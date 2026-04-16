"""
Test script for trained subTHz-CSI Conv3D joint RU + beam selection models.

Works with checkpoints from both:
  - train_subthz_csi_ru_beam_sel.py       (from-scratch 3-head training)
  - train_subthz_csi_ru_beam_transfer.py  (transfer-learning 3-head training)

Uses the same dataset split (results.csv, seed=42, 80/10/10) as the
training scripts so the test set is identical.
"""
import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import CsiDataset
from evaluation import evaluate_nmse_cdf, plot_spatial_nmse_delta, plot_ue_ru_locations, get_ru_positions
from train_subthz_csi_ru_beam_sel import (
    RUBeamSelectionCSIConv3D,
    normalize_csi,
    prepare_csi_input,
    validate,
    _extract_labels,
)


def main():
    # --- Configuration (edit these) ---
    checkpoint_path = 'stored_models_subTHz_input_ru_beam_sel/csi_conv3d_ru_beam_ep20_bs32_lr3e-04_cc16_cl3_fc256_do0.3_0409_1032/best_model.pt'
    dataset_folder = 'office_space_reduced_inline'
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load checkpoint
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    config = ckpt['config']
    print(f'Loaded checkpoint: {ckpt_path}')
    print(f'  trained epoch: {ckpt["epoch"]}')
    print(f'  val_loss:      {ckpt["val_loss"]:.4f}')
    print(f'  val_ru_id_acc: {ckpt["val_ru_id_acc"]:.2f}%')
    print(f'  val_ue_beam:   {ckpt["val_ue_beam_acc"]:.2f}%')
    print(f'  val_ru_beam:   {ckpt["val_ru_beam_acc"]:.2f}%')
    if 'phase' in ckpt:
        print(f'  best phase:    {ckpt["phase"]}')
    print(f'  config:        {config}')

    # Paths — always use results.csv (optimal beams) to match training scripts
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'
    labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'
    raw_data_path = f'../dataset/{dataset_folder}/ru_selection_labels/flickering_raw_data'

    config_path = Path(__file__).parent.parent / 'dataset' / dataset_folder / 'config_used.yaml'
    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)
    num_stripes = dataset_config['stripe_config']['N_stripes']
    num_rus_per_stripe = dataset_config['stripe_config']['N_RUs']

    # Load dataset with the same split as training
    mode = config.get('mode', 'subTHz')
    dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
        num_stripes=num_stripes,
        num_rus_per_stripe=num_rus_per_stripe,
    )

    valid_indices = [
        i for i, uid in enumerate(dataset.valid_users) if uid in dataset.labels_dict
    ]
    dataset = torch.utils.data.Subset(dataset, valid_indices)

    total_samples = len(dataset)
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, _val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f'\nTest set size: {len(test_dataset)}')

    # Build model from saved config
    sample_data, _, _ = dataset[0]
    csi_raw = sample_data['subthz_channel']
    num_aps = csi_raw.shape[0]
    num_ru_ids = dataset.dataset.num_ru_ids

    model = RUBeamSelectionCSIConv3D(
        num_aps=num_aps,
        num_ru_ids=num_ru_ids,
        conv_channels=config['conv_channels'],
        conv_layers=config['conv_layers'],
        proj_channels=config['proj_channels'],
        pooled_ue=config['pooled_ue'],
        pooled_ru=config['pooled_ru'],
        pooled_feat=config['pooled_feat'],
        fc_size=config['fc_size'],
        dropout=config['dropout'],
        kernel_ue=config.get('kernel_ue', 2),
        kernel_ru=config.get('kernel_ru', 4),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    # ------------------------------------------------------------------
    # Test accuracy (3 heads)
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    loss_weights = tuple(config.get('loss_weights', [1.0, 1.0, 1.0]))
    test_loss, test_ru_acc, test_ue_beam_acc, test_ru_beam_acc = validate(
        model, test_loader, device, criterion, loss_weights,
    )
    print(f'\nTest loss:            {test_loss:.4f}')
    print(f'Test RU ID accuracy:  {test_ru_acc:.2f}%')
    print(f'Test UE beam accuracy:{test_ue_beam_acc:.2f}%')
    print(f'Test RU beam accuracy:{test_ru_beam_acc:.2f}%')

    # ------------------------------------------------------------------
    # NMSE CDF evaluation — NN uses predicted RU + predicted beams
    # ------------------------------------------------------------------
    print('\nRunning NMSE CDF evaluation on test set...')
    model.eval()
    all_ue_ids = []
    all_ru_labels = []
    all_ru_preds = []
    all_ue_beam_preds = []
    all_ru_beam_preds = []

    with torch.no_grad():
        for data, labels, user_ids in tqdm(test_loader, desc='Inference', leave=False):
            subthz_csi = data['subthz_channel'].to(device)
            subthz_csi = prepare_csi_input(subthz_csi)
            subthz_csi = normalize_csi(subthz_csi)

            ru_logits, ue_beam_logits, ru_beam_logits = model(subthz_csi)

            all_ru_preds.extend(ru_logits.argmax(1).cpu().tolist())
            all_ue_beam_preds.extend(ue_beam_logits.argmax(1).cpu().tolist())
            all_ru_beam_preds.extend(ru_beam_logits.argmax(1).cpu().tolist())

            all_ru_labels.extend(labels['global_ru_id'].tolist())
            batch_uids = user_ids.tolist() if isinstance(user_ids, torch.Tensor) else list(user_ids)
            all_ue_ids.extend(batch_uids)

    # Convert predicted beam IDs (0-6) → beam angles (-30 … +30)
    beam_angles = CsiDataset.BEAM_ANGLES  # [-30, -20, -10, 0, 10, 20, 30]
    nn_ue_beam_angles = [beam_angles[bid] for bid in all_ue_beam_preds]
    nn_ru_beam_angles = [beam_angles[bid] for bid in all_ru_beam_preds]

    save_dir = ckpt_path.parent / 'post_training_eval'
    save_dir.mkdir(exist_ok=True)
    ru_x, ru_y = get_ru_positions(str(config_path))

    results_csv_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'
    nmse_results = evaluate_nmse_cdf(
        ue_ids=all_ue_ids,
        global_ru_id_labels=all_ru_labels,
        global_ru_id_predictions=all_ru_preds,
        results_csv_path=results_csv_path,
        raw_data_path=raw_data_path,
        num_rus_per_stripe=num_rus_per_stripe,
        nn_ue_beam_angles=nn_ue_beam_angles,
        nn_ru_beam_angles=nn_ru_beam_angles,
        save_path=str(save_dir / 'nmse_cdf_test_eval.png'),
        title='NMSE CDF: Optimal vs Conv3D RU + Beam Selection (subTHz CSI) — Test',
    )

    # ------------------------------------------------------------------
    # Spatial plots
    # ------------------------------------------------------------------
    ue_x = nmse_results['ue_x']
    ue_y = nmse_results['ue_y']

    # Training-set UE locations
    train_ue_ids = [
        dataset.dataset.valid_users[dataset.indices[i]]
        for i in train_dataset.indices
    ]
    plot_ue_ru_locations(
        ue_ids=train_ue_ids,
        results_csv_path=labels_path,
        ru_x=ru_x, ru_y=ru_y,
        save_path=str(save_dir / 'ue_ru_locations_train.png'),
        title='Training set: UE and RU locations',
    )

    # 1) Optimal RU optimal beams  vs  NN selected RU + beams
    plot_spatial_nmse_delta(
        ue_x, ue_y,
        nmse_a=nmse_results['optimal'],
        nmse_b=nmse_results['nn'],
        ru_x=ru_x, ru_y=ru_y,
        label_a='Optimal RU, optimal beams',
        label_b='NN RU + beams',
        save_path=str(save_dir / 'spatial_nmse_optimal_vs_nn.png'),
    )

    # 2) Best RU for fixed beams  vs  Closest RU fixed beams
    plot_spatial_nmse_delta(
        ue_x, ue_y,
        nmse_a=nmse_results['best_ru_fixed'],
        nmse_b=nmse_results['closest_fixed'],
        ru_x=ru_x, ru_y=ru_y,
        label_a='Best RU, fixed beams',
        label_b='Closest RU, fixed beams',
        save_path=str(save_dir / 'spatial_nmse_bestru_vs_closest.png'),
    )

    print('\nDone.')


if __name__ == '__main__':
    main()
