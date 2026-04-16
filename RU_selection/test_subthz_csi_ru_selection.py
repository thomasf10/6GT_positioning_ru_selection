"""
Test script for a trained subTHz-CSI Conv3D RU selection model.
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
from train_subthz_csi_ru_sel import (
    RUSelectionCSIConv3D,
    normalize_csi,
    prepare_csi_input,
    validate,
)


def main():
    # --- Configuration (edit these) ---
    checkpoint_path = 'stored_models_subTHz_input/csi_conv3d_ep20_bs32_lr3e-04_cc16_cl3_fc256_do0.3_0408_1024/best_model.pt'
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
    print(f'  config:        {config}')

    # Paths
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'
    raw_data_path = f'../dataset/{dataset_folder}/ru_selection_labels/flickering_raw_data'

    label_mode = config.get('label_mode', 'optimal_beams')
    if label_mode == 'fixed_beams':
        labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results_fixed_beams_0deg.csv'
    else:
        labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'
    print(f'  label_mode:    {label_mode}')

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

    model = RUSelectionCSIConv3D(
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

    # Test accuracy
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = validate(model, test_loader, device, criterion=criterion)
    print(f'\nTest loss:           {test_loss:.4f}')
    print(f'Test RU ID accuracy: {test_acc:.2f}%')

    # NMSE CDF evaluation
    print('\nRunning NMSE CDF evaluation on test set...')
    model.eval()
    all_ue_ids, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for data, labels, user_ids in tqdm(test_loader, desc='Inference', leave=False):
            subthz_csi = data['subthz_channel'].to(device)
            subthz_csi = prepare_csi_input(subthz_csi)
            subthz_csi = normalize_csi(subthz_csi)
            logits = model(subthz_csi)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            batch_labels = labels['global_ru_id'].tolist()
            batch_uids = user_ids.tolist() if isinstance(user_ids, torch.Tensor) else list(user_ids)
            all_ue_ids.extend(batch_uids)
            all_labels.extend(batch_labels)
            all_preds.extend(batch_preds)

    save_dir = ckpt_path.parent / 'post_training_eval'
    save_dir.mkdir(exist_ok=True)
    ru_x, ru_y = get_ru_positions(str(config_path))

    results_csv_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'
    nmse_results = evaluate_nmse_cdf(
        ue_ids=all_ue_ids,
        global_ru_id_labels=all_labels,
        global_ru_id_predictions=all_preds,
        results_csv_path=results_csv_path,
        raw_data_path=raw_data_path,
        num_rus_per_stripe=num_rus_per_stripe,
        save_path=str(save_dir / 'nmse_cdf_test_eval.png'),
        title='NMSE CDF: Optimal vs Conv3D RU Selection (subTHz CSI) — Test',
    )

    # RU positions (shared by all spatial plots) — derived from config
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

    # Spatial NMSE delta plots
    ue_x = nmse_results['ue_x']
    ue_y = nmse_results['ue_y']

    # 1) Optimal RU optimal beams  vs  NN selected RU fixed beams
    plot_spatial_nmse_delta(
        ue_x, ue_y,
        nmse_a=nmse_results['optimal'],
        nmse_b=nmse_results['nn'],
        ru_x=ru_x, ru_y=ru_y,
        label_a='Optimal RU, optimal beams',
        label_b='NN RU, fixed beams',
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
