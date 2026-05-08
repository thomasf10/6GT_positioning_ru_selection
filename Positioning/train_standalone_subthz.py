"""
Training script for standalone 2D positioning from sub-THz CSI.

Mirrors `train_standalone_sub10.py` exactly — same Conv3D backbone, same
2D regression head, same on-grid / saved-RU-split data setup — but reads
the `subthz_channel` from `CsiPositionDataset` and uses `mode='subTHz'`.

Compared to the sub-10 variant, the input tensor's first dimension is the
number of sub-THz RUs that send pilots (rather than the number of sub-10 APs);
otherwise the (ue_ants, ru_ants, subcarriers) layout is identical, so the
same backbone applies.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

import matplotlib.pyplot as plt
import matplot2tikz
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path so `dataset` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import CsiPositionDataset
from train_standalone_sub10 import (
    PositioningCSIConv3D,
    prepare_csi_input,
    train_epoch,
    validate,
    evaluate_position_cdf,
)


CSI_KEY = 'subthz_channel'


def main():
    # --- Paths ---
    dataset_folder = 'office_space_reduced_inline'
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'
    labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'

    # --- Training config ---
    mode = 'subTHz'
    batch_size = 8
    epochs = 150
    lr = 1e-3

    # Reuse the val/test split written by `RU_selection/train_csi_ru_beam_sel.py`
    # so this script evaluates on exactly the same UEs as the sub-10 variant.
    ru_split_path = Path('../RU_selection/stored_models_ru_beam_sel/csi_conv3d_ru_beam_ep1_bs32_lr3e-04_cc16_cl3_fc256_do0.3_0505_1135/split_user_ids.json')


    # --- Model config ---
    conv_channels = 16
    conv_layers = 3
    proj_channels = 16
    pooled_ue = 2
    pooled_ru = 2
    pooled_feat = 4
    fc_size = 256
    dropout = 0.3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%m%d_%H%M')
    run_name = (
        f'csi_conv3d_pos_subthz'
        f'_ep{epochs}_bs{batch_size}_lr{lr:.0e}'
        f'_cc{conv_channels}_cl{conv_layers}_fc{fc_size}_do{dropout}'
        f'_{timestamp}'
    )
    run_dir = Path('stored_models_positioning_subthz') / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {device}')
    print('Loading CSI positioning datasets...')

    # --- Load val/test user_ids from the RU/beam-selection split ---
    if not ru_split_path.exists():
        raise FileNotFoundError(
            f"RU split file not found at {ru_split_path}. Run "
            f"RU_selection/train_csi_ru_beam_sel.py first; it writes "
            f"split_user_ids.json into its run_dir."
        )
    with open(ru_split_path) as f:
        ru_split = json.load(f)
    saved_val_ids = list(ru_split['val_user_ids'])
    saved_test_ids = list(ru_split['test_user_ids'])
    print(f'Loaded RU split from {ru_split_path}: '
          f'val={len(saved_val_ids)}, test={len(saved_test_ids)}')

    # Train: all on-grid UEs.
    train_dataset = CsiPositionDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
        subset='on_grid',
    )

    # Val/test: the saved RU val/test lists, with on-grid UEs removed
    # (prevents on-grid UEs from leaking out of training into val/test).
    on_grid_ids = set(train_dataset.valid_users)
    val_dataset = CsiPositionDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
        user_ids=saved_val_ids,
        exclude_user_ids=on_grid_ids,
        max_users=160
    )
    test_dataset = CsiPositionDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
        user_ids=saved_test_ids,
        exclude_user_ids=on_grid_ids,
    )

    if len(train_dataset) < 1:
        raise ValueError('Train (on-grid) set is empty.')
    if len(val_dataset) < 1 or len(test_dataset) < 1:
        raise ValueError('Val or test set is empty after applying RU split.')

    # Hard guarantee that train is disjoint from val/test.
    train_ids = set(train_dataset.valid_users)
    val_ids = set(val_dataset.valid_users)
    test_ids = set(test_dataset.valid_users)
    overlap_train_test = train_ids & test_ids
    overlap_train_val = train_ids & val_ids
    if overlap_train_test or overlap_train_val:
        raise ValueError(
            f'Split leakage detected: train∩test={len(overlap_train_test)}, '
            f'train∩val={len(overlap_train_val)}'
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # subthz_channel raw shape: (n_rus, ue_ants, ru_ants, subcarriers, 2)
    sample_data, _sample_label, _ = train_dataset[0]
    csi_raw = sample_data[CSI_KEY]
    csi_prepared = prepare_csi_input(csi_raw.unsqueeze(0)).squeeze(0)
    num_rus = csi_raw.shape[0]
    num_ue_ants = csi_raw.shape[1]
    num_ru_ants = csi_raw.shape[2]

    print('Dataset summary:')
    print(f'  dataset_folder            = {dataset_folder}')
    print(f'  train subset              = on_grid')
    print(f'  val/test                  = loaded from {ru_split_path} (on-grid UEs removed)')
    print(f'  subthz_channel raw shape  = {tuple(csi_raw.shape)}')
    print(f'  subthz_channel conv input = {tuple(csi_prepared.shape)}')
    print(f'  num_rus                   = {num_rus}')
    print(f'  train/val/test sizes      = {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}')

    criterion = nn.MSELoss()

    # The Conv3D's `num_aps` argument is just the number of input channels of
    # the first Conv3d layer, which here corresponds to the number of sub-THz
    # RUs sending pilots.
    model = PositioningCSIConv3D(
        num_aps=num_rus,
        out_dim=2,
        conv_channels=conv_channels,
        conv_layers=conv_layers,
        proj_channels=proj_channels,
        pooled_ue=pooled_ue,
        pooled_ru=pooled_ru,
        pooled_feat=pooled_feat,
        fc_size=fc_size,
        dropout=dropout,
        kernel_ue=num_ue_ants,
        kernel_ru=num_ru_ants,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nModel & training summary:')
    print(f'  run_name                 = {run_name}')
    print(f'  -- Training --')
    print(f'  epochs                   = {epochs}')
    print(f'  batch_size               = {batch_size}')
    print(f'  lr                       = {lr}')
    print(f'  loss                     = MSE on (x, y)')
    print(f'  -- Architecture --')
    print(f'  conv_channels            = {conv_channels}')
    print(f'  conv_layers              = {conv_layers}')
    print(f'  kernel_size              = ({num_ue_ants}, {num_ru_ants}, 3)')
    print(f'  proj_channels            = {proj_channels}')
    print(f'  pooled_ue                = {pooled_ue}')
    print(f'  pooled_ru                = {pooled_ru}')
    print(f'  pooled_feat              = {pooled_feat}')
    print(f'  fc_size                  = {fc_size}')
    print(f'  dropout                  = {dropout}')
    print(f'  trainable parameters     = {num_params:,}')

    run_params = {
        'run_name': run_name,
        'dataset': {
            'dataset_folder': dataset_folder,
            'mode': mode,
            'csi_key': CSI_KEY,
            'train_subset': 'on_grid',
            'ru_split_path': str(ru_split_path),
            'ru_split_seed': ru_split.get('seed'),
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': 1e-4,
            'loss': 'MSE',
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
        },
        'architecture': {
            'num_rus': num_rus,
            'out_dim': 2,
            'conv_channels': conv_channels,
            'conv_layers': conv_layers,
            'kernel_size': [num_ue_ants, num_ru_ants, 3],
            'proj_channels': proj_channels,
            'pooled_ue': pooled_ue,
            'pooled_ru': pooled_ru,
            'pooled_feat': pooled_feat,
            'fc_size': fc_size,
            'dropout': dropout,
            'trainable_parameters': num_params,
        },
        'device': str(device),
    }
    params_path = run_dir / 'params.yaml'
    with open(params_path, 'w') as f:
        yaml.dump(run_params, f, default_flow_style=False, sort_keys=False)
    print(f'\nRun parameters saved to {params_path}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    writer = SummaryWriter(str(run_dir / 'logs'))
    best_ckpt_path = str(run_dir / 'best_model.pt')

    train_losses, val_losses = [], []
    train_errs, val_errs = [], []
    best_val_loss = float('inf')

    print('\nStarting training...\n')
    for epoch in range(epochs):
        t0 = time.perf_counter()

        tr_loss, tr_err = train_epoch(
            model, train_loader, optimizer, device, criterion, csi_key=CSI_KEY,
        )
        vl_loss, vl_err = validate(
            model, val_loader, device, criterion, csi_key=CSI_KEY,
        )
        scheduler.step(vl_loss)

        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val', vl_loss, epoch)
        writer.add_scalar('PositionError_m/train', tr_err, epoch)
        writer.add_scalar('PositionError_m/val', vl_err, epoch)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_errs.append(tr_err);    val_errs.append(vl_err)

        dt = time.perf_counter() - t0
        print(
            f'Epoch [{epoch + 1:03d}/{epochs}]  '
            f'loss={tr_loss:.4f}/{vl_loss:.4f}  '
            f'pos_err={tr_err:.3f}/{vl_err:.3f} m  '
            f'{dt:.1f}s'
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl_loss,
                'val_pos_err_m': vl_err,
                'config': {
                    'mode': mode,
                    'csi_key': CSI_KEY,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'lr': lr,
                    'conv_channels': conv_channels,
                    'conv_layers': conv_layers,
                    'proj_channels': proj_channels,
                    'pooled_ue': pooled_ue,
                    'pooled_ru': pooled_ru,
                    'pooled_feat': pooled_feat,
                    'fc_size': fc_size,
                    'dropout': dropout,
                    'kernel_ue': num_ue_ants,
                    'kernel_ru': num_ru_ants,
                },
            }, best_ckpt_path)

    writer.close()

    # --- Test evaluation ---
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_err = validate(model, test_loader, device, criterion, csi_key=CSI_KEY)

    print('\nTraining finished')
    print(f'Best val loss:       {best_val_loss:.4f}')
    print(f'Test loss:           {test_loss:.4f}')
    print(f'Test pos error:      {test_err:.3f} m')
    print(f'Best checkpoint:     {best_ckpt_path}')

    # --- CDF of per-sample positioning error on the test set ---
    evaluate_position_cdf(
        model, test_loader, device, run_dir,
        title='Standalone sub-THz: test positioning error CDF',
        csi_key=CSI_KEY,
    )

    # --- Training plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Curves')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(train_errs, label='Train Pos Error')
    axes[1].plot(val_errs, label='Val Pos Error')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Euclidean Error (m)')
    axes[1].set_title('Positioning Error')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plot_path = str(run_dir / 'training_plots.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    tikz_code = matplot2tikz.get_tikz_code(float_format='.4g')
    tikz_path = str(run_dir / 'training_plots.tex')
    with open(tikz_path, 'w', encoding='utf-8') as f:
        f.write(tikz_code)
    plt.close(fig)
    print(f'Training curves saved to: {plot_path}')
    print(f'TikZ saved to: {tikz_path}')


if __name__ == '__main__':
    main()
