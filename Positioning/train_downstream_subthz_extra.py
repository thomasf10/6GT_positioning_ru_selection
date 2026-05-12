"""
Sweep variant of `train_downstream_subthz.py`.

Runs the same downstream sub-THz positioning fine-tuning repeatedly across
a grid of (n_extra_train_users, freeze_backbone) combinations. Smaller n_extra
values are subsets of larger ones (deterministic given `extra_train_seed`).

All runs of a sweep land in a single `sweep_{timestamp}/` directory, with a
`sweep_summary.csv` aggregating the test metrics across the grid.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

import matplotlib.pyplot as plt
import matplot2tikz
import numpy as np
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
    train_epoch,
    validate,
    evaluate_position_cdf,
)
from train_downstream_sub10 import (
    load_pretrained_backbone,
    freeze_backbone_params,
)


CSI_KEY = 'subthz_channel'


def main():
    # --- Paths ---
    dataset_folder = 'office_space_reduced_inline'
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'
    labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'

    pretrained_ckpt_path = Path(
        '../RU_selection/stored_models_subTHz_input_ru_beam_sel/'
        'csi_conv3d_ru_beam_ep20_bs32_lr3e-04_cc16_cl3_fc256_do0.3_0415_1018/'
        'best_model.pt'
    )
    ru_split_path = Path(
        '../RU_selection/stored_models_ru_beam_sel/'
        'csi_conv3d_ru_beam_ep1_bs32_lr3e-04_cc16_cl3_fc256_do0.3_0505_1135/'
        'split_user_ids.json'
    )

    # --- Training config (shared across the sweep) ---
    mode = 'subTHz'
    batch_size = 8
    epochs = 150
    lr = 1e-3
    extra_train_seed = 42

    # --- Sweep ---
    n_extra_sweep = [100, 500, 1000, None]
    freeze_sweep = [True, False]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not pretrained_ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found at {pretrained_ckpt_path}."
        )
    pretrained_blob = torch.load(pretrained_ckpt_path, map_location='cpu')
    pretrained_config = pretrained_blob['config']
    conv_channels = pretrained_config['conv_channels']
    conv_layers = pretrained_config['conv_layers']
    proj_channels = pretrained_config['proj_channels']
    pooled_ue = pretrained_config['pooled_ue']
    pooled_ru = pretrained_config['pooled_ru']
    pooled_feat = pretrained_config['pooled_feat']
    fc_size = pretrained_config['fc_size']
    dropout = pretrained_config['dropout']
    kernel_ue_ckpt = pretrained_config['kernel_ue']
    kernel_ru_ckpt = pretrained_config['kernel_ru']

    sweep_timestamp = datetime.now().strftime('%m%d_%H%M')
    sweep_dir = Path('stored_models_positioning_subthz_downstream_extra') / f'sweep_{sweep_timestamp}'
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f'Using device:        {device}')
    print(f'Sweep dir:           {sweep_dir}')
    print(f'Pretrained ckpt:     {pretrained_ckpt_path}')

    if not ru_split_path.exists():
        raise FileNotFoundError(f"RU split file not found at {ru_split_path}.")
    with open(ru_split_path) as f:
        ru_split = json.load(f)
    saved_train_ids = list(ru_split['train_user_ids'])
    saved_val_ids = list(ru_split['val_user_ids'])
    saved_test_ids = list(ru_split['test_user_ids'])
    print(f'Loaded RU split: train={len(saved_train_ids)}, '
          f'val={len(saved_val_ids)}, test={len(saved_test_ids)}')

    on_grid_dataset = CsiPositionDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
        subset='on_grid',
    )
    on_grid_ids = set(on_grid_dataset.valid_users)

    extra_candidates_all = sorted(set(saved_train_ids) - on_grid_ids)
    max_extras = len(extra_candidates_all)
    rng = np.random.default_rng(extra_train_seed)
    shuffled_idx = rng.permutation(max_extras)
    shuffled_candidates = [extra_candidates_all[i] for i in shuffled_idx]
    print(f'On-grid UEs:        {len(on_grid_ids)}')
    print(f'Extra candidates:   {max_extras} (in train_user_ids, not on-grid)')

    # ----------------------------------------------------------------- #
    def run_one(n_extra_requested, freeze_backbone):
        if n_extra_requested is None:
            n_extra_label = 'all'
            n_pick = max_extras
        else:
            n_extra_label = str(n_extra_requested)
            n_pick = min(n_extra_requested, max_extras)
            if n_extra_requested > max_extras:
                print(
                    f'Warning: only {max_extras} extra UEs available, '
                    f'requested {n_extra_requested}.'
                )
        extra_ids = sorted(shuffled_candidates[:n_pick])

        combined_train_ids = sorted(on_grid_ids | set(extra_ids))
        train_dataset = CsiPositionDataset(
            subthz_path=subthz_path,
            sub10_path=sub10_path,
            labels_path=labels_path,
            mode=mode,
            user_ids=combined_train_ids,
        )
        held_out = set(train_dataset.valid_users)
        val_dataset = CsiPositionDataset(
            subthz_path=subthz_path,
            sub10_path=sub10_path,
            labels_path=labels_path,
            mode=mode,
            user_ids=saved_val_ids,
            exclude_user_ids=held_out,
            max_users=160,
        )
        test_dataset = CsiPositionDataset(
            subthz_path=subthz_path,
            sub10_path=sub10_path,
            labels_path=labels_path,
            mode=mode,
            user_ids=saved_test_ids,
            exclude_user_ids=held_out,
        )

        if len(train_dataset) < 1:
            raise ValueError('Train set is empty.')
        if len(val_dataset) < 1 or len(test_dataset) < 1:
            raise ValueError('Val or test set is empty after applying RU split.')

        train_ids = set(train_dataset.valid_users)
        if train_ids & set(val_dataset.valid_users) or train_ids & set(test_dataset.valid_users):
            raise ValueError('Split leakage detected between train and val/test.')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        sample_data, _sample_label, _ = train_dataset[0]
        csi_raw = sample_data[CSI_KEY]
        num_rus = csi_raw.shape[0]
        num_ue_ants = csi_raw.shape[1]
        num_ru_ants = csi_raw.shape[2]

        if (kernel_ue_ckpt, kernel_ru_ckpt) != (num_ue_ants, num_ru_ants):
            raise RuntimeError(
                f"Antenna mismatch: ckpt kernel=(ue={kernel_ue_ckpt}, "
                f"ru={kernel_ru_ckpt}) vs data=(ue={num_ue_ants}, ru={num_ru_ants})."
            )

        run_name = f'xtra{n_extra_label}_frz{int(freeze_backbone)}'
        run_dir = sweep_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n{"=" * 70}\n=== Run: {run_name} ===\n{"=" * 70}')
        print(f'  freeze_backbone={freeze_backbone}, '
              f'on_grid={len(on_grid_ids)} + extras={len(extra_ids)} '
              f'=> train_size={len(train_dataset)} '
              f'(val={len(val_dataset)}, test={len(test_dataset)})')

        criterion = nn.MSELoss()
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

        load_pretrained_backbone(model, pretrained_ckpt_path, device)
        if freeze_backbone:
            freeze_backbone_params(model)

        num_params_total = sum(p.numel() for p in model.parameters())
        num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        run_params = {
            'run_name': run_name,
            'sweep_dir': str(sweep_dir),
            'pretrained': {
                'ckpt_path': str(pretrained_ckpt_path),
                'freeze_backbone': freeze_backbone,
                'src_config': pretrained_config,
            },
            'dataset': {
                'dataset_folder': dataset_folder,
                'mode': mode,
                'csi_key': CSI_KEY,
                'train_subset': 'on_grid+extras',
                'n_on_grid': len(on_grid_ids),
                'n_extra_requested': n_extra_requested,
                'n_extra_picked': len(extra_ids),
                'extra_train_seed': extra_train_seed,
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
                'total_parameters': num_params_total,
                'trainable_parameters': num_params_trainable,
            },
            'device': str(device),
        }
        with open(run_dir / 'params.yaml', 'w') as f:
            yaml.dump(run_params, f, default_flow_style=False, sort_keys=False)

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        writer = SummaryWriter(str(run_dir / 'logs'))
        best_ckpt_path = str(run_dir / 'best_model.pt')

        train_losses, val_losses, train_errs, val_errs = [], [], [], []
        best_val_loss = float('inf')

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
                    'pretrained_ckpt': str(pretrained_ckpt_path),
                    'freeze_backbone': freeze_backbone,
                    'config': run_params,
                }, best_ckpt_path)

        writer.close()

        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        test_loss, test_err = validate(model, test_loader, device, criterion, csi_key=CSI_KEY)
        print(f'  Best val loss: {best_val_loss:.4f}  |  '
              f'Test loss: {test_loss:.4f}  |  Test pos err: {test_err:.3f} m')

        cdf_stats = evaluate_position_cdf(
            model, test_loader, device, run_dir,
            title=(f'Downstream sub-THz (+{len(extra_ids)} extras, '
                   f'frz={int(freeze_backbone)}): test positioning error CDF'),
            csi_key=CSI_KEY,
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(train_losses, label='Train Loss')
        axes[0].plot(val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
        axes[0].set_title(f'Loss ({run_name})'); axes[0].grid(True); axes[0].legend()
        axes[1].plot(train_errs, label='Train Pos Error')
        axes[1].plot(val_errs, label='Val Pos Error')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Mean Euclidean Error (m)')
        axes[1].set_title(f'Positioning Error ({run_name})'); axes[1].grid(True); axes[1].legend()
        plt.tight_layout()
        plt.savefig(run_dir / 'training_plots.png', dpi=150, bbox_inches='tight')
        with open(run_dir / 'training_plots.tex', 'w', encoding='utf-8') as f:
            f.write(matplot2tikz.get_tikz_code(float_format='.4g'))
        plt.close(fig)

        return {
            'n_extra_requested': n_extra_requested,
            'n_extra_label': n_extra_label,
            'n_extra_picked': len(extra_ids),
            'freeze_backbone': freeze_backbone,
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_pos_err_m': test_err,
            'mean_m': cdf_stats['mean_m'],
            'rmse_m': cdf_stats['rmse_m'],
            'p50_m': cdf_stats['p50_m'],
            'p90_m': cdf_stats['p90_m'],
            'p95_m': cdf_stats['p95_m'],
            'p99_m': cdf_stats['p99_m'],
        }

    sweep_results = []
    for freeze_backbone in freeze_sweep:
        for n_extra in n_extra_sweep:
            result = run_one(n_extra, freeze_backbone)
            sweep_results.append(result)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sweep_csv = sweep_dir / 'sweep_summary.csv'
    with open(sweep_csv, 'w', encoding='utf-8') as f:
        f.write(
            'n_extra_requested,n_extra_picked,freeze_backbone,'
            'train_size,val_size,test_size,'
            'best_val_loss,test_loss,test_mean_m,test_rmse_m,'
            'test_p50_m,test_p90_m,test_p95_m,test_p99_m\n'
        )
        for r in sweep_results:
            f.write(
                f'{r["n_extra_label"]},{r["n_extra_picked"]},'
                f'{int(r["freeze_backbone"])},'
                f'{r["train_size"]},{r["val_size"]},{r["test_size"]},'
                f'{r["best_val_loss"]:.6g},{r["test_loss"]:.6g},'
                f'{r["mean_m"]:.6g},{r["rmse_m"]:.6g},'
                f'{r["p50_m"]:.6g},{r["p90_m"]:.6g},'
                f'{r["p95_m"]:.6g},{r["p99_m"]:.6g}\n'
            )

    print(f'\n\n=== Sweep summary ({sweep_csv}) ===')
    print(
        f'{"n_extra":<8}{"frz":>5}{"train":>8}'
        f'{"mean[m]":>10}{"rmse[m]":>10}{"p50":>8}{"p90":>8}{"p95":>8}'
    )
    for r in sweep_results:
        print(
            f'{r["n_extra_label"]:<8}{int(r["freeze_backbone"]):>5}'
            f'{r["train_size"]:>8}'
            f'{r["mean_m"]:>10.3f}{r["rmse_m"]:>10.3f}'
            f'{r["p50_m"]:>8.3f}{r["p90_m"]:>8.3f}{r["p95_m"]:>8.3f}'
        )


if __name__ == '__main__':
    main()