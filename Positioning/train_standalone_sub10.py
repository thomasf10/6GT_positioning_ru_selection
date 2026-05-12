"""
Training script for standalone 2D positioning from sub-10 GHz CSI.

Single-head Conv3D model predicting the 2D position (x, y) of the UE.
Same Conv3D backbone as `RU_selection/train_csi_ru_beam_sel.py`, with the
three classification heads replaced by a single 2-output regression head.

Uses the on-grid subset of `CsiPositionDataset` (UEs sitting under an RU).
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
from tqdm import tqdm

# Add parent directory to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import CsiPositionDataset


# ---------------------------------------------------------------------------
# Preprocessing helpers (identical to RU/beam script)
# ---------------------------------------------------------------------------

def normalize_csi(sub10_csi):
    """Normalize CSI per sample by dividing by the max absolute value."""
    b = sub10_csi.shape[0]
    max_val = sub10_csi.view(b, -1).abs().max(dim=1).values.clamp(min=1e-8)
    return sub10_csi / max_val.view(b, 1, 1, 1, 1)


def prepare_csi_input(sub10_channel):
    """Reshape CSI tensor for Conv3D.

    Input:  (batch, n_aps, ue_ants, ru_ants, subcarriers, 2)  -- real / imag
    Output: (batch, n_aps, ue_ants, ru_ants, subcarriers * 2)
    """
    b, n_aps, ue, ru, sc, ri = sub10_channel.shape
    return sub10_channel.reshape(b, n_aps, ue, ru, sc * ri)


# ---------------------------------------------------------------------------
# Conv3D positioning model (same backbone, single regression head)
# ---------------------------------------------------------------------------

class PositioningCSIConv3D(nn.Module):
    def __init__(
        self,
        num_aps,
        out_dim=3,
        conv_channels=16,
        conv_layers=3,
        proj_channels=16,
        pooled_ue=2,
        pooled_ru=2,
        pooled_feat=4,
        fc_size=256,
        dropout=0.3,
        kernel_ue=2,
        kernel_ru=4,
    ):
        super().__init__()

        in_channels = num_aps
        layers = []
        current_channels = in_channels
        for layer_idx in range(conv_layers):
            out_channels = conv_channels * (2 ** layer_idx)
            stride_feat = 2 if layer_idx > 0 else 1
            k = (kernel_ue, kernel_ru, 3)
            p = (kernel_ue // 2, kernel_ru // 2, 1)
            layers.append(
                nn.Conv3d(
                    current_channels,
                    out_channels,
                    kernel_size=k,
                    stride=(1, 1, stride_feat),
                    padding=p,
                )
            )
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU())
            current_channels = out_channels

        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Conv3d(current_channels, proj_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.pre_fc_pool = nn.AdaptiveAvgPool3d((pooled_ue, pooled_ru, pooled_feat))

        fc_input_dim = proj_channels * pooled_ue * pooled_ru * pooled_feat

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.position_head = nn.Linear(fc_size, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        x = self.pre_fc_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return self.position_head(x)


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def _mean_euclidean_error(pred, target):
    return torch.linalg.norm(pred - target, dim=1).mean().item()


def train_epoch(model, train_loader, optimizer, device, criterion, csi_key='sub10_channel'):
    model.train()
    sum_loss = 0.0
    sum_err = 0.0
    n_samples = 0

    for data, position, _user_ids in tqdm(train_loader, desc='Train', leave=False):
        csi = data[csi_key].to(device)
        csi = prepare_csi_input(csi)
        csi = normalize_csi(csi)
        position = position.to(device)

        optimizer.zero_grad()
        pred = model(csi)
        loss = criterion(pred, position)
        loss.backward()
        optimizer.step()

        bs = position.size(0)
        sum_loss += loss.item() * bs
        sum_err += _mean_euclidean_error(pred.detach(), position) * bs
        n_samples += bs

    return sum_loss / n_samples, sum_err / n_samples


def validate(model, val_loader, device, criterion, csi_key='sub10_channel'):
    model.eval()
    sum_loss = 0.0
    sum_err = 0.0
    n_samples = 0

    with torch.no_grad():
        for data, position, _user_ids in tqdm(val_loader, desc='Validate', leave=False):
            csi = data[csi_key].to(device)
            csi = prepare_csi_input(csi)
            csi = normalize_csi(csi)
            position = position.to(device)

            pred = model(csi)
            loss = criterion(pred, position)

            bs = position.size(0)
            sum_loss += loss.item() * bs
            sum_err += _mean_euclidean_error(pred, position) * bs
            n_samples += bs

    return sum_loss / n_samples, sum_err / n_samples


# ---------------------------------------------------------------------------
# Test-set evaluation: CDF of per-sample positioning error (in metres)
# ---------------------------------------------------------------------------

def evaluate_position_cdf(model, test_loader, device, save_dir,
                          title='Test positioning error CDF',
                          csi_key='sub10_channel'):
    """Run inference on the test set, compute the per-sample Euclidean error
    (in metres), and save:

      - `test_pos_error_cdf.csv`   : sorted errors + CDF values (plot-ready)
      - `test_predictions.csv`     : per-sample raw data
                                     (user_id, true_x, true_y, pred_x, pred_y, error_m)
      - `test_pos_error_summary.csv` : one-line summary (n, mean, rmse, p50/90/95/99)
      - `test_pos_error_cdf.png`   : empirical CDF plot
      - `test_pos_error_cdf.tex`   : TikZ export of the CDF plot

    Returns a dict with the raw errors and key percentiles.
    """
    model.eval()
    all_errors = []
    all_true = []
    all_pred = []
    all_user_ids = []
    with torch.no_grad():
        for data, position, user_ids in tqdm(test_loader, desc='CDF eval', leave=False):
            csi = data[csi_key].to(device)
            csi = prepare_csi_input(csi)
            csi = normalize_csi(csi)
            position = position.to(device)

            pred = model(csi)
            err = torch.linalg.norm(pred - position, dim=1).cpu().numpy()
            all_errors.append(err)
            all_true.append(position.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
            if isinstance(user_ids, torch.Tensor):
                all_user_ids.extend(user_ids.cpu().tolist())
            else:
                all_user_ids.extend(list(user_ids))

    if not all_errors:
        raise ValueError('Test loader produced no samples; cannot build CDF.')
    errors = np.concatenate(all_errors)
    true_pos = np.concatenate(all_true, axis=0)
    pred_pos = np.concatenate(all_pred, axis=0)

    errors_sorted = np.sort(errors)
    cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)

    p50, p90, p95, p99 = np.percentile(errors, [50, 90, 95, 99])
    mean = float(errors.mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    print(
        f'Test positioning error (m): '
        f'mean={mean:.3f}  rmse={rmse:.3f}  '
        f'p50={p50:.3f}  p90={p90:.3f}  p95={p95:.3f}  p99={p99:.3f}'
    )

    save_dir = Path(save_dir)

    # --- CSV 1: sorted errors + empirical CDF (replots the figure directly) ---
    cdf_csv_path = save_dir / 'test_pos_error_cdf.csv'
    np.savetxt(
        cdf_csv_path,
        np.column_stack([errors_sorted, cdf]),
        delimiter=',',
        header='error_m,cdf',
        comments='',
    )

    # --- CSV 2: per-sample raw predictions (richer source for any later analysis) ---
    pred_csv_path = save_dir / 'test_predictions.csv'
    if true_pos.shape[1] != 2 or pred_pos.shape[1] != 2:
        raise ValueError(
            f'evaluate_position_cdf expects 2D positions; got true={true_pos.shape}, '
            f'pred={pred_pos.shape}.'
        )
    with open(pred_csv_path, 'w', encoding='utf-8') as f:
        f.write('user_id,true_x,true_y,pred_x,pred_y,error_m\n')
        for uid, (tx, ty), (px, py), err in zip(all_user_ids, true_pos, pred_pos, errors):
            f.write(f'{int(uid)},{tx:.6g},{ty:.6g},{px:.6g},{py:.6g},{err:.6g}\n')

    # --- CSV 3: one-line summary, easy to aggregate across runs ---
    summary_csv_path = save_dir / 'test_pos_error_summary.csv'
    with open(summary_csv_path, 'w', encoding='utf-8') as f:
        f.write('n,mean_m,rmse_m,p50_m,p90_m,p95_m,p99_m\n')
        f.write(
            f'{len(errors)},{mean:.6g},{rmse:.6g},'
            f'{float(p50):.6g},{float(p90):.6g},{float(p95):.6g},{float(p99):.6g}\n'
        )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(errors_sorted, cdf,
            label=f'mean={mean:.2f} m, p90={p90:.2f} m, p95={p95:.2f} m')
    for p_val in (p50, p90, p95):
        ax.axvline(p_val, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Position error (m)')
    ax.set_ylabel('CDF')
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc='lower right')

    plot_path = save_dir / 'test_pos_error_cdf.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    tikz_code = matplot2tikz.get_tikz_code(float_format='.4g')
    tikz_path = save_dir / 'test_pos_error_cdf.tex'
    with open(tikz_path, 'w', encoding='utf-8') as f:
        f.write(tikz_code)
    plt.close(fig)
    print(f'Test CDF saved to:      {plot_path}')
    print(f'TikZ saved to:          {tikz_path}')
    print(f'CDF CSV saved to:       {cdf_csv_path}')
    print(f'Per-sample CSV saved to: {pred_csv_path}')
    print(f'Summary CSV saved to:   {summary_csv_path}')

    return {
        'errors_m': errors,
        'true_pos': true_pos,
        'pred_pos': pred_pos,
        'user_ids': all_user_ids,
        'mean_m': mean,
        'rmse_m': rmse,
        'p50_m': float(p50),
        'p90_m': float(p90),
        'p95_m': float(p95),
        'p99_m': float(p99),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    dataset_folder = 'office_space_reduced_inline'
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'
    labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'

    # --- Training config ---
    mode = 'sub10'
    batch_size = 8
    epochs = 150
    lr = 1e-3

    # Reuse the val/test split written by `RU_selection/train_csi_ru_beam_sel.py`
    # so this script evaluates on exactly the same UEs. Set this to the
    # split_user_ids.json of the RU run you want to align with.
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
        f'csi_conv3d_pos'
        f'_ep{epochs}_bs{batch_size}_lr{lr:.0e}'
        f'_cc{conv_channels}_cl{conv_layers}_fc{fc_size}_do{dropout}'
        f'_{timestamp}'
    )
    run_dir = Path('stored_models_positioning_sub10') / run_name
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

    # Val/test: the saved RU val/test lists, but with on-grid UEs removed
    # (prevents on-grid UEs from leaking out of the training set into val/test).
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
        raise ValueError('Train (on-grid, minus held-out) set is empty.')
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

    sample_data, _sample_label, _ = train_dataset[0]
    csi_raw = sample_data['sub10_channel']  # (n_aps, ue, ru, sc, 2)
    csi_prepared = prepare_csi_input(csi_raw.unsqueeze(0)).squeeze(0)
    num_aps = csi_raw.shape[0]
    num_ue_ants = csi_raw.shape[1]
    num_ru_ants = csi_raw.shape[2]

    print('Dataset summary:')
    print(f'  dataset_folder           = {dataset_folder}')
    print(f'  train subset             = on_grid')
    print(f'  val/test                 = loaded from {ru_split_path} (on-grid UEs removed)')
    print(f'  sub10_channel raw shape  = {tuple(csi_raw.shape)}')
    print(f'  sub10_channel conv input = {tuple(csi_prepared.shape)}')
    print(f'  num_aps                  = {num_aps}')
    print(f'  train/val/test sizes     = {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}')

    criterion = nn.MSELoss()

    model = PositioningCSIConv3D(
        num_aps=num_aps,
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
            'num_aps': num_aps,
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

        tr_loss, tr_err = train_epoch(model, train_loader, optimizer, device, criterion)
        vl_loss, vl_err = validate(model, val_loader, device, criterion)
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
    test_loss, test_err = validate(model, test_loader, device, criterion)

    print('\nTraining finished')
    print(f'Best val loss:       {best_val_loss:.4f}')
    print(f'Test loss:           {test_loss:.4f}')
    print(f'Test pos error:      {test_err:.3f} m')
    print(f'Best checkpoint:     {best_ckpt_path}')

    # --- CDF of per-sample positioning error on the test set ---
    evaluate_position_cdf(
        model, test_loader, device, run_dir,
        title='Standalone sub-10 GHz: test positioning error CDF',
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
