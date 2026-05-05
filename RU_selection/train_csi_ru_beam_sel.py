"""
Training script for joint RU + beam selection from sub-10 GHz CSI.

Three-head Conv3D model predicting:
  1) global_ru_id   (num_stripes * num_rus_per_stripe classes)
  2) ue_beam_id     (7 classes: beam angles -30° … +30°)
  3) ru_beam_id     (7 classes)

Uses 'optimal_beams' label mode (results.csv) so that beam labels are
the true optimal beams, not fixed 0°.
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

from dataset.dataloaders import CsiDataset
from evaluation import evaluate_nmse_cdf, get_ru_positions


# ---------------------------------------------------------------------------
# Preprocessing helpers (identical to RU-only script)
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
# Three-head Conv3D model
# ---------------------------------------------------------------------------

class RUBeamSelectionCSIConv3D(nn.Module):
    """Conv3D model with three classification heads:
    global RU ID, UE beam ID, and RU beam ID."""

    NUM_BEAM_IDS = len(CsiDataset.BEAM_ANGLES)  # 7

    def __init__(
        self,
        num_aps,
        num_ru_ids=30,
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

        # --- Shared conv backbone ---
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

        # --- Shared FC ---
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # --- Task-specific heads ---
        self.ru_id_head = nn.Linear(fc_size, num_ru_ids)
        self.ue_beam_head = nn.Linear(fc_size, self.NUM_BEAM_IDS)
        self.ru_beam_head = nn.Linear(fc_size, self.NUM_BEAM_IDS)

    def forward(self, x):
        # x: (batch, n_aps, ue_ants, ru_ants, subcarriers*2)
        x = self.conv(x)
        x = self.proj(x)
        x = self.pre_fc_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return self.ru_id_head(x), self.ue_beam_head(x), self.ru_beam_head(x)


# ---------------------------------------------------------------------------
# Loss & label extraction helpers
# ---------------------------------------------------------------------------

def _extract_labels(labels, device):
    """Extract all three label tensors from a batch."""
    if isinstance(labels, dict):
        global_ru_ids = labels['global_ru_id'].to(device)
        ue_beam_ids = labels['ue_beam_id'].to(device)
        ru_beam_ids = labels['ru_beam_id'].to(device)
    else:
        global_ru_ids = torch.tensor(
            [l['global_ru_id'] for l in labels], dtype=torch.long, device=device)
        ue_beam_ids = torch.tensor(
            [l['ue_beam_id'] for l in labels], dtype=torch.long, device=device)
        ru_beam_ids = torch.tensor(
            [l['ru_beam_id'] for l in labels], dtype=torch.long, device=device)
    return global_ru_ids, ue_beam_ids, ru_beam_ids


def multi_task_loss(ru_logits, ue_beam_logits, ru_beam_logits,
                    labels, device, criterion, loss_weights=(1.0, 1.0, 1.0)):
    """Weighted sum of cross-entropy losses for the three heads."""
    global_ru_ids, ue_beam_ids, ru_beam_ids = _extract_labels(labels, device)

    loss_ru = criterion(ru_logits, global_ru_ids)
    loss_ue_beam = criterion(ue_beam_logits, ue_beam_ids)
    loss_ru_beam = criterion(ru_beam_logits, ru_beam_ids)

    total = (loss_weights[0] * loss_ru
             + loss_weights[1] * loss_ue_beam
             + loss_weights[2] * loss_ru_beam)

    return total, loss_ru.item(), loss_ue_beam.item(), loss_ru_beam.item()


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, device, criterion, loss_weights):
    model.train()
    sum_loss = 0.0
    ru_correct = ue_beam_correct = ru_beam_correct = 0
    total = 0

    for data, labels, _user_ids in tqdm(train_loader, desc='Train', leave=False):
        sub10_csi = data['sub10_channel'].to(device)
        sub10_csi = prepare_csi_input(sub10_csi)
        sub10_csi = normalize_csi(sub10_csi)

        optimizer.zero_grad()
        ru_logits, ue_beam_logits, ru_beam_logits = model(sub10_csi)
        loss, _, _, _ = multi_task_loss(
            ru_logits, ue_beam_logits, ru_beam_logits,
            labels, device, criterion, loss_weights,
        )
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        global_ru_ids, ue_beam_ids, ru_beam_ids = _extract_labels(labels, device)
        total += global_ru_ids.size(0)
        ru_correct += (ru_logits.argmax(1) == global_ru_ids).sum().item()
        ue_beam_correct += (ue_beam_logits.argmax(1) == ue_beam_ids).sum().item()
        ru_beam_correct += (ru_beam_logits.argmax(1) == ru_beam_ids).sum().item()

    n = len(train_loader)
    return (
        sum_loss / n,
        100.0 * ru_correct / total,
        100.0 * ue_beam_correct / total,
        100.0 * ru_beam_correct / total,
    )


def validate(model, val_loader, device, criterion, loss_weights):
    model.eval()
    sum_loss = 0.0
    ru_correct = ue_beam_correct = ru_beam_correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, _user_ids in tqdm(val_loader, desc='Validate', leave=False):
            sub10_csi = data['sub10_channel'].to(device)
            sub10_csi = prepare_csi_input(sub10_csi)
            sub10_csi = normalize_csi(sub10_csi)

            ru_logits, ue_beam_logits, ru_beam_logits = model(sub10_csi)
            loss, _, _, _ = multi_task_loss(
                ru_logits, ue_beam_logits, ru_beam_logits,
                labels, device, criterion, loss_weights,
            )
            sum_loss += loss.item()

            global_ru_ids, ue_beam_ids, ru_beam_ids = _extract_labels(labels, device)
            total += global_ru_ids.size(0)
            ru_correct += (ru_logits.argmax(1) == global_ru_ids).sum().item()
            ue_beam_correct += (ue_beam_logits.argmax(1) == ue_beam_ids).sum().item()
            ru_beam_correct += (ru_beam_logits.argmax(1) == ru_beam_ids).sum().item()

    n = len(val_loader)
    return (
        sum_loss / n,
        100.0 * ru_correct / total,
        100.0 * ue_beam_correct / total,
        100.0 * ru_beam_correct / total,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Paths ---
    dataset_folder = 'office_space_reduced_inline'
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'
    labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'

    config_path = Path(__file__).parent.parent / 'dataset' / dataset_folder / 'config_used.yaml'
    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)
    num_stripes = dataset_config['stripe_config']['N_stripes']
    num_rus_per_stripe = dataset_config['stripe_config']['N_RUs']

    # --- Training config ---
    mode = 'sub10'
    batch_size = 32
    epochs = 1
    lr = 3e-4

    # Loss weights for the three heads: [ru_id, ue_beam, ru_beam]
    loss_weights = (1.0, 1.0, 1.0)

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
        f'csi_conv3d_ru_beam'
        f'_ep{epochs}_bs{batch_size}_lr{lr:.0e}'
        f'_cc{conv_channels}_cl{conv_layers}_fc{fc_size}_do{dropout}'
        f'_{timestamp}'
    )
    run_dir = Path('stored_models_ru_beam_sel') / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {device}')
    print('Loading CSI dataset...')

    dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
        num_stripes=num_stripes,
        num_rus_per_stripe=num_rus_per_stripe,
    )

    # Filter to only indices that have a label entry
    valid_indices = [
        i for i, uid in enumerate(dataset.valid_users) if uid in dataset.labels_dict
    ]
    dataset = torch.utils.data.Subset(dataset, valid_indices)

    total_samples = len(dataset)
    if total_samples < 10:
        raise ValueError('Dataset too small for train/val/test split')

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator,
    )

    # Persist the train/val/test user_id lists so other scripts (e.g. positioning)
    # can reuse the exact same split.
    def _subset_user_ids(subset_obj):
        indices = list(subset_obj.indices)
        cur = subset_obj.dataset
        while isinstance(cur, torch.utils.data.Subset):
            indices = [cur.indices[i] for i in indices]
            cur = cur.dataset
        return [int(cur.valid_users[i]) for i in indices]

    split_user_ids = {
        'dataset_folder': dataset_folder,
        'mode': mode,
        'seed': 42,
        'sizes': {'train': train_size, 'val': val_size, 'test': test_size},
        'train_user_ids': _subset_user_ids(train_dataset),
        'val_user_ids': _subset_user_ids(val_dataset),
        'test_user_ids': _subset_user_ids(test_dataset),
    }
    split_path = run_dir / 'split_user_ids.json'
    with open(split_path, 'w') as f:
        json.dump(split_user_ids, f, indent=2)
    print(f'Train/val/test user_ids saved to {split_path}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_data, _sample_label, _ = dataset[0]
    csi_raw = sample_data['sub10_channel']  # (n_aps, ue, ru, sc, 2)
    csi_prepared = prepare_csi_input(csi_raw.unsqueeze(0)).squeeze(0)
    num_aps = csi_raw.shape[0]
    num_ue_ants = csi_raw.shape[1]
    num_ru_ants = csi_raw.shape[2]
    num_ru_ids = dataset.dataset.num_ru_ids
    num_beam_ids = RUBeamSelectionCSIConv3D.NUM_BEAM_IDS

    print('Dataset summary:')
    print(f'  dataset_folder           = {dataset_folder}')
    print(f'  num_stripes              = {num_stripes}')
    print(f'  num_rus_per_stripe       = {num_rus_per_stripe}')
    print(f'  num_ru_ids               = {num_ru_ids}')
    print(f'  num_beam_ids             = {num_beam_ids}')
    print(f'  sub10_channel raw shape  = {tuple(csi_raw.shape)}')
    print(f'  sub10_channel conv input = {tuple(csi_prepared.shape)}')
    print(f'  num_aps                  = {num_aps}')
    print(f'  train/val/test sizes     = {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}')

    criterion = nn.CrossEntropyLoss()

    model = RUBeamSelectionCSIConv3D(
        num_aps=num_aps,
        num_ru_ids=num_ru_ids,
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
    print(f'  loss_weights             = {loss_weights}')
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

    # Log all run parameters
    run_params = {
        'run_name': run_name,
        'dataset': {
            'dataset_folder': dataset_folder,
            'mode': mode,
            'num_stripes': num_stripes,
            'num_rus_per_stripe': num_rus_per_stripe,
            'num_ru_ids': num_ru_ids,
            'num_beam_ids': num_beam_ids,
            'total_samples': total_samples,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'seed': 42,
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': 1e-4,
            'loss_weights': list(loss_weights),
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
        },
        'architecture': {
            'num_aps': num_aps,
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
    train_ru_accs, val_ru_accs = [], []
    train_ue_beam_accs, val_ue_beam_accs = [], []
    train_ru_beam_accs, val_ru_beam_accs = [], []
    best_val_loss = float('inf')

    print('\nStarting training...\n')
    for epoch in range(epochs):
        t0 = time.perf_counter()

        tr_loss, tr_ru, tr_ue_b, tr_ru_b = train_epoch(
            model, train_loader, optimizer, device, criterion, loss_weights)
        vl_loss, vl_ru, vl_ue_b, vl_ru_b = validate(
            model, val_loader, device, criterion, loss_weights)
        scheduler.step(vl_loss)

        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val', vl_loss, epoch)
        writer.add_scalar('Accuracy/train_ru_id', tr_ru, epoch)
        writer.add_scalar('Accuracy/val_ru_id', vl_ru, epoch)
        writer.add_scalar('Accuracy/train_ue_beam', tr_ue_b, epoch)
        writer.add_scalar('Accuracy/val_ue_beam', vl_ue_b, epoch)
        writer.add_scalar('Accuracy/train_ru_beam', tr_ru_b, epoch)
        writer.add_scalar('Accuracy/val_ru_beam', vl_ru_b, epoch)

        train_losses.append(tr_loss);       val_losses.append(vl_loss)
        train_ru_accs.append(tr_ru);        val_ru_accs.append(vl_ru)
        train_ue_beam_accs.append(tr_ue_b); val_ue_beam_accs.append(vl_ue_b)
        train_ru_beam_accs.append(tr_ru_b); val_ru_beam_accs.append(vl_ru_b)

        dt = time.perf_counter() - t0
        print(
            f'Epoch [{epoch + 1:03d}/{epochs}]  '
            f'loss={tr_loss:.4f}/{vl_loss:.4f}  '
            f'ru_acc={tr_ru:.1f}/{vl_ru:.1f}%  '
            f'ue_beam={tr_ue_b:.1f}/{vl_ue_b:.1f}%  '
            f'ru_beam={tr_ru_b:.1f}/{vl_ru_b:.1f}%  '
            f'{dt:.1f}s'
        )

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': vl_loss,
                'val_ru_id_acc': vl_ru,
                'val_ue_beam_acc': vl_ue_b,
                'val_ru_beam_acc': vl_ru_b,
                'config': {
                    'mode': mode,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'lr': lr,
                    'loss_weights': list(loss_weights),
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
    test_loss, test_ru, test_ue_b, test_ru_b = validate(
        model, test_loader, device, criterion, loss_weights)

    print('\nTraining finished')
    print(f'Best val loss:       {best_val_loss:.4f}')
    print(f'Test loss:           {test_loss:.4f}')
    print(f'Test RU ID acc:      {test_ru:.2f}%')
    print(f'Test UE beam acc:    {test_ue_b:.2f}%')
    print(f'Test RU beam acc:    {test_ru_b:.2f}%')
    print(f'Best checkpoint:     {best_ckpt_path}')

    # --- Training plots ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(train_ru_accs, label='Train RU Acc')
    axes[1].plot(val_ru_accs, label='Val RU Acc')
    axes[1].plot(train_ue_beam_accs, label='Train UE Beam Acc', linestyle='--')
    axes[1].plot(val_ue_beam_accs, label='Val UE Beam Acc', linestyle='--')
    axes[1].plot(train_ru_beam_accs, label='Train RU Beam Acc', linestyle=':')
    axes[1].plot(val_ru_beam_accs, label='Val RU Beam Acc', linestyle=':')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
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

    # --- NMSE CDF evaluation on test set ---
    print('\nRunning NMSE CDF evaluation on test set...')
    raw_data_path = f'../dataset/{dataset_folder}/ru_selection_labels/flickering_raw_data'
    model.eval()
    all_ue_ids, all_ru_labels, all_ru_preds = [], [], []
    all_ue_beam_preds, all_ru_beam_preds = [], []

    with torch.no_grad():
        for data, labels, user_ids in tqdm(test_loader, desc='Inference', leave=False):
            sub10_csi = data['sub10_channel'].to(device)
            sub10_csi = prepare_csi_input(sub10_csi)
            sub10_csi = normalize_csi(sub10_csi)

            ru_logits, ue_beam_logits, ru_beam_logits = model(sub10_csi)

            all_ru_preds.extend(ru_logits.argmax(1).cpu().tolist())
            all_ue_beam_preds.extend(ue_beam_logits.argmax(1).cpu().tolist())
            all_ru_beam_preds.extend(ru_beam_logits.argmax(1).cpu().tolist())
            all_ru_labels.extend(labels['global_ru_id'].tolist())
            batch_uids = user_ids.tolist() if isinstance(user_ids, torch.Tensor) else list(user_ids)
            all_ue_ids.extend(batch_uids)

    # Convert predicted beam IDs (0-6) -> angles (-30 ... +30)
    beam_angles = CsiDataset.BEAM_ANGLES
    nn_ue_beam_angles = [beam_angles[bid] for bid in all_ue_beam_preds]
    nn_ru_beam_angles = [beam_angles[bid] for bid in all_ru_beam_preds]

    ru_x, ru_y = get_ru_positions(str(config_path))
    results_csv_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'

    evaluate_nmse_cdf(
        ue_ids=all_ue_ids,
        global_ru_id_labels=all_ru_labels,
        global_ru_id_predictions=all_ru_preds,
        results_csv_path=results_csv_path,
        raw_data_path=raw_data_path,
        num_rus_per_stripe=num_rus_per_stripe,
        nn_ue_beam_angles=nn_ue_beam_angles,
        nn_ru_beam_angles=nn_ru_beam_angles,
        save_path=str(run_dir / 'nmse_cdf_test.png'),
        title='NMSE CDF: Optimal vs Conv3D RU + Beam Selection (CSI) — Test',
    )


if __name__ == '__main__':
    main()
