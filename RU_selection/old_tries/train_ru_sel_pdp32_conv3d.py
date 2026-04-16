import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import PDP32Dataset
from evaluation import evaluate_nmse_cdf


def normalize_pdp_db(sub10_pdp, pdp_db_min=-120.0, pdp_db_max=-40.0):
    """Clamp PDP dB values to a fixed range and map them to [0, 1]."""
    sub10_pdp = torch.clamp(sub10_pdp, min=pdp_db_min, max=pdp_db_max)
    return (sub10_pdp - pdp_db_min) / (pdp_db_max - pdp_db_min)


class RUSelectionPDP32Conv3D(nn.Module):
    """Small Conv3D baseline for RU selection from PDP-32 inputs."""

    def __init__(
        self,
        num_aps,
        num_ru_ids=160,
        conv_channels=16,
        conv_layers=3,
        proj_channels=16,
        pooled_ue=2,
        pooled_ru=2,
        pooled_taps=4,
        fc_size=256,
        dropout=0.3,
    ):
        super().__init__()

        in_channels = num_aps
        layers = []
        current_channels = in_channels
        for layer_idx in range(conv_layers):
            out_channels = conv_channels * (2 ** layer_idx)
            stride_taps = 2 if layer_idx > 0 else 1
            layers.append(
                nn.Conv3d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(1, 1, stride_taps),
                    padding=1,
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
        self.pre_fc_pool = nn.AdaptiveAvgPool3d((pooled_ue, pooled_ru, pooled_taps))

        fc_input_dim = proj_channels * pooled_ue * pooled_ru * pooled_taps
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.ru_id_head = nn.Linear(fc_size, num_ru_ids)

    def forward(self, sub10_pdp):
        # Input is already PDP in shape (batch, n_aps, ue_ant, ru_ant, taps).
        x = self.conv(sub10_pdp)
        x = self.proj(x)
        x = self.pre_fc_pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return self.ru_id_head(x)


def ru_id_loss(ru_id_logits, labels, criterion=None):
    """Cross-entropy loss on global RU ID."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    device = ru_id_logits.device
    if isinstance(labels, dict):
        global_ru_ids = labels['global_ru_id'].to(device)
    else:
        global_ru_ids = torch.tensor(
            [label['global_ru_id'] for label in labels], dtype=torch.long, device=device
        )

    return criterion(ru_id_logits, global_ru_ids)


def train_epoch(model, train_loader, optimizer, device, criterion=None):
    model.train()
    total_loss = 0.0
    ru_id_correct = 0
    total = 0

    for data, labels, _user_ids in tqdm(train_loader, desc='Train', leave=False):
        sub10_pdp = data['sub10_pdp'].to(device)
        sub10_pdp = normalize_pdp_db(sub10_pdp)

        optimizer.zero_grad()
        ru_id_logits = model(sub10_pdp)
        loss = ru_id_loss(ru_id_logits, labels, criterion=criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if isinstance(labels, dict):
            global_ru_ids = labels['global_ru_id'].to(device)
        else:
            global_ru_ids = torch.tensor(
                [label['global_ru_id'] for label in labels], dtype=torch.long, device=device
            )

        _, preds = torch.max(ru_id_logits.data, 1)
        total += global_ru_ids.size(0)
        ru_id_correct += (preds == global_ru_ids).sum().item()

    avg_loss = total_loss / len(train_loader)
    ru_id_acc = 100.0 * ru_id_correct / total
    return avg_loss, ru_id_acc


def validate(model, val_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    ru_id_correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, _user_ids in tqdm(val_loader, desc='Validate', leave=False):
            sub10_pdp = data['sub10_pdp'].to(device)
            sub10_pdp = normalize_pdp_db(sub10_pdp)

            ru_id_logits = model(sub10_pdp)
            loss = ru_id_loss(ru_id_logits, labels, criterion=criterion)
            total_loss += loss.item()

            if isinstance(labels, dict):
                global_ru_ids = labels['global_ru_id'].to(device)
            else:
                global_ru_ids = torch.tensor(
                    [label['global_ru_id'] for label in labels], dtype=torch.long, device=device
                )

            _, preds = torch.max(ru_id_logits.data, 1)
            total += global_ru_ids.size(0)
            ru_id_correct += (preds == global_ru_ids).sum().item()

    avg_loss = total_loss / len(val_loader)
    ru_id_acc = 100.0 * ru_id_correct / total
    return avg_loss, ru_id_acc


def main():
    # Paths
    sub10_pdp_path = '../../dataset/office_space_inline/sub_10ghz_pdp_32_taps'
    labels_path = '../../dataset/office_space_inline/ru_selection_labels/results.csv'

    # Training config
    mode = 'sub10'
    batch_size = 32
    epochs = 100
    lr = 3e-4
    use_class_weights = True

    # Model config
    conv_channels = 16
    conv_layers = 3
    proj_channels = 16
    pooled_ue = 2
    pooled_ru = 2
    pooled_taps = 4
    fc_size = 256
    dropout = 0.3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_root = Path('../pdp32_results') / 'conv3d'
    checkpoint_dir = results_root / 'checkpoints'
    log_dir = results_root / 'logs'
    plot_dir = results_root / 'plots'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f'Using device: {device}')
    print('Loading PDP32 dataset...')

    dataset = PDP32Dataset(
        sub10_pdp_path=sub10_pdp_path,
        labels_path=labels_path,
        mode=mode,
    )

    total_samples = len(dataset)
    if total_samples < 10:
        raise ValueError('Dataset too small for train/val/test split')

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_data, _sample_label, _sample_user_id = dataset[0]
    pdp_shape = sample_data['sub10_pdp'].shape
    num_aps = pdp_shape[0]
    sample_pdp_raw = sample_data['sub10_pdp']
    sample_pdp_norm = normalize_pdp_db(sample_pdp_raw)

    print('Sample shapes:')
    print(f'  sub10_pdp shape = {pdp_shape}')
    print(f'  num_aps = {num_aps}')
    print(
        f'  sub10_pdp raw range = [{sample_pdp_raw.min().item():.2f}, {sample_pdp_raw.max().item():.2f}] dB'
    )
    print(
        f'  sub10_pdp normalized range = [{sample_pdp_norm.min().item():.3f}, {sample_pdp_norm.max().item():.3f}]'
    )
    print(f'Train/Val/Test sizes: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}')

    # Class weights from TRAIN split only
    num_ru_ids = dataset.num_ru_ids
    class_counts = torch.zeros(num_ru_ids, dtype=torch.float32)
    base_ds = train_dataset.dataset
    for idx in train_dataset.indices:
        user_id = base_ds.valid_users[idx]
        ru_id = base_ds.labels_dict[user_id]['global_ru_id']
        class_counts[ru_id] += 1.0

    present_mask = class_counts > 0
    n_present = int(present_mask.sum().item())
    n_train = float(class_counts.sum().item())

    class_weights = torch.zeros(num_ru_ids, dtype=torch.float32)
    class_weights[present_mask] = n_train / (n_present * class_counts[present_mask])
    class_weights[present_mask] /= class_weights[present_mask].mean()

    if use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print(
            f'Using class weights from train split: present_classes={n_present}/{num_ru_ids}, '
            f'min={class_weights[present_mask].min():.4f}, max={class_weights[present_mask].max():.4f}'
        )
    else:
        criterion = nn.CrossEntropyLoss()
        print('Using unweighted CrossEntropyLoss')

    model = RUSelectionPDP32Conv3D(
        num_aps=num_aps,
        num_ru_ids=num_ru_ids,
        conv_channels=conv_channels,
        conv_layers=conv_layers,
        proj_channels=proj_channels,
        pooled_ue=pooled_ue,
        pooled_ru=pooled_ru,
        pooled_taps=pooled_taps,
        fc_size=fc_size,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {num_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    run_name = 'pdp32_conv3d_ru_only'
    writer = SummaryWriter(str(log_dir / run_name))
    best_ckpt_path = str(checkpoint_dir / 'best_model_pdp32_conv3d_ru_only.pt')

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')

    print('\nStarting training...\n')
    for epoch in range(epochs):
        t0 = time.perf_counter()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion=criterion)
        val_loss, val_acc = validate(model, val_loader, device, criterion=criterion)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train_ru_id', train_acc, epoch)
        writer.add_scalar('Accuracy/val_ru_id', val_acc, epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f'Epoch [{epoch + 1:03d}/{epochs}]  '
            f'train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  '
            f'val_loss={val_loss:.4f}  val_acc={val_acc:.2f}%  '
            f'time={time.perf_counter() - t0:.2f}s'
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ru_id_acc': val_acc,
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
                        'pooled_taps': pooled_taps,
                        'fc_size': fc_size,
                        'dropout': dropout,
                        'use_class_weights': use_class_weights,
                    },
                },
                best_ckpt_path,
            )

    writer.close()

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc = validate(model, test_loader, device, criterion=criterion)

    # --- Collect test-set predictions for NMSE CDF evaluation ---
    print('\nRunning NMSE CDF evaluation on test set...')
    model.eval()
    all_ue_ids, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for data, labels, user_ids in tqdm(test_loader, desc='Inference', leave=False):
            sub10_pdp = data['sub10_pdp'].to(device)
            sub10_pdp = normalize_pdp_db(sub10_pdp)
            logits = model(sub10_pdp)
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()
            batch_labels = labels['global_ru_id'].tolist()
            batch_uids = user_ids.tolist() if isinstance(user_ids, torch.Tensor) else list(user_ids)
            all_ue_ids.extend(batch_uids)
            all_labels.extend(batch_labels)
            all_preds.extend(batch_preds)

    raw_data_path = '../../dataset/office_space_inline/ru_selection_labels/flickering_raw_data'
    evaluate_nmse_cdf(
        ue_ids=all_ue_ids,
        global_ru_id_labels=all_labels,
        global_ru_id_predictions=all_preds,
        results_csv_path=labels_path,
        raw_data_path=raw_data_path,
        save_path=str(plot_dir / 'nmse_cdf_test_pdp32_conv3d.png'),
        title='NMSE CDF: Optimal vs Conv3D RU Selection (PDP32)',
    )

    print('\nTraining finished')
    print(f'Best val loss: {best_val_loss:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    print(f'Test RU ID accuracy: {test_acc:.2f}%')
    print(f'Best checkpoint: {best_ckpt_path}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(train_accs, label='Train RU ID Acc')
    axes[1].plot(val_accs, label='Val RU ID Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plot_path = str(plot_dir / 'training_plots_pdp32_conv3d_ru_only.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f'Training curves saved to: {plot_path}')


if __name__ == '__main__':
    main()