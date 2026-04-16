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
from evaluation import evaluate_nmse_cdf


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
    return sub10_channel.permute(0, 1, 2, 3, 4, 5).reshape(b, n_aps, ue, ru, sc * ri)


class RUSelectionCSIConv3D(nn.Module):
    """Conv3D model for RU selection from sub-10 GHz CSI inputs."""

    def __init__(
        self,
        num_aps,
        num_ru_ids=160,
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
        self.ru_id_head = nn.Linear(fc_size, num_ru_ids)

    def forward(self, x):
        # x: (batch, n_aps, ue_ants, ru_ants, subcarriers*2)
        x = self.conv(x)
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
        subthz_csi = data['subthz_channel'].to(device)
        subthz_csi = prepare_csi_input(subthz_csi)
        subthz_csi = normalize_csi(subthz_csi)

        optimizer.zero_grad()
        ru_id_logits = model(subthz_csi)
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
            subthz_csi = data['subthz_channel'].to(device)
            subthz_csi = prepare_csi_input(subthz_csi)
            subthz_csi = normalize_csi(subthz_csi)

            ru_id_logits = model(subthz_csi)
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
    dataset_folder = 'office_space_reduced_inline'
    sub10_path = f'../dataset/{dataset_folder}/sub_10ghz_channels'
    subthz_path = f'../dataset/{dataset_folder}/sub_thz_channels'

    # Label mode: 'optimal_beams'  — best RU with optimal beam pair (from results.csv)
    #             'fixed_beams'    — best RU for fixed 0° beams (from results_fixed_beams_0deg.csv)
    label_mode = 'fixed_beams'

    if label_mode == 'fixed_beams':
        labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results_fixed_beams_0deg.csv'
    else:
        labels_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'

    config_path = Path(__file__).parent.parent / 'dataset' / dataset_folder / 'config_used.yaml'
    with open(config_path) as f:
        dataset_config = yaml.safe_load(f)
    num_stripes = dataset_config['stripe_config']['N_stripes']
    num_rus_per_stripe = dataset_config['stripe_config']['N_RUs']


    # Training config
    mode = 'subTHz'
    batch_size = 32
    epochs = 20
    lr = 3e-4
    use_class_weights = False # Set to True to use class weights from train split for CrossEntropyLoss (in case of class imbalance)

    # Model config
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
        f'csi_conv3d'
        f'_ep{epochs}_bs{batch_size}_lr{lr:.0e}'
        f'_cc{conv_channels}_cl{conv_layers}_fc{fc_size}_do{dropout}'
        f'_{label_mode}'
        f'_{timestamp}'
    )
    run_dir = Path('stored_models_subTHz_input') / run_name
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
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_data, _sample_label, _sample_user_id = dataset[0]
    subthz_csi_raw = sample_data['subthz_channel']
    csi_prepared = prepare_csi_input(subthz_csi_raw.unsqueeze(0)).squeeze(0)  # (n_rus, ue, ru, sc*2)
    num_rus = csi_prepared.shape[0]
    num_ue_ants = csi_prepared.shape[1]
    num_ru_ants = csi_prepared.shape[2]


    print('Dataset summary:')
    print(f'  dataset_folder           = {dataset_folder}')
    print(f'  label_mode               = {label_mode}')
    print(f'  num_stripes              = {num_stripes}')
    print(f'  num_rus_per_stripe       = {num_rus_per_stripe}')
    print(f'  num_ru_ids               = {num_stripes * num_rus_per_stripe} - check: {dataset.dataset.num_ru_ids}')
    print(f'  subthz_channel raw shape  = {tuple(subthz_csi_raw.shape)}')
    print(f'  subthz_channel conv input = {tuple(csi_prepared.shape)}')
    print(f'  train/val/test sizes     = {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}')


    # Class weights from TRAIN split only
    num_ru_ids = dataset.dataset.num_ru_ids
    class_counts = torch.zeros(num_ru_ids, dtype=torch.float32)
    # train_dataset is a Subset of a Subset; drill down to the original CsiDataset
    outer_subset = train_dataset.dataset   # the filtered Subset
    base_ds = outer_subset.dataset         # the original CsiDataset
    for idx in train_dataset.indices:
        original_idx = outer_subset.indices[idx]
        user_id = base_ds.valid_users[original_idx]
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

    model = RUSelectionCSIConv3D(
        num_aps=num_rus,
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
    print(f'  use_class_weights        = {use_class_weights}')
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

    # Log all run parameters to a YAML file in the run directory
    run_params = {
        'run_name': run_name,
        'dataset': {
            'dataset_folder': dataset_folder,
            'label_mode': label_mode,
            'mode': mode,
            'num_stripes': num_stripes,
            'num_rus_per_stripe': num_rus_per_stripe,
            'num_ru_ids': num_ru_ids,
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
            'use_class_weights': use_class_weights,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_factor': 0.5,
            'scheduler_patience': 10,
        },
        'architecture': {
            'num_aps': num_rus,
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
        scheduler.step(val_loss)

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
                        'label_mode': label_mode,
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

    raw_data_path = f'../dataset/{dataset_folder}/ru_selection_labels/flickering_raw_data'
    results_csv_path = f'../dataset/{dataset_folder}/ru_selection_labels/results.csv'
    evaluate_nmse_cdf(
        ue_ids=all_ue_ids,
        global_ru_id_labels=all_labels,
        global_ru_id_predictions=all_preds,
        results_csv_path=results_csv_path,
        raw_data_path=raw_data_path,
        num_rus_per_stripe=num_rus_per_stripe,
        save_path=str(run_dir / 'nmse_cdf_test.png'),
        title='NMSE CDF: Optimal vs Conv3D RU Selection (CSI)',
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
    plot_path = str(run_dir / 'training_plots.png')
    plt.savefig(plot_path)
    tikz_path = str(run_dir / 'training_plots.tex')
    matplot2tikz.save(tikz_path)
    plt.close(fig)
    print(f'Training curves saved to: {plot_path}')
    print(f'TikZ saved to: {tikz_path}')


if __name__ == '__main__':
    main()
