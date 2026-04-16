import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy as np
import time
from tqdm import tqdm

# Add parent directory to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import CsiDataset


# ==============================================================================
# MLP RU Selection Model (only ru_id output)
# ==============================================================================
class RUSelectionMLPModel(nn.Module):
    def __init__(self, input_size=1000, hidden_size=512, num_ru_ids=160):
        super(RUSelectionMLPModel, self).__init__()
        self.input_size = input_size

        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.ru_id_head = nn.Linear(hidden_size // 2, num_ru_ids)

    def forward(self, sub10_channel, subthz_channel):
        channels = []
        if sub10_channel is not None:
            channels.append(sub10_channel.reshape(sub10_channel.shape[0], -1))
        if subthz_channel is not None:
            channels.append(subthz_channel.reshape(subthz_channel.shape[0], -1))

        x = torch.cat(channels, dim=1) if len(channels) > 1 else channels[0]

        if x.shape[1] != self.input_size:
            if x.shape[1] < self.input_size:
                x = torch.cat([x, torch.zeros(x.shape[0], self.input_size - x.shape[1], device=x.device)], dim=1)
            else:
                x = x[:, :self.input_size]

        return self.ru_id_head(self.backbone(x))


# ==============================================================================
# Conv3D RU Selection Model — frequency-averaging variant
#
# Preprocessing:
#   Input:  (batch, n_aps, ue_ant, ru_ant, carriers, 2)
#   Step 1: mean over carriers dim → (batch, n_aps, ue_ant, ru_ant, 2)
#   Step 2: permute → (batch, 2, n_aps, ue_ant, ru_ant)
#   Conv3D sees: 2 input channels, spatial dims (n_aps × ue_ant × ru_ant)
#
# Architecture:
#   Conv3D stack (plain stride=1, padding=1) → AdaptiveAvgPool3d(1,1,1) → flatten → FC head
# ==============================================================================
class RUSelectionConv3DFreqAvg(nn.Module):
    """
    Conv3D RU selection model that averages over subcarriers before convolution.

    The carrier mean collapses 1024 frequencies into a single spatially-structured
    value (Re + Im separately), giving Conv3D a clean 3-D spatial input.
    """
    def __init__(
        self,
        num_ru_ids=160,
        conv_input_feature_dim=None,
        conv_channels=32,
        conv_layers=3,
        fc_size=512,
        dropout=0.3,
    ):
        super(RUSelectionConv3DFreqAvg, self).__init__()

        self.num_ru_ids = num_ru_ids
        self.conv_input_feature_dim = conv_input_feature_dim

        if self.conv_input_feature_dim is None:
            raise ValueError("conv_input_feature_dim must be provided.")

        # 2 input channels: averaged Re and Im parts
        in_channels = 2

        layers = []
        for i in range(conv_layers):
            out_channels = conv_channels * (2 ** i)
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # Collapse all spatial dims to 1×1×1
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_shared = nn.Sequential(
            nn.Linear(self.conv_input_feature_dim, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.ru_id_head = nn.Linear(fc_size, num_ru_ids)

    def _preprocess(self, x):
        """
        Input:  (batch, n_aps, ue_ant, ru_ant, carriers, 2)
        Output: (batch, 2, n_aps, ue_ant, ru_ant)

        Averages the carrier dimension (dim=-2) to compress 1024 frequencies
        into a single value per spatial location, keeping Re and Im as channels.
        """
        if x is None or x.numel() == 0:
            return None
        # Average over subcarriers: (..., carriers, 2) → (..., 2)
        x = x.mean(dim=-2)                      # (batch, n_aps, ue_ant, ru_ant, 2)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (batch, 2, n_aps, ue_ant, ru_ant)
        return x

    def forward(self, sub10_channel, subthz_channel):
        modalities = []
        for x in [sub10_channel, subthz_channel]:
            if x is not None and x.numel() > 0:
                x = self._preprocess(x)          # (batch, 2, d1, d2, d3)
                x = self.conv(x)
                x = self.pool(x).view(x.shape[0], -1)  # (batch, final_ch)
                modalities.append(x)

        if not modalities:
            raise ValueError("No channel inputs supplied to Conv3D model")

        x = torch.cat(modalities, dim=1) if len(modalities) > 1 else modalities[0]
        x = self.fc_shared(x)
        return self.ru_id_head(x)


# ==============================================================================
# Loss Function
# ==============================================================================
def ru_id_loss(ru_id_logits, labels, criterion=None):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    device = ru_id_logits.device

    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion = type(criterion)(weight=criterion.weight.to(device),
                                    label_smoothing=criterion.label_smoothing)

    if isinstance(labels, dict):
        global_ru_ids = labels['global_ru_id'].to(device)
    else:
        global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)

    return criterion(ru_id_logits, global_ru_ids)


# ==============================================================================
# Training Functions
# ==============================================================================
def train_epoch(model, train_loader, optimizer, device, criterion=None):
    model.train()
    total_loss = 0.0
    ru_id_correct = 0
    total = 0

    for batch_idx, (data, labels, user_ids) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
        sub10_channel = data['sub10_channel']
        subthz_channel = data['subthz_channel']

        if sub10_channel is not None:
            sub10_channel = sub10_channel.to(device)
        if subthz_channel is not None:
            subthz_channel = subthz_channel.to(device)

        optimizer.zero_grad()
        ru_id_logits = model(sub10_channel, subthz_channel)
        loss = ru_id_loss(ru_id_logits, labels, criterion=criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if isinstance(labels, dict):
            global_ru_ids = labels['global_ru_id'].to(device)
        else:
            global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)

        _, ru_id_pred = torch.max(ru_id_logits.data, 1)
        total += global_ru_ids.size(0)
        ru_id_correct += (ru_id_pred == global_ru_ids).sum().item()

    return total_loss / len(train_loader), 100 * ru_id_correct / total


def validate(model, val_loader, device, criterion=None):
    model.eval()
    total_loss = 0.0
    ru_id_correct = 0
    total = 0

    with torch.no_grad():
        for data, labels, user_ids in tqdm(val_loader, desc='Validate', leave=False):
            sub10_channel = data['sub10_channel']
            subthz_channel = data['subthz_channel']

            if sub10_channel is not None:
                sub10_channel = sub10_channel.to(device)
            if subthz_channel is not None:
                subthz_channel = subthz_channel.to(device)

            ru_id_logits = model(sub10_channel, subthz_channel)
            loss = ru_id_loss(ru_id_logits, labels, criterion=criterion)
            total_loss += loss.item()

            if isinstance(labels, dict):
                global_ru_ids = labels['global_ru_id'].to(device)
            else:
                global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)

            _, ru_id_pred = torch.max(ru_id_logits.data, 1)
            total += global_ru_ids.size(0)
            ru_id_correct += (ru_id_pred == global_ru_ids).sum().item()

    return total_loss / len(val_loader), 100 * ru_id_correct / total


# ==============================================================================
# Main Training Script
# ==============================================================================
def main():
    # ==============================================================================
    # Training Parameters
    # ==============================================================================
    subthz_path = '../dataset/office_space_inline/sub_thz_channels'
    sub10_path  = '../dataset/office_space_inline/sub_10ghz_channels'
    labels_path = '../dataset/office_space_inline/ru_selection_labels/results.csv'

    mode           = 'sub10'   # 'sub10', 'subTHz', or 'combined'
    batch_size     = 32
    epochs         = 150
    lr             = 1e-3
    device         = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = 'checkpoints'
    log_dir        = 'runs'

    # Model selection
    model_type = 'conv3d'  # 'mlp' or 'conv3d'

    # MLP hyperparameters
    mlp_input_size  = 1000
    mlp_hidden_size = 512

    # Conv3D (freq-avg) hyperparameters
    conv3d_channels  = 32    # base channel count; layer i uses channels * 2^i
    conv3d_layers    = 3     # number of conv layers
    conv3d_fc_size   = 512
    conv3d_dropout   = 0.3

    use_class_weights = True  # Set False to use unweighted CrossEntropyLoss

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    print(f"Using device: {device}")

    # ==============================================================================
    # Load Dataset
    # ==============================================================================
    print("\nLoading dataset...")
    full_dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode
    )

    total_samples = len(full_dataset)
    if total_samples < 10:
        raise ValueError("Dataset too small for train/val/test split")

    generator  = torch.Generator().manual_seed(42)
    train_size = int(0.8 * total_samples)
    val_size   = int(0.1 * total_samples)
    test_size  = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=0)

    # Sample shape info
    sample_data, sample_label, sample_user_id = train_dataset[0]
    sub10_raw_shape  = sample_data['sub10_channel'].shape
    subthz_raw_shape = sample_data['subthz_channel'].shape
    print("\nSample data shapes:")
    print(f"  sub10_channel  raw shape = {sub10_raw_shape}")
    print(f"  subthz_channel raw shape = {subthz_raw_shape}")

    # Feature dim per modality: conv produces conv_channels * 2^(layers-1) channels,
    # AdaptiveAvgPool3d(1,1,1) collapses spatial → flat size = that channel count.
    def conv_modality_feature_dim(raw_shape, conv_channels, conv_layers):
        if len(raw_shape) < 5 or raw_shape[0] == 0:
            return 0
        return conv_channels * (2 ** (conv_layers - 1))

    feat_sub10  = conv_modality_feature_dim(sub10_raw_shape,  conv3d_channels, conv3d_layers)
    feat_subthz = conv_modality_feature_dim(subthz_raw_shape, conv3d_channels, conv3d_layers)

    if feat_sub10 > 0 and feat_subthz > 0:
        conv_input_feature_dim = feat_sub10 + feat_subthz
    elif feat_sub10 > 0:
        conv_input_feature_dim = feat_sub10
    elif feat_subthz > 0:
        conv_input_feature_dim = feat_subthz
    else:
        conv_input_feature_dim = 0

    print(f"  conv_input_feature_dim = {conv_input_feature_dim}")
    print(f"\nTrain / Val / Test: {train_size} / {val_size} / {test_size}")

    # ==============================================================================
    # Class Imbalance Weighting
    # ==============================================================================
    num_ru_ids_for_weights = train_dataset.dataset.num_ru_ids
    class_counts = torch.zeros(num_ru_ids_for_weights, dtype=torch.float32)
    base_ds = train_dataset.dataset
    for idx in train_dataset.indices:
        user_id = base_ds.valid_users[idx]
        ru_id   = base_ds.labels_dict[user_id]['global_ru_id']
        class_counts[ru_id] += 1.0

    n_present    = (class_counts > 0).sum().item()
    n_train      = class_counts.sum().item()
    class_weights = torch.zeros(num_ru_ids_for_weights, dtype=torch.float32)
    present_mask  = class_counts > 0
    class_weights[present_mask] = n_train / (n_present * class_counts[present_mask])
    class_weights[present_mask] /= class_weights[present_mask].mean()

    if use_class_weights:
        print(f"\nClass weights ({n_present}/{num_ru_ids_for_weights} classes present):")
        print(f"  min={class_weights[present_mask].min():.4f}  "
              f"max={class_weights[present_mask].max():.4f}  "
              f"mean={class_weights[present_mask].mean():.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\nClass weighting disabled — using unweighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    # ==============================================================================
    # Initialize Model
    # ==============================================================================
    print("\nInitializing model...")
    num_ru_ids = train_dataset.dataset.num_ru_ids

    if model_type == 'mlp':
        model = RUSelectionMLPModel(
            input_size=mlp_input_size,
            hidden_size=mlp_hidden_size,
            num_ru_ids=num_ru_ids
        )
    elif model_type == 'conv3d':
        model = RUSelectionConv3DFreqAvg(
            num_ru_ids=num_ru_ids,
            conv_input_feature_dim=conv_input_feature_dim,
            conv_channels=conv3d_channels,
            conv_layers=conv3d_layers,
            fc_size=conv3d_fc_size,
            dropout=conv3d_dropout,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'mlp' or 'conv3d'.")

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {model_type}")
    print(f"Num RU IDs: {num_ru_ids}")
    print(f"Trainable parameters: {num_params:,}")
    print(model)

    print("\nHyperparameters:")
    print(f"  mode              = {mode}")
    print(f"  model_type        = {model_type}")
    print(f"  batch_size        = {batch_size}")
    print(f"  epochs            = {epochs}")
    print(f"  lr                = {lr}")
    print(f"  optimizer         = Adam")
    print(f"  scheduler         = StepLR(step_size=10, gamma=0.5)")
    print(f"  device            = {device}")
    print(f"  seed              = 42")
    print(f"  use_class_weights = {use_class_weights}")
    if model_type == 'conv3d':
        print(f"\n  Conv3D (freq-avg) specific:")
        print(f"    preprocessing       = mean over carriers → permute to (batch,2,d1,d2,d3)")
        print(f"    conv_channels       = {conv3d_channels}  (layer i → channels*2^i)")
        print(f"    conv_layers         = {conv3d_layers}")
        print(f"    pool                = AdaptiveAvgPool3d(1,1,1)")
        print(f"    conv_input_feature_dim = {conv_input_feature_dim}")
        print(f"    fc_size             = {conv3d_fc_size}")
        print(f"    dropout             = {conv3d_dropout}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    writer    = SummaryWriter(log_dir)

    best_val_loss   = float('inf')
    train_losses    = []
    val_losses      = []
    train_ru_id_accs = []
    val_ru_id_accs  = []

    # ==============================================================================
    # Training Loop
    # ==============================================================================
    print("\nStarting training...\n")
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        epoch_start = time.perf_counter()

        train_loss, train_ru_id_acc = train_epoch(model, train_loader, optimizer, device, criterion=criterion)
        val_loss,   val_ru_id_acc   = validate(model, val_loader, device, criterion=criterion)
        scheduler.step()

        writer.add_scalar('Loss/train',            train_loss,       epoch)
        writer.add_scalar('Loss/val',              val_loss,         epoch)
        writer.add_scalar('Accuracy/train_ru_id',  train_ru_id_acc,  epoch)
        writer.add_scalar('Accuracy/val_ru_id',    val_ru_id_acc,    epoch)

        print(f"  Train Loss: {train_loss:.4f}  RU Acc: {train_ru_id_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  RU Acc: {val_ru_id_acc:.2f}%")
        print(f"  Epoch time: {time.perf_counter() - epoch_start:.2f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(checkpoint_dir, f'best_model_{model_type}_{mode}_freq_avg.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ru_id_acc': val_ru_id_acc,
            }, ckpt_path)
            print(f"  Saved best checkpoint → {ckpt_path}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ru_id_accs.append(train_ru_id_acc)
        val_ru_id_accs.append(val_ru_id_acc)
        print()

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_ru_id_acc = validate(model, test_loader, device, criterion=criterion)
    print(f"Test Loss: {test_loss:.4f}  RU Acc: {test_ru_id_acc:.2f}%")

    # ==============================================================================
    # Plots
    # ==============================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_losses,    label='Train Loss', marker='o')
    axes[0].plot(val_losses,      label='Val Loss',   marker='s')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(train_ru_id_accs, label='Train RU Acc', marker='o')
    axes[1].plot(val_ru_id_accs,   label='Val RU Acc',   marker='s')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('RU ID Accuracy'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, f'training_plots_{model_type}_{mode}_freq_avg.png')
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
