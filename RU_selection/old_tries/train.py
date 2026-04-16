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
from tqdm import tqdm

# Add parent directory to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import CsiDataset


# ==============================================================================
# Multi-Task RU Selection Model
# ==============================================================================
class RUSelectionModel(nn.Module):
    """
    Multi-task RU selection model with 3 outputs:
    1. Global RU ID (which RU to select)
    2. UE Beam ID (which beam the UE should use)
    3. RU Beam ID (which beam the RU should use)
    """
    def __init__(self, input_size=1000, hidden_size=512, num_ru_ids=19, num_beam_ids=7):
        super(RUSelectionModel, self).__init__()
        self.input_size = input_size
        
        # Shared feature extraction backbone
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Task-specific intermediate layers (different capacities for different output sizes)
        self.ru_id_intermediate = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.beam_intermediate = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.ru_id_head = nn.Linear(hidden_size // 2, num_ru_ids)
        self.ue_beam_id_head = nn.Linear(hidden_size // 4, num_beam_ids)
        self.ru_beam_id_head = nn.Linear(hidden_size // 4, num_beam_ids)
    
    def forward(self, sub10_channel, subthz_channel):
        """
        Args:
            sub10_channel: (batch_size, nr_APs, ue_ants, ru_ants, subcarriers) or None
            subthz_channel: (batch_size, nr_RUs, ue_ants, ru_ants, subcarriers) or None
        
        Returns:
            ru_id_logits: (batch_size, num_ru_ids)
            ue_beam_id_logits: (batch_size, num_beam_ids)
            ru_beam_id_logits: (batch_size, num_beam_ids)
        """
        # Concatenate available channels
        channels = []
        
        if sub10_channel is not None:
            channels.append(sub10_channel.reshape(sub10_channel.shape[0], -1))
        
        if subthz_channel is not None:
            channels.append(subthz_channel.reshape(subthz_channel.shape[0], -1))
        
        # If both are available, concatenate them; otherwise use the one available
        if len(channels) > 1:
            x = torch.cat(channels, dim=1)
        else:
            x = channels[0]
        
        # Pad or truncate to expected input size
        if x.shape[1] != self.input_size:
            if x.shape[1] < self.input_size:
                # Pad with zeros
                x = torch.cat([x, torch.zeros(x.shape[0], self.input_size - x.shape[1], device=x.device)], dim=1)
            else:
                # Truncate
                x = x[:, :self.input_size]
        
        # Shared backbone
        features = self.backbone(x)
        
        # Task-specific heads with intermediate layers
        ru_id_logits = self.ru_id_head(self.ru_id_intermediate(features))
        ue_beam_id_logits = self.ue_beam_id_head(self.beam_intermediate(features))
        ru_beam_id_logits = self.ru_beam_id_head(self.beam_intermediate(features))
        
        return ru_id_logits, ue_beam_id_logits, ru_beam_id_logits


class RUSelectionConv3DModel(nn.Module):
    """3D Conv-based RU selection model"""
    def __init__(
        self,
        num_ru_ids=160,
        num_beam_ids=7,
        conv_channels=32,
        conv_layers=3,
        fc_size=512,
        dropout=0.3,
    ):
        super(RUSelectionConv3DModel, self).__init__()

        self.num_ru_ids = num_ru_ids
        self.num_beam_ids = num_beam_ids

        # 3D conv stack with adaptive handling of spatial dims
        in_channels = 2  # real + imag
        layers = []
        for i in range(conv_layers):
            out_channels = conv_channels * (2**i)
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU())
            # Only apply pooling on earlier layers; skip on later layers to avoid spatial collapse
            if i < conv_layers - 1:
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # Reduce to feature vector
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc_shared = nn.Sequential(
            nn.Linear(in_channels, fc_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_size, fc_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Task-specific intermediate layers (different capacities for different output sizes)
        self.ru_id_intermediate = nn.Sequential(
            nn.Linear(fc_size // 2, fc_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.beam_intermediate = nn.Sequential(
            nn.Linear(fc_size // 2, fc_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.ru_id_head = nn.Linear(fc_size // 2, num_ru_ids)
        self.ue_beam_id_head = nn.Linear(fc_size // 4, num_beam_ids)
        self.ru_beam_id_head = nn.Linear(fc_size // 4, num_beam_ids)

    def _preprocess(self, x):
        # x shape: (batch, d1, d2, d3, subcarriers, 2)
        # Reduce over subcarriers to make it 3D spatial + channels
        if x is None or x.numel() == 0:
            return None

        x = x.mean(dim=-2)  # shape: (batch, d1, d2, d3, 2)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # shape: (batch, 2, d1, d2, d3)
        return x

    def forward(self, sub10_channel, subthz_channel):
        modalities = []
        for x in [sub10_channel, subthz_channel]:
            if x is not None and x.numel() > 0:
                x = self._preprocess(x)
                # Apply conv layers with adaptive pooling
                x = self.conv(x)
                x = self.pool(x).view(x.shape[0], -1)
                modalities.append(x)

        if not modalities:
            raise ValueError("No channel inputs supplied to Conv3D model")

        x = torch.cat(modalities, dim=1) if len(modalities) > 1 else modalities[0]
        x = self.fc_shared(x)

        # Task-specific heads with intermediate layers
        ru_id_logits = self.ru_id_head(self.ru_id_intermediate(x))
        ue_beam_id_logits = self.ue_beam_id_head(self.beam_intermediate(x))
        ru_beam_id_logits = self.ru_beam_id_head(self.beam_intermediate(x))

        return ru_id_logits, ue_beam_id_logits, ru_beam_id_logits


# ==============================================================================
# Dummy Loss Function - Replace with your actual loss
# ==============================================================================
def multi_task_loss(ru_id_logits, ue_beam_id_logits, ru_beam_id_logits, labels, weights=None):
    """
    Multi-task loss function for RU selection.
    
    Args:
        ru_id_logits: Global RU ID predictions (batch_size, num_ru_ids)
        ue_beam_id_logits: UE beam ID predictions (batch_size, num_beam_ids)
        ru_beam_id_logits: RU beam ID predictions (batch_size, num_beam_ids)
        labels: Batch of label dicts
        weights: Loss weights for each task [ru_id_weight, ue_beam_weight, ru_beam_weight]
                 Default: [1.0, 1.0, 1.0]
    
    Returns:
        total_loss: scalar tensor
        loss_dict: dict with individual losses for logging
    """
    if weights is None:
        weights = [1.0, 1.0, 1.0]
    
    criterion = nn.CrossEntropyLoss()
    device = ru_id_logits.device
    
    # Extract targets from labels - support dict-of-tensors (default collate) and list-of-dicts (non-collate)
    if isinstance(labels, dict):
        global_ru_ids = labels['global_ru_id'].to(device)
        ue_beam_ids = labels['ue_beam_id'].to(device)
        ru_beam_ids = labels['ru_beam_id'].to(device)
    else:
        global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)
        ue_beam_ids = torch.tensor([label['ue_beam_id'] for label in labels], dtype=torch.long, device=device)
        ru_beam_ids = torch.tensor([label['ru_beam_id'] for label in labels], dtype=torch.long, device=device)

    # Compute individual losses
    ru_id_loss = criterion(ru_id_logits, global_ru_ids) * weights[0]
    ue_beam_loss = criterion(ue_beam_id_logits, ue_beam_ids) * weights[1]
    ru_beam_loss = criterion(ru_beam_id_logits, ru_beam_ids) * weights[2]
    
    # Total loss
    total_loss = ru_id_loss + ue_beam_loss + ru_beam_loss
    
    loss_dict = {
        'ru_id_loss': ru_id_loss.item(),
        'ue_beam_loss': ue_beam_loss.item(),
        'ru_beam_loss': ru_beam_loss.item(),
        'total_loss': total_loss.item()
    }
    
    return total_loss, loss_dict


# ==============================================================================
# Training Functions
# ==============================================================================
def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    ru_id_correct = 0
    ue_beam_correct = 0
    ru_beam_correct = 0
    total = 0
    
    for batch_idx, (data, labels, user_ids) in enumerate(tqdm(train_loader, desc='Train', leave=False)):
        # Move data to device
        sub10_channel = data['sub10_channel']
        subthz_channel = data['subthz_channel']
        
        if sub10_channel is not None:
            sub10_channel = sub10_channel.to(device)
        if subthz_channel is not None:
            subthz_channel = subthz_channel.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        ru_id_logits, ue_beam_logits, ru_beam_logits = model(sub10_channel, subthz_channel)
        
        # Calculate loss
        loss, loss_dict = multi_task_loss(ru_id_logits, ue_beam_logits, ru_beam_logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Calculate accuracy for each task (supports dict-of-tensors and list-of-dicts labels)
        if isinstance(labels, dict):
            global_ru_ids = labels['global_ru_id'].to(device)
            ue_beam_ids = labels['ue_beam_id'].to(device)
            ru_beam_ids = labels['ru_beam_id'].to(device)
        else:
            global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)
            ue_beam_ids = torch.tensor([label['ue_beam_id'] for label in labels], dtype=torch.long, device=device)
            ru_beam_ids = torch.tensor([label['ru_beam_id'] for label in labels], dtype=torch.long, device=device)

        _, ru_id_pred = torch.max(ru_id_logits.data, 1)
        _, ue_beam_pred = torch.max(ue_beam_logits.data, 1)
        _, ru_beam_pred = torch.max(ru_beam_logits.data, 1)
        
        total += global_ru_ids.size(0)
        ru_id_correct += (ru_id_pred == global_ru_ids).sum().item()
        ue_beam_correct += (ue_beam_pred == ue_beam_ids).sum().item()
        ru_beam_correct += (ru_beam_pred == ru_beam_ids).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    ru_id_acc = 100 * ru_id_correct / total
    ue_beam_acc = 100 * ue_beam_correct / total
    ru_beam_acc = 100 * ru_beam_correct / total
    
    return avg_loss, ru_id_acc, ue_beam_acc, ru_beam_acc


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    ru_id_correct = 0
    ue_beam_correct = 0
    ru_beam_correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels, user_ids in tqdm(val_loader, desc='Validate', leave=False):
            # Move data to device
            sub10_channel = data['sub10_channel']
            subthz_channel = data['subthz_channel']
            
            if sub10_channel is not None:
                sub10_channel = sub10_channel.to(device)
            if subthz_channel is not None:
                subthz_channel = subthz_channel.to(device)
            
            # Forward pass
            ru_id_logits, ue_beam_logits, ru_beam_logits = model(sub10_channel, subthz_channel)
            
            # Calculate loss
            loss, loss_dict = multi_task_loss(ru_id_logits, ue_beam_logits, ru_beam_logits, labels)
            total_loss += loss.item()
            
            # Calculate accuracy for each task (supports dict-of-tensors and list-of-dicts labels)
            if isinstance(labels, dict):
                global_ru_ids = labels['global_ru_id'].to(device)
                ue_beam_ids = labels['ue_beam_id'].to(device)
                ru_beam_ids = labels['ru_beam_id'].to(device)
            else:
                global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)
                ue_beam_ids = torch.tensor([label['ue_beam_id'] for label in labels], dtype=torch.long, device=device)
                ru_beam_ids = torch.tensor([label['ru_beam_id'] for label in labels], dtype=torch.long, device=device)

            _, ru_id_pred = torch.max(ru_id_logits.data, 1)
            _, ue_beam_pred = torch.max(ue_beam_logits.data, 1)
            _, ru_beam_pred = torch.max(ru_beam_logits.data, 1)
            
            total += global_ru_ids.size(0)
            ru_id_correct += (ru_id_pred == global_ru_ids).sum().item()
            ue_beam_correct += (ue_beam_pred == ue_beam_ids).sum().item()
            ru_beam_correct += (ru_beam_pred == ru_beam_ids).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    ru_id_acc = 100 * ru_id_correct / total
    ue_beam_acc = 100 * ue_beam_correct / total
    ru_beam_acc = 100 * ru_beam_correct / total
    
    return avg_loss, ru_id_acc, ue_beam_acc, ru_beam_acc


# ==============================================================================
# Main Training Script
# ==============================================================================
def main():
    # ==============================================================================
    # Training Parameters - Edit these to change training configuration
    # ==============================================================================
    subthz_path = '../dataset/office_space_inline/sub_thz_channels'
    sub10_path = '../dataset/office_space_inline/sub_10ghz_channels'
    labels_path = '../dataset/office_space_inline/ru_selection_labels/results.csv'
    
    mode = 'sub10'  # Choose: 'sub10', 'subTHz', or 'combined'
    batch_size = 32
    epochs = 50
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = 'checkpoints'
    log_dir = 'runs'
    
    # Model selection and hyperparameters
    model_type = 'conv3d'  # Choose: 'mlp' or 'conv3d'
    
    # MLP hyperparameters
    mlp_input_size = 1000
    mlp_hidden_size = 512
    
    # Conv3D hyperparameters
    conv3d_channels = 32
    conv3d_layers = 3
    conv3d_fc_size = 256
    conv3d_dropout = 0.3

    
    # Create checkpoint and log directories
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # ==============================================================================
    # Load Dataset
    # ==============================================================================
    print("\nLoading dataset...")
    train_dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode
    )
    
    # Split into train/val/test (80/10/10)
    total_samples = len(train_dataset)
    if total_samples < 10:
        raise ValueError("Dataset too small for train/val/test split")

    # reproducibility
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * total_samples)
    val_size = int(0.1 * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging, increase for production
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Data sample shape overview (raw + flattened)
    sample_data, sample_label, sample_user_id = train_dataset[0]
    sub10_raw_shape = sample_data['sub10_channel'].shape
    subthz_raw_shape = sample_data['subthz_channel'].shape
    sub10_flat = int(np.prod(sub10_raw_shape)) if sub10_raw_shape != () else 0
    subthz_flat = int(np.prod(subthz_raw_shape)) if subthz_raw_shape != () else 0
    print("\nSample data shapes:")
    print(f"  sub10_channel raw shape = {sub10_raw_shape}, flattened = {sub10_flat}")
    print(f"  subthz_channel raw shape = {subthz_raw_shape}, flattened = {subthz_flat}")
    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # ==============================================================================
    # Initialize Model, Optimizer, and TensorBoard
    # ==============================================================================
    print("\nInitializing model...")
    # Get number of RU IDs from dataset
    num_ru_ids = train_dataset.dataset.num_ru_ids
    num_beam_ids = 7  # 7 beam angles: [-30, -20, -10, 0, 10, 20, 30]

    if model_type == 'mlp':
        model = RUSelectionModel(
            input_size=mlp_input_size,
            hidden_size=mlp_hidden_size,
            num_ru_ids=num_ru_ids,
            num_beam_ids=num_beam_ids
        )
    elif model_type == 'conv3d':
        model = RUSelectionConv3DModel(
            num_ru_ids=num_ru_ids,
            num_beam_ids=num_beam_ids,
            conv_channels=conv3d_channels,
            conv_layers=conv3d_layers,
            fc_size=conv3d_fc_size,
            dropout=conv3d_dropout
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'mlp' or 'conv3d'.")

    model = model.to(device)

    # Print model summary info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model type: {model_type}")
    print(f"Model initialized with {num_ru_ids} global RU IDs and {num_beam_ids} beam IDs")
    print(f"Trainable parameters: {num_params:,}")
    print("Model architecture:")
    print(model)

    # Hyperparameter overview
    print("\nHyperparameters:")
    print(f"  mode = {mode}")
    print(f"  model_type = {model_type}")
    print(f"  batch_size = {batch_size}")
    print(f"  epochs = {epochs}")
    print(f"  lr = {lr}")
    print(f"  optimizer = Adam")
    print(f"  scheduler = StepLR(step_size=10, gamma=0.5)")
    print(f"  device = {device}")
    print(f"  seed = 42")
    print(f"  train/val/test split = {train_size}/{val_size}/{test_size}")
    
    if model_type == 'mlp':
        print(f"\n  MLP specific:")
        print(f"    input_size = {mlp_input_size}")
        print(f"    hidden_size = {mlp_hidden_size}")
    elif model_type == 'conv3d':
        print(f"\n  Conv3D specific:")
        print(f"    conv_channels = {conv3d_channels}")
        print(f"    conv_layers = {conv3d_layers}")
        print(f"    fc_size = {conv3d_fc_size}")
        print(f"    dropout = {conv3d_dropout}")

    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    train_ru_id_accs = []
    val_ru_id_accs = []
    train_ue_beam_accs = []
    val_ue_beam_accs = []
    train_ru_beam_accs = []
    val_ru_beam_accs = []
    
    # ==============================================================================
    # Training Loop
    # ==============================================================================
    print("\nStarting training...\n")
    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        
        # Train
        train_loss, train_ru_id_acc, train_ue_beam_acc, train_ru_beam_acc = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_ru_id_acc, val_ue_beam_acc, val_ru_beam_acc = validate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train_ru_id', train_ru_id_acc, epoch)
        writer.add_scalar('Accuracy/train_ue_beam', train_ue_beam_acc, epoch)
        writer.add_scalar('Accuracy/train_ru_beam', train_ru_beam_acc, epoch)
        writer.add_scalar('Accuracy/val_ru_id', val_ru_id_acc, epoch)
        writer.add_scalar('Accuracy/val_ue_beam', val_ue_beam_acc, epoch)
        writer.add_scalar('Accuracy/val_ru_beam', val_ru_beam_acc, epoch)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    RU ID Acc: {train_ru_id_acc:.2f}% | UE Beam Acc: {train_ue_beam_acc:.2f}% | RU Beam Acc: {train_ru_beam_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    RU ID Acc: {val_ru_id_acc:.2f}% | UE Beam Acc: {val_ue_beam_acc:.2f}% | RU Beam Acc: {val_ru_beam_acc:.2f}%")
        
        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_{model_type}_{mode}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ru_id_acc': val_ru_id_acc,
            }, checkpoint_path)
            print(f"  Saved best checkpoint to {checkpoint_path}")
        
        # Track metrics for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ru_id_accs.append(train_ru_id_acc)
        val_ru_id_accs.append(val_ru_id_acc)
        train_ue_beam_accs.append(train_ue_beam_acc)
        val_ue_beam_accs.append(val_ue_beam_acc)
        train_ru_beam_accs.append(train_ru_beam_acc)
        val_ru_beam_accs.append(val_ru_beam_acc)
        
        print()
    
    writer.close()
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Tensorboard logs saved to {log_dir}")
    print(f"Checkpoints saved to {checkpoint_dir}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_ru_id_acc, test_ue_beam_acc, test_ru_beam_acc = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"  RU ID Acc: {test_ru_id_acc:.2f}% | UE Beam Acc: {test_ue_beam_acc:.2f}% | RU Beam Acc: {test_ru_beam_acc:.2f}%")

    # ==============================================================================
    # Plot Loss and Accuracy
    # ==============================================================================
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Loss
    axes[0, 0].plot(train_losses, label='Train Loss', marker='o', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', marker='s', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot RU ID Accuracy
    axes[0, 1].plot(train_ru_id_accs, label='Train RU ID Acc', marker='o', linewidth=2)
    axes[0, 1].plot(val_ru_id_accs, label='Val RU ID Acc', marker='s', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('RU ID Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot UE Beam Accuracy
    axes[1, 0].plot(train_ue_beam_accs, label='Train UE Beam Acc', marker='o', linewidth=2)
    axes[1, 0].plot(val_ue_beam_accs, label='Val UE Beam Acc', marker='s', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('UE Beam ID Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot RU Beam Accuracy
    axes[1, 1].plot(train_ru_beam_accs, label='Train RU Beam Acc', marker='o', linewidth=2)
    axes[1, 1].plot(val_ru_beam_accs, label='Val RU Beam Acc', marker='s', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('RU Beam ID Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining curves saved to {plot_path}")
    plt.show()


if __name__ == '__main__':
    main()
