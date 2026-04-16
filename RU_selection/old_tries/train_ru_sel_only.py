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
import itertools
from tqdm import tqdm

# Add parent directory to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataloaders import CsiDataset


# ==============================================================================
# MLP RU Selection Model (only ru_id output)
# ==============================================================================
class RUSelectionMLPModel(nn.Module):
    """
    Simple MLP RU selection model with single output:
    Global RU ID (which RU to select)
    """
    def __init__(self, input_size=1000, hidden_size=512, num_ru_ids=160):
        super(RUSelectionMLPModel, self).__init__()
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
        
        # Single output head for RU ID
        self.ru_id_head = nn.Linear(hidden_size // 2, num_ru_ids)
    
    def forward(self, sub10_channel, subthz_channel):
        """
        Args:
            sub10_channel: (batch_size, nr_APs, ue_ants, ru_ants, subcarriers) or None
            subthz_channel: (batch_size, nr_RUs, ue_ants, ru_ants, subcarriers) or None
        
        Returns:
            ru_id_logits: (batch_size, num_ru_ids)
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
        
        # RU ID output
        ru_id_logits = self.ru_id_head(features)
        
        return ru_id_logits


# ==============================================================================
# Conv3D RU Selection Model (only ru_id output)
# ==============================================================================
class RUSelectionConv3DModelOnly(nn.Module):
    """3D Conv-based RU selection model for ru_id only"""
    def __init__(
        self,
        num_ru_ids=160,
        num_aps=1,
        input_type='frequency',
        pdp_num_taps=128,
        conv_input_feature_dim=None,
        conv_channels=32,
        conv_layers=3,
        conv_proj_channels=16,
        conv_tap_stride=2,
        conv_tap_dilation_growth=0,
        pre_fc_pool_ue=2,
        pre_fc_pool_ru=2,
        pre_fc_pool_taps=8,
        fc_size=512,
        dropout=0.3,
    ):
        super(RUSelectionConv3DModelOnly, self).__init__()

        self.num_ru_ids = num_ru_ids
        self.num_aps = num_aps
        self.input_type = input_type
        self.pdp_num_taps = pdp_num_taps
        self.conv_input_feature_dim = conv_input_feature_dim
        self.conv_proj_channels = conv_proj_channels
        self.conv_tap_stride = conv_tap_stride
        self.conv_tap_dilation_growth = conv_tap_dilation_growth
        self._printed_preprocess_shapes = False
        self._printed_forward_shapes = False

        if self.conv_input_feature_dim is None:
            raise ValueError("conv_input_feature_dim must be provided for non-lazy Conv3D head.")

        # Input channels depend on representation
        # frequency: 2 (Re/Im) per AP
        # pdp: 1 (power) per AP
        if input_type == 'frequency':
            in_channels = 2 * num_aps
        elif input_type == 'pdp':
            in_channels = num_aps
        else:
            raise ValueError(f"Unknown input_type: {input_type}. Choose 'frequency' or 'pdp'.")
        
        # 3D conv stack without pooling to preserve spatial information
        layers = []
        for i in range(conv_layers):
            out_channels = conv_channels * (2**i)
            dilation_tap = 1 + i * conv_tap_dilation_growth
            stride_tap = conv_tap_stride if i > 0 else 1
            layers.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(1, 1, stride_tap),
                    dilation=(1, 1, dilation_tap),
                    padding=(1, 1, dilation_tap),
                )
            )
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)

        # Learned channel compression before flattening to keep parameter count manageable
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, conv_proj_channels, kernel_size=1),
            nn.ReLU(),
        )

        # Adaptive pooling right before FC to control feature size
        self.pre_fc_pool = nn.AdaptiveAvgPool3d((pre_fc_pool_ue, pre_fc_pool_ru, pre_fc_pool_taps))

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
        Preprocess CSI tensor for Conv3D.

        Input shape: (batch, n_aps, ue_ant, ru_ant, carriers, 2)
        Output shape depends on input_type:
          - 'frequency': (batch, 2*n_aps, ue_ant, ru_ant, carriers)
          - 'pdp': (batch, n_aps, ue_ant, ru_ant, pdp_num_taps)
        """
        if x is None or x.numel() == 0:
            return None

        if not self._printed_preprocess_shapes:
            print(f"[preprocess] input_type={self.input_type}")
            print(f"[preprocess] raw input shape: {tuple(x.shape)}")
            print("[preprocess] raw dims: (batch, n_aps, ue_ant, ru_ant, carriers, re_im)")

        batch_size = x.shape[0]
        n_aps = x.shape[1]
        ue_ant = x.shape[2]
        ru_ant = x.shape[3]
        carriers = x.shape[4]
        
        if self.input_type == 'frequency':
            # Reshape to (batch, 2*n_aps, ue_ant, ru_ant, carriers)
            # Treats each AP's Re and Im as separate input channels
            x = x.permute(0, 1, 5, 2, 3, 4).contiguous()  # (batch, n_aps, 2, ue_ant, ru_ant, carriers)
            x = x.reshape(batch_size, 2 * n_aps, ue_ant, ru_ant, carriers)
            if not self._printed_preprocess_shapes:
                print(f"[preprocess] frequency output shape: {tuple(x.shape)}")
                print("[preprocess] frequency dims: (batch, channels=2*n_aps [Re/Im per AP], ue_ant, ru_ant, carriers)")
            
        elif self.input_type == 'pdp':
            # Convert to complex, compute PDP via IFFT, then power
            # x shape: (batch, n_aps, ue_ant, ru_ant, carriers, 2)
            # Convert Re/Im to complex
            x_complex = torch.complex(x[..., 0], x[..., 1])  # (batch, n_aps, ue_ant, ru_ant, carriers)
            
            # IFFT along carrier dimension (dim=-1) to get time domain (delay domain)
            x_time = torch.fft.ifft(x_complex, dim=-1)  # (batch, n_aps, ue_ant, ru_ant, carriers)
            
            # Power delay profile: magnitude squared
            x_pdp = torch.abs(x_time) ** 2  # (batch, n_aps, ue_ant, ru_ant, carriers)

            # Convert power to dB for a compressed dynamic range
            x_pdp = 10.0 * torch.log10(x_pdp + 1e-12)

            # Normalize each sample independently to [0, 1]
            pdp_min = x_pdp.amin(dim=(1, 2, 3, 4), keepdim=True)
            pdp_max = x_pdp.amax(dim=(1, 2, 3, 4), keepdim=True)
            x_pdp = (x_pdp - pdp_min) / (pdp_max - pdp_min + 1e-8)

            # Keep only configured number of taps (or zero-pad if requested taps exceed available carriers)
            if self.pdp_num_taps < carriers:
                x = x_pdp[..., :self.pdp_num_taps]
            elif self.pdp_num_taps > carriers:
                pad_len = self.pdp_num_taps - carriers
                pad = torch.zeros(*x_pdp.shape[:-1], pad_len, device=x_pdp.device, dtype=x_pdp.dtype)
                x = torch.cat([x_pdp, pad], dim=-1)
            else:
                x = x_pdp

            if not self._printed_preprocess_shapes:
                print(f"[preprocess] pdp taps requested: {self.pdp_num_taps}, carriers available: {carriers}")
                print(f"[preprocess] pdp output shape: {tuple(x.shape)}")
                print("[preprocess] pdp dims: (batch, channels=n_aps [power per AP], ue_ant, ru_ant, delay_taps)")
                print("[preprocess] pdp representation: 10*log10(power + eps), then per-sample min-max normalized to [0, 1]")

        if not self._printed_preprocess_shapes:
            self._printed_preprocess_shapes = True
        
        return x

    def forward(self, sub10_channel, subthz_channel):
        modalities = []
        for x in [sub10_channel, subthz_channel]:
            if x is not None and x.numel() > 0:
                x = self._preprocess(x)
                if not self._printed_forward_shapes:
                    print(f"[forward] after preprocess: {tuple(x.shape)}")
                x = self.conv(x)
                if not self._printed_forward_shapes:
                    print(f"[forward] after conv: {tuple(x.shape)}")
                    print("[forward] conv dims: (batch, conv_channels_out, ue_ant, ru_ant, reduced_taps)")
                x = self.proj(x)
                if not self._printed_forward_shapes:
                    print(f"[forward] after proj: {tuple(x.shape)}")
                    print("[forward] proj dims: (batch, conv_proj_channels, ue_ant, ru_ant, reduced_taps)")
                x = self.pre_fc_pool(x)
                if not self._printed_forward_shapes:
                    print(f"[forward] after pre_fc_pool: {tuple(x.shape)}")
                    print("[forward] pooled dims: (batch, conv_proj_channels, pooled_ue, pooled_ru, pooled_taps)")
                x = x.flatten(start_dim=1)
                if not self._printed_forward_shapes:
                    print(f"[forward] after flatten: {tuple(x.shape)}")
                modalities.append(x)

        if not modalities:
            raise ValueError("No channel inputs supplied to Conv3D model")

        x = torch.cat(modalities, dim=1) if len(modalities) > 1 else modalities[0]
        if not self._printed_forward_shapes:
            print(f"[forward] after modality concat: {tuple(x.shape)}")
        x = self.fc_shared(x)
        if not self._printed_forward_shapes:
            print(f"[forward] after fc_shared: {tuple(x.shape)}")

        ru_id_logits = self.ru_id_head(x)

        if not self._printed_forward_shapes:
            print(f"[forward] after ru_id_head: {tuple(ru_id_logits.shape)}")
            print("[forward] logits dims: (batch, num_ru_ids)")
            self._printed_forward_shapes = True

        return ru_id_logits


# ==============================================================================
# Loss Function - Simple Cross Entropy for RU ID only
# ==============================================================================
def ru_id_loss(ru_id_logits, labels, criterion=None):
    """
    Simple loss function for RU ID prediction only.
    
    Args:
        ru_id_logits: RU ID predictions (batch_size, num_ru_ids)
        labels: Batch of label dicts
        criterion: optional pre-built loss (e.g. with class weights); defaults to plain CrossEntropyLoss
    
    Returns:
        loss: scalar tensor
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    device = ru_id_logits.device
    
    # Move criterion weights to the same device if needed
    if hasattr(criterion, 'weight') and criterion.weight is not None:
        criterion = type(criterion)(weight=criterion.weight.to(device),
                                    label_smoothing=criterion.label_smoothing)
    
    # Extract targets from labels
    if isinstance(labels, dict):
        global_ru_ids = labels['global_ru_id'].to(device)
    else:
        global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)
    
    loss = criterion(ru_id_logits, global_ru_ids)
    
    return loss


# ==============================================================================
# Training Functions
# ==============================================================================
def train_epoch(model, train_loader, optimizer, device, criterion=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    ru_id_correct = 0
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
        ru_id_logits = model(sub10_channel, subthz_channel)
        
        # Calculate loss
        loss = ru_id_loss(ru_id_logits, labels, criterion=criterion)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Calculate accuracy
        if isinstance(labels, dict):
            global_ru_ids = labels['global_ru_id'].to(device)
        else:
            global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)
        
        _, ru_id_pred = torch.max(ru_id_logits.data, 1)
        
        total += global_ru_ids.size(0)
        ru_id_correct += (ru_id_pred == global_ru_ids).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    ru_id_acc = 100 * ru_id_correct / total
    
    return avg_loss, ru_id_acc


def validate(model, val_loader, device, criterion=None):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    ru_id_correct = 0
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
            ru_id_logits = model(sub10_channel, subthz_channel)
            
            # Calculate loss
            loss = ru_id_loss(ru_id_logits, labels, criterion=criterion)
            total_loss += loss.item()
            
            # Calculate accuracy
            if isinstance(labels, dict):
                global_ru_ids = labels['global_ru_id'].to(device)
            else:
                global_ru_ids = torch.tensor([label['global_ru_id'] for label in labels], dtype=torch.long, device=device)
            
            _, ru_id_pred = torch.max(ru_id_logits.data, 1)
            
            total += global_ru_ids.size(0)
            ru_id_correct += (ru_id_pred == global_ru_ids).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    ru_id_acc = 100 * ru_id_correct / total
    
    return avg_loss, ru_id_acc


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
    epochs = 100
    lr = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = 'checkpoints'
    log_dir = 'runs'
    
    # Model selection and hyperparameters
    model_type = 'conv3d'  # Choose: 'mlp' or 'conv3d'
    input_type = 'pdp'    # Choose: 'frequency' or 'pdp' (frequency domain or power delay profile)
    pdp_num_taps = 128    # Used only when input_type='pdp'
    
    # MLP hyperparameters
    mlp_input_size = 1000
    mlp_hidden_size = 512
    
    # Conv3D hyperparameters
    conv3d_channels = 16
    conv3d_layers = 3
    conv3d_proj_channels = 16
    conv3d_tap_stride = 2
    conv3d_tap_dilation_growth = 0
    pre_fc_pool_ue = 2
    pre_fc_pool_ru = 2
    pre_fc_pool_taps = 8
    conv3d_fc_size = 256
    conv3d_dropout = 0.3
    use_class_weights = True   # Set False to use unweighted CrossEntropyLoss

    # Hyperparameter search (Conv3D only)
    run_hparam_search = True
    max_hparam_trials = None  # Set an int to cap trials (e.g. 20), or None for full grid
    hparam_grid = {
        'pdp_num_taps': [16, 64],
        'conv3d_channels': [16, 32],
        'conv3d_layers': [2, 3],
        'conv3d_fc_size': [128, 256],
        'conv3d_dropout': [0.2, 0.5],
        'lr': [1e-3, 3e-4],
    }

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

    # Compute Conv3D flattened feature size after projection + adaptive pre-FC pooling
    def conv_modality_feature_dim(raw_shape, conv_proj_channels, pre_fc_pool_ue, pre_fc_pool_ru, pre_fc_pool_taps):
        # raw_shape is expected to be (n_aps, ue_ant, ru_ant, carriers, 2)
        if len(raw_shape) < 5:
            return 0
        n_aps_local = raw_shape[0]
        if n_aps_local == 0:
            return 0
        return int(conv_proj_channels * pre_fc_pool_ue * pre_fc_pool_ru * pre_fc_pool_taps)

    conv_feature_dim_sub10 = conv_modality_feature_dim(
        sub10_raw_shape,
        conv3d_proj_channels,
        pre_fc_pool_ue,
        pre_fc_pool_ru,
        pre_fc_pool_taps,
    )
    conv_feature_dim_subthz = conv_modality_feature_dim(
        subthz_raw_shape,
        conv3d_proj_channels,
        pre_fc_pool_ue,
        pre_fc_pool_ru,
        pre_fc_pool_taps,
    )

    # Match forward behavior: concatenate both modalities if both are present
    if conv_feature_dim_sub10 > 0 and conv_feature_dim_subthz > 0:
        conv_input_feature_dim = conv_feature_dim_sub10 + conv_feature_dim_subthz
    elif conv_feature_dim_sub10 > 0:
        conv_input_feature_dim = conv_feature_dim_sub10
    elif conv_feature_dim_subthz > 0:
        conv_input_feature_dim = conv_feature_dim_subthz
    else:
        conv_input_feature_dim = 0

    print(f"  conv_input_feature_dim = {conv_input_feature_dim}")
    
    # Extract num_aps from sample data (for Conv3D model)
    if len(sub10_raw_shape) > 0 and sub10_raw_shape[0] > 0:
        num_aps_sub10 = sub10_raw_shape[0]
    else:
        num_aps_sub10 = 1
    
    if len(subthz_raw_shape) > 0 and subthz_raw_shape[0] > 0:
        num_aps_subthz = subthz_raw_shape[0]
    else:
        num_aps_subthz = 1
    
    # Use the first available modality's num_aps (they should be consistent)
    num_aps = num_aps_sub10 if num_aps_sub10 > 0 else num_aps_subthz
    print(f"  Detected num_aps = {num_aps}")

    
    print(f"Train set size: {len(train_dataset)}")
    print(f"Val set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # ==============================================================================
    # Class Imbalance Weighting
    # ==============================================================================
    # Compute inverse-frequency class weights from the TRAINING set only.
    # Missing classes (never seen in training) get weight 0 so they don't affect loss.
    num_ru_ids_for_weights = train_dataset.dataset.num_ru_ids
    class_counts = torch.zeros(num_ru_ids_for_weights, dtype=torch.float32)
    base_ds = train_dataset.dataset
    for idx in train_dataset.indices:
        user_id = base_ds.valid_users[idx]
        ru_id = base_ds.labels_dict[user_id]['global_ru_id']
        class_counts[ru_id] += 1.0

    n_present = (class_counts > 0).sum().item()
    n_train = class_counts.sum().item()
    class_weights = torch.zeros(num_ru_ids_for_weights, dtype=torch.float32)
    present_mask = class_counts > 0
    # Balanced inverse-frequency: weight = n_train / (n_present * count)
    class_weights[present_mask] = n_train / (n_present * class_counts[present_mask])
    # Normalise so mean weight over present classes == 1 (keeps loss scale similar)
    class_weights[present_mask] /= class_weights[present_mask].mean()

    if use_class_weights:
        print(f"\nClass weights (training set, {n_present}/{num_ru_ids_for_weights} classes present):")
        print(f"  min weight = {class_weights[present_mask].min():.4f}")
        print(f"  max weight = {class_weights[present_mask].max():.4f}")
        print(f"  mean weight = {class_weights[present_mask].mean():.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("\nClass weighting disabled — using unweighted CrossEntropyLoss.")
        criterion = nn.CrossEntropyLoss()

    # ==============================================================================
    # Build Trial List (single run or hyperparameter grid search)
    # ==============================================================================
    if run_hparam_search and model_type != 'conv3d':
        raise ValueError("Hyperparameter search is implemented for model_type='conv3d' only.")

    if run_hparam_search:
        grid_keys = [
            'pdp_num_taps',
            'lr',
            'conv3d_channels',
            'conv3d_layers',
            'conv3d_fc_size',
            'conv3d_dropout',
        ]
        grid_values = [hparam_grid[k] for k in grid_keys]
        trial_configs = [dict(zip(grid_keys, vals)) for vals in itertools.product(*grid_values)]
        if max_hparam_trials is not None:
            trial_configs = trial_configs[:max_hparam_trials]
    else:
        trial_configs = [{
            'pdp_num_taps': pdp_num_taps,
            'lr': lr,
            'conv3d_channels': conv3d_channels,
            'conv3d_layers': conv3d_layers,
            'conv3d_fc_size': conv3d_fc_size,
            'conv3d_dropout': conv3d_dropout,
        }]

    print(f"\nPlanned trials: {len(trial_configs)}")
    if run_hparam_search:
        print("Hyperparameter search is ENABLED")
    else:
        print("Hyperparameter search is DISABLED (single run)")

    # Get number of RU IDs from dataset
    num_ru_ids = train_dataset.dataset.num_ru_ids
    trial_results = []

    for trial_idx, cfg in enumerate(trial_configs, start=1):
        trial_tag = f"trial_{trial_idx:03d}"
        trial_lr = cfg['lr']
        trial_pdp_num_taps = cfg['pdp_num_taps']
        trial_conv3d_channels = cfg['conv3d_channels']
        trial_conv3d_layers = cfg['conv3d_layers']
        trial_conv3d_fc_size = cfg['conv3d_fc_size']
        trial_conv3d_dropout = cfg['conv3d_dropout']

        # Compute Conv3D flattened feature size after projection + adaptive pre-FC pooling
        conv_feature_dim_sub10 = conv_modality_feature_dim(
            sub10_raw_shape,
            conv3d_proj_channels,
            pre_fc_pool_ue,
            pre_fc_pool_ru,
            pre_fc_pool_taps,
        )
        conv_feature_dim_subthz = conv_modality_feature_dim(
            subthz_raw_shape,
            conv3d_proj_channels,
            pre_fc_pool_ue,
            pre_fc_pool_ru,
            pre_fc_pool_taps,
        )
        if conv_feature_dim_sub10 > 0 and conv_feature_dim_subthz > 0:
            conv_input_feature_dim = conv_feature_dim_sub10 + conv_feature_dim_subthz
        elif conv_feature_dim_sub10 > 0:
            conv_input_feature_dim = conv_feature_dim_sub10
        elif conv_feature_dim_subthz > 0:
            conv_input_feature_dim = conv_feature_dim_subthz
        else:
            conv_input_feature_dim = 0

        print("\n" + "=" * 90)
        print(f"Starting {trial_tag} [{trial_idx}/{len(trial_configs)}]")
        print(f"  pdp_num_taps={trial_pdp_num_taps}, lr={trial_lr}, conv3d_channels={trial_conv3d_channels},")
        print(f"  conv3d_layers={trial_conv3d_layers}, conv3d_fc_size={trial_conv3d_fc_size}, conv3d_dropout={trial_conv3d_dropout}")
        print("=" * 90)

        # Build criterion per trial (same class weights, optional toggle)
        if use_class_weights:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        # Initialize model
        print("Initializing model...")
        if model_type == 'mlp':
            model = RUSelectionMLPModel(
                input_size=mlp_input_size,
                hidden_size=mlp_hidden_size,
                num_ru_ids=num_ru_ids
            )
        elif model_type == 'conv3d':
            model = RUSelectionConv3DModelOnly(
                num_ru_ids=num_ru_ids,
                num_aps=num_aps,
                input_type=input_type,
                pdp_num_taps=trial_pdp_num_taps,
                conv_input_feature_dim=conv_input_feature_dim,
                conv_channels=trial_conv3d_channels,
                conv_layers=trial_conv3d_layers,
                conv_proj_channels=conv3d_proj_channels,
                conv_tap_stride=conv3d_tap_stride,
                conv_tap_dilation_growth=conv3d_tap_dilation_growth,
                pre_fc_pool_ue=pre_fc_pool_ue,
                pre_fc_pool_ru=pre_fc_pool_ru,
                pre_fc_pool_taps=pre_fc_pool_taps,
                fc_size=trial_conv3d_fc_size,
                dropout=trial_conv3d_dropout
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choose 'mlp' or 'conv3d'.")

        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model type: {model_type}")
        print(f"Model initialized with {num_ru_ids} global RU IDs")
        print(f"Trainable parameters: {num_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=trial_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        trial_log_dir = os.path.join(log_dir, trial_tag)
        writer = SummaryWriter(trial_log_dir)

        best_val_loss = float('inf')
        best_checkpoint_path = os.path.join(
            checkpoint_dir,
            f'best_model_{model_type}_{mode}_ru_only_{trial_tag}.pt'
        )

        # Track losses for plotting
        train_losses = []
        val_losses = []
        train_ru_id_accs = []
        val_ru_id_accs = []

        print("Starting training...\n")
        for epoch in range(epochs):
            print(f"Epoch [{epoch + 1}/{epochs}]")
            epoch_start_time = time.perf_counter()

            train_loss, train_ru_id_acc = train_epoch(model, train_loader, optimizer, device, criterion=criterion)
            val_loss, val_ru_id_acc = validate(model, val_loader, device, criterion=criterion)

            scheduler.step()

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train_ru_id', train_ru_id_acc, epoch)
            writer.add_scalar('Accuracy/val_ru_id', val_ru_id_acc, epoch)

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"    RU ID Acc: {train_ru_id_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"    RU ID Acc: {val_ru_id_acc:.2f}%")
            epoch_elapsed = time.perf_counter() - epoch_start_time
            print(f"  Epoch Time: {epoch_elapsed:.2f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_ru_id_acc': val_ru_id_acc,
                    'trial_config': cfg,
                }, best_checkpoint_path)
                print(f"  Saved best checkpoint to {best_checkpoint_path}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_ru_id_accs.append(train_ru_id_acc)
            val_ru_id_accs.append(val_ru_id_acc)

            print()

        writer.close()
        print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
        print(f"Tensorboard logs saved to {trial_log_dir}")

        # Evaluate best checkpoint on test set
        ckpt = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        test_loss, test_ru_id_acc = validate(model, test_loader, device, criterion=criterion)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"  RU ID Acc: {test_ru_id_acc:.2f}%")

        # Plot Loss and Accuracy
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(train_losses, label='Train Loss', marker='o')
        axes[0].plot(val_losses, label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(train_ru_id_accs, label='Train RU ID Acc', marker='o')
        axes[1].plot(val_ru_id_accs, label='Val RU ID Acc', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation RU ID Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(log_dir, f'training_plots_{model_type}_{mode}_ru_only_{trial_tag}.png')
        plt.savefig(plot_path)
        print(f"Plots saved to {plot_path}")
        plt.close(fig)

        trial_results.append({
            'trial_idx': trial_idx,
            'trial_tag': trial_tag,
            'config': cfg,
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'test_ru_id_acc': test_ru_id_acc,
            'num_params': num_params,
            'checkpoint': best_checkpoint_path,
        })

    # Final summary across all trials
    best_trial = min(trial_results, key=lambda x: x['best_val_loss'])
    print("\n" + "=" * 90)
    print("Hyperparameter search summary")
    print(f"Total trials: {len(trial_results)}")
    print(f"Best trial: {best_trial['trial_tag']} (index {best_trial['trial_idx']})")
    print(f"Best val loss: {best_trial['best_val_loss']:.4f}")
    print(f"Best trial test loss: {best_trial['test_loss']:.4f}")
    print(f"Best trial test RU ID acc: {best_trial['test_ru_id_acc']:.2f}%")
    print(f"Best trial params: {best_trial['num_params']:,}")
    print("Best config:")
    for k, v in best_trial['config'].items():
        print(f"  {k} = {v}")
    print(f"Best checkpoint: {best_trial['checkpoint']}")

    summary_path = os.path.join(log_dir, 'hparam_search_summary_ru_only.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Hyperparameter search summary\n")
        f.write(f"Total trials: {len(trial_results)}\n")
        f.write(f"Best trial: {best_trial['trial_tag']}\n")
        f.write(f"Best val loss: {best_trial['best_val_loss']:.6f}\n")
        f.write(f"Best test loss: {best_trial['test_loss']:.6f}\n")
        f.write(f"Best test RU ID acc: {best_trial['test_ru_id_acc']:.4f}\n")
        f.write("Best config:\n")
        for k, v in best_trial['config'].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Best checkpoint: {best_trial['checkpoint']}\n\n")
        f.write("All trials:\n")
        for r in trial_results:
            f.write(
                f"{r['trial_tag']}: val={r['best_val_loss']:.6f}, "
                f"test={r['test_loss']:.6f}, acc={r['test_ru_id_acc']:.4f}, "
                f"params={r['num_params']}, cfg={r['config']}\n"
            )
    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
