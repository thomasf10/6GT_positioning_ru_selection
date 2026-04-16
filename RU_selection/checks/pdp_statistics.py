import sys
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

# Add workspace root to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.dataloaders import CsiDataset


# ==============================================================================
# User-selectable settings
# ==============================================================================
subthz_path = '../dataset/office_space_inline/sub_thz_channels'
sub10_path = '../dataset/office_space_inline/sub_10ghz_channels'
labels_path = '../dataset/office_space_inline/ru_selection_labels/results.csv'

mode = 'sub10'  # Choose: 'sub10', 'subTHz', or 'combined'


def compute_pdp_stats(csi_complex):
    """
    Compute PDP statistics from CSI.
    
    Args:
        csi_complex: Complex CSI array of shape (carriers,)
    
    Returns:
        dict with keys: power_linear_min, power_linear_max, power_db_min, power_db_max
    """
    pdp_complex = np.fft.ifft(csi_complex)
    pdp_power = np.abs(pdp_complex) ** 2
    pdp_db = 10 * np.log10(pdp_power + 1e-12)
    
    return {
        'power_linear_min': np.min(pdp_power),
        'power_linear_max': np.max(pdp_power),
        'power_db_min': np.min(pdp_db),
        'power_db_max': np.max(pdp_db),
    }


def extract_csi_antenna_pair(channel_tensor, ap_idx, ue_idx, tx_idx):
    """
    Extract complex CSI for a single antenna pair.
    
    Args:
        channel_tensor: Tensor of shape (n_aps, ue_ants, ru_ants, carriers, 2)
        ap_idx, ue_idx, tx_idx: Antenna pair indices
    
    Returns:
        Complex array of shape (carriers,)
    """
    if isinstance(channel_tensor, np.ndarray):
        channel_np = channel_tensor
    else:
        channel_np = np.asarray(channel_tensor)
    
    if channel_np.ndim != 5:
        raise ValueError(f'Expected shape (n_aps, ue_ants, ru_ants, carriers, 2), got {channel_np.shape}')
    
    n_aps, n_ue_ants, n_tx_ants, n_carriers, two_dim = channel_np.shape
    if two_dim != 2:
        raise ValueError(f'Last dimension must be 2, got {two_dim}')
    
    if not (0 <= ap_idx < n_aps):
        raise IndexError(f'ap_idx {ap_idx} out of range [0, {n_aps - 1}]')
    if not (0 <= ue_idx < n_ue_ants):
        raise IndexError(f'ue_idx {ue_idx} out of range [0, {n_ue_ants - 1}]')
    if not (0 <= tx_idx < n_tx_ants):
        raise IndexError(f'tx_idx {tx_idx} out of range [0, {n_tx_ants - 1}]')
    
    pair = channel_np[ap_idx, ue_idx, tx_idx, :, :]
    return pair[:, 0] + 1j * pair[:, 1]


def main():
    print(f"Loading dataset in '{mode}' mode...")
    dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
    )
    
    print(f"Found {len(dataset)} valid users\n")
    
    # Collect all statistics
    all_stats = []
    
    for sample_idx in range(len(dataset)):
        if sample_idx % 100 == 0:
            print(f"Processing sample {sample_idx} / {len(dataset)}...")
        
        try:
            data, label, user_id = dataset[sample_idx]
            
            # Get channel data
            channel_data = data.get('sub10_channel') if mode == 'sub10' else data.get('subthz_channel')
            if channel_data is None or channel_data.size == 0:
                continue
            
            # Convert to numpy if needed
            if hasattr(channel_data, 'numpy'):
                channel_np = channel_data.numpy()
            else:
                channel_np = np.asarray(channel_data)
            
            n_aps, n_ue_ants, n_tx_ants, n_carriers, two_dim = channel_np.shape
            
            # Iterate over all antenna pairs
            for ap_idx in range(n_aps):
                for ue_idx in range(n_ue_ants):
                    for tx_idx in range(n_tx_ants):
                        try:
                            csi_complex = extract_csi_antenna_pair(channel_np, ap_idx, ue_idx, tx_idx)
                            stats = compute_pdp_stats(csi_complex)
                            
                            all_stats.append({
                                'user_id': user_id,
                                'ap_idx': ap_idx,
                                'ue_ant_idx': ue_idx,
                                'tx_ant_idx': tx_idx,
                                'power_linear_min': stats['power_linear_min'],
                                'power_linear_max': stats['power_linear_max'],
                                'power_db_min': stats['power_db_min'],
                                'power_db_max': stats['power_db_max'],
                            })
                        except Exception as e:
                            print(f"  Error processing user {user_id}, AP {ap_idx}, UE {ue_idx}, TX {tx_idx}: {e}")
                            continue
        
        except Exception as e:
            print(f"Error loading sample {sample_idx}: {e}")
            continue
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(all_stats)
    
    print("\n" + "="*100)
    print("PDP STATISTICS SUMMARY")
    print("="*100)
    
    # Overall statistics
    print("\n** OVERALL STATISTICS ACROSS ALL ANTENNA PAIRS **")
    print(f"Total antenna pairs: {len(df)}")
    print(f"\nLinear Power (PDP):")
    print(f"  Min - Global minimum: {df['power_linear_min'].min():.6e}")
    print(f"  Min - Mean:           {df['power_linear_min'].mean():.6e}")
    print(f"  Min - Median:         {df['power_linear_min'].median():.6e}")
    print(f"  Min - Max:            {df['power_linear_min'].max():.6e}")
    print(f"  Max - Global minimum: {df['power_linear_max'].min():.6e}")
    print(f"  Max - Mean:           {df['power_linear_max'].mean():.6e}")
    print(f"  Max - Median:         {df['power_linear_max'].median():.6e}")
    print(f"  Max - Global maximum: {df['power_linear_max'].max():.6e}")
    
    print(f"\ndB Power (PDP):")
    print(f"  Min - Global minimum: {df['power_db_min'].min():.2f} dB")
    print(f"  Min - Mean:           {df['power_db_min'].mean():.2f} dB")
    print(f"  Min - Median:         {df['power_db_min'].median():.2f} dB")
    print(f"  Min - Max:            {df['power_db_min'].max():.2f} dB")
    print(f"  Max - Global minimum: {df['power_db_max'].min():.2f} dB")
    print(f"  Max - Mean:           {df['power_db_max'].mean():.2f} dB")
    print(f"  Max - Median:         {df['power_db_max'].median():.2f} dB")
    print(f"  Max - Global maximum: {df['power_db_max'].max():.2f} dB")
    
    # Per-user statistics
    print("\n" + "="*100)
    print("** PER-USER STATISTICS **")
    print("="*100)
    user_stats = df.groupby('user_id').agg({
        'power_linear_min': ['min', 'mean', 'max'],
        'power_linear_max': ['min', 'mean', 'max'],
        'power_db_min': ['min', 'mean', 'max'],
        'power_db_max': ['min', 'mean', 'max'],
    })
    
    print("\nPower Linear Min per user (min / mean / max):")
    for user_id, row in user_stats.iterrows():
        print(f"  User {user_id:4d}: {row[('power_linear_min', 'min')]:10.6e} / {row[('power_linear_min', 'mean')]:10.6e} / {row[('power_linear_min', 'max')]:10.6e}")
    
    print("\nPower Linear Max per user (min / mean / max):")
    for user_id, row in user_stats.iterrows():
        print(f"  User {user_id:4d}: {row[('power_linear_max', 'min')]:10.6e} / {row[('power_linear_max', 'mean')]:10.6e} / {row[('power_linear_max', 'max')]:10.6e}")
    
    print("\nPower dB Min per user (min / mean / max):")
    for user_id, row in user_stats.iterrows():
        print(f"  User {user_id:4d}: {row[('power_db_min', 'min')]:8.2f} / {row[('power_db_min', 'mean')]:8.2f} / {row[('power_db_min', 'max')]:8.2f} dB")
    
    print("\nPower dB Max per user (min / mean / max):")
    for user_id, row in user_stats.iterrows():
        print(f"  User {user_id:4d}: {row[('power_db_max', 'min')]:8.2f} / {row[('power_db_max', 'mean')]:8.2f} / {row[('power_db_max', 'max')]:8.2f} dB")
    
    # Per-AP statistics
    print("\n" + "="*100)
    print("** PER-AP STATISTICS **")
    print("="*100)
    ap_stats = df.groupby('ap_idx').agg({
        'power_linear_min': ['min', 'mean', 'max'],
        'power_linear_max': ['min', 'mean', 'max'],
        'power_db_min': ['min', 'mean', 'max'],
        'power_db_max': ['min', 'mean', 'max'],
    })
    
    print("\nPower Linear Min per AP (min / mean / max):")
    for ap_idx, row in ap_stats.iterrows():
        print(f"  AP {ap_idx:2d}: {row[('power_linear_min', 'min')]:10.6e} / {row[('power_linear_min', 'mean')]:10.6e} / {row[('power_linear_min', 'max')]:10.6e}")
    
    print("\nPower Linear Max per AP (min / mean / max):")
    for ap_idx, row in ap_stats.iterrows():
        print(f"  AP {ap_idx:2d}: {row[('power_linear_max', 'min')]:10.6e} / {row[('power_linear_max', 'mean')]:10.6e} / {row[('power_linear_max', 'max')]:10.6e}")
    
    print("\nPower dB Min per AP (min / mean / max):")
    for ap_idx, row in ap_stats.iterrows():
        print(f"  AP {ap_idx:2d}: {row[('power_db_min', 'min')]:8.2f} / {row[('power_db_min', 'mean')]:8.2f} / {row[('power_db_min', 'max')]:8.2f} dB")
    
    print("\nPower dB Max per AP (min / mean / max):")
    for ap_idx, row in ap_stats.iterrows():
        print(f"  AP {ap_idx:2d}: {row[('power_db_max', 'min')]:8.2f} / {row[('power_db_max', 'mean')]:8.2f} / {row[('power_db_max', 'max')]:8.2f} dB")
    
    # Per-antenna pair statistics
    print("\n" + "="*100)
    print("** PER-ANTENNA-PAIR STATISTICS **")
    print("="*100)
    pair_stats = df.groupby(['ue_ant_idx', 'tx_ant_idx']).agg({
        'power_linear_min': ['min', 'mean', 'max'],
        'power_linear_max': ['min', 'mean', 'max'],
        'power_db_min': ['min', 'mean', 'max'],
        'power_db_max': ['min', 'mean', 'max'],
    })
    
    print("\nPower Linear Min per antenna pair (min / mean / max):")
    for (ue_idx, tx_idx), row in pair_stats.iterrows():
        print(f"  UE_ant {ue_idx} / TX_ant {tx_idx}: {row[('power_linear_min', 'min')]:10.6e} / {row[('power_linear_min', 'mean')]:10.6e} / {row[('power_linear_min', 'max')]:10.6e}")
    
    print("\nPower Linear Max per antenna pair (min / mean / max):")
    for (ue_idx, tx_idx), row in pair_stats.iterrows():
        print(f"  UE_ant {ue_idx} / TX_ant {tx_idx}: {row[('power_linear_max', 'min')]:10.6e} / {row[('power_linear_max', 'mean')]:10.6e} / {row[('power_linear_max', 'max')]:10.6e}")
    
    print("\nPower dB Min per antenna pair (min / mean / max):")
    for (ue_idx, tx_idx), row in pair_stats.iterrows():
        print(f"  UE_ant {ue_idx} / TX_ant {tx_idx}: {row[('power_db_min', 'min')]:8.2f} / {row[('power_db_min', 'mean')]:8.2f} / {row[('power_db_min', 'max')]:8.2f} dB")
    
    print("\nPower dB Max per antenna pair (min / mean / max):")
    for (ue_idx, tx_idx), row in pair_stats.iterrows():
        print(f"  UE_ant {ue_idx} / TX_ant {tx_idx}: {row[('power_db_max', 'min')]:8.2f} / {row[('power_db_max', 'mean')]:8.2f} / {row[('power_db_max', 'max')]:8.2f} dB")
    
    # Save detailed results to CSV
    output_csv = f'pdp_statistics_{mode}.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n** Detailed results saved to {output_csv} **")


if __name__ == '__main__':
    main()
