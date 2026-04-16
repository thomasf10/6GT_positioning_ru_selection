"""
Analyze Power Delay Profile from CSI data.
Loads dataset, converts frequency-domain CSI to PDP, and visualizes sparsity.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add workspace root to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.dataloaders import CsiDataset


def csi_to_pdp(csi_complex):
    """
    Convert frequency-domain CSI to Power Delay Profile.
    
    Input: (n_aps, ue_ant, ru_ant, carriers, 2) with 2 being Re/Im
    Output: (n_aps, ue_ant, ru_ant, carriers) with power values
    """
    # Convert Re/Im to complex
    x_complex = torch.complex(csi_complex[..., 0], csi_complex[..., 1])
    
    # IFFT over carrier dimension to get time/delay domain
    x_time = torch.fft.ifft(x_complex, dim=-1)
    
    # Power delay profile: magnitude squared
    pdp = torch.abs(x_time) ** 2
    
    return pdp


def analyze_pdp_sparsity_single(pdp_single, num_taps_to_check=[10, 16]):
    """
    Compute energy concentration in first N taps for a single antenna pair PDP.
    
    Input: (carriers,)
    """
    total_energy = pdp_single.sum().item()
    if total_energy == 0:
        return {n: 0 for n in num_taps_to_check}, 0, pdp_single
    
    results = {}
    for n_taps in num_taps_to_check:
        if n_taps <= len(pdp_single):
            energy_in_taps = pdp_single[:n_taps].sum().item()
            percentage = 100 * energy_in_taps / total_energy
            results[n_taps] = percentage
    
    return results, total_energy, pdp_single


def main():
    # Dataset parameters
    subthz_path = '../dataset/office_space_inline/sub_thz_channels'
    sub10_path = '../dataset/office_space_inline/sub_10ghz_channels'
    labels_path = '../dataset/office_space_inline/ru_selection_labels/results.csv'
    
    mode = 'sub10'
    
    print("Loading dataset...")
    dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a few users
    num_users = 3
    sample_indices = np.random.choice(len(dataset), min(num_users, len(dataset)), replace=False)
    
    print(f"\nAnalyzing {len(sample_indices)} users with per-antenna-pair PDPs...\n")
    
    sparsity_stats = []
    
    for user_plot_idx, sample_idx in enumerate(sample_indices):
        data, label, user_id = dataset[sample_idx]
        
        sub10_channel = data['sub10_channel']  # torch tensor
        
        if sub10_channel.numel() == 0:
            print(f"Sample {sample_idx}: Empty channel, skipping")
            continue
        
        # Convert to PDP
        pdp = csi_to_pdp(sub10_channel)  # (n_aps, ue_ant, ru_ant, carriers)
        
        n_aps = pdp.shape[0]
        ue_ant = pdp.shape[1]
        ru_ant = pdp.shape[2]
        
        print(f"User {user_id} (sample {sample_idx}):")
        print(f"  Shape: {n_aps} APs × {ue_ant} UE antennas × {ru_ant} RU antennas = {n_aps * ue_ant * ru_ant} antenna pairs")
        
        # Create figure for this user with all antenna pairs
        num_antenna_pairs = n_aps * ue_ant * ru_ant
        grid_size = int(np.ceil(np.sqrt(num_antenna_pairs)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 12))
        fig.suptitle(f'User {user_id}: PDP per Antenna Pair', fontsize=14)
        axes = axes.flatten()
        
        pair_idx = 0
        for ap_idx in range(n_aps):
            for ue_idx in range(ue_ant):
                for ru_idx in range(ru_ant):
                    pdp_single = pdp[ap_idx, ue_idx, ru_idx, :]  # (carriers,)
                    
                    # Analyze this antenna pair
                    sparsity_data, total_energy, _ = analyze_pdp_sparsity_single(pdp_single)
                    sparsity_stats.append(sparsity_data)
                    
                    # Plot
                    ax = axes[pair_idx]
                    pdp_np = pdp_single.cpu().numpy()
                    
                    # Plot only first 16 taps as stem plot
                    pdp_np_truncated = pdp_np[:16]
                    markerline, stemlines, baseline = ax.stem(range(len(pdp_np_truncated)), pdp_np_truncated, basefmt=' ')
                    markerline.set_markerfacecolor('b')
                    markerline.set_markeredgecolor('b')
                    markerline.set_markersize(6)
                    stemlines.set_color('b')
                    stemlines.set_linewidth(1.5)
                    
                    ax.set_title(f'AP{ap_idx} UE{ue_idx} RU{ru_idx} (E:{total_energy:.2e})', fontsize=9)
                    ax.set_ylabel('Power')
                    ax.set_xlabel('Delay tap')
                    ax.set_xlim(-0.5, 15.5)
                    ax.grid(True, alpha=0.3, axis='y')
                    
                    pair_idx += 1
        
        # Hide unused subplots
        for idx in range(pair_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plot_path = f'pdp_analysis_user_{user_id}_16taps.png'
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved plot to {plot_path}\n")
        plt.close()
    
    print("="*60)
    print("OVERALL STATISTICS ACROSS ANTENNA PAIRS")
    print("="*60)
    
    if sparsity_stats:
        # Average the percentages
        num_taps_to_check = list(sparsity_stats[0].keys())
        avg_percentages = {}
        for n_taps in num_taps_to_check:
            pcts = [s[n_taps] for s in sparsity_stats if n_taps in s]
            if pcts:
                avg_percentages[n_taps] = np.mean(pcts)
        
        print(f"\nAnalyzed {len(sparsity_stats)} antenna pairs total")
        print("\nAverage energy concentration across antenna pairs:")
        for n_taps in sorted(avg_percentages.keys()):
            pct = avg_percentages[n_taps]
            print(f"  First {n_taps:3d} taps capture {pct:6.2f}% of energy on average")
        
        # Recommendation
        print("\nRecommendation for PDP truncation:")
        for n_taps in sorted(avg_percentages.keys()):
            if avg_percentages[n_taps] >= 90:
                print(f"  → Consider truncating to ~{n_taps} taps (captures ~90% of energy)")
                break
    
    print()


if __name__ == '__main__':
    main()
