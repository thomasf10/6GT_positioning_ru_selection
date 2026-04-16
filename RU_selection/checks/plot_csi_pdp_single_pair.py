import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# Add workspace root to path to allow imports from dataset module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.dataloaders import CsiDataset


# ==============================================================================
# User-selectable settings
# ==============================================================================
subthz_path = '../dataset/office_space_inline/sub_thz_channels'
#sub10_path = '../dataset/office_space_inline/sub_10ghz_channels_normalized_delay'
sub10_path = '../dataset/office_space_inline/sub_10ghz_channels'
labels_path = '../dataset/office_space_inline/ru_selection_labels/results.csv'

mode = 'sub10'  # Choose: 'sub10', 'subTHz', or 'combined'

# Select a sample either by dataset index or by explicit user_id.
# If selected_user_id is not None, it takes priority over dataset_index.
dataset_index = 0
selected_user_id = 235

# In combined mode, choose which modality to inspect.
channel_key = 'sub10_channel'  # 'sub10_channel' or 'subthz_channel'

# Antenna-pair selectors
ap_index = 0
tx_ant_index = 0   # Transmit / RU antenna index
ue_ant_index = 0   # User-equipment antenna index


def resolve_sample_index(dataset, dataset_index_local, selected_user_id_local):
    if selected_user_id_local is None:
        return dataset_index_local

    try:
        return dataset.valid_users.index(selected_user_id_local)
    except ValueError as exc:
        raise ValueError(
            f'user_id {selected_user_id_local} is not present in this dataset/mode.'
        ) from exc


def tensor_to_complex_1d(channel_tensor, ap_idx, ue_idx, tx_idx):
    if isinstance(channel_tensor, torch.Tensor):
        channel_np = channel_tensor.cpu().numpy()
    else:
        channel_np = np.asarray(channel_tensor)

    if channel_np.ndim != 5:
        raise ValueError(
            f'Expected channel tensor shape (n_aps, ue_ants, ru_ants, carriers, 2), got {channel_np.shape}'
        )

    n_aps, n_ue_ants, n_tx_ants, n_carriers, two_dim = channel_np.shape
    if two_dim != 2:
        raise ValueError(f'Last channel dimension must be 2 (real/imag), got {two_dim}')

    if not (0 <= ap_idx < n_aps):
        raise IndexError(f'ap_index {ap_idx} out of range [0, {n_aps - 1}]')
    if not (0 <= ue_idx < n_ue_ants):
        raise IndexError(f'ue_ant_index {ue_idx} out of range [0, {n_ue_ants - 1}]')
    if not (0 <= tx_idx < n_tx_ants):
        raise IndexError(f'tx_ant_index {tx_idx} out of range [0, {n_tx_ants - 1}]')

    pair = channel_np[ap_idx, ue_idx, tx_idx, :, :]
    return pair[:, 0] + 1j * pair[:, 1]


def compute_pdp(csi_complex):
    pdp_time = np.fft.ifft(csi_complex)
    pdp_power = np.abs(pdp_time) ** 2
    pdp_db = 10.0 * np.log10(pdp_power + 1e-12)
    return pdp_time, pdp_power, pdp_db


def main():
    dataset = CsiDataset(
        subthz_path=subthz_path,
        sub10_path=sub10_path,
        labels_path=labels_path,
        mode=mode,
    )

    sample_index = resolve_sample_index(dataset, dataset_index, selected_user_id)
    data, label, user_id = dataset[sample_index]

    selected_channel = data[channel_key]
    if selected_channel is None or selected_channel.numel() == 0:
        raise ValueError(
            f'Channel {channel_key} is empty for mode={mode}. Choose a valid mode/channel combination.'
        )

    csi_complex = tensor_to_complex_1d(selected_channel, ap_index, ue_ant_index, tx_ant_index)
    carriers = np.arange(csi_complex.shape[0])
    pdp_time, pdp_power, pdp_db = compute_pdp(csi_complex)
    taps = np.arange(pdp_power.shape[0])

    print('Selected sample information:')
    print(f'  dataset_index = {sample_index}')
    print(f'  user_id = {user_id}')
    print(f'  mode = {mode}')
    print(f'  channel_key = {channel_key}')
    print(f'  selected channel tensor shape = {tuple(selected_channel.shape)}')
    print(f'  ap_index = {ap_index}')
    print(f'  ue_ant_index = {ue_ant_index}')
    print(f'  tx_ant_index = {tx_ant_index}')
    if label is not None:
        print(f'  label = {label}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Load coordinates from the channel netCDF file attributes
    coords_str = ''
    try:
        if mode == 'sub10':
            channel_file = Path(sub10_path) / f'channels_sub10ghz_ue_{user_id}.nc'
        elif mode == 'subTHz':
            channel_file = Path(subthz_path) / f'channels_thz_ue_{user_id}.nc'
        else:
            channel_file = Path(sub10_path) / f'channels_sub10ghz_ue_{user_id}.nc'
        
        if channel_file.exists():
            ds = xr.open_dataset(channel_file)
            x = ds.attrs.get('user_x')
            y = ds.attrs.get('user_y')
            z = ds.attrs.get('user_z')
            if x is not None and y is not None and z is not None:
                coords_str = f', coords=({x:.2f}, {y:.2f}, {z:.2f})'
                print(f'Loaded coordinates: {coords_str}')
    except Exception as e:
        print(f'Warning: Could not load user coordinates: {e}')
    
    fig.suptitle(
        f'CSI / PDP for user_id={user_id}{coords_str}, {channel_key}, AP={ap_index}, UE ant={ue_ant_index}, TX ant={tx_ant_index}',
        fontsize=12,
    )

    axes[0, 0].plot(carriers, csi_complex.real, label='Real')
    axes[0, 0].plot(carriers, csi_complex.imag, label='Imag')
    axes[0, 0].set_title('CSI Components vs Carrier')
    axes[0, 0].set_xlabel('Carrier index')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True)
    axes[0, 0].legend()

    axes[0, 1].plot(carriers, np.abs(csi_complex), label='Magnitude')
    #axes[0, 1].plot(carriers, np.angle(csi_complex), label='Phase')
    axes[0, 1].set_title('CSI Magnitude / Phase')
    axes[0, 1].set_xlabel('Carrier index')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    axes[1, 0].stem(taps, pdp_power)
    axes[1, 0].set_title('PDP Power')
    axes[1, 0].set_xlabel('Delay tap index')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].grid(True)
    axes[1, 0].set_xlim([0, 64])

    axes[1, 1].plot(taps, pdp_db)
    axes[1, 1].set_title('PDP in dB')
    axes[1, 1].set_xlabel('Delay tap index')
    axes[1, 1].set_ylabel('Power (dB)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()