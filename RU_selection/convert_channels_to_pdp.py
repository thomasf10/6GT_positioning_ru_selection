import os
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def discover_user_ids(sub10_path):
    """Discover user IDs exactly like in dataloaders.py."""
    user_ids = []
    if os.path.exists(sub10_path):
        for file_name in os.listdir(sub10_path):
            match = re.search(r'ue_(\d+)', file_name)
            if match:
                user_ids.append(int(match.group(1)))
    return sorted(set(user_ids))


def choose_write_engine():
    """Pick an installed xarray backend engine for writing netCDF."""
    try:
        engines = set(xr.backends.list_engines().keys())
    except Exception:
        engines = set()

    for engine in ('netcdf4', 'h5netcdf', 'scipy'):
        if engine in engines:
            return engine
    return None


def convert_channels_to_pdp(input_dir, output_dir, num_taps=32):
    """
    Convert channel files (complex CSI) to PDP (dB) with limited taps.
    
    Args:
        input_dir: Directory containing channel .nc files
        output_dir: Directory to save converted .nc files
        num_taps: Number of delay taps to keep
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    user_ids = discover_user_ids(input_dir)
    print(f"Found {len(user_ids)} channel files in {input_dir}")

    if len(user_ids) == 0:
        print(f"Error: No channel files found in {input_dir}")
        return

    write_engine = choose_write_engine()
    if write_engine is None:
        print(
            "Warning: No explicit netCDF backend detected from xarray list_engines(). "
            "Will try default writer."
        )
    else:
        print(f"Using netCDF writer engine: {write_engine}")
    
    # Process each file
    for file_idx, user_id in enumerate(user_ids):
        channel_file = os.path.join(input_dir, f"channels_sub10ghz_ue_{user_id}.nc")
        channel_name = os.path.basename(channel_file)
        
        if (file_idx + 1) % 100 == 0 or file_idx == 0:
            print(f"Processing {file_idx + 1} / {len(user_ids)}: {channel_name}")
        
        try:
            # Load CSI data
            ds = xr.load_dataset(channel_file)
            
            # Extract channel tensor
            # Shape can be: (n_aps, ue_ants, ru_ants, carriers, 2) or (n_aps, ue_ants, ru_ants, carriers)
            channel_data = ds['channel'].values
            
            # Extract metadata
            attrs = dict(ds.attrs)
            
            # Handle both 5D (real/imag separate) and 4D (complex) formats
            if channel_data.ndim == 5:
                n_aps, n_ue_ants, n_ru_ants, n_carriers, two_dim = channel_data.shape
                is_complex_format = False
            elif channel_data.ndim == 4:
                n_aps, n_ue_ants, n_ru_ants, n_carriers = channel_data.shape
                is_complex_format = True
            else:
                raise ValueError(f"Unexpected channel shape: {channel_data.shape}. Expected 4D or 5D.")
            
            # Convert to PDP for each antenna path independently
            # Initialize PDP array: (n_aps, ue_ants, ru_ants, num_taps)
            pdp_db = np.zeros((n_aps, n_ue_ants, n_ru_ants, num_taps), dtype=np.float32)
            
            # Process each antenna combination
            for ap_idx in range(n_aps):
                for ue_idx in range(n_ue_ants):
                    for ru_idx in range(n_ru_ants):
                        # Extract complex CSI for this antenna path
                        if is_complex_format:
                            # Already complex
                            csi_complex = channel_data[ap_idx, ue_idx, ru_idx, :].astype(np.complex128)
                        else:
                            # Real and imaginary parts are separate
                            csi_complex = channel_data[ap_idx, ue_idx, ru_idx, :, 0] + \
                                         1j * channel_data[ap_idx, ue_idx, ru_idx, :, 1]
                        
                        # Compute PDP via IFFT
                        pdp_complex = np.fft.ifft(csi_complex)
                        pdp_power = np.abs(pdp_complex) ** 2
                        
                        # Convert to dB
                        pdp_db_full = 10 * np.log10(pdp_power + 1e-12)
                        
                        # Keep only the first num_taps taps
                        pdp_db[ap_idx, ue_idx, ru_idx, :] = pdp_db_full[:num_taps]
            
            # Create new xarray dataset with PDP data
            pdp_ds = xr.Dataset(
                {
                    'pdp': (['ap', 'ue_ant', 'ru_ant', 'tap'], pdp_db)
                },
                coords={
                    'ap': np.arange(n_aps),
                    'ue_ant': np.arange(n_ue_ants),
                    'ru_ant': np.arange(n_ru_ants),
                    'tap': np.arange(num_taps),
                },
                attrs=attrs
            )
            
            # Save to output file
            output_file = output_path / channel_name
            if write_engine is None:
                pdp_ds.to_netcdf(output_file)
            else:
                pdp_ds.to_netcdf(output_file, engine=write_engine)
            
        except Exception as e:
            print(f"Error processing {channel_name}: {e}")
            continue
    
    print(f"\nConversion complete! Output saved to {output_dir}")
    print(f"PDP shape: (n_aps, ue_ants, ru_ants, {num_taps})")


def main():
    # Define paths (relative to this script)
    input_dir = '../dataset/office_space_inline/sub_10ghz_channels'
    output_dir = '../dataset/office_space_inline/sub_10ghz_pdp_32_taps'
    num_taps = 32
    
    print("="*80)
    print(f"Converting CSI channels to PDP ({num_taps} taps, dB scale)")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print()
    
    convert_channels_to_pdp(input_dir, output_dir, num_taps)


if __name__ == '__main__':
    main()
