import xarray as xr
import numpy as np

def get_channel_by_stripe_ru(ds, stripe_idx, ru_idx):
    match = ((ds["stripe_idx"] == stripe_idx) & (ds["RU_idx"] == ru_idx))
    if match.any():
        tx_index = match.argmax().item()  # first match
        channel = ds["channel"].isel(tx_pair=tx_index)
        return channel
    else:
        print("No matching stripe/RU combination found.")
        return None

def channel_to_numpycomplex64(arr_struct):
    """
    Convert xarray DataArray channel which is a structured dtype with separate real and imaginary parts$
    back to numpy complex64 array.
    """
    arr_complex = arr_struct['r'] + 1j * arr_struct['i']
    return arr_complex.astype(np.complex64)

""" select desired UEs """

# Load the dataset
ds = xr.load_dataset(r"office_space_inline/ue_locations/ue_locations.nc")

# Print dataset overview
print(ds)



# # Or convert to NumPy for further processing
# positions_zone1 = grid_users[['x', 'y', 'z']].to_array().values.T  # shape: (num_users, 3)

# # select users based if they are on the grid or not
# on_grid_users = ds.where(ds['ue_on_stripe_grid'], drop=True)
# off_grid_users = ds.where(~ds['ue_on_stripe_grid'], drop=True)
#
# print(f"Users on grid: {on_grid_users.dims['user']}")
# print(f"Users off grid: {off_grid_users.dims['user']}")

# get user under stripe idx and ru idx
# todo loop over all grid users to create training dataset for positioning
stripe_idx = 0
ru_idx = 0
matched_user = ds.where(
    (ds['ue_on_stripe_grid']) &
    (ds['ue_stripe_idx'] == stripe_idx) &
    (ds['ue_ru_idx'] == ru_idx),
    drop=True
)
matched_user_idx = matched_user['user_id'].values

print(f"User at stripe: {stripe_idx} and RU: {ru_idx}: UE idx={matched_user_idx} x={matched_user['x'].values}, "
        f"y={matched_user['y'].values}, "
        f"z={matched_user['z'].values}")

""" load sub THz channel of matched user """
print(f'----------------------------- sub THz example -----------------------------')

user_id = int(matched_user_idx[0])  # Example user index
print(f'user_id: {user_id}')
file_path = f"office_space_inline\sub_thz_channels\channels_thz_ue_{user_id}.nc"

ds = xr.load_dataset(file_path)
print(ds)

# get subthz CSI of user
sub_thz_channel = ds['channel'].values
print(f'sub thz channel shape: {sub_thz_channel.shape}')
print(f'shape subthz channel: {sub_thz_channel.shape} - the axis correspond to: {ds.sizes} - (tx pair = radio units)')

# print some meta data
print(f"User {ds.attrs['user_idx']} is in zone: {ds.attrs['zone']}")
print(f"Location: x={ds.attrs['user_x']}, y={ds.attrs['user_y']}, z={ds.attrs['user_z']}")



""" load sub 10 GHz channel of matched user """

print(f'----------------------------- sub 10 GHz example -----------------------------')
file_path_sub10 = f"office_space_inline/sub_10ghz_channels/channels_sub10ghz_ue_{user_id}.nc"
ds_sub10 = xr.load_dataset(file_path_sub10)
print(ds_sub10)

# print all AP names
ap_names = ds_sub10["ap"].values
print("AP names:", ap_names)
ap_idx = 0
channel_sub10 = ds_sub10["channel"].sel(ap=ap_names[ap_idx]).values  # select 3 AP
#print(f"Channel shape for AP {ap_names[ap_idx]}: {channel_sub10.shape}")  # (rx_ant, tx_ant, subcarrier)
#print(f"Channel data for AP {ap_names[ap_idx]}:\n", channel_sub10)

# all APs
full_sub_10_channel = ds_sub10["channel"].values
print(f'shape full sub10 channel: {full_sub_10_channel.shape} - the axis correspond to: {ds_sub10.sizes}')