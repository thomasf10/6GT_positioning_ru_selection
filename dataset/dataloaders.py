import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re

invalid_ue_ideces = [761]

""" Dataset for Positioning """
# todo add option to make a dataset with only the points under the RUs as a training set
# use: on_grid_users = ds.where(ds['ue_on_stripe_grid'], drop=True) (see commented code below)
class CsiPositionDataset(Dataset):
    def __init__(self, subthz_path, sub10_path, labels_path, mode):
        self.subthz_path = subthz_path
        self.sub10_path = sub10_path
        self.labels_path = labels_path

        modes = ['sub10', 'subTHz', 'combined', 'pilot', 'sub10_pilot'] #todo pilot modes currently not implemented
        if mode not in modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {modes}"
            )
        self.mode = mode

        # --------------------------------------------------
        # Build list of VALID user_ids
        # --------------------------------------------------

        sub10_users = set()
        subthz_users = set()

        if os.path.exists(sub10_path):
            for f in os.listdir(sub10_path):
                match = re.search(r'ue_(\d+)', f)
                if match:
                    sub10_users.add(int(match.group(1)))

        if os.path.exists(subthz_path):
            for f in os.listdir(subthz_path):
                match = re.search(r'ue_(\d+)', f)
                if match:
                    subthz_users.add(int(match.group(1)))

        if mode == "sub10":
            self.valid_users = sorted(list(sub10_users))

        elif mode == "subTHz":
            self.valid_users = sorted(list(subthz_users))

        elif mode == "combined":
            # Only keep users that exist in BOTH folders
            self.valid_users = sorted(list(sub10_users & subthz_users))

        else:
            self.valid_users = sorted(list(sub10_users))

        print(f"Found {len(self.valid_users)} valid users.")

        print(f'-------------------------------------------------------------------- \nCsiDataset expected output shapes: \n'
              f'sub10 mode:  (batch_size, nr_APs, ue_ants, ru_ants, subcarriers) \n'
              f'subTHz mode: (batch_size, nr_RUs, ue_ants, ru_ants, subcarriers) \n'
              f'combined mode: [H_sub10, H_subTHz] \n'
              f'pilot mode: TODO not yet implemented \n'
              f'sub10_pilot mode: TODO not yet implemented \n'
              f'--------------------------------------------------------------------')

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        if self.mode == 'subTHz' or self.mode == 'combined':
            """ load sub THz CSI """
            subthz_dir = os.path.join(self.subthz_path, f"channels_thz_ue_{user_id}.nc")

            ds = xr.load_dataset(subthz_dir)

            # get subthz CSI of user
            sub_thz_channel = ds['channel'].values #shape: (bs, nr_RUs, ue_ant, ru_ant, subcarrier)

            # get position label
            x_label = ds.attrs['user_x']
            y_label = ds.attrs['user_y']
            z_label = ds.attrs['user_z'] # should be fixed
            position_label = np.array([x_label, y_label, z_label], dtype=np.float32) #shape: {bs, 3)

            if self.mode == 'subTHz':
                return sub_thz_channel, position_label, user_id

        if self.mode == 'sub10' or self.mode == 'combined':
            """ load sub 10 CSI """
            sub10_dir = os.path.join(self.sub10_path, f"channels_sub10ghz_ue_{user_id}.nc")

            ds_sub10 = xr.load_dataset(sub10_dir)
            #print(ds_sub10)

            sub_10_channel = ds_sub10["channel"].values #shape: (bs, nr_aps, ue_ant, ru_ant, subcarrier)

            # get position label
            x_label = ds.attrs['user_x']
            y_label = ds.attrs['user_y']
            z_label = ds.attrs['user_z'] # should be fixed
            position_label = np.array([x_label, y_label, z_label], dtype=np.float32) #shape: {bs, 3)

            if self.mode == 'sub10':
                return sub_10_channel, position_label, user_id

        if self.mode == 'combined':
            return sub_10_channel, sub_thz_channel, position_label, user_id

""" Dataset for RU selection """
class CsiDataset(Dataset):
    def __init__(self, subthz_path, sub10_path, labels_path, mode):
        self.subthz_path = subthz_path
        self.sub10_path = sub10_path
        self.labels_path = labels_path

        modes = ['sub10', 'subTHz', 'combined', 'pilot', 'sub10_pilot'] #todo pilot modes currently not implemented
        if mode not in modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {modes}"
            )
        self.mode = mode

        # todo get labels
        # todo load them from new dataframe that needs to be made

        # Build list of valid user_ids
        sub10_users = set()
        subthz_users = set()

        if os.path.exists(sub10_path):
            for f in os.listdir(sub10_path):
                match = re.search(r'ue_(\d+)', f)
                if match:
                    sub10_users.add(int(match.group(1)))

        if os.path.exists(subthz_path):
            for f in os.listdir(subthz_path):
                match = re.search(r'ue_(\d+)', f)
                if match:
                    subthz_users.add(int(match.group(1)))

        if mode == "sub10":
            self.valid_users = sorted(list(sub10_users))

        elif mode == "subTHz":
            self.valid_users = sorted(list(subthz_users))

        elif mode == "combined":
            # Only keep users that exist in BOTH folders
            self.valid_users = sorted(list(sub10_users & subthz_users))

        else:
            self.valid_users = sorted(list(sub10_users))

        print(f"Found {len(self.valid_users)} valid users.")

        # print shapes
        print(f'-------------------------------------------------------------------- \nCsiDataset expected output shapes: \n'
              f'sub10 mode:  (batch_size, nr_APs, ue_ants, ru_ants, subcarriers) \n'
              f'subTHz mode: (batch_size, nr_RUs, ue_ants, ru_ants, subcarriers) \n'
              f'combined mode: [H_sub10, H_subTHz] \n'
              f'pilot mode: TODO not yet implemented \n'
              f'sub10_pilot mode: TODO not yet implemented \n'
              f'--------------------------------------------------------------------')

    def __len__(self):
        return len(self.valid_users)

    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        if self.mode == 'subTHz' or self.mode == 'combined':
            """ load sub THz CSI """
            subthz_dir = os.path.join(self.subthz_path, f"channels_thz_ue_{user_id}.nc")

            ds = xr.load_dataset(subthz_dir)
            #print(ds)

            # get subthz CSI of user
            sub_thz_channel = ds['channel'].values #shape: (bs, nr_RUs, ue_ant, ru_ant, subcarrier)

            if self.mode == 'subTHz':
                return sub_thz_channel, user_id

        if self.mode == 'sub10' or self.mode == 'combined':
            """ load sub 10 CSI """
            sub10_dir = os.path.join(self.sub10_path, f"channels_sub10ghz_ue_{user_id}.nc")

            ds_sub10 = xr.load_dataset(sub10_dir)
            #print(ds_sub10)

            sub_10_channel = ds_sub10["channel"].values #shape: (bs, nr_aps, ue_ant, ru_ant, subcarrier)

            if self.mode == 'sub10':
                return sub_10_channel, user_id

        if self.mode == 'combined':
            return sub_10_channel, sub_thz_channel, user_id




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
#
# """ select desired UEs """
#
# # Load the dataset
# ds = xr.load_dataset(r"office_space_inline/ue_locations/ue_locations.nc")
#
# # Print dataset overview
# print(ds)
#
#
#
# # # Or convert to NumPy for further processing
# # positions_zone1 = grid_users[['x', 'y', 'z']].to_array().values.T  # shape: (num_users, 3)
#
# # # select users based if they are on the grid or not
# # on_grid_users = ds.where(ds['ue_on_stripe_grid'], drop=True)
# # off_grid_users = ds.where(~ds['ue_on_stripe_grid'], drop=True)
# #
# # print(f"Users on grid: {on_grid_users.dims['user']}")
# # print(f"Users off grid: {off_grid_users.dims['user']}")
#
# # get user under stripe idx and ru idx
# # todo loop over all grid users to create training dataset for positioning
# stripe_idx = 0
# ru_idx = 0
# matched_user = ds.where(
#     (ds['ue_on_stripe_grid']) &
#     (ds['ue_stripe_idx'] == stripe_idx) &
#     (ds['ue_ru_idx'] == ru_idx),
#     drop=True
# )
# matched_user_idx = matched_user['user_id'].values
#
# print(f"User at stripe: {stripe_idx} and RU: {ru_idx}: UE idx={matched_user_idx} x={matched_user['x'].values}, "
#         f"y={matched_user['y'].values}, "
#         f"z={matched_user['z'].values}")
#
# """ load sub THz channel of matched user """
# print(f'----------------------------- sub THz example -----------------------------')
#
# user_id = int(matched_user_idx[0])  # Example user index
# print(f'user_id: {user_id}')
# file_path = f"office_space_inline\sub_thz_channels\channels_thz_ue_{user_id}.nc"
#
# ds = xr.load_dataset(file_path)
# print(ds)
#
# # get subthz CSI of user
# sub_thz_channel = ds['channel'].values
# print(f'sub thz channel shape: {sub_thz_channel.shape}')
# print(f'shape subthz channel: {sub_thz_channel.shape} - the axis correspond to: {ds.sizes} - (tx pair = radio units)')
#
# # print some meta data
# print(f"User {ds.attrs['user_idx']} is in zone: {ds.attrs['zone']}")
# print(f"Location: x={ds.attrs['user_x']}, y={ds.attrs['user_y']}, z={ds.attrs['user_z']}")
#
#
#
# """ load sub 10 GHz channel of matched user """
#
# print(f'----------------------------- sub 10 GHz example -----------------------------')
# user_id = 0
# file_path_sub10 = f"office_space_inline/sub_10ghz_channels/channels_sub10ghz_ue_{user_id}.nc"
# ds_sub10 = xr.load_dataset(file_path_sub10)
# print(ds_sub10)
#
# # print all AP names
# ap_names = ds_sub10["ap"].values
# print("AP names:", ap_names)
# ap_idx = 0
# channel_sub10 = ds_sub10["channel"].sel(ap=ap_names[ap_idx]).values  # select 3 AP
# #print(f"Channel shape for AP {ap_names[ap_idx]}: {channel_sub10.shape}")  # (rx_ant, tx_ant, subcarrier)
# #print(f"Channel data for AP {ap_names[ap_idx]}:\n", channel_sub10)
#
# # all APs
# full_sub_10_channel = ds_sub10["channel"].values
# print(f'shape full sub10 channel: {full_sub_10_channel.shape} - the axis correspond to: {ds_sub10.sizes}')