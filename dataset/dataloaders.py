import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import re
import pandas as pd

invalid_ue_ideces = [761]

""" Dataset for Positioning """
class CsiPositionDataset(Dataset):
    """Positioning dataset with on/off-grid splitting.

    Args:
        subset: 'on_grid' (UEs sitting under an RU — typical training set),
                'off_grid' (UEs not on the stripe grid),
                or 'all' (no filtering). Ignored if `user_ids` is provided.
        max_users: optional cap on the number of users kept after subset
                filtering. Selection is reproducible via `seed`.
        seed: RNG seed used when truncating to `max_users`.
        user_ids: optional explicit list of user_ids. When provided, the
                dataset uses exactly these users (intersected with what is
                available on disk for `mode`); `subset` and `max_users` are
                ignored. Use this to reuse a split saved by another script.
        exclude_user_ids: optional iterable of user_ids to drop after all
                other filtering — e.g. to keep train disjoint from val/test.
    """

    VALID_SUBSETS = ('on_grid', 'off_grid', 'all')

    def __init__(
        self,
        subthz_path,
        sub10_path,
        labels_path,
        mode,
        subset='all',
        max_users=None,
        seed=42,
        user_ids=None,
        exclude_user_ids=None,
    ):
        self.subthz_path = subthz_path
        self.sub10_path = sub10_path
        self.labels_path = labels_path

        if subset not in self.VALID_SUBSETS:
            raise ValueError(
                f"Invalid subset '{subset}'. Must be one of: {self.VALID_SUBSETS}"
            )
        self.subset = subset
        self.max_users = max_users
        self.seed = seed

        modes = ['sub10', 'subTHz', 'combined', 'pilot', 'sub10_pilot'] #todo pilot modes currently not implemented
        if mode not in modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {modes}"
            )
        self.mode = mode

        # --------------------------------------------------
        # Available user_ids on disk for this mode
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
            available = sub10_users
        elif mode == "subTHz":
            available = subthz_users
        elif mode == "combined":
            available = sub10_users & subthz_users
        else:
            available = sub10_users

        # --------------------------------------------------
        # Either use explicit user_ids, or apply subset / max_users filters
        # --------------------------------------------------
        if user_ids is not None:
            requested = [int(uid) for uid in user_ids]
            self.valid_users = sorted(uid for uid in requested if uid in available)
            missing = [uid for uid in requested if uid not in available]
            if missing:
                print(
                    f"Warning: {len(missing)} of {len(requested)} requested user_ids not "
                    f"available on disk for mode='{mode}'; they were dropped."
                )
        else:
            self.valid_users = sorted(list(available))

            if subset != 'all':
                ue_locations_path = os.path.join(
                    os.path.dirname(sub10_path), "ue_locations", "ue_locations.nc"
                )
                if not os.path.exists(ue_locations_path):
                    raise FileNotFoundError(
                        f"UE locations file required for subset='{subset}' but not found at "
                        f"{ue_locations_path}."
                    )
                ue_ds = xr.load_dataset(ue_locations_path)
                on_grid_user_ids = set(
                    ue_ds.where(ue_ds['ue_on_stripe_grid'], drop=True)['user_id'].values
                )
                if subset == 'on_grid':
                    self.valid_users = [uid for uid in self.valid_users if uid in on_grid_user_ids]
                else:  # 'off_grid'
                    self.valid_users = [uid for uid in self.valid_users if uid not in on_grid_user_ids]
                print(f"Filtered to {len(self.valid_users)} {subset} users.")

            if max_users is not None and len(self.valid_users) > max_users:
                rng = np.random.default_rng(seed)
                picked = rng.choice(len(self.valid_users), size=max_users, replace=False)
                self.valid_users = sorted(self.valid_users[i] for i in picked)
                print(f"Capped to {len(self.valid_users)} users (max_users={max_users}, seed={seed}).")

        # --------------------------------------------------
        # Optional explicit exclusion (e.g. keep train disjoint from val/test)
        # --------------------------------------------------
        if exclude_user_ids:
            exclude_set = {int(uid) for uid in exclude_user_ids}
            before = len(self.valid_users)
            self.valid_users = [uid for uid in self.valid_users if uid not in exclude_set]
            removed = before - len(self.valid_users)
            if removed:
                print(f"Excluded {removed} user(s) via exclude_user_ids.")

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

        # Initialize data containers
        sub_10_channel = None
        sub_thz_channel = None
        position_label = None

        if self.mode == 'subTHz' or self.mode == 'combined':
            """ load sub THz CSI """
            subthz_dir = os.path.join(self.subthz_path, f"channels_thz_ue_{user_id}.nc")
            ds = xr.load_dataset(subthz_dir)

            # get subthz CSI of user
            sub_thz_channel = ds['channel'].values

            # 2D position label (x, y only — z is constant for this dataset)
            x_label = ds.attrs['user_x']
            y_label = ds.attrs['user_y']
            position_label = np.array([x_label, y_label], dtype=np.float32)

        if self.mode == 'sub10' or self.mode == 'combined':
            """ load sub 10 CSI """
            sub10_dir = os.path.join(self.sub10_path, f"channels_sub10ghz_ue_{user_id}.nc")
            ds_sub10 = xr.load_dataset(sub10_dir)

            sub_10_channel = ds_sub10["channel"].values

            # get position label (use sub10 dataset if not already set)
            if position_label is None:
                x_label = ds_sub10.attrs['user_x']
                y_label = ds_sub10.attrs['user_y']
                position_label = np.array([x_label, y_label], dtype=np.float32)

        # Convert complex CSI (real+imag) to float tensors with shape (..., 2),
        # matching CsiDataset so the same Conv3D backbone can be used.
        def channel_to_float_array(channel):
            if channel is None:
                return np.zeros((0,), dtype=np.float32)
            if channel.dtype.names is not None and 'r' in channel.dtype.names and 'i' in channel.dtype.names:
                channel = np.stack([channel['r'], channel['i']], axis=-1)
            elif np.iscomplexobj(channel):
                channel = np.stack([channel.real, channel.imag], axis=-1)
            return channel.astype(np.float32)

        sub_10_channel = channel_to_float_array(sub_10_channel)
        sub_thz_channel = channel_to_float_array(sub_thz_channel)

        data = {
            'sub10_channel': torch.from_numpy(sub_10_channel).float(),
            'subthz_channel': torch.from_numpy(sub_thz_channel).float(),
        }

        return data, torch.from_numpy(position_label), user_id

""" Dataset for RU selection """
class CsiDataset(Dataset):
    # Define beam angles and create angle-to-id mapping
    BEAM_ANGLES = [-30, -20, -10, 0, 10, 20, 30]
    ANGLE_TO_ID = {angle: idx for idx, angle in enumerate(BEAM_ANGLES)}
    ID_TO_ANGLE = {idx: angle for angle, idx in ANGLE_TO_ID.items()}
    
    def __init__(self, subthz_path, sub10_path, labels_path, mode, num_stripes=8, num_rus_per_stripe=20):
        self.subthz_path = subthz_path
        self.sub10_path = sub10_path
        self.labels_path = labels_path

        modes = ['sub10', 'subTHz', 'combined', 'pilot', 'sub10_pilot'] #todo pilot modes currently not implemented
        if mode not in modes:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {modes}"
            )
        self.mode = mode

        # Load labels from CSV
        self.labels_df = pd.read_csv(labels_path)

        # Use configurable grid for global RU IDs
        self.num_stripes = num_stripes
        self.num_rus_per_stripe = num_rus_per_stripe
        self.num_ru_ids = self.num_stripes * self.num_rus_per_stripe
        print(f"Using global RU ID space: {self.num_ru_ids} ({num_stripes} stripes x {num_rus_per_stripe} RUs)")

        # Create a mapping from ue_id to labels
        self.labels_dict = {}
        for _, row in self.labels_df.iterrows():
            ue_id = int(row['ue_id'])
            # Convert beam angles to beam IDs
            ue_beam_angle = int(row['ue_beam_id'])
            ru_beam_angle = int(row['ru_beam_id'])
            ue_beam_id = self.angle_to_beam_id(ue_beam_angle)
            ru_beam_id = self.angle_to_beam_id(ru_beam_angle)
            
            stripe_id = int(row['stripe_id'])
            ru_id = int(row['ru_id'])
            if not (0 <= stripe_id < self.num_stripes and 0 <= ru_id < self.num_rus_per_stripe):
                raise ValueError(f"Invalid stripe_id/ru_id pair: ({stripe_id}, {ru_id})")
            global_ru_id = stripe_id * self.num_rus_per_stripe + ru_id
            
            self.labels_dict[ue_id] = {
                'stripe_id': stripe_id,
                'ru_id': ru_id,
                'global_ru_id': global_ru_id,
                'ue_beam_id': ue_beam_id,
                'ru_beam_id': ru_beam_id
            }

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

    @staticmethod
    def angle_to_beam_id(angle):
        """Convert beam angle to beam ID.
        
        Args:
            angle: Beam angle in degrees [-30, -10, 0, 10, 30]
            
        Returns:
            Beam ID [0, 1, 2, 3, 4]
        """
        return CsiDataset.ANGLE_TO_ID[angle]
    
    @staticmethod
    def beam_id_to_angle(beam_id):
        """Convert beam ID to beam angle.
        
        Args:
            beam_id: Beam ID [0, 1, 2, 3, 4]
            
        Returns:
            Beam angle in degrees [-30, -10, 0, 10, 30]
        """
        return CsiDataset.ID_TO_ANGLE[beam_id]

    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        # Get label for this user
        label = self.labels_dict.get(user_id, None)
        
        # Initialize data containers
        sub_10_channel = None
        sub_thz_channel = None
        
        if self.mode == 'subTHz' or self.mode == 'combined':
            """ load sub THz CSI """
            subthz_dir = os.path.join(self.subthz_path, f"channels_thz_ue_{user_id}.nc")
            ds = xr.load_dataset(subthz_dir)
            sub_thz_channel = ds['channel'].values

        if self.mode == 'sub10' or self.mode == 'combined':
            """ load sub 10 CSI """
            sub10_dir = os.path.join(self.sub10_path, f"channels_sub10ghz_ue_{user_id}.nc")
            ds_sub10 = xr.load_dataset(sub10_dir)
            sub_10_channel = ds_sub10["channel"].values

        # Convert complex data (real+imag) to float tensors
        def channel_to_float_array(channel):
            if channel is None:
                return np.zeros((0,), dtype=np.float32)
            if np.iscomplexobj(channel):
                # Convert complex to two channels (real, imag)
                channel = np.stack([channel.real, channel.imag], axis=-1)
            return channel.astype(np.float32)

        sub_10_channel = channel_to_float_array(sub_10_channel)
        sub_thz_channel = channel_to_float_array(sub_thz_channel)

        data = {
            'sub10_channel': torch.from_numpy(sub_10_channel).float(),
            'subthz_channel': torch.from_numpy(sub_thz_channel).float()
        }

        return data, label, user_id


class PDP32Dataset(Dataset):
    # Keep the same RU/beam conventions as CsiDataset for compatibility.
    BEAM_ANGLES = [-30, -20, -10, 0, 10, 20, 30]
    ANGLE_TO_ID = {angle: idx for idx, angle in enumerate(BEAM_ANGLES)}
    ID_TO_ANGLE = {idx: angle for angle, idx in ANGLE_TO_ID.items()}

    def __init__(self, sub10_pdp_path, labels_path, mode='sub10'):
        self.sub10_pdp_path = sub10_pdp_path
        self.labels_path = labels_path

        if mode != 'sub10':
            raise ValueError("PDP32Dataset currently only supports mode='sub10'.")
        self.mode = mode

        # Load labels from CSV (same mapping strategy as CsiDataset)
        self.labels_df = pd.read_csv(labels_path)

        # Use fixed grid for global RU IDs: 8 stripes x 20 RUs per stripe => 160 values
        self.num_stripes = 8
        self.num_rus_per_stripe = 20
        self.num_ru_ids = self.num_stripes * self.num_rus_per_stripe
        print(f"Using fixed global RU ID space: {self.num_ru_ids} (8 stripes x 20 RUs)")

        self.labels_dict = {}
        for _, row in self.labels_df.iterrows():
            ue_id = int(row['ue_id'])
            ue_beam_angle = int(row['ue_beam_id'])
            ru_beam_angle = int(row['ru_beam_id'])
            ue_beam_id = self.angle_to_beam_id(ue_beam_angle)
            ru_beam_id = self.angle_to_beam_id(ru_beam_angle)

            stripe_id = int(row['stripe_id'])
            ru_id = int(row['ru_id'])
            if not (0 <= stripe_id < self.num_stripes and 0 <= ru_id < self.num_rus_per_stripe):
                raise ValueError(f"Invalid stripe_id/ru_id pair: ({stripe_id}, {ru_id})")
            global_ru_id = stripe_id * self.num_rus_per_stripe + ru_id

            self.labels_dict[ue_id] = {
                'stripe_id': stripe_id,
                'ru_id': ru_id,
                'global_ru_id': global_ru_id,
                'ue_beam_id': ue_beam_id,
                'ru_beam_id': ru_beam_id,
            }

        # Build list of valid user_ids from PDP folder
        sub10_users = set()
        if os.path.exists(sub10_pdp_path):
            for f in os.listdir(sub10_pdp_path):
                match = re.search(r'ue_(\d+)', f)
                if match:
                    sub10_users.add(int(match.group(1)))

        self.valid_users = sorted(list(sub10_users))
        print(f"Found {len(self.valid_users)} valid users.")

        print(
            '-------------------------------------------------------------------- \n'
            'PDP32Dataset expected output shapes: \n'
            'sub10 mode: (batch_size, nr_APs, ue_ants, ru_ants, 32_taps) \n'
            '--------------------------------------------------------------------'
        )

    def __len__(self):
        return len(self.valid_users)

    @staticmethod
    def angle_to_beam_id(angle):
        return PDP32Dataset.ANGLE_TO_ID[angle]

    @staticmethod
    def beam_id_to_angle(beam_id):
        return PDP32Dataset.ID_TO_ANGLE[beam_id]

    def __getitem__(self, idx):
        user_id = self.valid_users[idx]
        label = self.labels_dict.get(user_id, None)

        pdp_file = os.path.join(self.sub10_pdp_path, f"channels_sub10ghz_ue_{user_id}.nc")
        ds_pdp = xr.load_dataset(pdp_file)

        # Converter stores variable name 'pdp'. Keep fallback for compatibility.
        if 'pdp' in ds_pdp.data_vars:
            pdp_sub10 = ds_pdp['pdp'].values
        elif 'channel' in ds_pdp.data_vars:
            pdp_sub10 = ds_pdp['channel'].values
        else:
            raise KeyError(
                f"Expected variable 'pdp' (or fallback 'channel') in {pdp_file}, "
                f"found {list(ds_pdp.data_vars.keys())}"
            )

        if pdp_sub10.shape[-1] != 32:
            raise ValueError(f"Expected last dimension to be 32 taps, got shape {pdp_sub10.shape}")

        pdp_sub10 = pdp_sub10.astype(np.float32)
        data = {
            'sub10_pdp': torch.from_numpy(pdp_sub10).float(),
            'subthz_channel': torch.from_numpy(np.zeros((0,), dtype=np.float32)).float(),
        }

        return data, label, user_id




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