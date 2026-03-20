from dataset.dataloaders import CsiDataset, CsiPositionDataset
import os
import torch

if __name__ == '__main__':

    basepath = r'../dataset/office_space_inline'
    subthz_path = os.path.join(basepath, 'sub_thz_channels')
    sub10_path = os.path.join(basepath, 'sub_10ghz_channels')
    labels_path = os.path.join(basepath, 'ru_selection_labels', 'results.csv')
    modes = ['sub10', 'subTHz', 'combined', 'pilot', 'sub10_pilot']  # todo pilot modes currently not implemented
    mode = modes[2]
    print(f'SELECTED MODE: {mode}')

    train_set = CsiDataset(subthz_path,
                           sub10_path,
                           labels_path,
                           mode
                           )

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    data, label, user_id = next(iter(train_dataloader))

    print(f'{user_id=}')
    print(f'datakeys: {data.keys()}')
    print(f'labels: {label}')
    print(f"stripe: {label['stripe_id']} \n"
          f"ru: {label['ru_id']} \n"
          f"ue beam id: {int(label['ue_beam_id'])} - ue beam angle:{ train_set.beam_id_to_angle(int(label['ue_beam_id']))}\n"
          f"ru beam id: {int(label['ru_beam_id'])} - ru beam angle:{ train_set.beam_id_to_angle(int(label['ru_beam_id']))}\n")


    print(f'--------------- positioning example -------------------------')
    train_set = CsiPositionDataset(subthz_path,
                           sub10_path,
                           labels_path,
                           mode,
                           on_grid_only=True
                           )

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    data, label, user_id = next(iter(train_dataloader))
    print(f'{user_id=}')
    print(f'datakeys: {data.keys()}')
    print(f'position labels: {label}')

