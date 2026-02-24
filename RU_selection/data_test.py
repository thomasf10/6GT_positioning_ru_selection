from dataset.dataloaders import CsiDataset, CsiPositionDataset
import os
import torch

if __name__ == '__main__':

    basepath = r'../dataset/office_space_inline'
    subthz_path = os.path.join(basepath, 'sub_thz_channels')
    sub10_path = os.path.join(basepath, 'sub_10ghz_channels')
    labels_path = 'todo' #todo
    modes = ['sub10', 'subTHz', 'combined', 'pilot', 'sub10_pilot']  # todo pilot modes currently not implemented
    mode = modes[2]
    print(f'SELECTED MODE: {mode}')
    #
    # train_set = CsiDataset(subthz_path,
    #                        sub10_path,
    #                        labels_path,
    #                        mode
    #                        )
    #
    # train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    # sub10_channel, subthz_channel, user_ids = next(iter(train_dataloader))
    # print(f'{sub10_channel.shape}')
    # print(f'{subthz_channel.shape}')
    # print(f'{user_ids=}')


    train_set = CsiPositionDataset(subthz_path,
                           sub10_path,
                           labels_path,
                           mode
                           )

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
    sub10_channel, subthz_channel, position, user_ids = next(iter(train_dataloader))
    print(f'{sub10_channel.shape}')
    print(f'{subthz_channel.shape}')
    print(f'postions shape: {position.shape}')
    print(f'{position=}')
    print(f'position of first ue: {position[0]}')
    print(f'{user_ids=}')