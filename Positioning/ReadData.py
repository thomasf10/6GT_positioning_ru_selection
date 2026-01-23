from scipy.io import netcdf

import netCDF4


path_dataset = "E:\\6GT_positioning_ru_selection\\dataset\\office_space_inline\\"

path_sub10ghz_data = path_dataset+'sub_10ghz_channels\\'


ue_idx = 0

ue_data = netCDF4.Dataset(path_sub10ghz_data+'channels_sub10ghz_ue_'+str(ue_idx)+'.nc','r')
# temp = ue_data.variables[:]
print('test')
# print(temp)