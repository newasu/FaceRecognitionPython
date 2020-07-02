
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import lib
import pandas as pd
import numpy as np

# Import my own lib
import others.utilities as my_util

# Dataset path
dataset_name = 'CelebA'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA_features'])
filename_1 = 'CelebA_retinaface_1.txt'
filename_2 = 'CelebA_retinaface_2.txt'

# Read data
my_data_1 = pd.read_csv((dataset_path+filename_1), sep=" ", header=0)
my_data_2 = pd.read_csv((dataset_path+filename_2), sep=" ", header=0)
my_data_all = my_data_1.append(my_data_2)
del my_data_1, my_data_2

# my_data_all = pd.read_csv((dataset_path+'CelebA_retinaface.txt'), sep=" ", header=0)

# Sort
my_data_all = my_data_all.sort_values(by=['id', 'image_id'])

# Write
my_data_all.to_csv('CelebA_retinaface.txt', header=True, index=False, sep=' ', mode='a')

print('Finished')