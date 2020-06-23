
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Add Libraries
import numpy as np
import pandas as pd
import others.utilities as my_util

# Dataset path
# dataset 1
dataset_path_1 = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA(partial)_1'])
dataset_path_1 = dataset_path_1 + 'CelebAretinaface1_1000.txt'
my_data_1 = pd.read_csv(dataset_path_1, sep=" ", header=0)
# dataset 2
dataset_path_2 = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA(partial)_2'])
dataset_path_2 = dataset_path_2 + 'CelebAretinaface1001_2000.txt'
my_data_2 = pd.read_csv(dataset_path_2, sep=" ", header=0)
# dataset 3
dataset_path_3 = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA(partial)_3'])
dataset_path_3 = dataset_path_3 + 'CelebAretinaface2001_3000.txt'
my_data_3 = pd.read_csv(dataset_path_3, sep=" ", header=0)

# Concat files
my_data = pd.concat([my_data_1, my_data_2, my_data_3], ignore_index=True)
my_data = my_data.sort_values(by=['id', 'image_id'])
my_data = my_data.reset_index(drop=True)

del my_data_1, my_data_2, my_data_3

# Write to file
my_data.to_csv('CelebA_Retinaface_1_3000.txt', index=None, sep=' ')

# Test read wrote file
# dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA_partial'])
# dataset_path = dataset_path + 'CelebA_Retinaface_1_3000.txt'
# my_data = pd.read_csv(dataset_path, sep=" ", header=0)

print()