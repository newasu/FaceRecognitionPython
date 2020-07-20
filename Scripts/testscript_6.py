
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

#############################################################################################

# Path
# Dataset path
dataset_name = 'Diveface'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
dataset_path = dataset_path + 'Diveface_retinaface.txt'

#############################################################################################

# use_data_bet = [0, 100000]

# Read data
yy = pd.read_csv(dataset_path, sep=" ", header=0).id.values

# # Select only some classes
# yy = yy[np.where(np.logical_and(yy>=use_data_bet[0], yy<=use_data_bet[1]))]

# Split training and test set
# [exp_training_sep_idx, exp_test_sep_idx] = my_util.split_kfold_by_classes(yy, n_splits=numb_exp_kfold, random_state=0)
# del yy


