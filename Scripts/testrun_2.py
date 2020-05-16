
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import my lib
# from others.utilities import checkClassProportions
import others.utilities as my_util

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

# Read data
# Change data to cleaned dataset #################################################
dataset_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu\'s Mac/PhD\'s Degree/New/SourceCode/FaceRecognitionPython_data_store/Dataset/CelebA(partial)_1/CelebA_retinaface_1_1000.csv'
my_data = pd.read_csv(dataset_path, sep=",", header=0)

my_data_sep = my_data[my_data['id'].between(1, 5)]
# my_data_sep = my_data[my_data['id'].between(1, 10)].copy()
# my_data_sep = my_data_sep.reset_index()

xx = my_data_sep.feature.values
yy = my_data_sep.id.values

del dataset_path, my_data, my_data_sep

# count_sample_occurrence = my_util.checkClassProportions(yy)
# print(my_util.checkClassProportions(yy).round({'freq_perc': 2}))

# Get index for partition training/test set
tmp_round = 0
data_spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=tmp_round)
data_spliter.get_n_splits(xx, yy)
train_data_spliter = list()
test_data_spliter = list()
for train_index, test_index in data_spliter.split(xx, yy):
    train_data_spliter.append(train_index)
    test_data_spliter.append(test_index)
    # print(my_util.checkClassProportions(yy[train_index]).round({'freq_perc': 2}))
    # print(my_util.checkClassProportions(yy[test_index]).round({'freq_perc': 2}))

    kfold_data_spliter = StratifiedKFold(n_splits=5, shuffle=False, random_state=tmp_round)
    kfold_data_spliter.get_n_splits(xx[train_index], yy[train_index])
    kfold_train_data_spliter = list()
    kfold_test_data_spliter = list()
    for kfold_train_index, kfold_test_index in kfold_data_spliter.split(xx[train_index], yy[train_index]):
        kfold_train_data_spliter.append(kfold_train_index)
        kfold_test_data_spliter.append(kfold_test_index)

        xx_kfold = xx[train_data_spliter[tmp_round][kfold_train_index]]
        yy_kfold = yy[train_data_spliter[tmp_round][kfold_train_index]]
        # print(my_util.checkClassProportions(yy_kfold).round({'freq_perc': 2}))

    del kfold_data_spliter, kfold_train_data_spliter, kfold_test_data_spliter
    del kfold_train_index, kfold_test_index, xx_kfold, yy_kfold

    tmp_round = tmp_round + 1

del data_spliter, train_index, test_index

print()