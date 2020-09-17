import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import random

# import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_datasets as tfds

from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

exp = 'exp_7'
exp_name = exp + '_baseline'
dataset_name = 'Diveface'
dataset_exacted = 'vgg16' # vgg16 resnet50 retinaface
exp_name = exp_name + dataset_exacted

eval_set = ['test'] # training valid test

random_seed = 0
test_size = 0.3
valid_size = 0.1

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Summary path
summary_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'summary', exp])
#############################################################################################

# Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
# Label
class_data = my_data.id.values
id_data = my_data.data_id.values
x_data = my_data.iloc[:,8:].values
y_gender_data = my_data['gender'].values
y_race_data = (my_data['gender'] + '-' + my_data['ethnicity']).values
# Separate data
[training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, y_race_data, test_size=test_size, valid_size=valid_size, random_state=random_seed)
data_sep_idx = {'training': training_sep_idx, 'valid':valid_sep_idx, 'test':test_sep_idx}
del training_sep_idx, test_sep_idx, valid_sep_idx
del my_data

#############################################################################################

# Function
def assign_data(_sep_idx, _class_idx):
    _tmp_class_data = class_data[_sep_idx]
    _tmp_id_data = id_data[_sep_idx]
    _tmp_x_data = x_data[_sep_idx]
    if _class_idx in gender_class:
        _tmp_y_data = y_gender_data[_sep_idx]
    else:
        _tmp_y_data = y_race_data[_sep_idx]
    _idx = _tmp_y_data == _class_idx
    _tmp_class_data = _tmp_class_data[_idx]
    _tmp_id_data = _tmp_id_data[_idx]
    _tmp_x_data = _tmp_x_data[_idx]
    _tmp_y_data = _tmp_y_data[_idx]
    return _tmp_class_data, _tmp_id_data, _tmp_x_data, _tmp_y_data

def prepare_data_from_class(_class_idx, _eval_id_data, _eval_x_data, _eval_y_data):
    _feature_size = _eval_x_data[_class_idx[0]].shape[1]
    combined_id = np.empty(0)
    combined_x = np.empty((0, _feature_size))
    combined_y = np.empty(0)
    for _tmp_class_idx in _class_idx:
        combined_id = np.append(combined_id, eval_id_data[_tmp_class_idx])
        combined_x = np.vstack((combined_x, eval_x_data[_tmp_class_idx]))
        combined_y = np.append(combined_y, eval_y_data[_tmp_class_idx])
    return combined_id, combined_x, combined_y

def eval_perf(_combined_id, _combined_x, _combined_y):
    _feature_size = np.int((_combined_x.shape[1]/2))
    [predictedScores, predictedY, test_time] = distance_model.predict(_combined_x[:,0:_feature_size], _combined_x[:,_feature_size:], _combined_y, unique_class, 1, distance_metric='euclidean')
    _tmp_performance_metric = my_util.biometric_metric(_combined_y, predictedScores, 'POS', score_order='ascending', threshold_step=0.01)
    del _tmp_performance_metric['threshold'], _tmp_performance_metric['fmr'], _tmp_performance_metric['fnmr']
    return _tmp_performance_metric

#############################################################################################

gender_class = np.array(['female', 'male'])
race_class = np.array(['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian'])
distance_model = paired_distance_alg()
unique_class = {'pos':'POS', 'neg':'NEG'}

# Pair triplet
# Race
race_triplet_paired_list = {}
for eval_set_idx in eval_set:
    race_triplet_paired_list[eval_set_idx] = {}
    for race_class_idx in race_class:
        tmp_idx = y_race_data[data_sep_idx[eval_set_idx]] == race_class_idx
        race_triplet_paired_list[eval_set_idx][race_class_idx] = my_util.triplet_loss_paring(id_data[data_sep_idx[eval_set_idx]][tmp_idx], class_data[data_sep_idx[eval_set_idx]][tmp_idx], randomseed=random_seed)
        del tmp_idx

# Evaludate
for eval_set_idx in eval_set:
    print('eval set: ' + eval_set_idx)
    performance_metric = {}
    performance_metric[eval_set_idx] = {}

    eval_id_data = {}
    eval_x_data = {}
    eval_y_data = {}
    for race_class_idx in race_class:
        # Prepare data each class
        tmp_class_data, tmp_id_data, tmp_x_data, tmp_y_data = assign_data(data_sep_idx[eval_set_idx], race_class_idx)
        # Combination data
        [eval_x_data[race_class_idx], eval_y_data[race_class_idx], eval_id_data[race_class_idx]] = my_util.combination_rule_paired_list(tmp_x_data, tmp_id_data, race_triplet_paired_list[eval_set_idx][race_class_idx], combine_rule='concatenate')
        del tmp_class_data, tmp_id_data, tmp_x_data, tmp_y_data
        # Evaluate
        performance_metric[eval_set_idx][race_class_idx] = eval_perf(eval_id_data[race_class_idx], eval_x_data[race_class_idx], eval_y_data[race_class_idx])
        pass
    
    # Evaluate overall
    combined_id, combined_x, combined_y = prepare_data_from_class(race_class, eval_id_data, eval_x_data, eval_y_data)
    performance_metric[eval_set_idx]['overall'] = eval_perf(combined_id, combined_x, combined_y)
    del combined_id, combined_x, combined_y
    del eval_id_data, eval_x_data, eval_y_data

    # # Save
    pickle_write = open((summary_path + exp_name + '_run_' + str(random_seed) + '(' + eval_set_idx + ').pickle'), 'wb')
    pickle.dump(performance_metric, pickle_write)
    pickle_write.close()
    del pickle_write, performance_metric
    print('Saved')



print()
