
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

dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface
dataset_suffix = '_nonorm' + '_exp_9_b_90_e_200_a_30' # _nonorm _nonorm_exp_7 _nonorm_exp_9 _nonorm_exp_9_30_30

random_seed = 0
test_size = 0.3
valid_size = 0.1

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
# exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name])

#############################################################################################

# Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + dataset_suffix + '.txt'), sep=" ", header=0)
# Separate data
[training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, (my_data['gender'] + '-' + my_data['ethnicity']).values, test_size=test_size, valid_size=valid_size, random_state=random_seed)
# Assign data
# Label
tmp_label = (my_data['gender'] + '-' + my_data['ethnicity']).values
new_label = tmp_label
# Training data
x_training = my_data.iloc[training_sep_idx,8:].values
y_training = my_data.id.iloc[training_sep_idx].values
y_class_training = new_label[training_sep_idx]
y_id_training = my_data.data_id.iloc[training_sep_idx].values
# Test data
x_test = my_data.iloc[test_sep_idx,8:].values
y_test = my_data.id.iloc[test_sep_idx].values
y_class_test = new_label[test_sep_idx]
y_id_test = my_data.data_id.iloc[test_sep_idx].values

del my_data, tmp_label, new_label
del x_training, y_training, y_class_training, y_id_training

# Normalize features
if dataset_exacted == 'vgg16' or dataset_exacted == 'resnet50':
    x_test = preprocessing.normalize(x_test, norm='l2', axis=1, copy=True, return_norm=False)

# Pair triplets function
def pair_triplets(_x, _y, _id):
    triplet_paired_list = my_util.triplet_loss_paring(_id, _y, randomseed=random_seed)
    [combined_xx, combined_yy, combined_idd] = my_util.combination_rule_paired_list(_x, _id, triplet_paired_list, combine_rule='concatenate')
    return combined_xx, combined_yy, combined_idd

# Pair triplets 
uniqued_class = np.unique(y_class_test)
combined_x = np.empty((0, x_test.shape[1]*2))
combined_y = np.empty(0)
combined_id = np.empty(0)
for class_idx in uniqued_class:
    tmp_idx = y_class_test == class_idx
    combined_xx, combined_yy, combined_idd = pair_triplets(x_test[tmp_idx], y_test[tmp_idx], y_id_test[tmp_idx])
    combined_x = np.vstack((combined_x, combined_xx))
    combined_y = np.append(combined_y, combined_yy)
    combined_id = np.append(combined_id, combined_idd)
    del combined_xx, combined_yy, combined_idd
del x_test, y_test, y_id_test, y_class_test

# Prepare variable
sep_idx = int(combined_x.shape[1]/2)
unique_class = {'pos':'POS', 'neg':'NEG'}

# Train model with best params
distance_model = paired_distance_alg()

# Test model
[predictedScores, predictedY, test_time] = distance_model.predict(combined_x[:,0:sep_idx], combined_x[:,sep_idx:], combined_y, unique_class, 1, distance_metric='euclidean')



# Eval performance
# Biometric metrics
performance_metric = my_util.biometric_metric(combined_y, predictedScores, 'POS', score_order='ascending')

print(performance_metric)


