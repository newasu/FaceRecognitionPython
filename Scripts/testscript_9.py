
import sys
sys.path.append("././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import random

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util

#############################################################################################

exp_name = 'exp_9_alg_tlresnet50male'

dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface

random_seed = 0
test_size = 0.3
valid_size = 0.1

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name])

#############################################################################################

# img_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store'])
# img_bbbox = [65,66,125,166]
# img = cv2.imread(img_path + '0015_01.jpg')
# img = img[img_bbbox[1]:img_bbbox[1]+img_bbbox[3],img_bbbox[0]:img_bbbox[0]+img_bbbox[2],:]
# cv2.imwrite(img_path+'test.jpg', img)

# Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
# Separate data
[training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, (my_data['gender'] + '-' + my_data['ethnicity']).values, test_size=test_size, valid_size=valid_size, random_state=random_seed)
# Assign data
# Label
tmp_label = my_data['gender'].values
new_label = tmp_label
# Training data
data_id_training = my_data.id.iloc[training_sep_idx].values
x_training = my_data.iloc[training_sep_idx,8:].values
y_training = new_label[training_sep_idx]
# Validate data
data_id_valid = my_data.id.iloc[valid_sep_idx].values
x_valid = my_data.iloc[valid_sep_idx,8:].values
y_valid = new_label[valid_sep_idx]

proposed_model = tf.keras.models.load_model(exp_result_path + exp_name + '_run_' + str(random_seed))
# proposed_model.load_weights()
results = proposed_model.predict(x_valid)