import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import random
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

param = {'exp':'exp_7', 'exp_name': 'baseline',
         'model': ['baseline', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline'],  
         'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

# param = {'exp':'exp_7', 'exp_name': 'racebaseline',
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_240_e_50_a_1', 'b_360_e_50_a_1', 'b_270_e_50_a_1', 'b_240_e_50_a_1'], 
#          'epoch': [36, 32, 42, 25, 35, 28], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian'], 'class-model': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

# param = {'exp':'exp_8', 'exp_name': 'racebaseline',
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1'], 
#          'epoch': [36, 36, 36, 17, 17, 17], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian'], 'class-model': ['female', 'female', 'female', 'male', 'male', 'male']}

# param = {'exp':'exp_9', 'exp_name': 'racebaseline',
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_360_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1'], 
#          'epoch': [16, 16, 16, 17, 17, 17], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

exp = param['exp']
exp_name = exp + '_' + param['exp_name'] + '_' + param['exp'] # _baseline _racebaseline
dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface
exp_name = exp_name + '_' + dataset_exacted

eval_set = ['test'] # training valid test

random_seed = 0
test_size = 0.3
valid_size = 0.1

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Summary path
summary_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'summary', exp])
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
    _model_threshold = distance_model.train(_combined_x[:,0:_feature_size], _combined_x[:,_feature_size:], _combined_y, unique_class)
    predictedScores, predictedY, _ = distance_model.predict(_combined_x[:,0:_feature_size], _combined_x[:,_feature_size:], _combined_y, unique_class, _model_threshold[0], distance_metric='euclidean')
    _tmp_performance_metric = my_util.biometric_metric(_combined_y, predictedScores, 'POS', score_order='ascending', threshold_step=0.01)
    del _tmp_performance_metric['threshold'], _tmp_performance_metric['fmr'], _tmp_performance_metric['fnmr']
    _tmp_performance_metric.update(my_util.classification_performance_metric(_combined_y, predictedY, np.array(['NEG', 'POS'])))
    return _tmp_performance_metric

#############################################################################################

gender_class = np.array(['female', 'male'])
race_class = np.array(['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian'])
distance_model = paired_distance_alg()
unique_class = {'pos':'POS', 'neg':'NEG'}
# Feature size
if dataset_exacted == 'vgg16':
    feature_size = 4096
elif dataset_exacted == 'resnet50':
    feature_size = 2048
proposed_model_feature_size = 1024

# Extract features
if param['exp_name'] == 'baseline':
    exacted_data = np.empty((y_race_data.size, feature_size))
else:
    exacted_data = np.empty((y_race_data.size, proposed_model_feature_size))
for class_idx, class_val in enumerate(param['class']):
    tmp_idx = np.where(y_race_data == class_val)[0]
    feature_embedding = x_data[tmp_idx]
    if param['model'][class_idx] != 'baseline':
        # Load model
        model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', param['exp'], param['exp'] + '_alg_tl' + dataset_exacted + param['class-model'][class_idx] + '_' + param['model'][class_idx] + '_run_' + str(random_seed)])
        proposed_model = tf.keras.models.Sequential()
        proposed_model.add(tf.keras.layers.Dense(proposed_model_feature_size, input_dim=feature_size, activation='linear'))
        proposed_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
        proposed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss())
        proposed_model.load_weights(model_path + 'cp-' + str(param['epoch'][class_idx]).zfill(4) + '.ckpt')
        # Extract features
        feature_embedding = proposed_model.predict(feature_embedding)
        del model_path, proposed_model
    exacted_data[tmp_idx] = feature_embedding
    del tmp_idx, feature_embedding
x_data = exacted_data
del exacted_data

# Pair triplet
# Race
race_triplet_paired_list = {}
for eval_set_idx in eval_set:
    race_triplet_paired_list[eval_set_idx] = {}
    for race_class_idx in race_class:
        tmp_idx = y_race_data[data_sep_idx[eval_set_idx]] == race_class_idx
        race_triplet_paired_list[eval_set_idx][race_class_idx] = my_util.triplet_loss_paring(id_data[data_sep_idx[eval_set_idx]][tmp_idx], class_data[data_sep_idx[eval_set_idx]][tmp_idx], randomseed=random_seed)
        del tmp_idx

# Evaluate
for eval_set_idx in eval_set:
    print('eval set: ' + eval_set_idx)
    performance_metric = {}
    performance_metric[eval_set_idx] = {}

    eval_id_data = {}
    eval_x_data = {}
    eval_y_data = {}
    for race_class_idx in tqdm(race_class):
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

    # Save
    # pickle_write = open((summary_path + exp_name + '_run_' + str(random_seed) + '(' + eval_set_idx + ').pickle'), 'wb')
    # pickle.dump(performance_metric, pickle_write)
    # pickle_write.close()
    # del pickle_write, performance_metric
    # print('Saved')



print()
