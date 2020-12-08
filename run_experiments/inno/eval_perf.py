
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
import glob

import tensorflow as tf
import tensorflow_addons as tfa

from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

dataset_name = 'Diveface'
exp = 'inno'
exp_name = exp + '_alg_tl' # exp_7_alg_tl exp_9_alg_tl
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface
exp_name = exp_name + dataset_exacted
exp_name_suffix = '_b_300_e_50_a_1' # 30 60 90 120 150 180 210 240 270 300 330 360

# eval_set = ['training', 'valid'] # training valid test
eval_set = ['test']

# epoch = range(0,101)

random_seed = 0
test_size = 0.2
valid_size = 0

model_feature_size = 512

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Summary path
summary_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'summary', exp, exp_name + exp_name_suffix + '_run_' + str(random_seed)])
my_util.make_directory(summary_path)

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
# Randomly exclude to be equal to number of training samples in race classes
# exclude_perc = np.round(training_sep_idx.size/6/3/6)/np.round(training_sep_idx.size/6/3/2)
# [_, tmp_test_sep_idx, _] = my_util.split_data_by_id_and_classes(my_data.id.values[training_sep_idx], (my_data['gender'].iloc[training_sep_idx] + '-' + my_data['ethnicity'].iloc[training_sep_idx]).values, test_size=exclude_perc, valid_size=0, random_state=random_seed)
# training_sep_idx = training_sep_idx[tmp_test_sep_idx]
data_sep_idx = {'training': training_sep_idx, 'valid':valid_sep_idx, 'test':test_sep_idx}
# del exclude_perc, tmp_test_sep_idx
del training_sep_idx, test_sep_idx, valid_sep_idx
del my_data

#############################################################################################

# Function
def preprocess_data(_model, tmp_x_data):
    _tmp_x_data = _model.predict(tmp_x_data)
    _tmp_x_data = preprocessing.normalize(_tmp_x_data, norm='l2', axis=1, copy=True, return_norm=False)
    return _tmp_x_data

def eval_perf(_combined_id, _combined_x, _combined_y):
    [predictedScores, predictedY, test_time] = distance_model.predict(_combined_x[:,0:model_feature_size], _combined_x[:,model_feature_size:], _combined_y, unique_class, 1, distance_metric='euclidean')
    _tmp_performance_metric = my_util.biometric_metric(_combined_y, predictedScores, 'POS', score_order='ascending')
    del _tmp_performance_metric['threshold'], _tmp_performance_metric['fmr'], _tmp_performance_metric['fnmr']
    return _tmp_performance_metric

#############################################################################################

# Init
gender_class = np.array(['female', 'male'])
race_class = np.array(['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian'])
# uniqued_class = np.unique(y_race_data)
gender_in_race_class = pd.DataFrame(race_class)[0].str.split('-')
distance_model = paired_distance_alg()
unique_class = {'pos':'POS', 'neg':'NEG'}
chkp_fn = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name + exp_name_suffix + '_run_' + str(random_seed)])
chkp_fn = glob.glob(chkp_fn + 'cp-????.ckpt.index')
chkp_fn = [os.path.basename(name) for name in chkp_fn ]
chkp_fn = sorted(chkp_fn)

# Pair triplet
# Race
triplet_paired_list = {}
for eval_set_idx in eval_set:
    triplet_paired_list[eval_set_idx] = my_util.triplet_loss_paring(id_data[data_sep_idx[eval_set_idx]], class_data[data_sep_idx[eval_set_idx]], randomseed=random_seed)

# Initial triplets network model
model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name + exp_name_suffix + '_run_' + str(random_seed)])
proposed_model = tf.keras.models.Sequential()
proposed_model.add(tf.keras.layers.Dense(1024, input_dim=2048, activation='linear'))
proposed_model.add(tf.keras.layers.Dense(512, activation=None))
proposed_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
proposed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss())

# Evaluate
for eval_set_idx in eval_set:
    print('eval set: ' + eval_set_idx)
    performance_metric = {}
    performance_metric[eval_set_idx] = {}
    for epoch_val in tqdm(chkp_fn):
        epoch_idx = np.int(epoch_val[3:7])
        tmp_epoch_idx = str(epoch_idx).zfill(4)
        performance_metric[eval_set_idx][epoch_idx] = {}
        
        proposed_model.load_weights(model_path + 'cp-' + tmp_epoch_idx + '.ckpt')
        tmp_class_data = class_data[data_sep_idx[eval_set_idx]]
        tmp_id_data = id_data[data_sep_idx[eval_set_idx]]
        tmp_x_data = x_data[data_sep_idx[eval_set_idx]]
        tmp_y_data = y_gender_data[data_sep_idx[eval_set_idx]]
        tmp_x_data = preprocess_data(proposed_model, tmp_x_data)
        eval_x_data, eval_y_data, eval_id_data = my_util.combination_rule_paired_list(tmp_x_data, tmp_id_data, triplet_paired_list[eval_set_idx], combine_rule='concatenate')
        del tmp_class_data, tmp_id_data, tmp_x_data, tmp_y_data

        # Evaluate overall
        performance_metric[eval_set_idx][epoch_idx] = eval_perf(eval_id_data, eval_x_data, eval_y_data)
        
        print('AUC: ' + str(performance_metric[eval_set_idx][epoch_idx]['auc']) + ' - ' + str(epoch_idx))
        print('TAR@0d01: ' + str(performance_metric[eval_set_idx][epoch_idx]['tar_0d01']) + ' - ' + str(epoch_idx))

    # Save
    pickle_write = open((summary_path + exp_name + exp_name_suffix + '_run_' + str(random_seed) + '(' + eval_set_idx + ').pickle'), 'wb')
    pickle.dump(performance_metric, pickle_write)
    pickle_write.close()
    del pickle_write, performance_metric
    print('Saved')



print()
