
import sys
sys.path.append("././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

# Import my own lib
import others.utilities as my_util

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#############################################################################################

filename_comment = 'eer'

# param = {'exp':'exp_7', 
#          'model': ['b_270_e_50_a_1', 'b_240_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_360_e_50_a_1', 'b_240_e_50_a_1'], 
#          'epoch': [43, 31, 35, 41, 29, 28], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

param = {'exp':'exp_7', 
         'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_240_e_50_a_1', 'b_360_e_50_a_1', 'b_270_e_50_a_1', 'b_240_e_50_a_1'], 
         'epoch': [36, 32, 42, 25, 35, 28], 
         'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface

exp = param['exp']
exp_name = exp + '_alg_tl' + dataset_exacted

random_seed = 0

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])

#############################################################################################

# # Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
# Label
# class_data = my_data.id.values
# id_data = my_data.data_id.values
info_data = my_data.iloc[:,:8].values
x_data = my_data.iloc[:,8:].values
# y_gender_data = my_data['gender'].values
y_race_data = (my_data['gender'] + '-' + my_data['ethnicity']).values

#############################################################################################

# Feature size
if dataset_exacted == 'vgg16':
    feature_size = 4096
elif dataset_exacted == 'resnet50':
    feature_size = 2048
proposed_model_feature_size = 1024

race_classes = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
my_data_columns = my_data.columns[0:8]
my_data_columns = np.array(my_data_columns)
my_data_columns = np.append(my_data_columns, np.char.add(np.tile('feature_', (proposed_model_feature_size)), np.array(range(1, proposed_model_feature_size+1)).astype('U')))
del my_data

# Initial triplets network model
model_path = {}
proposed_model = {}
for class_idx, class_val in enumerate(param['class']):
    model_path[class_val] = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name + class_val + '_' + param['model'][class_idx] + '_run_' + str(random_seed)])
    proposed_model[class_val] = tf.keras.models.Sequential()
    proposed_model[class_val].add(tf.keras.layers.Dense(proposed_model_feature_size, input_dim=feature_size, activation='linear'))
    proposed_model[class_val].add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    proposed_model[class_val].compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss())
    proposed_model[class_val].load_weights(model_path[class_val] + 'cp-' + str(param['epoch'][class_idx]).zfill(4) + '.ckpt')

exacted_data = np.empty((y_race_data.size, proposed_model_feature_size))
for class_val in tqdm(param['class']):
    tmp_idx = np.where(y_race_data == class_val)[0]
    feature_embedding = x_data[tmp_idx]
    feature_embedding = proposed_model[class_val].predict(feature_embedding)
    exacted_data[tmp_idx] = feature_embedding
    del feature_embedding

exacted_data = np.concatenate((info_data, exacted_data), axis=1)
exacted_data = pd.DataFrame(exacted_data, columns=my_data_columns)

# Write
exacted_data.to_csv((dataset_path + 'Diveface_' + dataset_exacted + '_' + exp + '_run_' + str(random_seed) + '(' + filename_comment + ').txt'), header=True, index=False, sep=' ', mode='a')

print('Finished')


