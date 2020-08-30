
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

gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#############################################################################################

dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface

exp = 'exp_9'
exp_name = exp + '_alg_tl' + dataset_exacted
exp_name_suffix = '_b_150_e_200_a_50'
all_classes = ['female', 'male']

random_seed = 0

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])

# Model path
model_path = {}
for idx in all_classes:
    model_path[idx] = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name + idx]) + exp_name + idx + '_run_' + str(random_seed)

#############################################################################################

# # Feature size
# if dataset_exacted == 'vgg16':
#     feature_size = 4096
# elif dataset_exacted == 'resnet50':
#     feature_size = 2048
feature_size = 1024

# Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
my_data_columns = my_data.columns[0:8]
my_data_columns = np.array(my_data_columns)
my_data_columns = np.append(my_data_columns, np.char.add(np.tile('feature_', (feature_size)), np.array(range(1, feature_size+1)).astype('U')))
data_class = my_data['gender'].values
# Load model
model = {}
for idx in all_classes:
    model[idx] = tf.keras.models.load_model(model_path[idx])
# Exact feature embeddings
exacted_data = np.empty((0, feature_size))
for idx in tqdm(range(0, my_data.shape[0])):
    feature_embedding = my_data.iloc[idx, 8:].values.astype(np.float64)[None,:]
    feature_embedding = model[data_class[idx]].predict(feature_embedding)
    # Append
    exacted_data = np.vstack((exacted_data, feature_embedding))

exacted_data = np.concatenate((my_data.values[:,0:8], exacted_data), axis=1)
exacted_data = pd.DataFrame(exacted_data, columns=my_data_columns)

# Write
exacted_data.to_csv((dataset_path + 'Diveface_' + dataset_exacted + '_nonorm_' + exp + exp_name_suffix + '.txt'), header=True, index=False, sep=' ', mode='a')

print('Finished')


