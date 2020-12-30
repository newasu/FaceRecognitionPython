
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

#############################################################################################

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

dataset_name = 'Diveface'
exp = 'inno'
exp_name = exp + '_alg_tl'
dataset_exacted = 'resnet50'
exp_name = exp_name + dataset_exacted
exp_name_suffix = '_b_300_e_50_a_1'

random_seed = 0

#############################################################################################

# Initial triplets network model
model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name + exp_name_suffix + '_run_' + str(random_seed)])
proposed_model = tf.keras.models.Sequential()
proposed_model.add(tf.keras.layers.Dense(1024, input_dim=2048, activation='linear'))
proposed_model.add(tf.keras.layers.Dense(512, activation=None))
proposed_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
proposed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss())

proposed_model.load_weights(model_path + 'cp-' + '0017' + '.ckpt')

proposed_model.save('tp_model')
proposed_model.save('tp_model.h5')

del proposed_model

new_model = tf.keras.models.load_model('tp_model')
del new_model

new_model = tf.keras.models.load_model('tp_model.h5')
del new_model

print()