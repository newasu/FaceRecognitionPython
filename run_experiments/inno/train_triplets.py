
# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import random

import tensorflow as tf
import tensorflow_addons as tfa

from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util

#############################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

exp = 'inno'
exp_name = exp + '_alg_tl'
dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface
exp_name = exp_name + dataset_exacted

exp_name = exp_name

img_per_class = 3
numb_class_each = 100 # 400 2400

batch_size = img_per_class * numb_class_each
epoch = 50
learning_rate = 0.0001

training_augment = 1
valid_augment = 1

random_seed = 0
test_size = 0.2
valid_size = 0

exp_name = exp_name + '_b_' + str(batch_size) + '_e_' + str(epoch) + '_a_' + str(training_augment)

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, exp_name])
# Grid search path
gridsearch_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, (exp_name + '_run_' + str(random_seed))])
# Make directory
my_util.make_directory(exp_result_path)
my_util.make_directory(gridsearch_path)

#############################################################################################

# Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
# Separate data
[training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, (my_data['gender'] + '-' + my_data['ethnicity']).values, test_size=test_size, valid_size=valid_size, random_state=random_seed)
# Randomly exclude to be equal to number of training samples in race classes
# exclude_perc = np.round(training_sep_idx.size/6/img_per_class/6)/np.round(training_sep_idx.size/6/img_per_class/2)
# [_, tmp_test_sep_idx, _] = my_util.split_data_by_id_and_classes(my_data.id.values[training_sep_idx], (my_data['gender'].iloc[training_sep_idx] + '-' + my_data['ethnicity'].iloc[training_sep_idx]).values, test_size=exclude_perc, valid_size=0, random_state=random_seed)
# training_sep_idx = training_sep_idx[tmp_test_sep_idx]
# class_proportion = my_util.checkClassProportions((my_data['gender'] + '-' + my_data['ethnicity']).values)
# del exclude_perc, tmp_test_sep_idx
# Assign data
# Training data
data_id_training = my_data.id.iloc[training_sep_idx].values
x_training = my_data.iloc[training_sep_idx,8:].values
y_training = data_id_training

step_per_epoch = np.round(y_training.size/batch_size).astype(int) * training_augment
def generator(x_data, y_data):
    ind = np.argsort(y_data)
    y_data = y_data[ind]
    x_data = x_data[ind]
    del ind
    random_seed_count = 0
    sample_idx = np.empty(0)
    while True:
        # Feed data
        if sample_idx.size < batch_size:
            sample_idx = np.empty(0)
            # print(random_seed_count)
            shuffled_y = np.unique(y_data)
            random.Random(random_seed+random_seed_count).shuffle(shuffled_y)
            random_seed_count = random_seed_count + 1
            for y_idx in shuffled_y:
                tmp_sample_idx = np.in1d(y_data, y_idx).nonzero()[0]
                tmp_sample_idx = np.append(tmp_sample_idx, np.tile(tmp_sample_idx[-1], (img_per_class-len(tmp_sample_idx))))
                sample_idx = np.append(sample_idx, tmp_sample_idx)
            sample_idx = sample_idx.astype(int)
        tmp_sample_idx = sample_idx[0:batch_size]
        random.Random(random_seed+random_seed_count).shuffle(tmp_sample_idx)
        sample_idx = np.delete(sample_idx, range(0,batch_size))
        # print(x_data[tmp_sample_idx].shape, y_data[tmp_sample_idx].shape)
        yield x_data[tmp_sample_idx], y_data[tmp_sample_idx]

# generator(x_training, y_training, 2)

# Initial triplets network
proposed_model = tf.keras.models.Sequential()
proposed_model.add(tf.keras.layers.Dense(1024, input_dim=x_training.shape[1], activation='linear'))
# proposed_model.add(tf.keras.layers.Dropout(0.1))
proposed_model.add(tf.keras.layers.Dense(512, activation=None))
proposed_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
proposed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tfa.losses.TripletSemiHardLoss())

# Create a callback
checkpoint_path = gridsearch_path + 'cp-{epoch:04d}.ckpt'
my_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_weights_only=True)
proposed_model.save_weights(checkpoint_path.format(epoch=0))

# Train the network
tf.random.set_seed(random_seed)
history = proposed_model.fit(generator(x_training, y_training), steps_per_epoch=step_per_epoch, epochs=epoch, callbacks=my_callbacks)

# Save model
proposed_model.save(exp_result_path + exp_name + '_run_' + str(random_seed))
history.history['epoch'] = history.epoch
pickle_history = open((exp_result_path + exp_name + '_run_' + str(random_seed) + '.pickle'), 'wb')
pickle.dump(history.history, pickle_history)
pickle_history.close()


# Evaluate the network
# results = proposed_model.predict(test_dataset)

print()