
# Add project path to sys
import sys
# import pathlib
# my_current_path = pathlib.Path(__file__).parent.absolute()
# my_root_path = my_current_path.parent
# sys.path.insert(0, str(my_root_path))
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
import tensorflow_datasets as tfds

from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util

#############################################################################################
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

exp_name = 'exp_8_alg_tl'
dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface
exp_name = exp_name + dataset_exacted

train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
train_class = train_class[4]
exp_name = exp_name + train_class

batch_size = 90
epoch = 1000

random_seed = 0
test_size = 0.3
valid_size = 0.1

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name])
# Grid search path
gridsearch_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'gridsearch', (exp_name + '_run_' + str(random_seed))])
# Make directory
my_util.make_directory(exp_result_path)
my_util.make_directory(gridsearch_path)

#############################################################################################

# Load data
my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
# Separate data
[training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, (my_data['gender'] + '-' + my_data['ethnicity']).values, test_size=test_size, valid_size=valid_size, random_state=random_seed)
# Assign data
# Label
tmp_label = (my_data['gender'] + '-' + my_data['ethnicity']).values
new_label = np.zeros(tmp_label.shape)
new_label[tmp_label == train_class] = 1
# Training data
data_id_training = my_data.id.iloc[training_sep_idx].values
x_training = my_data.iloc[training_sep_idx,8:].values
y_training = new_label[training_sep_idx]
# Validate data
data_id_valid = my_data.id.iloc[valid_sep_idx].values
x_valid = my_data.iloc[valid_sep_idx,8:].values
y_valid = new_label[valid_sep_idx]

# img_per_class = 3
step_per_epoch = np.floor(((y_training.size/6)*2)/batch_size).astype(int)
validation_step_per_epoch = np.floor(((y_valid.size/6)*2)/batch_size).astype(int)
def generator(x_data, y_data):
    batch_step = batch_size/3
    pos_batch_step = np.floor(batch_step).astype(int)
    neg_batch_step = np.floor(batch_step*2).astype(int)
    random_seed_count = 0
    sample_pos_idx = np.where(y_data==1)[0]
    sample_neg_idx = np.where(y_data==0)[0]
    sample_idx = np.empty(0)
    while True:
        # Feed data
        if sample_idx.size == 0 or sample_idx.size < batch_size:
            # print(random_seed_count)
            shuffled_pos = sample_pos_idx
            shuffled_neg = sample_neg_idx
            random.Random(random_seed+random_seed_count).shuffle(shuffled_pos)
            random.Random(random_seed+random_seed_count).shuffle(shuffled_neg)
            random_seed_count = random_seed_count + 1
            # Prepare list
            while shuffled_pos.size >= pos_batch_step:
                sample_idx = np.append(sample_idx, shuffled_pos[0:pos_batch_step])
                sample_idx = np.append(sample_idx, shuffled_neg[0:neg_batch_step])
                shuffled_pos = np.delete(shuffled_pos, range(0,pos_batch_step))
                shuffled_neg = np.delete(shuffled_neg, range(0,neg_batch_step))
            sample_idx = sample_idx.astype(int)
        tmp_sample_idx = sample_idx[0:batch_size]
        sample_idx = np.delete(sample_idx, range(0,batch_size))
        # print(x_data[tmp_sample_idx].shape, y_data[tmp_sample_idx].shape)
        yield x_data[tmp_sample_idx], y_data[tmp_sample_idx]

# generator(x_training, y_training)

# Initial triplets network
proposed_model = tf.keras.models.Sequential()
proposed_model.add(tf.keras.layers.Dense(1024, input_dim=x_training.shape[1], activation='linear'))
# proposed_model.add(tf.keras.layers.Dropout(0.1))
# proposed_model.add(tf.keras.layers.Dense(512, activation=None))
proposed_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
proposed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss())

# Create a callback
checkpoint_path = gridsearch_path + 'cp-{epoch:04d}.ckpt'
my_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_weights_only=True)
proposed_model.save_weights(checkpoint_path.format(epoch=0))

# Train the network
tf.random.set_seed(random_seed)
history = proposed_model.fit(generator(x_training, y_training), steps_per_epoch=step_per_epoch, validation_data=generator(x_valid, y_valid), validation_steps=validation_step_per_epoch, epochs=epoch, callbacks=my_callbacks)

# Save model
proposed_model.save(exp_result_path + exp_name + '_run_' + str(random_seed))



# Evaluate the network
# results = proposed_model.predict(test_dataset)

print()