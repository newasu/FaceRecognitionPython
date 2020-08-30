
# Add project path to sys
import sys
import pathlib
# my_current_path = pathlib.Path(__file__).parent.absolute()
# my_root_path = my_current_path.parent.parent
# sys.path.insert(0, str(my_root_path))
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn import preprocessing
import dlib

import tensorflow as tf
# import tensorflow_addons as tfa
# import tensorflow_datasets as tfds

from keras_vggface import utils
from keras_vggface.vggface import VGGFace

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Import my own lib
import others.utilities as my_util

#############################################################################################

model_name = 'vgg16' # vgg16, resnet50

#############################################################################################

# Path
# Dataset path
dataset_name = 'Diveface'
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
raw_dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'DiveFace4K_cropped'])
# Model path
dlib_model_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Model', 'dlib'])

#############################################################################################

if model_name == 'vgg16':
    feature_size = 4096
elif model_name == 'resnet50':
    feature_size = 2048

# Data labels
my_data = pd.read_csv((dataset_path + 'Diveface_retinaface_nonorm_backup.txt'), sep=" ", header=0)
my_data_columns = my_data.columns[0:8]
my_data_columns = np.array(my_data_columns)
my_data_columns = np.append(my_data_columns, np.char.add(np.tile('feature_', (feature_size)), np.array(range(1, feature_size+1)).astype('U')))

# Declare face model
pretrained_model = VGGFace(model = model_name)
if model_name == 'vgg16':
    feature_layer = pretrained_model.get_layer('fc6/relu').output
    version = 1
elif model_name == 'resnet50':
    feature_layer = tf.keras.layers.Flatten(name='flatten')(pretrained_model.get_layer('avg_pool').output)
    version = 2
model = tf.keras.models.Model(pretrained_model.input, feature_layer)
# Declare face detector
# predictor_path = dlib_model_path + 'shape_predictor_5_face_landmarks.dat'
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor(predictor_path)
# Exact features
exacted_data = np.empty((0, feature_size))
for i in tqdm(range(0, my_data.shape[0])):
    img_path = raw_dataset_path + my_data.filepath[i][2:] + '/' + my_data.filename[i] + my_data.fileext[i]
    img = cv2.imread(img_path)
    # # Detect and align face
    # dets = detector(img, 1)
    # if len(dets) > 0:
    #     # Find the 5 face landmarks we need to do the alignment.
    #     faces = dlib.full_object_detections()
    #     for detection in dets:
    #         faces.append(sp(img, detection))
    #     img = dlib.get_face_chip(img, faces[0])
    #     del dets, faces
    #     # cv2.imwrite(my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store'])+'test.jpg', img)
    # else:
    #     print('Cannot detect face')
    # img = img/255
    # img = img.astype('float32')
    img = cv2.resize(img, (224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=version) # vgg = 1, resnet = 2
    feature_embedding = model.predict(img)
    
    # feature_embedding = preprocessing.normalize(feature_embedding, norm='l2', axis=1, copy=True, return_norm=False)
    exacted_data = np.vstack((exacted_data, feature_embedding))

exacted_data = np.concatenate((my_data.values[:,0:8], exacted_data), axis=1)
exacted_data = pd.DataFrame(exacted_data, columns=my_data_columns)

# Write
exacted_data.to_csv((dataset_path + 'Diveface_' + model_name + '_nonorm_new.txt'), header=True, index=False, sep=' ', mode='a')

print('Finished')