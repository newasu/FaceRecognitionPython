# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import os
import glob

import cv2
from mtcnn.mtcnn import MTCNN

import tensorflow as tf
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

# Import my own lib
import others.utilities as my_util

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#############################################################################################

# Path
# Dataset path
lfw_test_cleaned_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw', 'DevTest', 'cleaned'])
lfw_test_crop_face_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw', 'DevTest', 'cropped'])
# Make directory
my_util.make_directory(lfw_test_crop_face_path)

#############################################################################################

# Initialise dataframe
my_column_names = ['gender', 'ethnicity', 'id', 'pose', 'path']
my_data = pd.DataFrame(columns=my_column_names)

# List files
dir_list = glob.glob(lfw_test_cleaned_path + '*/*')

i = 0
for fp in dir_list:
    # print(str(i))
    i = i+1
    
    # Get path
    my_path = fp
    my_dir, my_filename = os.path.split(my_path)

    # Extract class
    my_dir = my_dir.split('/')[-1]
    my_gender, my_ethnicity = my_dir.split('-')

    # Extract ID
    my_filename = my_filename.split('.')[0]
    tmp = my_filename.rfind('_')
    my_id = my_filename[0:tmp]
    my_pose = int(my_filename[tmp+1:])
    
    # Append
    my_data = my_data.append({'gender':my_gender, 'ethnicity':my_ethnicity, 'id':my_id, 'pose':my_pose, 'path':my_path}, ignore_index=True)

# Sort
my_data = my_data.sort_values(by=['id', 'pose'], ignore_index=True)

# Unique
id_unique = my_data['id'].unique()

# Check correction
for id_u in id_unique:
    tmp = my_data[my_data['id'] == id_u][['gender', 'ethnicity']]
    if tmp['gender'].unique().size > 1 or tmp['ethnicity'].unique().size > 1:
        print(id_u)

# Initialise pretrained
pretrained_model = VGGFace(model = 'resnet50')
feature_layer = tf.keras.layers.Flatten(name='flatten')(pretrained_model.get_layer('avg_pool').output)
model = tf.keras.models.Model(pretrained_model.input, feature_layer)

# Detect face
exacted_data = np.empty((0, 2048))
detector = MTCNN()
for img_idx in range(0,my_data.shape[0]):
    print(str(img_idx))
    # Load image
    img = cv2.cvtColor(cv2.imread(my_data['path'][img_idx]), cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    # Select biggest face
    face_idx = 0
    if len(faces) > 1:
        facesize = []
        for face in faces:
            _, _, width, height = face['box']
            facesize.append(width*height)
        face_idx = np.argmax(facesize)
    # Crop face
    bbox = np.array(faces[face_idx]['box'])
    bbox[bbox<0] = 0
    x1, y1, width, height = bbox
    x2, y2 = x1 + width, y1 + height
    img = img[y1:y2, x1:x2]
    # Convert color space
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Extract feature by pretrained
    img = cv2.resize(img, (224,224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2) # vgg = 1, resnet = 2
    feature_embedding = model.predict(img)
    exacted_data = np.vstack((exacted_data, feature_embedding))
    # Save img
    # cv2.imwrite((lfw_test_crop_face_path + '/' + my_data.id.iloc[img_idx] + '_' + str(my_data.pose.iloc[img_idx]).zfill(4)) + '.jpg', img)

# Concatenate
my_data_columns = my_column_names
my_data_columns = np.array(my_data_columns)
my_data_columns = np.append(my_data_columns, np.char.add(np.tile('feature_', (2048)), np.array(range(1, 2048+1)).astype('U')))
my_data = np.concatenate((my_data.values, exacted_data), axis=1)
my_data = pd.DataFrame(my_data, columns=my_data_columns)

# Save
my_data.to_csv('DevTest_cleaned.txt', header=True, index=False, sep=' ')

print()
