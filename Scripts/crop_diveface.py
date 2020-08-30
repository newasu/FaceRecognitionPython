
# Add project path to sys
import sys
import pathlib
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn import preprocessing
import dlib
from mtcnn.mtcnn import MTCNN
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Import my own lib
import others.utilities as my_util

#############################################################################################

model_name = 'resnet50' # vgg16, resnet50

#############################################################################################

# Path
# Dataset path
dataset_name = 'Diveface'
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
raw_dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'DiveFace4K_Full'])
save_dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'DiveFace4K_cropped'])
# Model path
dlib_model_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Model', 'dlib'])

#############################################################################################

# Data labels
my_data = pd.read_csv((dataset_path + 'Diveface_retinaface_nonorm_backup.txt'), sep=" ", header=0)

# Declare face detector
predictor_path = dlib_model_path + 'shape_predictor_5_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
# Exact features
mtcnn_detector = MTCNN()
for i in tqdm(range(0, my_data.shape[0])):
    img_path = raw_dataset_path + my_data.filepath[i][2:] + '/' + my_data.filename[i] + my_data.fileext[i]
    img = cv2.imread(img_path)
    # Detect face
    isFound = 0
    dets = mtcnn_detector.detect_faces(img)
    if len(dets) > 0:
        # x1, y1, width, height = dets[0]['box']
        # x2, y2 = x1 + width, y1 + height
        # isFound = 1
        bbox = np.array(dets[0]['box'])
        isFound = 1
    else:
        dets = detector(img, 1)
        if len(dets) > 0:
            # x1 = dets[0].left()
            # y1 = dets[0].top()
            # width = dets[0].right()
            # height = dets[0].bottom()
            # x2, y2 = x1 + width, y1 + height
            # isFound = 1
            bbox = np.array([dets[0].left(), dets[0].top(), dets[0].right(), dets[0].bottom()])
            isFound = 1
    
    if isFound == 1:
        bbox[bbox<0] = 0
        x1, y1, width, height = bbox
        x2, y2 = x1 + width, y1 + height
        img = img[y1:y2, x1:x2]
    else:
        print('Not found face')
    
    # Write image
    image_save_path = save_dataset_path + my_data.filepath[i][2:]
    my_util.make_directory(image_save_path, doSilent=True)
    cv2.imwrite((image_save_path + '/' + my_data.filename[i] + my_data.fileext[i]), img)

print('Finished')