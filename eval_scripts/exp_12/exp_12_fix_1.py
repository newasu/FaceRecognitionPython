import sys
sys.path.append("././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import glob
from shutil import copyfile

import tensorflow as tf
import tensorflow_addons as tfad
from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
# from algorithms.selm import selm
from algorithms.welm import welm
from algorithms.paired_distance_alg import paired_distance_alg

# gpu_id = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# # Clear GPU cache
# tf.keras.backend.clear_session()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

#############################################################################################

eval_alg = 'baseline' # selm baseline
race_classify_mode = 'none' # auto manual none

#############################################################################################

# Path
# Dataset path
diveface_diveface_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
lfw_dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw', 'DevTest'])
# Model path
gender_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', 'exp_12', '_gender_welm', '_gender_welm' + '_run_0.npy'])
ethnicity_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', 'exp_12', '_ethnicity_welm', '_ethnicity_welm' + '_run_0.npy'])
# incorrect list path
incorrect_list_path = my_util.get_path(additional_path=['tmp']) + 'save_incorrect_list_' + eval_alg + '_' + race_classify_mode + '.txt'
# Save image path
save_img_path = my_util.get_path(additional_path=['tmp', 'images'])

#############################################################################################

def query_and_copy(search_idx, _incorrect_list, _my_list_files, _save_img_path):
    fname_anchor = _incorrect_list.iloc[search_idx]['fname_anchor']
    fname_compare = _incorrect_list.iloc[search_idx]['fname_compare']

    fname_anchor_path = _my_list_files[_my_list_files['filename']==fname_anchor]['path']
    fname_compare_path = _my_list_files[_my_list_files['filename']==fname_compare]['path']

    # copyfile(fname_anchor_path.values[0], (_save_img_path+'anchor_' + fname_anchor[0:fname_anchor.rfind('_')] + '.jpg'))
    # copyfile(fname_compare_path.values[0], (_save_img_path+'compare_' + fname_compare[0:fname_compare.rfind('_')] + '.jpg'))
    copyfile(fname_anchor_path.values[0], (_save_img_path + 'anchor' + '.jpg'))
    copyfile(fname_compare_path.values[0], (_save_img_path + 'compare' + '.jpg'))
    
    print(_incorrect_list.iloc[search_idx])

#############################################################################################

# Load model
# gender_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', 'exp_12', 'exp_12_gender_welm', 'exp_12_gender_welm' + '_run_0.npy'])
# ethnicity_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', 'exp_12', 'exp_12_ethnicity_welm', 'exp_12_ethnicity_welm' + '_run_0.npy'])
# gender_model = my_util.load_numpy_file(gender_model_path[:-1])
# ethnicity_model = my_util.load_numpy_file(ethnicity_model_path[:-1])

# Read fix list
incorrect_list = pd.read_csv(incorrect_list_path, sep=' ', header=None)
incorrect_list = incorrect_list[0].str.split('--', 8, expand=True)
incorrect_list.columns = ['fname_anchor', 'fname_compare', 'true_race_anchor', 'true_race_compare', 'predicted_race_anchor', 'predicted_race_compare', 'true_label', 'predicted_label']
incorrect_list = incorrect_list.assign(isFixed=np.tile(False, incorrect_list.shape[0])) 

# List files
my_list_files = pd.DataFrame(columns=['filename', 'path'])
dir_list = glob.glob(lfw_dataset_path  + 'cropped/*')
for fp in dir_list:
    # Get path
    my_path = fp
    my_dir, my_filename = os.path.split(my_path)
    my_list_files = my_list_files.append({'path':my_path, 'filename':my_filename}, ignore_index=True)

del fp, my_path, my_dir, my_filename, dir_list

# copyfile(source_path, destination_path)

query_and_copy(0, incorrect_list, my_list_files, save_img_path)



print()
