
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt

# Import my own lib
import others.utilities as my_util

#############################################################################################

dataset_exacted = 'resnet50'
exp = 'exp_7' # exp_7 exp_9
exp_name = exp + '_alg_tl'
exp_name = exp_name + dataset_exacted
exp_name_suffix = 'b_360_e_50_a_1'

exact_eval_set = ['training', 'valid', 'test']

train_class = ['female', 'male']
# train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

selected_epoch = [30, 30]

extracted_metric = ['auc', 'eer', 'tar_0', 'tar_0d01', 'tar_0d1', 'tar_1']

random_seed = 0
test_size = 0.3
valid_size = 0.1

model_feature_size = 1024

#############################################################################################

# Path
# Summary path
summary_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'summary', exp])

#############################################################################################

# Function
def initNaNVector(_vec_size):
    _tmp_vec = np.empty(_vec_size)
    _tmp_vec[:] = np.NaN
    return _tmp_vec

#############################################################################################

eval_class = ['overall', 'female', 'male', 'female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

# Baseline
if dataset_exacted == 'resnet50':
    baseline = {'auc':0.9769784915123455, 'eer':3.0694444444444446, 'tar_0':92.16666666666666, 'tar_0d01':92.16666666666666, 'tar_0d1':94.31944444444444, 'tar_1':97.35416666666666}

scores = {}
for exact_eval_set_idx in exact_eval_set:
    scores[exact_eval_set_idx] = {}
    # Prepare data
    full_exp_name = exp_name + '_' + exp_name_suffix + '_run_' + str(random_seed)
    file_path = summary_path + full_exp_name + os.sep + full_exp_name + '(' + exact_eval_set_idx + ')' + '.pickle'
    data = pickle.load(open(file_path, 'rb'))
    key_value = np.array(list(data[exact_eval_set_idx].keys()))
    tmp_metric = {}
    for extracted_metric_val in extracted_metric:
        tmp_metric[extracted_metric_val] = initNaNVector(key_value.max()+1)
    
    for eval_class_idx, eval_class_val in enumerate(eval_class):
        # Exact scores
        for key_value_idx in key_value:
            for extracted_metric_val in extracted_metric:
                tmp_metric[extracted_metric_val][key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class_val][extracted_metric_val]
        # # Append scores
        # if eval_class_idx == 0:
        #     for extracted_metric_val in extracted_metric:
        #         scores[exact_eval_set_idx][extracted_metric_val] = tmp_metric[extracted_metric_val][None,:]
        # else:
        #     if scores[exact_eval_set_idx]['auc'].shape[1] < tmp_metric[extracted_metric[0]].size:
        #         add_col_size = tmp_metric[extracted_metric[0]].size - scores[exact_eval_set_idx][extracted_metric[0]].shape[1]
        #         for extracted_metric_val in extracted_metric:
        #             scores[exact_eval_set_idx][extracted_metric_val] = np.pad(scores[exact_eval_set_idx][extracted_metric_val], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
        #     for extracted_metric_val in extracted_metric:
        #         scores[exact_eval_set_idx][extracted_metric_val] = np.vstack((scores[exact_eval_set_idx][extracted_metric_val], tmp_metric[extracted_metric_val]))



print()
