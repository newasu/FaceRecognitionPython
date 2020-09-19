
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

# AUC Valid
# param = {'exp':'exp_7', 
#          'model': ['b_210_e_50_a_1', 'b_210_e_50_a_1', 'b_270_e_50_a_1', 'b_270_e_50_a_1', 'b_330_e_50_a_1', 'b_240_e_50_a_1'], 
#          'epoch': [29, 12, 15, 25, 37, 18], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_8', 
#          'model': ['b_270_e_50_a_1', 'b_270_e_50_a_1', 'b_270_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1'], 
#          'epoch': [48, 48, 48, 49, 49, 49], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_9', 
#          'model': ['b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1'], 
#          'epoch': [49, 49, 49, 38, 38, 38], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

# EER Valid
# param = {'exp':'exp_7', 
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_240_e_50_a_1', 'b_360_e_50_a_1', 'b_270_e_50_a_1', 'b_240_e_50_a_1'], 
#          'epoch': [36, 32, 42, 25, 35, 28], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_8', 
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1'], 
#          'epoch': [36, 36, 36, 17, 17, 17], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_9', 
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_360_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1'], 
#          'epoch': [16, 16, 16, 17, 17, 17], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

# TAR 0.1 Valid
# param = {'exp':'exp_7', 
#          'model': ['b_270_e_50_a_1', 'b_240_e_50_a_1', 'b_240_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_240_e_50_a_1'], 
#          'epoch': [43, 31, 30, 41, 26, 28], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_8', 
#          'model': ['b_360_e_50_a_1', 'b_360_e_50_a_1', 'b_360_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1'], 
#          'epoch': [50, 50, 50, 37, 37, 37], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_9', 
#          'model': ['b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1'], 
#          'epoch': [17, 17, 17, 18, 18, 18], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

# TAR 0.01 Valid
# param = {'exp':'exp_7', 
#          'model': ['b_270_e_50_a_1', 'b_240_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_360_e_50_a_1', 'b_240_e_50_a_1'], 
#          'epoch': [43, 31, 35, 41, 29, 28], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
param = {'exp':'exp_8', 
         'model': ['b_210_e_50_a_1', 'b_210_e_50_a_1', 'b_210_e_50_a_1', 'b_300_e_50_a_1', 'b_300_e_50_a_1', 'b_300_e_50_a_1'], 
         'epoch': [21, 21, 21, 39, 39, 39], 
         'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}
# param = {'exp':'exp_9', 
#          'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1', 'b_330_e_50_a_1'], 
#          'epoch': [9, 9, 9, 17, 17, 17], 
#          'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

dataset_exacted = 'resnet50'
exp = param['exp'] # exp_7 exp_9
exp_name = exp + '_alg_tl'
exp_name = exp_name + dataset_exacted
# exp_name_suffix = param['model']

# exact_eval_set = ['training', 'valid', 'test']
exact_eval_set = ['valid', 'test']

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

gender_class = ['female', 'male']
race_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
eval_class = gender_class + race_class

# Baseline
if dataset_exacted == 'resnet50':
    baseline = {'auc':0.9769784915123455, 'eer':3.0694444444444446, 'tar_0':92.16666666666666, 'tar_0d01':92.16666666666666, 'tar_0d1':94.31944444444444, 'tar_1':97.35416666666666}

# Exact scores
scores = {}
for exact_eval_set_idx in exact_eval_set:
    scores[exact_eval_set_idx] = {}
    for model_val in param['model']:
        scores[exact_eval_set_idx][model_val] = {}
        # Prepare data
        full_exp_name = exp_name + '_' + model_val + '_run_' + str(random_seed)
        file_path = summary_path + full_exp_name + os.sep + full_exp_name + '(' + exact_eval_set_idx + ')' + '.pickle'
        data = pickle.load(open(file_path, 'rb'))
        key_value = np.array(list(data[exact_eval_set_idx].keys()))

        for eval_class_idx, eval_class_val in enumerate(eval_class):
            # Initial nan vector
            tmp_metric = {}
            for extracted_metric_val in extracted_metric:
                tmp_metric[extracted_metric_val] = initNaNVector(key_value.max()+1)
            # Exact scores
            for key_value_idx in key_value:
                for extracted_metric_val in extracted_metric:
                    tmp_metric[extracted_metric_val][key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class_val][extracted_metric_val]
            # Append scores
            scores[exact_eval_set_idx][model_val][eval_class_val] = {}
            for extracted_metric_val in extracted_metric:
                scores[exact_eval_set_idx][model_val][eval_class_val][extracted_metric_val] = tmp_metric[extracted_metric_val][None,:]
            del tmp_metric
        # Average
        scores[exact_eval_set_idx][model_val]['overall'] = {}
        for extracted_metric_val in extracted_metric:
            tmp_metric = np.empty((0, key_value.max()+1))
            for race_class_val in race_class:
                tmp_metric = np.vstack((tmp_metric, scores[exact_eval_set_idx][model_val][race_class_val][extracted_metric_val]))
            scores[exact_eval_set_idx][model_val]['overall'][extracted_metric_val] = np.average(tmp_metric, axis=0)

print('exp: ' + param['exp'])
print('Model: ')
print([i + ', ' + str(j) for i, j in zip(param['model'], param['epoch'])] )
print()

# Selected scores
selected_scores = {}
for exact_eval_set_val in exact_eval_set:
    selected_scores[exact_eval_set_val] = {}
    for model_idx, model_val in enumerate(param['model']):
        selected_scores[exact_eval_set_val][param['class'][model_idx]] = {}
        for extracted_metric_val in extracted_metric:
            selected_scores[exact_eval_set_val][param['class'][model_idx]][extracted_metric_val] = scores[exact_eval_set_val][model_val][param['class'][model_idx]][extracted_metric_val][0][param['epoch'][model_idx]]
# Bind into dataframe
tmp_metric = {}
for exact_eval_set_val in exact_eval_set:
    tmp_metric[exact_eval_set_val] = np.empty((0, len(race_class)))
    for extracted_metric_val in extracted_metric:
        tmp = np.empty(0)
        for race_class_val in race_class:
            tmp = np.append(tmp, selected_scores[exact_eval_set_val][race_class_val][extracted_metric_val])
        tmp_metric[exact_eval_set_val] = np.vstack((tmp_metric[exact_eval_set_val], tmp))
    selected_scores[exact_eval_set_val] = pd.DataFrame(tmp_metric[exact_eval_set_val], index=extracted_metric, columns=race_class)
    print()
    print(exact_eval_set_val)
    print(selected_scores[exact_eval_set_val])

# selected_scores = {}
# for exact_eval_set_val in exact_eval_set:
#     print()
#     print(exact_eval_set_val)
#     selected_scores[exact_eval_set_val] = {}
#     selected_scores[exact_eval_set_val][param['class']] = {}
#     for extracted_metric_val in extracted_metric:
#         selected_scores[exact_eval_set_val][param['class']][extracted_metric_val] = scores[exact_eval_set_val][param['class']][extracted_metric_val][0, param['epoch']]
#     print(selected_scores[exact_eval_set_val][param['class']])



print()
