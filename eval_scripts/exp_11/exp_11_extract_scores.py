
# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Experiment name
exp = 'exp_11'
exp_name = exp + '_alg_GenderEuclideanOneThreshold' # _alg_BaselineEuclidean _alg_GenderEuclidean _alg_RaceEuclidean _alg_selmEuclidSumPOS _alg_selmEuclidDistPOS _alg_selmEuclidMultiplyPOS _alg_selmEuclidMeanPOS _alg_elmRBFPOS _alg_elmRBFPOS_2 _alg_BaselineEuclideanOneThreshold _alg_GenderEuclideanOneThreshold

class_model = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
metric = ['auc', 'eer', 'accuracy', 'tar_1', 'tar_0d1', 'tar_0d01']

run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#############################################################################################

# Path
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp])

#############################################################################################

# Extract scores
scores = {}
for run_exp_round_idx, run_exp_round_val in enumerate(run_exp_round):  
    for class_model_idx, class_model_val in enumerate(class_model):
        data_folder = exp_name + '_' + class_model_val
        data = exp_result_path + data_folder + os.sep + data_folder + '_run_' + str(run_exp_round_val) + '.npy'
        data = my_util.load_numpy_file(data)
        for metric_val in metric:
            if run_exp_round_idx == 0 and class_model_idx == 0:
                scores[metric_val] = np.zeros((len(run_exp_round), len(class_model)))
            scores[metric_val][run_exp_round_idx,class_model_idx] = data[metric_val]
            # print(metric_val + ': ' + str(scores[metric_val][run_exp_round_idx,class_model_idx]))

# Bind into dataframe
avg_scores = {}
std_scores = {}
for metric_val in metric:
    scores[metric_val][np.isnan(scores[metric_val])] = 0
    # Average
    avg_scores[metric_val] = pd.DataFrame(np.append(np.average(scores[metric_val], axis=0), np.average(scores[metric_val])))
    avg_scores[metric_val] = avg_scores[metric_val].T
    avg_scores[metric_val].columns = class_model + ['average']
    # Standard deviation
    std_scores[metric_val] = pd.DataFrame(np.append(np.std(scores[metric_val], axis=0), np.average(np.std(scores[metric_val], axis=0))))
    std_scores[metric_val] = std_scores[metric_val].T
    std_scores[metric_val].columns = class_model + ['average']
    # Raw
    scores[metric_val] = pd.DataFrame(data=scores[metric_val], columns=class_model)
    # Display
    if metric_val != 'eer':
        avg_scores[metric_val] = avg_scores[metric_val] * 100
        std_scores[metric_val] = std_scores[metric_val] * 100
    disp_score = [str(a_) + ' ± ' + str(b_) for a_, b_ in zip(np.round(avg_scores[metric_val].values[0],2), np.round(std_scores[metric_val].values[0],2))]
    disp_score = pd.DataFrame(disp_score).T
    disp_score.columns = class_model + ['average']
    print(metric_val)
    print(disp_score)
    print()
    
    # print('Average ' + metric_val)
    # print(avg_scores[metric_val])
    # print('Standard deviation ' + metric_val)
    # print(std_scores[metric_val])
    # print()



print()
