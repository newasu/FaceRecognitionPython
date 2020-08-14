
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent.parent
sys.path.insert(0, str(my_root_path))

# Import lib
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from scipy.io import savemat

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Run exp_5_avg_classes.py

save_folder = 'exp_5'
load_folder = 'exp_5_avg_classes'
filenames = ['exp_5_alg_euclid_avg_classes', 'exp_5_alg_selmEuclidDistPOS_avg_classes', 'exp_5_alg_selmEuclidMeanPOS_avg_classes', 'exp_5_alg_selmEuclidMultiplyPOS_avg_classes', 'exp_5_alg_selmEuclidSumPOS_avg_classes']
# filenames = ['exp_5_alg_euclid_avg_classes', 'exp_5_alg_selmEuclidDistPOS_avg_classes', 'exp_5_alg_selmEuclidMeanPOS_avg_classes', 'exp_5_alg_selmEuclidMultiplyPOS_avg_classes', 'exp_5_alg_selmEuclidSumPOS_avg_classes', 'exp_5_alg_selmRBFDistPOS_avg_classes']

# Initial variables
exp_round = ['0', '1', '2', '3', '4']
data_classes = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
measurements = ['eer', 'fmr_1', 'fmr_0d1', 'fmr_0d01', 'fmr_0']
measurements_order = ['ascending', 'ascending', 'ascending', 'ascending', 'ascending']
mat_save_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'average_exp_result_mat'])
average_exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'average_exp_result'])

scores_dict = {}
ranked_dict = {}
avg_scores = {}
std_scores = {}
sum_ranked = {}
algo_names = []

# Exact data
for fn_idx in filenames:
    algo_names.append(fn_idx.split('_')[3])
    data = my_util.load_numpy_file((average_exp_result_path + load_folder + '/' + fn_idx + '.npy'))
    
    for exp_round_idx in exp_round:
        for ethnicity_idx in data_classes:
            for measurement_idx in measurements:
                # Initial array if first time
                if (measurement_idx in scores_dict) == False:
                    scores_dict[measurement_idx] = {}
                if (ethnicity_idx in scores_dict[measurement_idx]) == False:
                    scores_dict[measurement_idx][ethnicity_idx] = np.empty(0)
                scores_dict[measurement_idx][ethnicity_idx] = np.append(scores_dict[measurement_idx][ethnicity_idx], data[exp_round_idx][ethnicity_idx][measurement_idx])

# Sort and Rank data
for measurement_idx in range(0, len(measurements)):
    ranked_dict[measurements[measurement_idx]] = {}
    std_scores[measurements[measurement_idx]] = {}
    avg_scores[measurements[measurement_idx]] = {}
    sum_ranked[measurements[measurement_idx]] = {}
    for ethnicity_idx in data_classes:
        # Sort
        scores_dict[measurements[measurement_idx]][ethnicity_idx] = np.reshape(scores_dict[measurements[measurement_idx]][ethnicity_idx], (-1, len(exp_round))).T
        # STD
        std_scores[measurements[measurement_idx]][ethnicity_idx] = np.std(scores_dict[measurements[measurement_idx]][ethnicity_idx], axis=0)
        # Rank
        if measurements_order[measurement_idx] == 'ascending':
            ranked_dict[measurements[measurement_idx]][ethnicity_idx] = rankdata(scores_dict[measurements[measurement_idx]][ethnicity_idx], method='average', axis=1)
        else:
            ranked_dict[measurements[measurement_idx]][ethnicity_idx] = rankdata(-scores_dict[measurements[measurement_idx]][ethnicity_idx], method='average', axis=1)
        # Average scores and sum ranked
        avg_scores[measurements[measurement_idx]][ethnicity_idx] = np.average(scores_dict[measurements[measurement_idx]][ethnicity_idx], axis=0)
        sum_ranked[measurements[measurement_idx]][ethnicity_idx] = np.sum(ranked_dict[measurements[measurement_idx]][ethnicity_idx], axis=0)
        # if (measurements[measurement_idx] in avg_scores) == False:
        #     avg_scores[measurements[measurement_idx]] = np.empty((0, len(filenames)))
        #     sum_ranked[measurements[measurement_idx]] = np.empty((0, len(filenames)))
        # avg_scores[measurements[measurement_idx]] = np.vstack((avg_scores[measurements[measurement_idx]], np.average(scores_dict[measurements[measurement_idx]][ethnicity_idx], axis=0)))
        # sum_ranked[measurements[measurement_idx]] = np.vstack((sum_ranked[measurements[measurement_idx]], np.sum(ranked_dict[measurements[measurement_idx]][ethnicity_idx], axis=0)))

# Save data as mat
data = {'algo_names':algo_names, 'data_classes':data_classes, 'scores':scores_dict, 'ranked':ranked_dict, 'avg_scores':avg_scores, 'std_scores':std_scores, 'sum_ranked':sum_ranked}
my_util.save_numpy(data, (average_exp_result_path + save_folder), load_folder, doSilent=True)
savemat((mat_save_path + save_folder + '/' + load_folder + '.mat'), data)

print()
print()
print()