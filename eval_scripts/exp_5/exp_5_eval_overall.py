
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
from tqdm import tqdm

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Path
# Dataset path
dataset_name = 'Diveface'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
dataset_path = dataset_path + 'Diveface_retinaface.txt'

mat_save_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'average_exp_result_mat'])
exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result'])
average_exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'average_exp_result'])

# Load label
# my_data = pd.read_csv(dataset_path, sep=" ", header=0)
# my_label = pd.concat([my_data['gender'], (my_data['gender'] + '-' + my_data['ethnicity']), my_data['data_id']], axis=1)
# my_label.columns = ['gender', 'label', 'id']
# my_unique_labels = np.unique(my_label['label'])
# del my_data

#############################################################################################

save_folder = 'exp_5'
save_filename = 'exp_5_avg_overall'
filenames = ['exp_5_alg_euclid', 'exp_5_alg_selmEuclidDistPOS', 'exp_5_alg_selmEuclidMeanPOS', 'exp_5_alg_selmEuclidMultiplyPOS', 'exp_5_alg_selmEuclidSumPOS']
performance_sort_order = ['ascending', 'descending', 'descending', 'descending', 'descending', 'descending']

exp_round = ['0', '1', '2', '3', '4']
measurements = ['eer', 'fmr_1', 'fmr_0d1', 'fmr_0d01', 'fmr_0',]
measurements_order = ['ascending', 'ascending', 'ascending', 'ascending', 'ascending']
exact_list = ['test_image_id', 'trueY', 'predictedScores', 'label_classes']

scores_dict = {}
ranked_dict = {}
std_scores = {}
avg_scores = {}
std_scores = {}
sum_ranked = {}
algo_names = []

for filename_idx in tqdm(range(0, len(filenames))):
    algo_names.append(filenames[filename_idx].split('_')[3])
    data = my_util.exact_run_result_in_directory((exp_result_path + filenames[filename_idx] + '/'), exact_list)
    
    for fold_idx in range(0,data['trueY'].shape[0]):
        print('filename_idx: ' + str(filename_idx) + ', fold: ' + str(fold_idx))
        tmp_score = data['predictedScores'][fold_idx]
        if tmp_score.ndim > 1:
            tmp_score = tmp_score[:, ( data['label_classes'][fold_idx]=='POS')][:,0]
        eval_score = my_util.biometric_metric(data['trueY'][fold_idx], tmp_score, 'POS', score_order=performance_sort_order[filename_idx])
    
        for measurement_idx in measurements:
            # Initial array if first time
            if (measurement_idx in scores_dict) == False:
                scores_dict[measurement_idx] = {}
                scores_dict[measurement_idx]['overall'] = np.empty(0)
            scores_dict[measurement_idx]['overall'] = np.append(scores_dict[measurement_idx]['overall'], eval_score[measurement_idx])

# Sort and Rank data
for measurement_idx in range(0, len(measurements)):
    ranked_dict[measurements[measurement_idx]] = {}
    std_scores[measurements[measurement_idx]] = {}
    avg_scores[measurements[measurement_idx]] = {}
    sum_ranked[measurements[measurement_idx]] = {}
    # Sort
    scores_dict[measurements[measurement_idx]]['overall'] = np.reshape(scores_dict[measurements[measurement_idx]]['overall'], (-1, len(exp_round))).T
    # STD
    std_scores[measurements[measurement_idx]]['overall'] = np.std(scores_dict[measurements[measurement_idx]]['overall'], axis=0)
    # Rank
    if measurements_order[measurement_idx] == 'ascending':
        ranked_dict[measurements[measurement_idx]]['overall'] = rankdata(scores_dict[measurements[measurement_idx]]['overall'], method='average', axis=1)
    else:
        ranked_dict[measurements[measurement_idx]]['overall'] = rankdata(-scores_dict[measurements[measurement_idx]]['overall'], method='average', axis=1)
    # Average scores and sum ranked
    avg_scores[measurements[measurement_idx]]['overall'] = np.average(scores_dict[measurements[measurement_idx]]['overall'], axis=0)
    sum_ranked[measurements[measurement_idx]]['overall'] = np.sum(ranked_dict[measurements[measurement_idx]]['overall'], axis=0)
    # if (measurements[measurement_idx] in avg_scores) == False:
    #     avg_scores[measurements[measurement_idx]]= np.empty((0, len(filenames)))
    #     sum_ranked[measurements[measurement_idx]] = np.empty((0, len(filenames)))
    # avg_scores[measurements[measurement_idx]] = np.vstack((avg_scores[measurements[measurement_idx]], np.average(scores_dict[measurements[measurement_idx]]['overall'], axis=0)))
    # sum_ranked[measurements[measurement_idx]] = np.vstack((sum_ranked[measurements[measurement_idx]], np.sum(ranked_dict[measurements[measurement_idx]]['overall'], axis=0)))

# Save data as mat
data = {'algo_names':algo_names, 'data_classes':'overall', 'scores':scores_dict, 'ranked':ranked_dict, 'avg_scores':avg_scores, 'std_scores':std_scores, 'sum_ranked':sum_ranked}
my_util.save_numpy(data, (average_exp_result_path + save_folder), save_filename, doSilent=True)
savemat((mat_save_path + save_folder + '/' + save_filename + '.mat'), data)

print()
print()
print()


