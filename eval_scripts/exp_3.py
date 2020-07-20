
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import lib
import pandas as pd
import numpy as np

# Import my own lib
import others.utilities as my_util

#############################################################################################

exact_list = ['distanceFunc', 'hiddenNodePerc', 'regC', 'combine_rule', 'randomseed', 'label_classes', 'training_time', 'test_time', 'dataset_name', 'accuracy', 'auc', 'f1score', 'f1score_mean', 'eer', 'fmr_0d1', 'fmr_0d01', 'fnmr_0d1', 'fnmr_0d01']

exp_result_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/exp_result/'

euclid = my_util.exact_run_result_in_directory((exp_result_path + 'exp_3_alg_euclid/'), exact_list)
# selm_euclid_dist = my_util.exact_run_result_in_directory((exp_result_path + 'exp_3_alg_selmEuclidDist/'), exact_list)
selm_euclid_dist_pos = my_util.exact_run_result_in_directory((exp_result_path + 'exp_3_alg_selmEuclidDistPOS/'), exact_list)
# selm_rbf_dist = my_util.exact_run_result_in_directory((exp_result_path + 'exp_3_alg_selmRBFDist/'), exact_list)
# selm_rbf_dist_pos = my_util.exact_run_result_in_directory((exp_result_path + 'exp_3_alg_selmRBFDistPOS/'), exact_list)

# exacted_data = [euclid, selm_euclid_dist, selm_euclid_dist_pos, selm_rbf_dist, selm_rbf_dist_pos]
# result_names = ['euclid', 'selm_euclid_dist', 'selm_euclid_dist_pos', 'selm_rbf_dist', 'selm_rbf_dist_pos']

exacted_data = [euclid, selm_euclid_dist_pos]
result_names = ['euclid', 'selm_euclid_dist_pos']

[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_classes_result_eval_retrieve(exacted_data, result_names, 'f1score', metric_ordering='ascending') # multiply 100 to be %
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'accuracy', metric_ordering='ascending') # multiply 100 to be %
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'f1score_mean', metric_ordering='ascending')
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'auc', metric_ordering='ascending') # multiply 100 to be %
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'eer', metric_ordering='descending') # multiply 100 to be %
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'fmr_0d1', metric_ordering='descending')
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'fmr_0d01', metric_ordering='descending')
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'fnmr_0d1', metric_ordering='ascending')
[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.exact_result_eval_retrieve(exacted_data, result_names, 'fnmr_0d01', metric_ordering='ascending')

print()