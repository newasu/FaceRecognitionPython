
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

exact_list = ['distanceFunc', 'hiddenNodePerc', 'regC', 'combine_rule', 'randomseed', 'label_classes', 'training_time', 'test_time', 'dataset_name', 'accuracy', 'auc', 'f1score', 'f1score_mean', 'eer', 'fmr_0d1', 'fmr_0d01']

exp_result_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/exp_result/'

euclid = my_util.exact_run_result_in_directory((exp_result_path + 'test_1_alg_euclid/'), exact_list)
selm_sum = my_util.exact_run_result_in_directory((exp_result_path + 'test_1_alg_selm_cr_sum/'), exact_list)
selm_dist = my_util.exact_run_result_in_directory((exp_result_path + 'exp_1_alg_selmDist/'), exact_list)

exacted_data = [euclid, selm_sum, selm_dist]
result_names = ['euclid', 'selm_sum', 'selm_dist']

[retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat] = my_util.term_retrieve_exact_result(exacted_data, result_names, 'eer', order_metric='descending')

print()