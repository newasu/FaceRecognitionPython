
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import lib
import pandas as pd
import numpy as np
from scipy.io import savemat

# Import my own lib
import others.utilities as my_util

#############################################################################################

file_name = 'exp_3_alg_selmEuclidDistPOS'

save_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/exp_result_mat/'
exp_result_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/exp_result/'
exact_list = ['distanceFunc', 'hiddenNodePerc', 'regC', 'combine_rule', 'randomseed', 'label_classes', 'training_time', 'test_time', 'dataset_name', 'accuracy', 'auc', 'f1score', 'f1score_mean', 'eer', 'fmr_0d1', 'fmr_0d01', 'fnmr_0d1', 'fnmr_0d01']
exDict = my_util.exact_run_result_in_directory((exp_result_path + file_name + '/'), exact_list)

savemat((save_path + file_name + '.mat'), exDict)