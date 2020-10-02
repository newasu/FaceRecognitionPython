
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
exp_name = exp + '_alg_elmRBFPOS' # _alg_BaselineEuclidean _alg_GenderEuclidean _alg_RaceEuclidean _alg_selmEuclidSumPOS _alg_selmEuclidDistPOS _alg_selmEuclidMultiplyPOS _alg_selmEuclidMeanPOS _alg_elmRBFPOS

class_model = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#############################################################################################

# Path
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp])

#############################################################################################

# Extract scores
scores = {}
for run_exp_round_idx, run_exp_round_val in tqdm(enumerate(run_exp_round)):  
    for class_model_idx, class_model_val in enumerate(class_model):
        data_folder = exp_name + '_' + class_model_val
        data_name = data_folder + '_run_' + str(run_exp_round_val)
        data = exp_result_path + data_folder + os.sep + data_name + '.npy'
        data = my_util.load_numpy_file(data)
        
        # Eval performance
        pos_class_idx = data['label_classes'] == 'POS'
        # Performance metrics
        performance_metric = my_util.classification_performance_metric(data['trueY'], data['predictedY'], data['label_classes'])
        # Biometric metrics
        if data['predictedScores'].ndim == 1:
            performance_metric.update(my_util.biometric_metric(data['trueY'], np.ravel(data['predictedScores']), 'POS', score_order='ascending'))
        else:
            performance_metric.update(my_util.biometric_metric(data['trueY'], np.ravel(data['predictedScores'][:,pos_class_idx]), 'POS', score_order='descending'))
            
        # Save score
        data.update(performance_metric)
        my_util.save_numpy(data, exp_result_path + data_folder, data_name)