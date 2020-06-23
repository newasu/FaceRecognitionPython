import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
# from collections import Counter
# import itertools

import multiprocessing
from joblib import Parallel, delayed
# from tqdm import tqdm

# Import my lib
import others.utilities as my_util
from algorithm.welm import welm

welm_model = None

class selm(object):

    def __init__(self):
        global welm_model
        welm_model = welm()
        pass

    def train(self, trainingDataX, trainingDataY, **kwargs):
        return []

    def predict(self, testDataX, weights, beta, distanceFunc, label_classes, useTF=False):
        return []
    
    def grid_search_cv_parallel(self, kfold_training_idx, kfold_test_idx, trainingDataX, trainingDataY, trainingDataID, param_grid, exp_path, exp_name, **kwargs):
        
        # Assign params
        modelParams = {}
        modelParams['num_cores']  = multiprocessing.cpu_count()
        modelParams['useTF'] = False
        modelParams['combine_rule'] = 'sum'
        modelParams['cv_run'] = -1 # -1 = run all seed, else, run only defined seed
        modelParams['randomseed'] = 0
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))
        del key, value
        
        # Run only desire number in cv
        if modelParams['cv_run'] == -1:
            cv_run = list(range(0, len(kfold_training_idx)))
        else:
            if isinstance(modelParams['cv_run'], list):
                cv_run = modelParams['cv_run']
            else:
                cv_run = [modelParams['cv_run']]

        # Get save directory
        my_save_directory = exp_path + exp_name

        # Combine params all of possible combination
        param_list = np.array(np.meshgrid(param_grid['distanceFunc'], param_grid['hiddenNodePerc'], param_grid['regC'])).T
        param_list = param_list.reshape(-1, param_list.shape[-1])

        # Init result
        cv_results = pd.DataFrame(columns=['fold', 'distanceFunc', 'hiddenNodePerc', 'regC', 'auc', 'f1score']) # define column names
        
        # Run k-fold
        total_run = str(param_list.shape[0])
        for kfold_idx in cv_run:
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(trainingDataID[kfold_training_idx[kfold_idx]], trainingDataY[kfold_training_idx[kfold_idx]], randomseed=modelParams['randomseed'])
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(trainingDataX[kfold_training_idx[kfold_idx]], trainingDataID[kfold_training_idx[kfold_idx]], triplet_paired_list, combine_rule=modelParams['combine_rule'])
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(trainingDataID[kfold_test_idx[kfold_idx]], trainingDataY[kfold_test_idx[kfold_idx]], randomseed=modelParams['randomseed'])
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(trainingDataX[kfold_test_idx[kfold_idx]], trainingDataID[kfold_test_idx[kfold_idx]], triplet_paired_list, combine_rule=modelParams['combine_rule'])
            
            tmp_training_idx = np.arange(0, combined_training_yy.size)
            tmp_test_idx = np.arange(combined_training_yy.size, (combined_training_yy.size+combined_test_yy.size))

            other_param = {'kfold_idx':kfold_idx, 'randomseed':modelParams['randomseed'], 'exp_path':exp_path,'exp_name':exp_name, 'kfold_training_data_idx':tmp_training_idx, 'kfold_test_data_idx':tmp_test_idx, 'my_save_directory':my_save_directory, 'total_run':total_run, 'useTF':modelParams['useTF']}
            
            # Run grid search parallel
            tmp_cv_results = Parallel(n_jobs=modelParams['num_cores'])(delayed(welm_model.do_gridsearch_parallel)(gs_idx, np.concatenate((combined_training_xx, combined_test_xx)), np.concatenate((combined_training_yy, combined_test_yy)), np.concatenate((combined_training_id, combined_test_id)), param_list[gs_idx], other_param) for gs_idx in range(0, param_list.shape[0]))
            cv_results = cv_results.append(pd.DataFrame(tmp_cv_results), ignore_index=True)
            
            # # Run grid search by for loop
            # for gs_idx in range(0, param_list.shape[0]):
            #     tmp_cv_results = welm_model.do_gridsearch_parallel(gs_idx, np.concatenate((combined_training_xx, combined_test_xx)), np.concatenate((combined_training_yy, combined_test_yy)), np.concatenate((combined_training_id, combined_test_id)), param_list[gs_idx], other_param)
            #     cv_results = cv_results.append([tmp_cv_results], ignore_index=True)
            
            del tmp_cv_results
                
        # Average cv_results
        [cv_results, avg_cv_results] = my_util.average_gridsearch(cv_results, ['distanceFunc', 'hiddenNodePerc', 'regC'])

        return cv_results, avg_cv_results