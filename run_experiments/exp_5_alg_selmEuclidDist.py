# code testing

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
from algorithms.selm import selm
from algorithms.welm import welm

#############################################################################################

# Experiment name
exp_name = ('exp_5_alg_selmEuclidDist')
query_exp_name = ('exp_5_alg_selmEuclidDistPOS')
training_useTF = False
test_useTF = False

# Parameter settings
# Whole run round settings
run_exp_kfold = [0, 1, 2, 3, 4] # define randomseed as list
numb_exp_kfold = 5

# k-fold for training
numb_train_kfold = 5
cv_run = -1 # -1 = run all fold, else, run only define

# Algorithm parameters
param_grid = {'distanceFunc':'euclidean', 
              'kernel_param':0, 
              'hiddenNodePerc':(np.arange(1, 11)/10), 
              'regC':10**np.arange(-6, 7, dtype='float')}
combine_rule = 'distance'

pos_class = 'POS'

# Path
# Dataset path
dataset_name = 'Diveface'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
dataset_path = dataset_path + 'Diveface_retinaface.txt'
# Result path
exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name])
# Grid search path
gridsearch_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'gridsearch'])

#############################################################################################

use_data_bet = [0, 100000]

# Read data
yy = pd.read_csv(dataset_path, sep=" ", header=0).id.values

# # Select only some classes
yy = yy[np.where(np.logical_and(yy>=use_data_bet[0], yy<=use_data_bet[1]))]

# Split training and test set
[exp_test_sep_idx, exp_training_sep_idx] = my_util.split_kfold_by_classes(yy, n_splits=numb_exp_kfold, random_state=0)
del yy

# Run experiment
for exp_numb in run_exp_kfold:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Read on training data for each fold
    my_data = pd.read_csv(dataset_path, sep=" ", header=0)
    my_data = my_data[my_data['id'].between(use_data_bet[0], use_data_bet[1])]
    my_data = my_data.iloc[exp_training_sep_idx[exp_numb]]
    xx = my_data.iloc[:,8:].values
    yy = my_data.id.values
    image_id = my_data.data_id.values.astype(str)
    del my_data

    # Run experiment each iterative
    if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):
        # Run experiment if not available

        # Initial model
        selm_model = selm()
        
        # Generate k-fold index for training
        [kfold_training_idx, kfold_test_idx] = my_util.split_kfold_by_classes(yy, n_splits=numb_train_kfold, random_state=exp_numb)
        
        query_exp_name_seed = (query_exp_name + '_run_' + str(exp_numb))
        
        # Grid search
        [cv_results, avg_cv_results] = selm_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, xx, yy, image_id, param_grid, gridsearch_path, query_exp_name_seed, pos_class=pos_class, cv_run=cv_run, randomseed=exp_numb, useTF=training_useTF, combine_rule=combine_rule, num_cores=6)

        if cv_run == -1:
            # Clear and reload dataset
            del xx, yy, image_id
            my_data = pd.read_csv(dataset_path, sep=" ", header=0)
            my_data = my_data[my_data['id'].between(use_data_bet[0], use_data_bet[1])]
            xx = my_data.iloc[:,8:].values
            yy = my_data.id.values
            image_id = my_data.data_id.values.astype(str)
            del my_data
            
            # Initial model
            welm_model = welm()
            
            # Best params
            # avg_cv_results = avg_cv_results.sort_values(by='auc_pos', ascending=False)
            best_param = avg_cv_results.iloc[0]
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[exp_training_sep_idx[exp_numb]], yy[exp_training_sep_idx[exp_numb]], randomseed=exp_numb)
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(xx[exp_training_sep_idx[exp_numb]], image_id[exp_training_sep_idx[exp_numb]], triplet_paired_list, combine_rule=combine_rule)
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[exp_test_sep_idx[exp_numb]], yy[exp_test_sep_idx[exp_numb]], randomseed=exp_numb)
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(xx[exp_test_sep_idx[exp_numb]], image_id[exp_test_sep_idx[exp_numb]], triplet_paired_list, combine_rule=combine_rule)

            # Train model with best params
            [weights, weightID, beta, label_classes, training_time] = welm_model.train(
            combined_training_xx, combined_training_yy, 
            trainingDataID=combined_training_id, 
            distanceFunc=best_param.distanceFunc, 
            kernel_param=best_param.kernel_param,
            hiddenNodePerc=best_param.hiddenNodePerc, 
            regC=best_param.regC, 
            randomseed=exp_numb,
            useTF=test_useTF)

            # Test model
            [predictedScores, predictedY, test_time] = welm_model.predict(combined_test_xx, weights, beta, best_param.distanceFunc, best_param.kernel_param, label_classes, useTF=test_useTF)

            # Eval performance
            pos_class_idx = label_classes == pos_class
            # Performance metrics
            performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, label_classes)
            # Biometric metrics
            performance_metric.update(my_util.biometric_metric(combined_test_yy, predictedScores[:,pos_class_idx], pos_class, score_order='descending'))

            # Save score
            exp_result = {'distanceFunc':best_param.distanceFunc, 'kernel_param':best_param.kernel_param, 'hiddenNodePerc': best_param.hiddenNodePerc, 'regC':best_param.regC, 'combine_rule':combine_rule, 'randomseed': exp_numb, 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'selm', 'experiment_name': exp_name, 'trueY':combined_test_yy, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
            exp_result.update(performance_metric)
            my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

            print('Finished ' + exp_name_seed)

            del weights, weightID, beta, label_classes, training_time
            del predictedScores, predictedY, test_time
            del performance_metric, exp_result
            del combined_training_xx, combined_training_yy, combined_training_id
            del combined_test_xx, combined_test_yy, combined_test_id

        del kfold_training_idx, kfold_test_idx
        del cv_results, avg_cv_results

    else:
        print('The experiment ' + str(exp_numb) + ' already existed.')

print()
