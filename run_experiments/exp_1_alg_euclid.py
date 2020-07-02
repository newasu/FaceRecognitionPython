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
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

# Experiment name
exp_name = ('exp_1_alg_euclid')

# Parameter settings
# Whole run round settings
run_exp_kfold = [0, 1, 2, 3, 4] # define randomseed as list
numb_exp_kfold = 5

# k-fold for training
numb_train_kfold = 5
cv_run = -1 # -1 = run all fold, else, run only define

# Algorithm parameters
param_grid = {'distanceFunc':'euclidean'}

pos_class = 'POS'

# Path
# Dataset path
dataset_name = 'CelebA'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA_features'])
dataset_path = dataset_path + 'CelebA_retinaface.txt'
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
    xx = my_data.iloc[:,2:].values
    yy = my_data.id.values
    image_id = my_data.image_id.values
    del my_data

    if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):

        # Run experiment each iterative
        
        # from collections import Counter
        # data_class_freq = Counter(yy[train_index])
        # unique_class = np.array(list(data_class_freq.keys()))
        # data_class_freq = np.array(list(data_class_freq.values()))
        
        # Initial model
        distance_model = paired_distance_alg()
        
        # Generate k-fold index for training
        [kfold_training_idx, kfold_test_idx] = my_util.split_kfold_by_classes(yy, n_splits=numb_train_kfold, random_state=exp_numb)
        
        # Grid search
        [cv_results, avg_cv_results] = distance_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, xx, yy, image_id, param_grid, exp_name_seed, cv_run=cv_run, randomseed=exp_numb, threshold_decimal=2)

        if cv_run == -1:
            # Clear and reload dataset
            del xx, yy, image_id
            my_data = pd.read_csv(dataset_path, sep=" ", header=0)
            my_data = my_data[my_data['id'].between(use_data_bet[0], use_data_bet[1])]
            xx = my_data.iloc[:,2:].values
            yy = my_data.id.values
            image_id = my_data.image_id.values
            del my_data
            
            # Best params
            best_param = avg_cv_results.iloc[0]
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[exp_training_sep_idx[exp_numb]], yy[exp_training_sep_idx[exp_numb]], randomseed=exp_numb, num_cores=-1)
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(xx[exp_training_sep_idx[exp_numb]], image_id[exp_training_sep_idx[exp_numb]], triplet_paired_list, combine_rule='concatenate')
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[exp_test_sep_idx[exp_numb]], yy[exp_test_sep_idx[exp_numb]], randomseed=exp_numb, num_cores=-1)
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(xx[exp_test_sep_idx[exp_numb]], image_id[exp_test_sep_idx[exp_numb]], triplet_paired_list, combine_rule='concatenate')
            
            # Prepare variable
            sep_idx = int(combined_training_xx.shape[1]/2)
            unique_class = {'pos':pos_class, 'neg':'NEG'}

            # Train model with best params

            # Test model
            [predictedScores, predictedY, test_time] = distance_model.predict(combined_test_xx[:,0:sep_idx], combined_test_xx[:,sep_idx:], combined_test_yy, unique_class, best_param.classifier_threshold, distance_metric=best_param.distanceFunc)

            # Eval performance
            # Performance metrics
            performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, np.array(list(unique_class.values())))
            # Biometric metrics
            performance_metric.update(my_util.biometric_metric(combined_test_yy, predictedScores, pos_class, score_order='descending'))

            # Save score
            exp_result = {'distanceFunc':best_param.distanceFunc, 'randomseed': exp_numb, 'weightID': combined_training_id, 'label_classes': np.array(list(unique_class.values())), 'training_time': 0, 'test_time': test_time, 'algorithm': 'euclid', 'experiment_name': exp_name, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
            exp_result.update(performance_metric)
            my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

            print('Finished ' + exp_name_seed)

            del predictedScores, predictedY, test_time
            del performance_metric, exp_result

        del kfold_training_idx, kfold_test_idx
        del cv_results, avg_cv_results

    else:
        print('The experiment ' + str(exp_numb) + ' already existed.')

print()
