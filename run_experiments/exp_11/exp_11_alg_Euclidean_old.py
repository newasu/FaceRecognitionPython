# code testing

# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

# Experiment name
exp = 'exp_11'
exp_name = exp + '_alg_RaceEuclidean' # _alg_BaselineEuclidean _alg_RaceEuclidean _alg_GenderEuclidean
query_exp_name = exp_name

train_class_idx = 0
train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
train_class = train_class[train_class_idx]
exp_name = exp_name + '_' + train_class
query_exp_name = query_exp_name + '_' + train_class

dataset_name = 'Diveface'
dataset_exacted = 'resnet50'
dataset_exacted_model = ['exp_7', 'eer'] # exp_7 exp_8

# Whole run round settings
run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # define randomseed as list
test_size = 0.3
valid_size = 0.1

# k-fold for training
numb_train_kfold = 1
cv_run = -1 # -1 = run all fold, else, run only define

# Algorithm parameters
param_grid = {'distanceFunc':'euclidean'}
combine_rule = 'concatenate'

pos_class = 'POS'

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
exp_result_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, exp_name])
# Make directory
my_util.make_directory(exp_result_path)

#############################################################################################

# Run experiment
for exp_numb in run_exp_round:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Load data
    if 'BaselineEuclidean' in exp_name:
        my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm' + '.txt'), sep=" ", header=0)
    else:
        my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_' + dataset_exacted_model[0] + '_run_' + str(0) + '(' + dataset_exacted_model[1] + ').txt'), sep=" ", header=0)
    # Separate data
    my_data_race = (my_data['gender'] + '-' + my_data['ethnicity']).values
    [training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, my_data_race, test_size=test_size, valid_size=valid_size, random_state=exp_numb)
    # [tmp_training_sep_idx, _, _] = my_util.split_data_by_id_and_classes(my_data.id.values[training_sep_idx], my_data_race[training_sep_idx], test_size=0.9, valid_size=0, random_state=exp_numb)
    # training_sep_idx = training_sep_idx[tmp_training_sep_idx]
    # del tmp_training_sep_idx
    # Assign data
    # Training data
    training_sep_idx = training_sep_idx[my_data_race[training_sep_idx] == train_class]
    x_training = my_data.iloc[training_sep_idx,8:].values
    y_race_training = my_data_race[training_sep_idx]
    y_class_training = my_data.id.iloc[training_sep_idx].values
    y_id_training = my_data.data_id.iloc[training_sep_idx].values
    # Valid data
    valid_sep_idx = valid_sep_idx[my_data_race[valid_sep_idx] == train_class]
    x_valid = my_data.iloc[valid_sep_idx,8:].values
    y_race_valid = my_data_race[valid_sep_idx]
    y_class_valid = my_data.id.iloc[valid_sep_idx].values
    y_id_valid = my_data.data_id.iloc[valid_sep_idx].values
    # Test data
    test_sep_idx = test_sep_idx[my_data_race[test_sep_idx] == train_class]
    x_test = my_data.iloc[test_sep_idx,8:].values
    y_race_test = my_data_race[test_sep_idx]
    y_class_test = my_data.id.iloc[test_sep_idx].values
    y_id_test = my_data.data_id.iloc[test_sep_idx].values
    del my_data, my_data_race
    
    if 'BaselineEuclidean' in exp_name:
        x_training = preprocessing.normalize(x_training, norm='l2', axis=1, copy=True, return_norm=False)
        x_valid = preprocessing.normalize(x_valid, norm='l2', axis=1, copy=True, return_norm=False)
        x_test = preprocessing.normalize(x_test, norm='l2', axis=1, copy=True, return_norm=False)
    
    feature_size = x_training.shape[1]
    unique_class = {'pos':'POS', 'neg':'NEG'}

    # Run experiment each iterative
    if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):
        # Run experiment if not available

        # Initial model
        distance_model = paired_distance_alg()
        
        # Generate k-fold index for training
        # kfold_training_idx = [np.arange(0, training_sep_idx.size)]
        # kfold_test_idx = [np.arange(0, valid_sep_idx.size) + training_sep_idx.size]
        [_, tmp_kfold_training_idx] = my_util.split_kfold_by_classes(y_class_training, n_splits=10, random_state=exp_numb)
        [_, tmp_kfold_test_idx] = my_util.split_kfold_by_classes(y_class_valid, n_splits=10, random_state=exp_numb)
        kfold_training_idx = np.empty(0).astype(int)
        kfold_test_idx = np.empty(0).astype(int)
        for idx in range(0,len(tmp_kfold_training_idx)):
            kfold_training_idx = np.append(kfold_training_idx, tmp_kfold_training_idx[idx])
            kfold_test_idx = np.append(kfold_test_idx, tmp_kfold_test_idx[idx] + training_sep_idx.size)
        del tmp_kfold_training_idx, tmp_kfold_test_idx
        kfold_training_idx = [kfold_training_idx]
        kfold_test_idx = [kfold_test_idx]
        
        query_exp_name_seed = (query_exp_name + '_run_' + str(exp_numb))
        
        cv_results, avg_cv_results = distance_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, np.vstack((x_training, x_valid)), np.append(y_class_training, y_class_valid), np.append(y_id_training, y_id_valid), param_grid, exp_name, cv_run=cv_run, randomseed=exp_numb)
        
        best_param = avg_cv_results.iloc[0]

        # Construct triplet training dataset
        triplet_paired_list = my_util.triplet_loss_paring(y_id_training, y_class_training, randomseed=exp_numb)
        [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(x_training, y_id_training, triplet_paired_list, combine_rule=combine_rule)
        
        # Construct triplet test dataset
        triplet_paired_list = my_util.triplet_loss_paring(y_id_test, y_class_test, randomseed=exp_numb)
        [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(x_test, y_id_test, triplet_paired_list, combine_rule=combine_rule)
        
        label_classes = np.unique(combined_training_yy)
        
        predictedScores, predictedY, _ = distance_model.predict(combined_test_xx[:,0:feature_size], combined_test_xx[:,feature_size:], combined_test_yy, unique_class, best_param.classifier_threshold, distance_metric=param_grid['distanceFunc'])
        
        # Eval performance
        # Performance metrics
        performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, np.array(['NEG', 'POS']))
        # Biometric metrics
        performance_metric.update(my_util.biometric_metric(combined_test_yy, np.ravel(predictedScores), pos_class, score_order='ascending', threshold_step=0.01))

        # Save score
        exp_result = {'distanceFunc':best_param.distanceFunc, 'kernel_param':best_param.classifier_threshold, 'combine_rule':combine_rule, 'randomseed': exp_numb, 'label_classes': label_classes, 'algorithm': 'euclidean', 'experiment_name': exp_name, 'trueY':combined_test_yy, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
        exp_result.update(performance_metric)
        my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

        print('Finished ' + exp_name_seed)

        del best_param, performance_metric, label_classes
        del predictedScores, predictedY
        del combined_training_xx, combined_training_yy, combined_training_id
        del combined_test_xx, combined_test_yy, combined_test_id
        del exp_result

        del kfold_training_idx, kfold_test_idx
        del cv_results, avg_cv_results

    else:
        print('The experiment ' + str(exp_numb) + ' already existed.')

print()
