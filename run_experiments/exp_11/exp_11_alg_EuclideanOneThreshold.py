# code testing

# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

# Experiment name
exp = 'exp_11'
exp_name = exp + '_alg_GenderEuclidean' # _alg_BaselineEuclidean  _alg_GenderEuclidean
query_exp_name = exp_name
exp_name = exp_name + 'OneThreshold'

train_class_idx = [3, 4, 5] # [0, 1, 2, 3, 4, 5] [0, 1, 2] [3, 4, 5]
train_class = np.array(['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian'])
train_class = list(train_class[train_class_idx])
train_class_name = '-'
query_exp_name = query_exp_name + '_' + train_class_name.join(train_class)

dataset_name = 'Diveface'
dataset_exacted = 'resnet50'

# Whole run round settings
run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # define randomseed as list
test_size = 0.3
valid_size = 0.1

# k-fold for training
numb_train_kfold = 1
cv_run = -1 # -1 = run all fold, else, run only define

# Algorithm parameters
# param_grid = {'distanceFunc':'euclidean'}
# combine_rule = 'concatenate'

pos_class = 'POS'

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
exp_query_path = my_util.get_path(additional_path=['.', '.', 'mount','FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, query_exp_name])

#############################################################################################

# Run experiment
for exp_numb in run_exp_round:
    # Load data
    if 'BaselineEuclidean' in exp_name:
        my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm' + '.txt'), sep=" ", header=0)
    else:
        dataset_exacted_model = ''
        if 'GenderEuclidean' in exp_name:
            dataset_exacted_model = ['exp_8', 'eer'] # exp_7 for race, exp_8 for gender
        else:
            dataset_exacted_model = ['exp_7', 'eer'] # exp_7 for race, exp_8 for gender
        my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_' + dataset_exacted_model[0] + '_run_' + str(0) + '(' + dataset_exacted_model[1] + ').txt'), sep=" ", header=0)
    # Separate data
    my_data_race = (my_data['gender'] + '-' + my_data['ethnicity']).values
    [training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, my_data_race, test_size=test_size, valid_size=valid_size, random_state=exp_numb)
    
    # Assign idx
    tmp_training_sep_idx = np.empty(0)
    tmp_valid_sep_idx = np.empty(0)
    tmp_test_sep_idx = np.empty(0)
    for train_class_idx in train_class:
        tmp_training_sep_idx = np.append(tmp_training_sep_idx, training_sep_idx[my_data_race[training_sep_idx] == train_class_idx])
        tmp_valid_sep_idx = np.append(tmp_valid_sep_idx, valid_sep_idx[my_data_race[valid_sep_idx] == train_class_idx])
        tmp_test_sep_idx = np.append(tmp_test_sep_idx, test_sep_idx[my_data_race[test_sep_idx] == train_class_idx])
    
    # Train data
    training_sep_idx = tmp_training_sep_idx.astype(int)
    x_training = my_data.iloc[training_sep_idx,8:].values
    y_race_training = my_data_race[training_sep_idx]
    y_class_training = my_data.id.iloc[training_sep_idx].values
    y_id_training = my_data.data_id.iloc[training_sep_idx].values
    # Valid data
    valid_sep_idx = tmp_valid_sep_idx.astype(int)
    x_valid = my_data.iloc[valid_sep_idx,8:].values
    y_race_valid = my_data_race[valid_sep_idx]
    y_class_valid = my_data.id.iloc[valid_sep_idx].values
    y_id_valid = my_data.data_id.iloc[valid_sep_idx].values
    # Test data
    test_sep_idx = tmp_test_sep_idx.astype(int)
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
    
    # Initial model
    distance_model = paired_distance_alg()
    
    # Load model
    model = my_util.load_numpy_file(exp_query_path + query_exp_name + '_run_' + str(exp_numb) + '.npy')
    best_param = model['kernel_param']
    
    # Construct triplet training dataset
    triplet_paired_list = my_util.triplet_loss_paring(y_id_training, y_class_training, randomseed=exp_numb)
    [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(x_training, y_id_training, triplet_paired_list, combine_rule=model['combine_rule'])
    
    label_classes = np.unique(combined_training_yy)
    
    for race_idx in train_class:
        save_exp_name = exp_name + '_' + race_idx
        # Result path
        exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount','FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, save_exp_name])
        # Make directory
        my_util.make_directory(exp_result_path)
        # Experiment name each seed
        exp_name_seed = save_exp_name + '_run_' + str(exp_numb)
        
        sep_idx = y_race_test == race_idx
        # Construct triplet test dataset
        triplet_paired_list = my_util.triplet_loss_paring(y_id_test[sep_idx], y_class_test[sep_idx], randomseed=exp_numb)
        [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(x_test[sep_idx,:], y_id_test[sep_idx], triplet_paired_list, combine_rule=model['combine_rule'])
        
        predictedScores, predictedY, _ = distance_model.predict(combined_test_xx[:,0:feature_size], combined_test_xx[:,feature_size:], combined_test_yy, unique_class, model['kernel_param'], distance_metric=model['distanceFunc'])
        
        # Eval performance
        # Performance metrics
        performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, np.array(['NEG', 'POS']))
        # Biometric metrics
        performance_metric.update(my_util.biometric_metric(combined_test_yy, np.ravel(predictedScores), pos_class, score_order='ascending', threshold_step=0.01))

        # Save score
        exp_result = {'distanceFunc':model['distanceFunc'], 'kernel_param':model['kernel_param'], 'combine_rule':model['combine_rule'], 'randomseed': exp_numb, 'label_classes': label_classes, 'algorithm':model['algorithm'], 'experiment_name': save_exp_name, 'trueY':combined_test_yy, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
        exp_result.update(performance_metric)
        my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

        print('Finished ' + exp_name_seed)

        del performance_metric
        del predictedScores, predictedY
        del combined_test_xx, combined_test_yy, combined_test_id
        del exp_result
    
    del best_param, label_classes
    del combined_training_xx, combined_training_yy, combined_training_id

print()
