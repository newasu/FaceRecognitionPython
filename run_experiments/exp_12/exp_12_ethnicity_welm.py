# code testing

# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np

# Import my own lib
import others.utilities as my_util
from algorithms.welm import welm

#############################################################################################

# Experiment name
exp = 'exp_12'
exp_name = exp + '_ethnicity_welm'

dataset_name = 'Diveface'
dataset_exacted = 'resnet50'

# Parameter settings
num_used_cores = 1

# Whole run round settings
run_exp_round = [0] # define randomseed as list
test_size = 0.3
valid_size = 0.1

# Algorithm parameters
param_grid = {'distanceFunc':'euclidean', 
              'kernel_param':0, 
              'hiddenNodePerc': (np.arange(1, 5.5)/10),
              'regC':10**np.arange(-6, 6, dtype='float')}
combine_rule = 'concatenate'

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, exp_name])
# Grid search path
gridsearch_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name])
# Make directory
my_util.make_directory(exp_result_path)
my_util.make_directory(gridsearch_path)

#############################################################################################

# Initial model
welm_model = welm()

# Run experiment
for exp_numb in run_exp_round:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Load data
    my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_nonorm.txt'), sep=" ", header=0)
    # Separate data
    my_data_race = (my_data['gender'] + '-' + my_data['ethnicity']).values
    [training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(my_data.id.values, my_data_race, test_size=test_size, valid_size=valid_size, random_state=exp_numb)
    
    # Assign data
    # Training data
    # training_sep_idx = training_sep_idx[my_data_race[training_sep_idx] == train_class]
    x_training = my_data.iloc[training_sep_idx,8:].values
    y_race_training = my_data_race[training_sep_idx]
    y_ethnicity_training = my_data['ethnicity'].values[training_sep_idx]
    # y_gender_training = my_data['gender'].values[training_sep_idx]
    # y_class_training = my_data.id.iloc[training_sep_idx].values
    y_id_training = my_data.data_id.iloc[training_sep_idx].values
    # Test data
    # test_sep_idx = test_sep_idx[my_data_race[test_sep_idx] == train_class]
    x_test = my_data.iloc[test_sep_idx,8:].values
    y_race_test = my_data_race[test_sep_idx]
    y_ethnicity_test = my_data['ethnicity'].values[test_sep_idx]
    # y_gender_test = my_data['gender'].values[test_sep_idx]
    # y_class_test = my_data.id.iloc[test_sep_idx].values
    y_id_test = my_data.data_id.iloc[test_sep_idx].values
    # Valid data
    # valid_sep_idx = valid_sep_idx[my_data_race[valid_sep_idx] == train_class]
    x_valid = my_data.iloc[valid_sep_idx,8:].values
    y_race_valid = my_data_race[valid_sep_idx]
    y_ethnicity_valid = my_data['ethnicity'].values[valid_sep_idx]
    # y_gender_valid = my_data['gender'].values[valid_sep_idx]
    # y_class_valid = my_data.id.iloc[valid_sep_idx].values
    y_id_valid = my_data.data_id.iloc[valid_sep_idx].values
    del my_data, my_data_race
    
    # Generate k-fold index for training
    [_, tmp_kfold_training_idx] = my_util.split_kfold_by_classes(y_ethnicity_training, n_splits=5, random_state=exp_numb)
    [_, tmp_kfold_test_idx] = my_util.split_kfold_by_classes(y_ethnicity_valid, n_splits=5, random_state=exp_numb)
    kfold_training_idx = np.empty(0).astype(int)
    kfold_test_idx = np.empty(0).astype(int)
    for idx in range(0,len(tmp_kfold_training_idx)):
        kfold_training_idx = np.append(kfold_training_idx, tmp_kfold_training_idx[idx])
        kfold_test_idx = np.append(kfold_test_idx, tmp_kfold_test_idx[idx] + training_sep_idx.size)
    del tmp_kfold_training_idx, tmp_kfold_test_idx
    kfold_training_idx = [kfold_training_idx]
    kfold_test_idx = [kfold_test_idx]
    
    exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Grid search
    [cv_results, avg_cv_results] = welm_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, np.vstack((x_training, x_valid)), np.append(y_ethnicity_training, y_ethnicity_valid), np.append(y_id_training, y_id_valid), param_grid, gridsearch_path, exp_name_seed, cv_run=-1, randomseed=exp_numb, useTF=False, num_cores=num_used_cores)
    
    # Best params
    avg_cv_results = avg_cv_results.sort_values(by='f1score', ascending=False)
    best_param = avg_cv_results.iloc[0]
    
    # Train model with best params
    [weights, weightID, beta, label_classes, training_time] = welm_model.train(
    x_training, y_ethnicity_training, 
    trainingDataID=y_id_training, 
    distanceFunc=best_param.distanceFunc, 
    kernel_param=best_param.kernel_param,
    hiddenNodePerc=best_param.hiddenNodePerc, 
    regC=best_param.regC, 
    randomseed=exp_numb,
    useTF=False)
    
    # Test model
    [predictedScores, predictedY, test_time] = welm_model.predict(x_test, weights, beta, best_param.distanceFunc, best_param.kernel_param, label_classes, useTF=False)
    
    # Eval performance
    # Performance metrics
    performance_metric = my_util.classification_performance_metric(y_ethnicity_test, predictedY, label_classes)
    
    # Save score
    exp_result = {'distanceFunc':best_param.distanceFunc, 'kernel_param':best_param.kernel_param, 'hiddenNodePerc': best_param.hiddenNodePerc, 'regC':best_param.regC, 'randomseed': exp_numb, 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'welm', 'experiment_name': exp_name_seed, 'trueY':y_ethnicity_training, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':y_id_training, 'dataset_name':dataset_name}
    exp_result.update(performance_metric)
    my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

print()
