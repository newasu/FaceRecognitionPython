# code testing

# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np

# Import my own lib
import others.utilities as my_util
from algorithms.selm import selm
from algorithms.welm import welm

#############################################################################################

# Experiment name
exp = 'exp_10'
exp_name = exp + '_alg_selmEuclidMeanPOS'
query_exp_name = exp + '_alg_selmEuclidMeanPOS'

train_class_idx = 0
train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
train_class = train_class[train_class_idx]
exp_name = exp_name + '_' + train_class
query_exp_name = query_exp_name + '_' + train_class

dataset_name = 'Diveface'
dataset_exacted = 'resnet50'
dataset_exacted_model = ['exp_7', 'eer']

# Parameter settings
num_used_cores = 2

# Whole run round settings
run_exp_round = [0] # define randomseed as list
test_size = 0.3

# k-fold for training
numb_train_kfold = 5
cv_run = -1 # -1 = run all fold, else, run only define

# Algorithm parameters
param_grid = {'distanceFunc':'euclidean', 
              'kernel_param':0, 
              'hiddenNodePerc':(np.arange(1, 11)/10), 
              'regC':10**np.arange(-10, 11, dtype='float')}
combine_rule = 'mean'

pos_class = 'POS'

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
# Result path
exp_result_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, exp_name])
# Grid search path
gridsearch_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name])
# Make directory
my_util.make_directory(exp_result_path)
my_util.make_directory(gridsearch_path)

#############################################################################################

# Run experiment
for exp_numb in run_exp_round:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Load data
    my_data = pd.read_csv((dataset_path + dataset_name + '_' + dataset_exacted + '_' + dataset_exacted_model[0] + '_run_' + str(exp_numb) + '(' + dataset_exacted_model[1] + ').txt'), sep=" ", header=0)
    # Separate data
    my_data_race = (my_data['gender'] + '-' + my_data['ethnicity']).values
    [training_sep_idx, test_sep_idx, _] = my_util.split_data_by_id_and_classes(my_data.id.values, my_data_race, test_size=test_size, valid_size=0, random_state=exp_numb)
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
    # Test data
    test_sep_idx = test_sep_idx[my_data_race[test_sep_idx] == train_class]
    x_test = my_data.iloc[test_sep_idx,8:].values
    y_race_test = my_data_race[test_sep_idx]
    y_class_test = my_data.id.iloc[test_sep_idx].values
    y_id_test = my_data.data_id.iloc[test_sep_idx].values
    del my_data, my_data_race

    # Run experiment each iterative
    if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):
        # Run experiment if not available

        # Initial model
        selm_model = selm()
        
        # Generate k-fold index for training
        [kfold_training_idx, kfold_test_idx] = my_util.split_kfold_by_classes(y_class_training, n_splits=numb_train_kfold, random_state=exp_numb)
        
        query_exp_name_seed = (query_exp_name + '_run_' + str(exp_numb))
        
        # Grid search
        [cv_results, avg_cv_results] = selm_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, x_training, y_class_training, y_id_training, param_grid, gridsearch_path, query_exp_name_seed, pos_class=pos_class, cv_run=cv_run, randomseed=exp_numb, useTF=False, combine_rule=combine_rule, num_cores=num_used_cores)

        if cv_run == -1:
            
            # Initial model
            welm_model = welm()
            
            # Best params
            avg_cv_results = avg_cv_results.sort_values(by='auc_pos', ascending=False)
            best_param = avg_cv_results.iloc[0]
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(y_id_training, y_class_training, randomseed=exp_numb)
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(x_training, y_id_training, triplet_paired_list, combine_rule=combine_rule)
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(y_id_test, y_class_test, randomseed=exp_numb)
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(x_test, y_id_test, triplet_paired_list, combine_rule=combine_rule)

            # Train model with best params
            [weights, weightID, beta, label_classes, training_time] = welm_model.train(
            combined_training_xx, combined_training_yy, 
            trainingDataID=combined_training_id, 
            distanceFunc=best_param.distanceFunc, 
            kernel_param=best_param.kernel_param,
            hiddenNodePerc=best_param.hiddenNodePerc, 
            regC=best_param.regC, 
            randomseed=exp_numb,
            useTF=False)

            # Test model
            [predictedScores, predictedY, test_time] = welm_model.predict(combined_test_xx, weights, beta, best_param.distanceFunc, best_param.kernel_param, label_classes, useTF=False)

            # Eval performance
            pos_class_idx = label_classes == pos_class
            # Performance metrics
            performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, label_classes)
            # Biometric metrics
            performance_metric.update(my_util.biometric_metric(combined_test_yy, np.ravel(predictedScores[:,pos_class_idx]), pos_class, score_order='descending'))

            # Save score
            exp_result = {'distanceFunc':best_param.distanceFunc, 'kernel_param':best_param.kernel_param, 'hiddenNodePerc': best_param.hiddenNodePerc, 'regC':best_param.regC, 'combine_rule':combine_rule, 'randomseed': exp_numb, 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'selm', 'experiment_name': exp_name, 'trueY':combined_test_yy, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
            exp_result.update(performance_metric)
            my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

            print('Finished ' + exp_name_seed)
            
            print(exp_result)

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
