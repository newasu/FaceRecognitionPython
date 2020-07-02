# code testing

# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import libraries
import pandas as pd
import numpy as np

# Import my own lib
import others.utilities as my_util
from algorithms.paired_distance_alg import paired_distance_alg

#############################################################################################

# Experiment name
exp_name = ('test_1_alg_euclid')

# Parameter settings
numb_exp = [0, 1, 2, 3, 4] # define randomseed as list
cv_numb = 5
cv_run = -1 # -1 = run all seed, else, run only define
param_grid = {'distanceFunc':'euclidean'}

test_size = 0.3

# Path
# Dataset path
dataset_name = 'CelebA'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA_features'])
dataset_path = dataset_path + 'CelebA_retinaface.txt'
# Result path
exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name])

#############################################################################################

# Prepare save path
my_util.make_directory(exp_result_path)

# Read data
my_data = pd.read_csv(dataset_path, sep=" ", header=0)

# Select only some classes
my_data = my_data[my_data['id'].between(1, 1000)]

# Assign xx, yy and data id
xx = my_data.iloc[:,2:].values
yy = my_data.id.values
image_id = my_data.image_id.values

del dataset_path, my_data

# Run experiment
for exp_numb in numb_exp:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_seed_' + str(exp_numb))

    if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):
        # Get index for partition training/test set
        [train_index, test_index] = my_util.split_data_by_classes(yy, test_size=test_size, random_state=exp_numb)
        
        # Initial model
        distance_model = paired_distance_alg()

        # Run experiment each iterative
        # Generate k-fold index
        [kfold_training_idx, kfold_test_idx] = my_util.split_kfold_by_classes(yy[train_index], n_splits=cv_numb, random_state=exp_numb)
        
        # Grid search        
        [cv_results, avg_cv_results] = distance_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, xx[train_index], yy[train_index], image_id[train_index], param_grid, exp_name_seed, cv_run=cv_run, randomseed=exp_numb, threshold_decimal=2)

        if cv_run == -1:
            
            # Best params
            best_param = avg_cv_results.iloc[0]
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[train_index], yy[train_index], randomseed=exp_numb)
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(xx[train_index], image_id[train_index], triplet_paired_list, combine_rule='concatenate')
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[test_index], yy[test_index], randomseed=exp_numb)
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(xx[test_index], image_id[test_index], triplet_paired_list, combine_rule='concatenate')
            
            # Prepare variable
            sep_idx = int(combined_training_xx.shape[1]/2)
            unique_class = {'pos':'POS', 'neg':'NEG'}
            
            # Train model with best params

            # Test model
            [predictedScores, predictedY, test_time] = distance_model.predict(combined_test_xx[:,0:sep_idx], combined_test_xx[:,sep_idx:], combined_test_yy, unique_class, best_param.classifier_threshold, distance_metric=best_param.distanceFunc)

            # tmp = pd.DataFrame({'y_true':combined_test_yy, 'pred_scores':predictedScores, 'pred_y':predictedY})
            # tmp.to_csv('euclid.csv')

            # Eval performance
            # Performance metrics
            performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, np.array(list(unique_class.values())))
            # AUC
            performance_metric.update(my_util.binary_classes_auc(combined_test_yy, predictedScores, unique_class['pos']))
            # Biometric metrics
            performance_metric.update(my_util.biometric_metric(combined_test_yy, predictedScores, unique_class['pos'], score_order='descending'))

            # Save score
            exp_result = {'distanceFunc':best_param.distanceFunc, 'randomseed': exp_numb, 'weightID': combined_training_id, 'label_classes': np.array(list(unique_class.values())), 'training_time': 0, 'test_time': test_time, 'algorithm': 'euclid', 'experiment_name': exp_name, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
            exp_result.update(performance_metric)
            my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

            print('Finished ' + exp_name_seed)

            del predictedScores, predictedY, test_time
            del performance_metric, exp_result

        del train_index, test_index

    else:
        print('The experiment ' + str(exp_numb) + ' already existed.')
