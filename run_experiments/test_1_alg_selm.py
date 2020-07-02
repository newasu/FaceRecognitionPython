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
exp_name = ('test_1_alg_selm_cr_sum')
training_useTF = False
test_useTF = False

# Parameter settings
numb_exp = [0] # define randomseed as list
cv_numb = 5
cv_run = -1 # -1 = run all seed, else, run only define
param_grid = {'distanceFunc':'euclidean', 
'hiddenNodePerc':np.arange(0.1, 1.1, 0.1), 
'regC':10**np.arange(-6, 7, dtype='float')}
combine_rule = 'sum'

test_size = 0.3

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

        # Run experiment each iterative
            
        # from collections import Counter
        # data_class_freq = Counter(yy[train_index])
        # unique_class = np.array(list(data_class_freq.keys()))
        # data_class_freq = np.array(list(data_class_freq.values()))
        
        # Initial model
        selm_model = selm()
        
        # Generate k-fold index
        [kfold_training_idx, kfold_test_idx] = my_util.split_kfold_by_classes(yy[train_index], n_splits=cv_numb, random_state=exp_numb)
        
        # Grid search
        [cv_results, avg_cv_results] = selm_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, xx[train_index], yy[train_index], image_id[train_index], param_grid, gridsearch_path, exp_name_seed, cv_run=cv_run, randomseed=exp_numb, useTF=training_useTF, combine_rule=combine_rule)

        if cv_run == -1:
            # Initial model
            welm_model = welm()
            
            # Best params
            best_param = avg_cv_results.iloc[0]
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[train_index], yy[train_index], randomseed=exp_numb, num_cores=-1)
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(xx[train_index], image_id[train_index], triplet_paired_list, combine_rule=combine_rule)
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(image_id[test_index], yy[test_index], randomseed=exp_numb, num_cores=-1)
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(xx[test_index], image_id[test_index], triplet_paired_list, combine_rule=combine_rule)

            # Train model with best params
            [weights, weightID, beta, label_classes, training_time] = welm_model.train(
            combined_training_xx, combined_training_yy, 
            trainingDataID=combined_training_id, 
            distanceFunc=best_param.distanceFunc, 
            hiddenNodePerc=best_param.hiddenNodePerc, 
            regC=best_param.regC, 
            randomseed=exp_numb,
            useTF=test_useTF)

            # Test model
            [predictedScores, predictedY, test_time] = welm_model.predict(combined_test_xx, weights, beta, best_param.distanceFunc, label_classes, useTF=test_useTF)
            
            # tmp = pd.DataFrame({'y_true':combined_test_yy, 'pred_scores':np.array(predictedScores[:,pos_class_idx].T)[0], 'pred_y':predictedY.flatten()})
            # tmp.to_csv('selm.csv')

            pos_class_idx = label_classes == pos_class

            # Eval performance
            # Performance matrix
            performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, label_classes)
            # Biometric metrics
            performance_metric.update(my_util.biometric_metric(combined_test_yy, predictedScores[:,pos_class_idx], pos_class, score_order='ascending'))
            # AUC
            # performance_metric.update(my_util.cal_auc(combined_test_yy, predictedScores, label_classes))

            # Save score
            exp_result = {'distanceFunc':best_param.distanceFunc, 'hiddenNodePerc': best_param.hiddenNodePerc, 'regC':best_param.regC, 'combine_rule':combine_rule, 'randomseed': exp_numb, 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'selm', 'experiment_name': exp_name, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':image_id[test_index], 'dataset_name':dataset_name}
            exp_result.update(performance_metric)
            my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

            print('Finished ' + exp_name_seed)

            del weights, weightID, beta, label_classes, training_time
            del predictedScores, predictedY, test_time
            del performance_metric, exp_result

        del train_index, test_index

    else:
        print('The experiment ' + str(exp_numb) + ' already existed.')

print()
