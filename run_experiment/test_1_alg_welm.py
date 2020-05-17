# code testing

# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import lib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import itertools
from sklearn.utils.fixes import loguniform

# Import my own lib
import others.utilities as my_util
from WELM.welm import welm

#############################################################################################

# Experiment name
exp_name = ('test_1_alg_welm')

# Parameter settings
numb_exp = [0, 1] # define randomseed as list
cv_numb = 5
cv_run = -1 # -1 = run all seed, else, run only define
param_grid = {'distanceFunc':'euclidean', 
'hiddenNodePerc':[0.20, 0.40, 0.60, 0.80, 1.00], 
'regC':[0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}

# Path
# Dataset path
dataset_name = 'CelebA'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA(partial)_1'])
dataset_path = dataset_path + 'CelebAretinaface11000(clean).txt'
# Result path
exp_result_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'exp_result', exp_name])
# Grid search path
gridsearch_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Result', 'gridsearch'])

#############################################################################################

# Prepare save path
my_util.make_directory(exp_result_path)
print('Grid search will be saved at -> ' + exp_result_path)
print('Experiment results will be saved at -> ' + exp_result_path)

# Read data
my_data = pd.read_csv(dataset_path, sep=" ", header=0)

# Select only some classes
my_data_sep = my_data[my_data['id'].between(1, 100)]

# Assign xx, yy and data id
xx = my_data_sep.iloc[:,2:].values
yy = my_data_sep.id.values
image_id = my_data_sep.image_id.values

del dataset_path, my_data, my_data_sep

# Run experiment
for exp_numb in numb_exp:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_seed_' + str(exp_numb))

    if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):
        # Get index for partition training/test set
        data_spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=exp_numb)
        data_spliter.get_n_splits(xx, yy)

        # Run experiment each iterative
        for train_index, test_index in data_spliter.split(xx, yy):
            # Model
            m1 = welm()

            # Grid search
            [cv_results, avg_cv_results] = m1.grid_search_cv_parallel(xx[train_index], yy[train_index], image_id[train_index], param_grid, gridsearch_path, exp_name_seed, cv=cv_numb, cv_run=cv_run, randomseed=exp_numb)

            if cv_run == -1:
                # Best params
                best_param = avg_cv_results.iloc[0]

                # Train model with best params
                [weights, weightID, beta, label_classes, training_time] = m1.train(
                xx[train_index], 
                yy[train_index], 
                trainingDataID=image_id[train_index], 
                distanceFunc=best_param.distanceFunc, 
                hiddenNodePerc=best_param.hiddenNodePerc, 
                regC=best_param.regC, 
                randomseed=exp_numb)

                # Test model
                [predictedScores, predictedY, test_time] = m1.predict(xx[test_index], weights, beta, best_param.distanceFunc, label_classes)

                # Eval performance
                # Performance matrix
                performance_matrix = my_util.eval_classification_performance(yy[test_index], predictedY, label_classes)
                # AUC
                performance_matrix.update(my_util.cal_auc(yy[test_index], predictedScores, label_classes))
                # Accuracy
                performance_matrix['accuracy'] = my_util.cal_accuracy(yy[test_index], predictedY)

                # Save score
                exp_result = {'distanceFunc':best_param.distanceFunc, 'hiddenNodePerc': best_param.hiddenNodePerc, 'regC':best_param.regC, 'randomseed': exp_numb, 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'welm', 'experiment_name': exp_name, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':image_id[test_index], 'dataset_name':dataset_name}
                exp_result.update(performance_matrix)
                my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

                print('Finished ' + exp_name_seed)

                del weights, weightID, beta, label_classes, training_time
                del predictedScores, predictedY, test_time
                del performance_matrix, exp_result

        del data_spliter, train_index, test_index

    else:
        print('The experiment ' + str(exp_numb) + ' already existed.')

print()