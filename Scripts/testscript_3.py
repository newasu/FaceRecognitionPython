
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
training_useTF = False
test_useTF = False

# Parameter settings
param_grid = {'distanceFunc':'euclidean', 
'hiddenNodePerc':1.0, 
'regC':1}
combine_rule = 'sum'

test_size = 0.3

pos_class = 'POS'

# Path
# Dataset path
dataset_name = 'CelebA'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'CelebA_features'])
dataset_path = dataset_path + 'CelebA_retinaface.txt'

#############################################################################################

# Read data
my_data = pd.read_csv(dataset_path, sep=" ", header=0)

# Select only some classes
my_data = my_data[my_data['id'].between(1, 1000)]

# Assign xx, yy and data id
xx = my_data.iloc[:,2:].values
yy = my_data.id.values
image_id = my_data.image_id.values

del dataset_path, my_data

[train_index, test_index] = my_util.split_data_by_classes(yy, test_size=test_size, random_state=0)
selm_model = selm()
welm_model = welm()

# Construct triplet training dataset
triplet_paired_list = my_util.triplet_loss_paring(image_id[train_index], yy[train_index], randomseed=0, num_cores=-1)
[combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(xx[train_index], image_id[train_index], triplet_paired_list, combine_rule=combine_rule)

# Construct triplet test dataset
triplet_paired_list = my_util.triplet_loss_paring(image_id[test_index], yy[test_index], randomseed=0, num_cores=-1)
[combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(xx[test_index], image_id[test_index], triplet_paired_list, combine_rule=combine_rule)

# Train model with best params thresholding
[weights, weightID, optimal_threshold, beta, label_classes, training_time] = welm_model.train_thresholding(
combined_training_xx, combined_training_yy, pos_class,
trainingDataID=combined_training_id, 
distanceFunc=param_grid['distanceFunc'], 
hiddenNodePerc=param_grid['hiddenNodePerc'], 
regC=param_grid['regC'], 
threshold='', 
randomseed=0,
useTF=test_useTF)

# Test model thresholding
[predictedScores, predictedY, test_time] = welm_model.predict_thresholding(combined_test_xx, weights, beta, param_grid['distanceFunc'], optimal_threshold, label_classes, pos_class, useTF=test_useTF)

# Eval performance thresholding
# Performance matrix
performance_metric_thresholding = my_util.classification_performance_metric(combined_test_yy, predictedY, label_classes)
# Biometric metrics
performance_metric_thresholding.update(my_util.biometric_metric(combined_test_yy, predictedScores, pos_class, score_order='ascending'))



# Train model with best params
[weights, weightID, beta, label_classes, training_time] = welm_model.train(
combined_training_xx, combined_training_yy,
trainingDataID=combined_training_id, 
distanceFunc=param_grid['distanceFunc'], 
hiddenNodePerc=param_grid['hiddenNodePerc'], 
regC=param_grid['regC'], 
randomseed=0,
useTF=test_useTF)

# Test model thresholding
[predictedScores, predictedY, test_time] = welm_model.predict(combined_test_xx, weights, beta, param_grid['distanceFunc'], label_classes, useTF=test_useTF)

# Eval performance
pos_class_idx = label_classes == pos_class
# Performance matrix
performance_metric = my_util.classification_performance_metric(combined_test_yy, predictedY, label_classes)
# Biometric metrics
performance_metric.update(my_util.biometric_metric(combined_test_yy, predictedScores[:,pos_class_idx], pos_class, score_order='ascending'))

print()
