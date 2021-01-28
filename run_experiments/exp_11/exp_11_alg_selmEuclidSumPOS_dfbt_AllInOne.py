# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import os
import pandas as pd
import numpy as np
import random
from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
from algorithms.selm import selm
from algorithms.welm import welm

#############################################################################################

# Experiment name
exp = 'exp_11'
exp_name = exp + '_alg_selmEuclidSumPOS_dfbt_AllInOne'
# query_exp_name = exp_name
vl = 'df' # valid from df : diveface , lfw

dataset_name = 'Diveface'
dataset_exacted = 'resnet50'

# Parameter settings
num_used_cores = 1

# Whole run round settings
run_exp_round = [0] # define randomseed as list
training_rd = 0 # reduce size
test_size = 0.3
valid_size = 0.1

exp_name = exp_name + '_rd_' + str(training_rd).replace('.', 'd') + '_vl_' + vl

# k-fold for training
numb_train_kfold = 1
cv_run = -1 # -1 = run all fold, else, run only define

# Algorithm parameters
param_grid = {'distanceFunc':'euclidean', 
              'kernel_param':0, 
              'hiddenNodePerc':(np.arange(7, 0, -1)/10), 
              'regC':10**np.arange(10, -11, -1, dtype='float')}
combine_rule = 'sum'

pos_class = 'POS'

#############################################################################################

# Path
# Dataset path
diveface_dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
lfw_dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw'])
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, exp_name])
# Grid search path
gridsearch_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name])
# Make directory
my_util.make_directory(exp_result_path)
my_util.make_directory(gridsearch_path)

#############################################################################################

def searchIdx(queryID, src):
    orig_indices = src.argsort()
    return orig_indices[np.searchsorted(src[orig_indices], queryID)]

def siamese_layer(fa,fb,cr):
    if cr == 'sum':
        return fa+fb
    elif cr == 'mean':
        return (fa + fb)/2
    elif cr == 'multiply':
        return np.multiply(fa, fb)
    elif cr == 'dist':
        return np.absolute(fa - fb)

#############################################################################################

# Initial model
selm_model = selm()
welm_model = welm()

# Load LFW
lfw = pd.read_csv(lfw_dataset_path + 'DevTest' + os.sep + 'DevTest_balanced_triplet.txt', header=0, sep=' ')
lfw_id_pose = (lfw['id'] + '_' + lfw['pose'].astype(str)).values
pairsDevTest_POS = pd.read_csv(lfw_dataset_path + 'pairsDevTest_POS.txt', header=None, sep='\t')
pairsDevTest_NEG = pd.read_csv(lfw_dataset_path + 'pairsDevTest_NEG.txt', header=None, sep='\t')
# Test list POS
pairsDevTest_POS_1 = pairsDevTest_POS[[0, 1]].copy()
pairsDevTest_POS_2 = pairsDevTest_POS[[0, 2]].copy()
pairsDevTest_POS_1.columns = ['name', 'imagenum']
pairsDevTest_POS_2.columns = ['name', 'imagenum']
pairsDevTest_POS_1['person'] = pairsDevTest_POS_1.name.replace('_', ' ', regex=True)
pairsDevTest_POS_2['person'] = pairsDevTest_POS_2.name.replace('_', ' ', regex=True)
# Test list NEG
pairsDevTest_NEG_1 = pairsDevTest_NEG[[0, 1]].copy()
pairsDevTest_NEG_2 = pairsDevTest_NEG[[2, 3]].copy()
pairsDevTest_NEG_1.columns = ['name', 'imagenum']
pairsDevTest_NEG_2.columns = ['name', 'imagenum']
pairsDevTest_NEG_1['person'] = pairsDevTest_NEG_1.name.replace('_', ' ', regex=True)
pairsDevTest_NEG_2['person'] = pairsDevTest_NEG_2.name.replace('_', ' ', regex=True)
# Initialise test variables
test_idx_anchor = np.empty(0)
test_idx_compare = np.empty(0)
# Query POS
my_queryString = (pairsDevTest_POS_1['name'] + '_' + pairsDevTest_POS_1['imagenum'].astype(str)).values
test_idx_anchor = np.append(test_idx_anchor, searchIdx(my_queryString, lfw_id_pose))
my_queryString = (pairsDevTest_POS_2['name'] + '_' + pairsDevTest_POS_2['imagenum'].astype(str)).values
test_idx_compare = np.append(test_idx_compare, searchIdx(my_queryString, lfw_id_pose))
# Query NEG
my_queryString = (pairsDevTest_NEG_1['name'] + '_' + pairsDevTest_NEG_1['imagenum'].astype(str)).values
test_idx_anchor = np.append(test_idx_anchor, searchIdx(my_queryString, lfw_id_pose))
my_queryString = (pairsDevTest_NEG_2['name'] + '_' + pairsDevTest_NEG_2['imagenum'].astype(str)).values
test_idx_compare = np.append(test_idx_compare, searchIdx(my_queryString, lfw_id_pose))
# Convert to Int
test_idx_anchor = test_idx_anchor.astype(int)
test_idx_compare = test_idx_compare.astype(int)
# Assign
# Anchor
test_data_id_anchor = lfw.iloc[test_idx_anchor].index.values
test_gender_anchor = lfw.iloc[test_idx_anchor]['gender'].values
test_ethnicity_anchor = lfw.iloc[test_idx_anchor]['ethnicity'].values
test_id_anchor = lfw.iloc[test_idx_anchor]['id'].values
test_pose_anchor = lfw.iloc[test_idx_anchor]['pose'].values
test_filename_anchor = test_id_anchor + '_' + np.char.zfill(test_pose_anchor.astype(str), 4) + '.jpg'
test_feature_anchor = lfw.iloc[test_idx_anchor].values[:,5:]
test_race_anchor = test_gender_anchor + '-' + test_ethnicity_anchor
# Compare
test_data_id_compare = lfw.iloc[test_idx_compare].index.values
test_gender_compare = lfw.iloc[test_idx_compare]['gender'].values
test_ethnicity_compare = lfw.iloc[test_idx_compare]['ethnicity'].values
test_id_compare = lfw.iloc[test_idx_compare]['id'].values
test_pose_compare = lfw.iloc[test_idx_compare]['pose'].values
test_filename_compare = test_id_compare + '_' + np.char.zfill(test_pose_compare.astype(str), 4) + '.jpg'
test_feature_compare = lfw.iloc[test_idx_compare].values[:,5:]
test_race_compare = test_gender_compare + '-' + test_ethnicity_compare
# Label
test_data_id = (lfw.iloc[test_idx_anchor].index.astype(str) + '_' + lfw.iloc[test_idx_compare].index.astype(str)).values
test_id = test_id_anchor + '_' + test_pose_anchor.astype(str) + '-' + test_id_compare + '-' + test_pose_compare.astype(str)
test_valid_label = np.tile('POS', 500)
test_valid_label = np.append(test_valid_label, np.tile('NEG', 500))
# Normalise feature
test_feature_anchor = preprocessing.normalize(test_feature_anchor, norm='l2', axis=1, copy=True, return_norm=False)
test_feature_compare = preprocessing.normalize(test_feature_compare, norm='l2', axis=1, copy=True, return_norm=False)

del pairsDevTest_POS, pairsDevTest_NEG
del pairsDevTest_POS_1, pairsDevTest_POS_2, pairsDevTest_NEG_1, pairsDevTest_NEG_2
del my_queryString, test_idx_anchor, test_idx_compare

# Run experiment
for exp_numb in run_exp_round:
    # Experiment name each seed
    exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Load DiveFace data
    diveface = pd.read_csv((diveface_dataset_path + dataset_name + '_' + dataset_exacted + '_balanced_triplet.txt'), sep=' ', header=0)
    # Separate data
    diveface_race = (diveface['gender'] + '-' + diveface['ethnicity']).values
    [training_sep_idx, test_sep_idx, valid_sep_idx] = my_util.split_data_by_id_and_classes(diveface.id.values, diveface_race, test_size=test_size, valid_size=valid_size, random_state=exp_numb)
    # Reduce training size
    [tmp_training_sep_idx, _, _] = my_util.split_data_by_id_and_classes(diveface.id.values[training_sep_idx], diveface_race[training_sep_idx], test_size=training_rd, valid_size=0, random_state=exp_numb)
    training_sep_idx = training_sep_idx[tmp_training_sep_idx]
    del tmp_training_sep_idx
    # Assign data
    # Training data
    x_training = diveface.iloc[training_sep_idx,8:].values
    x_training = preprocessing.normalize(x_training, norm='l2', axis=1, copy=True, return_norm=False)
    y_race_training = diveface_race[training_sep_idx]
    y_class_training = diveface.id.iloc[training_sep_idx].values
    y_id_training = diveface.data_id.iloc[training_sep_idx].values
    
    # # Valid data
    if vl == 'lfw':
        # lfw
        x_valid = lfw.iloc[:,5:].values
        y_race_valid = (lfw.gender + '-' + lfw.ethnicity).values
        y_class_valid = lfw.id.values
        y_id_valid = lfw.index.values
    elif vl == 'df':
        # diveface
        x_valid = diveface.iloc[valid_sep_idx,8:].values
        y_race_valid = diveface_race[valid_sep_idx]
        y_class_valid = diveface.id.iloc[valid_sep_idx].values
        y_id_valid = diveface.data_id.iloc[valid_sep_idx].values
    x_valid = preprocessing.normalize(x_valid, norm='l2', axis=1, copy=True, return_norm=False)
    
    # # Test data
    # test_sep_idx = test_sep_idx[diveface_race[test_sep_idx] == train_class]
    # x_test = diveface.iloc[test_sep_idx,8:].values
    # y_race_test = diveface_race[test_sep_idx]
    # y_class_test = diveface.id.iloc[test_sep_idx].values
    # y_id_test = diveface.data_id.iloc[test_sep_idx].values
    del diveface, diveface_race
    
    # Run experiment each iterative
    # if not my_util.is_path_available(exp_result_path + exp_name_seed + '.npy'):
    
    # Run experiment if not available
    
    # Generate k-fold index for training
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
    
    query_exp_name_seed = (exp_name + '_run_' + str(exp_numb))
    
    # Grid search
    [cv_results, avg_cv_results] = selm_model.grid_search_cv_parallel(kfold_training_idx, kfold_test_idx, np.vstack((x_training, x_valid)), np.append(y_class_training, y_class_valid), np.append(y_id_training, y_id_valid), param_grid, gridsearch_path, query_exp_name_seed, pos_class=pos_class, cv_run=cv_run, randomseed=exp_numb, useTF=False, combine_rule=combine_rule, num_cores=num_used_cores)
    
    # Construct triplet training dataset
    triplet_paired_list = my_util.triplet_loss_paring(y_id_training, y_class_training, randomseed=exp_numb)
    [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(x_training, y_id_training, triplet_paired_list, combine_rule=combine_rule)
    
    combined_test_xx = siamese_layer(test_feature_anchor, test_feature_compare, combine_rule)
    combined_test_yy = test_valid_label
    combined_test_id = test_data_id
    
    # Best params
    avg_cv_results = avg_cv_results.sort_values(by='accuracy', ascending=False)
    best_param_list = avg_cv_results.iloc[0:6,:]
    bucket = {}
    performance_metric_bucket = {}
    for best_param_list_idx in range(0, best_param_list.shape[0]):
        print('Calculating best parameters ' + str(best_param_list_idx) + ' ...')
        best_param = best_param_list.iloc[best_param_list_idx]

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
        
        # Store
        bucket[best_param_list_idx] = {'weightID':weightID, 'beta':beta, 'label_classes':label_classes, 'training_time':training_time, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_time':test_time, 'performance_metric': performance_metric}
        
        performance_metric_bucket[best_param_list_idx] = {'auc':performance_metric['auc'], 'eer':performance_metric['eer'], 'accuracy':performance_metric['accuracy'],  'tar_1':performance_metric['tar_1'], 'tar_0d1':performance_metric['tar_0d1'], 'tar_0d01':performance_metric['tar_0d01']}
        
        del weights, weightID, beta, label_classes, training_time
        del predictedScores, predictedY, test_time
        del performance_metric

    performance_metric_bucket = pd.DataFrame(data=performance_metric_bucket).transpose()
    performance_metric_bucket = performance_metric_bucket.sort_values(by=['accuracy', 'tar_0d01', 'auc', 'eer', 'tar_0d01'], ascending=[False, False, False, True, False])
    best_param = best_param_list.iloc[performance_metric_bucket.index[0]]
    performance_metric = bucket[performance_metric_bucket.index[0]]['performance_metric']
    weightID = bucket[performance_metric_bucket.index[0]]['weightID']
    beta = bucket[performance_metric_bucket.index[0]]['beta']
    label_classes = bucket[performance_metric_bucket.index[0]]['label_classes']
    training_time = bucket[performance_metric_bucket.index[0]]['training_time']
    test_time = bucket[performance_metric_bucket.index[0]]['test_time']
    predictedScores = bucket[performance_metric_bucket.index[0]]['predictedScores']
    predictedY = bucket[performance_metric_bucket.index[0]]['predictedY']

    # Save score
    exp_result = {'distanceFunc':best_param.distanceFunc, 'kernel_param':best_param.kernel_param, 'hiddenNodePerc': best_param.hiddenNodePerc, 'regC':best_param.regC, 'combine_rule':combine_rule, 'randomseed': exp_numb, 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'selm', 'experiment_name': exp_name, 'trueY':combined_test_yy, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':combined_test_id, 'dataset_name':dataset_name}
    exp_result.update(performance_metric)
    my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

    print('Finished ' + exp_name_seed)
    
    print('exp_name: ' + exp_name_seed)
    print('auc: ' + str(exp_result['auc']))
    print('eer: ' + str(exp_result['eer']))
    print('accuracy: ' + str(exp_result['accuracy']))
    print('tar_1: ' + str(exp_result['tar_1']))
    print('tar_0d1: ' + str(exp_result['tar_0d1']))
    print('tar_0d01: ' + str(exp_result['tar_0d01']))
    
    print('regC: ' + str(exp_result['regC']))
    print('hiddenNodePerc: ' + str(exp_result['hiddenNodePerc']))

    del best_param, performance_metric
    del weightID, beta, label_classes, training_time
    del predictedScores, predictedY, test_time
    del combined_training_xx, combined_training_yy, combined_training_id
    del combined_test_xx, combined_test_yy, combined_test_id
    del exp_result
    del bucket, performance_metric_bucket

    del kfold_training_idx, kfold_test_idx
    del cv_results, avg_cv_results



print()
