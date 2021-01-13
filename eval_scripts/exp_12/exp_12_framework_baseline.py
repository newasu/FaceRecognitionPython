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
# from algorithms.welm import welm

#############################################################################################

# Experiment name
exp = 'exp_12'
exp_name = exp + '_framework_baseline'

model_exp = 'exp_11'
model_exp_name = model_exp + '_alg_BaselineEuclideanOneThreshold_female-asian'

train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

# Whole run round settings
run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # define randomseed as list

# dataset_name = 'lfw'
# dataset_exacted = 'resnet50'

# Parameter settings
num_used_cores = 1

#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw'])
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp, exp_name])
# Model path
model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', model_exp, model_exp_name])
# Make directory
# my_util.make_directory(exp_result_path)
# my_util.make_directory(gridsearch_path)

#############################################################################################

def searchIdx(queryID, src):
    orig_indices = src.argsort()
    return orig_indices[np.searchsorted(src[orig_indices], queryID)]

def evaluate(tfa, tfc, tvl, uc, md):
    # Classify
    predictedScores, predictedY, _ = distance_model.predict(tfa, tfc, tvl, uc, md['kernel_param'], distance_metric=md['distanceFunc'])
    
    # Eval performance
    # Performance metrics
    performance_metric = my_util.classification_performance_metric(tvl, predictedY, np.array(['NEG', 'POS']))
    # Biometric metrics
    performance_metric.update(my_util.biometric_metric(tvl, np.ravel(predictedScores), 'POS', score_order='ascending', threshold_step=0.01))
    
    return performance_metric

#############################################################################################

# Read txt
my_data = pd.read_csv(dataset_path + 'DevTest_cleaned.txt', header=0, sep=' ')
my_source = (my_data['id'] + '_' + my_data['pose'].astype(str)).values
pairsDevTest_POS = pd.read_csv(dataset_path + 'pairsDevTest_POS.txt', header=None, sep='\t')
pairsDevTest_NEG = pd.read_csv(dataset_path + 'pairsDevTest_NEG.txt', header=None, sep='\t')

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

# feature = my_data.values[:,5:]

# Initial model
distance_model = paired_distance_alg()
unique_class = {'pos':'POS', 'neg':'NEG'}
feature_size = 2048
label_classes = np.unique(['POS', 'NEG'])

# Initialise test variables
test_idx_anchor = np.empty(0)
test_idx_compare = np.empty(0)
test_gender_anchor = np.empty(0)
test_gender_compare = np.empty(0)
test_ethnicity_anchor = np.empty(0)
test_ethnicity_compare = np.empty(0)
test_id_anchor = np.empty(0)
test_id_compare = np.empty(0)
test_pose_anchor = np.empty(0)
test_pose_compare = np.empty(0)
test_feature_anchor = np.empty((0,feature_size))
test_feature_compare = np.empty((0,feature_size))

# Query POS
my_queryString = (pairsDevTest_POS_1['name'] + '_' + pairsDevTest_POS_1['imagenum'].astype(str)).values
test_idx_anchor = np.append(test_idx_anchor, searchIdx(my_queryString, my_source))
my_queryString = (pairsDevTest_POS_2['name'] + '_' + pairsDevTest_POS_2['imagenum'].astype(str)).values
test_idx_compare = np.append(test_idx_compare, searchIdx(my_queryString, my_source))

# Query NEG
my_queryString = (pairsDevTest_NEG_1['name'] + '_' + pairsDevTest_NEG_1['imagenum'].astype(str)).values
test_idx_anchor = np.append(test_idx_anchor, searchIdx(my_queryString, my_source))
my_queryString = (pairsDevTest_NEG_2['name'] + '_' + pairsDevTest_NEG_2['imagenum'].astype(str)).values
test_idx_compare = np.append(test_idx_compare, searchIdx(my_queryString, my_source))

test_idx_anchor = test_idx_anchor.astype(int)
test_idx_compare = test_idx_compare.astype(int)

# Assign
test_gender_anchor = np.append(test_gender_anchor, my_data.iloc[test_idx_anchor]['gender'])
test_ethnicity_anchor = np.append(test_ethnicity_anchor, my_data.iloc[test_idx_anchor]['ethnicity'])
test_id_anchor = np.append(test_id_anchor, my_data.iloc[test_idx_anchor]['id'])
test_pose_anchor = np.append(test_pose_anchor, my_data.iloc[test_idx_anchor]['pose']).astype(int)
test_feature_anchor = np.vstack((test_feature_anchor, my_data.iloc[test_idx_anchor].values[:,5:]))
test_race_anchor = test_gender_anchor + '-' + test_ethnicity_anchor
test_gender_compare = np.append(test_gender_compare, my_data.iloc[test_idx_compare]['gender'])
test_ethnicity_compare = np.append(test_ethnicity_compare, my_data.iloc[test_idx_compare]['ethnicity'])
test_id_compare = np.append(test_id_compare, my_data.iloc[test_idx_compare]['id'])
test_pose_compare = np.append(test_pose_compare, my_data.iloc[test_idx_compare]['pose']).astype(int)
test_feature_compare = np.vstack((test_feature_compare, my_data.iloc[test_idx_compare].values[:,5:]))
test_race_compare = test_gender_compare + '-' + test_ethnicity_compare
test_id = test_id_anchor + '_' + test_pose_anchor.astype(str) + '-' + test_id_compare + '-' + test_pose_compare.astype(str)
test_valid_label = np.tile('POS', 500)
test_valid_label = np.append(test_valid_label, np.tile('NEG', 500))

# Normalise
if 'baseline' in exp_name:
    test_feature_anchor = preprocessing.normalize(test_feature_anchor, norm='l2', axis=1, copy=True, return_norm=False)
    test_feature_compare = preprocessing.normalize(test_feature_compare, norm='l2', axis=1, copy=True, return_norm=False)

tmp_accuracy_all = np.empty(0)
tmp_auc_all = np.empty(0)
tmp_accuracy_race = np.empty((0, len(train_class)))
tmp_auc_race = np.empty((0, len(train_class)))

# Run experiment
for exp_numb in run_exp_round:
    exp_name_seed = exp_name + '_run_' + str(exp_numb)
    
    # Load model
    model = my_util.load_numpy_file(model_path + model_exp_name + '_run_' + str(exp_numb) + '.npy')
    
    # Evaluate all
    performance_metric = evaluate(test_feature_anchor, test_feature_compare, test_valid_label, unique_class, model)
    tmp_accuracy_all = np.append(tmp_accuracy_all, performance_metric['accuracy'])
    tmp_auc_all = np.append(tmp_auc_all, performance_metric['auc'])
    
    tmp_accuracy = np.empty(0)
    tmp_auc = np.empty(0)
    for race_idx in train_class:
        # Assign idx
        tmp_idx_anchor = test_race_anchor == race_idx
        tmp_idx_compare = test_race_compare == race_idx
        tmp_idx = tmp_idx_anchor + tmp_idx_compare
        
        # Evaluate race
        performance_metric = evaluate(test_feature_anchor[tmp_idx,:], test_feature_compare[tmp_idx,:], test_valid_label[tmp_idx], unique_class, model)
        tmp_accuracy = np.append(tmp_accuracy, performance_metric['accuracy'])
        tmp_auc = np.append(tmp_auc, performance_metric['auc'])
        
        del tmp_idx_anchor, tmp_idx_compare, tmp_idx
        del performance_metric
    
    # Append
    tmp_accuracy_race = np.vstack((tmp_accuracy_race, tmp_accuracy))
    tmp_auc_race = np.vstack((tmp_auc_race, tmp_auc))
    del tmp_accuracy, tmp_auc
    
    # Save score
    # exp_result = {'distanceFunc':model['distanceFunc'], 'kernel_param':model['kernel_param'], 'combine_rule':model['combine_rule'], 'randomseed': exp_numb, 'label_classes': label_classes, 'algorithm':model['algorithm'], 'experiment_name': exp_name_seed, 'trueY':test_valid_label, 'predictedScores':predictedScores, 'predictedY':predictedY, 'test_image_id':test_id, 'dataset_name':'lfw'}
    # exp_result.update(performance_metric)
    # my_util.save_numpy(exp_result, exp_result_path, exp_name_seed)

    print('Finished ' + exp_name_seed)
    
    del performance_metric
    # del predictedScores, predictedY
    # del exp_result

print()
