import sys
sys.path.append("././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfa

# Import my own lib
import others.utilities as my_util
from algorithms.selm import selm
from algorithms.welm import welm
from algorithms.paired_distance_alg import paired_distance_alg

gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#############################################################################################

filename_comment = 'eer'
param = {'exp':'exp_7', 
         'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_240_e_50_a_1', 'b_360_e_50_a_1', 'b_270_e_50_a_1', 'b_240_e_50_a_1'], 
         'epoch': [36, 32, 42, 25, 35, 28], 
         'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface

exp = param['exp']
exp_name = exp + '_alg_tl' + dataset_exacted

train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

# classifier
classifier_exp = 'exp_11'
classifier_model_rule = ['Mean', 'Mean', 'Mean', 'Multiply', 'Mean', 'Dist'] # Dist Mean Multiply Sum

# gender and ethnicity model
model_exp = 'exp_12'
gender_exp_name = model_exp + '_gender_welm'
ethnicity_exp_name = model_exp + '_ethnicity_welm'
mode = 'auto' # auto manual

# Whole run round settings
run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # define randomseed as list

random_seed = 0

#############################################################################################

# Path
# Dataset path
diveface_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw'])
# Model path
gender_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', model_exp, gender_exp_name, gender_exp_name + '_run_0.npy'])
ethnicity_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', model_exp, ethnicity_exp_name, ethnicity_exp_name + '_run_0.npy'])

#############################################################################################

def searchIdx(queryID, src):
    orig_indices = src.argsort()
    return orig_indices[np.searchsorted(src[orig_indices], queryID)]

def evaluate_selm(tfa, tfc, tra, trc, tvl, sr, uc, md, smr):
    predictedY = np.tile('NEG', tvl.size)
    data_id = my_data_triplet['data_id'].values
    orig_indices = data_id.argsort()
    
    for tc_idx, tc_val in enumerate(train_class):
        tmp_idx = tra[sr] == tc_val
        tmp_weight = pd.DataFrame(md[tc_val]['weightID'])[0].str.split('-',expand=True)
        weight_a_idx = orig_indices[np.searchsorted(data_id[orig_indices], tmp_weight[0].values.astype(int))]
        weight_b_idx = orig_indices[np.searchsorted(data_id[orig_indices], tmp_weight[1].values.astype(int))]
        # siamese layer
        tmp_feature = siamese_layer(tfa[sr][tmp_idx], tfc[sr][tmp_idx], smr[tc_idx])
        tmp_weight_feature = siamese_layer(my_data_triplet.iloc[weight_a_idx].values[:,8:], my_data_triplet.iloc[weight_b_idx].values[:,8:], smr[tc_idx])
        # Test model
        [_, tmp_predictedY, _] = welm_model.predict(tmp_feature, tmp_weight_feature, md[tc_val]['beta'], md[tc_val]['distanceFunc'], md[tc_val]['kernel_param'], md[tc_val]['label_classes'], useTF=False)
        # Bind into whole predicted list
        predictedY[np.where(sr)[0][tmp_idx]] = np.ravel(tmp_predictedY)
        
        print(tc_val + ': ' + str(my_util.cal_accuracy(tvl[np.where(sr)[0][tmp_idx]], tmp_predictedY)))
        
        del tmp_idx, tmp_weight, weight_a_idx, weight_b_idx
        del tmp_feature, tmp_weight_feature, tmp_predictedY
    
    # Assign NEG for samples were classified as not same race class
    predictedY[~sr] = 'NEG'
    
    # Eval performance
    # Performance metrics
    performance_metric = {'accuracy':my_util.cal_accuracy(tvl, predictedY)}
    
    return performance_metric

def predict_race(fa, fb, tla, tlb, m):
    if m == 'auto':
        pd_a = tla
        pd_b = tlb
    else:
        my_data = pd.read_csv((diveface_path + 'Diveface' + '_' + 'resnet50' + '_nonorm.txt'), sep=" ", header=0)
        data_id = my_data.data_id
        orig_indices = data_id.argsort()
        # Predict gender
        gender_model_weight_idx = orig_indices[np.searchsorted(data_id[orig_indices], gender_model['weightID'])].values
        [_, pd_a_gender, _] = welm_model.predict(fa, my_data.iloc[gender_model_weight_idx].values[:,8:], gender_model['beta'], gender_model['distanceFunc'], gender_model['kernel_param'], gender_model['label_classes'], useTF=False)
        [_, pd_b_gender, _] = welm_model.predict(fb, my_data.iloc[gender_model_weight_idx].values[:,8:], gender_model['beta'], gender_model['distanceFunc'], gender_model['kernel_param'], gender_model['label_classes'], useTF=False)
        # Predict ethnicity
        ethnicity_model_weight_idx = orig_indices[np.searchsorted(data_id[orig_indices], ethnicity_model['weightID'])].values
        [_, pd_a_ethnicity, _] = welm_model.predict(fa, my_data.iloc[ethnicity_model_weight_idx].values[:,8:], ethnicity_model['beta'], ethnicity_model['distanceFunc'], ethnicity_model['kernel_param'], ethnicity_model['label_classes'], useTF=False)
        [_, pd_b_ethnicity, _] = welm_model.predict(fb, my_data.iloc[ethnicity_model_weight_idx].values[:,8:], ethnicity_model['beta'], ethnicity_model['distanceFunc'], ethnicity_model['kernel_param'], ethnicity_model['label_classes'], useTF=False)
        pd_a = pd_a_gender + '-' + pd_a_ethnicity
        pd_b = pd_b_gender + '-' + pd_b_ethnicity
    pd_compare = pd_a == pd_b
    return pd_a, pd_b, pd_compare

def siamese_layer(fa,fb,cr):
    if cr == 'Sum':
        return fa+fb
    elif cr == 'Mean':
        return (fa + fb)/2
    elif cr == 'Multiply':
        return np.multiply(fa, fb)
    elif cr == 'Dist':
        return np.absolute(fa - fb)

#############################################################################################

# Feature size
if dataset_exacted == 'vgg16':
    feature_size = 4096
elif dataset_exacted == 'resnet50':
    feature_size = 2048
proposed_model_feature_size = 1024

# Initial triplets network model
triplet_model_path = {}
triplet_model = {}
for class_idx, class_val in enumerate(param['class']):
    # Triplet model
    if param['exp'] == 'exp_7':
        triplet_model_path[class_val] = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name + param['class'][class_idx] + '_' + param['model'][class_idx] + '_run_' + str(random_seed)])
    else:
        triplet_model_path[class_val] = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', exp, exp_name + param['class-model'][class_idx] + '_' + param['model'][class_idx] + '_run_' + str(random_seed)])
    triplet_model[class_val] = tf.keras.models.Sequential()
    triplet_model[class_val].add(tf.keras.layers.Dense(proposed_model_feature_size, input_dim=feature_size, activation='linear'))
    triplet_model[class_val].add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    triplet_model[class_val].compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfa.losses.TripletSemiHardLoss())
    triplet_model[class_val].load_weights(triplet_model_path[class_val] + 'cp-' + str(param['epoch'][class_idx]).zfill(4) + '.ckpt')
    
del triplet_model_path

gender_model = my_util.load_numpy_file(gender_model_path[:-1])
ethnicity_model = my_util.load_numpy_file(ethnicity_model_path[:-1])

# Initial model
distance_model = paired_distance_alg()
selm_model = selm()
welm_model = welm()
unique_class = {'pos':'POS', 'neg':'NEG'}
label_classes = np.unique(['POS', 'NEG'])

#############################################################################################

# Read txt
my_data_triplet = pd.read_csv((diveface_path + 'Diveface' + '_' + 'resnet50' + '_' + 'exp_7' + '_run_' + str(0) + '(' + filename_comment + ').txt'), sep=" ", header=0)
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

# Predict race
predicted_race_anchor, predicted_race_compare, same_race = predict_race(test_feature_anchor, test_feature_compare, test_race_anchor, test_race_compare, mode)
# my_util.cal_accuracy(test_race_anchor, predicted_race_anchor)
# my_util.cal_accuracy(test_race_compare, predicted_race_compare)
# my_util.cal_accuracy(np.append(test_race_anchor, test_race_compare) , np.append(predicted_race_anchor, predicted_race_compare))

# Extract triplet feature
test_exacted_feature_anchor = np.empty((test_race_anchor.size, proposed_model_feature_size))
test_exacted_feature_compare = np.empty((test_race_compare.size, proposed_model_feature_size))
for class_val in tqdm(param['class']):
    # Extract anchor
    tmp_idx = np.where(predicted_race_anchor == class_val)[0]
    feature_embedding = test_feature_anchor[tmp_idx]
    feature_embedding = triplet_model[class_val].predict(feature_embedding.astype(np.float64))
    test_exacted_feature_anchor[tmp_idx] = feature_embedding
    del tmp_idx, feature_embedding
    # Extracr compare
    tmp_idx = np.where(predicted_race_compare == class_val)[0]
    feature_embedding = test_feature_compare[tmp_idx]
    feature_embedding = triplet_model[class_val].predict(feature_embedding.astype(np.float64))
    test_exacted_feature_compare[tmp_idx] = feature_embedding
    del tmp_idx, feature_embedding

#############################################################################################

tmp_accuracy_all = np.empty(0)
# tmp_auc_all = np.empty(0)
# tmp_accuracy_race = np.empty((0, len(train_class)))
# tmp_auc_race = np.empty((0, len(train_class)))

# Run experiment
for exp_numb in run_exp_round:
    exp_name_seed = exp_name + '_run_' + str(exp_numb)
    
    # Load model
    model = {}
    for train_class_idx, train_class_val in enumerate(train_class):
        tmp_exp_name = classifier_exp + '_alg_selmEuclid' + classifier_model_rule[train_class_idx] + 'POS_' + train_class_val
        model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', classifier_exp, tmp_exp_name])
        model[train_class_val] = my_util.load_numpy_file(model_path + tmp_exp_name + '_run_' + str(exp_numb) + '.npy')
    
    # Evaluate all
    performance_metric = evaluate_selm(test_exacted_feature_anchor, test_exacted_feature_compare, test_race_anchor, test_race_compare, test_valid_label, same_race, unique_class, model, classifier_model_rule)
    tmp_accuracy_all = np.append(tmp_accuracy_all, performance_metric['accuracy'])
    
    print('Finished ' + exp_name_seed)
    
    del performance_metric
    
print(tmp_accuracy_all)
print(np.mean(tmp_accuracy_all))

print()
