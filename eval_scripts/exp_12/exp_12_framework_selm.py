import sys
sys.path.append("././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

import tensorflow as tf
import tensorflow_addons as tfad
from sklearn import preprocessing

# Import my own lib
import others.utilities as my_util
# from algorithms.selm import selm
from algorithms.welm import welm
from algorithms.paired_distance_alg import paired_distance_alg

gpu_id = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
# Clear GPU cache
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

#############################################################################################

# classifier
eval_mode = 'selm' # selm baseline
classifier_exp = 'exp_11'
classifier_model_rule = ['Sum', 'Sum', 'Sum', 'Sum', 'Sum', 'Sum'] # Dist Mean Multiply Sum
allinone = 'rd_0d5_vl_df' # rd_0d5_vl_df rd_0d2_vl_df rd_0d67_vl_df rd_0d6_vl_df
balanced_triplet = True     # rd_0_vl_df rd_0d2_vl_df rd_0d3_vl_df rd_0d5_vl_df

# gender and ethnicity model
race_classify_mode = 'none' # auto manual none
model_exp = 'exp_12'
gender_exp_name = model_exp + '_gender_welm'
ethnicity_exp_name = model_exp + '_ethnicity_welm'

# Experiment
exp_name = model_exp + '_framework_' + eval_mode + '-' + allinone + '_' + race_classify_mode
if balanced_triplet:
    exp_name = exp_name + '_dfbt'

triplet_filename_comment = 'eer'
triplet_param = {'exp':'exp_7', 
         'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_240_e_50_a_1', 'b_360_e_50_a_1', 'b_270_e_50_a_1', 'b_240_e_50_a_1'], 
         'epoch': [36, 32, 42, 25, 35, 28], 
         'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}

dataset_name = 'Diveface'
dataset_exacted = 'resnet50' # vgg16 resnet50 retinaface

triplet_model_exp = triplet_param['exp']
triplet_model_exp_name = triplet_model_exp + '_alg_tl' + dataset_exacted

train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

# Whole run round settings
run_exp_round = [0] # define randomseed as list [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

random_seed = 0

#############################################################################################

# Path
# Dataset path
diveface_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw'])
# Model path
gender_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', model_exp, gender_exp_name, gender_exp_name + '_run_0.npy'])
ethnicity_model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', model_exp, ethnicity_exp_name, ethnicity_exp_name + '_run_0.npy'])
# Save path
summary_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'summary', model_exp, exp_name])
# Make directory
my_util.make_directory(summary_path)

#############################################################################################

def searchIdx(queryID, src):
    orig_indices = src.argsort()
    return orig_indices[np.searchsorted(src[orig_indices], queryID)]

def evaluate(tfa, tfc, tra, trc, pra, prc, tvl, sr, uc, md, em):
    data_id = diveface['data_id'].values
    orig_indices = data_id.argsort()
    predictedY = np.tile('NEG', tvl.size)
    
    for tc_idx, tc_val in enumerate(train_class):
        tmp_idx = pra[sr] == tc_val
        # Predict
        if em == 'selm':
            tmp_predictedY = cal_selm(md[tc_val], data_id, orig_indices, tfa[sr][tmp_idx], tfc[sr][tmp_idx], md[tc_val]['combine_rule'])
        else:
            _, tmp_predictedY, _ = distance_model.predict(tfa[sr][tmp_idx], tfc[sr][tmp_idx], tvl[sr][tmp_idx], uc, md[tc_val]['kernel_param'], distance_metric=md[tc_val]['distanceFunc'])
        # Bind into whole predicted list
        predictedY[np.where(sr)[0][tmp_idx]] = np.ravel(tmp_predictedY)
        # Cal metric
        # tmp_metric = my_util.cal_accuracy(tvl[np.where(sr)[0][tmp_idx]], tmp_predictedY)
        # print(tc_val + ': ' + str(tmp_metric))
        del tmp_idx, tmp_predictedY
        
    # Assign NEG for samples were classified as not same race class
    predictedY[~sr] = 'NEG'
    # Eval performance
    correct_list = tvl == predictedY
    # Performance metrics
    performance_metric = {'accuracy':my_util.cal_accuracy(tvl, predictedY)}
    
    return performance_metric, predictedY, correct_list

def cal_selm(_md, _data_id, _orig_indices, _tfa, _tfc, _smr):
    # selm weight
    tmp_weight = pd.DataFrame(_md['weightID'])[0].str.split('-', expand=True)
    # weight_a_idx = _orig_indices[np.searchsorted(_data_id[_orig_indices], tmp_weight[0].values.astype(int))]
    # weight_b_idx = _orig_indices[np.searchsorted(_data_id[_orig_indices], tmp_weight[1].values.astype(int))]
    weight_a_idx = tmp_weight[0].values.astype(int)
    weight_b_idx = tmp_weight[1].values.astype(int)
    # siamese layer
    tmp_feature = siamese_layer(_tfa, _tfc, _smr)
    # tmp_weight_feature = siamese_layer(diveface.iloc[weight_a_idx].values[:,8:], diveface.iloc[weight_b_idx].values[:,8:], _smr)
    tmp_weight_feature = siamese_layer(selm_weight.loc[weight_a_idx].values, selm_weight.loc[weight_b_idx].values, _smr)
    # Test model
    [_, tmp_predictedY, _] = welm_model.predict(tmp_feature, tmp_weight_feature, _md['beta'], _md['distanceFunc'], _md['kernel_param'], _md['label_classes'], useTF=False)
    return tmp_predictedY

def predict_race(fa, fb, tla, tlb, m):
    if m == 'auto':
        data_id = diveface.data_id
        orig_indices = data_id.argsort()
        # Predict gender
        gender_model_weight_idx = orig_indices[np.searchsorted(data_id[orig_indices], gender_model['weightID'])].values
        [_, pd_a_gender, _] = welm_model.predict(fa, diveface.iloc[gender_model_weight_idx].values[:,8:], gender_model['beta'], gender_model['distanceFunc'], gender_model['kernel_param'], gender_model['label_classes'], useTF=False)
        [_, pd_b_gender, _] = welm_model.predict(fb, diveface.iloc[gender_model_weight_idx].values[:,8:], gender_model['beta'], gender_model['distanceFunc'], gender_model['kernel_param'], gender_model['label_classes'], useTF=False)
        # Predict ethnicity
        ethnicity_model_weight_idx = orig_indices[np.searchsorted(data_id[orig_indices], ethnicity_model['weightID'])].values
        [_, pd_a_ethnicity, _] = welm_model.predict(fa, diveface.iloc[ethnicity_model_weight_idx].values[:,8:], ethnicity_model['beta'], ethnicity_model['distanceFunc'], ethnicity_model['kernel_param'], ethnicity_model['label_classes'], useTF=False)
        [_, pd_b_ethnicity, _] = welm_model.predict(fb, diveface.iloc[ethnicity_model_weight_idx].values[:,8:], ethnicity_model['beta'], ethnicity_model['distanceFunc'], ethnicity_model['kernel_param'], ethnicity_model['label_classes'], useTF=False)
        pd_a = np.ravel(pd_a_gender + '-' + pd_a_ethnicity)
        pd_b = np.ravel(pd_b_gender + '-' + pd_b_ethnicity)
        pd_compare = pd_a == pd_b
    elif m == 'manual':
        pd_a = tla
        pd_b = tlb
        pd_compare = pd_a == pd_b
    elif m == 'none':
        pd_a = tla
        pd_b = tlb
        pd_compare = np.tile(True, pd_a.size)
    
    return pd_a, pd_b, pd_compare

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

# Feature size
if balanced_triplet:
    feature_size = 512
else:
    feature_size = 2048
proposed_model_feature_size = 1024

# Initial model
if eval_mode == 'selm':
    # selm_model = selm()
    welm_model = welm()
    
    if race_classify_mode != 'none':
        # Initial triplets network model
        triplet_model_path = {}
        triplet_model = {}
        for class_idx, class_val in enumerate(triplet_param['class']):
            # Triplet model
            if triplet_param['exp'] == 'exp_7':
                triplet_model_path[class_val] = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', triplet_model_exp, triplet_model_exp_name + triplet_param['class'][class_idx] + '_' + triplet_param['model'][class_idx] + '_run_' + str(random_seed)])
            else:
                triplet_model_path[class_val] = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'gridsearch', triplet_model_exp, triplet_model_exp_name + triplet_param['class-model'][class_idx] + '_' + triplet_param['model'][class_idx] + '_run_' + str(random_seed)])
            triplet_model[class_val] = tf.keras.models.Sequential()
            triplet_model[class_val].add(tf.keras.layers.Dense(proposed_model_feature_size, input_dim=feature_size, activation='linear'))
            triplet_model[class_val].add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
            triplet_model[class_val].compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tfad.losses.TripletSemiHardLoss())
            triplet_model[class_val].load_weights(triplet_model_path[class_val] + 'cp-' + str(triplet_param['epoch'][class_idx]).zfill(4) + '.ckpt')
        del triplet_model_path
    
else:
    distance_model = paired_distance_alg()

if race_classify_mode == 'auto':
    gender_model = my_util.load_numpy_file(gender_model_path[:-1])
    ethnicity_model = my_util.load_numpy_file(ethnicity_model_path[:-1])

unique_class = {'pos':'POS', 'neg':'NEG'}
label_classes = np.unique(['POS', 'NEG'])

#############################################################################################

# Read txt
if balanced_triplet:
    diveface = pd.read_csv((diveface_path + 'Diveface' + '_' + 'resnet50' + '_balanced_triplet.txt'), sep=" ", header=0)
else:
    diveface = pd.read_csv((diveface_path + 'Diveface' + '_' + 'resnet50' + '_nonorm.txt'), sep=" ", header=0)
diveface_race = (diveface.gender + '-' + diveface.ethnicity).values
# my_data_triplet = pd.read_csv((diveface_path + 'Diveface' + '_' + 'resnet50' + '_' + 'exp_7' + '_run_' + str(0) + '(' + triplet_filename_comment + ').txt'), sep=" ", header=0)
if balanced_triplet:
    lfw = pd.read_csv(dataset_path + 'DevTest' + os.sep + 'DevTest_balanced_triplet.txt', header=0, sep=' ')
else:
    lfw = pd.read_csv(dataset_path + 'DevTest' + os.sep + 'DevTest_cleaned_backup.txt', header=0, sep=' ')
lfw_id_pose = (lfw['id'] + '_' + lfw['pose'].astype(str)).values
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
test_idx_anchor = np.append(test_idx_anchor, searchIdx(my_queryString, lfw_id_pose))
my_queryString = (pairsDevTest_POS_2['name'] + '_' + pairsDevTest_POS_2['imagenum'].astype(str)).values
test_idx_compare = np.append(test_idx_compare, searchIdx(my_queryString, lfw_id_pose))

# Query NEG
my_queryString = (pairsDevTest_NEG_1['name'] + '_' + pairsDevTest_NEG_1['imagenum'].astype(str)).values
test_idx_anchor = np.append(test_idx_anchor, searchIdx(my_queryString, lfw_id_pose))
my_queryString = (pairsDevTest_NEG_2['name'] + '_' + pairsDevTest_NEG_2['imagenum'].astype(str)).values
test_idx_compare = np.append(test_idx_compare, searchIdx(my_queryString, lfw_id_pose))

test_idx_anchor = test_idx_anchor.astype(int)
test_idx_compare = test_idx_compare.astype(int)

# Assign
test_gender_anchor = np.append(test_gender_anchor, lfw.iloc[test_idx_anchor]['gender'])
test_ethnicity_anchor = np.append(test_ethnicity_anchor, lfw.iloc[test_idx_anchor]['ethnicity'])
test_id_anchor = np.append(test_id_anchor, lfw.iloc[test_idx_anchor]['id'])
test_pose_anchor = np.append(test_pose_anchor, lfw.iloc[test_idx_anchor]['pose']).astype(int)
test_filename_anchor = test_id_anchor + '_' + np.char.zfill(test_pose_anchor.astype(str), 4) + '.jpg'
test_feature_anchor = np.vstack((test_feature_anchor, lfw.iloc[test_idx_anchor].values[:,5:]))
test_race_anchor = test_gender_anchor + '-' + test_ethnicity_anchor
test_gender_compare = np.append(test_gender_compare, lfw.iloc[test_idx_compare]['gender'])
test_ethnicity_compare = np.append(test_ethnicity_compare, lfw.iloc[test_idx_compare]['ethnicity'])
test_id_compare = np.append(test_id_compare, lfw.iloc[test_idx_compare]['id'])
test_pose_compare = np.append(test_pose_compare, lfw.iloc[test_idx_compare]['pose']).astype(int)
test_filename_compare = test_id_compare + '_' + np.char.zfill(test_pose_compare.astype(str), 4) + '.jpg'
test_feature_compare = np.vstack((test_feature_compare, lfw.iloc[test_idx_compare].values[:,5:]))
test_race_compare = test_gender_compare + '-' + test_ethnicity_compare
test_id = test_id_anchor + '_' + test_pose_anchor.astype(str) + '-' + test_id_compare + '-' + test_pose_compare.astype(str)
test_valid_label = np.tile('POS', 500)
test_valid_label = np.append(test_valid_label, np.tile('NEG', 500))

# Predict race
predicted_race_anchor, predicted_race_compare, same_race = predict_race(test_feature_anchor, test_feature_compare, test_race_anchor, test_race_compare, race_classify_mode)
# my_util.cal_accuracy(test_race_anchor, predicted_race_anchor)
# my_util.cal_accuracy(test_race_compare, predicted_race_compare)
# my_util.cal_accuracy(np.append(test_race_anchor, test_race_compare) , np.append(predicted_race_anchor, predicted_race_compare))
# np.where(~(test_race_anchor==predicted_race_anchor))[0]
# np.where(~(test_race_compare==predicted_race_compare))[0]

# Extract triplet feature
if eval_mode == 'baseline' or race_classify_mode == 'none':
    test_exacted_feature_anchor = preprocessing.normalize(test_feature_anchor, norm='l2', axis=1, copy=True, return_norm=False)
    test_exacted_feature_compare = preprocessing.normalize(test_feature_compare, norm='l2', axis=1, copy=True, return_norm=False)
else:
    test_exacted_feature_anchor = np.empty((test_race_anchor.size, proposed_model_feature_size))
    test_exacted_feature_compare = np.empty((test_race_compare.size, proposed_model_feature_size))
    for class_val in tqdm(triplet_param['class']):
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
tmp_race_accuracy_all = np.empty((0, len(train_class)))
# tmp_auc_all = np.empty(0)
# tmp_accuracy_race = np.empty((0, len(train_class)))
# tmp_auc_race = np.empty((0, len(train_class)))
incorrect_list = np.empty(0)
incorrect_true_race = np.empty(0)
incorrect_prediceted_race = np.empty(0)
incorrect_true_label = np.empty(0)
incorrect_predicted_label = np.empty(0)

# Run experiment
for exp_numb in run_exp_round:
    exp_name_seed = exp_name + '_run_' + str(exp_numb)
    # Load model
    model = {}
    selm_weight_idx = np.empty(0, dtype=int)
    for train_class_idx, train_class_val in enumerate(train_class):
        if eval_mode == 'baseline':
            tmp_exp_name = classifier_exp + '_alg_BaselineEuclideanOneThreshold_' + train_class_val
        elif eval_mode == 'selm':
            if allinone == '':  # load race model
                tmp_exp_name = classifier_exp + '_alg_selmEuclid' + classifier_model_rule[train_class_idx] + 'POS_' + train_class_val
            else:   # load AllInOne model
                if balanced_triplet:
                    allinone_prefix = 'POS_dfbt_AllInOne_'
                else:
                    allinone_prefix = 'POS_AllInOne_'
                tmp_exp_name = classifier_exp + '_alg_selmEuclid' + classifier_model_rule[train_class_idx] + allinone_prefix + allinone
                
        model_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', classifier_exp, tmp_exp_name])
        model[train_class_val] = my_util.load_numpy_file(model_path + tmp_exp_name + '_run_' + str(exp_numb) + '.npy')
        if eval_mode == 'selm':
            selm_weight_idx = np.unique(np.append(selm_weight_idx, np.unique(np.array(pd.DataFrame(model[train_class_val]['weightID'])[0].str.split('-').values.tolist()).astype(int))))
    del tmp_exp_name, train_class_idx, train_class_val, model_path
    
    # Prepare weights for SELM
    if eval_mode == 'selm':
        # selm_weight_idx = selm_weight_idx - 1
        if race_classify_mode == 'none':    # resnet50 feature
            selm_weight = diveface.loc[selm_weight_idx-1].values[:,-feature_size:]
            selm_weight = preprocessing.normalize(selm_weight, norm='l2', axis=1, copy=True, return_norm=False)
        else:                               # gender-ethnicity feature
            selm_weight = np.empty((selm_weight_idx.size, proposed_model_feature_size))
            # extract feature
            for train_class_idx, train_class_val in enumerate(train_class):
                tmp_selm_weight_idx_race = np.where(diveface_race[selm_weight_idx-1] == train_class_val)[0]
                tmp_selm_weight_idx = selm_weight_idx[tmp_selm_weight_idx_race]
                tmp_selm_weight = diveface.loc[tmp_selm_weight_idx-1].values[:,-feature_size:]
                tmp_selm_weight = triplet_model[train_class_val].predict(tmp_selm_weight.astype(np.float64))
                selm_weight[tmp_selm_weight_idx_race] = tmp_selm_weight
                del tmp_selm_weight_idx_race, tmp_selm_weight_idx, tmp_selm_weight
            del train_class_idx, train_class_val
        selm_weight = pd.DataFrame(selm_weight)
        selm_weight.index = selm_weight_idx
    
    # Evaluate all
    performance_metric, predictedY, correct_list = evaluate(test_exacted_feature_anchor, test_exacted_feature_compare, test_race_anchor, test_race_compare, predicted_race_anchor, predicted_race_compare, test_valid_label, same_race, unique_class, model, eval_mode)
    # Append score
    tmp_race_performance_metric = np.empty(0)
    for tc_idx in train_class:
        tmp_idx = (test_race_anchor == tc_idx) + (test_race_compare == tc_idx)
        tmp_race_performance_metric = np.append(tmp_race_performance_metric, correct_list[tmp_idx].sum()/len(correct_list[tmp_idx]))
    tmp_race_accuracy_all = np.vstack((tmp_race_accuracy_all, tmp_race_performance_metric))
    del tmp_idx, tmp_race_performance_metric
    tmp_accuracy_all = np.append(tmp_accuracy_all, performance_metric['accuracy'])
    # Assign incorrect predicted list
    incorrect_list = np.append(incorrect_list, test_filename_anchor[~correct_list] + '--' + test_filename_compare[~correct_list])
    incorrect_true_race = np.append(incorrect_true_race, test_race_anchor[~correct_list] + '--' + test_race_compare[~correct_list])
    incorrect_prediceted_race = np.append(incorrect_prediceted_race, predicted_race_anchor[~correct_list] + '--' + predicted_race_compare[~correct_list])
    incorrect_true_label = np.append(incorrect_true_label, test_valid_label[~correct_list])
    incorrect_predicted_label = np.append(incorrect_predicted_label, predictedY[~correct_list])
    
    predicted_label = np.empty((0,4))
    predicted_label = np.vstack((predicted_label, ['true_label', 'predicted_label', 'race_anchor', 'race_compare']))
    predicted_label = np.vstack((predicted_label,np.concatenate((test_valid_label[:,None],predictedY[:,None],test_race_anchor[:,None],test_race_compare[:,None]),axis=1)))
    
    my_util.save_numpy(predicted_label, summary_path, exp_name_seed+'_label', doSilent=True)
    my_util.save_numpy(np.vstack((train_class, tmp_race_accuracy_all)), summary_path, exp_name_seed, doSilent=True)
    
    print('Finished ' + exp_name_seed)
    
    if eval_mode == 'selm':
        del selm_weight, selm_weight_idx
    del performance_metric, correct_list

print(train_class)
print(np.vstack((train_class, tmp_race_accuracy_all)))
print(np.mean(tmp_race_accuracy_all, axis=1))
print(np.mean(tmp_race_accuracy_all))
# print(tmp_accuracy_all)
# print(np.mean(tmp_accuracy_all))


# Save incorrect_list
save_incorrect_list = np.unique(incorrect_list[:,None] + '--' + incorrect_true_race[:,None] + '--' + incorrect_prediceted_race[:,None] + '--' + incorrect_true_label[:,None] + '--' + incorrect_predicted_label[:,None])
# np.savetxt('incorrect_list.txt', incorrect_list, delimiter=' ', fmt='%s')
# np.savetxt('incorrect_true_race.txt', incorrect_true_race, delimiter=' ', fmt='%s')
# np.savetxt('incorrect_prediceted_race.txt', incorrect_prediceted_race, delimiter=' ', fmt='%s')
# np.savetxt('save_incorrect_list_' + eval_mode + '-' + allinone + '_' + race_classify_mode + '.txt', save_incorrect_list, delimiter=' ', fmt='%s')

print()
