
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
import cv2
import os
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import my own lib
import others.utilities as my_util

#############################################################################################

exp = 'exp_7'
exp_name = exp + '_alg_tl'
dataset_exacted = 'resnet50'
exp_name = exp_name + dataset_exacted

# exp_name_suffix = [ 'b_240_e_30_a_1', 'b_240_e_30_a_5', 'b_30_e_30_a_30', 'b_60_e_30_a_30', 'b_90_e_30_a_30', 'b_120_e_30_a_30', 'b_150_e_30_a_30', 'b_60_e_30_a_90', 'b_90_e_30_a_90', 'b_120_e_30_a_90', 'b_150_e_30_a_90', 'b_270_e_30_a_10', 'b_300_e_30_a_10', 'b_600_e_40_a_1', 'b_90_e_50_a_10', 'b_120_e_50_a_10', 'b_150_e_50_a_10', 'b_150_e_50_a_1', 'b_180_e_50_a_10', 'b_210_e_50_a_10', 'b_240_e_50_a_10', 'b_270_e_100_a_1', 'b_300_e_100_a_1' ]
exp_name_suffix = ['b_30_e_50_a_1', 'b_60_e_50_a_1', 'b_90_e_50_a_1', 'b_120_e_50_a_1', 'b_150_e_50_a_1', 'b_180_e_50_a_1', 'b_210_e_50_a_1', 'b_240_e_50_a_1', 'b_270_e_50_a_1', 'b_300_e_50_a_1', 'b_330_e_50_a_1', 'b_360_e_50_a_1']

exact_eval_set = ['training', 'valid', 'test']
eval_class = 'overall' # overall female male female-asian female-black female-caucasian male-asian male-black male-caucasian

random_seed = 0

#############################################################################################

# Path
# Summary path
summary_path = my_util.get_path(additional_path=['.', 'FaceRecognitionPython_data_store', 'Result', 'summary', exp])

#############################################################################################

# Function
def initNaNVector(_vec_size):
    _tmp_vec = np.empty(_vec_size)
    _tmp_vec[:] = np.NaN
    return _tmp_vec

def plotLines(_metric, _exact_eval_set, _scores, _legends, _name):
    epoch_numb = np.arange(_scores[_exact_eval_set][_metric].shape[1])
    # Color
    cmap = plt.get_cmap('jet') # gist_ncar gnuplot jet
    colors = [cmap(i) for i in np.linspace(0, 1, len(_legends))]
    linewidth = 0.5
    plt.figure()
    # Plot baseline
    plt.plot(epoch_numb, np.tile(baseline[_metric], epoch_numb.size), '--', color='gray', linewidth=linewidth, label='Baseline')
    # Plot lines
    for _legends_idx, _legends_val in enumerate(_legends):
        plt.plot(epoch_numb, _scores[_exact_eval_set][_metric][_legends_idx,:], color=colors[_legends_idx], linewidth=linewidth, label=_legends_val)
    # Mark best
    if _metric == 'eer':
        best_score = np.amin(_scores[_exact_eval_set][_metric])
        txt_dist = -5
    else:
        best_score = np.amax(_scores[_exact_eval_set][_metric])
        txt_dist = 5
    best_coor = np.where(best_score == _scores[_exact_eval_set][_metric])
    plt.annotate(epoch_numb[best_coor[1]][0], (epoch_numb[best_coor[1]][0],best_score), textcoords="offset points", color=colors[best_coor[0][0]], xytext=(0,txt_dist), ha='center', va='center')
    plt.xlabel('EPOCH')
    # plt.ylabel('')
    plt.title(_name)
    plt.legend(loc='best', prop={'size': 5})
    # plt.show()
    plt.savefig(summary_path + exp + '_' + _name + '(' + _exact_eval_set + '-' + eval_class + ').pdf', format='pdf', transparent=True)
    # plt.savefig(summary_path + exp + '_' + _name + '(' + eval_class + ').eps', format='eps', transparent=True)
    # plt.savefig(summary_path + exp + '_' + _name + '(' + eval_class + ').svg', format='svg')
    pass

#############################################################################################

# Baseline
if dataset_exacted == 'resnet50':
    baseline = {'auc':0.9769784915123455, 'eer':3.0694444444444446, 'tar_0':92.16666666666666, 'tar_0d01':92.16666666666666, 'tar_0d1':94.31944444444444, 'tar_1':97.35416666666666}

scores = {}
for exact_eval_set_idx in exact_eval_set:
    scores[exact_eval_set_idx] = {}
    for exp_idx, exp_val in enumerate(exp_name_suffix):
        # Prepare data
        full_exp_name = exp_name + '_' + exp_val + '_run_' + str(random_seed)
        file_path = summary_path + full_exp_name + os.sep + full_exp_name + '(' + exact_eval_set_idx + ')' + '.pickle'
        data = pickle.load(open(file_path, 'rb'))
        key_value = np.array(list(data[exact_eval_set_idx].keys()))
        tmp_auc = initNaNVector(key_value.max()+1)
        tmp_eer = initNaNVector(key_value.max()+1)
        tmp_tar_0 = initNaNVector(key_value.max()+1)
        tmp_tar_0d01 = initNaNVector(key_value.max()+1)
        tmp_tar_0d1 = initNaNVector(key_value.max()+1)
        tmp_tar_1 = initNaNVector(key_value.max()+1)
        # Exact scores
        for key_value_idx in key_value:
            tmp_auc[key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class]['auc']
            tmp_eer[key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class]['eer']
            tmp_tar_0[key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class]['tar_0']
            tmp_tar_0d01[key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class]['tar_0d01']
            tmp_tar_0d1[key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class]['tar_0d1']
            tmp_tar_1[key_value_idx] = data[exact_eval_set_idx][key_value_idx][eval_class]['tar_1']
        # Append scores
        if exp_idx == 0:
            scores[exact_eval_set_idx]['auc'] = tmp_auc[None,:]
            scores[exact_eval_set_idx]['eer'] = tmp_eer[None,:]
            scores[exact_eval_set_idx]['tar_0'] = tmp_tar_0[None,:]
            scores[exact_eval_set_idx]['tar_0d01'] = tmp_tar_0d01[None,:]
            scores[exact_eval_set_idx]['tar_0d1'] = tmp_tar_0d1[None,:]
            scores[exact_eval_set_idx]['tar_1'] = tmp_tar_1[None,:]
        else:
            if scores[exact_eval_set_idx]['auc'].shape[1] < tmp_auc.size:
                add_col_size = tmp_auc.size - scores[exact_eval_set_idx]['auc'].shape[1]
                scores[exact_eval_set_idx]['auc'] = np.pad(scores[exact_eval_set_idx]['auc'], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
                scores[exact_eval_set_idx]['eer'] = np.pad(scores[exact_eval_set_idx]['eer'], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
                scores[exact_eval_set_idx]['tar_0'] = np.pad(scores[exact_eval_set_idx]['tar_0'], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
                scores[exact_eval_set_idx]['tar_0d01'] = np.pad(scores[exact_eval_set_idx]['tar_0d01'], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
                scores[exact_eval_set_idx]['tar_0d1'] = np.pad(scores[exact_eval_set_idx]['tar_0d1'], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
                scores[exact_eval_set_idx]['tar_1'] = np.pad(scores[exact_eval_set_idx]['tar_1'], ((0,0),(0,add_col_size)), 'constant', constant_values=np.nan)
            scores[exact_eval_set_idx]['auc'] = np.vstack((scores[exact_eval_set_idx]['auc'], tmp_auc))
            scores[exact_eval_set_idx]['eer'] = np.vstack((scores[exact_eval_set_idx]['eer'], tmp_eer))
            scores[exact_eval_set_idx]['tar_0'] = np.vstack((scores[exact_eval_set_idx]['tar_0'], tmp_tar_0))
            scores[exact_eval_set_idx]['tar_0d01'] = np.vstack((scores[exact_eval_set_idx]['tar_0d01'], tmp_tar_0d01))
            scores[exact_eval_set_idx]['tar_0d1'] = np.vstack((scores[exact_eval_set_idx]['tar_0d1'], tmp_tar_0d1))
            scores[exact_eval_set_idx]['tar_1'] = np.vstack((scores[exact_eval_set_idx]['tar_1'], tmp_tar_1))

# scores['auc'].max()
# scores['eer'].min()
# scores['tar_1'].max()
# scores['tar_0d1'].max()
# scores['tar_0d01'].max()
# scores['tar_1'].max()

# Plot
for exact_eval_set_idx in exact_eval_set:
    plotLines('auc', exact_eval_set_idx, scores, exp_name_suffix, 'AUC')
    plotLines('eer', exact_eval_set_idx, scores, exp_name_suffix, 'EER')
    plotLines('tar_0d01', exact_eval_set_idx, scores, exp_name_suffix, 'TAR 0.1')
    plotLines('tar_0d1', exact_eval_set_idx, scores, exp_name_suffix, 'TAR 0.01')

print()


