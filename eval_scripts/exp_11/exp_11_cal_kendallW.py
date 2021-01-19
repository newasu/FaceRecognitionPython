
# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
# import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Experiment name
exp = 'exp_11'
exp_name = exp + '_alg_'
methods = ['BaselineEuclideanOneThreshold', 'elmRBFPOS_2', 'selmEuclidSumPOS', 'selmEuclidDistPOS', 'selmEuclidMultiplyPOS', 'selmEuclidMeanPOS']
method_names = ['ResNet', 'ELM', 'SELM$_{Sum}$', 'SELM$_{Dist}$', 'SELM$_{Mult}$', 'SELM$_{Mean}$']

# methods = ['BaselineEuclideanOneThreshold', 'GenderEuclideanOneThreshold', 'RaceEuclidean']
# method_names = ['ID', 'GD', 'GED']

class_model = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
metric = ['auc', 'eer', 'accuracy', 'tar_1', 'tar_0d1', 'tar_0d01']

eval_metric = 'auc'

run_exp_round = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

#############################################################################################

# Path
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp])

#############################################################################################

# Extract scores
scores = {}
for method_idx, method_val in enumerate(methods):
    for run_exp_round_idx, run_exp_round_val in enumerate(run_exp_round):  
        add_idx = np.arange(0,len(class_model)) + (len(class_model) * run_exp_round_idx)
        for class_model_idx, class_model_val in enumerate(class_model):
            data_folder = exp_name + method_val + '_' + class_model_val
            data = exp_result_path + data_folder + os.sep + data_folder + '_run_' + str(run_exp_round_val) + '.npy'
            data = my_util.load_numpy_file(data)
            for metric_val in metric:
                if method_idx == 0 and run_exp_round_idx == 0 and class_model_idx == 0:
                    scores[metric_val] = np.zeros((len(methods), len(run_exp_round) * len(class_model)))
                scores[metric_val][method_idx,add_idx[class_model_idx]] = data[metric_val]
                # print(metric_val + ': ' + str(scores[metric_val][method_idx]))

rankOrder = {}
for metric_val in metric:
    if metric_val == 'eer':
        rankOrder[metric_val] =  stats.rankdata(scores[metric_val], method='average', axis=0)
    else:
        rankOrder[metric_val] =  stats.rankdata(-scores[metric_val], method='average', axis=0)

# score_exp = {}
rankOrder_exp = {}
rankOrder_sum_exp = {}
rankOrder_total_exp = {}
for metric_val in metric:
    # score_exp[metric_val] = np.zeros((len(methods),len(run_exp_round)))
    rankOrder_exp[metric_val] = np.zeros((len(methods),len(run_exp_round)))
    rankOrder_sum_exp[metric_val] = np.zeros((len(methods),len(run_exp_round)))
    # tmp_score = 0
    tmp_rankOrder = 0
    for run_exp_round_idx, run_exp_round_val in enumerate(run_exp_round):
        tmp_idx = np.arange(0,len(class_model)) + (len(class_model) * run_exp_round_val)
        tmp_score = np.sum(scores[metric_val][:,tmp_idx], axis=1)
        rankOrder_sum_exp[metric_val][:,run_exp_round_idx] = np.sum(rankOrder[metric_val][:,tmp_idx], axis=1)
        if metric_val == 'eer':
            rankOrder_exp[metric_val][:,run_exp_round_idx] =  stats.rankdata(tmp_score, method='average')
        else:
            rankOrder_exp[metric_val][:,run_exp_round_idx] =  stats.rankdata(-tmp_score, method='average')
    
    rankOrder_total_exp[metric_val] = np.sum(rankOrder_sum_exp[metric_val], axis=1)

#############################################################################################

# One-way ANOVA
# f, pv = stats.f_oneway(scores[eval_metric][0,:], scores[eval_metric][1,:], scores[eval_metric][2,:])
# Kendall's W
S, W, chi, cl = my_util.kendall_w(rankOrder[eval_metric].T)
print('Confidence level cal. from W: ' + str(cl))
print(stats.chi2.ppf(1-.005, df=rankOrder[eval_metric].shape[0]-1))
# stats.chi2.cdf(chi, rankOrder[eval_metric].shape[0]-1)

# compare selm sum vs mean
# rankOrder[eval_metric][2] # Sum
# rankOrder[eval_metric][5] # Mean

# max_idx = np.argmax([scores[eval_metric][2], scores[eval_metric][5]], axis=0)
# count = np.bincount(max_idx)
# element = np.nonzero(count)[0]
# np.vstack((element,count[element])).T

#############################################################################################

# Sort descending order
sort_idx = np.argsort(-rankOrder_total_exp[eval_metric])
rankOrder_total_exp[eval_metric] = rankOrder_total_exp[eval_metric][sort_idx]
rankOrder_sum_exp[eval_metric] = rankOrder_sum_exp[eval_metric][sort_idx,:]
method_names = np.array(method_names)[sort_idx]

# Plot ranking order
plt.rcParams['font.serif'] = 'Times New Roman'
color = cm.get_cmap('tab20c', len(run_exp_round))
fig = plt.figure() 
ax = plt.subplot(111)

width = 0.8
ind = np.arange(len(methods))
bottom = np.zeros(len(methods))
p = {}
p_legend = []
txt_legend = []
for run_exp_round_idx, run_exp_round_val in enumerate(run_exp_round):
    plot_val = rankOrder_sum_exp[eval_metric][:,run_exp_round_idx]
    p[run_exp_round_idx] = plt.bar(ind, plot_val, width, bottom=bottom, color=color.colors[run_exp_round_idx])
    bottom = bottom + plot_val
    p_legend.append(p[run_exp_round_idx][0])
    txt_legend.append('Exp. ' + str(run_exp_round_val+1))

# plot total
anno_p = {}
for p_idx, p_val in enumerate(p[0]):
    anno_p[p_idx] = plt.text(p_val.get_x()+p_val.get_width()/2., rankOrder_total_exp[eval_metric][p_idx], '%s'% rankOrder_total_exp[eval_metric][p_idx], ha='center', va='bottom')

# Set axis label
ax.set_xticks(ind)
ax.set_xticklabels(method_names, fontsize=12, fontname='Times New Roman')
ax.set_ylabel('Summation of Ranked Order', fontsize=13, fontname='Times New Roman')
# Set legend
my_legend = plt.legend(tuple(p_legend), tuple(txt_legend), prop={'family':'Times New Roman', 'size':12}, ncol=2, loc='upper right', framealpha=0, fancybox=True, shadow=False)
# Grid
ax.set_axisbelow(True)
ax.grid(b=True, which='major', axis='both', linestyle=':', color='gray', alpha=1)
# ax.minorticks_on()
# ax.grid(b=True, which='minor', axis='both', linestyle='-', color='#999999', alpha=0.2)
# Save
fig.savefig('rankingOrder.png')
fig.savefig('rankingOrder.pdf')



print()
