# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker

# Import my own lib
import others.utilities as my_util

#############################################################################################

result_exp = 'exp_12'
# result_alg = ['baseline-_none', 'selm-_auto', 'selm-rd_0d2_vl_df_none', 'selm-rd_0d5_vl_df_none', 'selm-rd_0d6_vl_df_none', 'selm-rd_0d67_vl_df_none', 'selm-rd_0_vl_df_none_dfbt', 'selm-rd_0d2_vl_df_none_dfbt', 'selm-rd_0d5_vl_df_none_dfbt']

result_alg = ['baseline-_none', 'selm-_auto', 'selm-rd_0d5_vl_df_none']
method_list = ['ResNet', 'SELM$_{Sum}^{GED}$', 'SELM$_{Sum}^{SI}$']

random_seed = 0

#############################################################################################

# Path
# Summary path
summary_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'summary', result_exp])

#############################################################################################

def cal_conmat(tl,pl):
    TN, FP, FN, TP = confusion_matrix(tl, pl, labels=['NEG', 'POS']).ravel()
    _FPR = (FP/(TN+FP)) * 100
    _FNR = (FN/(TP+FN)) * 100
    return _FPR, _FNR

#############################################################################################

score_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
plot_score_class = ['Female\nAsian', 'Female\nBlack', 'Female\nCaucasian', 'Male\nAsian', 'Male\nBlack', 'Male\nCaucasian']
scores = {}
labels = {}
FAR = {}
FRR = {}

# Load result
for rm in result_alg:
    tmp_fname = result_exp + '_framework_' + rm
    tmp_score = np.load(summary_path + tmp_fname + os.sep + tmp_fname + '_run_' + str(random_seed) + '.npy', allow_pickle=True)
    tmp_label = np.load(summary_path + tmp_fname + os.sep + tmp_fname + '_run_' + str(random_seed) + '_label.npy', allow_pickle=True)
    
    tmp_score = pd.DataFrame(tmp_score)
    tmp_score.columns = tmp_score.iloc[0]
    tmp_score = tmp_score[1:]
    
    tmp_label = pd.DataFrame(tmp_label)
    tmp_label.columns = tmp_label.iloc[0]
    tmp_label = tmp_label[1:]
    
    tmp_read_score = []
    for sc in score_class:
        tmp_read_score.append(np.float64(tmp_score[sc].values[0]))
    
    labels[rm] = tmp_label
    scores[rm] = tmp_read_score
    
    FAR[rm], FRR[rm] = cal_conmat(tmp_label.true_label.values, tmp_label.predicted_label.values)
    
del rm, sc, tmp_fname, tmp_score, tmp_label, tmp_read_score

scores = pd.DataFrame(scores).T
scores.columns = score_class
scores = scores * 100
# scores.mean(axis=1)

conmat = pd.DataFrame([FAR,FRR]).T
conmat.columns = ['FAR', 'FRR']
del FAR, FRR

ranks = scores.rank(axis=0, method='average', ascending=False)
sum_rank = ranks.sum(axis=1)

print('Sum of rank:')
print(sum_rank)

print('False Accept Rate:')
print(conmat.FAR)

print('False Reject Rate:')
print(conmat.FRR)

# Plot grouped bar

# Plot framework_perf
plt.rcParams['font.serif'] = 'Times New Roman'
fig = plt.figure() 
ax = plt.subplot(111)
# patterns = [ "|" , "\\" , "/" , "+" , "-", ".", "*","x", "o", "O" ]
patterns = ['', '/', '\\']
label_loc = np.arange(len(score_class))  # the label locations
width = 0.24  # the width of the bars

if (len(label_loc) % 2) == 0:
    x_pos = label_loc - ((len(result_alg)//2) * width)
else:
    x_pos = label_loc - (((len(result_alg)/2) * width) - (width/2))

rects = {}
for rl_idx, rl_val in enumerate(result_alg):
    tmp_plot_score = []
    for em in score_class:
        tmp_plot_score.append(scores[em][rl_val])
    rects[rl_val] = ax.bar(x_pos, tmp_plot_score, width, label=method_list[rl_idx], color='white', edgecolor='black', hatch=patterns[rl_idx])
    x_pos = x_pos + width
    
    # Text over bar
    for rect_idx, rect_val in enumerate(rects[rl_val]):
        tmp_plot_score = ranks[score_class[rect_idx]][rl_val]
        if tmp_plot_score.is_integer():
            tmp_plot_score = tmp_plot_score.astype(int)
        ax.annotate('{}'.format(tmp_plot_score), xy=(rect_val.get_x() + rect_val.get_width() / 2, rect_val.get_height()), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, rotation=0)
    del tmp_plot_score

# Set limit
ax.set_ylim([87, 103])
# Set axis label
ax.set_xticks(label_loc)
ax.set_xticklabels(plot_score_class, fontsize=13, fontname='Times New Roman')
ax.set_ylabel('Accuracy', fontsize=14, fontname='Times New Roman')
# Set legend
my_legend = plt.legend(prop={'family':'Times New Roman', 'size':13}, ncol=3, loc='upper right', framealpha=0, fancybox=True, shadow=False)
# Grid
ax.set_axisbelow(True)
ax.grid(b=True, which='major', axis='both', linestyle='-', color='gray', alpha=0.35)
ax.minorticks_on()
ax.grid(b=True, which='minor', axis='both', linestyle='-', color='#999999', alpha=0.15)
# Save
fig.savefig('framework_perf.png')
fig.savefig('framework_perf.pdf')



# Plot framework_perf

plot_col = ['False Accept Rate', 'False Reject Rate']

plt.rcParams['font.serif'] = 'Times New Roman'
fig = plt.figure() 
ax = plt.subplot(111)
# patterns = [ "|" , "\\" , "/" , "+" , "-", ".", "*","x", "o", "O" ]
patterns = ['', '/', '\\']
label_loc = np.arange(conmat.shape[1])  # the label locations
width = 0.25  # the width of the bars

if (len(label_loc) % 2) == 0:
    x_pos = label_loc - ((len(result_alg)//2) * width)
else:
    x_pos = label_loc - (((len(result_alg)/2) * width) - (width/2))

rects = {}
for rl_idx, rl_val in enumerate(result_alg):
    tmp_plot_score = []
    for em in conmat.columns:
        tmp_plot_score.append(conmat[em][rl_val])
    rects[rl_val] = ax.bar(x_pos, tmp_plot_score, width, label=method_list[rl_idx], color='white', edgecolor='black', hatch=patterns[rl_idx])
    x_pos = x_pos + width
    
    # Text over bar
    for rect_idx, rect_val in enumerate(rects[rl_val]):
        tmp_plot_score = rect_val.get_height()
        tmp_plot_score = np.round(tmp_plot_score, 2)
        if tmp_plot_score.is_integer():
            tmp_plot_score = tmp_plot_score.astype(int)
        ax.annotate('{}'.format(tmp_plot_score), xy=(rect_val.get_x() + rect_val.get_width() / 2, rect_val.get_height()), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=10, rotation=0)
    del tmp_plot_score

# Set limit
ax.set_ylim([0, 12])
# Set axis label
ax.set_xticks(label_loc)
ax.set_xticklabels(plot_col, fontsize=13, fontname='Times New Roman')
ax.set_ylabel('%', fontsize=14, fontname='Times New Roman')
# Set legend
my_legend = plt.legend(prop={'family':'Times New Roman', 'size':13}, ncol=1, loc='upper left', framealpha=0, fancybox=True, shadow=False)
# Grid
ax.set_axisbelow(True)
ax.grid(b=True, which='major', axis='both', linestyle='-', color='gray', alpha=0.35)
ax.minorticks_on()
ax.grid(b=True, which='minor', axis='both', linestyle='-', color='#999999', alpha=0.15)
# Save
fig.savefig('framework_conmat.png')
fig.savefig('framework_conmat.pdf')

print()
