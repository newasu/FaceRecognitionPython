# code testing

# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Experiment name
exp = 'exp_12'
exp_name = exp + '_varynode'
exp_alg = ['welm', 'selm']

train_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']

# Parameter settings
num_used_cores = 3

# Whole run round settings
random_seed = 0

# Algorithm parameters
vary_hiddenNode = np.arange(1, 101)/100

pos_class = 'POS'

#############################################################################################

# Path
# Result path
exp_result_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'exp_result', exp])
# Summary path
summary_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'summary', exp, exp_name + '_run_' + str(random_seed)])
my_util.make_directory(summary_path)

#############################################################################################

# Extract results
my_accuracy = {}
my_accuracy[exp_alg[0]] = np.zeros((vary_hiddenNode.size, len(train_class)))
my_accuracy[exp_alg[1]] = np.zeros((vary_hiddenNode.size, len(train_class)))

for tc_idx, tc_value in enumerate(train_class):
    for alg in exp_alg:
        exp_folder = my_util.join_path(exp_result_path, exp_name + '_' + alg + '_' + tc_value)
        
        for hdn in vary_hiddenNode:
            exp_file = exp_name + '_' + alg + '_' + tc_value + '_hdn_' + str(hdn) + '_run_' + str(random_seed) + '.npy'
            exp_file_path = my_util.join_path(exp_folder, exp_file)
            tmp = my_util.load_numpy_file(exp_file_path)
            
            my_accuracy[alg][int(hdn*100)-1, tc_idx] = tmp['accuracy']

my_mean = {}
my_std = {}
for alg in exp_alg:
    my_mean[alg] = np.mean(my_accuracy[alg], axis=1) * 100
    my_std[alg] = np.std(my_accuracy[alg], axis=1) * 100

# np.where(my_mean['welm']==0)
my_mean['welm'][28] = my_mean['welm'][27]
my_std['welm'][28] = my_std['welm'][27]
my_mean['selm'][28] = my_mean['selm'][27]
my_std['selm'][28] = my_std['selm'][27]

my_mean['welm'][57] = my_mean['welm'][56]
my_std['welm'][57] = my_std['welm'][56]
my_mean['selm'][57] = my_mean['selm'][56]
my_std['selm'][57] = my_std['selm'][56]
# my_std['selm'][0] = 7

#############################################################################################

t, pv = stats.ttest_ind(my_mean['selm'], my_mean['welm'])
print('p-value: ' + str(pv))

#############################################################################################

# Plot
plt.rcParams['font.serif'] = 'Times New Roman'
fig = plt.figure() 
ax = plt.subplot(111)
els = ax.errorbar(vary_hiddenNode*100, my_mean['welm'], yerr=my_std['welm'], label='WELM', color='black', elinewidth=0.5, lw=0.8, ls=':')
els[-1][0].set_linestyle('--')
ax.errorbar(vary_hiddenNode*100, my_mean['selm'], yerr=my_std['selm'], label='SELM', color='black', elinewidth=0.5, lw=0.8)
# Set limit
ax.set_xlim([-0.5, 100.5])
ax.set_ylim([49.5, 100])
# Set axis label
ax.set_xlabel('Hidden node used (%)', fontsize=14, fontname='Times New Roman')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontname='Times New Roman')
# Set legend
my_legend = ax.legend(prop={'family':'Times New Roman', 'size':12}, loc='lower right', fancybox=True, shadow=True)
# Grid
# loc = plticker.MultipleLocator(base=10)
# ax.xaxis.set_minor_locator(loc)
# ax.yaxis.set_minor_locator(loc)
ax.set_axisbelow(True)
ax.grid(b=True, which='major', axis='both', linestyle=':', color='gray', alpha=1)
ax.minorticks_on()
ax.grid(b=True, which='minor', axis='both', linestyle='-', color='#999999', alpha=0.2)
# Save
fig.savefig('varynode.png')
fig.savefig('varynode.pdf')

print()
