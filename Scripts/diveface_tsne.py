
# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm 
# import matplotlib.font_manager
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE

# Import my own lib
import others.utilities as my_util


#############################################################################################

# Path
# Dataset path
dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])

#############################################################################################

# Load DiveFace
# my_data = pd.read_csv((dataset_path + 'Diveface' + '_' + 'resnet50' + '_nonorm.txt'), sep=" ", header=0)
# x = my_data.iloc[:,8:].values
# y_gender = my_data['gender'].values
# y_ethnicity = my_data['ethnicity'].values
# y_label = y_gender + y_ethnicity
# y_int = pd.factorize(y_label)[0]
# y_cat = pd.Categorical(y_int)

# # Label
# y_label[y_label=='femaleasian'] = 'Female-Asian'
# y_label[y_label=='maleasian'] = 'Male-Asian'
# y_label[y_label=='femaleblack'] = 'Female-Black'
# y_label[y_label=='maleblack'] = 'Male-Black'
# y_label[y_label=='femalecaucasian'] = 'Female-Caucasian'
# y_label[y_label=='malecaucasian'] = 'Male-Caucasian'
y_label_unique = ['Female-Asian', 'Male-Asian', 'Female-Black', 'Male-Black', 'Female-Caucasian', 'Male-Caucasian']

# Calculate TSNE
# print('Calculating..')
# Single-core lib
# X_embedded = TSNE(n_components=2).fit_transform(x)
# Multicore lib
# X_embedded = TSNE(n_components=2, n_jobs=8).fit_transform(x)
# Save TSNE
# np.savetxt('x.csv', X_embedded, delimiter=',')
# np.savetxt('y.csv', y_int, delimiter=',')
# np.savetxt('y.txt', y_label, delimiter=',', fmt='%s')

# Load scatter data
x = np.loadtxt('x.csv', delimiter=',')
# y = np.loadtxt('y.csv', delimiter=',')
y_label = np.genfromtxt('y.txt', dtype='str')
# Plot TSNE
plt.rcParams['font.serif'] = 'Times New Roman'
color = cm.get_cmap('Set1', 9)
# color = cm.get_cmap('Set3', 12)
# color = cm.get_cmap('Paired', 6)
fig = plt.figure()
ax = plt.subplot(111)
for label_idx, label_val in enumerate(y_label_unique):
    tmp_idx = y_label == label_val
    ax.scatter(x[tmp_idx,0], x[tmp_idx,1], c=color.colors[None,label_idx], s=6, label=label_val, alpha=0.3, edgecolors='none')
# Set legend
my_legend = ax.legend(prop={'family':'Times New Roman', 'size':12}, loc='upper center', bbox_to_anchor=(0.5, 1.169), ncol=3, fancybox=True, shadow=True)
for handle in my_legend.legendHandles:
    handle.set_sizes([50])
    handle.set_alpha(1)
# Grid
ax.set_axisbelow(True)
ax.grid(linestyle=':', color='gray')
# Save plot
fig.savefig('diveface_tsne.png')
fig.savefig('diveface_tsne.pdf')
# fig.clf()

print()
