# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 
import matplotlib.ticker as plticker

# Import my own lib
import others.utilities as my_util

#############################################################################################

result_exp = 'exp_12'
result_alg = ['baseline-_none', 'selm-_auto', 'selm-rd_0d2_vl_df_none', 'selm-rd_0d5_vl_df_none', 'selm-rd_0d6_vl_df_none', 'selm-rd_0d67_vl_df_none']

random_seed = 0

#############################################################################################

# Path
# Summary path
summary_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Result', 'summary', result_exp])

#############################################################################################

score_class = ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']
scores = {}
labels = {}

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
del rm, sc, tmp_fname, tmp_score, tmp_label, tmp_read_score

scores = pd.DataFrame(scores).T
scores.columns = score_class
# scores.mean(axis=1)

ranks = scores.rank(axis=0, method='average', ascending=False)
sum_rank = ranks.sum(axis=1)

# Plot grouped bar


print()
