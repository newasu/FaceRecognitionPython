
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import lib
import pandas as pd
import numpy as np
from tqdm import tqdm
# Parallel
import multiprocessing
from joblib import Parallel, delayed

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Path
# Dataset path
dataset_name = 'Diveface'
dataset_path = my_util.get_current_path(additional_path=['FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
dataset_path = dataset_path + 'Diveface_retinaface.txt'

# Load label
my_data = pd.read_csv(dataset_path, sep=" ", header=0)
my_label = pd.concat([(my_data['gender'] + '-' + my_data['ethnicity']), my_data['filename']], axis=1)
my_label.columns = ['label', 'id']
my_unique_labels = np.unique(my_label['label'])
del my_data

#############################################################################################

def eval_result(data_id, data_score):
    tmp_data = pd.DataFrame(data_id)
    tmp = tmp_data[0].str.split('-')
    
    # Loopy Find label and score
    # test_labels = pd.DataFrame(columns=['fn0', 'fn1', 'score']) 
    # for tmp_idx in range(0, tmp.size):
    #     test_labels = test_labels.append(do_eval_result(data_score, tmp, tmp_idx), ignore_index=True)
    
    tmp_0 = find_first_index_each_row(tmp.str[0])
    tmp_1 = find_first_index_each_row(tmp.str[1])
    test_labels = pd.DataFrame({'fn0':tmp_0, 'fn1':tmp_1, 'score':data_score})
        
    return test_labels

def find_first_index_each_row(tmp_data_id):
    print('Indexing..')
    mt = tmp_data_id.values[:, None] == my_label['id'].values
    tmptmp = []
    for row_idx in tqdm(range(0, mt.shape[0])):
        tmptmp.append(my_label.loc[np.where(mt[row_idx,:])[0][0]]['label'])
        # print(str(row_idx+1) + '/' + str(mt.shape[0]))
    return tmptmp

def do_eval_result(ds, ttmp, ttmp_idx):
    tmp_score = ds[ttmp_idx]
    query_idx = np.where(my_label['id']==ttmp[ttmp_idx][0])[0][0]
    tmp_1 = my_label.loc[query_idx]['label']
    query_idx = np.where(my_label['id']==ttmp[ttmp_idx][1])[0][0]
    tmp_2 = my_label.loc[query_idx]['label']
    print(str(ttmp_idx+1) + '/' + str(ttmp.size))
    return {'fn0':tmp_1, 'fn1':tmp_2, 'score':tmp_score}

def eval_score(my_unique_label, score_mat, score_order='ascending'):
    print('evaluating score..')
    eval_score_dict = {}
    for tmp_unique_labels in tqdm(my_unique_label):
        tmp_idx = score_mat[['fn0', 'fn1']] == tmp_unique_labels
        tmp_idx = (tmp_idx['fn0'] | tmp_idx['fn1']).values
        tmp_score = score_mat['score'][tmp_idx].values
        eval_score_dict[tmp_unique_labels] = my_util.biometric_metric(score_mat['fn0'][tmp_idx].values, tmp_score, tmp_unique_labels, score_order=score_order)
    return eval_score_dict

def exact_result_all_seed(data, my_unique_label, seed, score_order='ascending'):
    result_score = {}
    eval_score_dict = {}
    for seed_idx in seed:
        print('Exacting seed ' + str(seed_idx))
        result_score[str(seed_idx)] = eval_result(data['test_image_id'][seed_idx], data['predictedScores'][seed_idx])
        eval_score_dict[str(seed_idx)] = eval_score(my_unique_label, result_score[str(seed_idx)], score_order=score_order)
    return result_score, eval_score_dict
    
#############################################################################################

# Load exp result
# tmp = my_util.load_numpy_file((exp_result_path + 'exp_3_alg_euclid/exp_3_alg_euclid_run_0.npy'))
exact_list = ['test_image_id', 'predictedScores']
exp_result_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/exp_result/'

seed = [0, 1, 2, 3, 4]

# Prepare data
# Euclidean
filename = 'exp_3_alg_euclid'
data = my_util.exact_run_result_in_directory((exp_result_path + filename + '/'), exact_list)
[result_score, eval_score_dict] = exact_result_all_seed(data, my_unique_labels, seed, score_order='descending')

# result_score = eval_result(data['test_image_id'][seed], data['predictedScores'][seed])
# eval_score_dict = eval_score(my_unique_labels, result_score, score_order='descending')

# concat_test_image_id = np.concatenate((data['test_image_id'][0], data['test_image_id'][1], data['test_image_id'][2], data['test_image_id'][3], data['test_image_id'][4]))
# concat_predictedScores = np.concatenate((data['predictedScores'][0], data['predictedScores'][1], data['predictedScores'][2], data['predictedScores'][3], data['predictedScores'][4]))
# result_score = eval_result(concat_test_image_id, concat_predictedScores)
# eval_score_dict = eval_score(my_unique_labels, result_score, score_order='descending')



print()
