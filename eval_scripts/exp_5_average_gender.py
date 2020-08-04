
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

exp_result_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/exp_result/'
average_exp_result = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu-Mac/PhDs-Degree/New/SourceCode/FaceRecognitionPython_data_store/Result/average_exp_result/'

# Load label
my_data = pd.read_csv(dataset_path, sep=" ", header=0)
my_label = pd.concat([my_data['gender'], (my_data['gender'] + '-' + my_data['ethnicity']), my_data['data_id']], axis=1)
my_label.columns = ['gender', 'label', 'id']
my_unique_labels = np.unique(my_label['gender'])
del my_data

#############################################################################################

def do_eval_result(ds, ttmp, ttmp_idx):
    tmp_score = ds[ttmp_idx]
    query_idx = np.where(my_label['id']==ttmp[ttmp_idx][0])[0][0]
    tmp_1 = my_label.loc[query_idx]['gender']
    query_idx = np.where(my_label['id']==ttmp[ttmp_idx][1])[0][0]
    tmp_2 = my_label.loc[query_idx]['gender']
    print(str(ttmp_idx+1) + '/' + str(ttmp.size))
    return {'fn0':tmp_1, 'fn1':tmp_2, 'score':tmp_score}

def eval_score(my_unique_label, score_mat, score_order='descending'):
    print('evaluating score..')
    eval_score_dict = {}
    for tmp_unique_labels in tqdm(my_unique_label):
        tmp_idx = score_mat[['fn0', 'fn1']] == tmp_unique_labels
        tmp_idx = (tmp_idx['fn0'] | tmp_idx['fn1']).values
        tmp_score = score_mat['score'][tmp_idx].values
        eval_score_dict[tmp_unique_labels] = my_util.biometric_metric(score_mat['trueY'][tmp_idx].values, tmp_score, 'POS', score_order=score_order)
    return eval_score_dict

def eval_result(data_id, data_trueY, data_score, data_label_classes):
    tmp_data = pd.DataFrame(data_id)
    tmp = tmp_data[0].str.split('-')
    
    if data_score.ndim > 1:
        tmp_score = data_score[:, (data_label_classes=='POS')][:,0]
    else:
        tmp_score = data_score
    
    tmp_0 = tmp.str[0].astype(int) - 1
    tmp_0 = my_label.loc[tmp_0]['gender'].values
    tmp_1 = tmp.str[1].astype(int) - 1
    tmp_1 = my_label.loc[tmp_1]['gender'].values
    test_labels = pd.DataFrame({'fn0':tmp_0, 'fn1':tmp_1, 'trueY':data_trueY, 'score':tmp_score})
    return test_labels

def exact_result_all_seed(data, my_unique_label, seed, score_order='descending'):
    result_score = {}
    eval_score_dict = {}
    for seed_idx in seed:
        print('Exacting seed ' + str(seed_idx))
        result_score[str(seed_idx)] = eval_result(data['test_image_id'][seed_idx], data['trueY'][seed_idx], data['predictedScores'][seed_idx], data['label_classes'][seed_idx])
        eval_score_dict[str(seed_idx)] = eval_score(my_unique_label, result_score[str(seed_idx)], score_order=score_order)
        eval_score_dict[str(seed_idx)]['all'] = {'auc': data['auc'][seed_idx], 'eer': data['eer'][seed_idx], 'fmr_fnmr_thresh':data['fmr_fnmr_thresh'][seed_idx], 'fmr': data['fmr'][seed_idx], 'fnmr': data['fnmr'][seed_idx], 'fmr_0d1': data['fmr_0d1'][seed_idx], 'fmr_0d01': data['fmr_0d01'][seed_idx], 'fnmr_0d1': data['fnmr_0d1'][seed_idx], 'fnmr_0d01': data['fnmr_0d01'][seed_idx]}
    return result_score, eval_score_dict

#############################################################################################

# Load exp result
# tmp = my_util.load_numpy_file((exp_result_path + 'exp_5_alg_euclid/exp_5_alg_euclid_run_0.npy'))
exact_list = ['test_image_id', 'trueY', 'predictedScores', 'label_classes', 'auc', 'eer', 'fmr_0d1', 'fmr_0d01', 'fnmr_0d1', 'fnmr_0d01', 'fmr', 'fnmr', 'fmr_fnmr_thresh']


seed = [0, 1, 2, 3, 4]

# Prepare data
# filename = 'exp_5_alg_euclid'
# filename = 'exp_5_alg_selmEuclidDist'
# filename = 'exp_5_alg_selmEuclidDistPOS'
# filename = 'exp_5_alg_selmEuclidSum'
# filename = 'exp_5_alg_selmEuclidSumPOS'
# filename = 'exp_5_alg_selmRBFDist'
filename = 'exp_5_alg_selmRBFDistPOS'



save_folder = 'exp_5_avg_genders'
save_name = filename + '_avg_genders'
data = my_util.exact_run_result_in_directory((exp_result_path + filename + '/'), exact_list)
[result_score, eval_score_dict] = exact_result_all_seed(data, my_unique_labels, seed, score_order='descending')
eval_score_dict['score'] = result_score
my_util.save_numpy(eval_score_dict, (average_exp_result + save_folder), save_name, doSilent=True)



print()
print()
print()


