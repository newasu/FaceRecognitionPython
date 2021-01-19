import sys
import psutil
from collections import Counter 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise
from scipy import linalg
from scipy import stats
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import warnings
import random
from tqdm import tqdm
# Parallel
import multiprocessing
from joblib import Parallel, delayed

from scipy.optimize import brentq
from scipy.interpolate import interp1d

def getsizeof(measured_data, unit='mb'):
    data_size = sys.getsizeof(measured_data)
    if unit == 'gb':
        return ((data_size/1024)/1024)/1024
    elif unit == 'mb':
        return (data_size/1024)/1024
    elif unit == 'kb':
        return data_size/1024
    else:
        return data_size

def get_available_memory(unit='mb'):
    free_mem = psutil.virtual_memory().available
    if unit == 'gb':
        return ((free_mem/1024)/1024)/1024
    elif unit == 'mb':
        return (free_mem/1024)/1024
    elif unit == 'kb':
        return free_mem/1024
    else:
        return free_mem

def split_kfold_by_classes(yy, n_splits=5, random_state=0):
    # Initial
    kfold_train_index = []
    kfold_test_index = []
    yy_dataframe = pd.DataFrame(yy)
    # Unique and shuffle classes
    kfold_idx = np.unique(yy)
    random.Random(random_state).shuffle(kfold_idx)
    kfold_idx = np.array_split(kfold_idx, n_splits)
    # Apportion train and test index
    tmp_copy = np.arange(0, yy.size)
    for loop_idx in range(0, n_splits):
        kfold_test_index.append(np.where(yy_dataframe.isin(kfold_idx[loop_idx]))[0])
        kfold_train_index.append(tmp_copy)
        kfold_train_index[loop_idx] = np.delete(kfold_train_index[loop_idx], kfold_test_index[loop_idx], None)
    return kfold_train_index, kfold_test_index

def split_data_by_classes(yy, test_size=0.3, random_state=0):
    unique_yy = np.unique(yy)
    # Find number of test samples
    if isinstance(test_size, float):
        numb_test_sample = np.round(len(unique_yy) * test_size)
    else:
        numb_test_sample = test_size
    # Raise error if number of test samples is bigger than number of classes
    if numb_test_sample >= len(unique_yy):
        raise Exception('utilities.split_data_by_classes: test_size is larger/equal than number of sample classes.')
    # Indexing
    numb_training_sample = int(len(unique_yy) - numb_test_sample)
    random.Random(random_state).shuffle(unique_yy)
    training_idx = unique_yy[0:numb_training_sample]
    test_idx = unique_yy[numb_training_sample:]
    training_idx = np.in1d(yy, training_idx).nonzero()[0]
    test_idx = np.in1d(yy, test_idx).nonzero()[0]
    return training_idx, test_idx

def split_data_by_id_and_classes(id_yy, class_yy, test_size=0.3, valid_size=0.0, random_state=0):
    unique_id_yy = np.unique(id_yy)
    unique_class_yy = np.unique(class_yy)
    # Find number of test/valid samples
    if isinstance(test_size, float):
        numb_test_sample = int(np.round(len(unique_id_yy) * test_size))
        numb_valid_sample = int(np.round(len(unique_id_yy) * valid_size))
    else:
        numb_test_sample = test_size
        numb_valid_sample = valid_size
    numb_training_sample = (len(unique_id_yy) - (numb_test_sample+numb_valid_sample))
    # Find labels each id
    if unique_id_yy.shape < class_yy.shape:
        print('Finding label of id..')
        id_class_yy = np.empty(0)
        for id_idx in tqdm(unique_id_yy):
            id_class_yy = np.append(id_class_yy, class_yy[np.where(id_yy == id_idx)[0][0]])
    else:
        id_class_yy = class_yy
    training_idx = np.empty(0)
    test_idx = np.empty(0)
    valid_idx = np.empty(0)
    numb_training_sample_each_class = int(np.round(numb_training_sample/unique_class_yy.size))
    numb_test_sample_each_class = int(np.round(numb_test_sample/unique_class_yy.size))
    # numb_valid_sample_each_class = int(np.round(numb_valid_sample/unique_class_yy.size))
    for class_idx in unique_class_yy:
        shuffled_samples = np.where(id_class_yy==class_idx)[0]
        shuffled_samples = unique_id_yy[shuffled_samples]
        random.Random(random_state).shuffle(shuffled_samples)
        training_idx = np.append(training_idx, np.in1d(id_yy, shuffled_samples[0:numb_training_sample_each_class]).nonzero()[0])
        shuffled_samples = np.delete(shuffled_samples, range(0,numb_training_sample_each_class))
        test_idx = np.append(test_idx, np.in1d(id_yy, shuffled_samples[0:numb_test_sample_each_class]).nonzero()[0])
        shuffled_samples = np.delete(shuffled_samples, range(0,numb_test_sample_each_class))
        valid_idx = np.append(valid_idx, np.in1d(id_yy, shuffled_samples).nonzero()[0])

    return training_idx.astype(int), test_idx.astype(int), valid_idx.astype(int)

def time_counter():
    return time.perf_counter()

def convert_nan_to_zero(my_mat):
    if isinstance(my_mat, list):
        my_mat = np.array(my_mat)

    if my_mat.size > 1:
        my_mat[np.isnan(my_mat)] = 0
    else:
        if np.isnan(my_mat):
            my_mat = 0
    return my_mat

def join_path(main_path, additional_path):
    tmp_path = main_path
    if isinstance(additional_path, str):
        tmp_path = os.path.join(tmp_path, additional_path)
    elif isinstance(additional_path, list):
        for sub_directory in additional_path:
            tmp_path = os.path.join(tmp_path, sub_directory)
        tmp_path = tmp_path + os.path.sep
    return tmp_path

def get_path(additional_path='', create_if_not_exists=False):
    # tmp_path = os.path.dirname(os.getcwd())
    tmp_path = os.getcwd()
    for gp in additional_path:
        if gp == '.':
            tmp_path = os.path.dirname(tmp_path)
        else:
            tmp_path = join_path(tmp_path, gp)
    return tmp_path + os.sep

def make_directory(directory_path, doSilent=False):
    if is_path_available(directory_path):
        if not doSilent:
            print('Directory already exists')
    else:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        if not doSilent:
            print('Directory is created: ' + directory_path)
    pass

def is_path_available(checked_path):
    return os.path.exists(checked_path)

def checkClassProportions(yy):
    tmp_size_all = yy.size
    tmp_count = Counter(yy)
    tmp_count = sorted(tmp_count.items())
    tmp_count = list(map(list, tmp_count))
    tmp_count = pd.DataFrame.from_records(tmp_count)
    tmp_count.columns = ['class', 'freq']
    tmp_count['freq_perc'] = tmp_count.apply(lambda row: row.freq/tmp_size_all, axis=1)
    return tmp_count

def roc_curve_ascending_order(y_true, y_pred_score, pos_label):
    max_dist = max(y_pred_score)
    pred_score = np.array([1-val/max_dist for val in y_pred_score])
    [fpr, tpr, thresholds] = roc_curve(y_true, pred_score, pos_label=pos_label)
    return fpr, tpr, thresholds

def cal_auc(y_true, y_pred_score, y_label):
    # bin_y_true = label_binarize(y_true, classes=y_label)
    eval_score = np.empty(0)
    for tmp_auc_idx in range(0, y_label.size):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            [fpr, tpr, thresholds] = roc_curve(y_true, y_pred_score[:, tmp_auc_idx], pos_label=y_label[tmp_auc_idx])
        # [fpr, tpr, thresholds] = roc_curve(bin_y_true[:, tmp_auc_idx], y_pred_score[:, tmp_auc_idx], pos_label=1)
        # eval_score.append(auc(fpr, tpr))
        eval_score = np.append(eval_score, auc(fpr, tpr))
    # eval_score[np.isnan(eval_score)] = 0
    eval_score = convert_nan_to_zero(eval_score)
    eval_score = {'auc':eval_score, 'auc_mean': np.mean(eval_score)}
    return eval_score

def binary_classes_auc(y_true, y_pred_score, pos_label):
    # max_dist = max(y_pred_score)
    # pred = np.array([1-e/max_dist for e in y_pred_score])
    # [fpr, tpr, thresholds] = roc_curve(y_true, pred, pos_label=pos_label)
    [fpr, tpr, thresholds] = roc_curve_ascending_order(y_true, y_pred_score, pos_label)
    return {'auc':auc(fpr, tpr)}

def classification_performance_metric(y_true, y_pred, label_name):
    conf_mat = multilabel_confusion_matrix(y_true, y_pred, labels=label_name)
    # tmp_accuracy = []
    tmp_precision = []
    tmp_recall = []
    tmp_f1score = []
    for tmp_conf_idx in range(0, label_name.size):
        tmp_tn = conf_mat[tmp_conf_idx][0][0]
        tmp_fn = conf_mat[tmp_conf_idx][1][0]
        tmp_tp = conf_mat[tmp_conf_idx][1][1]
        tmp_fp = conf_mat[tmp_conf_idx][0][1]

        # tmp_accuracy.append( (tmp_tn+tmp_tp) / (tmp_tn+tmp_tp+tmp_fn+tmp_fp) )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tmp_precision.append(tmp_tp/(tmp_tp+tmp_fp))
            tmp_precision[tmp_conf_idx] = float(convert_nan_to_zero(tmp_precision[tmp_conf_idx]))
            tmp_recall.append(tmp_tp/(tmp_tp+tmp_fn))
            tmp_recall[tmp_conf_idx] = float(convert_nan_to_zero(tmp_recall[tmp_conf_idx]))

        # Avoid division by zero
        try:
            tmp_f1score.append(2 * ((tmp_precision[tmp_conf_idx]*tmp_recall[tmp_conf_idx]) / (tmp_precision[tmp_conf_idx]+tmp_recall[tmp_conf_idx])))
        except ZeroDivisionError:
            tmp_f1score.append(0.0)
            
    tmp_f1score = convert_nan_to_zero(tmp_f1score)
    eval_score = {}
    eval_score['accuracy'] = cal_accuracy(y_true, y_pred)
    eval_score['precision'] = np.array(tmp_precision)
    eval_score['recall'] = np.array(tmp_recall)
    eval_score['f1score'] = np.array(tmp_f1score)
    # Mean
    # eval_score['accuracy_mean'] = np.mean(eval_score['accuracy'])
    eval_score['precision_mean'] = np.mean(eval_score['precision'])
    eval_score['recall_mean'] = np.mean(eval_score['recall'])
    eval_score['f1score_mean'] = np.mean(eval_score['f1score'])
    # del conf_mat, tmp_tn, tmp_fn, tmp_tp, tmp_fp
    
    return eval_score

def biometric_metric(y_true, y_pred_score, pos_label, score_order='ascending', threshold_step=0.0001):
    # FMR (False Match Rate) = should reject but accpet
    # FNMR (False Non-Match Rate) = should accpet but reject
    # EER = Equal Error Rate = crossing point of False Matches meet False Non-Matches (FMR=FNMR)
    if score_order == 'ascending':
        [fpr, tpr, thresholds] = roc_curve_ascending_order(y_true, y_pred_score, pos_label)
    elif score_order == 'descending':
        [fpr, tpr, thresholds] = roc_curve(y_true, y_pred_score, pos_label=pos_label)
    eval_score = {'auc':auc(fpr, tpr), 'eer':cal_eer(fpr, tpr, thresholds)}
    eval_score.update(cal_fmr_fnmr(y_true, y_pred_score, pos_label, score_order=score_order, threshold_step=threshold_step))
    return eval_score

def cal_fmr_fnmr(y_true, y_pred_score, pos_label, score_order='ascending', threshold_step=0.0001):
    # Find range of thresholds
    threshold_min = np.floor(y_pred_score.min() / threshold_step) * threshold_step
    threshold_max = np.ceil(y_pred_score.max() / threshold_step) * threshold_step
    thresholds = np.arange(threshold_min, threshold_max, threshold_step)
    # Find classes idx
    pos_idx = y_true == pos_label
    neg_idx = ~pos_idx
    pos_size = np.sum(pos_idx)
    neg_size = np.sum(neg_idx)
    neg_label = np.unique(y_true)
    neg_label = neg_label[neg_label != pos_label][0]
    # Prepare variables
    if score_order == 'ascending':
        thresholds = np.flip(thresholds)
    fmr = np.empty(0)
    fnmr = np.empty(0)
    # Calculate
    for threshold_classifier in thresholds:
        if score_order == 'descending':
            thresholded_class = y_pred_score >= threshold_classifier
        else:
            thresholded_class = y_pred_score < threshold_classifier
        # FMR
        temp_fmr = np.sum(thresholded_class[neg_idx]==True)/neg_size
        # FNMR
        temp_fnmr = np.sum(thresholded_class[pos_idx]==False)/pos_size
        # Append
        fmr = np.append(fmr, temp_fmr)
        fnmr = np.append(fnmr, temp_fnmr)
        
    def eval_fmr_fnmr(_fmr, _fnmr, _thres, _y_true, _y_pred_score, _thresholds, _pos_label, _neg_label, _score_order):
        # FMR and TAR
        tmp_idx = np.where(fmr <= _thres)[0]
        if tmp_idx.size == 0:
            tmp_fmr = np.nan
            tmp_tar = np.nan
        else:
            tmp_fmr = _fnmr[tmp_idx[0]] * 100
            y_pred = np.tile(_neg_label, _y_true.shape)
            if _score_order == 'descending':
                y_pred[_y_pred_score >= _thresholds[tmp_idx[0]]] = _pos_label
            else:
                y_pred[_y_pred_score < _thresholds[tmp_idx[0]]] = _pos_label
            p_idx = _y_true == _pos_label
            tmp_tar = np.sum(_y_true[p_idx] == y_pred[p_idx]) / np.sum(p_idx)
            # tmp_tar = cal_accuracy(_y_true, y_pred) * 100
        # FNMR
        tmp_idx = np.where(_fnmr >= _thres)[0]
        if tmp_idx.size == 0:
            tmp_fnmr = np.nan
        else:
            tmp_fnmr = _fmr[tmp_idx[0]] * 100
        return tmp_fmr, tmp_tar, tmp_fnmr
        
    # FMR & TAR
    fmr_1, tar_1, fnmr_1 = eval_fmr_fnmr(fmr, fnmr, 0.01, y_true, y_pred_score, thresholds, pos_label, neg_label, score_order)  # at 1%
    fmr_0d1, tar_0d1, fnmr_0d1 = eval_fmr_fnmr(fmr, fnmr, 0.001, y_true, y_pred_score, thresholds, pos_label, neg_label, score_order)  # at 0.1%
    fmr_0d01, tar_0d01, fnmr_0d01 = eval_fmr_fnmr(fmr, fnmr, 0.0001, y_true, y_pred_score, thresholds, pos_label, neg_label, score_order)  # at 0.01%
    fmr_0, tar_0, fnmr_0 = eval_fmr_fnmr(fmr, fnmr, 0, y_true, y_pred_score, thresholds, pos_label, neg_label, score_order)  # at 0%
    
    if score_order == 'ascending':
        thresholds = np.flip(thresholds)
        fmr = np.flip(fmr)
        fnmr = np.flip(fnmr)
    return {'threshold':thresholds, 'fmr':fmr, 'fnmr':fnmr, 'fmr_1':fmr_1, 'fmr_0d1':fmr_0d1, 'fmr_0d01':fmr_0d01, 'fmr_0':fmr_0, 'fnmr_1':fnmr_1, 'fnmr_0d1':fnmr_0d1, 'fnmr_0d01':fnmr_0d01, 'fnmr_0':fnmr_0, 'tar_1':tar_1, 'tar_0d1':tar_0d1, 'tar_0d01':tar_0d01, 'tar_0':tar_0}

def cal_eer(fpr, tpr, thresholds):
    intersection_x = line_intersection(fpr, tpr, [0, 1], [1, 0])[0]
    eer = intersection_x[0]
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100

def cal_accuracy(y_true, y_pred):
    # list([x[0] for x in y_true])
    eval_score = accuracy_score(y_true.tolist(), y_pred.tolist())
    return eval_score

def load_numpy_file(save_path):
    return np.load(save_path, allow_pickle=True).item()

def save_numpy(model, save_directory, save_name, doSilent=True):
    # Make directory
    make_directory(save_directory, doSilent=True)
    save_path = os.path.join(save_directory, (save_name + '.npy'))
    # Save file
    np.save(save_path, model)
    if not doSilent:
        print()
        print('Model was saved at -> ' + save_path)
        print()
    pass

def average_gridsearch(cv_results, sortby):
    metric_order = np.array([['auc_pos', 'ascending'], ['auc', 'ascending'], ['f1score', 'ascending'], ['eer', 'descending']])
    sortby.append('fold')
    eval_metric_all = cv_results.columns.values
    for sortby_val in sortby:
        eval_metric_all = np.delete(eval_metric_all, np.where(eval_metric_all == sortby_val)[0][0])
    metric_sortby = []
    sortby_order = []
    for eval_metric_all_val in eval_metric_all:
        metric_sortby.append(eval_metric_all_val)
        find_idx = np.where(eval_metric_all_val == metric_order)[0]
        if find_idx.size > 0:
            if metric_order[find_idx][0][1] == 'ascending':
                tmp_sort_order = False
            else:
                tmp_sort_order = True
        else:
            tmp_sort_order = False
        sortby_order.append(tmp_sort_order)
    
    numb_fold = np.unique(cv_results.fold).size
    # Sort each fold in table
    cv_results = cv_results.sort_values(by=sortby)
    sortby.remove('fold')
    # Average
    avg_cv_results = cv_results[sortby].iloc[::numb_fold].values
    avg_result = {}
    for ag_idx in eval_metric_all:
        avg_result[ag_idx] = np.cumsum(cv_results[ag_idx].values, 0)[numb_fold-1::numb_fold]/float(numb_fold)
        avg_result[ag_idx][1:] = avg_result[ag_idx][1:] - avg_result[ag_idx][:-1]
        avg_cv_results = np.concatenate((avg_cv_results, avg_result[ag_idx][:, None]), axis=1)
    # Bind into dataframe
    avg_cv_results = pd.DataFrame(data=avg_cv_results, columns=cv_results.columns[1:].values)
    avg_cv_results = avg_cv_results.sort_values(by=metric_sortby, ascending=sortby_order)
    # Clear index
    cv_results = cv_results.reset_index(drop=True)
    return cv_results, avg_cv_results

def calculate_kernel(m1, m2, kernelFunc, kernel_param=None, useTF=False):
    kernel_mat = []
    if useTF:
        
        if kernelFunc == 'euclidean':
            print('Computing euclidean using Tensorflow')
            # m1_tensor = tf.convert_to_tensor(np.asarray(m1, np.float32), np.float32)
            # m2_tensor = tf.convert_to_tensor(np.asarray(m2, np.float32), np.float32)
            # simKernel = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(m1_tensor, 1) - m2_tensor), 2), 2)
            # simKernel = simKernel.numpy()
            
            m2_tensor = tf.convert_to_tensor(np.asarray(m2, np.float32), np.float32)
            kernel_mat = np.empty((1, m2.shape[0]))
            
            # Compute distance each row of m1 and whole of m2
            for tmp_idx in range(0, m1.shape[0]):
                m1_tensor = tf.convert_to_tensor(np.asarray(m1[tmp_idx,:,None].T, np.float32), np.float32)
                kernel_row = tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(m1_tensor, 1)-m2_tensor), 2), 2)
                kernel_mat = np.concatenate((kernel_mat, kernel_row.numpy()), axis=0)
            
            # Delete not-used variable
            kernel_mat = np.delete(kernel_mat, 0, 0)
            del kernel_row, m1_tensor, m2_tensor
        else:
            raise Exception('There is no kernelFunc match in condition in calculate_kernel function.')
            
    else:
        if kernelFunc == 'euclidean':
            # simKernel = euclidean_distances(m1, m2, squared=False)
            kernel_mat = pairwise_distances(m1, m2, metric=kernelFunc)
        elif kernelFunc == 'cosine':
            kernel_mat = pairwise.cosine_similarity(m1, m2)
        elif kernelFunc == 'rbf':
            if kernel_param == 0:
                gamma = None
            else:
                gamma = kernel_param
            kernel_mat = pairwise.rbf_kernel(m1, m2, gamma=gamma)
        elif kernelFunc == 'sigmoid':
            kernel_mat = pairwise.sigmoid_kernel(m1, m2, gamma=kernel_param, coef0=kernel_param)
        elif kernelFunc == 'polynomial':
            kernel_mat = pairwise.polynomial_kernel(m1, m2, degree=kernel_param, gamma=kernel_param, coef0=kernel_param)
    
    return kernel_mat

def cal_inv_func(pass_inv_data):
    # inv_data = np.linalg.pinv(pass_inv_data)
    temp_inv_data = linalg.inv(pass_inv_data)
    return temp_inv_data

def triplet_loss_paring(data_id, data_class, **kwargs):
    print('triplet_paring')
    # Assign params
    funcParams = {}
    funcParams['num_cores'] = 1
    funcParams['randomseed'] = 0
    for key, value in kwargs.items():
        if key in funcParams:
            funcParams[key] = value
        else:
            raise Exception('Error key ({}) exists in dict'.format(key))
    del key, value
    
    if funcParams['num_cores'] == '-1':
        funcParams['num_cores'] = multiprocessing.cpu_count()
    
    # bind data into dataframe and shuffle the orders
    data = pd.DataFrame({'image_id': data_id, 'image_class': data_class})
    data = data.sample(frac=1, random_state=funcParams['randomseed']).reset_index()
    # Unique classes and shuffle the orders
    data_class_freq = Counter(data_class)
    unique_class = np.array(list(data_class_freq.keys()))
    data_class_freq = np.array(list(data_class_freq.values()))
    # unique_class = np.unique(img_class)
    # Delete class that contains member below two
    if data_class_freq.min() < 2:
        print('Some classes contain member less than two.')
        print('Those classes will be deleted...')
        unique_class = np.delete(unique_class, np.where(data_class_freq < 2))
        # data_class_freq = np.delete(data_class_freq, np.where(data_class_freq < 2))
    del data_class_freq
    unique_class = sorted(unique_class)
    random.Random(funcParams['randomseed']).shuffle(unique_class)
    
    # Parallel pair
    # triplet_dataset = Parallel(n_jobs=funcParams['num_cores'])(delayed(do_triplet_loss_paring)(unique_class_idx, data, unique_class) for unique_class_idx in range(0, len(unique_class)))
    # triplet_dataset = pd.DataFrame(triplet_dataset)
    
    # Looppy pair
    triplet_dataset = pd.DataFrame([])
    for unique_class_idx in tqdm(range(0, len(unique_class))):
        tmp_triplet_dataset = do_triplet_loss_paring(unique_class_idx, data, unique_class)
        triplet_dataset = triplet_dataset.append(pd.DataFrame([tmp_triplet_dataset]), ignore_index=True)
        
    return triplet_dataset

def do_triplet_loss_paring(unique_class_idx, data, unique_class):
    tmp = data.query('image_class == ' + str(unique_class[unique_class_idx]))
    anchor_id = tmp.iloc[0].image_id
    anchor_idx = tmp.iloc[0][0]
    positive_id = tmp.iloc[1].image_id
    positive_idx = tmp.iloc[1][0]
    positive_class = tmp.iloc[1].image_class
    if unique_class_idx == len(unique_class)-1: # The last sample must to be paired with first sample
        tmp = data.query('image_class == ' + str(unique_class[0]))    
    else:
        tmp = data.query('image_class == ' + str(unique_class[(unique_class_idx+1)]))
    negative_id = tmp.iloc[0].image_id
    negative_idx = tmp.iloc[0][0]
    negative_class = tmp.iloc[0].image_class
    
    tmp_triplet_dataset = {
        'anchor_id':anchor_id, 
        'anchor_idx':anchor_idx,
        'positive_id':positive_id, 
        'positive_class':positive_class,
        'positive_idx':positive_idx, 
        'negative_id':negative_id, 
        'negative_class':negative_class,
        'negative_idx':negative_idx}
    # print('triplet_loss_paring: ' + str(unique_class_idx+1) + '/' + str(len(unique_class)))
    
    return tmp_triplet_dataset

def combination_rule_paired_list(dataXX, data_id, paired_list, combine_rule='sum'):
    # print('combination_rule_paired_list')
    # Find data index in data_id
    tmp_anchor_idx = paired_list.anchor_id.values[:, None] == data_id
    tmp_positive_idx = paired_list.positive_id.values[:, None] == data_id
    tmp_negative_idx = paired_list.negative_id.values[:, None] == data_id
    tmptmp_anchor_idx = []
    tmptmp_positive_idx = []
    tmptmp_negative_idx = []
    tmptmp_positive_data_id = []
    tmptmp_negative_data_id = []
    for idx in range(0, tmp_anchor_idx.shape[0]):
        tmptmp_anchor_idx.append(np.where(tmp_anchor_idx[idx,:])[0][0])
        tmptmp_positive_idx.append(np.where(tmp_positive_idx[idx,:])[0][0])
        tmptmp_negative_idx.append(np.where(tmp_negative_idx[idx,:])[0][0])
        tmptmp_positive_data_id.append(paired_list.anchor_id[idx].astype(str) + '-' + paired_list.positive_id[idx].astype(str))
        tmptmp_negative_data_id.append(paired_list.anchor_id[idx].astype(str) + '-' + paired_list.negative_id[idx].astype(str))
    del tmp_anchor_idx, tmp_positive_idx, tmp_negative_idx
    # retrieve data by idx
    tmp_anchor_feature = dataXX[tmptmp_anchor_idx,:]
    tmp_positive_feature = dataXX[tmptmp_positive_idx,:]
    tmp_negative_feature = dataXX[tmptmp_negative_idx,:]
    del tmptmp_anchor_idx, tmptmp_positive_idx, tmptmp_negative_idx
    # Combine
    if combine_rule == 'sum':
        tmp_positive_feature = tmp_anchor_feature + tmp_positive_feature
        tmp_negative_feature = tmp_anchor_feature + tmp_negative_feature
    elif combine_rule == 'minus':
        tmp_positive_feature = tmp_anchor_feature - tmp_positive_feature
        tmp_negative_feature = tmp_anchor_feature - tmp_negative_feature
    elif combine_rule == 'multiply':
        tmp_positive_feature = np.multiply(tmp_anchor_feature, tmp_positive_feature)
        tmp_negative_feature = np.multiply(tmp_anchor_feature, tmp_negative_feature)
    elif combine_rule == 'distance':
        tmp_positive_feature = np.absolute(tmp_anchor_feature - tmp_positive_feature)
        tmp_negative_feature = np.absolute(tmp_anchor_feature - tmp_negative_feature)
    elif combine_rule == 'mean':
        tmp_positive_feature = (tmp_anchor_feature + tmp_positive_feature)/2
        tmp_negative_feature = (tmp_anchor_feature + tmp_negative_feature)/2
    elif combine_rule == 'concatenate':
        tmp_positive_feature = np.concatenate((tmp_anchor_feature, tmp_positive_feature), axis=1)
        tmp_negative_feature = np.concatenate((tmp_anchor_feature, tmp_negative_feature), axis=1)
    del tmp_anchor_feature
    # Arrange feature
    combined_xx = np.concatenate((tmp_positive_feature, tmp_negative_feature), axis=1)
    combined_xx = np.reshape(combined_xx, ((tmp_positive_feature.shape[0]*2), -1 ))
    # Arrange class
    combined_yy = np.concatenate((np.tile('POS', len(tmptmp_positive_data_id))[:,None], np.tile('NEG', len(tmptmp_negative_data_id))[:,None]), axis=1)
    combined_yy = np.reshape(combined_yy, (combined_yy.size, -1 ))
    combined_yy = np.squeeze(combined_yy)
    # Arrange data id
    tmptmp_positive_data_id = np.array(tmptmp_positive_data_id)
    tmptmp_negative_data_id = np.array(tmptmp_negative_data_id)
    combined_data_id = np.concatenate((tmptmp_positive_data_id[:,None], tmptmp_negative_data_id[:,None]), axis=1)
    combined_data_id = np.reshape(combined_data_id, (combined_yy.size, -1 ))
    combined_data_id = np.squeeze(combined_data_id)
    return combined_xx, combined_yy, combined_data_id

def combination_rule_paired_list_1(dataXX, data_id, paired_list, combine_rule='sum'):
    # Initial
    if combine_rule == 'concatenate':
        combined_xx = np.empty((0, (dataXX.shape[1]*2)), np.float64)
    else:
        combined_xx = np.empty((0, dataXX.shape[1]), np.float64)
    combined_yy = []
    combined_data_id = []
    # Assign and Combine by rule
    for idx in range(0, paired_list.shape[0]):
        # Find data by id
        tmp_anchor_idx = np.where(data_id == paired_list.anchor_id[idx])[0][0]
        tmp_positive_idx = np.where(data_id == paired_list.positive_id[idx])[0][0]
        tmp_negative_idx = np.where(data_id == paired_list.negative_id[idx])[0][0]
        tmp_anchor_feature = dataXX[tmp_anchor_idx, :, None].T
        tmp_positive_feature = dataXX[tmp_positive_idx, :, None].T
        tmp_negative_feature = dataXX[tmp_negative_idx, :, None].T
        # Combine
        if combine_rule == 'sum':
            tmp_positive_feature = tmp_anchor_feature + tmp_positive_feature
            tmp_negative_feature = tmp_anchor_feature + tmp_negative_feature
        elif combine_rule == 'minus':
            tmp_positive_feature = tmp_anchor_feature - tmp_positive_feature
            tmp_negative_feature = tmp_anchor_feature - tmp_negative_feature
        elif combine_rule == 'multiply':
            tmp_positive_feature = np.multiply(tmp_anchor_feature, tmp_positive_feature)
            tmp_negative_feature = np.multiply(tmp_anchor_feature, tmp_negative_feature)
        elif combine_rule == 'distance':
            tmp_positive_feature = np.absolute(tmp_anchor_feature - tmp_positive_feature)
            tmp_negative_feature = np.absolute(tmp_anchor_feature - tmp_negative_feature)
        elif combine_rule == 'mean':
            tmp_positive_feature = (tmp_anchor_feature + tmp_positive_feature)/2
            tmp_negative_feature = (tmp_anchor_feature + tmp_negative_feature)/2
        elif combine_rule == 'concatenate':
            tmp_positive_feature = np.concatenate((tmp_anchor_feature, tmp_positive_feature), axis=1)
            tmp_negative_feature = np.concatenate((tmp_anchor_feature, tmp_negative_feature), axis=1)
        # Bind feature
        combined_xx = np.append(combined_xx, tmp_positive_feature, axis=0)
        combined_xx = np.append(combined_xx, tmp_negative_feature, axis=0)
        # Bind class
        combined_yy.append('POS')
        combined_yy.append('NEG')
        # Bind data id
        combined_data_id.append(paired_list.anchor_id[idx] + '-' + paired_list.positive_id[idx])
        combined_data_id.append(paired_list.anchor_id[idx] + '-' + paired_list.negative_id[idx])
        print('combination_rule_paired_list: ' + str(idx+1) + '/' + str(paired_list.shape[0]))
    combined_yy = np.array(combined_yy)
    combined_data_id = np.array(combined_data_id)
    return combined_xx, combined_yy, combined_data_id

def line_intersection(line1_x, line1_y, line2_x, line2_y):
    line1 = LineString(np.column_stack((line1_x, line1_y)))
    line2 = LineString(np.column_stack((line2_x, line2_y)))
    intersection = line1.intersection(line2)
    if intersection.geom_type == 'MultiPoint':
        intersection = np.asarray(LineString(intersection).xy)
        intersection_x = intersection[0]
        intersection_y = intersection[1]
    elif intersection.geom_type == 'LineString':
        intersection = np.asarray(intersection.xy)
        intersection_x = intersection[0]
        intersection_y = intersection[1]
    elif intersection.geom_type == 'Point':
        intersection_x = np.array([intersection.x])
        intersection_y = np.array([intersection.y])
    elif intersection.geom_type == 'MultiLineString':
        intersection = np.asarray(intersection[0].xy)
        intersection_x = intersection[0]
        intersection_y = intersection[1]
    else:
        intersection_x = np.empty(0)
        intersection_y = np.empty(0)
    # Plot
    # plt.plot(line1_x, line1_y, '-')
    # plt.plot(line2_x, line2_y, '-')
    # plt.plot(intersection_x, intersection_y, 'o')
    # plt.show()
    # plt.close()
    return intersection_x, intersection_y

def find_optimal_threshold_two_clases(score_mat, true_y, unique_y, threshold_decimal=2):
    if score_mat.ndim > 1:
        if type(score_mat) == np.matrix:
            score_mat = np.array(score_mat.T)[0]
        else:
            score_mat = score_mat.flatten()
    threshold_step = 10**-threshold_decimal
    # Calculate distance
    dist_min = np.floor(score_mat.min() * (10**threshold_decimal))/(10**threshold_decimal)
    dist_max = np.ceil(score_mat.max() * (10**threshold_decimal))/(10**threshold_decimal)
    # Vary threshold
    dist_confusion = np.empty((0, 3), np.float64)
    for threshold_idx in np.arange(dist_min, (dist_max+threshold_step), threshold_step):
        thresholded_distance_class = np.tile(unique_y['neg'], true_y.shape)
        thresholded_distance_class[score_mat < threshold_idx] = unique_y['pos']
        [tn, fp, fn, tp] = confusion_matrix(true_y, thresholded_distance_class).ravel()
        dist_confusion = np.append(dist_confusion, np.expand_dims(np.array([threshold_idx, tp, tn]), axis=0), axis=0)
    del dist_min, dist_max, thresholded_distance_class, tn, fp, fn, tp
    # Normalize accuracy matrix
    dist_confusion[:,1] = dist_confusion[:,1]/max(dist_confusion[:,1])
    dist_confusion[:,2] = dist_confusion[:,2]/max(dist_confusion[:,2])
    # Find best threshold by finding crossing point between two lines
    [intersection_x, intersection_y] = line_intersection(dist_confusion[:,0], dist_confusion[:,1], dist_confusion[:,0], dist_confusion[:,2])
    optimal_threshold = intersection_x
    return optimal_threshold

def exact_run_result_in_directory(result_directory_path, exact_list):
    file_in_directory = os.listdir(result_directory_path)
    if '.DS_Store' in file_in_directory: file_in_directory.remove('.DS_Store')
    file_in_directory = sorted(file_in_directory)
    exacted_result = {}
    exacted_result['filenames'] = np.asarray(file_in_directory)
    # Load result
    result = {}
    for x in range(0, len(file_in_directory)):
        result[x] = load_numpy_file(result_directory_path + file_in_directory[x])
    # Exact result
    for exact_name in exact_list:
        tmp = []
        for x in range(0, len(file_in_directory)):
            if exact_name in result[x]:
                tmp.append(result[x][exact_name])
            else:
                tmp.append(np.nan)
        exacted_result[exact_name] = np.asarray(tmp)
    return exacted_result

def exact_result_eval_retrieve(exacted_data, data_names, term_finding, metric_ordering='descending'):
    tmp = [x[term_finding] for x in exacted_data]
    tmp = np.vstack(tmp).T
    if isinstance(tmp[0,0], np.number):
        avg_mat = np.average(tmp, axis=0)
        avg_mat = pd.DataFrame(np.average(tmp, axis=0)[np.newaxis,:], columns=data_names)
        if metric_ordering == 'descending':
            ranked_mat = stats.rankdata(-tmp, method='average', axis=1)
        else:
            ranked_mat = stats.rankdata(tmp, method='average', axis=1)
        sum_ranked_mat = np.sum(ranked_mat, axis=0)
        ranked_mat = pd.DataFrame(ranked_mat, columns=data_names)
        sum_ranked_mat = pd.DataFrame(sum_ranked_mat[np.newaxis,:], columns=data_names)
    else:
        avg_mat = np.nan
        ranked_mat = np.nan
        sum_ranked_mat = np.nan
    retrieved_mat = pd.DataFrame(tmp, columns=data_names)
    return retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat

def exact_classes_result_eval_retrieve(exacted_data, data_names, term_finding, metric_ordering='descending', class_name_finding='label_classes'):
    # unique class
    unique_class = []
    for x in exacted_data:
        unique_class.append(x[class_name_finding])
    unique_class = np.unique(unique_class)

    # Inital
    tmp_exacted_data = {}
    for x in unique_class:
        tmp_exacted_data[x] = []
        for y in data_names:
            tmp_exacted_data[x].append([])
        
    # Seperate classes
    for x in range(0, len(exacted_data)):
        tmp_data = exacted_data[x][term_finding]
        tmp_class = exacted_data[x][class_name_finding]
        for y in unique_class:
            # tmp_exacted_data[y][term_finding].append(tmp_data[tmp_class==y])
            tmp_exacted_data[y][x] = {term_finding:tmp_data[tmp_class==y]}
            
    # Exact
    retrieved_mat = {}
    avg_mat = {}
    ranked_mat = {}
    sum_ranked_mat = {}
    for x in unique_class:
        [retrieved_mat[x], avg_mat[x], ranked_mat[x], sum_ranked_mat[x]] = exact_result_eval_retrieve(tmp_exacted_data[x], data_names, term_finding, metric_ordering=metric_ordering)
    
    return retrieved_mat, avg_mat, ranked_mat, sum_ranked_mat

def limit_cpu_used(cpu_used_perc=0.8):
    return np.int(np.round(multiprocessing.cpu_count() * cpu_used_perc))

def ceil(a, precision=0):
    return np.round((a + (0.5 * 10**(-precision))), precision)

def floor(a, precision=0):
    return np.round((a - (0.5 * 10**(-precision))), precision)

def kendall_w(expt_ratings):
    if expt_ratings.ndim!=2:
        raise 'ratings matrix must be 2-dimensional'
    k = expt_ratings.shape[0] # number of experimnet
    n = expt_ratings.shape[1] # candidates
    denom = k**2*(n**3-n)
    rating_sums = np.sum(expt_ratings, axis=0)
    S = n*np.var(rating_sums)
    W = 12*S/denom
    chi_square = k*(n-1)*W
    confidence_level = stats.chi2.cdf(chi_square, n-1)
    return S, W, chi_square, confidence_level


