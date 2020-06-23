
from collections import Counter 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
# from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from collections import Counter
from scipy import linalg
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
# Parallel
import multiprocessing
from joblib import Parallel, delayed

from scipy.optimize import brentq
from scipy.interpolate import interp1d

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

def get_current_path(additional_path=''):
    tmp_path = os.path.dirname(os.getcwd())
    tmp_path = join_path(tmp_path, additional_path)
    return tmp_path

def make_directory(directory_path, doSilent=False):
    if is_path_available(directory_path):
        if not doSilent:
            print('Directory already exists')
    else:
        print('Directory was created')
        Path(directory_path).mkdir(parents=True, exist_ok=True)
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

def roc_curve_descending_order(y_true, y_pred_score, pos_label):
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
    [fpr, tpr, thresholds] = roc_curve_descending_order(y_true, y_pred_score, pos_label)
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

def biometric_metric(y_true, y_pred_score, pos_label, score_order='ascending'):
    # FMR (False Match Rate) = should reject but accpet
    # FNMR (False Non-Match Rate) = should accpet but reject
    # EER = Equal Error Rate = crossing point of False Matches meet False Non-Matches (FMR=FNMR)
    if score_order == 'ascending':
        [fpr, tpr, thresholds] = roc_curve(y_true, y_pred_score, pos_label=pos_label)
    else:
        [fpr, tpr, thresholds] = roc_curve_descending_order(y_true, y_pred_score, pos_label)
    
    eval_score = {'auc':auc(fpr, tpr), 'eer':cal_eer(fpr, tpr, thresholds)}
    
    eval_score.update(cal_far_frr(y_true, y_pred_score, pos_label, score_order=score_order))
    
    return eval_score

def cal_far_frr(y_true, y_pred_score, pos_label, score_order='ascending', threshold_step=0.0001):
    # Find range of thresholds
    threshold_min = np.floor(y_pred_score.min() / threshold_step) * threshold_step
    threshold_max = np.ceil(y_pred_score.max() / threshold_step) * threshold_step
    thresholds = np.arange(threshold_min, threshold_max, threshold_step)
    # Find classes idx
    pos_idx = y_true == pos_label
    neg_idx = ~pos_idx
    pos_size = np.sum(pos_idx)
    neg_size = np.sum(neg_idx)
    # Prepare variables
    if score_order == 'descending':
        thresholds = np.flip(thresholds)
    fmr = np.empty(0)
    fnmr = np.empty(0)
    # Calculate
    for threshold_classifier in thresholds:
        if score_order == 'ascending':
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
    # FMR
    # at 0.1%
    tmp_idx = np.where(fmr < 0.001)[0]
    if tmp_idx.size == 0:
        fmr_0d1 = np.nan
    else:
        fmr_0d1 = fnmr[tmp_idx[0]] * 100
    # at 0.01%
    tmp_idx = np.where(fmr < 0.0001)[0]
    if tmp_idx.size == 0:
        fmr_0d01 = np.nan
    else:
        fmr_0d01 = fnmr[tmp_idx[0]] * 100
    # FNMR
    # at 0.1%
    tmp_idx = np.where(fnmr > 0.001)[0]
    if tmp_idx.size == 0:
        fnmr_0d1 = np.nan
    else:
        fnmr_0d1 = fmr[tmp_idx[0]] * 100
    # at 0.01%
    tmp_idx = np.where(fnmr > 0.0001)[0]
    if tmp_idx.size == 0:
        fnmr_0d01 = np.nan
    else:
        fnmr_0d01 = fmr[tmp_idx[0]] * 100
    
    # if score_order == 'descending':
    #     thresholds = np.flip(thresholds)
    #     fmr = np.flip(fmr)
    #     fnmr = np.flip(fnmr)
    return {'fmr_fnmr_thresh':thresholds, 'fmr':fmr, 'fnmr':fnmr, 'fmr_0d1':fmr_0d1, 'fmr_0d01':fmr_0d01, 'fnmr_0d1':fnmr_0d1, 'fnmr_0d01':fnmr_0d01}

def cal_eer(fpr, tpr, thresholds):
    intersection_x = line_intersection(fpr, tpr, [0, 1], [1, 0])[0]
    eer = intersection_x[0]
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    return eer

def cal_accuracy(y_true, y_pred):
    # list([x[0] for x in y_true])
    eval_score = accuracy_score(y_true.tolist(), y_pred.tolist())
    return eval_score

def load_numpy_file(save_path):
    return np.load(save_path ,allow_pickle='TRUE').item()

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

def average_gridsearch(cv_results, sortby, eval_metric=[['auc', 'descending'], ['f1score', 'descending']]):
    metric_sortby = []
    sortby_order = []
    for tmp_eval_metric in eval_metric:
        metric_sortby.append(tmp_eval_metric[0])
        if tmp_eval_metric[1] == 'descending':
            sortby_order.append(False)
        else:
            sortby_order.append(True)
    
    numb_fold = np.unique(cv_results.fold).size
    # Average each fold in table
    cv_results = cv_results.sort_values(by=(sortby + ['fold']))
    # Average AUC
    avg_auc = np.cumsum(cv_results.auc.values, 0)[numb_fold-1::numb_fold]/float(numb_fold)
    avg_auc[1:] = avg_auc[1:] - avg_auc[:-1]
    # Average F1 scores
    avg_f1score = np.cumsum(cv_results.f1score.values, 0)[numb_fold-1::numb_fold]/float(numb_fold)
    avg_f1score[1:] = avg_f1score[1:] - avg_f1score[:-1]
    # Average params
    avg_param = cv_results[sortby].iloc[::numb_fold].values
    # Average cv_results
    avg_cv_results = np.concatenate((avg_param, avg_auc[:, None], avg_f1score[:, None]), axis=1)
    avg_cv_results = pd.DataFrame(data=avg_cv_results, columns=cv_results.columns[1:].values)
    avg_cv_results = avg_cv_results.sort_values(by=metric_sortby, ascending=sortby_order)
    # Clear index
    cv_results = cv_results.reset_index(drop=True)
    return cv_results, avg_cv_results

def calculate_kernel(m1, m2, kernelFunc, useTF=False):
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
        # simKernel = euclidean_distances(m1, m2, squared=False)
        kernel_mat = pairwise_distances(m1, m2, metric=kernelFunc)
    
    return kernel_mat

def cal_inv_func(pass_inv_data):
    # inv_data = np.linalg.pinv(pass_inv_data)
    temp_inv_data = linalg.inv(pass_inv_data)
    return temp_inv_data

def triplet_loss_paring(data_id, data_class, **kwargs):
    # Assign params
    funcParams = {}
    funcParams['num_cores']  = 1
    funcParams['randomseed'] = 0
    for key, value in kwargs.items():
        if key in funcParams:
            funcParams[key] = value
        else:
            raise Exception('Error key ({}) exists in dict'.format(key))
    del key, value
    
    if funcParams['num_cores'] == 'all':
        funcParams['num_cores']  = multiprocessing.cpu_count()
    
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
    triplet_dataset = Parallel(n_jobs=funcParams['num_cores'])(delayed(do_triplet_loss_paring)(unique_class_idx, data, unique_class) for unique_class_idx in range(0, len(unique_class)))
    triplet_dataset = pd.DataFrame(triplet_dataset)
    
    # Looppy pair
    # triplet_dataset = pd.DataFrame([])
    # for unique_class_idx in range(0, len(unique_class)):
    #     tmp_triplet_dataset = do_triplet_loss_paring(unique_class_idx, data, unique_class)
    #     triplet_dataset = triplet_dataset.append(pd.DataFrame([tmp_triplet_dataset]), ignore_index=True)
        
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
    print('triplet_loss_paring: ' + str(unique_class_idx+1) + '/' + str(len(unique_class)))
    
    return tmp_triplet_dataset

def combination_rule_paired_list(dataXX, data_id, paired_list, combine_rule='sum'):
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

