
from collections import Counter 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
import warnings

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
    tmp_count['freq_perc'] = tmp_count.apply(lambda row: row.freq/tmp_size_all, axis = 1)
    return tmp_count

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

def eval_classification_performance(y_true, y_pred, label_name):
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
    # eval_score['accuracy'] = np.array(tmp_accuracy)
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

def average_gridsearch(cv_results, sortby):
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
    avg_cv_results = avg_cv_results.sort_values(by=['auc', 'f1score'], ascending=[False, False])
    # Clear index
    cv_results = cv_results.reset_index(drop=True)
    return cv_results, avg_cv_results