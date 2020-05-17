import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from collections import Counter
from scipy import linalg
import itertools

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Import my lib
import others.utilities as my_util

class welm(object):

    def __init__(self):
        pass

    def train(self, trainingDataX, trainingDataY, **kwargs):

        # Check number of data sizes
        if trainingDataX.shape[0] != trainingDataY.shape[0]:
            raise Exception('number of data size is not equal')

        # Assign params
        modelParams = {}
        modelParams['hiddenNodePerc'] = 100
        modelParams['regC'] = 1
        modelParams['randomseed'] = 0
        modelParams['distanceFunc'] = 'euclidean'
        modelParams['trainingDataID'] = np.array(range(0, trainingDataX.shape[0]))
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))

        [weights, trainingWeightDataID] = self.initHidden(trainingDataX, modelParams['trainingDataID'], modelParams['hiddenNodePerc'], modelParams['randomseed'])
        tic = my_util.time_counter()
        [beta, label_classes] = self.trainModel(trainingDataX, trainingDataY, weights, modelParams['regC'], modelParams['distanceFunc'])
        toc = my_util.time_counter()
        # Timer
        run_time = toc-tic

        # print(modelParams)
        # print(weights)
        # print(trainingWeightDataID)
        # print(beta)

        return weights, trainingWeightDataID, beta, label_classes, run_time

    def initHidden(self, trainingDataX, trainingDataID, hiddenNodePerc, randomseed):
        if hiddenNodePerc < 0:
            raise Exception('The range of hidden node is wrong')
        else:
            if isinstance(hiddenNodePerc, int):
                hiddenNodeNum = hiddenNodePerc
            else:
                hiddenNodeNum = round(hiddenNodePerc * trainingDataX.shape[0])
            
        hiddenNodeNum = min(hiddenNodeNum, trainingDataX.shape[0])
        if not isinstance(hiddenNodeNum, int):
            hiddenNodeNum = hiddenNodeNum.astype(int)

        if hiddenNodeNum == 0:
            hiddenNodeNum = 1
            
        random.seed( randomseed )
        weightIdx = random.sample(range(trainingDataX.shape[0]), hiddenNodeNum)
        weights = trainingDataX[weightIdx, :]
        trainingWeightDataID = trainingDataID[weightIdx]
        return weights, trainingWeightDataID

    def trainModel(self, trainingDataX, trainingDataY, weights, regC, distanceFunc):
        simKernel = self.calculate_kernel(trainingDataX, weights, distanceFunc)

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(trainingDataY)

        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        trainingDataY_onehot = onehot_encoder.fit_transform(integer_encoded)

        # Balance classes
        label_classes = label_encoder.classes_
        class_freq = trainingDataY_onehot.sum(axis=0)
        max_freq = max(class_freq)
        penalized_value = np.sqrt(max_freq / class_freq)
        penalized_array = penalized_value[integer_encoded]
        # penalized_array = np.matrix(np.tile(penalized_array, [1, 1, simKernel.shape[1]]))

        H = np.matrix(np.multiply(penalized_array, simKernel))
        del penalized_array, simKernel
        Two_H = H.T * H
        regC_mat = np.matrix((1/regC) * np.identity(H.shape[1]))
        # inv_data = np.linalg.pinv(Two_H + regC_mat)
        # inv_data = linalg.inv(Two_H + regC_mat)
        inv_data = np.matrix(self.cal_inv_func(Two_H + regC_mat))
        del Two_H, regC_mat
        beta = inv_data * (H.T * (trainingDataY_onehot * penalized_value))

        return beta, label_classes

    def predict(self, testDataX, weights, beta, distanceFunc, label_classes):
        tic = my_util.time_counter()
        simKernel = self.calculate_kernel(testDataX, weights, distanceFunc)
        predictedScores = simKernel * beta
        toc = my_util.time_counter()
        predictedY = np.argmax(predictedScores, axis=1)
        predictedY = label_classes[predictedY]
        # Timer
        run_time = toc-tic

        return predictedScores, predictedY, run_time

    def calculate_kernel(self, m1, m2, distanceFunc):
        # simKernel = euclidean_distances(m1, m2)
        # simKernel = pairwise_distances(m1, m2, metric='euclidean')
        simKernel = pairwise_distances(m1, m2, metric=distanceFunc)
        
        return simKernel

    def cal_inv_func(self, pass_inv_data):
        # inv_data = np.linalg.pinv(pass_inv_data)
        temp_inv_data = linalg.inv(pass_inv_data)

        return temp_inv_data

    def grid_search_cv(self, trainingDataX, trainingDataY, trainingDataID, param_grid, exp_path, exp_name, **kwargs):

        # Assign params
        modelParams = {}
        modelParams['cv'] = 5
        modelParams['cv_run'] = -1 # -1 = run all seed, else, run only define
        modelParams['randomseed'] = 0
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))
        del key, value

        # Run only desire number in cv
        if modelParams['cv_run'] == -1:
            cv_run = list(range(0, modelParams['cv']))
        else:
            if isinstance(modelParams['cv_run'], list):
                cv_run = modelParams['cv_run']
            else:
                cv_run = [modelParams['cv_run']]

        # Get save directory
        my_save_directory = exp_path + exp_name
        
        # Generate k-Fold indices
        kfold_data_spliter = StratifiedKFold(n_splits=modelParams['cv'], shuffle=True, random_state=modelParams['randomseed'])
        kfold_data_spliter.get_n_splits(trainingDataX, trainingDataY)
        kfold_train_data_spliter = list()
        kfold_test_data_spliter = list()
        for kfold_train_index, kfold_test_index in kfold_data_spliter.split(trainingDataX, trainingDataY):
            kfold_train_data_spliter.append(kfold_train_index)
            kfold_test_data_spliter.append(kfold_test_index)

        # Combine params all of possible combination
        param_list = np.array(np.meshgrid(param_grid['distanceFunc'], param_grid['hiddenNodePerc'], param_grid['regC'])).T.reshape(-1,3)

        # Init result
        cv_results = pd.DataFrame(columns=['fold', 'distanceFunc', 'hiddenNodePerc', 'regC', 'auc', 'f1score'])

        # Run k-fold
        count_run = 0
        total_run = str(param_list.shape[0] * len(cv_run))
        for kfold_idx in cv_run:

            # Run grid search
            for gs_param in param_list:
                # Prepare save filename
                my_save_name = ('kf_' + str(kfold_idx) + '_dtf_' + gs_param[0] + '_hdn_' + gs_param[1] + '_rc_' + gs_param[2] + '_rd_' + str(modelParams['randomseed']) )
                my_save_name = my_save_name.replace(".", "d")
                my_save_path = my_util.join_path(my_save_directory, (my_save_name + '.npy'))
                
                if my_util.is_path_available(my_save_path):
                    # Load finished experiment
                    my_model = my_util.load_numpy_file(my_save_path)

                else:
                    # Run experiment
                    # Prepare used parameters
                    tmp_distanceFunc = gs_param[0]
                    tmp_hiddenNodePerc = gs_param[1].astype(type(param_grid['hiddenNodePerc'][0]))
                    tmp_regC = gs_param[2].astype(type(param_grid['regC'][0]))

                    # Train model
                    [weights, weightID, beta, label_classes, training_time] = self.train(trainingDataX[kfold_train_data_spliter[kfold_idx]], 
                    trainingDataY[kfold_train_data_spliter[kfold_idx]], 
                    trainingDataID=trainingDataID[kfold_train_data_spliter[kfold_idx]], 
                    distanceFunc=tmp_distanceFunc, 
                    hiddenNodePerc=tmp_hiddenNodePerc, 
                    regC=tmp_regC, 
                    randomseed=modelParams['randomseed'])

                    # Test model
                    [predictedScores, predictedY, test_time] = self.predict(trainingDataX[kfold_test_data_spliter[kfold_idx]], weights, beta, tmp_distanceFunc, label_classes)

                    # Evaluate performance
                    # AUC
                    eval_scores = my_util.cal_auc(trainingDataY[kfold_test_data_spliter[kfold_idx]], predictedScores, label_classes)
                    # Accuracy
                    eval_scores['accuracy'] = my_util.cal_accuracy(trainingDataY[kfold_test_data_spliter[kfold_idx]], predictedY)
                    # Performance matrix
                    performance_matrix = my_util.eval_classification_performance(trainingDataY[kfold_test_data_spliter[kfold_idx]], predictedY, label_classes)
                    
                    # Prepare model to save
                    my_model = {'distanceFunc':tmp_distanceFunc, 'hiddenNodePerc': tmp_hiddenNodePerc, 'regC':tmp_regC, 'randomseed': modelParams['randomseed'], 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'welm', 'experiment_name': exp_name}
                    my_model.update(eval_scores)
                    my_model.update({'f1score_mean': performance_matrix['f1score_mean']})
                    # Remove model to reduce file size
                    del my_model['weightID'], my_model['beta']

                    # Save model
                    my_util.save_numpy(my_model, my_save_directory, my_save_name)

                    del tmp_distanceFunc, tmp_hiddenNodePerc, tmp_regC
                    del weights, weightID, beta, label_classes, predictedScores, predictedY
                    del eval_scores, performance_matrix
                
                # Bind a model performance result into the table
                cv_results = cv_results.append({'fold': kfold_idx, 
                'distanceFunc': my_model['distanceFunc'], 
                'hiddenNodePerc': my_model['hiddenNodePerc'], 
                'regC': my_model['regC'], 
                'auc': my_model['auc_mean'],
                'f1score': my_model['f1score_mean']}, 
                ignore_index=True)

                count_run = count_run + 1
                print( str(count_run) + '/' + total_run )

                del my_model, my_save_path, my_save_name

        [cv_results, avg_cv_results] = my_util.average_gridsearch(cv_results, ['distanceFunc', 'hiddenNodePerc', 'regC'])

        return cv_results, avg_cv_results

    def do_gridsearch_parallel(self, gs_idx, trainingDataX, trainingDataY, trainingDataID, gs_param, other_param):
        # Prepare used parameters
        tmp_distanceFunc = gs_param[0]
        tmp_hiddenNodePerc = gs_param[1]
        if tmp_hiddenNodePerc.find('.') == -1:
            tmp_hiddenNodePerc = tmp_hiddenNodePerc.astype(int)
        else:
            tmp_hiddenNodePerc = tmp_hiddenNodePerc.astype(float)
        tmp_regC = gs_param[2].astype(float)
        
        # Prepare save filename
        my_save_name = ('kf_' + str(other_param['kfold_idx']) + '_dtf_' + str(tmp_distanceFunc) + '_hdn_' + str(tmp_hiddenNodePerc) + '_rc_' + str(tmp_regC) + '_rd_' + str(other_param['randomseed']) )
        my_save_name = my_save_name.replace(".", "d")
        my_save_path = my_util.join_path(other_param['my_save_directory'], (my_save_name + '.npy'))
        
        if my_util.is_path_available(my_save_path):
            # Load finished experiment
            my_model = my_util.load_numpy_file(my_save_path)

        else:
            # Run experiment

            # Train model
            [weights, weightID, beta, label_classes, training_time] = self.train(trainingDataX[other_param['kfold_train_data_spliter']], 
            trainingDataY[other_param['kfold_train_data_spliter']], 
            trainingDataID=trainingDataID[other_param['kfold_train_data_spliter']], 
            distanceFunc=tmp_distanceFunc, 
            hiddenNodePerc=tmp_hiddenNodePerc, 
            regC=tmp_regC, 
            randomseed=other_param['randomseed'])

            # Test model
            [predictedScores, predictedY, test_time] = self.predict(trainingDataX[other_param['kfold_test_data_spliter']], weights, beta, tmp_distanceFunc, label_classes)

            # Evaluate performance
            # AUC
            eval_scores = my_util.cal_auc(trainingDataY[other_param['kfold_test_data_spliter']], predictedScores, label_classes)
            # Accuracy
            eval_scores['accuracy'] = my_util.cal_accuracy(trainingDataY[other_param['kfold_test_data_spliter']], predictedY)
            # Performance matrix
            performance_matrix = my_util.eval_classification_performance(trainingDataY[other_param['kfold_test_data_spliter']], predictedY, label_classes)
            
            # Prepare model to save
            my_model = {'distanceFunc':tmp_distanceFunc, 'hiddenNodePerc': tmp_hiddenNodePerc, 'regC':tmp_regC, 'randomseed': other_param['randomseed'], 'weightID': weightID, 'beta': beta, 'label_classes': label_classes, 'training_time': training_time, 'test_time': test_time, 'algorithm': 'welm', 'experiment_name': other_param['exp_name']}
            my_model.update(eval_scores)
            my_model.update({'f1score_mean': performance_matrix['f1score_mean']})
            # Remove model to reduce file size
            del my_model['weightID'], my_model['beta']

            # Save model
            my_util.save_numpy(my_model, other_param['my_save_directory'], my_save_name)
        
        # Bind a model performance result into the table
        tmp_cv_results = {'fold': other_param['kfold_idx'], 
        'distanceFunc': my_model['distanceFunc'], 
        'hiddenNodePerc': my_model['hiddenNodePerc'], 
        'regC': my_model['regC'], 
        'auc': my_model['auc_mean'],
        'f1score': my_model['f1score_mean']}

        print('Fold: ' + str(other_param['kfold_idx']) + ', gs_idx: ' + str(gs_idx+1) + '/' +  other_param['total_run'])

        return tmp_cv_results

    def grid_search_cv_parallel(self, trainingDataX, trainingDataY, trainingDataID, param_grid, exp_path, exp_name, **kwargs):

        # Assign params
        modelParams = {}
        modelParams['num_cores']  = multiprocessing.cpu_count()
        modelParams['cv'] = 5
        modelParams['cv_run'] = -1 # -1 = run all seed, else, run only define
        modelParams['randomseed'] = 0
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))
        del key, value

        # Run only desire number in cv
        if modelParams['cv_run'] == -1:
            cv_run = list(range(0, modelParams['cv']))
        else:
            if isinstance(modelParams['cv_run'], list):
                cv_run = modelParams['cv_run']
            else:
                cv_run = [modelParams['cv_run']]

        # Get save directory
        my_save_directory = exp_path + exp_name
        
        # Generate k-Fold indices
        kfold_data_spliter = StratifiedKFold(n_splits=modelParams['cv'], shuffle=True, random_state=modelParams['randomseed'])
        kfold_data_spliter.get_n_splits(trainingDataX, trainingDataY)
        kfold_train_data_spliter = list()
        kfold_test_data_spliter = list()
        for kfold_train_index, kfold_test_index in kfold_data_spliter.split(trainingDataX, trainingDataY):
            kfold_train_data_spliter.append(kfold_train_index)
            kfold_test_data_spliter.append(kfold_test_index)

        # Combine params all of possible combination
        param_list = np.array(np.meshgrid(param_grid['distanceFunc'], param_grid['hiddenNodePerc'], param_grid['regC'])).T
        param_list = param_list.reshape(-1, param_list.shape[-1])

        # Init result
        cv_results = pd.DataFrame(columns=['fold', 'distanceFunc', 'hiddenNodePerc', 'regC', 'auc', 'f1score']) # define column names

        # Run k-fold
        total_run = str(param_list.shape[0])
        for kfold_idx in cv_run:

            other_param = {'kfold_idx':kfold_idx, 'randomseed':modelParams['randomseed'], 'exp_path':exp_path,'exp_name':exp_name, 'kfold_train_data_spliter':kfold_train_data_spliter[kfold_idx], 'kfold_test_data_spliter':kfold_test_data_spliter[kfold_idx], 'my_save_directory':my_save_directory, 'total_run':total_run}
            
            # Run grid search
            tmp_cv_results = Parallel(n_jobs=modelParams['num_cores'])(delayed(self.do_gridsearch_parallel)(gs_idx, trainingDataX, trainingDataY, trainingDataID, param_list[gs_idx], other_param) for gs_idx in range(0, param_list.shape[0]))
            # for gs_param in param_list:
            #     cv_results = self.do_gridsearch_parallel(trainingDataX, trainingDataY, trainingDataID, gs_param, other_param)
            
            cv_results = cv_results.append(pd.DataFrame(tmp_cv_results), ignore_index=True)
            del tmp_cv_results
                
        # Average cv_results
        [cv_results, avg_cv_results] = my_util.average_gridsearch(cv_results, ['distanceFunc', 'hiddenNodePerc', 'regC'])

        return cv_results, avg_cv_results