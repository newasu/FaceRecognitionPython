import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

import multiprocessing
from joblib import Parallel, delayed

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
        modelParams['hiddenNodePerc'] = 1.0
        modelParams['regC'] = 1
        modelParams['randomseed'] = 0
        modelParams['distanceFunc'] = 'euclidean'
        modelParams['trainingDataID'] = np.array(range(0, trainingDataX.shape[0]))
        modelParams['useTF'] = False
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))

        [weights, trainingWeightDataID] = self.initHidden(trainingDataX, modelParams['trainingDataID'], modelParams['hiddenNodePerc'], modelParams['randomseed'])
        tic = my_util.time_counter()
        [beta, label_classes] = self.trainModel(trainingDataX, trainingDataY, weights, modelParams['regC'], modelParams['distanceFunc'], useTF=modelParams['useTF'])
        toc = my_util.time_counter()
        # Timer
        run_time = toc-tic

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

    def trainModel(self, trainingDataX, trainingDataY, weights, regC, distanceFunc, useTF=False):
        simKernel = my_util.calculate_kernel(trainingDataX, weights, distanceFunc, useTF=useTF)

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
        inv_data = np.matrix(my_util.cal_inv_func(Two_H + regC_mat))
        del Two_H, regC_mat
        beta = inv_data * (H.T * (trainingDataY_onehot * penalized_value))

        return beta, label_classes

    def predict(self, testDataX, weights, beta, distanceFunc, label_classes, useTF=False):
        tic = my_util.time_counter()
        simKernel = my_util.calculate_kernel(testDataX, weights, distanceFunc, useTF=useTF)
        predictedScores = simKernel * beta
        toc = my_util.time_counter()
        predictedY = np.argmax(predictedScores, axis=1)
        predictedY = label_classes[predictedY]
        # Timer
        run_time = toc-tic

        return predictedScores, predictedY, run_time

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
            [weights, weightID, beta, label_classes, training_time] = self.train(trainingDataX[other_param['kfold_training_data_idx']], 
            trainingDataY[other_param['kfold_training_data_idx']], 
            trainingDataID=trainingDataID[other_param['kfold_training_data_idx']], 
            distanceFunc=tmp_distanceFunc, 
            hiddenNodePerc=tmp_hiddenNodePerc, 
            regC=tmp_regC, 
            randomseed=other_param['randomseed'],
            useTF=other_param['useTF'])

            # Test model
            [predictedScores, predictedY, test_time] = self.predict(trainingDataX[other_param['kfold_test_data_idx']], weights, beta, tmp_distanceFunc, label_classes, useTF=other_param['useTF'])

            # Evaluate performance
            # AUC
            eval_scores = my_util.cal_auc(trainingDataY[other_param['kfold_test_data_idx']], predictedScores, label_classes)
            # Accuracy
            eval_scores['accuracy'] = my_util.cal_accuracy(trainingDataY[other_param['kfold_test_data_idx']], predictedY)
            # Performance matrix
            performance_matrix = my_util.classification_performance_metric(trainingDataY[other_param['kfold_test_data_idx']], predictedY, label_classes)
            
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

        print('WELM-Fold: ' + str(other_param['kfold_idx']) + ', gs_idx: ' + str(gs_idx+1) + '/' +  other_param['total_run'])

        return tmp_cv_results

    def grid_search_cv_parallel(self, kfold_training_idx, kfold_test_idx, trainingDataX, trainingDataY, trainingDataID, param_grid, exp_path, exp_name, **kwargs):

        # Assign params
        modelParams = {}
        modelParams['num_cores']  = multiprocessing.cpu_count()
        modelParams['useTF'] = False
        # modelParams['cv'] = len(kfold_training_idx)
        modelParams['cv_run'] = -1 # -1 = run all seed, else, run only defined seed
        modelParams['randomseed'] = 0
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))
        del key, value

        # Run only desire number in cv
        if modelParams['cv_run'] == -1:
            cv_run = list(range(0, len(kfold_training_idx)))
        else:
            if isinstance(modelParams['cv_run'], list):
                cv_run = modelParams['cv_run']
            else:
                cv_run = [modelParams['cv_run']]

        # Get save directory
        my_save_directory = exp_path + exp_name

        # Combine params all of possible combination
        param_list = np.array(np.meshgrid(param_grid['distanceFunc'], param_grid['hiddenNodePerc'], param_grid['regC'])).T
        param_list = param_list.reshape(-1, param_list.shape[-1])

        # Init result
        cv_results = pd.DataFrame(columns=['fold', 'distanceFunc', 'hiddenNodePerc', 'regC', 'auc', 'f1score']) # define column names

        # Run k-fold
        total_run = str(param_list.shape[0])
        for kfold_idx in cv_run:

            other_param = {'kfold_idx':kfold_idx, 'randomseed':modelParams['randomseed'], 'exp_path':exp_path,'exp_name':exp_name, 'kfold_training_data_idx':kfold_training_idx[kfold_idx], 'kfold_test_data_idx':kfold_test_idx[kfold_idx], 'my_save_directory':my_save_directory, 'total_run':total_run, 'useTF':modelParams['useTF']}
            
            # Run grid search parallel
            tmp_cv_results = Parallel(n_jobs=modelParams['num_cores'])(delayed(self.do_gridsearch_parallel)(gs_idx, trainingDataX, trainingDataY, trainingDataID, param_list[gs_idx], other_param) for gs_idx in range(0, param_list.shape[0]))
            cv_results = cv_results.append(pd.DataFrame(tmp_cv_results), ignore_index=True)
            
            # Run grid search by for loop
            # for gs_idx in range(0, param_list.shape[0]):
            #     tmp_cv_results = self.do_gridsearch_parallel(gs_idx, trainingDataX, trainingDataY, trainingDataID, param_list[gs_idx], other_param)
            #     cv_results = cv_results.append([tmp_cv_results], ignore_index=True)
            
            # del tmp_cv_results
                
        # Average cv_results
        [cv_results, avg_cv_results] = my_util.average_gridsearch(cv_results, ['distanceFunc', 'hiddenNodePerc', 'regC'])

        return cv_results, avg_cv_results