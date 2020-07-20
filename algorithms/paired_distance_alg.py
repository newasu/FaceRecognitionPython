import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import paired_distances
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import my lib
import others.utilities as my_util

class paired_distance_alg(object):

    def train(self, trainingDataX_anchor, trainingDataX_compared, trainingDataY, unique_y, threshold_decimal=2, distance_metric='euclidean'):
        tic = my_util.time_counter()
        # Calculate distance
        dist_mat = paired_distances(trainingDataX_anchor, trainingDataX_compared, metric=distance_metric)
        # dist_min = np.floor(dist_mat.min() * (10**threshold_decimal))/(10**threshold_decimal)
        # dist_max = np.ceil(dist_mat.max() * (10**threshold_decimal))/(10**threshold_decimal)
        # threshold_step = 10**-threshold_decimal
        # # Vary threshold
        # dist_confusion = np.empty((0, 3), np.float64)
        # for threshold_idx in np.arange(dist_min, (dist_max+threshold_step), threshold_step):
        #     thresholded_distance_class = np.tile(unique_y['neg'], trainingDataY.shape)
        #     thresholded_distance_class[dist_mat < threshold_idx] = unique_y['pos']
        #     [tn, fp, fn, tp] = confusion_matrix(trainingDataY, thresholded_distance_class).ravel()
        #     dist_confusion = np.append(dist_confusion, np.expand_dims(np.array([threshold_idx, tp, tn]), axis=0), axis=0)
        # del dist_mat, dist_min, dist_max, thresholded_distance_class, tn, fp, fn, tp
        # # Normalize accuracy matrix
        # dist_confusion[:,1] = dist_confusion[:,1]/max(dist_confusion[:,1])
        # dist_confusion[:,2] = dist_confusion[:,2]/max(dist_confusion[:,2])
        # # Find best threshold by finding crossing point between two lines
        # [intersection_x, intersection_y] = my_util.line_intersection(dist_confusion[:,0], dist_confusion[:,1], dist_confusion[:,0], dist_confusion[:,2])
        # classifier_threshold = intersection_x[0]
        classifier_threshold = my_util.find_optimal_threshold_two_clases(dist_mat, trainingDataY, unique_y, threshold_decimal=threshold_decimal)
        classifier_threshold = classifier_threshold[0]
        toc = my_util.time_counter()
        # Plot
        # plt.plot(dist_confusion[:,0], dist_confusion[:,1], '-')
        # plt.plot(dist_confusion[:,0], dist_confusion[:,2], '-')
        # plt.plot(intersection_x, intersection_y, 'o')
        # plt.show()
        # plt.close()
        # Timer
        run_time = toc-tic
        return classifier_threshold, run_time

    def predict(self, trainingDataX_anchor, trainingDataX_compared, trainingDataY, unique_y, classifier_threshold, distance_metric='euclidean'):
        tic = my_util.time_counter()
        predictedScores = paired_distances(trainingDataX_anchor, trainingDataX_compared, metric=distance_metric)
        predictedY = np.tile(unique_y['neg'], trainingDataY.shape)
        predictedY[predictedScores < classifier_threshold] = unique_y['pos']
        toc = my_util.time_counter()
        # Timer
        run_time = toc-tic
        return predictedScores, predictedY, run_time
    
    def grid_search_cv_parallel(self, kfold_training_idx, kfold_test_idx, trainingDataX, trainingDataY, trainingDataID, param_grid, exp_name, **kwargs):
        # Assign params
        modelParams = {}
        modelParams['cv_run'] = -1 # -1 = run all seed, else, run only defined seed
        modelParams['randomseed'] = 0
        modelParams['threshold_decimal'] = 2
        # modelParams['threshold_step'] = 10**-modelParams['threshold_decimal']
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

        # Init result
        cv_results = pd.DataFrame(columns=['fold', 'distanceFunc', 'classifier_threshold', 'auc', 'f1score']) # initial dataframe
        
        # Run k-fold
        for kfold_idx in cv_run:
            
            # Construct triplet training dataset
            triplet_paired_list = my_util.triplet_loss_paring(trainingDataID[kfold_training_idx[kfold_idx]], trainingDataY[kfold_training_idx[kfold_idx]], randomseed=modelParams['randomseed'])
            [combined_training_xx, combined_training_yy, combined_training_id] = my_util.combination_rule_paired_list(trainingDataX[kfold_training_idx[kfold_idx]], trainingDataID[kfold_training_idx[kfold_idx]], triplet_paired_list, combine_rule='concatenate')
            
            # Construct triplet test dataset
            triplet_paired_list = my_util.triplet_loss_paring(trainingDataID[kfold_test_idx[kfold_idx]], trainingDataY[kfold_test_idx[kfold_idx]], randomseed=modelParams['randomseed'])
            [combined_test_xx, combined_test_yy, combined_test_id] = my_util.combination_rule_paired_list(trainingDataX[kfold_test_idx[kfold_idx]], trainingDataID[kfold_test_idx[kfold_idx]], triplet_paired_list, combine_rule='concatenate')
            
            tmp_training_idx = np.arange(0, combined_training_yy.size)
            tmp_test_idx = np.arange(combined_training_yy.size, (combined_training_yy.size+combined_test_yy.size))

            # Prepare variable
            sep_idx = int(combined_training_xx.shape[1]/2)
            unique_class = {'pos':'POS', 'neg':'NEG'}

            # Train
            [classifier_threshold, training_time] = self.train(combined_training_xx[:,0:sep_idx], combined_training_xx[:,sep_idx:], combined_training_yy, unique_class, threshold_decimal=modelParams['threshold_decimal'], distance_metric=param_grid['distanceFunc'])
            
            # plt.plot(dist_confusion[:,0], dist_confusion[:,1], '-')
            # plt.plot(dist_confusion[:,0], dist_confusion[:,2], '-')
            # plt.plot(intersection_x, intersection_y, 'o')
            # plt.show()
            # plt.close()
            
            # Test
            [predictedScores, predictedY, test_time] = self.predict(combined_test_xx[:,0:sep_idx], combined_test_xx[:,sep_idx:], combined_test_yy, unique_class, classifier_threshold, distance_metric=param_grid['distanceFunc'])
            
            # Evaluate performance
            # AUC
            eval_scores = my_util.binary_classes_auc(combined_test_yy, predictedScores, unique_class['pos'])
            # Performance matrix
            performance_matrix = my_util.classification_performance_metric(combined_test_yy, predictedY, np.array(list(unique_class.values())))
            eval_scores.update({'f1score_mean': performance_matrix['f1score_mean']})
            eval_scores.update({'accuracy': performance_matrix['accuracy']})
            
            # Bind a model performance result into the table
            tmp_cv_results = {'fold': kfold_idx, 'distanceFunc': param_grid['distanceFunc'], 'classifier_threshold': classifier_threshold, 'auc': eval_scores['auc'], 'f1score': eval_scores['f1score_mean']}
            cv_results = cv_results.append([tmp_cv_results], ignore_index=True)
            
            del eval_scores, tmp_cv_results
                
        # Average cv_results
        [cv_results, avg_cv_results] = my_util.average_gridsearch(cv_results, ['distanceFunc', 'classifier_threshold'], eval_metric=[['auc', 'descending'], ['f1score', 'descending']])

        return cv_results, avg_cv_results