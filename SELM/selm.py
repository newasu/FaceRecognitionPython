import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from collections import Counter
from scipy import linalg

class selm(object):

    def __init__(self):
        pass

    def train(self, trainingDataX, trainingDataY, trainingDataID, **kwargs):

        # Check number of data sizes
        if trainingDataX.shape[0] != trainingDataY.shape[0] or trainingDataX.shape[0] != trainingDataID.shape[0]:
            raise Exception('number of data size is not equal')

        # Assign params
        modelParams = {}
        modelParams['hiddenNodePerc'] = 100
        modelParams['regC'] = 1
        modelParams['randomseed'] = 1
        modelParams['distanceFunc'] = 'euclidean'
        for key, value in kwargs.items():
            if key in modelParams:
                modelParams[key] = value
            else:
                raise Exception('Error key ({}) exists in dict'.format(key))

        [weights, trainingWeightDataID] = self.initHidden(trainingDataX, trainingDataID, modelParams['hiddenNodePerc'], modelParams['randomseed'])
        [beta, label_classes] = self.trainModel(trainingDataX, trainingDataY, weights, modelParams['regC'], modelParams['distanceFunc'])

        # print(modelParams)
        # print(weights)
        # print(trainingWeightDataID)
        # print(beta)

        return weights, trainingWeightDataID, beta, label_classes

    def initHidden(self, trainingDataX, trainingDataID, hiddenNodePerc, randomseed):
        NumHiddenNode = round((hiddenNodePerc/100) * trainingDataX.shape[0])
        NumHiddenNode = min(NumHiddenNode, trainingDataX.shape[0])
        if NumHiddenNode == 0:
            NumHiddenNode = 1
            
        random.seed( randomseed )
        weightIdx = random.sample(range(trainingDataX.shape[0]), NumHiddenNode)
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
        penalized_array = np.matrix(np.tile(penalized_array, [1, 1, penalized_array.shape[0]]))

        H = np.multiply(penalized_array, simKernel)
        del penalized_array, simKernel
        Two_H = H.T * H
        regC_mat = (1/regC) * np.identity(H.shape[0])
        # inv_data = np.linalg.pinv(Two_H + regC_mat)
        # inv_data = linalg.inv(Two_H + regC_mat)
        inv_data = self.cal_inv_func(Two_H + regC_mat)
        del Two_H, regC_mat
        beta = inv_data * (H.T * (trainingDataY_onehot * penalized_value))

        return beta, label_classes

    def predict(self, testDataX, weights, beta, distanceFunc, label_classes):
        simKernel = self.calculate_kernel(testDataX, weights, distanceFunc)
        predictedScores = simKernel * beta
        predictedY = np.argmax(predictedScores, axis=1)
        predictedY = label_classes[predictedY]

        return predictedScores, predictedY

    def calculate_kernel(self, m1, m2, distanceFunc):
        # simKernel = euclidean_distances(m1, m2)
        # simKernel = pairwise_distances(m1, m2, metric='euclidean')
        simKernel = pairwise_distances(m1, m2, metric=distanceFunc)
        
        return simKernel

    def cal_inv_func(self, pass_inv_data):
        # inv_data = np.linalg.pinv(pass_inv_data)
        temp_inv_data = linalg.inv(pass_inv_data)

        return temp_inv_data
