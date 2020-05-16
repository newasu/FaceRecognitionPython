
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import libs
import numpy as np
# import pandas as pd
from WELM.welm import welm

# Param settings
randomseed = 25
hiddenNodePerc = 100
regC = 1
distanceFunc = 'euclidean'

v1 = np.array([1, 1, 1, 1, 1])
v2 = np.array([2, 2, 2, 2, 2])
v3 = np.array([3, 3, 3, 3, 3])
v4 = np.array([4, 4, 4, 4, 4])

my_trainingDataX = np.matrix([v1, v2, v3, v4])
my_trainingDataID = np.squeeze(np.asarray(my_trainingDataX[:,1]))
my_trainingDataY = np.array([0, 1, 1, 1])
my_trainingDataY = my_trainingDataY.astype(str)
my_trainingDataY[ my_trainingDataY == '1' ] = 'POS'
my_trainingDataY[ my_trainingDataY == '0' ] = 'NEG'

v5 = np.array([5, 5, 5, 5, 5])
v6 = np.array([6, 6, 6, 6, 6])
v7 = np.array([7, 7, 7, 7, 7])
my_testDataX = np.matrix([v5, v6, v7])
my_testDataID = np.squeeze(np.asarray(my_testDataX[:,1]))
my_testDataY = np.array([1, 1, 0])
my_testDataY = my_testDataY.astype(str)
my_testDataY[ my_testDataY == '1' ] = 'POS'
my_testDataY[ my_testDataY == '0' ] = 'NEG'

print()
# print(my_trainingDataX)
# print(my_trainingDataY)
# print(my_trainingDataID)

m1 = welm()
[weights, weightID, beta, label_classes] = m1.train(my_trainingDataX, my_trainingDataY, trainingDataID=my_trainingDataID, hiddenNodePerc=hiddenNodePerc, regC=regC, randomseed=randomseed, distanceFunc=distanceFunc)
[predictedScores, predictedY] = m1.predict(my_testDataX, weights, beta, distanceFunc, label_classes)

print(predictedScores)
print(predictedY)

