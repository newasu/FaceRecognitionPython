
# Add project path to sys
import sys
import pathlib
my_current_path = pathlib.Path(__file__).parent.absolute()
my_root_path = my_current_path.parent
sys.path.insert(0, str(my_root_path))

# Import my lib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

from WELM.welm import welm

dataset_path = '/Users/Wasu/Library/Mobile Documents/com~apple~CloudDocs/newasu\'s Mac/PhD\'s Degree/New/SourceCode/FaceRecognitionPython_data_store/Dataset/CelebA(partial)_1/CelebA_retinaface_1_1000.csv'
my_data = pd.read_csv(dataset_path, sep=",", header=0)
my_data.feature = my_data.feature.str.split(expand=True).astype(float).values.tolist()
# my_data.to_csv('CelebA_retinaface_1_1000(v2).csv', index=False, header=True)

train_idx = []
test_idx = []
for i in range(1, 5):
    temp_all = my_data.index[my_data['id'] == i]
    temp_train = my_data.index[my_data['id'] == i][0:i]
    temp_test = my_data.index[my_data['id'] == i][(i):]
    # print(i)
    # print(temp_all)
    # print(temp_train)
    # print(temp_test)

    train_idx.extend(temp_train)
    test_idx.extend(temp_test)

del temp_all, temp_train, temp_test

# np.matrix(my_data.feature.to_list()).shape
xx_train = np.matrix(my_data.iloc[train_idx].feature.to_list())
yy_train = my_data.iloc[train_idx].id.values.astype(str)
xx_test = np.matrix(my_data.iloc[test_idx].feature.to_list())
yy_test = my_data.iloc[test_idx].id.values.astype(str)
train_sample_id = my_data.iloc[train_idx].image_id.values

del dataset_path, my_data, test_idx, train_idx

m1 = welm()
[weights, weightID, beta, label_classes, training_time] = m1.train(xx_train, yy_train, trainingDataID=train_sample_id, 
    hiddenNodePerc=0.5, regC=1, randomseed=1, distanceFunc='euclidean')
[predictedScores, predictedY, test_time] = m1.predict(xx_test, weights, beta, 'euclidean', label_classes)

print(predictedScores)
print(predictedY)

print()

# accuracy_score(yy_test.tolist(), predictedY.T[0].tolist())
