# Add project path to sys
import sys
sys.path.append("./././")

# Import lib
import pandas as pd
import numpy as np
from shutil import copyfile

# Import my own lib
import others.utilities as my_util

#############################################################################################

# Path
# Dataset path
# dataset_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'Diveface'])
lfw_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw', 'lfw-deepfunneled'])
lfw_label_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw'])
lfw_test_img_path = my_util.get_path(additional_path=['.', '.', 'mount', 'FaceRecognitionPython_data_store', 'Dataset', 'lfw', 'DevTest'])

#############################################################################################

# Function
def query_and_copy(_converted_label, qr_person, qr_imagenum, qr_name, lfw_path, lfw_test_img_path):
    print(qr_name + ' ' + qr_person + ' ' + str(qr_imagenum))
    tmp_query = _converted_label[(_converted_label.person == qr_person) & (_converted_label.imagenum == qr_imagenum)].iloc[0]
    tmp_filename = qr_name + '_' + str(tmp_query.imagenum).zfill(4) + '.jpg'
    source_path = lfw_path + qr_name + '/' + tmp_filename
    destination_path = lfw_test_img_path + tmp_query.race + '/' + tmp_filename
    copyfile(source_path, destination_path)

#############################################################################################

# Read txt
my_label = pd.read_csv(lfw_label_path + 'lfw_attributes.txt', header=0, sep='\t', usecols=['person', 'imagenum', 'Male', 'Asian', 'Indian', 'Black', 'White'])
pairsDevTest_POS = pd.read_csv(lfw_label_path + 'pairsDevTest_POS.txt', header=None, sep='\t')
pairsDevTest_NEG = pd.read_csv(lfw_label_path + 'pairsDevTest_NEG.txt', header=None, sep='\t')

# Assign label
my_gender = np.tile(np.array('male', dtype=object), my_label.shape[0])
my_gender[my_label['Male']<0] = 'female'
my_ethnicity_idx = np.argmax(my_label[['Asian', 'Indian', 'Black', 'White']].values, axis=1)
master_ethnicity = ['asian', 'black', 'black', 'caucasian']
my_ethnicity = []
my_race = []
for i in range(0,my_ethnicity_idx.size):
    my_ethnicity.append(master_ethnicity[my_ethnicity_idx[i]])
    my_race.append(my_gender[i] + '-' + my_ethnicity[i])
    
converted_label = pd.DataFrame(data={'person':my_label['person'], 'imagenum':my_label['imagenum'], 'gender':my_gender, 'ethnicity':my_ethnicity, 'race':my_race})

# Test list POS
pairsDevTest_POS_1 = pairsDevTest_POS[[0, 1]].copy()
pairsDevTest_POS_2 = pairsDevTest_POS[[0, 2]].copy()
pairsDevTest_POS_1.columns = ['name', 'imagenum']
pairsDevTest_POS_2.columns = ['name', 'imagenum']
pairsDevTest_POS_1['person'] = pairsDevTest_POS_1.name.replace('_', ' ', regex=True)
pairsDevTest_POS_2['person'] = pairsDevTest_POS_2.name.replace('_', ' ', regex=True)

# Test list NEG
pairsDevTest_NEG_1 = pairsDevTest_NEG[[0, 1]].copy()
pairsDevTest_NEG_2 = pairsDevTest_NEG[[2, 3]].copy()
pairsDevTest_NEG_1.columns = ['name', 'imagenum']
pairsDevTest_NEG_2.columns = ['name', 'imagenum']
pairsDevTest_NEG_1['person'] = pairsDevTest_NEG_1.name.replace('_', ' ', regex=True)
pairsDevTest_NEG_2['person'] = pairsDevTest_NEG_2.name.replace('_', ' ', regex=True)

# Make DevTest folders
for fol in converted_label.race.unique():
    my_util.make_directory(lfw_test_img_path + fol)

# Fix label POS
tmp = converted_label[converted_label.person == 'Abdullah Gul'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':16, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Jake Gyllenhaal'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':2, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Giuseppe Gibilisco'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':2, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Will Smith'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':2, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Jose Dirceu'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':2, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Luis Horna'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':2, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Richard Gere'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':1, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
tmp = converted_label[converted_label.person == 'Sean OKeefe'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':5, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)

# Fix label NEG
tmp = converted_label[converted_label.person == 'Cesar Maia'].iloc[0]
converted_label = converted_label.append({'person':tmp.person, 'imagenum':2, 'gender':tmp.gender, 'ethnicity':tmp.ethnicity, 'race':tmp.race}, ignore_index=True)
# tmp = converted_label[converted_label.person == 'Jeffrey Pfeffer'].iloc[0]
converted_label = converted_label.append({'person':'Jeffrey Pfeffer', 'imagenum':1, 'gender':'male', 'ethnicity':'caucasian', 'race':'male-caucasian'}, ignore_index=True)
# tmp = converted_label[converted_label.person == 'Luis Pujols'].iloc[0]
converted_label = converted_label.append({'person':'Luis Pujols', 'imagenum':1, 'gender':'male', 'ethnicity':'black', 'race':'male-black'}, ignore_index=True)
# tmp = converted_label[converted_label.person == 'Marricia Tate'].iloc[0]
converted_label = converted_label.append({'person':'Marricia Tate', 'imagenum':1, 'gender':'female', 'ethnicity':'black', 'race':'female-black'}, ignore_index=True)

# Query and Copy POS
for i in range(0,pairsDevTest_POS_1.shape[0]):
    print(str(i))
    query_and_copy(converted_label, pairsDevTest_POS_1.person.iloc[i], pairsDevTest_POS_1.imagenum.iloc[i], pairsDevTest_POS_1.name.iloc[i], lfw_path, lfw_test_img_path)
    query_and_copy(converted_label, pairsDevTest_POS_2.person.iloc[i], pairsDevTest_POS_2.imagenum.iloc[i], pairsDevTest_POS_2.name.iloc[i], lfw_path, lfw_test_img_path)


# Query and Copy NEG
for i in range(0,pairsDevTest_NEG_1.shape[0]):
    print(str(i))
    query_and_copy(converted_label, pairsDevTest_NEG_1.person.iloc[i], pairsDevTest_NEG_1.imagenum.iloc[i], pairsDevTest_NEG_1.name.iloc[i], lfw_path, lfw_test_img_path)
    query_and_copy(converted_label, pairsDevTest_NEG_2.person.iloc[i], pairsDevTest_NEG_2.imagenum.iloc[i], pairsDevTest_NEG_2.name.iloc[i], lfw_path, lfw_test_img_path)

print()
