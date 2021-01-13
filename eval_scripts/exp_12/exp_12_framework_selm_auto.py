
-0.4293732888327213
-0.6948100835852529
-1.1014336426087539

info_data[0:3,0:4]
np.mean(proposed_model['male-black'].predict(x_data[0:3,:]), axis=1)

np.mean(exacted_data.iloc[0:3,8:].values, axis=1)

filename_comment = 'eer'
param = {'exp':'exp_7', 
         'model': ['b_180_e_50_a_1', 'b_180_e_50_a_1', 'b_240_e_50_a_1', 'b_360_e_50_a_1', 'b_270_e_50_a_1', 'b_240_e_50_a_1'], 
         'epoch': [36, 32, 42, 25, 35, 28], 
         'class': ['female-asian', 'female-black', 'female-caucasian', 'male-asian', 'male-black', 'male-caucasian']}