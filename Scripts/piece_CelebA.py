
import pandas as pd
import cv2
import shutil
import numpy as np

# Params
# number_user = range(0, 500)
# number_user = range(500, 1000)
number_user = range(2000, 3000)

# File paths
id_path = '/Users/Wasu/Downloads/CelebA/Anno/identity_CelebA.txt'
file_path = '/Users/Wasu/Downloads/CelebA/img_align_celeba'
bbox_path = '/Users/Wasu/Downloads/CelebA/Anno/list_bbox_celeba.txt'
sub_dataset = '/Users/Wasu/Downloads/CelebA(partial)_3'

# Read files
id_df = pd.read_csv(id_path, sep=" ", header=None)
id_df.columns = ['image_id', 'id']
id_df_unique = id_df.id.sort_values().unique()
bbox_df = pd.read_csv(bbox_path, delim_whitespace=True, skiprows=1)

# Init var
tmp_user_label = list()

# user id
# tmp_user_id = 1 # from 1 : id_df_unique.size
for tmp_user_id in number_user:
    # find user in dataset
    tmp_user_idx = id_df.index[id_df[:]['id'] == tmp_user_id+1]
    tmp_user_idx = id_df.iloc[tmp_user_idx]

    # Lookup into each user
    # tmp_user_sub_idx = 0 # from 0 : tmp_user_idx.size
    for tmp_user_sub_idx in range(tmp_user_idx.shape[0]):
        # Read image
        # temp_img = cv2.imread(file_path + '/' + tmp_user_idx['id_name'].iloc[tmp_user_sub_idx])

        # Draw image
        # cv2.imshow('image', temp_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Copy file
        original = file_path + '/' + tmp_user_idx['image_id'].iloc[tmp_user_sub_idx]
        target = sub_dataset + '/img_align_celeba/' + tmp_user_idx['image_id'].iloc[tmp_user_sub_idx]
        shutil.copyfile(original, target)

        # Add label to list
        tmp_user_label.append([tmp_user_idx.iloc[tmp_user_sub_idx].image_id, tmp_user_idx.iloc[tmp_user_sub_idx].id])
        
    print(tmp_user_id+1)

# Write label file
# tmp_user_label = pd.DataFrame(tmp_user_label)
with open(sub_dataset + '/Anno/identity_CelebA.txt', 'w+') as datafile_id:
    np.savetxt(datafile_id, tmp_user_label, fmt=['%s','%s'])

print('Finished')


