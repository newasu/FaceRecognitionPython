
import pandas as pd

# id_range = range(0, 10)

# File paths
id_path = '/Users/Wasu/Downloads/CelebA/Anno/identity_CelebA.txt'
image_name = '/Users/Wasu/Downloads/CelebA(partial)_1/features/CelebA_features_1_1000_name.csv'
image_feature = '/Users/Wasu/Downloads/CelebA(partial)_1/features/CelebA_features_1_1000_features.csv'
save_path = '/Users/Wasu/Downloads/CelebA(partial)_1/features/CelebA_retinaface_1_1000.csv'

# Read files
id_df = pd.read_csv(id_path, sep=" ", header=None)
id_df.columns = ['image_id', 'id']
id_df_unique = id_df.id.sort_values().unique()
image_name_df = pd.read_csv(image_name, sep=" ", header=None)
image_feature_df = pd.read_csv(image_feature, sep=",", header=None)

tmp_list = pd.DataFrame(columns=['image_id', 'id', 'feature'])
for tmp_i, _ in image_name_df.iterrows():
    tmp_image_id = image_name_df.iloc[tmp_i].item()
    tmp_id = id_df.loc[id_df['image_id'] == image_name_df.iloc[tmp_i][0]].id.item()
    tmp_feature = image_feature_df.iloc[tmp_i].values
    tmp_list = tmp_list.append({'image_id':tmp_image_id, 'id':tmp_id, 'feature':tmp_feature}, ignore_index=True)
    print(tmp_i)

# tmp_list = pd.read_csv(save_path, sep=",", header=0)
# tmp_list.sort_values(by=['id','image_id'], inplace=True)
tmp_list.to_csv(save_path, index=False, header=True)

print()


