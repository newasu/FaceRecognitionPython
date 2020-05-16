
import os

my_path = '/Users/Wasu/Downloads/CelebA(partial)_1/img_align_celeba/'
onlyfiles = next(os.walk(my_path))[2] #dir is your directory path as string
print(len(onlyfiles))

tmp_1 = 0
img_list = list()
for r, d, f in os.walk(my_path):
  for file in f:
    img_list.append(file)
    tmp_1 = tmp_1 + 1
    print(tmp_1)


with open('/Users/Wasu/Downloads/CelebA/img_align_celeba/img_name.txt', 'w') as f:
    for item in img_list:
        f.write("%s\n" % item)