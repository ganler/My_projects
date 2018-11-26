import os
import pandas as pd
import shutil

paths = ['0', '1']
dict_ = {'image_name': [], 'tags': []}
new_dir = 'AP'

os.mkdir(new_dir)

for tag, path in enumerate(paths):
    for i, file in enumerate(os.listdir(path)):
        os.rename(os.path.join(path, file), os.path.join(path, path + '-' + str(i) + ".png"))
        dict_['tags'].append(str(tag))
        dict_['image_name'].append(path + '-' + str(i))
        shutil.copy(os.path.join(path, path + '-' + str(i) + ".png"), new_dir)

pd.DataFrame(dict_).to_csv('AP_info.csv')