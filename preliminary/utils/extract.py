#coding:utf-8

import os 
import shutil

path = r'C:/Users/18292/Desktop/xuelang_dataset/process/local/Cropedflaw_2/'
new_path = r'C:/Users/18292/Desktop/xuelang_dataset/process/local/Cropedflaw_22/'

for root,dirs,files in os.walk(path):
    for i in range(len(files)):
        if(files[i][-3:] == 'jpg'):
            file_path = root+'/'+files[i]
            new_file_path = new_path+'/'+files[i]
            shutil.copy(file_path,new_file_path)