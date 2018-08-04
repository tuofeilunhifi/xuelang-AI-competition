#coding:utf-8

import os
import random
import shutil
import skimage.io as io

 

def copyFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir)
    for filename in pathDir:
        print (filename)
    str = 'fileDir*.jpg' # fileDir的路径+*.jpg表示文件下的所有jpg图片
    coll = io.ImageCollection(str)
    print(len(coll)) #打印图片数量
    num = 96
    print(num)
    sample = random.sample(pathDir,num)
    for name in sample:
        shutil.copy(fileDir+name,tarDir+name)

if __name__ == '__main__':
   fileDir = r"C:/Users/18292/Desktop/train/train0/" #填写要读取图片文件夹的路径
   tarDir = r"C:/Users/18292/Desktop/train/train00/" #填写保存随机读取图片文件夹的路径

   copyFile(fileDir,tarDir)
   print ('ok')

