# author by LYS 2017/5/24
# for Deep Learning course
'''
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
'''
import os, random, shutil
 
 
def copyFile(fileDir):
    # 1
	pathDir = os.listdir(fileDir)
 
    # 2
	sample = random.sample(pathDir, 132)
	print (sample)
	
	# 3
	for name in sample:
		shutil.move(fileDir+name, tarDir+name)
if __name__ == '__main__':
	fileDir = r"C:/Users/18292/Desktop/xuelang_dataset/process/train_normal/1/"
	tarDir = r'C:/Users/18292/Desktop/xuelang_dataset/process/train_normal/11/'
	copyFile(fileDir)