import os
import shutil
import pandas as pd
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from Model import getNet
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
#获取数据

batch_size=16
imgsize=(1920,1920)

model=getNet()
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["acc"])
model.summary()

base_log_dir = '/home/jychen/xuelang/code_5/code/logs'
os.makedirs(base_log_dir, exist_ok=True)
log_dir = os.path.join(base_log_dir, "md")
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)
model_save_path = os.path.join(base_log_dir, "1920_480_6.h5")
callbacks = [
    TensorBoard(log_dir, batch_size=batch_size),
    ModelCheckpoint(model_save_path, monitor='val_acc', verbose=1, save_best_only=True,period=1),
    ReduceLROnPlateau('val_acc', factor=0.1, patience=3, verbose=1, mode='max'),
    EarlyStopping('val_acc', patience=5, mode='max')
]

model.load_weights(model_save_path)#,by_name=True,skip_mismatch=True

def pred():
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            '/home/jychen/xuelang/code_5/code/data/test/',
            target_size=imgsize,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False
            )

    namelist=test_generator.filenames.copy()
    for i in range(len(namelist)):
        namelist[i]=namelist[i].replace("t/","")


    rt=model.predict_generator(test_generator,verbose=True, workers=4, use_multiprocessing=True)
    rt=np.reshape(rt,newshape=(rt.shape[0],))
    rt=list(rt)
    
    pr = []

    for t in rt:
        if t>0.999999:
            t = 0.999999
        if t<0.000001:
            t = 0.000001
        pr.append(t)

    data=pd.DataFrame({
                    "filename":namelist,
                    "probability":pr
    })
    print("head")
    print(data.head(20))
    print("tail")
    print(data.tail(20))
    data.to_csv("/home/jychen/xuelang/code_5/code/data/test/end_2.csv",index=False)

pred()


