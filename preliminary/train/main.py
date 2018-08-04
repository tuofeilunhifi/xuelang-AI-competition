import os
import shutil
import pandas as pd
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from Model import getNet
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
#获取数据

batch_size= 8
imgsize=(1920,1920)
INIT_LR = 1e-4
#GPU_COUNT = 6

model=getNet()
#model=multi_gpu_model(model,GPU_COUNT)
optimizer = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
model.summary()

base_log_dir = '/home/jychen/xuelang/code_8/logs'
os.makedirs(base_log_dir, exist_ok=True)
log_dir = os.path.join(base_log_dir, "md")
shutil.rmtree(log_dir, ignore_errors=True)
os.makedirs(log_dir, exist_ok=True)
model_save_path = os.path.join(base_log_dir, "1920_480_6.h5")
callbacks = [
    TensorBoard(log_dir, batch_size=batch_size),
    ModelCheckpoint(model_save_path, monitor='acc', verbose=1, save_best_only=True,period=1),
    ReduceLROnPlateau('acc', factor=0.1, patience=3, verbose=1, mode='max'),
    EarlyStopping('acc', patience=5, mode='max')
]

model.load_weights(model_save_path)


train_datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by dataset std
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in 0 to 180 degrees
    width_shift_range=0.2,  # randomly shift images horizontally
    height_shift_range=0.2,  # randomly shift images vertically
    shear_range=0.,  # set range for random shear
    zoom_range=0.1,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    fill_mode='nearest',  # set mode for filling points outside the input boundaries
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=True,  # randomly flip images
    rescale=1./255,  # set rescaling factor (applied before any other transformation)
    preprocessing_function=None,  # set function that will be applied on each input
    data_format=None,  # image data format, either "channels_first" or "channels_last"
)  # fraction of images reserved for validation (strictly between 0 and 1)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        '/home/jychen/xuelang/code_2/code/data_4/imgtrain/',
        target_size=imgsize,
        batch_size=batch_size,
        class_mode='binary',
        interpolation="bicubic"

        )

validation_generator = val_datagen.flow_from_directory(
        '/home/jychen/xuelang/code_2/code/data_4/imgval/',
        target_size=imgsize,
        batch_size=batch_size,
        class_mode='binary',
        interpolation="bicubic"
        )


#eval_datagen = ImageDataGenerator(rescale=1./255)
#evalidation_generator = eval_datagen.flow_from_directory(
#        'data/data4/val2/',
#        target_size=imgsize,
#        batch_size=batch_size,
#        class_mode='binary',
#        interpolation="bicubic"
#       )


H = model.fit_generator(
         train_generator,
         epochs=20,
         validation_data=validation_generator,
         callbacks=callbacks,
         workers=4
         )

plt.style.use("ggplot")
plt.figure()
N = 20
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Invoice classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

def pred():
    test_datagen = ImageDataGenerator(rescale=1./255)

    test_generator = test_datagen.flow_from_directory(
            'data/test/',
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

    data=pd.DataFrame({
                    "filename":namelist,
                    "probability":rt
    })
    print(data.head(10))
    data.to_csv("rt.csv",index=False)



# rt=model.evaluate_generator(generator=evalidation_generator,workers=4,use_multiprocessing=True)
# print(rt)


#pred()


