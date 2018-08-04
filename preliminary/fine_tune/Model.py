from keras.layers import Input, Dense, concatenate, Softmax, multiply, Embedding, Reshape, Lambda,Add
from keras.models import Model
# from CapsLayout import capsLayer
from keras.layers import *
import tensorflow as tf
from xception import Xception
from keras.applications.nasnet import NASNetLarge
def getNet():
    input1 = Input(shape=(1920,1920,3), name="input1")
    x=SeparableConv2D(3,(3,3),strides=2,padding="same",activation="relu")(input1)
    x=SeparableConv2D(3,(3,3),strides=2,padding="same",activation="relu")(x)
    # x=BatchNormalization()(x)
    x=Xception(weights=None,input_shape=(480,480,3),include_top=False)(x)
    # x=NASNetLarge(weights="imagenet",input_shape=(331,331,3),include_top=False)(x)
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)
    x=GlobalAveragePooling2D()(x)
    x=Dense(1,activation="sigmoid")(x)

    # 编译模型
    model = Model(inputs=input1, outputs=x)
    return model

