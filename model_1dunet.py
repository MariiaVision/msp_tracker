import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def unet(pretrained_weights = None,input_size = (100,1)):
    inputs = Input(input_size)
    conv1 = Conv1D(32, 1, activation = 'relu')(inputs)
    pool1 = MaxPooling1D(2)(conv1)
    conv2 = Conv1D(64, 1, activation = 'relu')(pool1)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling1D(2)(drop2)

    conv3 = Conv1D(128, 1, activation = 'relu')(pool2)
    drop3 = Dropout(0.5)(conv3)


    up7 = Conv1D(64, 1, activation = 'relu')(UpSampling1D(2)(drop3))
    merge7 = concatenate([conv2,up7],axis=2)
    conv7 = Conv1D(64, 1, activation = 'relu')(merge7)

    up9 = Conv1D(32, 1, activation = 'relu')(UpSampling1D(2)(conv7))
    merge9 = concatenate([conv1,up9],axis=2)
    conv9 = Conv1D(32, 1, activation = 'relu')(merge9)
    conv10 = Conv1D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)
    
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])


    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


