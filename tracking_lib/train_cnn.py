#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training CNN of the candidate prunning with new data
"""
from keras import backend as K
import keras
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, ZeroPadding2D, AveragePooling2D, Flatten, MaxPooling2D, Dropout, Conv2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.utils import layer_utils, np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import skimage 
from skimage import io
import argparse
import glob
from tqdm import tqdm

class CnnPrunning:
    '''
    CNN model for the candidates pruning
    '''
    
    def __init__(self):
        self.model=0

    def shallowCNN(self, include_top=True, input_tensor=None, 
                       input_shape=(16,16,1), pooling=None, classes=2, box_size_val=16):
        
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(classes, activation='softmax'))
    
        print(model.summary())
        return model
 

    def create_shallowCNN(self, input_shape_custom=(16,16,3), input_classes=2):
 
      self.model = self. shallowCNN(include_top=True, input_tensor=None, 
                             input_shape=input_shape_custom, pooling=None, classes=input_classes)
      
    def read_from_file(self,  path, box_size=16, label=0):
        segment_array=[]
        label_array=[]
        files=glob.glob(path+"/*")

        for filename in files:
            
            segment = io.imread(filename)
#            segment = skimage.color.rgb2gray(img) 
            segment=(segment-np.min(segment))/(np.max(segment)-np.min(segment))
            segment_array.append(segment)
            label_array.append(label)


        return np.array(segment_array), np.array(label_array)


def save_imgs(file_img, file_txt, path_save, box_size=16, pos=0):
    '''
    extract and save ROIs to the given folders
    '''
    
    def substract_bg_single(img_set, pos, step):
        '''
        substract image and in-situ bg
        '''
        start_i = pos-step # start frame
        if start_i<0:
            start_i=0
        end_i = start_i+2*step # end frame
        if end_i>=img_set.shape[0]:
            end_i=img_set.shape[0]
            start_i=end_i-2*step
    
        insitu=np.min(img_set[start_i:end_i], axis=0) # insitu calculation 
        
        # removing background by substraction
        img_3ch=np.copy(img_set[pos])
        img_new= img_3ch-insitu
            
        return img_new

    # data augmentation 
    datagen = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0,
            height_shift_range=0,
            shear_range=0.2,
            zoom_range=0.0,
            horizontal_flip=True,
            fill_mode='nearest')
    
    img_set=skimage.io.imread(file_img)

    f = open(file_txt, 'r')
    for line in tqdm(f):    
        field = line.split()
        
        frameN=int(int(field[3]))-1
        
        img= substract_bg_single(img_set, frameN, step=100) 
        img=(img-np.min(img))/(np.max(img)-np.min(img))
        
        for new_pos in range(0,2):
        
            x_step=np.random.randint(low=-1, high=2)
            y_step=np.random.randint(low=-1, high=2)
            center=[int(float(field[2])+x_step), int(float(field[1])+y_step)]
            if int(center[0]-box_size/2)>0 and int(box_size/2+center[0])< img.shape[0] and int(center[1]-box_size/2)>0 and int(box_size/2+center[1])<img.shape[1]:
                segment=img[int(center[0]-box_size/2):int(box_size/2+center[0]),int(center[1]-box_size/2):int(box_size/2+center[1])]
        
                x = segment.reshape((1,) + segment.shape)
                x=x.reshape((x.shape[0],x.shape[1],x.shape[2], 1))
                i = 0
                # generate images
                for batch in datagen.flow(x, batch_size=1,
                                          save_to_dir=path_save, save_prefix='particle_'+str(pos)+"_"+str(field[0]), save_format='png'):

                    i += 1
                    if i > 3:
                        break 
    f.close()        
    
def get_args():
    parser = argparse.ArgumentParser(description='train CNN model for MSP detector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-nof', '--number_of_files', type=int, default=1,
                        help='Number of image sequences', dest='number_of_files')
    
    parser.add_argument('-mp', '--movie_path', type=str, default="", nargs='+',
                        help='Path to the image sequence tiff file', dest='movie_path')
    
    parser.add_argument('-dpp', '--positive_coordinates_path', type=str, default="", nargs='+',
                        help='path to the file with vesicle coordinates', dest='positive_coordinates_path')
    
    parser.add_argument('-lpp', '--negative_coordinates_path', type=str, default="", nargs='+',
                        help='path to the file with not-vesicle(negative examples) coordinates', dest='negative_coordinates_path')
    
    parser.add_argument('-d', '--save_images_path', dest='save_images_path', type=str, default=False,
                        help='path where the extracted ROIs will be stored')
    
    parser.add_argument('-rs', '--roi_size', dest='roi_size', type=int, default=16,
                        help='size of the ROI for training')
    
    parser.add_argument('-smp', '--save_model_path', dest='save_model_path', type=str, default="",
                        help='size of the ROI for training')    
    return parser.parse_args()


if __name__ == '__main__':
    
    args = get_args()
    

    #extract, augment and save ROIs
    path_positive=args.save_images_path+"positive"
    path_negative=args.save_images_path+"negative"
    
    print("creating ROI images to ", args.save_images_path, "folder ...")
    for pos in range(0,args.number_of_files):
        
        movie_path=args.movie_path[pos]
        positive_coordinates_path=args.positive_coordinates_path[pos]
        negative_coordinates_path=args.negative_coordinates_path[pos]
        
        
        #positive
        save_imgs(movie_path, positive_coordinates_path, path_positive, args.roi_size, pos)
        
        #negative
        save_imgs(movie_path, negative_coordinates_path, path_negative, args.roi_size, pos)

    
    #training model
    print(" building model and dataset .... ")
    modelCNN=CnnPrunning()
    
    #read the data
    data_x_p2, data_y_p2=modelCNN.read_from_file(path_positive,  box_size=args.roi_size, label=1)
    data_x_p1, data_y_p1=modelCNN.read_from_file(path_negative,  box_size=args.roi_size, label=0)
    
    print("vesicles", data_x_p1.shape)
    print("non-vesicles", data_x_p2.shape)
    
    x_data=np.concatenate((data_x_p1,data_x_p2), axis=0)
    y_data=np.concatenate((data_y_p1,data_y_p2), axis=0)
    y_data = np_utils.to_categorical(y_data, 2)
    
    #normilise
    x_data=x_data/np.max(np.max(x_data))
    
    #reshape
    x_data=x_data.reshape((x_data.shape[0],x_data.shape[1],x_data.shape[2], 1))
    
    #split the data
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=32)
    
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)
    
    modelCNN.create_shallowCNN((X_test.shape[1],X_test.shape[2],1), input_classes=2)
    modelCNN.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc', 'mse'])
    
    #summary
    modelCNN.model.summary()
    
    # checkpoint
    filepath=args.save_model_path+"cnn-model-best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    #tensorboard
    tb_callback=keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
              write_graph=True, write_images=True)
    callbacks_list = [tb_callback, checkpoint]
    
    # training
    crn50 = modelCNN.model.fit(x=X_train, y=y_train, batch_size=args.roi_size, 
            epochs=40, callbacks=callbacks_list, verbose=1, validation_data=(X_test, y_test), shuffle=True)
    
    
    #save the model

    # serialize weights to HDF5
    modelCNN.model.save_weights(args.save_model_path+"cnn-model-last.hdf5")
    print("Saved to the disk ")
