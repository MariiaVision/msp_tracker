#!/bin/bash

conda activate msp

################### provide path to your data below 

NUMBER_FILES=2 # number of image sequences used for the training 

# paths to the image sequence (should be tiff format, single channel)
MOVIE_PATH_1="/path/to/tif/sequence/1"
MOVIE_PATH_2="/path/to/tif/sequence/2"
MOVIE_PATH_3="/path/to/tif/sequence/3"


# paths to the txt file with coordinates of the positive samples
POSITIVE_COORDINATES_PATH_1="/path/to/txt/file/with/positivie/samples/for/sequence 1"
POSITIVE_COORDINATES_PATH_2="/path/to/txt/file/with/positivie/samples/for/sequence 2"
POSITIVE_COORDINATES_PATH_3="/path/to/txt/file/with/positivie/samples/for/sequence 3"


# paths to the txt file with coordinates of the negative samples
NEGATIVE_COORDINATES_PATH_1="/path/to/txt/file/with/negative/samples/for/sequence 1"
NEGATIVE_COORDINATES_PATH_2="/path/to/txt/file/with/negative/samples/for/sequence 2"
NEGATIVE_COORDINATES_PATH_3="/path/to/txt/file/with/negative/samples/for/sequence 3"



# path where the positive and negative samples should be stored
IMAGE_PATH="/path/to/a folder/"

#path where the new weight for the model will be stored
MODEL_PATH="/path/to/a folder/"

# size of the region of interest used for training (16 or 32), the value depends on the particle size 
ROI_SIZE=16 

###############################


#create folders
mkdir $IMAGE_PATH
mkdir $IMAGE_PATH"positive"
mkdir $IMAGE_PATH"negative"

# add all the MOVIE_PATH,POSITIVE_COORDINATES_PATH, NEGATIVE_COORDINATES_PATH_1
# keep the same order of the files
python tracking_lib/train_cnn.py --number_of_files $NUMBER_FILES --movie_path $MOVIE_PATH_1 $MOVIE_PATH_2 $MOVIE_PATH_3  --positive_coordinates_path $POSITIVE_COORDINATES_PATH_1 $POSITIVE_COORDINATES_PATH_2 $POSITIVE_COORDINATES_PATH_3  --negative_coordinates_path $NEGATIVE_COORDINATES_PATH_1 $NEGATIVE_COORDINATES_PATH_2 $NEGATIVE_COORDINATES_PATH_3  --save_images_path $IMAGE_PATH --roi_size $ROI_SIZE --save_model_path $MODEL_PATH

