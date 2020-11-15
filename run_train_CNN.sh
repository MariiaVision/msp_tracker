#!/bin/bash

source activate gui_tracker

NUMBER_FILES=1
MOVIE_PATH_1="/home/mariaa/NANOSCOPY/data/data_for_processing/STED/Area05/Image0005_Channel_1.tif"
MOVIE_PATH_2="/home/mariaa/NANOSCOPY/data/data_for_processing/STED/Area05/Image0005_Channel_1.tif"

POSITIVE_COORDINATES_PATH_1="/home/mariaa/NANOSCOPY/data/data_for_processing/STED/Area05/Image0005_Channel_1_pos.txt"
POSITIVE_COORDINATES_PATH_2="/home/mariaa/NANOSCOPY/data/data_for_processing/STED/Area05/Image0005_Channel_1_pos.txt"

NEGATIVE_COORDINATES_PATH_1="/home/mariaa/NANOSCOPY/data/data_for_processing/STED/Area05/Image0005_Channel_1_neg.txt"
NEGATIVE_COORDINATES_PATH_2="/home/mariaa/NANOSCOPY/data/data_for_processing/STED/Area05/Image0005_Channel_1_neg.txt"

IMAGE_PATH="/home/mariaa/NANOSCOPY/VESICLE_TRACKING/code/TEST/"

MODEL_PATH="/home/mariaa/NANOSCOPY/VESICLE_TRACKING/code/TEST/"

ROI_SIZE=32
#create folders

mkdir $IMAGE_PATH"positive"
mkdir $IMAGE_PATH"negative"

python tracking_lib/train_cnn.py --number_of_files $NUMBER_FILES --movie_path $MOVIE_PATH_1  --positive_coordinates_path $POSITIVE_COORDINATES_PATH_1   --negative_coordinates_path $NEGATIVE_COORDINATES_PATH_1  --save_images_path $IMAGE_PATH --roi_size $ROI_SIZE --save_model_path $MODEL_PATH
