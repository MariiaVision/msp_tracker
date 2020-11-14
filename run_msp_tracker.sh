#!/bin/bash

source activate gui_tracker

ROOT_FOLDER="/media/mariaa/tracking/" 

MOVIE_PATH=$ROOT_FOLDER"test_data/2018_06_12-50min_bio_10fps_AiryscanProcessing-f150-200.tif"

DETECTION_PARAMETERS_PATH=$ROOT_FOLDER"test_data/2018_06_12-50min_bio_10fps_AiryscanProcessing-f150-200_detection_set.txt"

LINKING_PARAMETERS_PATH=$ROOT_FOLDER"test_data/2018_06_12-50min_bio_10fps_AiryscanProcessing-f150-200_linking_set.txt"

USE_EXIST_DETECTION="True"

DETECTION_PATH=$ROOT_FOLDER"test_data/2018_06_12-50min_bio_10fps_AiryscanProcessing-f150-200_detection.txt"

RESULT_PATH=$ROOT_FOLDER"test_data/2018_06_12-50min_bio_10fps_AiryscanProcessing-f150-200_tracks_1.txt"



python tracking_lib/tracking_pipeline.py --movie_path $MOVIE_PATH --detection_parameters_path $DETECTION_PARAMETERS_PATH  --linking_parameters_path $LINKING_PARAMETERS_PATH --use_existing_detection $USE_EXIST_DETECTION --detection_path $DETECTION_PATH --result_path $RESULT_PATH
