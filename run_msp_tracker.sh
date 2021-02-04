#!/bin/bash

conda activate msp

################## SET PATHS BELOW: 

# path to the image sequence (the file should be tiff format, single channel)
MOVIE_PATH="/path/to/the/tif/file"

# path to the file with detection parameters (created with the MSP-tracker GUI)
DETECTION_PARAMETERS_PATH="/path/to/the/parameter/file"

#path to the file with linking parameters (created with the MSP-tracker GUI)
LINKING_PARAMETERS_PATH="/path/to/the/parameter/file"

# True or False: set it to False to run the detection part of the tracker, and True to use existing detections from a file
USE_EXIST_DETECTION=False

# path to the file with the detection for the current image sequence. 
# When USE_EXIST_DETECTION is False - detections will be saved there, otherwise (when True) - detections will be read from the file
DETECTION_PATH="/path/to/the/detection/file"

# path where to save the trajectories and name of the file
RESULT_PATH="/path/to/the/file/with/results"


# running the code
python tracking_lib/tracking_pipeline.py --movie_path $MOVIE_PATH --detection_parameters_path $DETECTION_PARAMETERS_PATH  --linking_parameters_path $LINKING_PARAMETERS_PATH --use_existing_detection $USE_EXIST_DETECTION --detection_path $DETECTION_PATH --result_path $RESULT_PATH
