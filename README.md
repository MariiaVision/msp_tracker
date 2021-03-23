# MSP-package
This repository presents two modules with graphical interface: 
1) MSP-tracker: particle tracker  (set parameters, check the results, run on complete movie) with extra module of membrane segmentation
2) MSP-viewer: track viewer to visualise the tracks, provide statistics, also it allows to modify/delete/add new tracks if necessary.


The basis of the tracking approach is described in the paper "Protein Tracking By CNN-Based Candidate Pruning And Two-Step Linking With Bayesian Network" M Dmitrieva, H L Zenner,J Richens,D St Johnston and J Rittscher, MLSP 2019:  https://ieeexplore.ieee.org/abstract/document/8918873 

Please, see manual for the detailed description of the software and its functionality

### Software installation

#### Mac
1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages

#### Windows
1. Install Anaconda: https://docs.anaconda.com/anaconda/install/windows/
2. Launch Anaconda Prompt
3. Navigate to the source directory and create conda environment with: `conda env create -f environment_win.yml`

#### Linux
1. Install conda: https://docs.anaconda.com/anaconda/install/linux/
2. In the terminal open the directory with the software
3. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages

### Usage 

#### MSP-tracker
1. In necessary update the environment: `conda env update --file environment.yml`
2. In a terminal (Anaconda prompt for Windows) navigate to the software folder, activate the environment: `conda activate msp'
3. To run the software: `python msptracker.py`:
  - load the image sequence
  - set detection parameters
  - set linking parameter
  - run for the entire image sequence
  - select membrane, set parameters and run for the entire image sequence 
  
2a. For window machine double click on `run_tracker_windows.bat` - it should do all the work


#### Run tracker without GUI
Use 'run_msp_tracker.sh' to run the tracker when the parameters for detection and linking are saved into a file. It is preferable to use the option when the image sequence is large and the tracking can take some time.

1.  Following variables should be adjusted:
  - MOVIE_PATH - path to the image sequence (should be tiff format, single channel)
  - DETECTION_PARAMETERS_PATH - path to the file with detection parameters (created with the MSP-tracker GUI)
  - LINKING_PARAMETERS_PATH - path to the file with linking parameters (created with the MSP-tracker GUI)
  - USE_EXIST_DETECTION - "True" or "False": set it to False to run the detection part of the tracker, and True to use existing detections
  - DETECTION_PATH - path to the file with the detection for the current image sequence. When USE_EXIST_DETECTION is"False" - detections will be saved there, otherwise (when "True") - detections will be read from the file
  - RESULT_PATH - path to save the trajectories
  2. run the script from the command line 'bash run_msp_tracker.sh'

#### Train the CNN model for candidate pruning with your data
To train the model with a new data, use bash script 'run_train_CNN.sh'
1. Following variables should be adjusted: 
  - NUMBER_FILES - number of image sequences will be used for the training 
  - MOVIE_PATH_1 ... MOVIE_PATH_N - paths to the image sequence (should be tiff format, single channel)
  - POSITIVE_COORDINATES_PATH_1 ... POSITIVE_COORDINATES_PATH_N - paths to the txt file with coordinates of the positive samples
  - NEGATIVE_COORDINATES_PATH_1 ... NEGATIVE_COORDINATES_PATH_N - paths to the txt file with coordinates of the negative samples
  - IMAGE_PATH - path where the positive and negative samples can be stored
  - MODEL_PATH - path where the new weight for the model will be stored
2. Adjust 'python tracking_lib/train_cnn.py ... ' to include all the provided files
3. run the bash script  'bash run_train_CNN.sh'
4. Copy the trained model 'cnn-model-best.hdf5' to the folder with the existing weights, rename it and select it when setting parameters in MSP-tracker

##### Preparing data for training
The training data contains an image sequences paried up with two txt files. The txt files contain coordinates of the postivie samples(vesicles) and negative samples(non-vesicles) in separate files. One raw represents a single sample with the following order (position, x, y, frame). 

To create the txt file, you can use ImageJ multi-point tool and with ctrl+M copy the coordinates into a new txt file. It is important to include large variety in the vesicle class. Non-vesicle class would include background, bright blobs, noisy areas without vesicles.



#### Trackviewer
1. If necessary update the environment: `conda env update --file environment.yml`
2. In a terminal, activate the environment: `conda activate msp`
3. To run the software: `python mspviewer.py`:
  - load protein movie
  - select the tracks (.txt file, json format or csv file)
  - optional: select membrane movie

