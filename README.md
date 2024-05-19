# MSP-package
The multiscale paricle (MSP) package provides users with tools for particle tracking, trajectories review and characterisation.
This repository contains two modules with graphical-user interface: 
1) MSP-tracker: particle tracker  with extra module of membrane segmentation
2) MSP-viewer: track viewer to visualise the tracks, correct trajectories if required and provide statisticsy.


The software package and its applications are decribed in "Tracking exocytic vesicle movements reveals the spatial control of secretion in epithelial cells"
Jennifer H. Richens, Mariia Dmitrieva, Helen L. Zenner, Nadine Muschalik, Richard Butler, Jade Glashauser, Carolina Camelo, Stefan Luschnig, Sean Munro, Jens Rittscher, Daniel St Johnston
bioRxiv 2024.01.25.577201; doi: https://doi.org/10.1101/2024.01.25.577201  and basis of the tracking approach is described in "Protein Tracking By CNN-Based Candidate Pruning And Two-Step Linking With Bayesian Network" M Dmitrieva, H L Zenner,J Richens,D St Johnston and J Rittscher, MLSP 2019:  https://ieeexplore.ieee.org/abstract/document/8918873

Please, see manual for the detailed description of the software and its functionality

!!! Coming soon: docker container will be added to run the MSP-tracker more easily. 

### Software installation

#### Mac
1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment_mac.yml` and press enter. It should install all the required packages

#### Windows
1. Install Anaconda: https://docs.anaconda.com/anaconda/install/windows/
2. Launch Anaconda Prompt
3. Navigate to the source directory and create conda environment with: `conda env create -f environment_win.yml`

#### Linux
1. Install conda: https://docs.anaconda.com/anaconda/install/linux/
2. In the terminal open the directory with the software
3. In the terminal type `conda env create -f environment_linux.yml` and press enter. It should install all the required packages

### Usage 

#### MSP-tracker
1. In necessary update the environment: `conda env update --file environment_file_name.yml`
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


#### Trackviewer
1. If necessary update the environment: `conda env update --file environment_file_name.yml`
2. In a terminal, activate the environment: `conda activate msp`
3. To run the software: `python mspviewer.py`:
  - load protein movie
  - select the tracks (.txt file (json format) or csv file)
  - optional: select membrane movie

