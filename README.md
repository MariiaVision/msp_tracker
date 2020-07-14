# MSP-tracker
This repository presents two modules with graphical interface: 
1) particle tracker  (set parameters, check the results, run on complete movie) with extra module of membrane segmentation
2) track viewer to visualise the tracks, provide statistics, also it allows to modify/delete/add new tracks if necessary.


The basis of the tracking approach is described in the paper "Protein Tracking By CNN-Based Candidate Pruning And Two-Step Linking With Bayesian Network" M Dmitrieva, H L Zenner,J Richens,D St Johnston and J Rittscher, MLSP 2019:  https://ieeexplore.ieee.org/abstract/document/8918873 


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
2. In a terminal (Anaconda prompt for Windows) navigate to the software folder, activate the environment: `source activate gui_tracker` (`conda activate gui_tracker` for windows)
3. To run the software: `python gui_tracker.py`:
  - Load the movie
  - set detection parameters
  - set linking parameter
  - run it in all the movie
  - select membrane, set parameters and run in all the movie
  
2a. For window machine double click on `run_tracker_windows.bat` -it will do all the work



#### Trackviewer
1. In necessary update the environment: `conda env update --file environment.yml`
2. In a terminal, activate the environment: `source activate gui_tracker`
3. To run the software: `python gui_trackviewer.py`:
  - 1st load protein movie
  - then select the tracks (.txt file, json format)
  - optional: select membrane movie

