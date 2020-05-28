# gui_tracking
This repository presents two GUIs: 
1) tracking protein vesicles  (set parameters, check the results, run on complete movie)  
2) track viewer to visualise the detected tracks, provides statistics and it also allows track modification.

## Tracker 

The approach is described in the paper "Protein Tracking By CNN-Based Candidate Pruning And Two-Step Linking With Bayesian Network" M Dmitrieva, H L Zenner,J Richens,D St Johnston and J Rittscher, MLSP 2019:  https://ieeexplore.ieee.org/abstract/document/8918873 
### Software installation
#### Mac
1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages
5. To run the software: activate the environment with `source activate gui_tracker` and run it with `python gui_tracker.py`

#### Windows
1. Install Anaconda: https://docs.anaconda.com/anaconda/install/windows/
2. Launch Anaconda Prompt
3. Navigate to the source directory and create conda environment with: `conda env create -f environment_win.yml`
4. To run the software: activate the environment with `conda activate gui_tracker` and run the tracker with `python gui_tracker.py` or double click on `run_tracker_windows.bat`

#### Linux
1. Install conda: https://docs.anaconda.com/anaconda/install/linux/
2. In the terminal open the directory with the software
3. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages
4. To run the software: activate the environment with `source activate gui_tracker` and run it with `python gui_tracker.py`

### Usage
1. In a terminal (Anaconda prompt for Windows) navigate to the software folder, activate the environment: `source activate gui_tracker` (`conda activate gui_tracker` for windows)
2. In necessary update the environment: `conda env update --file environment.yml`
3. To run the software: `python gui_tracker.py`
4. For window machine double click on `run_tracker_windows.bat` will do all the work



## Trackviewer
### Software installation
#### Mac
1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages

### Usage
1. In a terminal, activate the environment: `source activate gui_tracker`
2. In necessary update the environment: `conda env update --file environment.yml`
3. To run the software: `python gui_trackviewer.py`:
  - 1st load protein movie
  - then select the tracks (.txt file, json format)
  - for fusion event detection select membrane (segmented membrane)

