# gui_tracking
This repository presents two GUIs: 
1) tracking protein vesicles  (set parameters, check the results, run on complete movie)  
2) track viewer to visualise the detected tracks, provides statistics and it also allows track modification.

## Tracker 

### Software installation
#### Mac
1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment_tracker.yml` and press enter. It should install all the required packages

#### Windows
1. Install Anaconda: https://docs.anaconda.com/anaconda/install/windows/
2. Launch Anaconda Prompt
3. Navigate to the source directory and create conda environment with: `conda env create -f environment_tracker_win.yml`
4. Activate the environment with `conda activate gui_tracker`
5. To run software:  `python gui_tracker.py`


#### Linux
1. Install conda: https://docs.anaconda.com/anaconda/install/linux/
2. In the terminal open the directory with the software
3. In the terminal type `conda env create -f environment_tracker.yml` and press enter. It should install all the required packages

### Usage
1. In a terminal, activate the environment: `source activate gui_tracker` ( `conda activate gui_tracker` for windows)
2. In necessary update the environment: `conda env update --file environment_tracker.yml`
3. To run the software: `python gui_tracker.py`
  
### to do list:

- [ ] show number of passes
- [ ] add  license
- [ ] clean the code 
- [ ] give a proper name
- [ ] add progress bar



## Trackviewer
### Software installation
#### Mac
1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages

### Usage
1. In a terminal, activate the environment: `source activate gui_tracks`
2. In necessary update the environment: `conda env update --file environment.yml`
3. To run the software: `python gui_trackviewer_2.py`:
  - 1st load protein movie
  - then select the tracks (.txt file, json format)
  - for fusion event detection select membrane (segmented membrane)


### to do list:

- [ ] create help description
- [ ] give a proper name
- [ ] include different trajectory segmentation 
- [ ] change to different system to read the tracks
- [ ] change from list to a table with sorting option
- [ ] change plotting to grab the coordinates from the image
- [ ] clean the code
- [ ] add  license
