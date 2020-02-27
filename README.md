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

#### Linux
1. Install conda: https://docs.anaconda.com/anaconda/install/linux/
2. In the terminal open the directory with the software
3. In the terminal type `conda env create -f environment_tracker.yml` and press enter. It should install all the required packages

### Usage
1. In a terminal, activate the environment: `source activate gui_tracker`
2. In necessary update the environment: `conda env update --file environment_tracker.yml`
3. To run the software: `python gui_tracker.py`
  
### to do list:
- [ ] add progress bar
- [ ] set colours
- [ ] fix zooming from frame to frame
- [ ] clean the code 
- [ ] create a proper description in "help"
- [ ] give a proper name



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
- [ ] include differenttrajectory segmentation
- [ ] make scaling transit from frame to frame -> self.ax.set_ylim(val) doesn't allow "home" button
- [ ] change to different system to read the tracks
- [ ] change from list to a table with sorting option
- [ ] change plotting to grab the coordinates from the image
- [ ] clean the code
- [ ] give a proper name
