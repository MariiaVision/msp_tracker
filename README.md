# gui_tracking
The software visualises the detected tracks, provides statistics and allows track modification.

## Software installation

1. Install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2. Open the folder which contains the software in finder.
3. In finder, right click on code folder and select `New Terminal at Folder`
4. In the terminal type `conda env create -f environment.yml` and press enter. It should install all the required packages

## Usage
1. In a terminal, activate the environment: `source activate gui_tracks`
1a. In necessary update the environment: `conda env update --file environment.yml`
2. To run the software: `python gui_trackviewer_2.py`:
  - 1st load protein movie
  - then select the tracks (.txt file, json format)
  - for fusion event detection select membrane (segmented membrane)


## to do list:
- [ ] include motion map with A-P orientation
- [ ] change from list to a table with sorting option
- [ ] change plotting to grab the coordinates from the image
- [ ] clean the code
