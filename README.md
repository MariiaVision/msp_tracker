# gui_tracking
the software visualises the detected tracks, provides statistics and allows track modification.

## Software installation:

1) install conda:  https://docs.anaconda.com/anaconda/install/mac-os/
2) create virual environment: conda env create -f environment.yml
3) in the teminal type "conda env create -f environment.yml" and press enter. It should install all the required packages

4) activate the environment: source activate gui_tracks
5) to run the software python gui_trackviewer_2.py:
  - 1st load protein movie 
  - then select the tracks (.txt file, json format) 
  - for fusion event detection select membrane (segmented membrane) 


## to do list:
- change from list to a table with sorting option
- change plotting to grab the coordinates from the image
- clean the code
- include trajectory segmentation into the speed evaluation
- count crossing
