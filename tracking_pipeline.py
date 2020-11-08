#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tracking pipeline to run with defined parameters
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # not to show warnings

import sys
sys.path.append('./tracking_lib/')

import numpy as np
import scipy as sp



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tqdm import tqdm
import skimage
from skimage import io
import json 

from set_tracking import  TrackingSetUp

#paths:

movie_path="/home/mariaa/NANOSCOPY/VESICLE_TRACKING/Warwick_data/200729 GFP-TPD54 Fast imaging 002.nd2--OMERO ID_257292.tif"
movie=skimage.io.imread(movie_path)

print("movie ", movie.shape)
detection_parameters_path="/home/mariaa/NANOSCOPY/VESICLE_TRACKING/Warwick_data/200729 GFP-TPD54 Fast imaging 002.nd2--OMERO ID_257292_detection_set.txt"
linking_parameters_path="/home/mariaa/NANOSCOPY/VESICLE_TRACKING/Warwick_data/200729 GFP-TPD54 Fast imaging 002.nd2--OMERO ID_257292_linking_set.txt"

result_path="/home/mariaa/NANOSCOPY/VESICLE_TRACKING/Warwick_data/200729 GFP-TPD54 Fast imaging 002.nd2--OMERO ID_257292_tracks.txt"

# # # # # # set tracking
detector=TrackingSetUp()

# read parameters and set them
detector.start_frame=0
            
detector.end_frame=10 #movie.shape[0]

detector.detection_parameters_from_file(param_path=detection_parameters_path)

detector.linking_parameters_from_file(param_path=linking_parameters_path)

detector.movie=movie

#print parameters 
print("----------------detection parameters -----------------")
print(" substract_bg_step", detector.substract_bg_step)
print(" c", detector.c)
print(" sigma_min", detector.sigma_min)
print(" sigma_max", detector.sigma_max)
print(" min_distance", detector.min_distance)
print(" threshold_rel", detector.threshold_rel)
print(" box_size", detector.box_size)
print(" detection_threshold", detector.detection_threshold)
print(" gaussian_fit", detector.gaussian_fit)
print(" expected_radius", detector.expected_radius)
print(" box_size_fit", detector.box_size_fit)
print(" cnn_model", detector.cnn_model_path)
        
        
print("\n ---------------- linkin parameters -----------------")
print(" tracker_distance_threshold", detector.tracker_distance_threshold)
print(" tracker_max_skipped_frame", detector.tracker_max_skipped_frame)
print(" tracker_max_track_length ", detector.tracker_max_track_length)

print(" \n number of pass, ", detector.tracklinking_Npass)
print(" topology", detector.tracklinking_path1_topology)
print(" tracklinking_path1_connectivity_threshold", detector.tracklinking_path1_connectivity_threshold)
print(" tracklinking_path1_frame_gap", detector.tracklinking_path1_frame_gap_1)
print(" tracklinking_path1_distance_limit", detector.tracklinking_path1_distance_limit)
print(" tracklinking_path1_direction_limit", detector.tracklinking_path1_direction_limit)
print(" tracklinking_path1_speed_limit", detector.tracklinking_path1_speed_limit)
print(" tracklinking_path1_intensity_limit", detector.tracklinking_path1_intensity_limit)

if detector.tracklinking_Npass>1:
    print("\n topology 2", detector.tracklinking_path2_topology)
    print(" tracklinking_path2_connectivity_threshold", detector.tracklinking_path2_connectivity_threshold)
    print(" tracklinking_path2_frame_gap_1", detector.tracklinking_path2_frame_gap_1)
    print(" tracklinking_path2_distance_limit", detector.tracklinking_path2_distance_limit)
    print(" tracklinking_path2_direction_limit", detector.tracklinking_path2_direction_limit)
    print(" tracklinking_path2_speed_limit", detector.tracklinking_path2_speed_limit)
    print(" tracklinking_path2_intensity_limit", detector.tracklinking_path2_intensity_limit)

if detector.tracklinking_Npass>2:
    print("\n topology 3", detector.tracklinking_path1_topology)
    print(" tracklinking_path3_connectivity_threshold", detector.tracklinking_path3_connectivity_threshold)
    print(" tracklinking_path3_frame_gap_1", detector.tracklinking_path3_frame_gap_1)
    print(" tracklinking_path3_distance_limit", detector.tracklinking_path3_distance_limit)
    print(" tracklinking_path3_direction_limit", detector.tracklinking_path3_direction_limit)
    print(" tracklinking_path3_speed_limit", detector.tracklinking_path3_speed_limit)
    print(" tracklinking_path3_intensity_limit", detector.tracklinking_path3_intensity_limit)
#run tracking
final_tracks=detector.linking()


# save tracks        
with open(result_path, 'w') as f:
    json.dump(final_tracks, f, ensure_ascii=False)
        