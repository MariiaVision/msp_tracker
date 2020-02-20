#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:41:16 2019

@author: maria dmitrieva
"""

import sys
sys.path.append('./tracking_lib/')

import numpy as np
import scipy as sp

import copy
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import csv

# for plotting
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import skimage
from skimage import io
from scipy.ndimage import gaussian_filter1d
import json        
import cv2
import imageio
import math
from skimage.feature import peak_local_max

from fusion_events import FusionEvent 

from trajectory_segmentation_msd import TrajectorySegment
from sys import platform as _platform

from set_tracking import  TrackingSetUp


    
class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("MSBNtracker")
        parent.configure(background='white')
        
        #set the window size        
        self.window_width = int(parent.winfo_screenwidth()/2) # half the monitor width
        self.window_height = int(parent.winfo_screenheight()*0.8)  # 0.9 of the monitor height
#        parent.geometry(str(self.window_width)+"x"+str(self.window_height)) #"1200x1000")

        # menu
        menu = tk.Menu(parent)
        parent.config(menu=menu)
        parent.configure(background='white')
        helpmenu=tk.Menu(menu)        
        menu.add_cascade(label="Help", menu=helpmenu)
        
        def HelpAbout():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("About ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
            # widget in there with a text 
            lb_start = tk.Label(master=self.new_window_help, text=" Information about the software ",  bg='white')
            lb_start.grid(row=0, column=0, pady=self.pad_val*2, padx=self.pad_val*3)            
            
        def HelpDetection():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("Detection ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
            # widget in there with a text 
            lb_start = tk.Label(master=self.new_window_help, text=" Information about the Detection ",  bg='white')
            lb_start.grid(row=0, column=0, pady=self.pad_val*2, padx=self.pad_val*3)       
            
        def HelpLinking():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("Linking ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
            # widget in there with a text 
            lb_start = tk.Label(master=self.new_window_help, text=" Information about the Linking ",  bg='white')
            lb_start.grid(row=0, column=0, pady=self.pad_val*2, padx=self.pad_val*3)         
            
            
        def HelpTracking():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("Tracking ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
            # widget in there with a text 
            lb_start = tk.Label(master=self.new_window_help, text=" Information about the Tracking ",  bg='white')
            lb_start.grid(row=0, column=0, pady=self.pad_val*2, padx=self.pad_val*3)       
            
            
        helpmenu.add_command(label="About...", command=HelpAbout)
        helpmenu.add_command(label="Detection", command=HelpDetection)
        helpmenu.add_command(label="Linking", command=HelpLinking)
        helpmenu.add_command(label="Run tracking", command=HelpTracking)
        
        # set movie and class for parameter settings
        self.movie=np.ones((1,200,200))
        self.detector=TrackingSetUp()
        
        

        # main paths          
        self.movie_protein_path="not defined"
        self.result_path="not defined"
        
        # TABs 
        # set the style 
        style = ttk.Style()
        style.configure("TNotebook", foreground="black", background="white")
        
        tab_parent = ttk.Notebook(parent) # create tabs
        tab_detection = ttk.Frame(tab_parent, style="TNotebook")
        
        tab_parent.add(tab_detection, text=" Detection ")
        
        
        tab_linking = ttk.Frame(tab_parent)
        tab_parent.add(tab_linking, text=" Linking ")

        
        tab_run = ttk.Frame(tab_parent)
        tab_parent.add(tab_run, text=" Run tracking ")        
        
        tab_parent.pack(expand=1, fill='both')
        
        
        # detection
        detectionFrame = tk.Frame(master=tab_detection)
        detectionFrame.pack(expand=1, fill='both')
        
        # linking 
        linkingFrame = tk.Frame(master=tab_linking)
        linkingFrame.pack(expand=1, fill='both')
        
        # run 
        runFrame = tk.Frame(master=tab_run)
        runFrame.pack(expand=1, fill='both')
        


    ########################## DETECTION ######################
     # # # # # # # # # # # # # # # # # # # # # #   
        detectionFrame.configure(background='white')
        
        ############################################

        self.frame_pos=0
        self.movie_length=1
        self.monitor_switch_detection=0
        self.pad_val=5
        self.dpi=100
        self.img_width=self.window_height*0.8
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        self.button_length=np.max((10,int(self.window_width/100)))
        self.filename_final_tracking="unnamed_tracking_results.txt"
        
        #############################################

        # Framework: place monitor and view point
        self.viewFrame_detection = tk.Frame(master=detectionFrame, width=int(self.window_width*0.6), height=self.window_height, bg="white")
        self.viewFrame_detection.grid(row=0, column=0, pady=self.pad_val, padx=self.pad_val)   

           
        # place parameters and buttons
        self.parametersFrame_detection = tk.Frame(master=detectionFrame, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.parametersFrame_detection.grid(row=0, column=11, columnspan=1, rowspan=10, pady=self.pad_val, padx=self.pad_val)    




     # # # # # # # # # # # # # # # # # # # # # #    
        # Framework: place monitor 
        self.button_mv = tk.Button(self.viewFrame_detection,text="   Select vesicle movie   ", command=self.select_vesicle_movie_detection, width=20)
        self.button_mv.grid(row=0, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)        

        var_plot_detection = tk.IntVar()
        
        def update_detection_switch():            
            self.monitor_switch_detection=var_plot_detection.get()
            # change image
            self.show_frame_detection()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(self.viewFrame_detection, text=" original image ", variable=var_plot_detection, value=0, bg='white', command =update_detection_switch )
        self.M1.grid(row=3, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(self.viewFrame_detection, text=" candidates ", variable=var_plot_detection, value=1, bg='white',command = update_detection_switch ) #  command=sel)
        self.M2.grid(row=3, column=3, columnspan=3,  pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(self.viewFrame_detection, text=" detection ", variable=var_plot_detection, value=2, bg='white',command = update_detection_switch ) #  command=sel)
        self.M3.grid(row=3, column=6, columnspan=3, pady=self.pad_val, padx=self.pad_val)
  
        

        # plot bg
        self.figd, self.axd = plt.subplots(1,1, figsize=self.figsize_value, dpi=self.dpi)
        self.axd.axis('off')
        self.figd.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.show_frame_detection() 

   #   next and previous buttons
        def show_values_detection(v):
            self.frame_pos=int(v)
            self.show_tracks() 
          
        self.scale_movie = tk.Scale(self.viewFrame_detection, from_=0, to=self.movie_length-1, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values_detection)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

        self.set_detection_parameters_frame()
        
  # # # # # # # # # # # # # # # # # # # # # # # # #
        

############################ Linking #################

        #set the window colour        
        linkingFrame.configure(background='white')
        
        ############################################

        self.monitor_switch_linking=0
        self.pad_val=5
        self.dpi=100
        self.img_width=self.window_height*0.8
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        self.button_length=np.max((10,int(self.window_width/100)))
        self.track_data_framed={} 
    
        
        self.color_list_plot=["#00FFFF", "#7FFFD4", "#0000FF", "#8A2BE2", "#7FFF00", "#D2691E", "#FF7F50", "#DC143C",
            "#008B8B", "#8B008B", "#FF8C00", "#E9967A", "#FF1493", "#9400D3", "#FF00FF", "#B22222",
            "#FFD700", "#ADFF2F", "#FF69B4", "#ADD8E6", "#F08080", "#90EE90", "#20B2AA", "#C71585", "#FF00FF"]        
        #############################################
    
    
        # Framework: place monitor and view point
        self.viewFrame_linking = tk.Frame(master=linkingFrame, width=int(self.window_width*0.6), height=self.window_height, bg="white")
        self.viewFrame_linking.grid(row=0, column=0, pady=self.pad_val, padx=self.pad_val)   
    
           
        # place parameters and buttons
        self.parametersFrame_linking = tk.Frame(master=linkingFrame, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.parametersFrame_linking.grid(row=0, column=11, columnspan=1, rowspan=10, pady=self.pad_val, padx=self.pad_val)    
    
  
        # Framework: place monitor 
        self.button_mv = tk.Button(self.viewFrame_linking,text="   Select vesicle movie   ", command=self.select_vesicle_movie_linking, width=20)
        self.button_mv.grid(row=0, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)        
    
        var_plot_linking = tk.IntVar()
        
        def update_linking_switch():            
            self.monitor_switch_linking=var_plot_linking.get()
            # change image
            self.show_frame_linking()
    
        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(self.viewFrame_linking, text=" original image ", variable=var_plot_linking, value=0, bg='white', command =update_linking_switch )
        self.M1.grid(row=3, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(self.viewFrame_linking, text=" tracklets ", variable=var_plot_linking, value=1, bg='white',command = update_linking_switch ) #  command=sel)
        self.M2.grid(row=3, column=3, columnspan=3,  pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(self.viewFrame_linking, text=" tracks ", variable=var_plot_linking, value=2, bg='white',command = update_linking_switch ) #  command=sel)
        self.M3.grid(row=3, column=6, columnspan=3, pady=self.pad_val, padx=self.pad_val)
      
        
    
        # plot bg
        self.figl, self.axl = plt.subplots(1,1, figsize=self.figsize_value, dpi=self.dpi)
        self.axl.axis('off')
        self.figl.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.show_frame_linking() 
    
       #   next and previous buttons
        def show_values_linking(v):
            self.frame_pos=int(v)
            self.show_frame_linking() 
          
        self.scale_movie = tk.Scale(self.viewFrame_linking, from_=0, to=self.movie_length-1, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values_linking)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
        self.set_linking_parameters_frame()
        


#################################  run tracking: runFrame ##############################
        
        #set the window colour        
        runFrame.configure(background='white')

        # Framework: place monitor and view point
        self.action_frame = tk.Frame(master=runFrame, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.action_frame.grid(row=0, column=0, pady=self.pad_val, padx=self.pad_val)   

        # place parameters and buttons
        self.gap_frame = tk.Frame(master=runFrame, width=int(self.window_width*0.1), height=self.window_height, bg="white")
        self.gap_frame.grid(row=0, column=1, columnspan=1, rowspan=10, pady=self.pad_val, padx=self.pad_val)      

        lbl3 = tk.Label(master=self.gap_frame, text=" ",  bg='white', width=int(self.button_length), height=int(self.button_length/1.5))
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val)   
        
        
        # place parameters and buttons
        self.information_frame = tk.Frame(master=runFrame, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.information_frame.grid(row=0, column=2, columnspan=1, rowspan=10, pady=self.pad_val, padx=self.pad_val)    
        
        # set start and end frame
        lb_start = tk.Label(master=self.action_frame, text=" start frame ",  bg='white')
        lb_start.grid(row=0, column=0) 
        v=tk.StringVar(self.action_frame, value=str(self.detector.start_frame))
        self.r_start_frame = tk.Entry(self.action_frame, width=self.button_length, text=v)
        self.r_start_frame.grid(row=0, column=1, pady=self.pad_val, padx=self.pad_val)        
 
        lb_start = tk.Label(master=self.action_frame, text=" end frame ",  bg='white')
        lb_start.grid(row=0, column=2) 
        v=tk.StringVar(self.action_frame, value=str(self.movie.shape[0]))
        self.r_end_frame = tk.Entry(self.action_frame, width=self.button_length, text=v)
        self.r_end_frame.grid(row=0, column=3, pady=self.pad_val, padx=self.pad_val)    

        def define_result_path():
            
            filename=tk.filedialog.asksaveasfilename(title = "Save tracking results ")
            # save in parameters
            if not filename:
                print("file was not given")
            else:
                self.result_path=filename
                self.show_parameters()
                
        def update_info():
            self.show_parameters()

        # button to set 
        lbl3 = tk.Button(master=self.action_frame, text=" Where to save results ", command=define_result_path, width=int(self.button_length*1.5), bg="gray")
        lbl3.grid(row=4, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val)

        # button to set 
        lbl3 = tk.Button(master=self.action_frame, text=" Update the info ", command=update_info, width=int(self.button_length*1.5), bg="gray")
        lbl3.grid(row=5, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val)
          # empty space
        lbl3 = tk.Label(master=self.action_frame, text=" ",  bg='white', height=int(self.button_length/4))
        lbl3.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)  
        
        # button to run tracking        
        lbl3 = tk.Button(master=self.action_frame, text=" RUN TRACKING  ", command=self.run_tracking, width=self.button_length*2, bg="#02a17a")
        lbl3.grid(row=7, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)

        # show parameters
        self.show_parameters()
        
        ####################################################
        
    # parameters 
    def show_parameters(self):
        #### show parameters : parametersFrame_linking
        
        lbl3 = tk.Label(master=self.information_frame, text=" - - - - - IMPORTANT PATHS: - - - - - ",  bg='white')
        lbl3.grid(row=1, column=0, columnspan=4, pady=self.pad_val*3, padx=self.pad_val*3) 
        
        lbl3 = tk.Label(master=self.information_frame, text=" Original protein channel:  "+ self.movie_protein_path,  bg='white')
        lbl3.grid(row=2, column=0, pady=self.pad_val, padx=self.pad_val) 
        
        lbl3 = tk.Label(master=self.information_frame, text=" Final result fill be saved to: "+ self.result_path,  bg='white')
        lbl3.grid(row=3, column=0, pady=self.pad_val, padx=self.pad_val) 
        
        lbl3 = tk.Label(master=self.information_frame, text=" - - - - - PARAMETERS - - - - - ",  bg='white')
        lbl3.grid(row=4, column=0, columnspan=4, pady=self.pad_val*3, padx=self.pad_val*3) 
        
        lbl3 = tk.Label(master=self.information_frame, text=" CANDIDATES DETECTION ",  bg='white')
        lbl3.grid(row=5, column=0, columnspan=4, pady=self.pad_val*2, padx=self.pad_val*2) 
        
    # substract_bg_step background substraction step 

        lbl3 = tk.Label(master=self.information_frame, text=" Background subtraction based on  "+ str(self.detector.substract_bg_step)+" frames",  bg='white')
        lbl3.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val) 
        
    # threshold coef

        lbl3 = tk.Label(master=self.information_frame, text=" Threshold coefficient  "+ str(self.detector.c),  bg='white')
        lbl3.grid(row=7, column=0, pady=self.pad_val, padx=self.pad_val) 

    # sigma
        lbl3 = tk.Label(master=self.information_frame, text=" Sigma from  "+ str(self.detector.sigma_min)+" to "+str(self.detector.sigma_max),  bg='white')
        lbl3.grid(row=8, column=0, pady=self.pad_val, padx=self.pad_val)
        
    # min_distance min distance minimum distance between two max after MSSEF

        lbl3 = tk.Label(master=self.information_frame, text=" Minimum distance between detections "+str(self.detector.min_distance)+" pix",  bg='white')
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val) 
        
    # self.threshold_rel min pix value in relation to the image
    
        lbl3 = tk.Label(master=self.information_frame, text=" Relevant peak height "+str(self.detector.threshold_rel),  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 

            
          # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', height=int(self.button_length/4))
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val)        
                
        
        lbl3 = tk.Label(master=self.information_frame, text=" CANDIDATES PRUNING ",  bg='white')
        lbl3.grid(row=12, column=0, columnspan=4, pady=self.pad_val*2, padx=self.pad_val*2) 
        
    #self.box_size=16 # bounding box size for detection
        lbl3 = tk.Label(master=self.information_frame, text=" Region of Interest size "+str(self.detector.box_size)+" pix",  bg='white')
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val) 


    # detection_threshold threshold for the CNN based classification
        lbl3 = tk.Label(master=self.information_frame, text=" Threshold coefficient "+str(self.detector.detection_threshold),  bg='white')
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val)
    
    # gaussian_fit gaussian fit
        lbl3 = tk.Label(master=self.information_frame, text=" Gaussian fit:  "+str(self.detector.gaussian_fit),  bg='white')
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val) 
    
    # cnn_model cnn model 
        lbl3 = tk.Label(master=self.information_frame, text=" Loaded CNN model: "+self.detector.cnn_model_path.split("/")[-1],  bg='white')
        lbl3.grid(row=16, column=0, pady=self.pad_val, padx=self.pad_val)
    
  # # # # # #  # #

         # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', height=int(self.button_length/4))
        lbl3.grid(row=17, column=0, pady=self.pad_val, padx=self.pad_val) 

        
        lbl3 = tk.Label(master=self.information_frame, text=" TRACKER: STEP 1 ",  bg='white')
        lbl3.grid(row=20, column=0, columnspan=4, pady=self.pad_val*2, padx=self.pad_val*2) 
        
    # Maximum distance to link 
    
        lbl3 = tk.Label(master=self.information_frame, text=" Maximum distance to link "+str(self.detector.tracker_distance_threshold)+" pix",  bg='white')
        lbl3.grid(row=21, column=0, pady=self.pad_val, padx=self.pad_val) 
        
    # Maximum skipped frames
    
        lbl3 = tk.Label(master=self.information_frame, text=" Maximum skipped frames  "+str(self.detector.tracker_max_skipped_frame),  bg='white')
        lbl3.grid(row=22, column=0, pady=self.pad_val, padx=self.pad_val) 
    
    # Maximum track length
        lbl3 = tk.Label(master=self.information_frame, text=" Maximum track length  "+str(self.detector.tracker_max_track_length)+" frames",  bg='white')
        lbl3.grid(row=23, column=0, pady=self.pad_val, padx=self.pad_val)
        
        
          # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', height=int(self.button_length/4))
        lbl3.grid(row=24, column=0, pady=self.pad_val, padx=self.pad_val)        
        
        lbl3 = tk.Label(master=self.information_frame, text=" TRACKER: STEP 2 - tracklinking ",  bg='white')
        lbl3.grid(row=25, column=0, columnspan=4, pady=self.pad_val*2, padx=self.pad_val*2) 
        
    # Topology
    
        lbl3 = tk.Label(master=self.information_frame, text=" Bayesian network topology: "+self.detector.tracklinking_path1_topology,  bg='white')
        lbl3.grid(row=26, column=0, pady=self.pad_val, padx=self.pad_val)
    
    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.information_frame, text=" Connectivity threshold "+str(self.detector.tracklinking_path1_connectivity_threshold),  bg='white')
        lbl3.grid(row=27, column=0, pady=self.pad_val, padx=self.pad_val)        
        
    # tracklinking_path1_frame_gap_0 
        lbl3 = tk.Label(master=self.information_frame, text=" Small temporal gap "+str(self.detector.tracklinking_path1_frame_gap_0) +" frames  ",  bg='white')
        lbl3.grid(row=28, column=0, pady=self.pad_val, padx=self.pad_val)
        
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.information_frame, text=" Large temporal gap "+str(self.detector.tracklinking_path1_frame_gap_1)+" frames ", bg='white')
        lbl3.grid(row=29, column=0, pady=self.pad_val, padx=self.pad_val)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Distance limit "+str(self.detector.tracklinking_path1_distance_limit)+" pix ",  bg='white')
        lbl3.grid(row=30, column=0, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Orientation similarity limit "+str(self.detector.tracklinking_path1_direction_limit)+" degrees ",  bg='white')
        lbl3.grid(row=31, column=0, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Speed similarity limit "+str(self.detector.tracklinking_path1_speed_limit),  bg='white')
        lbl3.grid(row=32, column=0, pady=self.pad_val, padx=self.pad_val) 
    
    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Intensity similarity limit "+str(self.detector.tracklinking_path1_intensity_limit),  bg='white')
        lbl3.grid(row=33, column=0, pady=self.pad_val, padx=self.pad_val) 
    
        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Threshold of track length "+str(self.detector.tracklinking_path1_track_duration_limit)+" frames ",  bg='white')
        lbl3.grid(row=34, column=0, pady=self.pad_val, padx=self.pad_val) 
     
    
         # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', height=int(self.button_length/4))
        lbl3.grid(row=35, column=0, pady=self.pad_val, padx=self.pad_val) 
        
     
      # # # # # # # # # # # # # # # # # # # # # # # # # 
  
    def set_linking_parameters_frame(self):

        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" TRACKER: STEP 1 ",  bg='white')
        lbl3.grid(row=0, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Maximum distance to link 
    
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Maximum distance to link, pix ",  bg='white')
        lbl3.grid(row=1, column=0) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracker_distance_threshold))
        self.l_tracker_distance_threshold = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracker_distance_threshold.grid(row=1, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # Maximum skipped frames
    
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Maximum skipped frames  ",  bg='white')
        lbl3.grid(row=2, column=0) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracker_max_skipped_frame))
        self.l_tracker_max_skipped_frame = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracker_max_skipped_frame.grid(row=2, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # Maximum track length
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Maximum track length  ",  bg='white')
        lbl3.grid(row=3, column=0)
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracker_max_track_length))
        self.l_tracker_max_track_length = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracker_max_track_length.grid(row=3, column=1, pady=self.pad_val, padx=self.pad_val)
        
        
        
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=4, column=0, pady=self.pad_val, padx=self.pad_val)        
        
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" TRACKER: step 2 - tracklinking ",  bg='white')
        lbl3.grid(row=5, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Topology
    
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Bayesian network topology ",  bg='white')
        lbl3.grid(row=7, column=0)     
        self.comboTopology = ttk.Combobox(master=self.parametersFrame_linking, 
                            values=[
                                    "complete", 
                                    "no_intensity",
                                    "no_orientation",
                                    "no_motion",
                                    "no_gap"]) # comboTopology.get()
        self.comboTopology.grid(row=7, column=1) 
        self.comboTopology.current(0)
    
    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Connectivity threshold [0,1]  ",  bg='white')
        lbl3.grid(row=8, column=0)
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_connectivity_threshold))
        self.l_tracklinking_path1_connectivity_threshold = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_connectivity_threshold.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val) 
        
        
    # tracklinking_path1_frame_gap_0 
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Small temporal gap, frames  ",  bg='white')
        lbl3.grid(row=9, column=0)
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_frame_gap_0))
        self.l_tracklinking_path1_frame_gap_0 = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_frame_gap_0.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Large temporal gap, frames ", bg='white')
        lbl3.grid(row=9, column=2, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_frame_gap_1))
        self.l_tracklinking_path1_frame_gap_1 = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_frame_gap_1.grid(row=9, column=3, pady=self.pad_val, padx=self.pad_val)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Distance limit, pix ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_distance_limit))
        self.l_tracklinking_path1_distance_limit = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_distance_limit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Orientation similarity limit, degrees ",  bg='white')
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_direction_limit))
        self.l_tracklinking_path1_direction_limit = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_direction_limit.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Speed siimilarity limit ",  bg='white')
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_speed_limit))
        self.l_tracklinking_path1_speed_limit = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_speed_limit.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Intensity similarity limit, partition ",  bg='white')
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_intensity_limit))
        self.l_tracklinking_path1_intensity_limit = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_intensity_limit.grid(row=13, column=1, pady=self.pad_val, padx=self.pad_val)
    
    
        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Threshold of track length, frames ",  bg='white')
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.tracklinking_path1_track_duration_limit))
        self.l_tracklinking_path1_track_duration_limit = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.l_tracklinking_path1_track_duration_limit.grid(row=14, column=1, pady=self.pad_val, padx=self.pad_val)
     
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" ",  bg='white', height=int(self.button_length/4))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val) 
    
    # test range 
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" Testing from frame  ",  bg='white')
        lbl3.grid(row=16, column=0)
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.start_frame))
        self.start_frame = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.start_frame.grid(row=16, column=1, pady=self.pad_val, padx=self.pad_val)
    
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" to frame  ", bg='white')
        lbl3.grid(row=16, column=2, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_linking, value=str(self.detector.end_frame))
        self.end_frame = tk.Entry(self.parametersFrame_linking, width=self.button_length, text=v)
        self.end_frame.grid(row=16, column=3, pady=self.pad_val, padx=self.pad_val)
        
      # # # # # #  # #
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=18, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame_linking, text=" Run test ", command=self.run_test_linking, width=self.button_length*2, bg="#02f17a")
        lbl3.grid(row=19, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking, text=" Save to file ", command=self.save_to_file_linking, width=self.button_length*2, bg="#00917a")
        lbl3.grid(row=20, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking, text=" Read from file ", command=self.read_from_file_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=21, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   
            
      # # # # # # # # # # # # # # # # # # # # # # # # #       

    def set_detection_parameters_frame(self):

        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" CANDIDATES DETECTION ",  bg='white')
        lbl3.grid(row=0, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # substract_bg_step background substraction step 

        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Background evaluation: N frames ",  bg='white')
        lbl3.grid(row=1, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.substract_bg_step))
        self.d_substract_bg_step = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_substract_bg_step.grid(row=1, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # threshold coef

        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Threshold coefficient  ",  bg='white')
        lbl3.grid(row=2, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.c))
        self.d_c = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_c.grid(row=2, column=1, pady=self.pad_val, padx=self.pad_val)

    # sigma
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Sigma : from  ",  bg='white')
        lbl3.grid(row=3, column=0)
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.sigma_min))
        self.d_sigma_min = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_sigma_min.grid(row=3, column=1, pady=self.pad_val, padx=self.pad_val)
         
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" to ", bg='white')
        lbl3.grid(row=3, column=2, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.sigma_max))
        self.d_sigma_max = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_sigma_max.grid(row=3, column=3, pady=self.pad_val, padx=self.pad_val)
        
    # min_distance min distance minimum distance between two max after MSSEF

        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Minimum distance between detections  ",  bg='white')
        lbl3.grid(row=4, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.min_distance))
        self.d_min_distance = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_min_distance.grid(row=4, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # self.threshold_rel min pix value in relation to the image
    
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Relevant peak height ",  bg='white')
        lbl3.grid(row=5, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.threshold_rel))
        self.d_threshold_rel = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_threshold_rel.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val)

            
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)        
                
        
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" CANDIDATES PRUNING ",  bg='white')
        lbl3.grid(row=7, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    #self.box_size=16 # bounding box size for detection
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Region of Interest size ",  bg='white')
        lbl3.grid(row=8, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.box_size))
        self.d_box_size = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_box_size.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val)


    # detection_threshold threshold for the CNN based classification
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Threshold coefficient ",  bg='white')
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.detection_threshold))
        self.d_detection_threshold = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_detection_threshold.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # gaussian_fit gaussian fit
        def clickgaussian_fit():
            self.detector.gaussian_fit=self.gaussianValue.get()
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Gaussian fit (True/False) ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        self.gaussianValue=tk.BooleanVar()
        self.gaussianValue.set(True)
        self.d_gaussian_fit = tk.Checkbutton(self.parametersFrame_detection, text='', var=self.gaussianValue, command=clickgaussian_fit)
        self.d_gaussian_fit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)

    
    # cnn_model cnn model 
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Load CNN model ", command=self.load_cnn_model, width=self.button_length)
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val)  
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=self.detector.cnn_model_path.split("/")[-1],  bg='white')
        lbl3.grid(row=11, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
    
  # # # # # #  # #

         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Run test ", command=self.run_test_detection, width=self.button_length*2, bg="#02f17a")
        lbl3.grid(row=15, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Save to file ", command=self.save_to_file_detection, width=self.button_length*2, bg="#00917a")
        lbl3.grid(row=16, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Read from file ", command=self.read_from_file_detection, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=17, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   
    

        
    def load_cnn_model(self):
        # choose the file
        filename = tk.filedialog.askopenfilename(title = "Select CNN model")
        
        # save in parameters
        if not filename:
            print("file was not chosen")
        else:
            self.detector.cnn_model_path=filename
            self.set_detection_parameters_frame()
        
    def collect_detection_parameters(self):
        
        # parameters: candidate detection
        if self.d_substract_bg_step.get()!='':
            self.detector.substract_bg_step=int(self.d_substract_bg_step.get())
            
        if self.d_c.get()!='':
            self.detector.c=float(self.d_c.get())   
            
        if self.d_sigma_min.get()!='':
            self.detector.sigma_min=float(self.d_sigma_min.get())
            
        if self.d_sigma_max.get()!='':
            self.detector.sigma_max=float(self.d_sigma_max.get())
            
        if self.d_min_distance.get()!='':
            self.detector.min_distance=float(self.d_min_distance.get())
            
        if self.d_threshold_rel.get()!='':
            self.detector.threshold_rel=float(self.d_threshold_rel.get())
            
        # parameters: candidate pruning    
        if self.d_box_size.get()!='':
            self.detector.box_size=int(self.d_box_size.get())
            
        if self.d_detection_threshold.get()!='':
            self.detector.detection_threshold=float(self.d_detection_threshold.get())
            
        self.show_parameters()
        
    def run_test_detection(self):
        
        # read parameters from the buttons

        self.collect_detection_parameters()

        # movie 
        self.detector.movie=self.movie
        print("----------------parameters -----------------")
        print(" substract_bg_step", self.detector.substract_bg_step)
        print(" c", self.detector.c)
        print(" sigma_min", self.detector.sigma_min)
        print(" sigma_max", self.detector.sigma_max)
        print(" min_distance", self.detector.min_distance)
        print(" threshold_rel", self.detector.threshold_rel)
        print(" box_size", self.detector.box_size)
        print(" detection_threshold", self.detector.detection_threshold)
        print(" gaussian_fit", self.detector.gaussian_fit)
        print(" cnn_model", self.detector.cnn_model_path)
        
        print("\n running detection for frame ", self.frame_pos, " ...")

        self.detector.detection(int(self.frame_pos))
        self.show_frame_detection()        
        

    def save_to_file_detection(self):
        
        # update parameters
        self.collect_detection_parameters()
        
        # choose the file
        filename=tk.filedialog.asksaveasfilename(title = "Save parameters into json file")
        
        
        # save into the file
        if not filename:
            print("file was not chosen. Nothing will be saved")
        else:
            self.detector.detection_parameter_path= filename
            self.set_detection_parameters_frame()
            self.detector.detection_parameter_to_file()
        
            print(" Parameters are in the file ", self.detector.detection_parameter_path)

    def read_from_file_detection(self):
        
        # choose file
        filename = tk.filedialog.askopenfilename(title = "Open file with parameters ")
        # read from the file
        if not filename:
            print("file was not chosen")
        else:
            self.detector.detection_parameters_from_file(filename)     
        
            # update frame
            self.set_detection_parameters_frame()
        
            print(" Parameters are read from the file ", filename)

        
    def show_frame_detection(self):    
        
        # plot image
        self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        
        self.axd.clear() # clean the plot 
        self.axd.imshow(self.image, cmap="gray")
        self.axd.axis('off')  
        
        # plot results
        
   
        if self.monitor_switch_detection==1: # candidates
            if len(self.detector.detection_candidates)>0:
                for i in range(0, len(np.asarray(self.detector.detection_candidates))):
                    circle=plt.Circle((np.asarray(self.detector.detection_candidates)[i,1], np.asarray(self.detector.detection_candidates)[i,0]), 3, color="b", fill=False)
                    self.axd.add_artist(circle)    
        elif self.monitor_switch_detection==2: # detection
            if len(self.detector.detection_vesicles)>0:
                for i in range(0, len(np.asarray(self.detector.detection_vesicles))):
                    circle=plt.Circle((np.asarray(self.detector.detection_vesicles)[i,1], np.asarray(self.detector.detection_vesicles)[i,0]), 3, color="r", fill=False)
                    self.axd.add_artist(circle)    


        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.figd, master=self.viewFrame_detection)
        self.canvas.get_tk_widget().grid(row=5, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        toolbarFrame = tk.Frame(master=self.viewFrame_detection)
        toolbarFrame.grid(row=10, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()
      
#            
    def select_vesicle_movie_linking(self):
        
        filename = tk.filedialog.askopenfilename()
        if not filename:
            print("File was not chosen")
        else:   
            self.movie_protein_path=filename
        
            # read files 
            self.movie=skimage.io.imread(self.movie_protein_path)
            self.movie_length=self.movie.shape[0]  
            lbl1 = tk.Label(master=self.viewFrame_detection, text="movie: "+self.movie_protein_path.split("/")[-1], bg='white')
            lbl1.grid(row=1, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)
              
            lbl1 = tk.Label(master=self.viewFrame_linking, text="movie: "+self.movie_protein_path.split("/")[-1], bg='white')
            lbl1.grid(row=1, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)
            
        
        # plot image
        self.show_frame_linking()
        self.show_frame_detection()
        
   #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_frame_linking() 
            self.show_frame_detection()
          
        self.scale_movie = tk.Scale(self.viewFrame_linking,  from_=0, to=self.movie_length-1, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        self.scale_movie = tk.Scale(self.viewFrame_detection,  from_=0, to=self.movie_length-1, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)  
 
            
#            
    def select_vesicle_movie_detection(self):
        
        filename = tk.filedialog.askopenfilename()
        if not filename:
            print("File was not chosen")
        else:   
            self.movie_protein_path=filename
        
            # read files 
            self.movie=skimage.io.imread(self.movie_protein_path)
            self.movie_length=self.movie.shape[0]  
            lbl1 = tk.Label(master=self.viewFrame_detection, text="movie: "+self.movie_protein_path.split("/")[-1], bg='white')
            lbl1.grid(row=1, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)
              
            lbl1 = tk.Label(master=self.viewFrame_linking, text="movie: "+self.movie_protein_path.split("/")[-1], bg='white')
            lbl1.grid(row=1, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)
            
        
            # plot image
            self.show_frame_linking()
            self.show_frame_detection()
            
        
   #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_frame_detection() 
            self.show_frame_linking()
          
        self.scale_movie = tk.Scale(self.viewFrame_detection,  from_=0, to=self.movie_length-1, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

        self.scale_movie = tk.Scale(self.viewFrame_linking,  from_=0, to=self.movie_length-1, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
          
 
    def collect_linking_parameters(self):
        
        # parameters: TRACKER: STEP 1
        if self.l_tracker_distance_threshold.get()!='':
            self.detector.tracker_distance_threshold=int(self.l_tracker_distance_threshold.get())
            
        if self.l_tracker_max_skipped_frame.get()!='':
            self.detector.tracker_max_skipped_frame=int(self.l_tracker_max_skipped_frame.get())
            
        if self.l_tracker_max_track_length.get()!='':
            self.detector.tracker_max_track_length=int(self.l_tracker_max_track_length.get())
            
            
        # parameters: TRACKER: step 2 - tracklinking    
        if self.comboTopology.get()!='':
            self.detector.tracklinking_path1_topology=str(self.comboTopology.get())
            
        if self.l_tracklinking_path1_connectivity_threshold.get()!='':
            self.detector.tracklinking_path1_connectivity_threshold=float(self.l_tracklinking_path1_connectivity_threshold.get())
            
        if self.l_tracklinking_path1_frame_gap_0.get()!='':
            self.detector.tracklinking_path1_frame_gap_0=int(self.l_tracklinking_path1_frame_gap_0.get())
            
        if self.l_tracklinking_path1_frame_gap_1.get()!='':
            self.detector.tracklinking_path1_frame_gap_1=int(self.l_tracklinking_path1_frame_gap_1.get())
            
        if self.l_tracklinking_path1_distance_limit.get()!='':
            self.detector.tracklinking_path1_distance_limit=float(self.l_tracklinking_path1_distance_limit.get())
            
        if self.l_tracklinking_path1_direction_limit.get()!='':
            self.detector.tracklinking_path1_direction_limit=float(self.l_tracklinking_path1_direction_limit.get())
            
        if self.l_tracklinking_path1_speed_limit.get()!='':
            self.detector.tracklinking_path1_speed_limit=float(self.l_tracklinking_path1_speed_limit.get())
            
        if self.l_tracklinking_path1_intensity_limit.get()!='':
            self.detector.tracklinking_path1_intensity_limit=float(self.l_tracklinking_path1_intensity_limit.get())
            
        # start and end frame
            
        if self.start_frame.get()!='':
            self.detector.start_frame=int(self.start_frame.get())
            
        if self.end_frame.get()!='':
            self.detector.end_frame=int(self.end_frame.get())
        
        self.show_parameters()
        
    def run_test_linking(self):
        # read parameters from the buttons

        self.collect_linking_parameters()

        # movie 
        self.detector.movie=self.movie
        print("----------------parameters -----------------")
        print(" tracker_distance_threshold", self.detector.tracker_distance_threshold)
        print(" tracker_max_skipped_frame", self.detector.tracker_max_skipped_frame)
        print(" tracker_max_track_length \n", self.detector.tracker_max_track_length)
        
        print(" topology", self.detector.tracklinking_path1_topology)
        print(" tracklinking_path1_connectivity_threshold", self.detector.tracklinking_path1_connectivity_threshold)
        print(" tracklinking_path1_frame_gap_0", self.detector.tracklinking_path1_frame_gap_0)
        print(" tracklinking_path1_frame_gap_1", self.detector.tracklinking_path1_frame_gap_1)
        print(" tracklinking_path1_distance_limit", self.detector.tracklinking_path1_distance_limit)
        print(" tracklinking_path1_direction_limit", self.detector.tracklinking_path1_direction_limit)
        print(" tracklinking_path1_speed_limit", self.detector.tracklinking_path1_speed_limit)
        print(" tracklinking_path1_intensity_limit /n", self.detector.tracklinking_path1_intensity_limit)
        
        print(" start_frame", self.detector.start_frame)
        print(" end_frame", self.detector.end_frame)
        
        print("\n running linking ...")

        self.detector.linking()
        self.show_frame_linking()        
    
    def save_to_file_linking(self):
        # update parameters
        self.collect_linking_parameters()

        # choose the file
        filename=tk.filedialog.asksaveasfilename(title = "Save parameters into json file")
        
        if not filename:
            print("File was not chosen. Nothing will be saved! ")
        else:   
            self.detector.linking_parameter_path=filename
            self.detector.linking_parameter_to_file()
        
    def read_from_file_linking(self):
        
        # choose file
        filename = tk.filedialog.askopenfilename(title = "Open file with  linking parameters ")
        
        # read from the file  
        if not filename:
            print("File was not chosen. ")
        else:
            self.detector.linking_parameters_from_file(filename)
        
            # update frame
            self.set_linking_parameters_frame()

        
    def track_to_frame(self, data={}):
        # change data arrangment from tracks to frames
        self.track_data_framed={}
        self.track_data_framed.update({'frames':[]})
        
        for n_frame in range(0,self.movie_length):
            
            frame_dict={}
            frame_dict.update({'frame': n_frame})
            frame_dict.update({'tracks': []})

            #rearrange the data
            for trackID in data:
                p=data[trackID]
                if n_frame in p['frames']: # if the frame is in the track
                    frame_index=p['frames'].index(n_frame) # find position in the track
                    
                    new_trace=p['trace'][0:frame_index+1] # copy all the traces before the frame
                    frame_dict['tracks'].append({'trackID': p['trackID'], 'trace': new_trace}) # add to the list
                    
                    
            self.track_data_framed['frames'].append(frame_dict) # add the dictionary
            
    def show_frame_linking(self):    
    
        # plot image
        self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        
        self.axl.clear() # clean the plot 
        self.axl.imshow(self.image, cmap="gray")
        self.axl.axis('off')  
        
        # define the tracks to plot
       
        if self.monitor_switch_linking==0: # none
            self.track_to_frame()
        elif self.monitor_switch_linking==1: # tracklets
            self.track_to_frame(self.detector.tracklets)
        elif self.monitor_switch_linking==2: # tracks
            self.track_to_frame(self.detector.tracks)
    
        # plotting
        if  bool(self.track_data_framed): # if the dict is not empty
            # plot tracks
    #            print("self.frame_pos ", self.frame_pos)
    #            print(self.track_data_framed['frames'][self.frame_pos])
            for p in self.track_data_framed['frames'][self.frame_pos]['tracks']:
                trace=p['trace']
                plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                plt.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.figl, master=self.viewFrame_linking)
        self.canvas.get_tk_widget().grid(row=5, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        toolbarFrame = tk.Frame(master=self.viewFrame_linking)
        toolbarFrame.grid(row=10, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()


 ############# Running the code  ##################

    def run_tracking(self):
        '''
        running the final tracking 
        '''
        if self.r_start_frame.get()!='':
            self.detector.start_frame=int(self.r_start_frame.get())
            
        if self.r_end_frame.get()!='':
            self.detector.end_frame=int(self.r_end_frame.get())
        
        self.detector.movie=self.movie
        self.final_tracks=self.detector.linking()
        
        
        # save tracks        
        with open(self.result_path, 'w') as f:
            json.dump(self.final_tracks, f, ensure_ascii=False)
        
    
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    
    def close_app():
        quit()
    #closing the window
    root.protocol('WM_DELETE_WINDOW', close_app)

    tk.mainloop()
