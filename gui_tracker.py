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
#        print("window_width : ", self.window_width, ", window_height : ", self.window_height) #Width x Height
#        parent.geometry(str(self.window_width)+"x"+str(self.window_height)) #"1200x1000")
        
        
        # TABs 
        # set the stryle 
        style = ttk.Style()
        style.configure("TNotebook", foreground="black", background="white")
        
        tab_parent = ttk.Notebook(parent) # create tabs
        tab_detection = ttk.Frame(tab_parent, style="TNotebook")
        
        tab_parent.add(tab_detection, text=" Detection ")
        
        
        tab_linking = ttk.Frame(tab_parent)
        tab_parent.add(tab_linking, text=" Linking ")
        
        tab_membrane = ttk.Frame(tab_parent)
        tab_parent.add(tab_membrane, text=" Membrane segmentation ")
    
            
        tab_parent.pack(expand=1, fill='both')
        
        
        # detection
        detectionFrame = tk.Frame(master=tab_detection)
        detectionFrame.pack(expand=1, fill='both')
        DetectionViewer(detectionFrame)
        
        # linking 
        linkingFrame = tk.Frame(master=tab_linking)
        linkingFrame.pack(expand=1, fill='both')
        LinkingViewer(linkingFrame)
                        
        #closing the window
        parent.protocol('WM_DELETE_WINDOW', self.close_app)

        tk.mainloop()


    def close_app(self):
        self.quit()

class DetectionViewer(tk.Frame):
    '''
    class for the detection
    '''
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        #set the window size        
        self.window_width = int(master.winfo_screenwidth()/2) # half the monitor width
        self.window_height = int(master.winfo_screenheight()*0.8)  # 0.8 of the monitor height
        master.configure(background='white')
        
        ############################################
        self.movie=np.ones((1,200,200)) # matrix with data
        self.frame_pos=0
        self.movie_length=0
        self.detector=TrackingSetUp()
        self.monitor_switch=0
        self.pad_val=5
        self.dpi=100
        self.img_width=self.window_height*0.8
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        self.button_length=np.max((10,int(self.window_width/100)))
        
        
        #############################################
        
#        firstLabelTabOne = tk.Label(viewFrame, text=" Setting detection parameters", bg="white")
#        firstLabelTabOne.grid(row=0, column=0, padx=15, pady=15)

        # Framework: place monitor and view point
        self.viewFrame = tk.Frame(master=self.master, width=int(self.window_width*0.6), height=self.window_height, bg="white")
        self.viewFrame.grid(row=0, column=0, pady=self.pad_val, padx=self.pad_val)   

           
        # place parameters and buttons
        self.parametersFrame = tk.Frame(master=self.master, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.parametersFrame.grid(row=0, column=11, columnspan=1, rowspan=10, pady=self.pad_val, padx=self.pad_val)    




     # # # # # # # # # # # # # # # # # # # # # #    
        # Framework: place monitor 
        self.button_mv = tk.Button(self.viewFrame,text="   Select vesicle movie   ", command=self.select_vesicle_movie, width=20)
        self.button_mv.grid(row=0, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)        

        var_plot_detection = tk.IntVar()
        
        def update_detection_switch():            
            self.monitor_switch=var_plot_detection.get()
            # change image
            self.show_frame()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(self.viewFrame, text=" original image ", variable=var_plot_detection, value=0, bg='white', command =update_detection_switch )
        self.M1.grid(row=3, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(self.viewFrame, text=" candidates ", variable=var_plot_detection, value=1, bg='white',command = update_detection_switch ) #  command=sel)
        self.M2.grid(row=3, column=3, columnspan=3,  pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(self.viewFrame, text=" detection ", variable=var_plot_detection, value=2, bg='white',command = update_detection_switch ) #  command=sel)
        self.M3.grid(row=3, column=6, columnspan=3, pady=self.pad_val, padx=self.pad_val)
  
        

        # plot bg
        self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize_value, dpi=self.dpi)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.show_frame() 

   #   next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_tracks() 
          
        self.scale_movie = tk.Scale(self.viewFrame, from_=0, to=self.movie_length, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(self.viewFrame, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=7, column=1, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 

        
        buttonnext = tk.Button(self.viewFrame, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=7, column=7, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)  

        
  # # # # # # # # # # # # # # # # # # # # # # # # # 

        lbl3 = tk.Label(master=self.parametersFrame, text=" CANDIDATES DETECTION ",  bg='white')
        lbl3.grid(row=0, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # substract_bg_step background substraction step 

        lbl3 = tk.Label(master=self.parametersFrame, text=" Background evaluation: N frames ",  bg='white')
        lbl3.grid(row=1, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.substract_bg_step))
        self.d_substract_bg_step = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_substract_bg_step.grid(row=1, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # threshold coef

        lbl3 = tk.Label(master=self.parametersFrame, text=" Threshold coefficient  ",  bg='white')
        lbl3.grid(row=2, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.c))
        self.d_c = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_c.grid(row=2, column=1, pady=self.pad_val, padx=self.pad_val)

    # sigma
        lbl3 = tk.Label(master=self.parametersFrame, text=" Sigma : from  ",  bg='white')
        lbl3.grid(row=3, column=0)
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.sigma_min))
        self.d_sigma_min = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_sigma_min.grid(row=3, column=1, pady=self.pad_val, padx=self.pad_val)
         
        lbl3 = tk.Label(master=self.parametersFrame, text=" to ", bg='white')
        lbl3.grid(row=3, column=2, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.sigma_max))
        self.d_sigma_max = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_sigma_max.grid(row=3, column=3, pady=self.pad_val, padx=self.pad_val)
        
    # min_distance min distance minimum distance between two max after MSSEF

        lbl3 = tk.Label(master=self.parametersFrame, text=" Minimum distance between detections  ",  bg='white')
        lbl3.grid(row=4, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.min_distance))
        self.d_min_distance = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_min_distance.grid(row=4, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # self.threshold_rel min pix value in relation to the image
    
        lbl3 = tk.Label(master=self.parametersFrame, text=" Relevant peak height ",  bg='white')
        lbl3.grid(row=5, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.threshold_rel))
        self.d_threshold_rel = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_threshold_rel.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val)

            
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)        
                
        
        lbl3 = tk.Label(master=self.parametersFrame, text=" CANDIDATES PRUNING ",  bg='white')
        lbl3.grid(row=7, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    #self.box_size=16 # bounding box size for detection
        lbl3 = tk.Label(master=self.parametersFrame, text=" Region of Interest size ",  bg='white')
        lbl3.grid(row=8, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.box_size))
        self.d_box_size = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_box_size.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val)


    # detection_threshold threshold for the CNN based classification
        lbl3 = tk.Label(master=self.parametersFrame, text=" Threshold coefficient ",  bg='white')
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.detection_threshold))
        self.d_detection_threshold = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_detection_threshold.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # gaussian_fit gaussian fit
        lbl3 = tk.Label(master=self.parametersFrame, text=" Gaussian fit (True/False) ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.gaussian_fit))
        self.d_gaussian_fit = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.d_gaussian_fit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # cnn_model cnn model 
        lbl3 = tk.Button(master=self.parametersFrame, text=" Load CNN model ", command=self.load_cnn_model, width=self.button_length)
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val)  
        lbl3 = tk.Label(master=self.parametersFrame, text=self.detector.cnn_model_path,  bg='white')
        lbl3.grid(row=11, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
    
  # # # # # #  # #

         # empty space
        lbl3 = tk.Label(master=self.parametersFrame, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame, text=" Run test ", command=self.run_test, width=self.button_length*2, bg="#02f17a")
        lbl3.grid(row=15, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame, text=" Save to file ", command=self.save_to_file, width=self.button_length*2, bg="#00917a")
        lbl3.grid(row=16, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame, text=" Read from file ", command=self.read_from_file, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=17, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   
        
  # # # # # # # # # # # # # # # # # # # # # # # # #       
        
    def load_cnn_model(self):
        
        print("\n loading model here! \n")
        
    def run_test(self):
        
        print("\n run_test! \n")

    def save_to_file(self):
        
        print("\n save_to_file! \n")

    def read_from_file(self):
        
        print("\n read_from_file! \n")

        
    def move_to_previous(self):
        
        if self.frame_pos!=0:
            self.frame_pos-=1
 #       self.show_tracks()
        self.scale_movie.set(self.frame_pos) 
        
    def move_to_next(self):
        
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
  #      self.show_tracks()
        self.scale_movie.set(self.frame_pos) 
        
        
    def show_frame(self):    

        # plot image
        self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        
        self.ax.clear() # clean the plot 
        self.ax.imshow(self.image, cmap="gray")
        self.ax.axis('off')  
        
        # plot results
        
   
        if self.monitor_switch==1: # candidates
            if len(self.detector.detection_candidates)>0:
                for i in range(0, len(np.asarray(self.detector.detection_candidates))):
                    circle=plt.Circle((np.asarray(self.detector.detection_candidates)[i,1], np.asarray(self.detector.detection_candidates)[i,0]), 3, color="b", fill=False)
                    self.ax.add_artist(circle)    
        elif self.monitor_switch==2: # detection
            if len(self.detector.detection_vesicles)>0:
                for i in range(0, len(np.asarray(self.detector.detection_vesicles))):
                    circle=plt.Circle((np.asarray(self.detector.detection_vesicles)[i,1], np.asarray(self.detector.detection_vesicles)[i,0]), 3, color="b", fill=False)
                    self.ax.add_artist(circle)    


        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viewFrame)
        self.canvas.get_tk_widget().grid(row=5, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        toolbarFrame = tk.Frame(master=self.viewFrame)
        toolbarFrame.grid(row=10, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()
      
            
#            
    def select_vesicle_movie(self):
        
        filename = tk.filedialog.askopenfilename()
        self.movie_file=filename
        
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]  
        lbl1 = tk.Label(master=self.viewFrame, text="movie: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=1, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)
        
        # create a none-membrane movie
        self.membrane_movie=np.ones(self.movie.shape)
        
        # plot image
        self.show_frame()
        
        
   #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_frame() 
          
        self.scale_movie = tk.Scale(self.viewFrame,  from_=0, to=self.movie_length, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(self.viewFrame, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=7, column=1, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 
        
        buttonnext = tk.Button(self.viewFrame, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=7, column=7, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)   
 
class LinkingViewer(tk.Frame):
    '''
    class for the detection
    '''
    def __init__(self, master):
        tk.Frame.__init__(self, master)

        #set the window size        
        self.window_width = int(master.winfo_screenwidth()/2) # half the monitor width
        self.window_height = int(master.winfo_screenheight()*0.8)  # 0.8 of the monitor height
        master.configure(background='white')
        
        ############################################
        self.movie=np.ones((1,200,200)) # matrix with data
        self.frame_pos=0
        self.movie_length=0
        self.detector=TrackingSetUp()
        self.monitor_switch=0
        self.pad_val=5
        self.dpi=100
        self.img_width=self.window_height*0.8
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        self.button_length=np.max((10,int(self.window_width/100)))
        
        
        #############################################


        # Framework: place monitor and view point
        self.viewFrame = tk.Frame(master=self.master, width=int(self.window_width*0.6), height=self.window_height, bg="white")
        self.viewFrame.grid(row=0, column=0, pady=self.pad_val, padx=self.pad_val)   

           
        # place parameters and buttons
        self.parametersFrame = tk.Frame(master=self.master, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.parametersFrame.grid(row=0, column=11, columnspan=1, rowspan=10, pady=self.pad_val, padx=self.pad_val)    




     # # # # # # # # # # # # # # # # # # # # # #    
        # Framework: place monitor 
        self.button_mv = tk.Button(self.viewFrame,text="   Select vesicle movie   ", command=self.select_vesicle_movie, width=20)
        self.button_mv.grid(row=0, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)        

        var_plot_linking = tk.IntVar()
        
        def update_linking_switch():            
            self.monitor_switch=var_plot_linking.get()
            # change image
            self.show_frame()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(self.viewFrame, text=" original image ", variable=var_plot_linking, value=0, bg='white', command =update_linking_switch )
        self.M1.grid(row=3, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(self.viewFrame, text=" tracklets ", variable=var_plot_linking, value=1, bg='white',command = update_linking_switch ) #  command=sel)
        self.M2.grid(row=3, column=3, columnspan=3,  pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(self.viewFrame, text=" tracks ", variable=var_plot_linking, value=2, bg='white',command = update_linking_switch ) #  command=sel)
        self.M3.grid(row=3, column=6, columnspan=3, pady=self.pad_val, padx=self.pad_val)
  
        

        # plot bg
        self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize_value, dpi=self.dpi)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.show_frame() 

   #   next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_tracks() 
          
        self.scale_movie = tk.Scale(self.viewFrame, from_=0, to=self.movie_length, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(self.viewFrame, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=7, column=1, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 

        
        buttonnext = tk.Button(self.viewFrame, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=7, column=7, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)  

        
  # # # # # # # # # # # # # # # # # # # # # # # # # 

        lbl3 = tk.Label(master=self.parametersFrame, text=" TRACKER: STEP 1 ",  bg='white')
        lbl3.grid(row=0, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Maximum distance to link 

        lbl3 = tk.Label(master=self.parametersFrame, text=" Maximum distance to link, pix ",  bg='white')
        lbl3.grid(row=1, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracker_distance_threshold))
        self.l_tracker_distance_threshold = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracker_distance_threshold.grid(row=1, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # Maximum skipped frames

        lbl3 = tk.Label(master=self.parametersFrame, text=" Maximum skipped frames  ",  bg='white')
        lbl3.grid(row=2, column=0) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracker_max_skipped_frame))
        self.l_tracker_max_skipped_frame = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracker_max_skipped_frame.grid(row=2, column=1, pady=self.pad_val, padx=self.pad_val)

    # Maximum track length
        lbl3 = tk.Label(master=self.parametersFrame, text=" Maximum track length  ",  bg='white')
        lbl3.grid(row=3, column=0)
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracker_max_track_length))
        self.l_tracker_max_track_length = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracker_max_track_length.grid(row=3, column=1, pady=self.pad_val, padx=self.pad_val)
        
        
        
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=4, column=0, pady=self.pad_val, padx=self.pad_val)        
        
        lbl3 = tk.Label(master=self.parametersFrame, text=" TRACKER: step 2 - tracklinking ",  bg='white')
        lbl3.grid(row=5, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Topology

        lbl3 = tk.Label(master=self.parametersFrame, text=" Bayesian network topology ",  bg='white')
        lbl3.grid(row=7, column=0)     
        comboTopology = ttk.Combobox(master=self.parametersFrame, 
                            values=[
                                    "complete", 
                                    "no_intensity",
                                    "no_orientation",
                                    "no_motion",
                                    "no_gap"]) # comboTopology.get()
        comboTopology.grid(row=7, column=1) 
        comboTopology.current(0)

    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.parametersFrame, text=" Connectivity threshold [0,1]  ",  bg='white')
        lbl3.grid(row=8, column=0)
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_connectivity_threshold))
        self.l_tracklinking_path1_connectivity_threshold = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_connectivity_threshold.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val) 
        
        
    # tracklinking_path1_frame_gap_0 
        lbl3 = tk.Label(master=self.parametersFrame, text=" Small temporal gap, frames  ",  bg='white')
        lbl3.grid(row=9, column=0)
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_frame_gap_0))
        self.l_tracklinking_path1_frame_gap_0 = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_frame_gap_0.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.parametersFrame, text=" Large temporal gap, frames ", bg='white')
        lbl3.grid(row=9, column=2, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_frame_gap_1))
        self.l_tracklinking_path1_frame_gap_1 = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_frame_gap_1.grid(row=9, column=3, pady=self.pad_val, padx=self.pad_val)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.parametersFrame, text=" Distance limit, pix ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_distance_limit))
        self.l_tracklinking_path1_distance_limit = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_distance_limit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.parametersFrame, text=" Orientation similarity limit, degrees ",  bg='white')
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_direction_limit))
        self.l_tracklinking_path1_direction_limit = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_direction_limit.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val)

    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.parametersFrame, text=" Speed siimilarity limit ",  bg='white')
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_speed_limit))
        self.l_tracklinking_path1_speed_limit = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_speed_limit.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val)

    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.parametersFrame, text=" Intensity similarity limit, partition ",  bg='white')
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_intensity_limit))
        self.l_tracklinking_path1_intensity_limit = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_intensity_limit.grid(row=13, column=1, pady=self.pad_val, padx=self.pad_val)

    
        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.parametersFrame, text=" Threshold of track length, frames ",  bg='white')
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame, value=str(self.detector.tracklinking_path1_track_duration_limit))
        self.l_tracklinking_path1_track_duration_limit = tk.Entry(self.parametersFrame, width=self.button_length, text=v)
        self.l_tracklinking_path1_track_duration_limit.grid(row=14, column=1, pady=self.pad_val, padx=self.pad_val)
    
    
  # # # # # #  # #

         # empty space
        lbl3 = tk.Label(master=self.parametersFrame, text=" ",  bg='white', height=int(self.button_length/2))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame, text=" Run test ", command=self.run_test, width=self.button_length*2, bg="#02f17a")
        lbl3.grid(row=16, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame, text=" Save to file ", command=self.save_to_file, width=self.button_length*2, bg="#00917a")
        lbl3.grid(row=17, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame, text=" Read from file ", command=self.read_from_file, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=18, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   
        
  # # # # # # # # # # # # # # # # # # # # # # # # #       
        

        
    def run_test(self):
        
        print("\n run_test! \n")

    def save_to_file(self):
        
        print("\n save_to_file! \n")

    def read_from_file(self):
        
        print("\n read_from_file! \n")

        
    def move_to_previous(self):
        
        if self.frame_pos!=0:
            self.frame_pos-=1
 #       self.show_tracks()
        self.scale_movie.set(self.frame_pos) 
        
    def move_to_next(self):
        
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
  #      self.show_tracks()
        self.scale_movie.set(self.frame_pos) 
        
        
    def show_frame(self):    

        # plot image
        self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        
        self.ax.clear() # clean the plot 
        self.ax.imshow(self.image, cmap="gray")
        self.ax.axis('off')  
        
        # plot results
        
   
        if self.monitor_switch==1: # candidates
            if len(self.detector.detection_candidates)>0:
                for i in range(0, len(np.asarray(self.detector.detection_candidates))):
                    circle=plt.Circle((np.asarray(self.detector.detection_candidates)[i,1], np.asarray(self.detector.detection_candidates)[i,0]), 3, color="b", fill=False)
                    self.ax.add_artist(circle)    
        elif self.monitor_switch==2: # detection
            if len(self.detector.detection_vesicles)>0:
                for i in range(0, len(np.asarray(self.detector.detection_vesicles))):
                    circle=plt.Circle((np.asarray(self.detector.detection_vesicles)[i,1], np.asarray(self.detector.detection_vesicles)[i,0]), 3, color="b", fill=False)
                    self.ax.add_artist(circle)    


        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viewFrame)
        self.canvas.get_tk_widget().grid(row=5, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        toolbarFrame = tk.Frame(master=self.viewFrame)
        toolbarFrame.grid(row=10, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()
      
            
#            
    def select_vesicle_movie(self):
        
        filename = tk.filedialog.askopenfilename()
        self.movie_file=filename
        
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]  
        lbl1 = tk.Label(master=self.viewFrame, text="movie: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=1, column=0, columnspan=9, pady=self.pad_val, padx=self.pad_val)
        
        # create a none-membrane movie
        self.membrane_movie=np.ones(self.movie.shape)
        
        # plot image
        self.show_frame()

   #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_frame() 
          
        self.scale_movie = tk.Scale(self.viewFrame,  from_=0, to=self.movie_length, tickinterval=100, length=self.img_width, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=7, column=2, columnspan=5,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(self.viewFrame, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=7, column=1, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 
        
        buttonnext = tk.Button(self.viewFrame, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=7, column=7, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)   
               
    
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
