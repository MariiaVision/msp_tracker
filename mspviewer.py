
#########################################################
#
#  MSP-viewer GUI  v 0.3
#        
#########################################################


import numpy as np
import scipy as sp

import copy
import tkinter as tk
from tkinter import filedialog

import csv
import datetime
# for plotting
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure
from matplotlib import cm

import skimage
from skimage import io
from scipy.ndimage import gaussian_filter1d
import json        
import cv2
import imageio
import math
from skimage.feature import peak_local_max
from tqdm import tqdm

from viewer_lib.utils import SupportFunctions

from viewer_lib.trajectory_segmentation import TrajectorySegment

import os
import os.path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.rcParams.update({'figure.max_open_warning': 0})

class MainVisual(tk.Frame):
    '''
    Frame for the main msp-viewer window
    
    '''
    
    # choose the files and visualise the tracks on the data
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master

        
        #set the window size        
        self.window_width = int(master.winfo_screenwidth()/2.5) # half the monitor width
        self.window_height = int(master.winfo_screenheight()*0.7)  # 0.9 of the monitor height

        
        #colours for plotting tracks        
        self.color_list_plot=["#00FFFF", "#7FFFD4", "#0000FF", "#8A2BE2", "#7FFF00", "#D2691E", "#FF7F50", "#DC143C",
            "#008B8B", "#8B008B", "#FF8C00", "#E9967A", "#FF1493", "#9400D3", "#FF00FF", "#B22222",
            "#FFD700", "#ADFF2F", "#FF69B4", "#ADD8E6", "#F08080", "#90EE90", "#20B2AA", "#C71585", "#FF00FF"]
            
        self.color_list=[(200, 0, 0), (0, 255, 0), (0, 0, 255), (200, 155, 0),
                    (100, 255, 5), (255, 10, 120), (255, 127, 255),
                    (127, 0, 255), (200, 0, 127), (177, 0, 20), (12, 200, 0), (0, 114, 255), (255, 20, 0),
                    (0, 255, 255), (255, 100, 100), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
        
        self.movie_file=" " # path to the movie file
        self.track_file=" "# path to the file with tracking data
        self.movie=np.ones((1,200,200)) # matrix with data
        self.track_data_original={}
        self.track_data={'tracks':[]} # original tracking data
        self.track_data_filtered={'tracks':[]}  # filtered tracking data  
        self.track_data_framed={}  # tracking data arranged by frames  
        self.stat_data=[] # stat data to save csv file
        self.new_trackID=1 # default value or new track ID
        self.speed_sliding_window = 1 # length of the sliding window to evaluate speed
        
        # segmentation 
        self.tg = TrajectorySegment()     
        self.tg.window_length=8

        #filter parameters
        self.filter_duration=[0, float('Inf')]
        self.filter_length=[0, float('Inf')] 
        self.filter_length_total=[0, float('Inf')] 
        self.filter_speed=[0, float('Inf')] 
        self.filter_orientation=[float('-Inf'), float('Inf')] 
        self.filter_zoom=0 # option to include tracks only in zoomed area
        
        self.xlim_zoom=[0,float('Inf')]
        self.ylim_zoom=[float('Inf'), 0]
        
        self.frame_pos=0
        self.movie_length=1
        self.monitor_switch=0 # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.monitor_axis=0 # 0 - not to show axis, 1- show axis
        self.mode_orientation_diagram=0 # mode of the orientation diagram: 0 - based on number of tracks, 1 - based on the distance, 2-combined (normalised distance)
        self.pad_val=1
        self.axis_name="A,P" # axis names for the orientation plot
        self.img_resolution=100 # resolution of the image, default is 100 nm/pix 
        self.frame_rate=4 # frame rate, default is 4 f/s
        # 
        self.figsize_value=(4,4) # figure size
        # 
        self.deleted_tracks_N=0
        self.created_tracks_N=0
        self.filtered_tracks_N=0
        
        self.ap_axis=0
        self.plot_range_coordinates=[0,0] # zoom parameter
        
        # define window  proportions in relation to the monitor size
        self.button_length=np.max((20,int(self.window_width/50)))
        self.pad_val=2
        self.dpi=100
        self.img_width=self.window_height*0.6
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        
        ### frames ###
    
        # Framework: filtering tracks
        self.is_zoom_filtered=False # check is any tracks were filtered by zoom
        
        self.filterframe= tk.Frame(root, bg='white') 
        self.filterframe.grid(row=2, column=5, columnspan=4,rowspan=4,  pady=self.pad_val, padx=self.pad_val) 
        
        # Framework: saving/plotting results
        self.resultbuttonframe= tk.Frame(root, bg='white')
        self.resultbuttonframe.grid(row=13, column=5, columnspan=4,rowspan=4,  pady=self.pad_val, padx=self.pad_val)
        
        
        # Framework: place monitor navigation
        self.viewFrametool = tk.Frame(root, bg='white')
        self.viewFrametool.grid(row=13, column=0, columnspan=4,rowspan=2,  pady=self.pad_val, padx=self.pad_val)  

        # Framework: list navigation
        self.listframework = tk.Frame(root, bg='white') 
        self.listframework.grid(row=7, column=5, columnspan=4,rowspan=6,  pady=self.pad_val, padx=self.pad_val)           

        
     # # # # # # menu to choose files  # # # # # #
        
        self.button_mv = tk.Button(text="   Select image sequence   ", command=self.select_vesicle_movie, width=self.button_length)
        self.button_mv.grid(row=1, column=0, columnspan=2, pady=self.pad_val*3, padx=self.pad_val)
        
        self.button2 = tk.Button(text="Select file with tracks", command=self.select_track, width=self.button_length)
        self.button2.grid(row=1, column=2, columnspan=2, pady=self.pad_val, padx=self.pad_val)
  
        
#    # # # # # # Radiobuttone:tracks # # # # # # #   
        var = tk.IntVar()
        
        def update_monitor_switch():            
            self.monitor_switch=var.get()
            self.show_tracks()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(root, text="track and IDs", variable=var, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=3, column=0, pady=self.pad_val, padx=self.pad_val)  
        
        self.R2 = tk.Radiobutton(root, text=" only tracks ", variable=var, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=3, column=1, columnspan=2, pady=self.pad_val, padx=self.pad_val)
        
        self.R3 = tk.Radiobutton(root, text="    none    ", variable=var, value=2, bg='white',command=update_monitor_switch ) #  command=sel)
        self.R3.grid(row=3, column=3, pady=self.pad_val, padx=self.pad_val)
        
#    # # # # # # Radiobuttone:axis # # # # # # #   
        var_axis = tk.IntVar()
        
        def update_monitor_switch():            
            self.monitor_axis=var_axis.get()
            self.show_tracks()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(root, text="axis off", variable=var_axis, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=4, column=0,  pady=self.pad_val, padx=self.pad_val)  
        
        self.R2 = tk.Radiobutton(root, text=" axis on ", variable=var_axis, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=4, column=1, columnspan=2, pady=self.pad_val, padx=self.pad_val)     
        
#    # # # # # #  resolution in time and space   # # # # # # #  
            
        res_lb = tk.Label(master=root, text=" resolution (nm/pix) : ", width=self.button_length, bg='white')
        res_lb.grid(row=5, column=0, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.img_resolution))
        self.res_parameter = tk.Entry(root, width=10, text=v)
        self.res_parameter.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val)
            
        lbl3 = tk.Label(master=root, text=" frame rate (f/sec) : ", width=self.button_length, bg='white')
        lbl3.grid(row=5, column=2, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.frame_rate))
        self.frame_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
        self.frame_parameter.grid(row=5, column=3, pady=self.pad_val, padx=self.pad_val)        
        
            
        # AP axis 
        ap_lb = tk.Label(master=root, text=" Axis orientation ", width=self.button_length, bg='white')
        ap_lb.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.ap_axis))
        self.ap_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
        self.ap_parameter.grid(row=6, column=1, pady=self.pad_val, padx=self.pad_val)
            
        lbl3 = tk.Label(master=root, text="Axis  (A,B): ", width=self.button_length, bg='white')
        lbl3.grid(row=6, column=2, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.axis_name))
        self.axis_name_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
        self.axis_name_parameter.grid(row=6, column=3, pady=self.pad_val, padx=self.pad_val)   
       
        
        #update the list

        # show the list of data with scroll bar       
        self.scrollbar_tracks = tk.Scrollbar(master=self.listframework, orient="vertical")
        self.scrollbar_tracks.grid(row=12, column=9,  sticky=tk.N+tk.S,padx=self.pad_val)

        self.listNodes_tracks = tk.Listbox(master=self.listframework, width=self.button_length*3, height=int(self.img_width/20),  font=("Times", 12), selectmode='single')
        self.listNodes_tracks.grid(row=12, column=5, columnspan=4, sticky=tk.N+tk.S,padx=self.pad_val)
        self.listNodes_tracks.config(yscrollcommand=self.scrollbar_tracks.set)
        self.listNodes_tracks.bind('<Double-1>', self.tracklist_on_select)

        self.scrollbar_tracks.config(command=self.listNodes_tracks.yview)
        
        #delete button
        
        self.deletbutton = tk.Button(master=self.resultbuttonframe, text="DELETE TRACK", command=self.detele_track_question, width=int(self.button_length*0.8),  bg='red')
        self.deletbutton.grid(row=13, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        # add button

        self.deletbutton = tk.Button(master=self.resultbuttonframe, text="ADD TRACK", command=self.new_track_question, width=int(self.button_length*0.8),  bg='green')
        self.deletbutton.grid(row=14, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val) 

        # duplicate button
        self.duplicatebutton = tk.Button(master=self.resultbuttonframe, text="DUPLICATE TRACK", command=self.duplicate_track_question, width=int(self.button_length*0.8),  bg='green')
        self.duplicatebutton.grid(row=15, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
        # duplicate button
        self.duplicatebutton = tk.Button(master=self.resultbuttonframe, text="MERGE TRACKS", command=self.merge_track_question, width=int(self.button_length*0.8),  bg='green')
        self.duplicatebutton.grid(row=16, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
        
        self.list_update()        
        
        
 # # # # # # #   Filtering frame   # # # # # # #  


        # trackID
        lbl3 = tk.Label(master=self.filterframe, text="Track ID: ", width=int(self.button_length/2), bg='white')
        lbl3.grid(row=2, column=5, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_track_number = tk.Entry(self.filterframe, width=int(self.button_length))
        self.txt_track_number.grid(row=2, column=6, columnspan=3,pady=self.pad_val, padx=self.pad_val)

        # duration
        lbl3 = tk.Label(master=self.filterframe, text="Duration (sec): from ", width=self.button_length*2, bg='white')
        lbl3.grid(row=3, column=5, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_duration_from = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_duration_from.grid(row=3, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl3 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl3.grid(row=3, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_duration_to = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_duration_to.grid(row=3, column=8, pady=self.pad_val, padx=self.pad_val)


        # Length       
        lbl3 = tk.Label(master=self.filterframe, text="Net travelled distance (nm): from ", width=int(self.button_length*2), bg='white')
        lbl3.grid(row=4, column=5)
        
        self.txt_length_from = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_length_from.grid(row=4, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl3 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl3.grid(row=4, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_length_to = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_length_to.grid(row=4, column=8, pady=self.pad_val, padx=self.pad_val)  
               
        
        lbl3 = tk.Label(master=self.filterframe, text="Total travelled distance (nm): from ", width=int(self.button_length*2), bg='white')
        lbl3.grid(row=5, column=5)
        
        self.txt_length_from_total = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_length_from_total.grid(row=5, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl3 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl3.grid(row=5, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_length_to_total = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_length_to_total.grid(row=5, column=8, pady=self.pad_val, padx=self.pad_val)  
        
        # curvilinear moving speed
        lbl4 = tk.Label(master=self.filterframe, text="Mean curvilinear moving speed : from ", width=int(self.button_length*2), bg='white')
        lbl4.grid(row=6, column=5)
        
        self.txt_speed_from = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_speed_from.grid(row=6, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl5 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl5.grid(row=6, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_speed_to = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_speed_to.grid(row=6, column=8, pady=self.pad_val, padx=self.pad_val)
        
        
        # orientation
        lbl14 = tk.Label(master=self.filterframe, text="Trajectory net orientation : from ", width=int(self.button_length*2), bg='white')
        lbl14.grid(row=7, column=5)
        
        self.txt_orientation_from = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_orientation_from.grid(row=7, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl15 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl15.grid(row=7, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_orientation_to = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_orientation_to.grid(row=7, column=8, pady=self.pad_val, padx=self.pad_val)
        
        
        
        # Radio button zoom
        var_filter_zoom = tk.IntVar()
        
        def update_monitor_switch():            
            self.filter_zoom=var_filter_zoom.get()
            
        lbl5 = tk.Label(master=self.filterframe, text=" Trajectories included for zoomed area: ", width=int(self.button_length*2), bg='white')
        lbl5.grid(row=12, column=5)

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(master=self.filterframe, text=" at least one point", variable=var_filter_zoom, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=12, column=6,  pady=self.pad_val, padx=self.pad_val)  
        
        self.R2 = tk.Radiobutton(master=self.filterframe, text=" all points ", variable=var_filter_zoom, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=12, column=8, pady=self.pad_val, padx=self.pad_val)          
        
        # button to filter
        
        self.buttonFilter = tk.Button(master=self.filterframe, text=" Filter ", command=self.filtering, width=self.button_length)
        self.buttonFilter.grid(row=13, column=4, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
        self.buttonFilter = tk.Button(master=self.filterframe, text=" Update ", command=self.filtering, width=self.button_length)
        self.buttonFilter.grid(row=14, column=4, columnspan=4, pady=self.pad_val, padx=self.pad_val)  
 


# # # # # # # # # # # # Plot and save data  # # # # # # 

        # motion map         
        button_save=tk.Button(master=self.resultbuttonframe, text="orientation: save map", command=self.plot_motion_map, width=int(self.button_length*1.5))
        button_save.grid(row=13, column=6, pady=self.pad_val, padx=self.pad_val)

        button_save=tk.Button(master=self.resultbuttonframe, text="orientation: save info", command=self.save_orientation_info, width=int(self.button_length*1.5))
        button_save.grid(row=14, column=6, pady=self.pad_val, padx=self.pad_val)
        
        button_save=tk.Button(master=self.resultbuttonframe, text="orientation: joint map", command=self.plot_multiple_motion_map, width=int(self.button_length*1.5))
        button_save.grid(row=15, column=6, pady=self.pad_val, padx=self.pad_val)        
        
        # button to save all the tracks on the image
        button_save=tk.Button(master=self.resultbuttonframe, text="trajectories: save image", command=self.save_track_plot, width=int(self.button_length*1.5))
        button_save.grid(row=13, column=7, pady=self.pad_val, padx=self.pad_val)
        
        # button to save movie
        button_save=tk.Button(master=self.resultbuttonframe, text="trajectories: save movie", command=self.save_movie, width=int(self.button_length*1.5))
        button_save.grid(row=14, column=7, pady=self.pad_val, padx=self.pad_val)
              
        # save information about all the filtered trajectories
        button_save=tk.Button(master=self.resultbuttonframe, text="trajectories: save info to csv", command=self.save_data_csv, width=int(self.button_length*1.5))
        button_save.grid(row=15, column=7, pady=self.pad_val, padx=self.pad_val)  
        
        # save corrected tracks
        button_save=tk.Button(master=self.resultbuttonframe, text="trajectories: save changes", command=self.save_in_file, width=int(self.button_length*1.5))
        button_save.grid(row=16, column=7, pady=self.pad_val, padx=self.pad_val)  
        
      # # # # # # movie  # # # # # # 
        
        # plot bg
        self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize_value)
        self.ax.axis('off')
        
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=12, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        self.toolbarFrame = tk.Frame(master=root)
        self.toolbarFrame.grid(row=15, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        
        self.show_tracks() 
        
    
        # next and previous buttons

        def show_values(v):
            self.frame_pos=int(v)
            self.show_tracks() 
          
        self.scale_movie = tk.Scale(self.viewFrametool, from_=0, to=self.movie_length, tickinterval=100, length=self.figsize_value[1]*self.dpi, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=0, column=1,  sticky=tk.W)
        
        buttonbefore = tk.Button(self.viewFrametool, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=0, column=0, sticky=tk.E) 

        
        buttonnext = tk.Button(self.viewFrametool, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=0, column=2, sticky=tk.W)        
    
    def save_track_plot(self):
        '''
        save the plot of all the tracks on a single frame
        '''
        
                
        # request file name
        save_file = tk.filedialog.asksaveasfilename()  
        
        if not save_file:
            print("File name was not provided. The data was not saved.")
        else:

            plt.figure()
            plt.imshow(self.image, cmap="gray")
            for trackID in range(0, len(self.track_data_filtered['tracks'])):
                track=self.track_data_filtered['tracks'][trackID]
                
                if len(track['trace'])>0:
                    trace=track['trace']
                    plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(trackID)%len(self.color_list_plot)])     
                    if self.monitor_switch==0:
                        plt.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(track['trackID']), fontsize=10, color=self.color_list_plot[int(trackID)%len(self.color_list_plot)])
                
            # plot
            if not(save_file.endswith(".png")):
                save_file += ".png"  
            
                if os.path.isfile(save_file)==True:
                    # add date if the file exists already
                    now = datetime.datetime.now()
                    save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]
                
            # save the image
            plt.savefig(save_file)
            # close the image
            plt.close()     
        
        
    def update_movie_parameters(self):
        '''
        read movie parameters and orientation given by the user
        
        '''
        
        # read movie parameters
        if self.res_parameter.get()!='':
            self.img_resolution=int(self.res_parameter.get())
        
        if self.frame_parameter.get()!='':
            self.frame_rate=float(self.frame_parameter.get())
        
        #read ap axis
        if self.ap_parameter.get()!='':
            self.ap_axis=int(self.ap_parameter.get())
            
        if self.axis_name_parameter.get()!='':
            self.axis_name=self.axis_name_parameter.get()
        
    def save_orientation_info(self):
        '''
        save track orientation into file
        
        '''        
        
        # request file name name
        save_file = tk.filedialog.asksaveasfilename()
         
        if not save_file:
            print("File name was not provided. The data was not saved.")
        else:
            
            #read ap axis
            if self.ap_parameter.get()!='':
                self.ap_axis=int(self.ap_parameter.get())
                
            if self.axis_name_parameter.get()!='':
                self.axis_name=self.axis_name_parameter.get()
      
            #calculate orientation for each trajectory
            orintation_array=[]
            distance_array=[]
            
            for trackID in range(0, len(self.track_data_filtered['tracks'])):
                track=self.track_data_filtered['tracks'][trackID]
            #    calculate parameters
                point_start=track['trace'][0]
                point_end=track['trace'][-1]
    
                # calculate orientation
                y=point_end[1]-point_start[1]
                x=point_end[0]-point_start[0]

                orintation_move=(math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360                
                
                if orintation_move>180:
                    orintation_move=abs(orintation_move-360)
                    
                # calculate distance
                net_displacement=np.round(np.sqrt((x)**2+(y)**2),2)*self.img_resolution
                
                
                orintation_array.append(orintation_move) 
                distance_array.append(net_displacement) 
                
            # save the array into the file
    
            
            if not(save_file.endswith(".txt")):
                save_file += ".txt"  
                
                if os.path.isfile(save_file)==True:
                    # add date if the file exists already
                    now = datetime.datetime.now()
                    save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]
                
                
            # save in json format                    
            with open(save_file, 'w') as f:
                json.dump({'orientation':orintation_array, 'distance': distance_array}, f, ensure_ascii=False) 
        
    def plot_multiple_motion_map(self):
        '''
        load and plot multiple orientations together
        
        '''
        
        def cancel_window():
            '''
            destroy the window
            
            '''
            try:
                self.choose_diagram_settings.destroy()
            except:
                pass
                    
            
        def run_main_code():
            '''
            the main code run after OK button
            '''
                        # plot the image
                       
            # create the joint orientation list            
            orientation_all=[]
            orientation=[]
            distance_all=[]
            title_text=" trajectory orientation " # title of the plot
            
            for file_name in load_files:
                
                #read from json format 
                with open(file_name) as json_file:  
                    orientation_new = json.load(json_file)
                    
                orientation+=orientation_new['orientation']       
                distance_all+=orientation_new['distance'] 
                
            
            for pos in orientation:
                if pos==180:
                    pos=179.99
                orientation_all+=[pos]
                
            
            # axis name               
            if self.axis_name_parameter.get()!='':
                self.axis_name=self.axis_name_parameter.get()
                
                
            axis_name=self.axis_name.split(",")
            if axis_name[0]:
                first_name=axis_name[0]
            else:
                first_name=" "
            
            if axis_name[1]:
                second_name=axis_name[1]
            else:
                second_name=" "
                
                
            #define weights:
            
            
            if self.mode_orientation_diagram==0: # track based
                distance_array=[1]*len(distance_all)
                title_text=title_text+"\n based on track count "
            elif self.mode_orientation_diagram==1: # distance based 
                distance_array=distance_all
                distance_array=[x / 1000 for x in distance_array]
                title_text=title_text+"\n based on net distance travelled [$\mu$m]"
            else:
                print("something went wrong")
            
            # plot and save 
            orientation_fig=plt.figure(figsize=(10,8))
            
            ax_new = plt.subplot(111, projection='polar')
            
            bin_size=10
            
            a , b=np.histogram(orientation_all, bins=np.arange(0, 360+bin_size, bin_size), weights=distance_array)
            centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1]) 

                


            # if scale is provided 
            
            if self.set_range.get()!='':
                ylim_max=int(self.set_range.get())
            
                ax_new.set_ylim([0, ylim_max])
                
            # if scale factor for the diagram is provided
            if self.set_norm_val.get()!='':
                
                norm_factor=float(self.set_norm_val.get())
                a=a/norm_factor
                title_text=title_text+" normalised by "+str(norm_factor)
                
            
            if self.mode_orientation_diagram==0: # track based
                plt.xticks(np.radians((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180)),
                   ["\n \n  \n "+second_name+"\n \n number of tracks", '10', '20', '30', '40', '50', '60', '70', '80', '90' , '100', '110', '120', '130', '140', '150', '160', '170' ,first_name])
                 
                ax_new.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.7', alpha=0.5)
                ax_new.set_theta_direction(1)
                ax_new.set_title(title_text)
                ax_new.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)) # provide only integer
                
            elif self.mode_orientation_diagram==1: # distance based
                plt.xticks(np.radians((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180)),
                   ["\n \n  \n "+second_name+"\n \n total net distance, $\mu$m", '10', '20', '30', '40', '50', '60', '70', '80', '90' , '100', '110', '120', '130', '140', '150', '160', '170' ,first_name])
                
                ax_new.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.7', alpha=0.5)
                ax_new.set_theta_direction(1)                
                ax_new.set_title(title_text)
                
            else: 
                print("something went wrong")        
            
            ax_new.set_thetamin(0)
            ax_new.set_thetamax(180)
            #set a window
            self.show_orientation_map_win = tk.Toplevel( bg='white')
            self.show_orientation_map_win.title(" orientation plot ")
            self.canvas_img = FigureCanvasTkAgg(orientation_fig, master=self.show_orientation_map_win)
            self.canvas_img.get_tk_widget().pack(expand = tk.YES, fill = tk.BOTH)
            self.canvas_img.draw()        
        
            # request file name
            save_file = tk.filedialog.asksaveasfilename()         
            
            if not save_file:
                print("File name was not provided. The data was not saved.")
            else:
                
                if not(save_file.endswith(".png")):
                    save_file += ".png"     

                    if os.path.isfile(save_file)==True:
                        # add date if the file exists already
                        now = datetime.datetime.now()
                        save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]
                
                plt.savefig(save_file) 
                
            # close the widnow with question
            cancel_window()                

                

            
            # load multiple files
        load_files = tk.filedialog.askopenfilenames(title='Choose all files together')
        print(load_files)
         
        if not load_files:
            print("Files were not selected. The data will not be processed.")
        else:
                        
                    
            # ask for the orientation diagram mode
            
            #default value of diagram mode=0
            self.mode_orientation_diagram=0
            # open new window
            self.choose_diagram_settings = tk.Toplevel(root,  bg='white')
            self.choose_diagram_settings.title(" ")
        
            
            self.qnewtext = tk.Label(master=self.choose_diagram_settings, text=" Plot orientation diagram based on  " ,  bg='white', font=("Times", 10))
            self.qnewtext.grid(row=0, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
            
            # radiobutton to choose
            
            var_diagram_switch = tk.IntVar()
            
            def update_switch():            
                self.mode_orientation_diagram=var_diagram_switch.get()
                
         
            segmentation_switch_off = tk.Radiobutton(master=self.choose_diagram_settings,text=" track count ",variable=var_diagram_switch, value=0, bg='white', command =update_switch )
            segmentation_switch_off.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)   
    
            segmentation_switch_msd = tk.Radiobutton(master=self.choose_diagram_settings,text=" Net distance travelled",variable=var_diagram_switch, value=1, bg='white', command =update_switch )
            segmentation_switch_msd.grid(row=1, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)    
            
#            segmentation_switch_unet = tk.Radiobutton(master=self.choose_diagram_settings,text=" Net distance travelled \n normalised by movie length", variable=var_diagram_switch, value=2, bg='white', command =update_switch )
#            segmentation_switch_unet.grid(row=1, column=3, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.qnewtext = tk.Label(master=self.choose_diagram_settings, text=" To set diagram range provide max value to display:  " ,  bg='white', font=("Times", 10))
            self.qnewtext.grid(row=2, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val)          
            
            self.set_range = tk.Entry(master=self.choose_diagram_settings, width=int(self.button_length/2))
            self.set_range.grid(row=2, column=4, pady=self.pad_val, padx=self.pad_val)  
            
            self.qnewtext = tk.Label(master=self.choose_diagram_settings, text=" Scale factor to normalise the values:  " ,  bg='white', font=("Times", 10))
            self.qnewtext.grid(row=3, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val)          
            
            self.set_norm_val = tk.Entry(master=self.choose_diagram_settings, width=int(self.button_length/2))
            self.set_norm_val.grid(row=3, column=4, pady=self.pad_val, padx=self.pad_val)  
                    
                    
            self.newbutton = tk.Button(master=self.choose_diagram_settings, text=" OK ", command=run_main_code, width=int(self.button_length/2),  bg='green')
            self.newbutton.grid(row=4, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.choose_diagram_settings, text=" Cancel ", command=cancel_window, width=int(self.button_length/2))
            self.deletbutton.grid(row=4, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
  
                                    

        
    def plot_motion_map(self):
        '''
        plot motion map with given AP
        
        '''       
        def cancel_window():
            '''
            destroy the window
            
            '''
            try:
                self.choose_diagram_settings.destroy()
            except:
                pass
                    
        def run_main_code():
            '''
            the main code run after OK button
            
                '''
        # read ap axis
            if self.ap_parameter.get()!='':
                self.ap_axis=int(self.ap_parameter.get())
                
            if self.axis_name_parameter.get()!='':
                self.axis_name=self.axis_name_parameter.get()
      
            orintation_array=[]
            distance_array=[]
            title_text=" trajectory orientation " # title of the plot
            
            orientation_map_figure = plt.figure(figsize=(15,6))
            plt.axis('off')
            ax = orientation_map_figure.add_subplot(121)
            
            # set zoom as in the main window
            # read limits
            xlim_old=self.ax.get_xlim()
            ylim_old=self.ax.get_ylim()
            
            lim_x0=int(ylim_old[1])
            lim_x1=int(ylim_old[0])
            lim_y0=int(xlim_old[0]) # because y-axis is inverted
            lim_y1=int(xlim_old[1]) # because y-axis is inverted

            img_to_show=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
            img_to_show=img_to_show[lim_x0:lim_x1,lim_y0:lim_y1]
            ax.imshow(img_to_show, cmap='bone')
            
            
            # position of axis
    
            axis_name=self.axis_name.split(",")
            
            if axis_name[0]:
                first_name=axis_name[0]
            else:
                first_name=" "
            
            if axis_name[1]:
                second_name=axis_name[1]
            else:
                second_name=" "
    
            arrow_a=[int(self.image.shape[0]/10),int(self.image.shape[1]/10)]
            dist=int(self.image.shape[1]/10)
            arrow_b=[int(dist*math.cos(math.radians(self.ap_axis-90))+arrow_a[0]),int(dist*math.sin(math.radians(self.ap_axis-90))+arrow_a[1])]
            
            # check that the points are not outside the view
    
            if arrow_b[0]<int(self.image.shape[0]/10):
                # move the points
                arrow_a[0]=arrow_a[0]+int(self.image.shape[0]/10)
                arrow_b[0]=arrow_b[0]+int(self.image.shape[0]/10)
    
            if arrow_b[1]<int(self.image.shape[1]/10):
                # move the points
                arrow_a[1]=arrow_a[1]+int(self.image.shape[1]/10)
                arrow_b[1]=arrow_b[1]+int(self.image.shape[1]/10)
                
            ax.plot([arrow_a[1], arrow_b[1]], [arrow_a[0], arrow_b[0]],  color='w', alpha=0.7)
            ax.text(arrow_a[1], arrow_a[0]-3,  second_name, color='magenta', size=12, alpha=0.7)
            ax.text(arrow_b[1], arrow_b[0]-3,  first_name, color='green', size=12, alpha=0.7)
    
            for trackID in range(0, len(self.track_data_filtered['tracks'])):
                track=self.track_data_filtered['tracks'][trackID]
                
                # calculate parameters
                point_start=track['trace'][0]
                point_end=track['trace'][-1]
    
                # calculate orientation
                y=point_end[1]-point_start[1]
                x=point_end[0]-point_start[0]                

                orintation_move=(math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360
                
                # define weights
                
                if self.mode_orientation_diagram==0: # track based
                    net_displacement=1
                else: # distance based or combined
                     net_displacement=np.round(np.sqrt((x)**2+(y)**2),2)*self.img_resolution
    
                if orintation_move>180:
                    
                    orintation_move=abs(orintation_move-360)
                    
                if orintation_move==180:
                    orintation_move=179.99
                
                orintation_array.append(orintation_move)
                distance_array.append(net_displacement)
                
                # select colour
                if orintation_move<45:
                    color='magenta'
                elif orintation_move>135: 
                    color='green'
                else:
                    color='gold'
                    
                plt.arrow(point_start[1],point_start[0], point_end[1]-point_start[1], point_end[0]-point_start[0], head_width=3.00, head_length=2.0, 
                          fc=color, ec=color, length_includes_head = True)
            
            if self.mode_orientation_diagram==1: # distance mode
               
            # move to micrometers for histogram
                distance_array=[x / 1000 for x in distance_array]
                title_text=title_text+"\n based on net distance travelled [$\mu$m]"
                
            if self.mode_orientation_diagram==0: 
                title_text=title_text+"\n based on track count "
                
            bin_size=10       
            
            a , b=np.histogram(orintation_array, bins=np.arange(0, 360+bin_size, bin_size), weights=distance_array)
            centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1]) 
                
            
            ax = orientation_map_figure.add_subplot(122, projection='polar')

            # if scale is provided 
            
            if self.set_range.get()!='':
                ylim_max=int(self.set_range.get())            
                ax.set_ylim([0, ylim_max])
                
                
            if self.set_norm_val.get()!='':
                norm_factor=float(self.set_norm_val.get())
                a=a/norm_factor
                title_text=title_text+" normalised by "+str(norm_factor)
                
            if self.mode_orientation_diagram==0: # track based
    
                plt.xticks(np.radians((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180)),
                   ["\n \n  \n "+second_name+"\n \n number of tracks", '10', '20', '30', '40', '50', '60', '70', '80', '90' , '100', '110', '120', '130', '140', '150', '160', '170' ,first_name])
                ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.7', alpha=0.5)
                ax.set_theta_direction(1)
            
                ax.set_title(title_text)

                ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True)) # provide only integer
                
                
            elif self.mode_orientation_diagram==1: # distance based    
                plt.xticks(np.radians((0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180)),
                   ["\n \n  \n "+second_name+"\n \n total net distance, $\mu$m", '10', '20', '30', '40', '50', '60', '70', '80', '90' , '100', '110', '120', '130', '140', '150', '160', '170' ,first_name])
                ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.7', alpha=0.5)
                ax.set_theta_direction(1)
                
                ax.set_title(title_text)


            else: # distance normalised by the track count
                print("something went wrong")     
                
            
            ax.set_thetamin(0)
            ax.set_thetamax(180)        
            #set a window
            self.show_orientation_map_win = tk.Toplevel( bg='white')
            self.show_orientation_map_win.title(" orientation plot ")
            self.canvas_img = FigureCanvasTkAgg(orientation_map_figure, master=self.show_orientation_map_win)
            self.canvas_img.get_tk_widget().pack(expand = tk.YES, fill = tk.BOTH)
            self.canvas_img.draw()
                        
            # request file name
            save_file = tk.filedialog.asksaveasfilename() 
            
            if not save_file:
                print("File name was not provided. The data was not saved. ")
            else: 
                           # plot
                if not(save_file.endswith(".png")):
                    save_file += ".png"        
                
                    if os.path.isfile(save_file)==True:
                        # add date if the file exists already
                        now = datetime.datetime.now()
                        save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]
                          
                plt.savefig(save_file) 
                
            # close the widnow with question
            cancel_window()
                
                
                        
        ###  show the results in a separate window  ###
                
        # ask for the orientation diagram mode
        
            
        #default value of diagram mode=0
        self.mode_orientation_diagram=0
        
        # open new window
        self.choose_diagram_settings = tk.Toplevel(root,  bg='white')
        self.choose_diagram_settings.title(" ")
    
        
        self.qnewtext = tk.Label(master=self.choose_diagram_settings, text=" Plot orientation diagram based on  " ,  bg='white', font=("Times", 10))
        self.qnewtext.grid(row=0, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
        
        # radiobutton to choose
        
        var_diagram_switch = tk.IntVar()
        
        def update_switch():            
            self.mode_orientation_diagram=var_diagram_switch.get()
            
     
        segmentation_switch_off = tk.Radiobutton(master=self.choose_diagram_settings,text=" track count ",variable=var_diagram_switch, value=0, bg='white', command =update_switch )
        segmentation_switch_off.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)   

        segmentation_switch_msd = tk.Radiobutton(master=self.choose_diagram_settings,text=" Net distance travelled",variable=var_diagram_switch, value=1, bg='white', command =update_switch )
        segmentation_switch_msd.grid(row=1, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)    
        
#        segmentation_switch_unet = tk.Radiobutton(master=self.choose_diagram_settings,text=" Net distance travelled \n normalised by the movie length", variable=var_diagram_switch, value=2, bg='white', command =update_switch )
#        segmentation_switch_unet.grid(row=1, column=3, columnspan=1, pady=self.pad_val, padx=self.pad_val)             
        
        self.qnewtext = tk.Label(master=self.choose_diagram_settings, text=" To set diagram range provide max value to display:  " ,  bg='white', font=("Times", 10))
        self.qnewtext.grid(row=2, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val)          
        
        self.set_range = tk.Entry(master=self.choose_diagram_settings, width=int(self.button_length/2))
        self.set_range.grid(row=2, column=4, pady=self.pad_val, padx=self.pad_val)  
    

            
        self.qnewtext = tk.Label(master=self.choose_diagram_settings, text=" Scale factor to normalise the values:  " ,  bg='white', font=("Times", 10))
        self.qnewtext.grid(row=3, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val)          
        
        self.set_norm_val = tk.Entry(master=self.choose_diagram_settings, width=int(self.button_length/2))
        self.set_norm_val.grid(row=3, column=4, pady=self.pad_val, padx=self.pad_val)  
                        
            
        self.newbutton = tk.Button(master=self.choose_diagram_settings, text=" OK ", command=run_main_code, width=int(self.button_length/2),  bg='green')
        self.newbutton.grid(row=4, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        self.deletbutton = tk.Button(master=self.choose_diagram_settings, text=" Cancel ", command=cancel_window, width=int(self.button_length/2))
        self.deletbutton.grid(row=4, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
  
                
        
############################################################
        

        
    def save_movie(self):
        '''
        save image sequence with plotted trajectories
        
        '''
        
        # update movie parameters
        self.update_movie_parameters()
        # run save movie
        
        f_start=0
        f_end=self.movie.shape[0]
        
        # request file name
        save_file = tk.filedialog.asksaveasfilename() 
        
        if not save_file:
            print("File name was not provided. The data was not saved. ")
        else: 
        
            # read limits
            xlim_old=self.ax.get_xlim()
            ylim_old=self.ax.get_ylim()
            
            lim_x0=int(ylim_old[1])
            lim_x1=int(ylim_old[0])
            lim_y0=int(xlim_old[0]) # because y-axis is inverted
            lim_y1=int(xlim_old[1]) # because y-axis is inverted

            saved_movie=self.movie[f_start:f_end,lim_x0:lim_x1,lim_y0:lim_y1]
            
            
            try: 
                 # save tiff file
                final_img_set=np.zeros((saved_movie.shape[0],saved_movie.shape[1],saved_movie.shape[2], 3), dtype=np.uint8)
    
                for frameN in range(0, saved_movie.shape[0]):
              
                    plot_info=self.track_data_framed['frames'][frameN]['tracks']
                    frame_img=saved_movie[frameN,:,:]
                    # make a colour image frame
                    orig_frame = np.zeros((saved_movie.shape[1], saved_movie.shape[2], 3), dtype=np.uint8)
            
                    img=frame_img/np.max(frame_img)
                    orig_frame [:,:,0] = img/np.max(img)*256
                    orig_frame [:,:,1] = img/np.max(img)*256
                    orig_frame [:,:,2] = img/np.max(img)*256

                    for p in plot_info:
                        trace=p['trace']
                        trackID=p['trackID']
                        
                        clr = trackID % len(self.color_list)
                        if (len(trace) > 1):
                            for j in range(len(trace)-1):
                                # draw trace line
                                point1=trace[j]
                                point2=trace[j+1]
                                x1 = int(point1[1])-lim_y0
                                y1 = int(point1[0])-lim_x0
                                x2 = int(point2[1])-lim_y0
                                y2 = int(point2[0])-lim_x0                        
                                cv2.line(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                         self.color_list[clr], 1)
                                
                    # Display the resulting tracking frame
                    cv2.imshow(' Processing ... ', orig_frame)
                    final_img_set[frameN,:,:,:]=orig_frame

                #save the file
                final_img_set=final_img_set/np.max(final_img_set)*255
                final_img_set=final_img_set.astype('uint8')
                
                if not(save_file.endswith(".tif") or save_file.endswith(".tiff")):
                    save_file += ".tif"
                    
                    if os.path.isfile(save_file)==True:
                        # add date if the file exists already
                        now = datetime.datetime.now()
                        save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]
                
                    
                imageio.volwrite(save_file, final_img_set)
                
                cv2.destroyAllWindows()
                
                print("movie location: ", save_file)
            
            except: 

                print("Something went wrong! The movie will not be saved! Try saving smaller region. ")
            
            
            
    
    def save_in_file(self):
        '''
        save corrected trajectories to json and csv files
        
        '''
        
        def save_data():
            
            try:
                cancel_window()
            except:
                pass
            
            # ask for the file location        
            save_file = tk.filedialog.asksaveasfilename()
            
            if not save_file:
                print("File name was not provided. The data was not saved. ")
                
            else: 
                # save txt file with json format            
                if not(save_file.endswith(".txt")):                
                    if save_file.endswith(".csv"):
                        save_file =save_file.split(".csv")[0]+ ".txt"
                    else:
                        save_file += ".txt"  
    
                    if os.path.isfile(save_file)==True:
                        # add date if the file exists already
                        now = datetime.datetime.now()
                        save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]
                    
                    
                # create dictionary with settings
                save_dict=self.track_data_filtered.copy()
                self.update_movie_parameters()
                param_dict={"viewer_set":{"resolution (nm/pix)": self.img_resolution, "frame rate (f/sec)":self.frame_rate, "axis orientation": self.ap_axis, "axis": self.axis_name}}
                save_dict.update(param_dict)
                
                with open(save_file, 'w') as f:
                    json.dump(save_dict, f, ensure_ascii=False) 
                    
                print("tracks are saved in json format to  ", save_file)                
                              
                # save tracks to csv file 
                # prepare csv file                
                tracks_data=[]
                
                tracks_data.append([ 'TrackID', 'x', 'y', 'frame'])   
                for trajectory in self.track_data_filtered["tracks"]:
                    new_frames=trajectory["frames"]
                    new_trace=trajectory["trace"]
                    trackID=trajectory["trackID"]
                    for pos in range(0, len(new_frames)):
                        point=new_trace[pos]
                        frame=new_frames[pos]
                        tracks_data.append([trackID, point[0], point[1],  frame])
        
        
                # save to csv 
                result_path_csv =save_file.split(".txt")[0]+ ".csv"
                    
                        
                with open(result_path_csv, 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(tracks_data)
                    csvFile.close()
    
                    
    
                print("                 in csv format to  ", result_path_csv)
                
            
        def cancel_window():
            self.save_data_window.destroy()
            
        # if zoom is on

#        xlim=self.ax.get_xlim()
#        ylim=self.ax.get_ylim()
#        print(xlim, ylim)
#        print(self.movie.shape)
#        print(xlim[0]!=0, ylim[1]!=0, xlim[1]!=self.movie.shape[2], ylim[0]!=self.movie.shape[1])
#        zoom= xlim[0]!=0 or ylim[1]!=0 or xlim[1]!=self.movie.shape[2] or ylim[0]!=self.movie.shape[1]
        print(self.is_zoom_filtered)
        if self.is_zoom_filtered==True:
          self.save_data_window = tk.Toplevel(root,  bg='white')
          self.save_data_window.title("ATTENTION !!! ")
          
          qnewtext = tk.Label(master=self.save_data_window, text="  Trajectories are filtered based on the zoomed area. \n Only tracks in this area will be included. " ,  bg='white', font=("Times", 10))
          qnewtext.grid(row=0, column=0, columnspan=2, pady=self.pad_val*2, padx=self.pad_val*2) 

        
          newbutton = tk.Button(master=self.save_data_window, text=" Continue ", command=save_data)
          newbutton.grid(row=2, column=0, columnspan=1, pady=self.pad_val*2, padx=self.pad_val*2) 
            
          deletbutton = tk.Button(master=self.save_data_window, text=" Cancel ", command=cancel_window)
          deletbutton.grid(row=2, column=1, columnspan=1, pady=self.pad_val*2, padx=self.pad_val*2)
        else:
            save_data()
            

    def move_to_previous(self):
        '''
        move back one frame
        '''
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.scale_movie.set(self.frame_pos) 
        
    def move_to_next(self):
        '''
        move forward one frame
        '''
        
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.scale_movie.set(self.frame_pos) 
        
        
    def show_tracks(self):  
        '''
        update image monitor with plots
        
        '''
               
        # update movie parameters
        self.update_movie_parameters()

        
        # read limits
        xlim_old=self.ax.get_xlim()
        ylim_old=self.ax.get_ylim()


        # plot image
        self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        
        self.ax.clear() # clean the plot 
        self.ax.imshow(self.image, cmap="gray")
        
        self.ax.axis('off')

        if  self.track_data_framed and self.monitor_switch<2:

            # plot tracks
            plot_info=self.track_data_framed['frames'][self.frame_pos]['tracks']
            for p in plot_info:
                trace=p['trace']
                self.ax.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                if self.monitor_switch==0:
                    self.ax.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])

        # plot axis
        if self.monitor_axis==1:
            # position of axis
            axis_name=self.axis_name.split(",")
            if axis_name[0]:
                first_name=axis_name[0]
            else:
                first_name=" "
            
            if axis_name[1]:
                second_name=axis_name[1]
            else:
                second_name=" "
            
            # get position and orientation
            
            image_size_y=abs(xlim_old[0]-xlim_old[1])
            image_size_x=abs(ylim_old[0]-ylim_old[1])
            
            dist=np.max((5,int(round(np.min((image_size_x, image_size_y))/10, 0))))

            
            arrow_a=[int(round(dist+np.min((ylim_old[0], ylim_old[1])),0)),int(round(dist+np.min((xlim_old[0], xlim_old[1])),0))]
            
            arrow_b=[int(round(dist*math.cos(math.radians(self.ap_axis-90))+arrow_a[0], 0)),int(round(dist*math.sin(math.radians(self.ap_axis-90))+arrow_a[1],0))]
                                

            # check that the points are not outside the view

            if arrow_b[0]>np.max((ylim_old[0], ylim_old[1]))-dist: #arrow_b[0]<int(self.image.shape[0]/10):
                # move the points
                arrow_a[0]=arrow_a[0]-dist
                arrow_b[0]=arrow_b[0]-dist

            if arrow_b[1]>np.max((xlim_old[0], xlim_old[1]))-dist: #arrow_b[1]<int(self.image.shape[1]/10):
                # move the points
                arrow_a[1]=arrow_a[1]-dist
                arrow_b[1]=arrow_b[1]-dist
                
                
            if arrow_b[0]<np.min((ylim_old[0], ylim_old[1]))+dist: #arrow_b[0]<int(self.image.shape[0]/10):
                # move the points
                arrow_a[0]=arrow_a[0]+dist
                arrow_b[0]=arrow_b[0]+dist

            if arrow_b[1]<np.min((xlim_old[0], xlim_old[1]))+dist: #arrow_b[1]<int(self.image.shape[1]/10):
                # move the points
                arrow_a[1]=arrow_a[1]+dist
                arrow_b[1]=arrow_b[1]+dist
                

#            arrow_length=[int(dist*math.cos(math.radians(self.ap_axis))),int(dist*math.sin(math.radians(self.ap_axis)))]
#            self.ax.arrow(arrow_a[0], arrow_a[1], arrow_length[0], arrow_length[1],  color='r', alpha=0.5, width=0.3, head_length=3, head_width=4, length_includes_head=True)

            self.ax.plot([arrow_a[1], arrow_b[1]], [arrow_a[0], arrow_b[0]],  color='r', alpha=0.5)
            self.ax.text(arrow_a[1]-2, arrow_a[0]-2,  second_name, color='r', size=9, alpha=0.5)
            self.ax.text(arrow_b[1]-2, arrow_b[0]-2,  first_name, color='r', size=9, alpha=0.5)

        #set the same "zoom"        
        self.ax.set_xlim(xlim_old[0],xlim_old[1])
        self.ax.set_ylim(ylim_old[0],ylim_old[1])
        
        # inver y-axis as set_ylim change the orientation
        if ylim_old[0]<ylim_old[1]:
            self.ax.invert_yaxis()



            
            
        self.canvas.draw()
        # place buttons
        
        def new_home(): # zoom
            # zoom out
        
            self.ax.set_xlim(0,self.movie.shape[2])
            self.ax.set_ylim(0,self.movie.shape[1])
            
            self.show_tracks()   

        def offclick_zoomin(event):
            
            self.ax.set_xlim(np.min((self.plot_range_coordinates[0],float(event.xdata))),np.max((self.plot_range_coordinates[0],float(event.xdata))))
            self.ax.set_ylim(np.min((self.plot_range_coordinates[1],float(event.ydata))), np.max((self.plot_range_coordinates[1],float(event.ydata))))
                
            self.show_tracks() 
            
            
        def onclick_zoomin(event):       
            self.plot_range_coordinates[0]=float(event.xdata)
            self.plot_range_coordinates[1]=float(event.ydata)            
            
        def zoom_in():
            self.canvas.mpl_connect('button_press_event', onclick_zoomin)
            self.canvas.mpl_connect('button_release_event', offclick_zoomin)           
            
        button_zoomin = tk.Button(master=self.toolbarFrame, text=" zoom in ", command=zoom_in)
        button_zoomin.grid(row=0, column=0,  columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
        button_zoomout = tk.Button(master=self.toolbarFrame, text=" zoom out ", command=new_home)
        button_zoomout.grid(row=0, column=1,  columnspan=1, pady=self.pad_val, padx=self.pad_val)

                       

    def filtering(self, position_based=True):
        '''
        filtering tracks
        
        '''
        
        self.is_zoom_filtered=False
        
        def cancel_window():
            '''
            destroy the window
            
            '''
            try:
                self.choose_traj_segmentation.destroy()
            except:
                pass
                    
        
        def run_filtering():       
            print("filtering for net length: ", self.filter_length, "filtering for total length: ", self.filter_length_total, ";   duration: ", self.filter_duration, ";   speed: ", self.filter_speed, ";   orientation: ", self.filter_orientation) #, ";   final stop duration: ", self.filter_stop)
    
            # filtering 
            self.track_data_filtered={}
            self.track_data_filtered.update({'tracks':[]})
            
            # check through the tracks
            for p in tqdm(self.track_data['tracks']):
                
                # check length
                if len(p['trace'])>1:
                    
                    # check length
                    track_duration=(p['frames'][-1]-p['frames'][0]+1)/self.frame_rate
                   
                    #check net and total distance
                    x_0=np.asarray(p['trace'])[0,0]
                    y_0=np.asarray(p['trace'])[0,1]
                    
                    x_e=np.asarray(p['trace'])[-1,0]
                    y_e=np.asarray(p['trace'])[-1,1]
                    
                    #net displacement
                    track_length=np.round(np.sqrt((x_e-x_0)**2+(y_e-y_0)**2),2)*self.img_resolution
                    
                    #total displacement
                    x_from=np.asarray(p['trace'])[0:-1,0] 
                    y_from=np.asarray(p['trace'])[0:-1,1] 
                    x_to=np.asarray(p['trace'])[1:,0] 
                    y_to=np.asarray(p['trace'])[1:,1] 
                    
                    track_length_total=np.round(np.sum(np.sqrt((x_to-x_from)**2+(y_to-y_from)**2)),2)*self.img_resolution
                    
                    # orientation
                                          
                    pointB=p['trace'][-1]                        
                    pointA=p['trace'][0]    
                    
                    y=pointB[1]-pointA[1]
                    x=pointB[0]-pointA[0]
                    net_direction=int((math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360)
                    
                    #scale to 0-180
                    
                    if net_direction>180:
                        net_direction=abs(net_direction-360)
                    
                    if self.filter_orientation[0]>180:
                        self.filter_orientation[0]=abs(self.filter_orientation[0]-360)
                        
                    if self.filter_orientation[1]>180:
                        self.filter_orientation[1]=abs(self.filter_orientation[1]-360)
                        
                    
                else:
                    track_duration=0
                    track_length=0
                    track_length_total=0
                    net_direction=0
    
                    # variables to evaluate the trackS
                length_var=track_length>=self.filter_length[0] and track_length<=self.filter_length[1]
                length_total_var=track_length_total>=self.filter_length_total[0] and track_length_total<=self.filter_length_total[1]
                duration_var=track_duration>=self.filter_duration[0] and track_duration<=self.filter_duration[1]
                orientation_var=net_direction>=np.min((self.filter_orientation[0],self.filter_orientation[1])) and net_direction<=np.max((self.filter_orientation[0],self.filter_orientation[1]))
                
                if self.txt_track_number.get()=='':
                    filterID=True 
                else:                
                    # get the list of trackIDs
                    track_list_filter=list(map(int,self.txt_track_number.get().split(",")))
                    filterID =  p['trackID'] in track_list_filter
                    
                # check position of the tracks
                self.xlim_zoom=self.ax.get_xlim()
                self.ylim_zoom=self.ax.get_ylim()
                
                # get zoom data
                if len(p['trace'])>0:
                    zz_x_0=np.asarray(p['trace'])[:,1]>=self.xlim_zoom[0]
                    zz_x_1=np.asarray(p['trace'])[:,1]<=self.xlim_zoom[1]
                    
                    zz_y_0=np.asarray(p['trace'])[:,0]<=self.ylim_zoom[0]
                    zz_y_1=np.asarray(p['trace'])[:,0]>=self.ylim_zoom[1]
                    
                    zz=zz_x_0*zz_x_1*zz_y_0*zz_y_1

                    if self.filter_zoom==0: # any point
                        
                        zoom_filter=np.any(zz==True)
                    else: # all points
                        #check all the points are inside
                        zoom_filter=np.all(zz==True)    
                else:
                    zoom_filter=True
                

                # Force zoom to be always True if Zoom filtering is not needed
                if position_based==False:
                    zoom_filter=True
                    
                #change value for zoomed filter indicator if needed
                if zoom_filter==False:
                    self.is_zoom_filtered=True
                    
                if length_var==True and length_total_var==True and duration_var==True and filterID==True and zoom_filter==True and orientation_var==True:
                                    # check speed limitation
                    if movement_1==True or movement_2==True:
                        # evaluate motion 
                        p['motion']=self.motion_type_evaluate(p, traj_segm_switch_var=self.traj_segmentation_var)
                        
                        #calculate speed
                        moving_speeds=self.tg.calculate_speed(p, "movement")
                        curvilinear_speed_move=np.round(moving_speeds[0]*self.img_resolution*self.frame_rate,0)
                    
                        speed_filter=curvilinear_speed_move>self.filter_speed[0] and curvilinear_speed_move<self.filter_speed[1]
                    else:
                        speed_filter=True
                    
                    if speed_filter==True:
                        self.track_data_filtered['tracks'].append(p)
                        
                        
            self.track_to_frame()
            
            #update the list
            self.list_update()
            
            #plot the filters
            lbl2 = tk.Label(master=self.listframework, text="filtered tracks: "+str(len(self.track_data['tracks'])-len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 12))
            lbl2.grid(row=8, column=7, columnspan=2, pady=self.pad_val, padx=self.pad_val)          
            
            cancel_window()
            
            
                
        # update movie parameters
        self.update_movie_parameters()
        
        #read variables
        
        if self.txt_duration_from.get()=='':
            self.filter_duration[0]=0
        else:
            self.filter_duration[0]=float(self.txt_duration_from.get())

        if self.txt_duration_to.get()=='':
            self.filter_duration[1]=float('Inf')
        else:
            self.filter_duration[1]=float(self.txt_duration_to.get())                        

        if self.txt_length_from.get()=='':
            self.filter_length[0]=0
        else:
            self.filter_length[0]=float(self.txt_length_from.get())

        if self.txt_length_to.get()=='':
            self.filter_length[1]=float('Inf')
        else:
            self.filter_length[1]=float(self.txt_length_to.get())  


        if self.txt_length_from_total.get()=='':
            self.filter_length_total[0]=0
        else:
            self.filter_length_total[0]=float(self.txt_length_from_total.get())

        if self.txt_length_to_total.get()=='':
            self.filter_length_total[1]=float('Inf')
        else:
            self.filter_length_total[1]=float(self.txt_length_to_total.get())  
        
        if self.txt_speed_from.get()=='':
            self.filter_speed[0]=0
            movement_1=False
        else:
            self.filter_speed[0]=float(self.txt_speed_from.get())
            movement_1=True
            
        if self.txt_speed_to.get()=='':
            self.filter_speed[1]=float('Inf')
            movement_2=False
        else:
            self.filter_speed[1]=float(self.txt_speed_to.get())                     
            movement_2=True

            
        if self.txt_orientation_from.get()=='':
            self.filter_orientation[0]=float('-Inf')
        else:
            self.filter_orientation[0]=float(self.txt_orientation_from.get())                     
            
        if self.txt_orientation_to.get()=='':
            self.filter_orientation[1]=float('Inf')
        else:
            self.filter_orientation[1]=float(self.txt_orientation_to.get())                     

            
        
        #choose trajectory segmentation type if needed and run filtering        
        if movement_1==True or movement_2==True:
                    
            # open new window
            self.choose_traj_segmentation = tk.Toplevel(root,  bg='white')
            self.choose_traj_segmentation.title(" ")
            
            #default value -> no segmentation
            self.traj_segmentation_var=1
            
            self.qnewtext = tk.Label(master=self.choose_traj_segmentation, text=" Select trajectory segmentation mode:  " ,  bg='white', font=("Times", 10))
            self.qnewtext.grid(row=0, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
            
            # radiobutton to choose
            
            var_traj_segm_switch = tk.IntVar()
            
            def update_traj_segm_switch():            
                self.traj_segmentation_var=var_traj_segm_switch.get()
                 
    
            segmentation_switch_msd = tk.Radiobutton(master=self.choose_traj_segmentation,text=" MSD based ",variable=var_traj_segm_switch, value=1, bg='white', command =update_traj_segm_switch )
            segmentation_switch_msd.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)    
            
            segmentation_switch_unet = tk.Radiobutton(master=self.choose_traj_segmentation,text=" U-Net based ", variable=var_traj_segm_switch, value=2, bg='white', command =update_traj_segm_switch )
            segmentation_switch_unet.grid(row=1, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
                
            self.newbutton = tk.Button(master=self.choose_traj_segmentation, text=" OK ", command=run_filtering, width=int(self.button_length/2),  bg='green')
            self.newbutton.grid(row=3, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.choose_traj_segmentation, text=" Cancel ", command=cancel_window, width=int(self.button_length/2))
            self.deletbutton.grid(row=3, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
        else:
            run_filtering()
            
        
    def tracklist_on_select(self, even):
        position_in_list=self.listNodes_tracks.curselection()[0]
        
        # creating a new window with class TrackViewer                
        self.new_window = tk.Toplevel(self.master)
        
         
        # create the track set with motion
        this_track=self.track_data_filtered['tracks'][position_in_list]
        motion_type=[0]*len(this_track['frames'])
        this_track['motion']=motion_type
        
        
        # update movie and ap-axis parameters
        self.update_movie_parameters()
        
  
        view_range=[self.ax.get_xlim(),self.ax.get_ylim()]

        TrackViewer(self.new_window, this_track, self.movie, 
                    self.img_resolution, self.frame_rate, self.ap_axis, self.axis_name, self.speed_sliding_window, view_range, self.track_data_framed)
            


    def detele_track_question(self):
        '''
        function for the delete track button
        '''
        # close windows if open
        self.cancel_action()
        
        # get  trackID from the list
        try:
            
            # open new window
            self.delete_window = tk.Toplevel(root ,  bg='white')
            self.delete_window.title(" Delete the track ")
            self.delete_window.geometry("+20+20")

            self.qdeletetext = tk.Label(master=self.delete_window, text="delete track "+str(self.track_data_filtered['tracks'][self.listNodes_tracks.curselection()[0]]['trackID'])+" ?",  bg='white', font=("Times", 10), width=self.button_length*2)
            self.qdeletetext.grid(row=0, column=0,  columnspan=2, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.delete_window, text=" OK ", command=self.detele_track, width=int(self.button_length/2),  bg='red')
            self.deletbutton.grid(row=1, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.delete_window, text=" Cancel ", command=self.cancel_action, width=int(self.button_length/2),  bg='green')
            self.deletbutton.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 

        except:
            self.delete_window = tk.Toplevel(root ,  bg='white')
            self.delete_window.title(" Delete the track ")
            self.delete_window.geometry("+20+20")
            self.qdeletetext = tk.Label(master=self.delete_window, text=" Track is not selected! ",  bg='white', font=("Times", 10), width=self.button_length*2)
            self.qdeletetext.grid(row=0, column=0,  columnspan=2, pady=self.pad_val, padx=self.pad_val) 

        
        
    def detele_track(self):
        '''
        delete selected track
        '''
        self.deleted_tracks_N+=1
        delete_trackID=self.track_data_filtered['tracks'][self.listNodes_tracks.curselection()[0]]['trackID']
        
        pos=0
        for p in self.track_data['tracks']:
            
            if p['trackID']==delete_trackID:
                self.track_data['tracks'].remove(p)
                
            pos+=1

        print("track ", delete_trackID, "is removed")
        
        #visualise without the track
        self.filtering()
        self.track_to_frame()
        
        #update the list
        self.list_update()
        
        #close the window
        self.cancel_action()
        
        
    def  merge_track_question(self):
        '''
        function for the "merge tracks" button
        '''
        # close windows if open
        self.cancel_action()
        
        # open new window
        self.create_window = tk.Toplevel(root ,  bg='white')
        self.create_window.title(" Merge the tracks")
        
        self.qnewtext = tk.Label(master=self.create_window, text=" Provide ID of the tracks to merge below (use ',' for separation): " ,  bg='white', font=("Times", 10))
        self.qnewtext.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
        
        self.merge_ids_text = tk.Entry(self.create_window, width=int(self.button_length*2))
        self.merge_ids_text.grid(row=1, column=0,  columnspan=2, pady=self.pad_val, padx=self.pad_val)              
        
        #define new trackID:
        for p in tqdm(self.track_data['tracks']):
            
            self.new_trackID=int(np.max((self.new_trackID, p['trackID'])))
        self.new_trackID+=1
        
        qnewtext1 = tk.Label(master=self.create_window, text=" ID of the new merged track: " ,  bg='white', font=("Times", 10))
        qnewtext1.grid(row=2, column=0, pady=self.pad_val, padx=self.pad_val) 
        
        v = tk.StringVar(root, value=str(self.new_trackID))
        self.trackID_parameter = tk.Entry(self.create_window, width=int(self.button_length/2), text=v)
        self.trackID_parameter.grid(row=2, column=2, pady=self.pad_val, padx=self.pad_val)

        self.newbutton = tk.Button(master=self.create_window, text=" OK ", command=self.merge_track, width=int(self.button_length/2),  bg='green')
        self.newbutton.grid(row=3, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        self.deletbutton = tk.Button(master=self.create_window, text=" Cancel ", command=self.cancel_action, width=int(self.button_length/2),  bg='green')
        self.deletbutton.grid(row=3, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
    def merge_track(self):
        '''
        merge tracks
        '''
        if self.merge_ids_text.get()!='':
            try:
                merge_ids=[int(idpos) for idpos in self.merge_ids_text.get().split(",")]
            except:
                 print("Some of the provided tracks are not exist!")   
        else:
            merge_ids=[]
            
        # convert IDs to positions in list
        list_of_ids=[self.track_data_filtered['tracks'][pos]["trackID"] for pos in range(0, len(self.track_data_filtered['tracks']))]
        
        try:
            pos_in_list=[np.where(np.asarray(list_of_ids)==pos)[0][0] for pos in merge_ids]
        except:
            print("Some of the provided tracks are not exist or filtered out!")
            pos_in_list=[]

        
        # define the tracks order 
        
        first_frames=[self.track_data_filtered['tracks'][pos]['frames'][0] for pos in pos_in_list]

        merging_order=np.argsort(first_frames)
        
        # merge the tracks
        
        frames=[]
        trace=[]
        
        for k in merging_order:
            i=pos_in_list[k]
            track=self.track_data_filtered['tracks'][i]
            frames=frames+track["frames"]
            trace=trace+track["trace"]
        

        if len(frames)!=0:
            
#            motion=[0]*len(trace)
            new_track={"trackID":int(self.new_trackID), "trace":trace, "frames":frames} #, "motion": motion}
            
            self.track_data['tracks'].append(new_track)
            
            #visualise without the track
            self.filtering()
            self.track_to_frame()
            
            #update the list
            self.list_update()
            
            # close the windows
            self.cancel_action()     
                
            print("Tracks ", merge_ids, " are mergerd, the new track ID is ", self.new_trackID)

    def duplicate_track_question(self):
        '''
        function for the "duplicate track" button
        
        '''
        
        # close windows if open
        self.cancel_action()
        
        # open new window
        self.create_window = tk.Toplevel(root ,  bg='white')
        self.create_window.title(" Duplicate the track")
        
        self.qnewtext = tk.Label(master=self.create_window, text="duplicate  track  "+str(self.track_data_filtered['tracks'][self.listNodes_tracks.curselection()[0]]['trackID'])+" ? new track ID: " ,  bg='white', font=("Times", 10), width=self.button_length*2)
        self.qnewtext.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
                    
        #define new trackID:
        for p in tqdm(self.track_data['tracks']):
            
            self.new_trackID=np.max((self.new_trackID, p['trackID']))
        self.new_trackID+=1
        
        v = tk.StringVar(root, value=str(self.new_trackID))
        self.trackID_parameter = tk.Entry(self.create_window, width=int(self.button_length/2), text=v)
        self.trackID_parameter.grid(row=0, column=2, pady=self.pad_val, padx=self.pad_val)

            
        self.newbutton = tk.Button(master=self.create_window, text=" OK ", command=self.duplicate_track, width=int(self.button_length/2),  bg='green')
        self.newbutton.grid(row=1, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        self.deletbutton = tk.Button(master=self.create_window, text=" Cancel ", command=self.cancel_action, width=int(self.button_length/2),  bg='green')
        self.deletbutton.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
        
        
    def new_track_question(self):
        '''
        function for the new track button
        '''
        # close windows if open
        self.cancel_action()
        
        
        # open new window
        self.create_window = tk.Toplevel(root ,  bg='white')
        self.create_window.title(" Create new track")
        self.create_window.geometry("+20+20")
        
        self.qnewtext = tk.Label(master=self.create_window, text="create new track ?  track ID: " ,  bg='white', font=("Times", 10), width=self.button_length*2)
        self.qnewtext.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
        
                    
        #define new trackID:
        for p in tqdm(self.track_data['tracks']):
            
            self.new_trackID=np.max((self.new_trackID, p['trackID']))
        self.new_trackID+=1
        
        v = tk.StringVar(root, value=str(self.new_trackID))
        self.trackID_parameter = tk.Entry(self.create_window, width=int(self.button_length/2), text=v)
        self.trackID_parameter.grid(row=0, column=2, pady=self.pad_val, padx=self.pad_val)

            
        self.newbutton = tk.Button(master=self.create_window, text=" OK ", command=self.create_track, width=int(self.button_length/2),  bg='green')
        self.newbutton.grid(row=1, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        self.deletbutton = tk.Button(master=self.create_window, text=" Cancel ", command=self.cancel_action, width=int(self.button_length/2),  bg='green')
        self.deletbutton.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)

    def cancel_action(self):
        '''
        destroy the windows
        '''
        
        try:
            self.create_window.destroy()
        except: 
            pass
        
        try:
            self.delete_window.destroy()        
        except: 
            pass
        
        
    def duplicate_track(self):
        '''
        duplicate a track
        '''
        
        # read ap axis
        if self.trackID_parameter.get()!='':
            self.new_trackID=int(self.trackID_parameter.get())
            
        # update counting of the new tracks
        self.created_tracks_N+=1
        duplicate_trackID=self.track_data_filtered['tracks'][self.listNodes_tracks.curselection()[0]]['trackID']
        
        for p in self.track_data['tracks']:
            
            if p['trackID']==duplicate_trackID:
                duplicated_track=copy.deepcopy(p)
            
        
        new_track={"trackID":self.new_trackID, "trace":duplicated_track['trace'], "frames":duplicated_track['frames']}
        
        self.track_data['tracks'].append(new_track)
        
        print(" track is duplicated with new trackID ", self.new_trackID)
        
        #visualise without the track
        self.filtering()
        self.track_to_frame()
        
        #update the list
        self.list_update()
        
        # close the windows
        self.cancel_action()        
        
        
    def create_track(self):
        '''
        create a track
        '''
        
        #read ap axis
        if self.trackID_parameter.get()!='':
            print(self.trackID_parameter.get())
            self.new_trackID=int(self.trackID_parameter.get())
            
        self.created_tracks_N+=1
        
        p={"trackID":self. new_trackID, "trace":[], "frames":[]}
        
        self.track_data['tracks'].append(p)
        
        print("new track ", self.new_trackID, "is created")
        
        #visualise without the track
        self.filtering()
        self.track_to_frame()
        
        #update the list
        self.list_update()
        
        # close the windows
        self.cancel_action()   

        # open the windows 
        
        
        # creating a new window with class TrackViewer                
        self.new_window = tk.Toplevel(self.master)
        
         
        # create the track set with motion
        this_track=self.track_data['tracks'][-1]
        this_track['motion']=[]
        
        
        # update movie and ap-axis parameters
        self.update_movie_parameters()
        
  
        view_range=[self.ax.get_xlim(),self.ax.get_ylim()]

        TrackViewer(self.new_window, this_track, self.movie, 
                    self.img_resolution, self.frame_rate, self.ap_axis, self.axis_name, self.speed_sliding_window, view_range, self.track_data_framed)
             

    def list_update(self):
        '''
        update track list
        '''
                
        # update movie parameters
        self.update_movie_parameters()
        
        # show track statistics
        try:
            self.text_deleted_tracks.destroy()
            self.text_filtered_tracks.destroy()
            self.text_total_tracks.destroy()
        except:
            pass
        
        
        self.text_total_tracks = tk.Label(master=self.listframework, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), width=int(self.button_length*1.5), bg='white',  font=("Times", 14, "bold"))
        self.text_total_tracks.grid(row=7, column=5, columnspan=4, pady=self.pad_val, padx=self.pad_val)        

            
        self.text_deleted_tracks = tk.Label(master=self.listframework, text="deleted tracks: "+str(self.deleted_tracks_N), width=int(self.button_length*1.5), bg='white',  font=("Times", 12,))
        self.text_deleted_tracks.grid(row=8, column=5, columnspan=2, pady=self.pad_val, padx=self.pad_val)         

        self.text_filtered_tracks = tk.Label(master=self.listframework, text="filtered tracks: "+str(len(self.track_data['tracks'])-len(self.track_data_filtered['tracks'])), width=int(self.button_length*1.5), bg='white',  font=("Times", 12))
        self.text_filtered_tracks.grid(row=8, column=7, columnspan=2, pady=self.pad_val, padx=self.pad_val)          
        
        

       # plot the tracks from filtered folder 
        try:
           self.listNodes_tracks.delete(0,tk.END)
        except:
            pass
                
        
        for p in self.track_data_filtered['tracks']:
            
            #calculate length and duration
            if len(p['trace'])>0:
                start_track_frame=p['frames'][0]
            else:
                start_track_frame=0
            
            # add to the list
            self.listNodes_tracks.insert(tk.END, "ID: "+str(p['trackID'])+" start frame: "+str(start_track_frame))        

            
            
    def select_vesicle_movie(self):
        '''
        function for select particle movie button
        '''
        
        filename = tk.filedialog.askopenfilename()
        if not filename:
            print("File was not selected")
        else:  
            self.movie_file=filename
            root.update()
            
            # read files 
            self.movie=skimage.io.imread(self.movie_file)
            self.movie_length=self.movie.shape[0]  
            try:
                self.lbl1.destroy()
            except:
                pass
                
            self.lbl1 = tk.Label(master=root, text="movie: "+self.movie_file.split("/")[-1], bg='white')
            self.lbl1.grid(row=2, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
             
            
            # set axes
            #set the same "zoom"
            self.ax.viewLim.x0=0
            self.ax.viewLim.x1=self.movie.shape[2] 
            self.ax.viewLim.y0=0
            self.ax.viewLim.y1=self.movie.shape[1]         
            
            # plot image
            self.show_tracks()    
            
            # next and previous buttons
            def show_values(v):
                self.frame_pos=int(v)
                self.show_tracks() 
                  
                
            self.scale_movie = tk.Scale(self.viewFrametool, from_=0, to=self.movie_length-1, tickinterval=100, length=self.figsize_value[1]*self.dpi, width=10, orient="horizontal", command=show_values)
            self.scale_movie.set(0)        
            self.scale_movie.grid(row=0, column=1,  sticky=tk.W)
            
            # rearrange the tracks in order of the frames
            try:
                self.track_to_frame()
            except:
                pass
            
    
    def select_track(self):
        '''
        load data with tracks from file
        
        '''
        
        global folder_path_output  
        filename = tk.filedialog.askopenfilename()
        
        if not filename:
            print("File was not selected")
        else:  
            self.track_file=filename
            
            # reset counts of deleted tracks
            self.deleted_tracks_N=0
            
            
            if self.track_file.endswith(".csv"):# read csv 
                # read file
                
                json_tracks={"tracks":[]}
                trackID_new=-1
                track={}
                with open(self.track_file, newline='') as f:
                    reader = csv.reader(f)

                    for row in reader:

                        if row:  # if the row is not empty
                            if row[0]!="TrackID" and row[0]:
                                trackID=int(row[0])
                                point=[float(row[1]), float(row[2])]
                                frame=int(row[3])
                                
                                if trackID!=trackID_new: # new track
                                    #save the previous track
                                    if bool(track)==True :
                                        json_tracks["tracks"].append(track)
                                    
                                    trackID_new=trackID
                                    track={"trackID":trackID, "trace":[point], "frames": [frame]}
                                else: # update the existing track
                                    track["trace"].append(point)
                                    track["frames"].append(frame)
                                    
                                    
                    #save the last track
                    if bool(track)==True :
                        json_tracks["tracks"].append(track)                        

                    self.track_data_original=json_tracks
    
                
            else: # read json in txt 
                       
                #read  the tracks data 
                with open(self.track_file) as json_file:  
        
                    self.track_data_original = json.load(json_file)
                    
                    
                # to save from dictionary to dict-list format:
                if 'tracks' not in self.track_data_original.keys(): # format exported from the tracker
                    
                    
                    self.track_data={'tracks':[]}
                    
                    for pos in self.track_data_original:
                        p=self.track_data_original[pos]
                        self.track_data['tracks'].append(p)
                        
                    self.track_data_original=self.track_data
                    
                else:  # format exported from the viewer, can include viewer settings
                    
                    try: # try to read viewer settings
                        print("file contains viewer settings, they will be updated")
                        read_parameters = self.track_data_original["viewer_set"]
                        print(read_parameters)
                        
                        try:
                            self.res_parameter.destroy()
                            self.frame_parameter.destroy()
                        except:
                            pass
                            
                        
                        # img resolution
                        self.img_resolution=read_parameters["resolution (nm/pix)"]
                        v = tk.StringVar(root, value=str(self.img_resolution))
                        self.res_parameter = tk.Entry(root, width=10, text=v)
                        self.res_parameter.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val)      
                        
                        # frame rate
                        self.frame_rate=read_parameters["frame rate (f/sec)"]
                        v = tk.StringVar(root, value=str(self.frame_rate))
                        self.frame_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
                        self.frame_parameter.grid(row=5, column=3, pady=self.pad_val, padx=self.pad_val)  

                        # axis orientation
                        self.ap_axis=read_parameters["axis orientation"]
                        v = tk.StringVar(root, value=str(self.ap_axis))
                        self.ap_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
                        self.ap_parameter.grid(row=6, column=1, pady=self.pad_val, padx=self.pad_val)
                        
                        # axis names
                        self.axis_name=read_parameters["axis"]        
                        v = tk.StringVar(root, value=str(self.axis_name))
                        self.axis_name_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
                        self.axis_name_parameter.grid(row=6, column=3, pady=self.pad_val, padx=self.pad_val)  
           
                        
                    except:
                        pass
                            
            #continue as it was before
            self.track_data=copy.deepcopy(self.track_data_original)
            self.track_data_filtered=self.track_data 
            self.track_to_frame()
                
            self.list_update()      
            
        
    def motion_type_evaluate(self, track_data_original, traj_segm_switch_var=2):
        '''
        provide motion type evaluation to select directed movement for speed evaluation
        
        '''
        
        if traj_segm_switch_var==0: # no segmentation required
            motion_type=[0] * len(track_data_original['frames'])
            
        elif  traj_segm_switch_var==1: # MSD based segmentation
            # set trajectory length
            self.tg.window_length=10
            # run segmentation
            segmentation_result=self.tg.msd_based_segmentation(track_data_original['trace'])
            motion_type=segmentation_result[:len(track_data_original['frames'])]
            
        else: # U-Net based segmentation
            # set trajectory length
            self.tg.window_length=8
            # run segmentation
            segmentation_result=self.tg.unet_segmentation(track_data_original['trace'])
            motion_type=segmentation_result[:len(track_data_original['frames'])]
        
        return motion_type
    
    def track_to_frame(self):
        '''
        recalculate loaded tracks to frame-based list for visualisation
        
        '''
        
        # change data arrangment from tracks to frames
        self.track_data_framed={}
        self.track_data_framed.update({'frames':[]})
        
        for n_frame in range(0,self.movie_length):
            
            frame_dict={}
            frame_dict.update({'frame': n_frame})
            frame_dict.update({'tracks': []})
            
            #rearrange the data
            for p in self.track_data_filtered['tracks']:
                if n_frame in p['frames']: # if the frame is in the track
                    frame_index=p['frames'].index(n_frame) # find position in the track
                    
                    new_trace=p['trace'][0:frame_index+1] # copy all the traces before the frame
                    frame_dict['tracks'].append({'trackID': p['trackID'], 'trace': new_trace}) # add to the list
                    
                    
            self.track_data_framed['frames'].append(frame_dict) # add the dictionary

    
    def save_data_csv(self):
        '''
        save csv file with parameters of all the trajectories
        '''
        
        def cancel_window():
            '''
            destroy the window
            
            '''
            try:
                self.choose_traj_segmentation.destroy()
            except:
                pass
                    
        
        def run_main_parameters_set():
            '''
            the main code run after OK button
            
            '''
            
            # read the speed interval
            
            self.speed_sliding_window = float(self.speed_window_position.get())
            
            # destroy the question window 
            cancel_window()
            
            
            
            # select file location and name
            save_file = tk.filedialog.asksaveasfilename()
            
            if not save_file:
                print("File name was not provided. The data was not saved. ")
                
            else:         
                
                # create the data for saving
                self.stat_data=[]
                self.stat_data.append(['','', 'settings: ', str(self.img_resolution)+' nm/pix', str(self.frame_rate)+' fps',' ',' ',' ',' ',' ' ]) 
                self.stat_data.append(['Filters: ','', '', '','','','','','','','','','' ]) 
                self.stat_data.append(['','', 'TrackID: ', self.txt_track_number.get(),'','','','','','','','','' ]) 
                self.stat_data.append(['','', ' Duration (sec): ', self.txt_duration_from.get(),' - ',self.txt_duration_to.get(),'','','','','','','' ]) 
                self.stat_data.append(['','', 'Net travelled distance (nm): ', self.txt_length_from.get(),' - ',self.txt_length_to.get(),'','','','','','','' ]) 
                self.stat_data.append(['','', 'Total travelled distance (nm): ', self.txt_length_from_total.get(),' - ',self.txt_length_to_total.get(),'','','','','','','' ]) 
                self.stat_data.append(['','','Mean curvilinear speed: moving (nm/sec): ', self.txt_speed_from.get(),' - ',self.txt_speed_to.get(),'','','','','','','' ]) 
                self.stat_data.append(['','','Zoom : ', 'x :',str(self.ylim_zoom),'  y:', str(self.xlim_zoom),'','','','','','' ]) 
                self.stat_data.append(['','',' ', '','','','','','','','','','' ]) 
                self.stat_data.append(['Track ID', 'Start frame', ' Total distance travelled (nm)',  'Net distance travelled (nm)', 
                                 ' Maximum distance travelled (nm)', ' Total trajectory time (sec)',  
                                 ' Net orientation (degree)', 'Mean curvilinear speed: average (nm/sec)', 'Mean straight-line speed: average (nm/sec)',
                                 'Mean curvilinear speed: moving (nm/sec)', 'Mean straight-line speed: moving (nm/sec)', 'Max curvilinear speed per segment: moving (nm/sec)', "Mean brightness (normalised [0,1]])", "Mean brightness (normalised by max)"  ]) 
        
        
                print("Total number of tracks to process: ", len(self.track_data_filtered['tracks']))
                for trackID in range(0, len(self.track_data_filtered['tracks'])):
                    print(" track ", trackID+1)
                    track=self.track_data_filtered['tracks'][trackID]
                    trajectory=track['trace']
                    if len(track['trace'])>0:
                        x=np.asarray(trajectory)[:,0]    
                        y=np.asarray(trajectory)[:,1]
                        x_0=np.asarray(trajectory)[0,0]
                        y_0=np.asarray(trajectory)[0,1]
                        
                        x_e=np.asarray(trajectory)[-1,0]
                        y_e=np.asarray(trajectory)[-1,1]
                        
                        displacement_array=np.sqrt((x-x_0)**2+(y-y_0)**2)*self.img_resolution

                        # max displacement
                        max_displacement=np.round(np.max(displacement_array),2)
                        
                        # displacement from start to the end
                        net_displacement=np.round(np.sqrt((x_e-x_0)**2+(y_e-y_0)**2),2)*self.img_resolution
                        
                        # total displacement
                        x_from=np.asarray(trajectory)[0:-1,0] 
                        y_from=np.asarray(trajectory)[0:-1,1] 
                        x_to=np.asarray(trajectory)[1:,0] 
                        y_to=np.asarray(trajectory)[1:,1] 
                        
                        total_displacement=np.round(np.sum(np.sqrt((x_to-x_from)**2+(y_to-y_from)**2)),2)*self.img_resolution
                        
                        #orientation
                        pointB=trajectory[-1]                        
                        pointA=trajectory[0]    
                        
                        y=pointB[1]-pointA[1]
                        x=pointB[0]-pointA[0]
                        net_direction=int((math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360)           
                        
                        # frames        
                        time=(track['frames'][-1]-track['frames'][0])/self.frame_rate
                        
                        # speed 
                                
                        #evaluate motion 
                        track['motion']=self.motion_type_evaluate(track, traj_segm_switch_var=self.traj_segmentation_var)
                        average_speeds=self.tg.calculate_speed(track, "average")
                        average_mcs=np.round(average_speeds[0]*self.img_resolution*self.frame_rate,0)
                        average_msls=np.round(average_speeds[1]*self.img_resolution*self.frame_rate,0)
                                             
                        moving_speeds=self.tg.calculate_speed(track, "movement")
                        try:
                            moving_mcs=np.round(moving_speeds[0]*self.img_resolution*self.frame_rate,0)
                            moving_msls=np.round(moving_speeds[1]*self.img_resolution*self.frame_rate,0)
                        except:
                            moving_mcs=None
                            moving_msls=None
                            
                            
           #             moving_maxcs=np.round(moving_speeds[2]*self.img_resolution*self.frame_rate,0)
           
                                         
                        max_curvilinear_segment=self.tg.max_speed_segment(track, int(self.speed_sliding_window*self.frame_rate))["speed"]
                        try:
                            moving_maxsegcs=np.round(max_curvilinear_segment*self.img_resolution*self.frame_rate,0)
                        except:
                            moving_maxsegcs=None

                        try:
                            _, _, intensity_mean_segment, intensity_mean_roi,_=SupportFunctions.intensity_calculation(self, self.movie, track['trace'], track['frames'])
                            intensity_mean_segment=np.round(intensity_mean_segment,5)
                            intensity_mean_roi=np.round(intensity_mean_roi,5)
                        except:
                            intensity_mean_segment=None
                            intensity_mean_roi=None

#                        moving_maxsegcs=np.round(moving_speeds[3]*self.img_resolution*self.frame_rate,0)
                        
                        
                        
                        self.stat_data.append([track['trackID'], track['frames'][0], total_displacement ,net_displacement,
                                                 max_displacement, time,
                                                 net_direction, average_mcs, average_msls, moving_mcs, moving_msls, moving_maxsegcs, intensity_mean_segment, intensity_mean_roi])
                    else:
                        self.stat_data.append([track['trackID'], None, None ,None,None, None, None, None, None, None, None, None, None, None, None])
                        
        
                if not(save_file.endswith(".csv")):
                    save_file += ".csv"
                    
                    if os.path.isfile(save_file)==True:
                        # add date if the file exists already
                        now = datetime.datetime.now()
                        save_file=save_file.split(".")[0]+"("+str(now.day)+"-"+str(now.month)+"_"+str(now.hour)+"-"+str(now.minute)+")"+"."+save_file.split(".")[-1]

        
                with open(save_file, 'w') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerows(self.stat_data)
        
                csvFile.close()
                 
                print("csv file has been saved to ", save_file)    
                
                
                
                
        # update movie parameters
        self.update_movie_parameters()
        
        # ask what trajectory segmentation to use
                
        # open new window
        self.choose_traj_segmentation = tk.Toplevel(root,  bg='white')
        self.choose_traj_segmentation.title(" ")
        
        #default value -> no segmentation
        self.traj_segmentation_var=0
        
        self.qnewtext = tk.Label(master=self.choose_traj_segmentation, text=" Select trajectory segmentation mode:  " ,  bg='white', font=("Times", 10))
        self.qnewtext.grid(row=0, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
        
        # radiobutton to choose
        
        var_traj_segm_switch = tk.IntVar()
        
        def update_traj_segm_switch():            
            self.traj_segmentation_var=var_traj_segm_switch.get()
            
     
        segmentation_switch_off = tk.Radiobutton(master=self.choose_traj_segmentation,text=" without ",variable=var_traj_segm_switch, value=0, bg='white', command =update_traj_segm_switch )
        segmentation_switch_off.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)   

        segmentation_switch_msd = tk.Radiobutton(master=self.choose_traj_segmentation,text=" MSD based ",variable=var_traj_segm_switch, value=1, bg='white', command =update_traj_segm_switch )
        segmentation_switch_msd.grid(row=1, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)    
        
        segmentation_switch_unet = tk.Radiobutton(master=self.choose_traj_segmentation,text=" U-Net based ", variable=var_traj_segm_switch, value=2, bg='white', command =update_traj_segm_switch )
        segmentation_switch_unet.grid(row=1, column=3, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
            
        
        self.lbpose1 = tk.Label(master=self.choose_traj_segmentation, text="segment speed evalution \n time interval, sec", bg='white')
        self.lbpose1.grid(row=2, column=1,  columnspan=3, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        v_speed = tk.StringVar(root, value=str(self.speed_sliding_window))
        self.speed_window_position = tk.Entry(self.choose_traj_segmentation, width=int(self.button_length/3), text=v_speed)
        self.speed_window_position.grid(row=2, column=3, pady=self.pad_val, padx=self.pad_val)   
        
        
            
        self.newbutton = tk.Button(master=self.choose_traj_segmentation, text=" OK ", command=run_main_parameters_set, width=int(self.button_length/2),  bg='green')
        self.newbutton.grid(row=3, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        self.deletbutton = tk.Button(master=self.choose_traj_segmentation, text=" Cancel ", command=cancel_window, width=int(self.button_length/2))
        self.deletbutton.grid(row=3, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
  
                
        
############################################################

class TrackViewer(tk.Frame):
    '''
    class for the window of the individual tracks 
    '''
    def __init__(self, master, track_data, movie,img_resolution, frame_rate, ap_axis, axis_name, speed_sliding_window, frame_zoom=None, track_data_framed=[]):
        tk.Frame.__init__(self, master)

        master.configure(background='white')
        
        self.viewer = master
        self.frame_zoom=frame_zoom
        
        #set the window size        
        self.window_width = int(master.winfo_screenwidth()/2.5) # of the monitor width
        self.window_height = int(master.winfo_screenheight()*0.7)  # of the monitor height

        
        # save important data
        self.track_data_framed=track_data_framed
        self.track_data=track_data
        self.movie=movie
        self.frames=track_data['frames']
        self.motion=track_data['motion']
        self.trace=track_data['trace']
        self.id=track_data['trackID']
        if len(track_data['frames'])>0:
            self.frame_pos=track_data['frames'][0]
        else:
            self.frame_pos=0
    
        self.frame_pos_to_change=() # frame which can be changed
        self.movie_length=self.movie.shape[0] # movie length
        self.plot_switch=0 # switch between plotting/not plotting tracks
        self.img_resolution=img_resolution # resolution of the movie
        self.frame_rate=frame_rate # movie frame rate
        self.ap_axis=ap_axis
        self.axis_name=axis_name
        self.img_range=[[0,0],[0,0]] # range of image ccordinates used for plotting [[x_min, x_max],[y_min, y_max]]
        
        
        # segmentation 
        self.traj_segm_switch_var=0 # calculate and show motion type
        self.speed_graph_var=0 # show curvilinear speed in colours
        self.speed_sliding_window=speed_sliding_window # sliding window interval for the speed evaluation
        self.tg = TrajectorySegment()     
        self.tg.window_length=8            
        #update motion information
        self.motion=self.motion_type_evaluate(self.track_data)
        self.track_data['motion']=self.motion 
        
        self.pixN_basic=100 # margin size 
        self.vesicle_patch_size=16
        
        self.traj_segm_switch_var=0 # calculate and show motion type
        
        #track evaluation 
        self.displacement_array=[]
        self.max_displacement=0
        self.net_displacement=0
        self.total_distance=0
        
        #colours for plotting tracks        
        self.color_list_plot=["#00FFFF", "#7FFFD4", "#0000FF", "#8A2BE2", "#7FFF00", "#D2691E", "#FF7F50", "#DC143C",
            "#008B8B", "#8B008B", "#FF8C00", "#E9967A", "#FF1493", "#9400D3", "#FF00FF", "#B22222",
            "#FFD700", "#ADFF2F", "#FF69B4", "#ADD8E6", "#F08080", "#90EE90", "#20B2AA", "#C71585", "#FF00FF"]
        
        # change the name to add track ID
        master.title("TrackViewer: track ID "+str(self.id))
        
        
        
        # placing sizes
        self.button_length=np.max((10,int(self.window_width/70)))
        self.pad_val=2
        self.dpi=100
        self.img_width=int(self.window_height*0.7)
        self.figsize_value=(int(self.window_height*0.8/self.dpi), int(self.window_height*0.4/self.dpi))

    
    ############### lists ##############################
    
        # with parameters 
        
        self.listNodes_parameters = tk.Listbox(master=self.viewer, width=int(self.button_length*6),  font=("Times", 10), selectmode='single')
        self.listNodes_parameters.grid(row=6, column=1,  columnspan=4, sticky=tk.N+tk.S, pady=self.pad_val, padx=self.pad_val)
        
        # with detections        
            
        def tracklist_on_select(even):
            try:
                
                self.frame_pos_to_change=self.listNodes.curselection()
            except:
                pass

                # show the list of data with scroll bar
        self.detection_lbend = tk.Label(master=self.viewer, text="LIST OF DETECTIONS:  ",  bg='white', font=("Times", 12))
        self.detection_lbend.grid(row=1, column=5, columnspan=3, pady=self.pad_val, padx=self.pad_val)
        
        self.scrollbar = tk.Scrollbar(master=self.viewer, orient="vertical")
        self.scrollbar.grid(row=2, column=8, rowspan=5,  sticky=tk.N+tk.S)
        
        self.listNodes = tk.Listbox(master=self.viewer, width=int(self.button_length*3), font=("Times", 10), selectmode='multiple')
        self.listNodes.grid(row=2, column=5, columnspan=3, rowspan=5 , sticky=tk.N+tk.S, pady=self.pad_val)
        self.listNodes.config(yscrollcommand=self.scrollbar.set)
        self.listNodes.bind('<<ListboxSelect>>', tracklist_on_select)
        self.scrollbar.config(command=self.listNodes.yview)

        
     # # #  build layout of the frame
        self.show_list()   
        
        # movie control
        
        
        self.fig_indwin, self.ax_indwin= plt.subplots(1,1, figsize=(int(self.img_width/self.dpi), int(self.img_width/self.dpi)))
        self.ax_indwin.axis('off')
        self.fig_indwin.tight_layout()
        
        def update_position_text(x,y): 
            # to add
            try:
                self.txt_position_coordinates.destroy()
                v = tk.StringVar(root, value=str(round(x,4))+","+str(round(y,4)))
                # add coordinates
                self.txt_position_coordinates = tk.Entry(self.add_position_window, width=int(self.button_length*2) , textvariable=v)
                self.txt_position_coordinates.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val) 
                
                # add frame
                self.txt_frame.destroy()
                v_frame = tk.StringVar(root, value=str(self.frame_pos))          
                self.txt_frame = tk.Entry(self.add_position_window, width=int(self.button_length), textvariable=v_frame)
                self.txt_frame.grid(row=0, column=11)                 
#                print(" coordinates: " , str(x), str(y), ";  frame: ", str(self.frame_pos))
                
            except:
                pass
            
           # to correct
            try:
                self.txt_position.destroy()
                v = tk.StringVar(root, value=str(round(x,4))+","+str(round(y,4)))
                self.txt_position = tk.Entry(self.correct_position_window, width=int(self.button_length*2) , textvariable=v)
                self.txt_position.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val) 
#                print(" coordinates: " , str(x), str(y))
                                
            except:
                pass            
        def onclick(event):
            
            update_position_text(float(event.ydata)+self.img_range[0][0], float(event.xdata)+self.img_range[1][0])
        
        # DrawingArea
        self.canvas_indwin = FigureCanvasTkAgg(self.fig_indwin, master=self.viewer)
        self.canvas_indwin.draw()
        self.canvas_indwin.get_tk_widget().grid(row=4, column=1, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        self.canvas_indwin.mpl_connect('button_press_event', onclick)        
        
        
        
        self.plot_image()
        
        # plot displacement
        self.fig_displacment, self.ax_displacement = plt.subplots(1, 1, figsize=self.figsize_value)  
        
        normi = matplotlib.colors.Normalize(vmin=0, vmax=2000);
        self.cbar=self.fig_displacment.colorbar(cm.ScalarMappable(norm=normi, cmap='rainbow') )
        self.cbar.ax.set_ylabel(" for curvilinear speed, nm/sec")
        
                 
        self.canvas_displacement = FigureCanvasTkAgg(self.fig_displacment, master=self.viewer)
        self.canvas_displacement.get_tk_widget().grid(row=3, column=9, columnspan=4, rowspan=2, pady=self.pad_val, padx=self.pad_val)   

        self.plot_displacement()
        
        # plot intensity graph    
        self.fig_intensity, self.ax_intensity = plt.subplots(1,1, figsize=self.figsize_value)        
        self.canvas_intensity = FigureCanvasTkAgg(self.fig_intensity, master=self.viewer)            
        self.canvas_intensity.get_tk_widget().grid(row=5, column=9, columnspan=4, rowspan=3, pady=self.pad_val, padx=self.pad_val)  
             
        
        self.intensity_calculation()
        
        # plot parameters
        self.show_parameters()
        

        # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.plot_image() 
                   
        self.scale_movie = tk.Scale(master=self.viewer, from_=0, to=self.movie_length-1, tickinterval=100, length=int(self.img_width), width=5, orient="horizontal", command=show_values)
        self.scale_movie.set(self.frame_pos)        
        self.scale_movie.grid(row=5, column=2, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(master=self.viewer, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 
        
        buttonnext = tk.Button(master=self.viewer, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=5, column=4, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)              
        
     # buttons to change the position
     
        buttonnext = tk.Button(master=self.viewer,text="change", command=self.change_position, width=int(self.button_length/2))
        buttonnext.grid(row=0, column=5, pady=self.pad_val, padx=self.pad_val)     

        buttonnext = tk.Button(master=self.viewer,text="delete", command=self.delete_position, width=int(self.button_length/2))
        buttonnext.grid(row=0, column=6, pady=self.pad_val, padx=self.pad_val)     
        
        buttonnext = tk.Button(master=self.viewer,text="add", command=self.add_position, width=int(self.button_length/2))
        buttonnext.grid(row=0, column=7, pady=self.pad_val, padx=self.pad_val)    
        
        
     # button segmentation choice
        self.lbftracjsegm = tk.Label(master=self.viewer, text=" Trajectory segmentation: ",  bg='white')
        self.lbftracjsegm.grid(row=0, column=9, columnspan=3, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

        var_traj_segm_switch = tk.IntVar()
        speed_show_switch = tk.IntVar()
        
        def update_segment_segm_switch():            
            
            self.speed_graph_var = speed_show_switch.get()
            self.speed_sliding_window = float(self.speed_window_position.get())  # in sec
                       
            # change image
            self.plot_image()
            
            # change plot
            self.plot_displacement()
            
            # change parameters
            self.show_parameters()
            
        def update_traj_segm_switch():            
            
            self.traj_segm_switch_var=var_traj_segm_switch.get()
            self.speed_graph_var = speed_show_switch.get()
            self.speed_sliding_window = float(self.speed_window_position.get())  # in sec
            
            #update motion information
            self.motion=self.motion_type_evaluate(self.track_data)
            self.track_data['motion']=self.motion            
            # change image
            self.plot_image()
            
            # change plot
            self.plot_displacement()
            
            # change parameters
            self.show_parameters()
            
     
        self.segmentation_switch_off = tk.Radiobutton(master=self.viewer,text="without",variable=var_traj_segm_switch, value=0, bg='white', command =update_traj_segm_switch )
        self.segmentation_switch_off.grid(row=1, column=9, pady=self.pad_val, padx=self.pad_val)     

        self.segmentation_switch_msd = tk.Radiobutton(master=self.viewer,text="MSD based",variable=var_traj_segm_switch, value=1, bg='white', command =update_traj_segm_switch )
        self.segmentation_switch_msd.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val)     
        
        self.segmentation_switch_unet = tk.Radiobutton(master=self.viewer,text="U-Net based", variable=var_traj_segm_switch, value=2, bg='white', command =update_traj_segm_switch )
        self.segmentation_switch_unet.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)  
     
        self.lbftracjsegm = tk.Label(master=self.viewer, text="Visualise curvilinear speed for movement: ",  bg='white')
        self.lbftracjsegm.grid(row=2, column=9, columnspan=3, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        
        self.segmentation_switch_off = tk.Radiobutton(master=self.viewer,text=" off ",variable=speed_show_switch, value=0, bg='white', command =update_traj_segm_switch )
        self.segmentation_switch_off.grid(row=3, column=9, pady=self.pad_val, padx=self.pad_val)     

        self.segmentation_switch_msd = tk.Radiobutton(master=self.viewer,text=" on ",variable=speed_show_switch, value=1, bg='white', command =update_traj_segm_switch )
        self.segmentation_switch_msd.grid(row=3, column=10, pady=self.pad_val, padx=self.pad_val)    
        
        self.lbpose1 = tk.Label(master=self.viewer, text="segment speed evalution \n time interval, sec", bg='white')
        self.lbpose1.grid(row=3, column=11, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)  
        
        v_speed = tk.StringVar(root, value=str(self.speed_sliding_window))
        self.speed_window_position = tk.Entry(self.viewer, width=int(self.button_length/3), text=v_speed)
        self.speed_window_position.grid(row=3, column=12, pady=self.pad_val, padx=self.pad_val)   
         
                
          

    # # # # # #  Radiobutton : tracks on/off/motion # # # # # # #          
    # plotting switch 
        var = tk.IntVar()
        def update_monitor_plot():            
            self.plot_switch=var.get()
            self.plot_image()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(master=self.viewer, text=" tracks on  ", variable=var, value=0, bg='white', command =update_monitor_plot )
        self.R1.grid(row=1, column=2, columnspan=1,  pady=self.pad_val, padx=self.pad_val)  

        self.R2 = tk.Radiobutton(master=self.viewer, text=" tracks off ", variable=var, value=1, bg='white',command = update_monitor_plot ) #  command=sel)
        self.R2.grid(row=1, column=3, columnspan=1,  pady=self.pad_val, padx=self.pad_val)
        
        self.R3 = tk.Radiobutton(master=self.viewer, text=" motion type ", variable=var, value=2, bg='white',command = update_monitor_plot ) #  command=sel)
        self.R3.grid(row=2, column=2, columnspan=1,  pady=self.pad_val*2, padx=self.pad_val)
        
        self.R3 = tk.Radiobutton(master=self.viewer, text=" all tracks ", variable=var, value=5, bg='white',command = update_monitor_plot ) #  command=sel)
        self.R3.grid(row=2, column=3, columnspan=1,  pady=self.pad_val*2, padx=self.pad_val)

        
        
    
    def calculate_direction(self, trace):
        '''
        calculate average angle of the direction
        '''
        pointB=trace[-1]                        
        pointA=trace[0]
        changeInX = pointB[0] - pointA[0]
        changeInY = pointB[1] - pointA[1]
        
        return int((math.degrees(math.atan2(changeInY,changeInX))+360-90-self.ap_axis)%360)
#        return int(abs(math.degrees(math.atan2(changeInX,changeInY))-self.ap_axis)%360) 

        
    def change_position(self):
        '''
        correct position coordinate values
        
        '''
        
        self.action_cancel()
        
        self.correct_position_window = tk.Toplevel(root, bg='white')
        self.correct_position_window.title(" Correct coordinates ")
        self.correct_position_window.geometry("+20+20")

        self.lbframechange = tk.Label(master=self.correct_position_window, text="Make changes in frame: "+str(self.frames[self.frame_pos_to_change[0]]), bg='white')
        self.lbframechange.grid(row=0, column=10, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

        self.lbpose = tk.Label(master=self.correct_position_window, text=" x, y ", bg='white')
        self.lbpose.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)  
        
        self.txt_position = tk.Entry(self.correct_position_window, width=int(self.button_length*2))
        self.txt_position.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)                
        

        self.buttonOK= tk.Button(master=self.correct_position_window,text=" apply ", command=self.action_apply_change, width=int(self.button_length/2))
        self.buttonOK.grid(row=2, column=10, pady=self.pad_val, padx=self.pad_val)   
        
        self.button_cancel= tk.Button(master=self.correct_position_window,text=" cancel ", command=self.action_cancel, width=int(self.button_length/2))
        self.button_cancel.grid(row=2, column=11, pady=self.pad_val, padx=self.pad_val)     
        
       
    def action_apply_change(self):
        '''
        apply the provided correction to the given position
        
        '''
        
        self.trace[self.frame_pos_to_change[0]]=[float(self.txt_position.get().split(',')[0]), float(self.txt_position.get().split(',')[1])]
        
        
        self.track_data['trace']=self.trace
        self.track_data['frames']=self.frames
        
        self.motion=self.motion_type_evaluate(self.track_data)
        self.track_data['motion']=self.motion
        
        # update visualisation
        self.show_list()  
        
        self.plot_image()
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
        self.action_cancel()
        
        self.frame_pos_to_change=()
        

        
    def action_cancel(self):
        '''
        remove all the widgets related to changes in trajectory
        
        '''

        try: 
            self.add_position_window.destroy()
        except: 
            pass
        
        try:
            self.delete_position_window.destroy()
        except: 
            pass
        try: 
            self.correct_position_window.destroy()  
        except: 
            pass    
        try: 
            self.add_coordinates_frame_window.destroy()
        except:
            pass

        
    def delete_position(self):
        '''
        delete selected position - question window
        '''
        
        self.action_cancel()
        
        self.delete_position_window = tk.Toplevel(root, bg='white')
        self.delete_position_window.title(" Delete ")
        self.delete_position_window.geometry("+20+20")
        
        
        self.lbframechange = tk.Label(master=self.delete_position_window, text="   Do you want to delete frame "+str([self.frames[x] for x in self.frame_pos_to_change])+" ?   ", bg='white')
        self.lbframechange.grid(row=0, column=10, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)              
        

        self.buttonOKdel= tk.Button(master=self.delete_position_window,text=" delete ", command=self.action_apply_delete, width=int(self.button_length/2), bg='red')
        self.buttonOKdel.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val)  
        
        self.button_cancel= tk.Button(master=self.delete_position_window,text=" cancel ", command=self.action_cancel, width=int(self.button_length/2))
        self.button_cancel.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)     
        
    def action_apply_delete(self):
        '''
        delete selected position
        
        '''
        frame_pos_to_change_sorted=sorted(self.frame_pos_to_change, reverse=True)
        for frame_pos_del in frame_pos_to_change_sorted:
            del self.trace[frame_pos_del] 
            del self.frames[frame_pos_del] 
        
        self.track_data['trace']=self.trace
        self.track_data['frames']=self.frames
        
        self.motion=self.motion_type_evaluate(self.track_data)
        self.track_data['motion']=self.motion
        
        # update visualisation
        self.show_list()  
        
        self.plot_image()
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
        self.action_cancel()
        self.frame_pos_to_change=()
        
    def add_position(self): 
        '''
        add new position with frame number with coordinates - question window
        
        '''
        
        self.action_cancel()   

        
        # open new window
       
        self.add_position_window = tk.Toplevel( bg='white')
        self.add_position_window.title(" Create new ")
        self.add_position_window.geometry("+20+20")
        
        self.lbframechange = tk.Label(master=self.add_position_window, text=" Add frame: ", bg='white')
        self.lbframechange.grid(row=0, column=10, pady=self.pad_val, padx=self.pad_val)

        self.txt_frame = tk.Entry(self.add_position_window, width=int(self.button_length))
        self.txt_frame.grid(row=0, column=11)                
        

        self.lbpose = tk.Label(master=self.add_position_window, text=" coordinates: x, y ", bg='white')
        self.lbpose.grid(row=1, columnspan=1, column=10, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_position_coordinates = tk.Entry(self.add_position_window, width=int(self.button_length*2))
        self.txt_position_coordinates.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)                
        

        self.buttonOK_add= tk.Button(master=self.add_position_window,text=" apply ", command=self.action_apply_add, width=int(self.button_length/2))
        self.buttonOK_add.grid(row=2, column=10, pady=self.pad_val, padx=self.pad_val)   

        self.button_cancel= tk.Button(master=self.add_position_window,text=" apply & add ", command=self.action_apply_add_extra)
        self.button_cancel.grid(row=2, column=11, pady=self.pad_val, padx=self.pad_val) 
        
        self.button_cancel= tk.Button(master=self.add_position_window,text=" cancel ", command=self.action_cancel, width=int(self.button_length/2))
        self.button_cancel.grid(row=2, column=12, pady=self.pad_val, padx=self.pad_val)     

        
    def action_apply_add(self):
        '''
        create the position with given parameters
        
        '''
        
        # get new location
        location_val=[float(self.txt_position_coordinates.get().split(',')[0]), float(self.txt_position_coordinates.get().split(',')[1])]
        frame_val=int(self.txt_frame.get())
        
        # where to insert the postion
        if len(self.frames)==0:
            pos=0
            
        elif frame_val<self.frames[0]: # at the start
            pos=0
        
        elif frame_val>self.frames[-1]: # at the end
            pos=len(self.frames)+1
            
        else: #somewhere in the middle
            
            diff_array=frame_val-np.asarray(self.frames)
            
            # find closest temporal position 
            val=np.argmin(diff_array[diff_array>=0])
            pos=val+1
            
            
        self.trace.insert(pos,location_val)
        self.frames.insert(pos,frame_val)
        self.track_data['trace']=self.trace
        self.track_data['frames']=self.frames
        
        self.motion=self.motion_type_evaluate(self.track_data)
        self.track_data['motion']=self.motion


        # update visualisation
        self.show_list()     
        
        self.plot_image()
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
        # remove the widgets
        self.action_cancel()
        


        
    def action_apply_add_extra(self):
        '''
        create the position with given parameters and start a new add function
        
        '''
        
        # get new location
        location_val=[float(self.txt_position_coordinates.get().split(',')[0]), float(self.txt_position_coordinates.get().split(',')[1])]
        frame_val=int(self.txt_frame.get())
        
        # where to insert the postion
        
        if len(self.frames)==0:
            pos=0
            
        elif frame_val<self.frames[0]: # at the start
            pos=0
        
        elif frame_val>self.frames[-1]: # at the end
            pos=len(self.frames)+1
            
        else: #somewhere in the middle
            
            diff_array=frame_val-np.asarray(self.frames)
            
            # find closest temporal position 
            val=np.argmin(diff_array[diff_array>=0])
            pos=val+1

        self.trace.insert(pos,location_val)
        self.frames.insert(pos,frame_val)
        self.track_data['trace']=self.trace
        self.track_data['frames']=self.frames
        
        self.motion=self.motion_type_evaluate(self.track_data)
        self.track_data['motion']=self.motion
        
        
        self.frame_pos=frame_val+1

        # update visualisation
        self.show_list()     
        
        self.plot_image()
        self.scale_movie.set(self.frame_pos)
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
#        # remove the widgets
#        self.action_cancel()
        
        
        # open new add pos
        self.add_position()

    def move_to_previous(self):
        '''
        one step back
        
        '''
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.plot_image()
        self.scale_movie.set(self.frame_pos)
        
    def move_to_next(self):
        '''
        one step forward
        
        '''
        if self.frame_pos!=self.movie_length-1:
            self.frame_pos+=1
        self.plot_image() 
        self.scale_movie.set(self.frame_pos)            
    
    
    def plot_image(self):
        '''
        arrange the image viewer
        
        '''
                
        img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])


        #calculate window position        
        
        if len(self.trace)==0:
            # new track -> show a zoomed in main window frame_zoom

            left_point_y=int(self.frame_zoom[0][0])
            right_point_y=int(self.frame_zoom[0][1])
            top_point_x=int(self.frame_zoom[1][1])
            bottom_point_x=int(self.frame_zoom[1][0])
            
        else:
            left_point_y=int(np.min(np.asarray(self.trace)[:,1])-self.pixN_basic)
            right_point_y=int(np.max(np.asarray(self.trace)[:,1])+self.pixN_basic)
            top_point_x=int(np.min(np.asarray(self.trace)[:,0])-self.pixN_basic)
            bottom_point_x=int(np.max(np.asarray(self.trace)[:,0])+self.pixN_basic)
        
        
        
        
        # for y
        if left_point_y>=0 and right_point_y<img.shape[1]:
            y_min=left_point_y
            y_max=right_point_y
        elif left_point_y<0 and right_point_y<img.shape[1]:
            y_min=0
#            y_max=np.min([y_min+2*self.pixN_basic, (img.shape[1]-1)])  
            y_max=right_point_y
        elif left_point_y>=0 and right_point_y>=img.shape[1]:
            y_max=img.shape[1]-1
#            y_min=np.max([0, y_max-2*self.pixN_basic])
            y_min=left_point_y
        else:
            y_min=0
            y_max=img.shape[1]-1
            
        # for x
        
        if top_point_x>=0 and bottom_point_x<img.shape[0]:
            x_min=top_point_x
            x_max=bottom_point_x
        elif top_point_x<0 and bottom_point_x<img.shape[0]:
            x_min=0
#            x_max=np.min([x_min+2*self.pixN_basic, (img.shape[0]-1)])  
            x_max=bottom_point_x
        elif top_point_x>=0 and bottom_point_x>=img.shape[0]:
            x_max=img.shape[0]-1
#            x_min=np.max([0, x_max-2*self.pixN_basic])
            x_min=top_point_x
        else:
            x_min=0
            x_max=img.shape[0]-1       
            
        # update the range for click action
        self.img_range=[[x_min, x_max], [y_min, y_max]]
        
        # extract the region
        region=img[x_min:x_max, y_min:y_max]
        
        blue_c=np.linspace(0., 1., len(self.trace))
        red_c=1-np.linspace(0., 1., len(self.trace))
        
        self.ax_indwin.clear()
        self.ax_indwin.imshow(region, cmap="gray")
        self.ax_indwin.axis('off')
        
        if self.plot_switch==5:
            
            # plot tracks
            '''
            
            left_point_y=int(self.frame_zoom[0][0])
            right_point_y=int(self.frame_zoom[0][1])
            top_point_x=int(self.frame_zoom[1][1])
            bottom_point_x=int(self.frame_zoom[1][0])
            
            '''
            plot_info=self.track_data_framed['frames'][self.frame_pos]['tracks']
            
            for p in plot_info:
                
                # check that the track is in the area
                
                trace=p['trace']
                
                ys=np.asarray(trace)[:,1]
                xs=np.asarray(trace)[:,0]
                margin=5
                condition_x= np.any(xs-margin>x_min) and np.any(xs+margin<x_max) 
                condition_y= np.any(ys-margin>y_min) and np.any(ys+margin<y_max) 
                
                
                if condition_x and condition_y:
                    self.ax_indwin.plot(np.asarray(trace)[:,1]-y_min,np.asarray(trace)[:,0]-x_min,  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                    self.ax_indwin.text(np.asarray(trace)[0,1]-y_min-5,np.asarray(trace)[0,0]-x_min-5, str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])

            
        else:
        
            if len(self.trace)>0:
                if self.plot_switch==0: # print full trajectory
                    for pos in range(0, len(self.trace)-1):
                        self.ax_indwin.plot(np.asarray(self.trace)[pos:pos+2,1]- y_min,np.asarray(self.trace)[pos:pos+2,0]-x_min,  color=(red_c[pos],0,blue_c[pos]))
                
                    self.ax_indwin.text(np.asarray(self.trace)[-1,1]- y_min,np.asarray(self.trace)[-1,0]- x_min, "  END  ", fontsize=16, color="b")
                    self.ax_indwin.plot(np.asarray(self.trace)[-1,1]- y_min,np.asarray(self.trace)[-1,0]- x_min,  "bo",)  
                    
                    self.ax_indwin.text(np.asarray(self.trace)[0,1]- y_min,np.asarray(self.trace)[0,0]- x_min, "  START  ", fontsize=16, color="r")
                    
                    self.ax_indwin.plot(np.asarray(self.trace)[0,1]- y_min,np.asarray(self.trace)[0,0]- x_min,  "ro",)
                    
                elif self.plot_switch==2 and self.traj_segm_switch_var>0: # plotting motion type
                    #define colour
                    red_c= (abs(np.array(self.motion)-1)).tolist()
                    green_c= self.motion
                    for pos in range(1, len(self.trace)):
                        self.ax_indwin.plot(np.asarray(self.trace)[pos-1:pos+1,1]- y_min,np.asarray(self.trace)[pos-1:pos+1,0]-x_min,  color=(red_c[pos],green_c[pos],0))
          

        self.canvas_indwin.draw()
        

    def show_parameters(self): 
        '''
        calculate and print trajectory parameters
        
        '''

        # show the list of data with scroll bar
        try:
            self.listNodes_parameters.delete(0,tk.END)
        except:
            pass
                
        
        if len(self.track_data['frames'])>1:
        
            average_speeds=self.tg.calculate_speed(self.track_data, "average")
            moving_speeds=self.tg.calculate_speed(self.track_data, "movement")
            
            max_curvilinear_segment=self.tg.max_speed_segment(self.track_data, int(self.speed_sliding_window*self.frame_rate))["speed"]
            
            try:
                max_curvilinear_segment=np.round(max_curvilinear_segment*self.img_resolution*self.frame_rate,0)
            except:
                 max_curvilinear_segment=None  
                 
            try:
                mean_curvilinear_average=np.round(average_speeds[0]*self.img_resolution*self.frame_rate,0)
            except:
                 mean_curvilinear_average=None  
                 
                 
            try:
                mean_straightline_average=np.round(average_speeds[1]*self.img_resolution*self.frame_rate,0)
            except:
                 mean_straightline_average=None  
                 
                 
            try:
                mean_curvilinear_moving=np.round(moving_speeds[0]*self.img_resolution*self.frame_rate,0)
            except:
                 mean_curvilinear_moving=None  
            try:
                mean_straightline_moving=np.round(moving_speeds[1]*self.img_resolution*self.frame_rate,0)
            except:
                 mean_straightline_moving=None  
            
            try:
                _, _, intensity_mean_1, intensity_mean_2,_=SupportFunctions.intensity_calculation(self, self.movie, self.trace, self.frames, self.vesicle_patch_size)
            except:
                intensity_mean_1=None
                intensity_mean_2=None
#                        
                
                
           # add to the list
            self.listNodes_parameters.insert(tk.END, " Total distance travelled                    "+str(np.round(self.total_distance*self.img_resolution,2))+" nm") 
    
            self.listNodes_parameters.insert(tk.END, " Net distance travelled                      "+str(np.round(self.net_displacement*self.img_resolution,2))+" nm")  
    
            self.listNodes_parameters.insert(tk.END, " Maximum distance travelled                  "+str(np.round(self.max_displacement*self.img_resolution,2))+" nm")
            
            self.listNodes_parameters.insert(tk.END, " Total trajectory time                       "+str(np.round((self.frames[-1]-self.frames[0])/self.frame_rate,5))+" sec")
    
            self.listNodes_parameters.insert(tk.END, " Net orientation                             "+str(self.calculate_direction(self.trace))+ " degrees")
    
#            self.listNodes_parameters.insert(tk.END, " Mean brightness (normilised [0,1])                 "+str(np.round(intensity_mean_1,5)))  
            
            self.listNodes_parameters.insert(tk.END, " Mean brightness (normalised)                "+str(np.round(intensity_mean_2,5)))
            
            self.listNodes_parameters.insert(tk.END, " Mean curvilinear speed: average             "+str(mean_curvilinear_average)+" nm/sec")
     
            self.listNodes_parameters.insert(tk.END, " Mean straight-line speed: average           "+str(mean_straightline_average)+" nm/sec")
    
            self.listNodes_parameters.insert(tk.END, " Mean curvilinear speed: moving              "+str(mean_curvilinear_moving)+" nm/sec")
    
            self.listNodes_parameters.insert(tk.END, " Mean straight-line speed: moving            "+str(mean_straightline_moving)+" nm/sec")
    
       #     self.listNodes_parameters.insert(tk.END, " Max curvilinear speed: moving               "+str(np.round(moving_speeds[2]*self.img_resolution*self.frame_rate,0))+" nm/sec")
    
            self.listNodes_parameters.insert(tk.END, " Max curvilinear speed over a segment : moving  "+str(max_curvilinear_segment)+" nm/sec")

        
    def show_list(self): 
        '''
        arrangement for the position list
        
        '''

        try:
            self.listNodes.delete(0,tk.END)
        except:
            pass
                
        
       # plot the track positions
        for i in range(0, len(self.frames)):
             # add to the list
            self.listNodes.insert(tk.END, str(self.frames[i])+":  "+str(np.round(self.trace[i], 4,)))    
            
            # colour gaps
            try: 
                prev_frame=self.frames[i-1]
            except:
                prev_frame=0
                    
            if self.frames[i]-prev_frame!=1 and i!=0:
                self.listNodes.itemconfig(tk.END, {'fg': 'red'})




        
    def intensity_calculation(self):
        '''
        Calculates changes in intersity for the given track
        and plot it into a canva
        '''

        intensity_array_1, intensity_array_2, intensity_mean_1, intensity_mean_2,check_border=SupportFunctions.intensity_calculation(self, self.movie, self.trace, self.frames, self.vesicle_patch_size)
        
        # plotting
        try :
            
            self.ax_intensity.clear()           
#            self.ax_intensity.plot(self.frames, intensity_array_1, "-b", label="normalised [0,1]")
            self.ax_intensity.plot(self.frames, intensity_array_2, "-k", label="normalised by max")
            
#            self.ax_intensity.plot(self.frames, intensity_array_1, "-b", label="segmented vesicle")
#            self.ax_intensity.plot(self.frames, intensity_array_2, "-k", label="without segmentation")

#            self.ax_intensity.plot(self.frames, (intensity_array_1-np.min(intensity_array_1))/np.max(((np.max(intensity_array_1)-np.min(intensity_array_1)), 0.00001)), "-b", label="segmented vesicle")
#    
##            self.ax_intensity.set_ylabel("intensity", fontsize='small')
#            self.ax_intensity.plot(self.frames, (intensity_array_2-np.min(intensity_array_2))/np.max(((np.max(intensity_array_2)-np.min(intensity_array_2)), 0.00001)), "-k", label="without segmentation")
            if check_border==0:
                self.ax_intensity.set_title('Vesicle intensity (normalised) per frame', fontsize='small')
            else:
                self.ax_intensity.set_title('Vesicle intensity: fail to compute for all frames!', fontsize='small')
#            self.ax_intensity.legend(fontsize='small')   
            self.ax_intensity.set_ylim(top = 1.1, bottom = 0)            
            
            # DrawingArea
            self.canvas_intensity.draw()
            
        except:
            pass
    def plot_displacement(self):
        '''
        displacement plot
        
        '''
        def color_map_color(value, cmap_name='rainbow', vmin=0, vmax=10):

            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(cmap_name)  # PiYG

            rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
            color = matplotlib.colors.rgb2hex(rgb)
            return color
        
        
        trajectory=self.trace
        try:
            #calculate the displacement
            x=np.asarray(trajectory)[:,0]    
            y=np.asarray(trajectory)[:,1]
            x_0=np.asarray(trajectory)[0,0]
            y_0=np.asarray(trajectory)[0,1]
            
            x_e=np.asarray(trajectory)[-1,0]
            y_e=np.asarray(trajectory)[-1,1]
            
            self.displacement_array=np.sqrt((x-x_0)**2+(y-y_0)**2)
            #calculate all type of displacements
            # max displacement
            self.max_displacement=np.round(np.max(self.displacement_array),2)
            
            # displacement from start to the end
            self.net_displacement=np.round(np.sqrt((x_e-x_0)**2+(y_e-y_0)**2),2)
            
            # total displacement
            x_from=np.asarray(trajectory)[0:-1,0] 
            y_from=np.asarray(trajectory)[0:-1,1] 
            x_to=np.asarray(trajectory)[1:,0] 
            y_to=np.asarray(trajectory)[1:,1] 
            
            self.total_distance=np.round(np.sum(np.sqrt((x_to-x_from)**2+(y_to-y_from)**2)),2) 
            
     
            disaplcement=self.displacement_array*self.img_resolution
            
                    
            self.ax_displacement.clear()
            
            if self.speed_graph_var==0: 
            
                # plot displacement colour
                
                for i in range(1, len(self.motion)):
    
                    if self.motion[i]==0:
                        colourV='r'
                    else:
                        colourV='g'
    
                    self.ax_displacement.plot((self.frames[i-1],self.frames[i]), (disaplcement[i-1],disaplcement[i]), colourV)
                    
                    
            else: # self.speed_graph_var==1:
                
                speed_dict=self.tg.max_speed_segment(self.track_data, int(self.speed_sliding_window*self.frame_rate))
    
                
                speed_disp=np.sqrt((x_to-x_from)**2+(y_to-y_from)**2)
                
                speed_array=speed_disp*self.frame_rate*self.img_resolution
                for i in range(1, len(self.motion)):
    
                    if self.motion[i]==0:
                        colourV='k'
                    else:
                        colourV=color_map_color(speed_array[i-1], vmin=0, vmax=2000)
    
                    self.ax_displacement.plot((self.frames[i-1],self.frames[i]), (disaplcement[i-1],disaplcement[i]), colourV)
                
                #plot fastest segment
                
                if speed_dict['frames']!=[]:
                
                    fastest_segment_start=np.where(np.asarray(self.frames)==int(speed_dict['frames'][0]))[0][0]
                    fastest_segment_end=np.where(np.asarray(self.frames)==int(speed_dict['frames'][1]))[0][0]
                    
                    fastest_speed_value=speed_dict['speed']*self.img_resolution*self.frame_rate
                              
                    
                    colourV=color_map_color(fastest_speed_value, vmin=0, vmax=2000)
                    self.ax_displacement.plot((speed_dict['frames'][0]-0.6,speed_dict['frames'][1]-0.6), (disaplcement[fastest_segment_start],disaplcement[fastest_segment_end]), colourV)
                    
    
                    frame_val=(speed_dict['frames'][0]+speed_dict['frames'][1])/2
                    displacement_val=(disaplcement[fastest_segment_start]+disaplcement[fastest_segment_end])/2
                    self.ax_displacement.text(frame_val,displacement_val, str(int(np.round(fastest_speed_value, 0)))+" nm", fontsize='8') #, c=colourV)
               
                #colourbar 
    #            normi = matplotlib.colors.Normalize(vmin=np.min(speed_array), vmax=np.max(speed_array));
    #            self.cbar.set_clim(vmin=np.min(speed_array),vmax=np.max(speed_array))
            
    
            
            #        self.ax_displacement.set_ylabel('Displacement (nm)', fontsize='small')
    
            self.ax_displacement.set_title('Displacement (nm) per frame', fontsize='small')

            # DrawingArea
            self.canvas_displacement.draw()
            
        except:
            pass

    def motion_type_evaluate(self, track_data_original):
        '''
        provide motion type evaluation to select directed movement for speed evaluation
        '''
        
        if self.traj_segm_switch_var==0: # no segmentation required
            motion_type=[0] * len(track_data_original['frames'])
            
        elif  self.traj_segm_switch_var==1: # MSD based segmentation
            # set trajectory length
            self.tg.window_length=10
            # run segmentation
            segmentation_result=self.tg.msd_based_segmentation(track_data_original['trace'])
            motion_type=segmentation_result[:len(track_data_original['frames'])]
            
        else: # U-Net based segmentation
            # set trajectory length
            self.tg.window_length=8
            # run segmentation
            segmentation_result=self.tg.unet_segmentation(track_data_original['trace'])
            motion_type=segmentation_result[:len(track_data_original['frames'])]

        
        return motion_type   
        
class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("MSP-viewer 0.3")
        parent.configure(background='white')
        
        self.main = MainVisual(parent)

        parent.protocol('WM_DELETE_WINDOW', self.close_app)

        tk.mainloop()

    def close_app(self):
        self.quit()
        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
