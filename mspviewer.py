
#########################################################
#
#  MSP-viewer GUI 
#        
#########################################################


import numpy as np
import scipy as sp

import copy
import tkinter as tk
from tkinter import filedialog

import csv

# for plotting
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import Event
from matplotlib.figure import Figure

import skimage
from skimage import io
from scipy.ndimage import gaussian_filter1d
import json        
import cv2
import imageio
import math
from skimage.feature import peak_local_max
from tqdm import tqdm
from viewer_lib.fusion_events import FusionEvent 

from viewer_lib.trajectory_segmentation import TrajectorySegment

import os
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
        self.membrane_movie=[]
        self.track_data_original={}
        self.track_data={'tracks':[]} # original tracking data
        self.track_data_filtered={'tracks':[]}  # filtered tracking data  
        self.track_data_framed={}  # tracking data arranged by frames  
        self.stat_data=[] # stat data to save csv file
        
        # segmentation 
        self.tg = TrajectorySegment()     
        self.tg.window_length=8

        #filter parameters
        self.filter_duration=[0, 10000]
        self.filter_length=[0, 100000]   
        self.filter_speed=[0, 100000] 
        self.filter_zoom=0 # option to include tracks only in zoomed area
        
        self.xlim_zoom=[0,10000]
        self.ylim_zoom=[10000, 0]
        
        self.frame_pos=0
        self.movie_length=1
        self.monitor_switch=0 # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.monitor_axis=0 # 0 - not to show axis, 1- show axis
        self.memebrane_switch=0 # 0 - don't show the membrane, 1 -s how the membrane
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
        
        # define window  proportions in relation to the monitor size
        self.button_length=np.max((20,int(self.window_width/50)))
        self.pad_val=2
        self.dpi=100
        self.img_width=self.window_height*0.6
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        
        ### frames ###
    
        # Framework: filtering tracks
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
        
        self.button_mv = tk.Button(text="   Select particle movie   ", command=self.select_vesicle_movie, width=self.button_length)
        self.button_mv.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val)

        self.button_mm = tk.Button(text="   Select membrane movie   ", command=self.select_membrane_movie, width=self.button_length)
        self.button_mm.grid(row=0, column=2, columnspan=2,pady=self.pad_val, padx=self.pad_val)
        
        self.button2 = tk.Button(text="Select file with tracks", command=self.select_track, width=self.button_length)
        self.button2.grid(row=1, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
  

     # # # # # #  Radiobutton: membrane on/off # # # # # # #   
        var_membrane = tk.IntVar()
        
        def update_membrane_switch():            
            self.memebrane_switch=var_membrane.get()
            # change image
            self.show_tracks()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(root, text="without membrane", variable=var_membrane, value=0, bg='white', command =update_membrane_switch )
        self.M1.grid(row=3, column=0, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(root, text=" with membrane ", variable=var_membrane, value=1, bg='white',command = update_membrane_switch ) #  command=sel)
        self.M2.grid(row=3, column=1, columnspan=2, pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(root, text=" with border ", variable=var_membrane, value=2, bg='white',command = update_membrane_switch ) #  command=sel)
        self.M3.grid(row=3, column=3, pady=self.pad_val, padx=self.pad_val)
        
        
#    # # # # # # Radiobuttone:tracks # # # # # # #   
        var = tk.IntVar()
        
        def update_monitor_switch():            
            self.monitor_switch=var.get()
            self.show_tracks()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(root, text="track and IDs", variable=var, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=4, column=0, pady=self.pad_val, padx=self.pad_val)  
        
        self.R2 = tk.Radiobutton(root, text=" only tracks ", variable=var, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=4, column=1, columnspan=2, pady=self.pad_val, padx=self.pad_val)
        
        self.R3 = tk.Radiobutton(root, text="    none    ", variable=var, value=2, bg='white',command=update_monitor_switch ) #  command=sel)
        self.R3.grid(row=4, column=3, pady=self.pad_val, padx=self.pad_val)
        
#    # # # # # # Radiobuttone:axis # # # # # # #   
        var_axis = tk.IntVar()
        
        def update_monitor_switch():            
            self.monitor_axis=var_axis.get()
            self.show_tracks()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(root, text="axis off", variable=var_axis, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=5, column=0,  pady=self.pad_val, padx=self.pad_val)  
        
        self.R2 = tk.Radiobutton(root, text=" axis on ", variable=var_axis, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=5, column=1, columnspan=2, pady=self.pad_val, padx=self.pad_val)     
        
#    # # # # # #  resolution in time and space   # # # # # # #  
            
        res_lb = tk.Label(master=root, text=" resolution (nm/pix) : ", width=self.button_length, bg='white')
        res_lb.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.img_resolution))
        self.res_parameter = tk.Entry(root, width=10, text=v)
        self.res_parameter.grid(row=6, column=1, pady=self.pad_val, padx=self.pad_val)
            
        lbl3 = tk.Label(master=root, text=" frame rate (f/sec) : ", width=self.button_length, bg='white')
        lbl3.grid(row=6, column=2, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.frame_rate))
        self.frame_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
        self.frame_parameter.grid(row=6, column=3, pady=self.pad_val, padx=self.pad_val)        
        
            
        # AP axis 
        ap_lb = tk.Label(master=root, text=" Axis orientation ", width=self.button_length, bg='white')
        ap_lb.grid(row=7, column=0, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.ap_axis))
        self.ap_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
        self.ap_parameter.grid(row=7, column=1, pady=self.pad_val, padx=self.pad_val)
            
        lbl3 = tk.Label(master=root, text="Axis  (A,B): ", width=self.button_length, bg='white')
        lbl3.grid(row=7, column=2, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.axis_name))
        self.axis_name_parameter = tk.Entry(root, width=int(self.button_length/2), text=v)
        self.axis_name_parameter.grid(row=7, column=3, pady=self.pad_val, padx=self.pad_val)   
       
        
        #update the list
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
        lbl3 = tk.Label(master=self.filterframe, text="Max travelled distance (nm): from ", width=int(self.button_length*2), bg='white')
        lbl3.grid(row=4, column=5)
        
        self.txt_length_from = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_length_from.grid(row=4, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl3 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl3.grid(row=4, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_length_to = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_length_to.grid(row=4, column=8, pady=self.pad_val, padx=self.pad_val)  
        
        # curvilinear moving speed
        lbl4 = tk.Label(master=self.filterframe, text="Mean curvilinear moving speed : from ", width=int(self.button_length*2), bg='white')
        lbl4.grid(row=5, column=5)
        
        self.txt_speed_from = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_speed_from.grid(row=5, column=6, pady=self.pad_val, padx=self.pad_val)
        
        lbl5 = tk.Label(master=self.filterframe, text="to", bg='white')
        lbl5.grid(row=5, column=7, pady=self.pad_val, padx=self.pad_val)
        
        self.txt_speed_to = tk.Entry(self.filterframe, width=int(self.button_length/2))
        self.txt_speed_to.grid(row=5, column=8, pady=self.pad_val, padx=self.pad_val)
        
        # Radio button zoom
        var_filter_zoom = tk.IntVar()
        
        def update_monitor_switch():            
            self.filter_zoom=var_filter_zoom.get()
            
        lbl5 = tk.Label(master=self.filterframe, text=" Trajectories included for zoomed area: ", width=int(self.button_length*2), bg='white')
        lbl5.grid(row=6, column=5)

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(master=self.filterframe, text=" at least one point", variable=var_filter_zoom, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=6, column=6,  pady=self.pad_val, padx=self.pad_val)  
        
        self.R2 = tk.Radiobutton(master=self.filterframe, text=" all points ", variable=var_filter_zoom, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=6, column=8, pady=self.pad_val, padx=self.pad_val)          
        
        # button to filter
        
        self.buttonFilter = tk.Button(master=self.filterframe, text="Filter", command=self.filtering, width=self.button_length)
        self.buttonFilter.grid(row=7, column=4, columnspan=4, pady=self.pad_val, padx=self.pad_val)  


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
        button_save=tk.Button(master=self.resultbuttonframe, text="trajectories: save updates", command=self.save_in_file, width=int(self.button_length*1.5))
        button_save.grid(row=16, column=7, pady=self.pad_val, padx=self.pad_val)  
        
      # # # # # # movie  # # # # # # 
        
        # plot bg
        self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize_value)
        self.ax.axis('off')
        
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
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
            # plot
            
            plt.figure()
            plt.imshow(self.image, cmap="gray")
            for trackID in range(0, len(self.track_data_filtered['tracks'])):
                track=self.track_data_filtered['tracks'][trackID]
                trace=track['trace']
                plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(trackID)%len(self.color_list_plot)])     
                if self.monitor_switch==0:
                    plt.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(track['trackID']), fontsize=10, color=self.color_list_plot[int(trackID)%len(self.color_list_plot)])
            
            if self.memebrane_switch==2:
                #extract skeleton
                skeleton = skimage.morphology.skeletonize(self.membrane_movie[self.frame_pos,:,:]).astype(np.int)
                # create an individual cmap with red colour
                cmap_new = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['red','red'],256)
                cmap_new._init() 
                alphas = np.linspace(0, 0.8, cmap_new.N+3)
                cmap_new._lut[:,-1] = alphas
                #plot the membrane border on the top
                plt.imshow(skeleton, interpolation='nearest', cmap=cmap_new)         
            
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
            self.frame_rate=int(self.frame_parameter.get())
        
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
            
            for trackID in range(0, len(self.track_data_filtered['tracks'])):
                track=self.track_data_filtered['tracks'][trackID]
            #    calculate parameters
                point_start=track['trace'][0]
                point_end=track['trace'][-1]
    
                # calculate orientation
                y=point_end[1]-point_start[1]
                x=point_end[0]-point_start[0]
                orintation_move=(math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360
                
                orintation_array.append(orintation_move)    
                
            # save the array into the file
    
            
            if not(save_file.endswith(".txt")):
                save_file += ".txt"  
                
            # save in json format                    
            with open(save_file, 'w') as f:
                json.dump({'orientation':orintation_array}, f, ensure_ascii=False) 
        
    def plot_multiple_motion_map(self):
        '''
        load and plot multiple orientations together
        
        '''
        
        # load multiple files
        load_files = tk.filedialog.askopenfilenames(title='Choose all files together')
        print(load_files)
         
        if not load_files:
            print("Files were not selected. The data will not be processed.")
        else:
                    
            # plot the image
                       
            # create the joint orientation list            
            orientation_all=[]
            
            for file_name in load_files:
                
                #read from json format 
                with open(file_name) as json_file:  
                    orientation_new = json.load(json_file)
                    
                orientation_all+=orientation_new['orientation']        
            
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
                
                
            # plot and save 
            orientation_fig=plt.figure(figsize=(8,8))
            
            ax_new = plt.subplot(111, projection='polar')
            
            bin_size=10
            a , b=np.histogram(orientation_all, bins=np.arange(0, 360+bin_size, bin_size))
            centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1]) 
    
            plt.xticks(np.radians((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)),
               [second_name, '30', '60', '90' , '120', '150',first_name,'210', '240', '270', '300', '330'])
            ax_new.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
            ax_new.set_theta_direction(1)
            ax_new.set_title(" movement orientation \n based on track count ")            

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
                plt.savefig(save_file) 
                
                
        
    def plot_motion_map(self):
        '''
        plot motion map with given AP
        
        '''
        
        # show the results in a separate window
        
    #read ap axis
        if self.ap_parameter.get()!='':
            self.ap_axis=int(self.ap_parameter.get())
            
        if self.axis_name_parameter.get()!='':
            self.axis_name=self.axis_name_parameter.get()
  
        orintation_array=[]
        
        orientation_map_figure = plt.figure(figsize=(15,6))
        plt.axis('off')
        ax = orientation_map_figure.add_subplot(121)
        ax.imshow(self.movie[1,:,:]/np.max(self.movie[1,:,:])+self.membrane_movie[1,:,:]/np.max(self.membrane_movie[1,:,:])*0.6, cmap='bone') 
               
        
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
            
        ax.plot([arrow_a[1], arrow_b[1]], [arrow_a[0], arrow_b[0]],  color='g', alpha=0.7)
        ax.text(arrow_a[1], arrow_a[0]-5,  second_name, color='g', size=12, alpha=0.7)
        ax.text(arrow_b[1], arrow_b[0]-5,  first_name, color='g', size=12, alpha=0.7)

        for trackID in range(0, len(self.track_data_filtered['tracks'])):
            track=self.track_data_filtered['tracks'][trackID]
            
            # calculate parameters
            point_start=track['trace'][0]
            point_end=track['trace'][-1]

            # calculate orientation
            y=point_end[1]-point_start[1]
            x=point_end[0]-point_start[0]
            orintation_move=(math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360
            
            orintation_array.append(orintation_move)
   
            color='r'
            plt.arrow(point_start[1],point_start[0], point_end[1]-point_start[1], point_end[0]-point_start[0], head_width=3.00, head_length=2.0, 
                      fc=color, ec=color, length_includes_head = True)
        
        bin_size=10
        a , b=np.histogram(orintation_array, bins=np.arange(0, 360+bin_size, bin_size))
        centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])   
        
        ax = orientation_map_figure.add_subplot(122, projection='polar')


        plt.xticks(np.radians((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)),
           [second_name, '30', '60', '90' , '120', '150',first_name,'210', '240', '270', '300', '330'])
        ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
        ax.set_theta_direction(1)
        ax.set_title(" movement orientation \n based on track count ")
        
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
           
    
            if not(save_file.endswith(".png")):
                save_file += ".png"        
            plt.savefig(save_file) 
            
            
        
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
            saved_membrane=self.membrane_movie[f_start:f_end,lim_x0:lim_x1,lim_y0:lim_y1]
    
            if saved_movie.shape[1]<700 and saved_movie.shape[2]<700: # hard threshold to create tiff stack or video sequence
                # save tiff file
                final_img_set=np.zeros((saved_movie.shape[0],saved_movie.shape[1],saved_movie.shape[2], 3))
    
                for frameN in range(0, saved_movie.shape[0]):
              
                    plot_info=self.track_data_framed['frames'][frameN]['tracks']
                    frame_img=saved_movie[frameN,:,:]
                    membrane_img=saved_membrane[frameN,:,:]
                    # make a colour image frame
                    orig_frame = np.zeros((saved_movie.shape[1], saved_movie.shape[2], 3))
            
                    img=frame_img/np.max(frame_img)+membrane_img*0.2
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
                    cv2.imshow('Tracking', orig_frame)
                    final_img_set[frameN,:,:,:]=orig_frame
    
                #save the file
                final_img_set=final_img_set/np.max(final_img_set)*255
                final_img_set=final_img_set.astype('uint8')
                
                if not(save_file.endswith(".tif") or save_file.endswith(".tiff")):
                    save_file += ".tif"
                    
                imageio.volwrite(save_file, final_img_set)
            else:
                # save avi file for large movies
    
                if not(save_file.endswith(".avi")):
                    save_file += ".avi"
    
                out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), 4.0, (self.movie.shape[1], self.movie.shape[2]))
        
                for frameN in range(0, saved_movie.shape[0]):
              
                    plot_info=self.track_data_framed['frames'][frameN]['tracks']
                    frame_img=saved_movie[frameN,:,:]
                    membrane_img=saved_movie[frameN,:,:]
                    
                    # create a colour image frame
                    orig_frame = np.zeros((saved_movie.shape[1], saved_movie.shape[2], 3))
            
                    img=frame_img/np.max(frame_img)+membrane_img*0.2
                    orig_frame [:,:,0] = img/np.max(img)*256
                    orig_frame [:,:,1] = img/np.max(img)*256
                    orig_frame [:,:,2] = img/np.max(img)*256
                    
                    for p in plot_info:
                        trace=p['trace']
                        trackID=p['trackID']
                        
                        clr = trackID % len(self.color_list)
                        if (len(trace) > 1):
                            for j in range(len(trace)-1):
                                # Draw trace line
                                point1=trace[j]
                                point2=trace[j+1]
                                x1 = int(point1[1])
                                y1 = int(point1[0])
                                x2 = int(point2[1])
                                y2 = int(point2[0])                        
                                cv2.line(orig_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                         self.color_list[clr], 1)
                                
                    # Display the resulting tracking frame
                    cv2.imshow('Tracking', orig_frame)
                    # write the flipped frame
                    out.write(np.uint8(orig_frame))
        
                    
                out.release()
    
            cv2.destroyAllWindows()
            
            print("movie location: ", save_file)
            
    
    def save_in_file(self):
        '''
        save corrected trajectories to json and csv files
        
        '''
        
        # ask for the file location        
        save_file = tk.filedialog.asksaveasfilename()
        
        if not save_file:
            print("File name was not provided. The data was not saved. ")
            
        else: 
            # save txt file with json format            
            if not(save_file.endswith(".txt")):
                save_file += ".txt"  
                
            with open(save_file, 'w') as f:
                json.dump(self.track_data_filtered, f, ensure_ascii=False) 
                
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
        if self.memebrane_switch==0:
            self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        elif self.memebrane_switch==1:
            self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])+self.membrane_movie[self.frame_pos,:,:]/4
        else:
            self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])

        self.ax.clear() # clean the plot 
        self.ax.imshow(self.image, cmap="gray")
        self.ax.axis('off')

        if  self.track_data_framed and self.monitor_switch<=1:

            # plot tracks
            plot_info=self.track_data_framed['frames'][self.frame_pos]['tracks']
            for p in plot_info:
                trace=p['trace']
                self.ax.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                if self.monitor_switch==0:
                    self.ax.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])
        if self.memebrane_switch==2:
            #extract skeleton
            skeleton = skimage.morphology.skeletonize(self.membrane_movie[self.frame_pos,:,:]).astype(np.int)
            
            # create an individual cmap with red colour
            cmap_new = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['red','red'],256)
            cmap_new._init() # create the _lut array, with rgba values
            alphas = np.linspace(0, 0.8, cmap_new.N+3)
            cmap_new._lut[:,-1] = alphas
            
            #plot the membrane border on the top
            self.ax.imshow(skeleton, interpolation='nearest', cmap=cmap_new)

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

            self.ax.plot([arrow_a[1], arrow_b[1]], [arrow_a[0], arrow_b[0]],  color='r', alpha=0.5)
            self.ax.text(arrow_a[1]-2, arrow_a[0]-2,  second_name, color='r', size=9, alpha=0.5)
            self.ax.text(arrow_b[1]-2, arrow_b[0]-2,  first_name, color='r', size=9, alpha=0.5)
            
        #set the same "zoom"        
        self.ax.set_xlim(xlim_old[0],xlim_old[1])
        self.ax.set_ylim(ylim_old[0],ylim_old[1])
        
        # inver y-axis as set_ylim change the orientation
        if ylim_old[0]<ylim_old[1]:
            self.ax.invert_yaxis()
        
        
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=12, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        # toolbar
        toolbarFrame = tk.Frame(master=root)
        toolbarFrame.grid(row=15, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        
        # update home button

        def new_home( *args, **kwargs):
            
            # zoom out            
            self.ax.set_xlim(0,self.movie.shape[2])
            self.ax.set_ylim(0,self.movie.shape[1])


            self.show_tracks()
            
        NavigationToolbar2Tk.home = new_home

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()
        
        

    def filtering(self):
        '''
        filtering tracks
        
        '''
        
        def cancel_window():
            '''
            destroy the window
            
            '''
            try:
                self.choose_traj_segmentation.destroy()
            except:
                pass
                    
        
        def run_filtering():       
            print("filtering for length: ", self.filter_length, ";   duration: ", self.filter_duration, ";   speed: ", self.filter_speed) #, ";   final stop duration: ", self.filter_stop)
    
            # filtering 
            self.track_data_filtered={}
            self.track_data_filtered.update({'tracks':[]})
            
            # check through the tracks
            for p in tqdm(self.track_data['tracks']):
                
                # check length
                if len(p['trace'])>0:
                    point_start=p['trace'][0]
                    # check length
                    track_duration=(p['frames'][-1]-p['frames'][0]+1)/self.frame_rate
                    # check maximum displacement between any two positions in track
                    track_length=np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2))*self.img_resolution
                   
                    
                else:
                    track_duration=0
                    track_length=0
    
                    # variables to evaluate the trackS
                length_var=track_length>=self.filter_length[0] and track_length<=self.filter_length[1]
                duration_var=track_duration>=self.filter_duration[0] and track_duration<=self.filter_duration[1]
                
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
                zz_x_0=np.asarray(p['trace'])[:,1]>self.xlim_zoom[0]
                zz_x_1=np.asarray(p['trace'])[:,1]<self.xlim_zoom[1]
                
                zz_y_0=np.asarray(p['trace'])[:,0]<self.ylim_zoom[0]
                zz_y_1=np.asarray(p['trace'])[:,0]>self.ylim_zoom[1]
                
                zz=zz_x_0*zz_x_1*zz_y_0*zz_y_1
    
                if self.filter_zoom==0: # any point
                    
                    zoom_filter=np.any(zz==True)
                else: # all points
                    #check all the points are inside
                    zoom_filter=np.all(zz==True)            
                

    
                if length_var==True and duration_var==True and filterID==True and zoom_filter==True:
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
            self.filter_duration[1]=1000
        else:
            self.filter_duration[1]=float(self.txt_duration_to.get())                        

        if self.txt_length_from.get()=='':
            self.filter_length[0]=0
        else:
            self.filter_length[0]=float(self.txt_length_from.get())

        if self.txt_length_to.get()=='':
            self.filter_length[1]=10000
        else:
            self.filter_length[1]=float(self.txt_length_to.get())  

        
        if self.txt_speed_from.get()=='':
            self.filter_speed[0]=0
            movement_1=False
        else:
            self.filter_speed[0]=float(self.txt_speed_from.get())
            movement_1=True
            
        if self.txt_speed_to.get()=='':
            self.filter_speed[1]=10000
            movement_2=False
        else:
            self.filter_speed[1]=float(self.txt_speed_to.get())                     
            movement_2=True
            
        
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
            self.newbutton.grid(row=2, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.choose_traj_segmentation, text=" Cancel ", command=cancel_window, width=int(self.button_length/2),  bg='green')
            self.deletbutton.grid(row=2, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
        else:
            run_filtering()
            
        

    def list_update(self):
        '''
        update track list
        '''
                
        # update movie parameters
        self.update_movie_parameters()
        
        def tracklist_on_select(even):
            position_in_list=listNodes.curselection()[0]
            
            # creating a new window with class TrackViewer
            self.new_window = tk.Toplevel(self.master)
             
            # create the track set with motion
            this_track=self.track_data_filtered['tracks'][position_in_list]
            motion_type=[0]*len(this_track['frames'])
            this_track['motion']=motion_type
            
            
            # update movie and ap-axis parameters
            self.update_movie_parameters()
            
            TrackViewer(self.new_window, this_track, self.movie, self.membrane_movie, 
                        self.img_resolution, self.frame_rate, self.ap_axis, self.axis_name)
            
            
        def detele_track_question():
            '''
            function for the delete track button
            '''
            # close windows if open
            cancel_action()
            
            # get  trackID from the list
            try:
                
                # open new window
                self.delete_window = tk.Toplevel(root)
                self.delete_window.title(" Delete the track ")
    
                self.qdeletetext = tk.Label(master=self.delete_window, text="delete track "+str(self.track_data_filtered['tracks'][listNodes.curselection()[0]]['trackID'])+" ?",  bg='white', font=("Times", 10), width=self.button_length*2)
                self.qdeletetext.grid(row=0, column=0,  columnspan=2, pady=self.pad_val, padx=self.pad_val) 
                
                self.deletbutton = tk.Button(master=self.delete_window, text=" OK ", command=detele_track, width=int(self.button_length/2),  bg='red')
                self.deletbutton.grid(row=1, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
                
                self.deletbutton = tk.Button(master=self.delete_window, text=" Cancel ", command=cancel_action, width=int(self.button_length/2),  bg='green')
                self.deletbutton.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 

            except:
                self.delete_window = tk.Toplevel(root)
                self.delete_window.title(" Delete the track ")
                self.qdeletetext = tk.Label(master=self.delete_window, text=" Track is not selected! ",  bg='white', font=("Times", 10), width=self.button_length*2)
                self.qdeletetext.grid(row=0, column=0,  columnspan=2, pady=self.pad_val, padx=self.pad_val) 

            
            
        def detele_track():
            '''
            delete selected track
            '''
            self.deleted_tracks_N+=1
            delete_trackID=self.track_data_filtered['tracks'][listNodes.curselection()[0]]['trackID']
            
            pos=0
            for p in self.track_data['tracks']:
                
                if p['trackID']==delete_trackID:
                    print("found")
                    self.track_data['tracks'].remove(p)
                    
                pos+=1

            print("track ", delete_trackID, "is removed")
            
            #visualise without the track
            self.filtering()
            self.track_to_frame()
            
            #update the list
            self.list_update()
            
            #close the window
            cancel_action()

        def duplicate_track_question():
            '''
            function for the duplicate track button
            
            '''
            
            # close windows if open
            cancel_action()
            
            self.new_trackID=1000
            
            # open new window
            self.create_window = tk.Toplevel(root)
            self.create_window.title(" Duplicate the track")
            
            self.qnewtext = tk.Label(master=self.create_window, text="duplicate  track  "+str(self.track_data_filtered['tracks'][listNodes.curselection()[0]]['trackID'])+" ? new track ID: " ,  bg='white', font=("Times", 10), width=self.button_length*2)
            self.qnewtext.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
            v = tk.StringVar(root, value=str(self.new_trackID))
            self.trackID_parameter = tk.Entry(self.create_window, width=int(self.button_length/2), text=v)
            self.trackID_parameter.grid(row=0, column=2, pady=self.pad_val, padx=self.pad_val)

                
            self.newbutton = tk.Button(master=self.create_window, text=" OK ", command=duplicate_track, width=int(self.button_length/2),  bg='green')
            self.newbutton.grid(row=1, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.create_window, text=" Cancel ", command=cancel_action, width=int(self.button_length/2),  bg='green')
            self.deletbutton.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)
            
            
            
        def new_track_question():
            '''
            function for the new track button
            '''
            # close windows if open
            cancel_action()
            
            self.new_trackID=1000
            
            # open new window
            self.create_window = tk.Toplevel(root)
            self.create_window.title(" Create new track")
            
            self.qnewtext = tk.Label(master=self.create_window, text="create new track ?  track ID: " ,  bg='white', font=("Times", 10), width=self.button_length*2)
            self.qnewtext.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
            v = tk.StringVar(root, value=str(self.new_trackID))
            self.trackID_parameter = tk.Entry(self.create_window, width=int(self.button_length/2), text=v)
            self.trackID_parameter.grid(row=0, column=2, pady=self.pad_val, padx=self.pad_val)

                
            self.newbutton = tk.Button(master=self.create_window, text=" OK ", command=create_track, width=int(self.button_length/2),  bg='green')
            self.newbutton.grid(row=1, column=0, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=self.create_window, text=" Cancel ", command=cancel_action, width=int(self.button_length/2),  bg='green')
            self.deletbutton.grid(row=1, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)

        def cancel_action():
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
            
            
        def duplicate_track():
            '''
            duplicate a track
            '''
            
            # read ap axis
            if self.trackID_parameter.get()!='':
                self.new_trackID=int(self.trackID_parameter.get())
                
            # update counting of the new tracks
            self.created_tracks_N+=1
            duplicate_trackID=self.track_data_filtered['tracks'][listNodes.curselection()[0]]['trackID']
            
            for p in self.track_data['tracks']:
                
                if p['trackID']==duplicate_trackID:
                    duplicated_track=p
                
            
            new_track={"trackID":self.new_trackID, "trace":duplicated_track['trace'], "frames":duplicated_track['frames']}
            
            self.track_data['tracks'].append(new_track)
            
            print(" track is duplicated with new trackID ", self.new_trackID)
            
            #visualise without the track
            self.filtering()
            self.track_to_frame()
            
            #update the list
            self.list_update()
            
            # close the windows
            cancel_action()        
            
            
        def create_track():
            '''
            create a track
            '''
            
            #read ap axis
            if self.trackID_parameter.get()!='':
                print(self.trackID_parameter.get())
                self.new_trackID=int(self.trackID_parameter.get())
                
            self.created_tracks_N+=1
            
            p={"trackID":self. new_trackID, "trace":[[0,0]], "frames":[0]}
            
            self.track_data['tracks'].append(p)
            
            print("new track ", self.new_trackID, "is created")
            
            #visualise without the track
            self.filtering()
            self.track_to_frame()
            
            #update the list
            self.list_update()
            
            # close the windows
            cancel_action()
            
            
        lbl2 = tk.Label(master=self.listframework, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), width=int(self.button_length*1.5), bg='white',  font=("Times", 14, "bold"))
        lbl2.grid(row=7, column=5, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        
        # show track statistics
        lbl2 = tk.Label(master=self.listframework, text="deleted tracks: "+str(self.deleted_tracks_N), width=int(self.button_length*1.5), bg='white',  font=("Times", 12,))
        lbl2.grid(row=8, column=5, columnspan=2, pady=self.pad_val, padx=self.pad_val)         

        lbl2 = tk.Label(master=self.listframework, text="filtered tracks: "+str(len(self.track_data['tracks'])-len(self.track_data_filtered['tracks'])), width=int(self.button_length*1.5), bg='white',  font=("Times", 12))
        lbl2.grid(row=8, column=7, columnspan=2, pady=self.pad_val, padx=self.pad_val)          
        
        # show the list of data with scroll bar       
        scrollbar = tk.Scrollbar(master=self.listframework, orient="vertical")
        scrollbar.grid(row=12, column=9,  sticky=tk.N+tk.S,padx=self.pad_val)

        listNodes = tk.Listbox(master=self.listframework, width=self.button_length*3, height=int(self.img_width/20),  font=("Times", 12), selectmode='single')
        listNodes.grid(row=12, column=5, columnspan=4, sticky=tk.N+tk.S,padx=self.pad_val)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<Double-1>', tracklist_on_select)

        scrollbar.config(command=listNodes.yview)
        
        #delete button
        
        deletbutton = tk.Button(master=self.resultbuttonframe, text="DELETE TRACK", command=detele_track_question, width=int(self.button_length*0.8),  bg='red')
        deletbutton.grid(row=13, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        # add button

        deletbutton = tk.Button(master=self.resultbuttonframe, text="ADD TRACK", command=new_track_question, width=int(self.button_length*0.8),  bg='green')
        deletbutton.grid(row=14, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val) 

        # duplicate button
        duplicatebutton = tk.Button(master=self.resultbuttonframe, text="DUPLICATE TRACK", command=duplicate_track_question, width=int(self.button_length*0.8),  bg='green')
        duplicatebutton.grid(row=15, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val)

       # plot the tracks from filtered folder 
        for p in self.track_data_filtered['tracks']:
            
            #calculate length and duration
            if len(p['trace'])>0:
                start_track_frame=p['frames'][0]
            else:
                start_track_frame=0
            
            # add to the list
            listNodes.insert(tk.END, "ID: "+str(p['trackID'])+" start frame: "+str(start_track_frame))        

            
            
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
            
            # create a none-membrane movie
            self.membrane_movie=np.ones(self.movie.shape)
            
             
            
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
            

    def select_membrane_movie(self):
        '''
        function for select membrane movie button
        '''
        
        filename = tk.filedialog.askopenfilename()
        if not filename:
            print("File was not selected")
        else:  
            # read files 
            self.membrane_movie=skimage.io.imread(filename)
            #normalise the membrane values
            self.membrane_movie=self.membrane_movie/np.max(self.membrane_movie)
            
    
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
            
            
            if self.track_file.endswith(".csv"):# read csv 
                # read file
    
                json_tracks={"tracks":[]}
                trackID_new=-1
                track={}
                with open(self.track_file, newline='') as f:
                    reader = csv.reader(f)
                    try:
                        for row in reader:
                            if row[0]!="TrackID":
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
                    except:
                        pass
                    self.track_data_original=json_tracks
    
                
            else: # read json in txt 
                       
                #read  the tracks data 
                with open(self.track_file) as json_file:  
        
                    self.track_data_original = json.load(json_file)
                    
                    
                # to save from dictionary to dict-list format:
                if 'tracks' not in self.track_data_original.keys():
                    
                    
                    self.track_data={'tracks':[]}
                    
                    for pos in self.track_data_original:
                        p=self.track_data_original[pos]
                        self.track_data['tracks'].append(p)
                        
                    self.track_data_original=self.track_data
                    
                
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
                self.stat_data.append(['','', 'Max travelled distance (nm): ', self.txt_length_from.get(),' - ',self.txt_length_to.get(),'','','','','','','' ]) 
                self.stat_data.append(['','','Mean curvilinear speed: moving (nm/sec): ', self.txt_speed_from.get(),' - ',self.txt_speed_to.get(),'','','','','','','' ]) 
                self.stat_data.append(['','','Zoom : ', 'x :',str(self.ylim_zoom),'  y:', str(self.xlim_zoom),'','','','','','' ]) 
                self.stat_data.append(['','',' ', '','','','','','','','','','' ]) 
                self.stat_data.append(['Track ID', 'Start frame', ' Total distance travelled (nm)',  'Net distance travelled (nm)', 
                                 ' Maximum distance travelled (nm)', ' Total trajectory time (sec)',  
                                 ' Net orientation (degree)', 'Mean curvilinear speed: average (nm/sec)', 'Mean straight-line speed: average (nm/sec)',
                                 'Mean curvilinear speed: moving (nm/sec)', 'Mean straight-line speed: moving (nm/sec)', 'Max curvilinear speed: moving (nm/sec)', 'Max curvilinear speed per segment: moving (nm/sec)'  ]) 
        
        
                print("Total number of tracks to process: ", len(self.track_data_filtered['tracks']))
                for trackID in range(0, len(self.track_data_filtered['tracks'])):
                    print(" track ", trackID+1)
                    track=self.track_data_filtered['tracks'][trackID]
                    trajectory=track['trace']
                    
                    x=np.asarray(trajectory)[:,0]    
                    y=np.asarray(trajectory)[:,1]
                    x_0=np.asarray(trajectory)[0,0]
                    y_0=np.asarray(trajectory)[0,1]
                    
                    x_e=np.asarray(trajectory)[-1,0]
                    y_e=np.asarray(trajectory)[-1,1]
                    
                    displacement_array=np.sqrt((x-x_0)**2+(y-y_0)**2)*self.img_resolution
                    #calculate all type of displacementsself.ax.get_ylim()
                    # max displacement
                    max_displacement=np.round(np.max(displacement_array),2)
                    
                    # displacement from start to the end
                    net_displacement=np.round(np.sqrt((x_e-x_0)**2+(y_e-y_0)**2),2)*self.img_resolution
                    
                    # total displacement
                    x_from=np.asarray(trajectory)[0:-1,0] 
                    y_from=np.asarray(trajectory)[0:-1,1] 
                    x_to=np.asarray(trajectory)[1:,0] 
                    y_to=np.asarray(trajectory)[1:,1] 
                    
                    #orientation
                    total_displacement=np.round(np.sum(np.sqrt((x_to-x_from)**2+(y_to-y_from)**2)),2)*self.img_resolution
                    pointB=trajectory[-1]                        
                    pointA=trajectory[0]    
                    
                    y=pointB[1]-pointA[1]
                    x=pointB[0]-pointA[0]
                    net_direction=int((math.degrees(math.atan2(y,x))+360-90-self.ap_axis)%360)
                    
                    # frames        
                    time=(track['frames'][-1]-track['frames'][0]+1)/self.frame_rate
                    
                    # speed 
                            
                    #evaluate motion 
                    track['motion']=self.motion_type_evaluate(track, traj_segm_switch_var=self.traj_segmentation_var)
                    average_mcs=np.round(self.tg.calculate_speed(track, "average")[0]*self.img_resolution*self.frame_rate,0)
                    average_msls=np.round(self.tg.calculate_speed(track, "average")[1]*self.img_resolution*self.frame_rate,0)
                                         
                    moving_speeds=self.tg.calculate_speed(track, "movement")
                    moving_mcs=np.round(moving_speeds[0]*self.img_resolution*self.frame_rate,0)
                    moving_msls=np.round(moving_speeds[1]*self.img_resolution*self.frame_rate,0)
                    moving_maxcs=np.round(moving_speeds[2]*self.img_resolution*self.frame_rate,0)
                    moving_maxsegcs=np.round(moving_speeds[3]*self.img_resolution*self.frame_rate,0)
                    
                    
                    
                    self.stat_data.append([track['trackID'], track['frames'][0], total_displacement ,net_displacement,
                                             max_displacement, time,
                                             net_direction, average_mcs, average_msls, moving_mcs, moving_msls, moving_maxcs, moving_maxsegcs])
                    
        
                if not(save_file.endswith(".csv")):
                        save_file += ".csv"
        
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
        
            
        self.newbutton = tk.Button(master=self.choose_traj_segmentation, text=" OK ", command=run_main_parameters_set, width=int(self.button_length/2),  bg='green')
        self.newbutton.grid(row=2, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        self.deletbutton = tk.Button(master=self.choose_traj_segmentation, text=" Cancel ", command=cancel_window, width=int(self.button_length/2),  bg='green')
        self.deletbutton.grid(row=2, column=2, columnspan=1, pady=self.pad_val, padx=self.pad_val)
        
  
                
        
############################################################

class TrackViewer(tk.Frame):
    '''
    class for the window of the individual tracks 
    '''
    def __init__(self, master, track_data, movie, membrane_movie, img_resolution, frame_rate, ap_axis, axis_name):
        tk.Frame.__init__(self, master)

        master.configure(background='white')
        
        self.viewer = master
        
        #set the window size        
        self.window_width = int(master.winfo_screenwidth()/2.5) # of the monitor width
        self.window_height = int(master.winfo_screenheight()*0.7)  # of the monitor height

        
        # save important data
        self.track_data=track_data
        self.movie=movie
        self.membrane_movie=membrane_movie
        self.frames=track_data['frames']
        self.motion=track_data['motion']
        self.trace=track_data['trace']
        self.id=track_data['trackID']
        self.frame_pos=track_data['frames'][0]

        self.frame_pos_to_change=0 # frame which can be changed
        self.movie_length=self.movie.shape[0] # movie length
        self.plot_switch=0 # switch between plotting/not plotting tracks
        self.img_resolution=img_resolution # resolution of the movie
        self.frame_rate=frame_rate # movie frame rate
        self.ap_axis=ap_axis
        self.axis_name=axis_name
        
        # segmentation 
        self.traj_segm_switch_var=0 # calculate and show motion type
        self.tg = TrajectorySegment()     
        self.tg.window_length=8            
        #update motion information
        self.motion=self.motion_type_evaluate(self.track_data)
        self.track_data['motion']=self.motion 
        
        self.pixN_basic=100 # margin size 
        self.vesicle_patch_size=10
        
        self.membrane_switch=0 # switch between membrane and no membrane
        self.traj_segm_switch_var=0 # calculate and show motion type
        
        #track evaluation 
        self.displacement_array=[]
        self.max_displacement=0
        self.net_displacement=0
        self.total_distance=0
        
        # change the name to add track ID
        master.title("TrackViewer: track ID "+str(self.id))
        
        # placing sizes
        self.button_length=np.max((10,int(self.window_width/70)))
        self.pad_val=2
        self.dpi=100
        self.img_width=int(self.window_height*0.6)
        self.figsize_value=(int(self.img_width/self.dpi), int(self.img_width/self.dpi*0.75))
        
        
     # # #  build layout of the frame
        self.show_list()   
        
        # movie control
        self.plot_image()
        
        # plot displacement
        
        self.plot_displacement()
        
        # plot intensity graph
        self.intensity_calculation()
        
        # plot parameters
        self.show_parameters()
        

        # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.plot_image() 
                   
        self.scale_movie = tk.Scale(master=self.viewer, from_=0, to=self.movie_length, tickinterval=100, length=int(self.img_width), width=5, orient="horizontal", command=show_values)
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
        
        def update_traj_segm_switch():            
            self.traj_segm_switch_var=var_traj_segm_switch.get()
            
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
                
          
    # # # # # #  Radiobutton : membrane on/off # # # # # # #   
        var_membrane = tk.IntVar()
        
        def update_membrane_switch():            
            self.membrane_switch=var_membrane.get()
            # change image
            self.plot_image()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(master=self.viewer, text="without membrane", variable=var_membrane, value=0, bg='white', command =update_membrane_switch )
        self.M1.grid(row=0, column=1, columnspan=1, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(master=self.viewer, text=" with membrane ", variable=var_membrane, value=1, bg='white',command = update_membrane_switch ) #  command=sel)
        self.M2.grid(row=0, column=2, columnspan=2, pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(master=self.viewer, text=" with border ", variable=var_membrane, value=2, bg='white',command = update_membrane_switch ) #  command=sel)
        self.M3.grid(row=0, column=4, pady=self.pad_val, padx=self.pad_val)

    # # # # # #  Radiobutton : tracks on/off/motion # # # # # # #          
    # plotting switch 
        var = tk.IntVar()
        def update_monitor_plot():            
            self.plot_switch=var.get()
            self.plot_image()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(master=self.viewer, text=" tracks on  ", variable=var, value=0, bg='white', command =update_monitor_plot )
        self.R1.grid(row=1, column=1, columnspan=1,  pady=self.pad_val, padx=self.pad_val)  

        self.R2 = tk.Radiobutton(master=self.viewer, text=" tracks off ", variable=var, value=1, bg='white',command = update_monitor_plot ) #  command=sel)
        self.R2.grid(row=1, column=2, columnspan=2,  pady=self.pad_val, padx=self.pad_val)
        
        self.R3 = tk.Radiobutton(master=self.viewer, text=" motion type ", variable=var, value=2, bg='white',command = update_monitor_plot ) #  command=sel)
        self.R3.grid(row=1, column=4, columnspan=1,  pady=self.pad_val, padx=self.pad_val)
    
    
    def calculate_direction(self, trace):
        '''
        calculate average angle of the direction
        '''
        pointB=trace[-1]                        
        pointA=trace[0]
        changeInX = pointB[0] - pointA[0]
        changeInY = pointB[1] - pointA[1]
        
        return int((math.degrees(math.atan2(changeInY,changeInX))+360-90-self.ap_axis)%360)

        
    def change_position(self):
        '''
        correct position coordinate values
        
        '''
        
        self.action_cancel()
        
        self.correct_position_window = tk.Toplevel(root, bg='white')
        self.correct_position_window.title(" Correct coordinates ")

        self.lbframechange = tk.Label(master=self.correct_position_window, text="Make changes in frame: "+str(self.frames[self.frame_pos_to_change]), bg='white')
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
        
        self.trace[self.frame_pos_to_change]=[float(self.txt_position.get().split(',')[0]), float(self.txt_position.get().split(',')[1])]
        
        
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
        
        self.lbframechange = tk.Label(master=self.delete_position_window, text="   Do you want to delete frame "+str(self.frames[self.frame_pos_to_change])+" ?   ", bg='white')
        self.lbframechange.grid(row=0, column=10, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)              
        

        self.buttonOKdel= tk.Button(master=self.delete_position_window,text=" apply ", command=self.action_apply_delete, width=int(self.button_length/2))
        self.buttonOKdel.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val)  
        
        self.button_cancel= tk.Button(master=self.delete_position_window,text=" cancel ", command=self.action_cancel, width=int(self.button_length/2))
        self.button_cancel.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)     
        
    def action_apply_delete(self):
        '''
        delete selected position
        
        '''

        del self.trace[self.frame_pos_to_change] 
        del self.frames[self.frame_pos_to_change] 
        
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
        
    def add_position(self): 
        '''
        add new position with frame number with coordinates - question window
        
        '''
        
        self.action_cancel()   

        
        # open new window
       
        self.add_position_window = tk.Toplevel( bg='white')
        self.add_position_window.title(" Create new ")
        self.lbframechange = tk.Label(master=self.add_position_window, text=" Add frame: ", bg='white')
        self.lbframechange.grid(row=0, column=10, pady=self.pad_val, padx=self.pad_val)

        self.txt_frame = tk.Entry(self.add_position_window, width=int(self.button_length))
        self.txt_frame.grid(row=0, column=11)                
        

        self.lbpose = tk.Label(master=self.add_position_window, text=" new coordinates: x, y ", bg='white')
        self.lbpose.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val)  
        
        self.txt_position_coordinates = tk.Entry(self.add_position_window, width=int(self.button_length*2))
        self.txt_position_coordinates.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)                
        

        self.buttonOK_add= tk.Button(master=self.add_position_window,text=" apply ", command=self.action_apply_add, width=int(self.button_length/2))
        self.buttonOK_add.grid(row=2, column=10, pady=self.pad_val, padx=self.pad_val)   

        self.button_cancel= tk.Button(master=self.add_position_window,text=" cancel ", command=self.action_cancel, width=int(self.button_length/2))
        self.button_cancel.grid(row=2, column=11, pady=self.pad_val, padx=self.pad_val)     

        
    def action_apply_add(self):
        '''
        create the position with given parameters
        
        '''
        
        # get new location
        location_val=[float(self.txt_position_coordinates.get().split(',')[0]), float(self.txt_position_coordinates.get().split(',')[1])]
        frame_val=int(self.txt_frame.get())
        
        # where to insert the postion
        if frame_val<self.frames[0]: # at the start
            pos=0
        
        elif frame_val>self.frames[-1]: # at the end
            pos=len(self.frames)+1
        else: #somewhere in the middle
            diff_array=np.asarray(self.frames)-frame_val
            diff_array_abs=abs(diff_array)
            val=min(abs(diff_array_abs))
            if min(diff_array)>0:
                pos=diff_array_abs.tolist().index(val)
            elif min(diff_array)<=0:
                pos=diff_array_abs.tolist().index(val)+1

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
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.plot_image() 
        self.scale_movie.set(self.frame_pos)            
    
    
    def plot_image(self):
        '''
        arrange the image viewer
        
        '''

        fig = plt.figure(figsize=(int(self.img_width/self.dpi), int(self.img_width/self.dpi)))
        plt.axis('off')
        fig.tight_layout()
        
        if self.membrane_switch==0:
            img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        elif self.membrane_switch==1:
            img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])+0.1*self.membrane_movie[self.frame_pos,:,:]
        else:
            img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])


        #calculate window position        
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
            y_max=np.min([y_min+self.pixN_basic, (img.shape[1]-1)])  
        elif left_point_y>=0 and right_point_y>=img.shape[1]:
            y_max=img.shape[1]-1
            y_min=np.max([0, y_max-self.pixN_basic])
        else:
            y_min=0
            y_max=img.shape[1]-1
            
        # for x
        
        if top_point_x>=0 and bottom_point_x<img.shape[0]:
            x_min=top_point_x
            x_max=bottom_point_x
        elif top_point_x<0 and bottom_point_x<img.shape[0]:
            x_min=0
            x_max=np.min([x_min+self.pixN_basic, (img.shape[0]-1)])  
        elif top_point_x>=0 and bottom_point_x>=img.shape[0]:
            x_max=img.shape[0]-1
            x_min=np.max([0, x_max-self.pixN_basic])
        else:
            x_min=0
            x_max=img.shape[0]-1       
            
        region=img[x_min:x_max, y_min:y_max]
        
        blue_c=np.linspace(0., 1., len(self.trace))
        red_c=1-np.linspace(0., 1., len(self.trace))
    
        self.im = plt.imshow(region, cmap="gray")
        
        if self.plot_switch==0: # print full trajectory
            for pos in range(0, len(self.trace)-1):
                plt.plot(np.asarray(self.trace)[pos:pos+2,1]- y_min,np.asarray(self.trace)[pos:pos+2,0]-x_min,  color=(red_c[pos],0,blue_c[pos]))
        
            plt.text(np.asarray(self.trace)[-1,1]- y_min,np.asarray(self.trace)[-1,0]- x_min, "  END  ", fontsize=16, color="b")
            plt.plot(np.asarray(self.trace)[-1,1]- y_min,np.asarray(self.trace)[-1,0]- x_min,  "bo",)  
            
            plt.text(np.asarray(self.trace)[0,1]- y_min,np.asarray(self.trace)[0,0]- x_min, "  START  ", fontsize=16, color="r")
            
            plt.plot(np.asarray(self.trace)[0,1]- y_min,np.asarray(self.trace)[0,0]- x_min,  "ro",)
            
        elif self.plot_switch==2 and self.traj_segm_switch_var>0: # plotting motion type
            #define colour
            red_c= (abs(np.array(self.motion)-1)).tolist()
            green_c= self.motion
            for pos in range(1, len(self.trace)):
                plt.plot(np.asarray(self.trace)[pos-1:pos+1,1]- y_min,np.asarray(self.trace)[pos-1:pos+1,0]-x_min,  color=(red_c[pos],green_c[pos],0))
            
        # on click -> get coordinates        
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
            
            update_position_text(float(event.ydata), float(event.xdata))
        
        #plot the border of the membrane if chosen
        if self.membrane_switch==2:
            #extract skeleton 
            skeleton = skimage.morphology.skeletonize(self.membrane_movie[self.frame_pos,x_min:x_max, y_min:y_max]).astype(np.int)
            # create an individual cmap with red colour
            cmap_new = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['red','red'],256)
            cmap_new._init() # create the _lut array, with rgba values
            alphas = np.linspace(0, 0.8, cmap_new.N+3)
            cmap_new._lut[:,-1] = alphas
            #plot the membrane border on the top
            plt.imshow(skeleton, interpolation='nearest', cmap=cmap_new)     
        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=self.viewer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=4, column=1, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        canvas.mpl_connect('button_press_event', onclick)

    def show_parameters(self): 
        '''
        calculate and print trajectory parameters
        
        '''

        # show the list of data with scroll bar
               
        listNodes_parameters = tk.Listbox(master=self.viewer, width=int(self.button_length*3),  font=("Times", 10), selectmode='single')
        listNodes_parameters.grid(row=6, column=1,  columnspan=4, sticky=tk.N+tk.S, pady=self.pad_val, padx=self.pad_val)

       # add to the list
        listNodes_parameters.insert(tk.END, " Total distance travelled                    "+str(np.round(self.total_distance*self.img_resolution,2))+" nm") 

        listNodes_parameters.insert(tk.END, " Net distance travelled                      "+str(np.round(self.net_displacement*self.img_resolution,2))+" nm")  

        listNodes_parameters.insert(tk.END, " Maximum distance travelled                  "+str(np.round(self.max_displacement*self.img_resolution,2))+" nm")
        
        listNodes_parameters.insert(tk.END, " Total trajectory time                       "+str(np.round((self.frames[-1]-self.frames[0]+1)/self.frame_rate,5))+" sec")

        listNodes_parameters.insert(tk.END, " Net orientation                             "+str(self.calculate_direction(self.trace))+ " degrees")

        listNodes_parameters.insert(tk.END, " Mean curvilinear speed: average             "+str(np.round(self.tg.calculate_speed( self.track_data, "average")[0]*self.img_resolution*self.frame_rate,0))+" nm/sec")
 
        listNodes_parameters.insert(tk.END, " Mean straight-line speed: average           "+str(np.round(self.tg.calculate_speed( self.track_data, "average")[1]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Mean curvilinear speed: moving              "+str(np.round(self.tg.calculate_speed( self.track_data, "movement")[0]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Mean straight-line speed: moving            "+str(np.round(self.tg.calculate_speed( self.track_data, "movement")[1]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Max curvilinear speed: moving               "+str(np.round(self.tg.calculate_speed( self.track_data, "movement")[2]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Max curvilinear speed per segment : moving  "+str(np.round(self.tg.calculate_speed( self.track_data, "movement")[3]*self.img_resolution*self.frame_rate,0))+" nm/sec")
      

        
    def show_list(self): 
        '''
        arrangement for the position list
        
        '''
        
        def tracklist_on_select(even):
            try:
                self.frame_pos_to_change=listNodes.curselection()[0]
            except:
                pass

                # show the list of data with scroll bar
        lbend = tk.Label(master=self.viewer, text="LIST OF DETECTIONS:  ",  bg='white', font=("Times", 12))
        lbend.grid(row=1, column=5, columnspan=3, pady=self.pad_val, padx=self.pad_val)
        
        scrollbar = tk.Scrollbar(master=self.viewer, orient="vertical")
        scrollbar.grid(row=2, column=8, rowspan=5,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=self.viewer, width=int(self.button_length*3), font=("Times", 10), selectmode='single')
        listNodes.grid(row=2, column=5, columnspan=3, rowspan=5 , sticky=tk.N+tk.S, pady=self.pad_val)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<<ListboxSelect>>', tracklist_on_select)
        scrollbar.config(command=listNodes.yview)
        
       # plot the track positions
        for i in range(0, len(self.frames)):
             # add to the list
            listNodes.insert(tk.END, str(self.frames[i])+":  "+str(np.round(self.trace[i], 4)))     





        
    def intensity_calculation(self):
        '''
        Calculates changes in intersity for the given track
        and plot it into a canva
        '''
        trace=self.trace
        frames=self.frames
        patch_size=self.vesicle_patch_size
        
        def img_segmentation(img_segment, int_size, box_size):
            '''
            the function segments the image based on the thresholding and watershed segmentation
            the only center part of the segmented part is taked into account.
            '''
    
        # calculate threshold based on the centre
            threshold=np.mean(img_segment[int(box_size/2-int_size):int(box_size/2+int_size), int(box_size/2-int_size):int(box_size/2+int_size)])
        #    thresholding to get the mask
            mask=np.zeros(np.shape(img_segment))
            mask[img_segment>threshold]=1
        
            # separate the objects in image
        ## Generate the markers as local maxima of the distance to the background
            distance = sp.ndimage.distance_transform_edt(mask)
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
            markers = sp.ndimage.label(local_maxi)[0]
        
            # segment the mask
            segment = skimage.morphology.watershed(-distance, markers, mask=mask)
           
        # save the segment which is only in the centre
            val=segment[int(box_size/2), int(box_size/2)]
            segment[segment!=val]=0
            segment[segment==val]=1
    
            return segment
        
        #extract images
        track_img=np.zeros((len(trace),patch_size,patch_size))
        
        intensity_array_1=[]
        intensity_array_2=[]
        
        check_border=0 
        for N in range(0,len(frames)):
            frameN=frames[N]
            point=trace[N]
            x_min=int(point[0]-patch_size/2)
            x_max=int(point[0]+patch_size/2)
            y_min=int(point[1]-patch_size/2)
            y_max=int(point[1]+patch_size/2)
            
            if x_min>0 and y_min>0 and x_max<self.movie.shape[1] and y_max<self.movie.shape[2]:
                
                # create img
                track_img[N,:,:]= self.movie[frameN, x_min:x_max, y_min:y_max]
                
                #segment img
                int_size=5
                segmented_vesicle=img_segmentation(track_img[N,:,:]/np.max(track_img[N,:,:]), int_size, patch_size)
                
                #calculate mean intensity inside the segment                
                intensity_1=np.sum(track_img[N,:,:]*segmented_vesicle)/np.sum(segmented_vesicle)
                intensity_2=np.sum(track_img[N,:,:])/(patch_size*patch_size)
                intensity_array_1.append(intensity_1)
                intensity_array_2.append(intensity_2)
            else:
                check_border=1
                intensity_array_1.append(0)
                intensity_array_2.append(0)
        
        # plotting
        fig1 = plt.figure(figsize=self.figsize_value)
        plt.plot(frames, (intensity_array_1-np.min(intensity_array_1))/(np.max(intensity_array_1)-np.min(intensity_array_1)), "-b", label="segmented vesicle")

        plt.ylabel("intensity", fontsize='small')
        plt.plot(frames, (intensity_array_2-np.min(intensity_array_2))/(np.max(intensity_array_2)-np.min(intensity_array_2)), "-k", label="without segmentation")
        if check_border==0:
            plt.title('Vesicle intensity per frame', fontsize='small')
        else:
            plt.title('Vesicle intensity: fail to compute for all frames!', fontsize='small')
        plt.legend(fontsize='small')   
        plt.ylim(top = 1.1, bottom = 0)            
        
        # DrawingArea
        canvas1 = FigureCanvasTkAgg(fig1, master=self.viewer)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=5, column=9, columnspan=4, rowspan=3, pady=self.pad_val, padx=self.pad_val)  
                     
        
    def plot_displacement(self):
        '''
        displacement plot
        
        '''
        trajectory=self.trace
        
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
        
        fig = plt.figure(figsize=self.figsize_value)
        
        disaplcement=self.displacement_array*self.img_resolution
        
        # plot dosplacement colour
        for i in range(1, len(self.motion)):
            if self.motion[i]==0:
                colourV='r'
            else:
                colourV='g'
            plt.plot((self.frames[i-1],self.frames[i]), (disaplcement[i-1],disaplcement[i]), colourV)
            
        plt.ylabel('Displacement (nm)', fontsize='small')

        plt.title('Displacement per frame', fontsize='small')
        
        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=self.viewer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=9, columnspan=4, rowspan=2, pady=self.pad_val, padx=self.pad_val)   

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
        parent.title("MSP-viewer 0.1")
        parent.configure(background='white')
        
        self.main = MainVisual(parent)

        parent.protocol('WM_DELETE_WINDOW', self.close_app)

        tk.mainloop()

    def close_app(self):
        self.quit()
        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
