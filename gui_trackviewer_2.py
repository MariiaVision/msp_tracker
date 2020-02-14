#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:41:16 2019

@author: maria dmitrieva
"""
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

class MainVisual(tk.Frame):
    # choose the files and visualise the tracks on the data
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        
        #colours for plotting tracks
        
        self.color_list_plot=["#00FFFF", "#7FFFD4", "#0000FF", "#8A2BE2", "#7FFF00", "#D2691E", "#FF7F50", "#DC143C",
            "#008B8B", "#8B008B", "#FF8C00", "#E9967A", "#FF1493", "#9400D3", "#FF00FF", "#B22222",
            "#FFD700", "#ADFF2F", "#FF69B4", "#ADD8E6", "#F08080", "#90EE90", "#20B2AA", "#C71585", "#FF00FF"]
            
        self.color_list=[(200, 0, 0), (0, 255, 0), (0, 0, 255), (200, 155, 0),
                    (100, 255, 5), (255, 10, 120), (255, 127, 255),
                    (127, 0, 255), (200, 0, 127), (177, 0, 20), (12, 200, 0), (0, 114, 255), (255, 20, 0),
                    (0, 255, 255), (255, 100, 100), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
        
        self.movie_file=" " # path to the move file
        self.track_file=" "# path to the file with tracking data (json format)
        self.movie=np.ones((1,200,200)) # matrix with data
        self.membrane_movie=[]
        self.track_data_original={}
        self.track_data={'tracks':[]} # original tracking data
        self.track_data_filtered={'tracks':[]}  # filtered tracking data  
        self.track_data_framed={}  # tracking data arranged by frames  
        self.stat_data=[] # stat data to save csv file
        
        # segmentation 
        self.tg = TrajectorySegment()     
        self.tg.window_length=10

        #filter parameters
        self.filter_duration=[0, 1000]
        self.filter_length=[0, 10000]   
        self.filter_stop=[0, 10000] 
        
        self.frame_pos=0
        self.movie_length=1
        self.monitor_switch=0 # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.memebrane_switch=0 # 0 - don't show the membrane, 1-show the membrane
        self.pad_val=1
        
        self.img_resolution=100 # resolution of the image, default is 100 nm/pix 
        self.frame_rate=4 # frame rate, default is 4 f/s
        # 
        self.figsize_value=(4,4) # image sizeS
        # 
        self.deleted_tracks_N=0
        self.created_tracks_N=0
        self.filtered_tracks_N=0
        
        self.max_movement_stay=1.0
        self.ap_axis=90
     # # # # # # menu to choose files and print data # # # # # #
        
        self.button_mv = tk.Button(text="   Select vesicle movie   ", command=self.select_vesicle_movie, width=20)
        self.button_mv.grid(row=0, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val)

        self.button_mm = tk.Button(text="   Select membrane movie   ", command=self.select_membrane_movie, width=20)
        self.button_mm.grid(row=0, column=2, columnspan=2,pady=self.pad_val, padx=self.pad_val)
        
        self.button2 = tk.Button(text="Select file with tracks", command=self.select_track, width=30)
        self.button2.grid(row=1, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        #update the list
        self.list_update()  
        
#        self.buttonShow = tk.Button(text="Show tracks", command=self.show_tracks, width=30)
#        self.buttonShow.grid(row=2, column=2, pady=self.pad_val, padx=self.pad_val)  

#    # # # # # # filter choice:membrane on/off # # # # # # #   
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
#    # # # # # # filter choice:tracks # # # # # # #   
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
        
        # resolution in time and space    
            
        res_lb = tk.Label(master=root, text=" resolution (nm/pix) : ", width=20, bg='white')
        res_lb.grid(row=5, column=0, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.img_resolution))
        self.res_parameter = tk.Entry(root, width=10, text=v)
        self.res_parameter.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val)
            
        lbl3 = tk.Label(master=root, text="frame rate (f/sec) : ", width=20, bg='white')
        lbl3.grid(row=5, column=2, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.frame_rate))
        self.frame_parameter = tk.Entry(root, width=10, text=v)
        self.frame_parameter.grid(row=5, column=3, pady=self.pad_val, padx=self.pad_val)        
        
            
        # AP axis 
        ap_lb = tk.Label(master=root, text=" AP axis: ", width=20, bg='white')
        ap_lb.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.ap_axis))
        self.ap_parameter = tk.Entry(root, width=10, text=v)
        self.ap_parameter.grid(row=6, column=1, pady=self.pad_val, padx=self.pad_val)
            
        lbl3 = tk.Label(master=root, text="frame rate (f/sec) : ", width=20, bg='white')
        lbl3.grid(row=5, column=2, pady=self.pad_val, padx=self.pad_val)
        v = tk.StringVar(root, value=str(self.frame_rate))
        self.frame_parameter = tk.Entry(root, width=10, text=v)
        self.frame_parameter.grid(row=5, column=3, pady=self.pad_val, padx=self.pad_val)   
       # list switchL # 0 - all, 1 

        # trackID
        lbl3 = tk.Label(master=root, text="Track ID: ", width=20, bg='white')
        lbl3.grid(row=0, column=5, pady=self.pad_val, padx=self.pad_val)
        self.txt_track_number = tk.Entry(root, width=10)
        self.txt_track_number.grid(row=0, column=6, pady=self.pad_val, padx=self.pad_val)

        # duration
        lbl3 = tk.Label(master=root, text="Duration (sec): from ", width=40, bg='white')
        lbl3.grid(row=1, column=5, pady=self.pad_val, padx=self.pad_val)
        self.txt_duration_from = tk.Entry(root, width=10)
        self.txt_duration_from.grid(row=1, column=6, pady=self.pad_val, padx=self.pad_val)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=1, column=7, pady=self.pad_val, padx=self.pad_val)
        self.txt_duration_to = tk.Entry(root, width=10)
        self.txt_duration_to.grid(row=1, column=8, pady=self.pad_val, padx=self.pad_val)


        # Length       
        lbl3 = tk.Label(master=root, text="Max travelled distance (nm): from ", width=40, bg='white')
        lbl3.grid(row=2, column=5)
        self.txt_length_from = tk.Entry(root, width=10)
        self.txt_length_from.grid(row=2, column=6, pady=self.pad_val, padx=self.pad_val)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=2, column=7, pady=self.pad_val, padx=self.pad_val)
        self.txt_length_to = tk.Entry(root, width=10)
        self.txt_length_to.grid(row=2, column=8, pady=self.pad_val, padx=self.pad_val)     
        
        # Stop duration
        lbl3 = tk.Label(master=root, text="Final stop duration (sec): from ", width=40, bg='white')
        lbl3.grid(row=3, column=5)
        self.txt_stop_from = tk.Entry(root, width=10)
        self.txt_stop_from.grid(row=3, column=6, pady=self.pad_val, padx=self.pad_val)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=3, column=7, pady=self.pad_val, padx=self.pad_val)
        self.txt_stop_to = tk.Entry(root, width=10)
        self.txt_stop_to.grid(row=3, column=8, pady=self.pad_val, padx=self.pad_val)    
        
        # Stop tolerance set
        lbl3 = tk.Label(master=root, text="Stop position tolerance (nm): ", width=40, bg='white')
        lbl3.grid(row=4, column=5)
        v = tk.StringVar(root, value=str(self.max_movement_stay*self.img_resolution))
        self.txt_stop_tolerance = tk.Entry(root, width=10, text=v)
        self.txt_stop_tolerance.grid(row=4, column=6, pady=self.pad_val, padx=self.pad_val)  

        # Fusion events
        lbl3 = tk.Label(master=root, text="Fusion: distance to membrane (nm): ", width=40, bg='white')
        lbl3.grid(row=5, column=5)
        v = tk.StringVar(root, value=str(0))
        self.fusion_distance = tk.Entry(root, width=10, text=v)
        self.fusion_distance.grid(row=5, column=6, pady=self.pad_val, padx=self.pad_val)   

        
        # button to filter
        
        self.buttonFilter = tk.Button(text="Filter", command=self.filtering, width=10)
        self.buttonFilter.grid(row=6, column=4, columnspan=2, pady=self.pad_val, padx=self.pad_val)  
        
        # count membrane crossing
        self.buttonFilter = tk.Button(text="Crossing membrane", command=self.crossing_membrane, width=20)
        self.buttonFilter.grid(row=6, column=6, columnspan=1,  pady=self.pad_val, padx=self.pad_val)             
        
        # fusion events and statistics
        
        self.buttonFilter = tk.Button(text="Fusion events", command=self.find_fusion, width=20)
        self.buttonFilter.grid(row=6, column=7, columnspan=2,  pady=self.pad_val, padx=self.pad_val)           
        
  
        
        # button to update changes
        
        button_save=tk.Button(master=root, text="update", command=self.update_data, width=14)
        button_save.grid(row=4, rowspan=2, column=7, pady=self.pad_val, padx=self.pad_val)


        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

        # motion map 

        button_save=tk.Button(master=root, text="motion map", command=self.plot_motion_map, width=14)
        button_save.grid(row=13, column=7, pady=self.pad_val, padx=self.pad_val)
        
        # button to save all the tracks on the image
        
        button_save=tk.Button(master=root, text="save tracks plot", command=self.save_track_plot, width=14)
        button_save.grid(row=14, column=7, pady=self.pad_val, padx=self.pad_val)
        
        # button to update changes
        
        button_save=tk.Button(master=root, text="save movie", command=self.save_movie, width=12)
        button_save.grid(row=13, column=8, pady=self.pad_val, padx=self.pad_val)
        
        # save button
     
        button_save=tk.Button(master=root, text="tracks to json", command=self.save_in_file, width=12)
        button_save.grid(row=16, column=8, pady=self.pad_val, padx=self.pad_val)        

        button_save=tk.Button(master=root, text="info to csv", command=self.save_data_csv, width=12)
        button_save.grid(row=14, column=8, pady=self.pad_val, padx=self.pad_val)  
        
      # # # # # # movie  # # # # # # 
        
        # plot bg
        self.fig, self.ax = plt.subplots(1,1, figsize=self.figsize_value)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        self.show_tracks() 
        
    
   #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_tracks() 
          
        self.scale_movie = tk.Scale(root, from_=0, to=self.movie_length, tickinterval=100, length=400, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=13, column=1, columnspan=3,rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=13, column=0, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 

        
        buttonnext = tk.Button(text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=13, column=4, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)        
    
    def find_fusion(self):
        '''
        fusion event detection based on last frame position
        '''
        print("fusion event detection ....")
        

#        membrane_movie_deliated=sp.ndimage.binary_dilation(self.membrane_movie, iterations=1)
        
        # detect events
        event_count=FusionEvent(self.movie, self.movie, self.membrane_movie, self.track_data_filtered)
        event_count.max_movement_stay=self.max_movement_stay
        
        # assign the distance value
        if self.fusion_distance.get()=='':
            event_count.distance_to_membrane==0
        else:
            event_count.distance_to_membrane=float(self.fusion_distance.get())/self.img_resolution
            
        self.track_data_filtered=event_count.find_fusion()
        
        # DrawingArea
        novi = tk.Toplevel()
        canvas = tk.Canvas(novi, width = 640, height = 480)
        canvas.pack(expand = tk.YES, fill = tk.BOTH)
        gif1 = tk.PhotoImage(file = 'subplot.png')
                                    #image not visual
        canvas.create_image(0,0, image = gif1, anchor = tk.NW)
        #assigned the gif1 to the canvas object
        canvas.gif1 = gif1
        
        # update the view 
        self.track_to_frame()
        
        #update the list
        self.list_update()   
        
        
    def crossing_membrane(self):
        '''
        find trajectories which crossing membrane and travelling between cells
        '''
        crossing_membrane=[]
        # create map
        membrane=1-self.membrane_movie[self.frame_pos,:,:]
        cells_labeled=sp.ndimage.label(membrane)[0]
        regions=np.unique(np.asarray(cells_labeled))

        # iterate over the track list
        for trackID in range(0, len(self.track_data_filtered['tracks'])):
            track=self.track_data_filtered['tracks'][trackID]
            
            # extract trajectory affiliation
            region_list=[]
            on_membrane=[]
            
            for pos in track['trace']:

                region_list.append(cells_labeled[pos[0], pos[1]])
                on_membrane.append(membrane[pos[0], pos[1]])
            

            # check if 3 different area is there
            regions=np.unique(np.asarray(region_list))
            number_of_regions=len(regions)
            
            
            #case 1: vesicle travels between cells and not stop on membrane
            case_1= number_of_regions==2 and np.all(on_membrane)!=0
            
            #case 2: vesicle travels between cells 
            case_2=number_of_regions>=3

            
            if case_1 or case_2:
                crossing_membrane.append(track)     
                
        # replace the filtered list
        self.track_data_filtered={'tracks': crossing_membrane}
        
        # update the view 
        self.track_to_frame()
        
        #update the list
        self.list_update()           
        
    def save_track_plot(self):
        '''
        plot and save the plot of all the tracks on a single frame
        '''
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
            cmap_new._init() # create the _lut array, with rgba values
            alphas = np.linspace(0, 0.8, cmap_new.N+3)
            cmap_new._lut[:,-1] = alphas
            #plot the membrane border on the top
            plt.imshow(skeleton, interpolation='nearest', cmap=cmap_new)        
        
        # request file name name
        save_file = tk.filedialog.asksaveasfilename()      
        
        # save the image
        plt.savefig(save_file)
        # close the image
        plt.close()     
        
    def plot_motion_map(self):
        '''
        plot motion map woth given AP
        '''
        def from_cartesian_to_polar(x,y):
            r=math.sqrt(x**2+y**2)
            angle=math.degrees(math.atan2(y,x))
            return r, (angle+360)%360
        orintation_array=[]
        orientation_array_length_related=[]
        
        fig = plt.figure(figsize=(15,6))
        plt.axis('off')
        ax = fig.add_subplot(121)
        ax.imshow(self.movie[1,:,:]/np.max(self.movie[1,:,:])+self.membrane_movie[1,:,:]/np.max(self.membrane_movie[1,:,:])*0.6, cmap='bone') 
               
        for trackID in range(0, len(self.track_data_filtered['tracks'])):
            track=self.track_data_filtered['tracks'][trackID]
        #    calculate parameters
            point_start=track['trace'][0]
            point_end=track['trace'][-1]

            # calculate orientation
            r, orintation_move=from_cartesian_to_polar(point_end[0]-point_start[0], point_end[1]-point_start[1])
            
            for pos in range(int(r*100)):
                orientation_array_length_related.append(orintation_move)
            
            orintation_array.append(orintation_move)
   
            color='r'
            plt.arrow(point_start[1],point_start[0], point_end[1]-point_start[1], point_end[0]-point_start[0], head_width=3.00, head_length=2.0, 
                      fc=color, ec=color, length_includes_head = True)
        
        bin_size=10
        a , b=np.histogram(orintation_array, bins=np.arange(0, 360+bin_size, bin_size))
        centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])   
        
        ax = fig.add_subplot(122, projection='polar')
#        plt.xticks(np.radians(range(0, 30, 180)),['0', '30', '60', '90', '120', '150', '180'])
        #read ap axis
        if self.ap_parameter.get()!='':
            self.ap_axis=int(self.ap_parameter.get())

        plt.xticks(np.radians((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330,  (self.ap_axis+90)%360, (self.ap_axis+270)%360)),
                   ['270', '300', '330', '0' , '30', '60','90', '120', '150', '180', '210', '240', '\n \n AP', '\n AP'])
        ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
        ax.set_theta_zero_location("S")
        ax.set_theta_direction(1)
        ax.set_title(" movement orientation \n based on track count ")
        
#        plt.show()
        
        # request file name
        save_file = tk.filedialog.asksaveasfilename() 

        if not(save_file.endswith(".png")):
            save_file += ".png"        
#        filename='/home/mariaa/NANOSCOPY/VESICLE_TRACKING/tracking_result/spinning_disk/movement_map/movement_map_'+file_name+'_cell_'+str(cellN)+'.png'
        plt.savefig(save_file) 
        
        
        
    def save_movie(self):
        length=self.movie.shape[0]
#        final_img_set = np.zeros((length, self.movie.shape[1], self.movie.shape[2], 3))
        
        # request file name
        save_file = tk.filedialog.asksaveasfilename() 
        
        if self.movie.shape[1]<300 and self.movie.shape[2]<300 or self.movie.shape[2]<200:
            #save tiff file for small cell solution
            final_img_set=np.zeros((self.movie.shape[0],self.movie.shape[1],self.movie.shape[2], 3))

            for frameN in range(0, length):
          
                plot_info=self.track_data_framed['frames'][frameN]['tracks']
                frame_img=self.movie[frameN,:,:]
                membrane_img=self.membrane_movie[frameN,:,:]
                # Make a colour image frame
                orig_frame = np.zeros((self.movie.shape[1], self.movie.shape[2], 3))
        
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
                final_img_set[frameN,:,:,:]=orig_frame

            #save the file
            final_img_set=final_img_set/np.max(final_img_set)*255
            final_img_set=final_img_set.astype('uint8')
            
            if not(save_file.endswith(".tif") or save_file.endswith(".tiff")):
                save_file += ".tif"
                
            imageio.volwrite(save_file, final_img_set)
        else:
            #save avi file for large movies

            if not(save_file.endswith(".avi")):
                save_file += ".avi"

    #        cv2.VideoWriter_fourcc(*'XVID')
    #        cv2.VideoWriter_fourcc('M','J','P','G')
            out = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), 4.0, (self.movie.shape[1], self.movie.shape[2]))
    
            for frameN in range(0, length):
          
                plot_info=self.track_data_framed['frames'][frameN]['tracks']
                frame_img=self.movie[frameN,:,:]
                membrane_img=self.membrane_movie[frameN,:,:]
                # Make a colour image frame
                orig_frame = np.zeros((self.movie.shape[1], self.movie.shape[2], 3))
        
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
        
        save_filename = tk.filedialog.asksaveasfilename()
        with open(save_filename+'.txt', 'w') as f:
            json.dump(self.track_data_filtered, f, ensure_ascii=False) 
        print("tracks are saved in  ", save_filename, " file")

    
    def update_data(self):
        '''
        update changed parameters
        '''
        
        if self.frame_parameter.get()!='':
            self.frame_rate=float(self.frame_parameter.get())

        if self.res_parameter.get()!='':
            self.img_resolution=float(self.res_parameter.get())        
            
        if self.txt_stop_tolerance.get()!='':
            self.max_movement_stay=float(self.txt_stop_tolerance.get())/self.img_resolution
        
        
        self.list_update()
        self.track_to_frame()
        self.show_tracks()
          

    def move_to_previous(self):
        
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.show_tracks()
        self.scale_movie.set(self.frame_pos) 
        
    def move_to_next(self):
        
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.show_tracks()
        self.scale_movie.set(self.frame_pos) 
        
        
    def show_tracks(self):    

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
                plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                if self.monitor_switch==0:
                    plt.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])
        if self.memebrane_switch==2:
            #extract skeleton
            skeleton = skimage.morphology.skeletonize(self.membrane_movie[self.frame_pos,:,:]).astype(np.int)
            # create an individual cmap with red colour
            cmap_new = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',['red','red'],256)
            cmap_new._init() # create the _lut array, with rgba values
            alphas = np.linspace(0, 0.8, cmap_new.N+3)
            cmap_new._lut[:,-1] = alphas
            #plot the membrane border on the top
            plt.imshow(skeleton, interpolation='nearest', cmap=cmap_new)
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=12, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        # toolbar
        toolbarFrame = tk.Frame(master=root)
        toolbarFrame.grid(row=18, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()
        
        

    def filtering(self):
        
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
            
        if self.txt_stop_from.get()=='':
            self.filter_stop[0]=0
        else:
            self.filter_stop[0]=float(self.txt_stop_from.get())

        if self.txt_stop_to.get()=='':
            self.filter_stop[1]=10000
        else:
            self.filter_stop[1]=float(self.txt_stop_to.get())          
            
        if self.txt_stop_tolerance.get()!='':
            self.max_movement_stay=float(self.txt_stop_tolerance.get())/self.img_resolution
            
        print("filtering for length: ", self.filter_length, ";   duration: ", self.filter_duration, ";   final stop duration: ", self.filter_stop)

        # filtering 
        self.track_data_filtered={}
        self.track_data_filtered.update({'tracks':[]})
        
        # check through the tracks
        for p in self.track_data['tracks']:
            
            # check length
            if len(p['trace'])>0:
                point_start=p['trace'][0]
                # check length
                track_duration=(p['frames'][-1]-p['frames'][0]+1)/self.frame_rate
                # check maximum displacement between any two positions in track
                track_length=np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2))*self.img_resolution
                # check stop length

                track_stop=FusionEvent.calculate_stand_length(self, p['trace'], p['frames'], self.max_movement_stay)/self.frame_rate
                
            else:
                track_duration=0
                track_length=0
                track_stop=0

                # variables to evaluate the trackS
            length_var=track_length>=self.filter_length[0] and track_length<=self.filter_length[1]
            duration_var=track_duration>=self.filter_duration[0] and track_duration<=self.filter_duration[1]
            stop_var=track_stop>=self.filter_stop[0] and track_duration<=self.filter_stop[1]
            
            if self.txt_track_number.get()=='':
                filterID=True 
            else:                
                filterID=p['trackID']== int(self.txt_track_number.get())

            if length_var==True and duration_var==True and filterID==True and stop_var==True:
                    self.track_data_filtered['tracks'].append(p)
        self.track_to_frame()
        
        #update the list
        self.list_update()
        
        #plot the filters
        lbl2 = tk.Label(master=root, text="filtered tracks: "+str(len(self.track_data['tracks'])-len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 12))
        lbl2.grid(row=8, column=7, columnspan=2, pady=self.pad_val, padx=self.pad_val)          

    def list_update(self):
        
        def tracklist_on_select(even):
            position_in_list=listNodes.curselection()[0]
            
            # creating a new window with class TrackViewer
            self.new_window = tk.Toplevel(self.master)
             
            # create the track set with motion
            this_track=self.track_data_filtered['tracks'][position_in_list]
            motion_type=self.motion_type_evaluate(this_track)
            this_track['motion']=motion_type
            
            TrackViewer(self.new_window, this_track, self.movie, self.membrane_movie, 
                        self.max_movement_stay, self.img_resolution, self.frame_rate)
            
            
        def detele_track_question():
            
            self.qdeletetext = tk.Label(master=root, text="delete track "+str(self.track_data_filtered['tracks'][listNodes.curselection()[0]]['trackID'])+" ?",  bg='white', font=("Times", 10), width=25)
            self.qdeletetext.grid(row=13, column=6, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
            
            self.deletbutton = tk.Button(master=root, text=" OK ", command=detele_track, width=5,  bg='red')
            self.deletbutton.grid(row=13, column=8, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
        def detele_track():
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
            
            #close the question widgets
            
            try: 
                self.qdeletetext.destroy()
            except: print("skip this window, as not open")

            try: 
                self.deletbutton.destroy()
            except: print("skip this window, as not open")            
            
        def new_track_question():
            
            self.qnewtext = tk.Label(master=root, text="create new track "+str(len(self.track_data_filtered['tracks'])+1)+" ?",  bg='white', font=("Times", 10), width=25)
            self.qnewtext.grid(row=14, column=6, columnspan=2, pady=self.pad_val, padx=self.pad_val) 
            
            self.newbutton = tk.Button(master=root, text=" OK ", command=create_track, width=5,  bg='green')
            self.newbutton.grid(row=14, column=8, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
            
        def create_track():
            self.created_tracks_N+=1
            new_trackID=len(self.track_data_filtered['tracks'])+1
            
            p={"trackID": new_trackID, "trace":[[0,0]], "frames":[0], "skipped_frames": 0}
            
            self.track_data['tracks'].append(p)
            
            
            
            print("new track ", new_trackID, "is created")
            
            #visualise without the track
            self.filtering()
            self.track_to_frame()
            
            #update the list
            self.list_update()
            
            #close the question widgets
            
            try: 
                self.qnewtext.destroy()
            except: print("skip this window, as not open")

            try: 
                self.newbutton.destroy()
            except: print("skip this window, as not open")            
            
            
            
        lbl2 = tk.Label(master=root, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 14, "bold"))
        lbl2.grid(row=7, column=5, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        
        # show track statistics
        lbl2 = tk.Label(master=root, text="deleted tracks: "+str(self.deleted_tracks_N), width=30, bg='white',  font=("Times", 12,))
        lbl2.grid(row=8, column=5, columnspan=2, pady=self.pad_val, padx=self.pad_val)        

#        lbl2 = tk.Label(master=root, text="created tracks: "+str(self.created_tracks_N), width=30, bg='white',  font=("Times", 12))
#        lbl2.grid(row=6, column=7, columnspan=2, pady=self.pad_val, padx=self.pad_val)  

        lbl2 = tk.Label(master=root, text="filtered tracks: "+str(len(self.track_data['tracks'])-len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 12))
        lbl2.grid(row=8, column=7, columnspan=2, pady=self.pad_val, padx=self.pad_val)          
        
        # show the list of data with scroll bar
        
        scrollbar = tk.Scrollbar(master=root, orient="vertical")
        scrollbar.grid(row=12, column=9,  sticky=tk.N+tk.S,padx=self.pad_val)

        listNodes = tk.Listbox(master=root, width=60,  font=("Times", 12), selectmode='single')
        listNodes.grid(row=12, column=5, columnspan=4, sticky=tk.N+tk.S,padx=self.pad_val)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<Double-1>', tracklist_on_select)

        scrollbar.config(command=listNodes.yview)
        
        #delete button
        
        deletbutton = tk.Button(master=root, text="DELETE TRACK", command=detele_track_question, width=15,  bg='red')
        deletbutton.grid(row=13, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        # add button

        deletbutton = tk.Button(master=root, text="ADD TRACK", command=new_track_question, width=15,  bg='green')
        deletbutton.grid(row=14, column=5, columnspan=1, pady=self.pad_val, padx=self.pad_val) 
        
        #

        
       # plot the tracks from filtered folder 
        for p in self.track_data_filtered['tracks']:
            
            #calculate length and duration
            if len(p['trace'])>0:
                point_start=p['trace'][0]
                track_duration=p['frames'][-1]-p['frames'][0]+1    
                track_length=round(np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2)),1)
                start_track_frame=p['frames'][0]
            else:
                track_duration=0
                track_length=0
                start_track_frame=0
            
            # add to the list
            listNodes.insert(tk.END, "ID: "+str(p['trackID'])+" start frame: "+str(start_track_frame))        

            
            
    def select_vesicle_movie(self):
        
        filename = tk.filedialog.askopenfilename()
        self.movie_file=filename
        root.update()
        
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]  
        lbl1 = tk.Label(master=root, text="movie: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=2, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)
        
        # create a none-membrane movie
        self.membrane_movie=np.ones(self.movie.shape)
        
        # plot image
        self.show_tracks()
        
           #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.show_tracks() 
        self.scale_movie = tk.Scale(root, from_=0, to=self.movie_length-1, tickinterval=100, length=400, width=10, orient="horizontal", command=show_values)
        self.scale_movie.set(0)        
        self.scale_movie.grid(row=13, column=1, columnspan=3, rowspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

    def select_membrane_movie(self):
        
        filename = tk.filedialog.askopenfilename()
        # read files 
        self.membrane_movie=skimage.io.imread(filename)
        #normalise the membrane values
        self.membrane_movie=self.membrane_movie/np.max(self.membrane_movie)
#        self.membrane_movie[self.membrane_movie>0]=1
    
    def select_track(self):
        
        global folder_path_output  
        filename = tk.filedialog.askopenfilename()
        self.track_file=filename
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
        
        
    def motion_type_evaluate(self, track_data_original):
        '''
        provide motion type evaluation to select directed movement for speed evaluation
        '''

        segmentation_result=self.tg.msd_based_segmentation(track_data_original['trace'])
        motion_type=segmentation_result[:len(track_data_original['frames'])]
        #motion_type=[1]*len(track_data_original['frames'])
        
        return motion_type
    
    def track_to_frame(self):
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
        save csv file
        '''
        # create the data for saving
        self.stat_data.append(['','', 'settings: ', str(self.img_resolution)+' nm/pix', str(self.frame_rate)+' fps' ]) 
        self.stat_data.append(['Track ID', 'Start frame', ' Total distance travelled (nm)',  'Net distance travelled (nm)', 
                         ' Maximum distance travelled (nm)', ' Total trajectory time (sec)', ' Final stop duration (sec)', 
                         ' Net direction (degree)', 'Mean curvilinear speed: average (nm/sec)', 'Mean straight-line speed: average (nm/sec)',
                         'Mean curvilinear speed: moving (nm/sec)', 'Mean straight-line speed: moving (nm/sec)' ]) 
        #['TrackID', 'Start frame', 'Total distance travelled',  'Net distance travelled', 
        #'Maximum distance travelled', 'Total trajectory time',  'Stop duration', 
        # 'Net direction', 'Average speed', 'Speed of movement' ]
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
            #calculate all type of displacements
            # max displacement
            max_displacement=np.round(np.max(displacement_array),2)
            
            # displacement from start to the end
            net_displacement=np.round(np.sqrt((x_e-x_0)**2+(y_e-y_0)**2),2)*self.img_resolution
            
            # total displacement
            x_from=np.asarray(trajectory)[0:-1,0] 
            y_from=np.asarray(trajectory)[0:-1,1] 
            x_to=np.asarray(trajectory)[1:,0] 
            y_to=np.asarray(trajectory)[1:,1] 
            
            #direction
            total_displacement=np.round(np.sum(np.sqrt((x_to-x_from)**2+(y_to-y_from)**2)),2)*self.img_resolution
            pointB=trajectory[-1]                        
            pointA=trajectory[0]            
            net_direction=int(math.degrees(math.atan2((pointB[1] - pointA[1]),(pointB[0] - pointA[0]))) )
            
            # frames        
            time=(track['frames'][-1]-track['frames'][0]+1)/self.frame_rate
            
            # speed 
                    
            #evaluate motion 
            track['motion']=self.motion_type_evaluate(track)
            average_mcs=np.round(self.calculate_speed(track, "average")[0]*self.img_resolution*self.frame_rate,0)
            average_msls=np.round(self.calculate_speed(track, "average")[1]*self.img_resolution*self.frame_rate,0)
                                 
            moving_mcs=np.round(self.calculate_speed(track, "movement")[0]*self.img_resolution*self.frame_rate,0)
            moving_msls=np.round(self.calculate_speed(track, "movement")[1]*self.img_resolution*self.frame_rate,0)
            #stop duration
            stop_t=FusionEvent.calculate_stand_length(self,trajectory, track['frames'], self.max_movement_stay)/self.frame_rate
            
            
            
            self.stat_data.append([track['trackID'], track['frames'][0], total_displacement ,net_displacement,
                                     max_displacement, time, stop_t, 
                                     net_direction, average_mcs, average_msls, moving_mcs, moving_msls, ''])
            
        # select file location and name
        save_file = tk.filedialog.asksaveasfilename()
        if not(save_file.endswith(".csv")):
                save_file += ".csv"

        with open(save_file, 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(self.stat_data)

        csvFile.close()
         
        print("csv file has been saved to ", save_file)    
        
    def calculate_speed(self, track, mode="average"): # mode: "average"/"movement"
        '''
        calculate speed of vesicle movement
        '''
        trajectory=track['trace']
        frames=track['frames']
        motion=track['motion']

        
        #Mean curvilinear speed

        # separated arrays for coordinates
        x_1=np.asarray(trajectory)[1:,0]    
        y_1=np.asarray(trajectory)[1:,1]   

        x_2=np.asarray(trajectory)[0:-1,0]    
        y_2=np.asarray(trajectory)[0:-1,1]   

        # calculate the discplacement

        sqr_disp_back=np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)        

        if mode=="average":  
            
            # sum of all the displacements                   
            disp=np.sum(sqr_disp_back)
            
            # frames        
            time=(frames[-1]-frames[0])

        else: # movement mode
            
            disp=np.sum(np.asarray(motion[:-1])*sqr_disp_back)
            time=np.max((1,np.sum(np.asarray(motion)[:-1])))
           
        #speed        
        curvilinear_speed=disp/time    
        
        # straightline_speed
        if  mode=="average":
            
            straightline_dist=sqr_disp_back=np.sqrt((x_2[0]-x_1[-1])**2+(y_2[0]-y_1[-1])**2)
            straightline_time=(frames[-1]-frames[0])
            straightline_speed=straightline_dist/straightline_time
        else:
            move_switch=0
            start=[0,0]
            end=[0,0]
            distance=0
            frame_n=0
            for pos in range(0, len(motion)-1):

                move_pos=motion[pos]
                if move_pos==1 and move_switch==0: # switching to the moving
                    move_switch=1 # switch to moving mode
                    start=trajectory[pos]
                    
                elif move_pos==0 and move_switch==1: #  end of motion
                    move_switch=0 # switch off moving mode
                    end=trajectory[pos]   
                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                
                elif move_pos==1 and move_switch==1and pos!=(len(motion)-2): # continue moving
                    frame_n=frame_n+1
                    
                elif move_pos==1 and move_switch==1 and pos==(len(motion)-2):
                    frame_n=frame_n+2
                    end=trajectory[pos+1]
                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            
            frame_n=np.max((1, frame_n))
            straightline_speed=distance/frame_n

            
        return curvilinear_speed, straightline_speed # pix/frame
############################################################

class TrackViewer(tk.Frame):
    '''
    class for the individual track viewer
    '''
    def __init__(self, master, track_data, movie, membrane_movie, max_movement_stay, img_resolution, frame_rate):
        tk.Frame.__init__(self, master)

        master.configure(background='white')
#        master.geometry("1200x800") #Width x Height
        
        self.viewer = master
        

        
        # save important data
        self.track_data=track_data
        self.movie=movie
        self.membrane_movie=membrane_movie
        self.frames=track_data['frames']
        self.motion=track_data['motion']
        self.trace=track_data['trace']
        self.id=track_data['trackID']
        self.frame_pos=track_data['frames'][0]
        self.figsize_value=(5,3) # figure size
        self.frame_pos_to_change=0 # frame which can be changed
        self.movie_length=self.movie.shape[0] # movie length
        self.plot_switch=0 # switch between plotting/not plotting tracks
        self.img_resolution=img_resolution # resolution of the movie
        self.frame_rate=frame_rate # movie frame rate
        
        self.max_movement_stay=max_movement_stay # evaluate stopped vesicle - movement within the threshold
        
        self.pixN_basic=100 # margin size 
        self.vesicle_patch_size=10
        
        self.membrane_switch=0 # switch between membrane and no membrane
        
        #track evaluation 
        self.displacement_array=[]
        self.max_displacement=0
        self.net_displacement=0
        self.total_distance=0
        # change the name to add track ID
        master.title("TrackViewer: track ID "+str(self.id))
        
        # interface settings
        
        self.pad_val=1
        
        
     # # # lay out of the frame
        self.show_list()   
        

        
        # movie control
        self.plot_image()
        
        # plot displacement
        
        self.plot_displacement()
        
        # plot intensity graph
        self.intensity_calculation()
        
        # plot parameters
        self.show_parameters()
        

   #  #  # # # # next and previous buttons
        def show_values(v):
            self.frame_pos=int(v)
            self.plot_image() 
          
        self.scale_movie = tk.Scale(master=self.viewer, from_=0, to=self.movie_length, tickinterval=100, length=400, width=5, orient="horizontal", command=show_values)
        self.scale_movie.set(self.frame_pos)        
        self.scale_movie.grid(row=5, column=2, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        buttonbefore = tk.Button(master=self.viewer, text=" << ", command=self.move_to_previous, width=5)
        buttonbefore.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.E) 
        
        buttonnext = tk.Button(master=self.viewer, text=" >> ", command=self.move_to_next, width=5)
        buttonnext.grid(row=5, column=4, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)            
#        
#        buttonbefore = tk.Button(master=self.viewer,text="previous", command=self.move_to_previous, width=10)
#        buttonbefore.grid(row=5, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
#
#        lbframe = tk.Label(master=self.viewer, text=" frame: "+str(self.frame_pos), width=20, bg='white')
#        lbframe.grid(row=5, column=2, columnspan=2, pady=self.pad_val, padx=self.pad_val)
#        
#        buttonnext = tk.Button(master=self.viewer,text="next", command=self.move_to_next, width=10)
#        buttonnext.grid(row=5, column=4, pady=self.pad_val, padx=self.pad_val, sticky=tk.E)
        
     # buttins to change the position
     
        buttonnext = tk.Button(master=self.viewer,text="change", command=self.change_position, width=10)
        buttonnext.grid(row=0, column=5, pady=self.pad_val, padx=self.pad_val)     

        buttonnext = tk.Button(master=self.viewer,text="delete", command=self.delete_position, width=10)
        buttonnext.grid(row=0, column=6, pady=self.pad_val, padx=self.pad_val)     
        
        buttonnext = tk.Button(master=self.viewer,text="add", command=self.add_position, width=10)
        buttonnext.grid(row=0, column=7, pady=self.pad_val, padx=self.pad_val)    
        
        
          
#    # # # # # # filter choice:membrane on/off # # # # # # #   
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
        


    def calculate_speed(self, trajectory, frames, mode="average"): # mode: "average"/"movement"
        '''
        calculate speed of vesicle movement
        '''
        #Mean curvilinear speed

        # separated arrays for coordinates
        x_1=np.asarray(trajectory)[1:,0]    
        y_1=np.asarray(trajectory)[1:,1]   

        x_2=np.asarray(trajectory)[0:-1,0]    
        y_2=np.asarray(trajectory)[0:-1,1]   

        # calculate the discplacement

        sqr_disp_back=np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)        

        if mode=="average":  
            
            # sum of all the displacements                   
            disp=np.sum(sqr_disp_back)
            
            # frames        
            time=(frames[-1]-frames[0])

        else: # movement mode
            
            disp=np.sum(np.asarray(self.motion[:-1])*sqr_disp_back)
            time=np.max((1,np.sum(np.asarray(self.motion)[:-1])))
           
        #speed        
        curvilinear_speed=disp/time    
        
        # straightline_speed
        if  mode=="average":
            
            straightline_dist=sqr_disp_back=np.sqrt((x_2[0]-x_1[-1])**2+(y_2[0]-y_1[-1])**2)
            straightline_time=(frames[-1]-frames[0])
            straightline_speed=straightline_dist/straightline_time
        else:
            move_switch=0
            start=[0,0]
            end=[0,0]
            distance=0
            frame_n=0
            for pos in range(0, len(self.motion)-1):

                move_pos=self.motion[pos]
                if move_pos==1 and move_switch==0: # switching to the moving
                    move_switch=1 # switch to moving mode
                    start=self.trace[pos]
                    
                elif move_pos==0 and move_switch==1: #  end of motion
                    move_switch=0 # switch off moving mode
                    end=self.trace[pos]   
                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                
                elif move_pos==1 and move_switch==1and pos!=(len(self.motion)-2): # continue moving
                    frame_n=frame_n+1
                    
                elif move_pos==1 and move_switch==1 and pos==(len(self.motion)-2):
                    frame_n=frame_n+2
                    end=self.trace[pos+1]
                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            
            frame_n=np.max((1, frame_n))
            straightline_speed=distance/frame_n

            
        return curvilinear_speed, straightline_speed # pix/frame
    
    def calculate_direction(self, trace):
        '''
        calculate average angle of the direction
        '''
        pointB=trace[-1]                        
        pointA=trace[0]
        changeInX = pointB[0] - pointA[0]
        changeInY = pointB[1] - pointA[1]
         
        return int(math.degrees(math.atan2(changeInY,changeInX)) )



        
    def change_position(self):
        
        self.action_cancel()

        self.lbframechange = tk.Label(master=self.viewer, text="Make changes in frame: "+str(self.frames[self.frame_pos_to_change]), width=30, bg='white')
        self.lbframechange.grid(row=0, column=10, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

        self.lbpose = tk.Label(master=self.viewer, text=" x, y ", width=10, bg='white')
        self.lbpose.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)  
        
        self.txt_position = tk.Entry(self.viewer, width=15)
        self.txt_position.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)                
        

        self.buttonOK= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_change, width=10)
        self.buttonOK.grid(row=2, column=10, pady=self.pad_val, padx=self.pad_val)   
        
        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=2, column=11, pady=self.pad_val, padx=self.pad_val)          
        
    def action_apply_change(self):
        
        self.trace[self.frame_pos_to_change]=[int(self.txt_position.get().split(',')[0]), int(self.txt_position.get().split(',')[1])]
        
        # update visualisation
        self.show_list()  
        
        self.plot_image()
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
        self.action_cancel()
        

        
    def action_cancel(self):
        #remove all the widgets related to make changes

        try: 
            self.lbframechange.destroy()
        except: print("skip this window, as not open")
        
        try:
            self.lbpose.destroy()
        except: print("skip this window, as not open")
        try: 
            self.txt_position.destroy()  
        except: print("skip this window, as not open")      
        try: 
            self.txt_frame.destroy()
        except: print("skip this window, as not open")
        
        try: 
            self.button_cancel.destroy()  
        except: print("skip this window, as not open")      
        try: 
            self.buttonOK.destroy()
        except: print("skip this window, as not open")
        try: 
            self.buttonOKdel.destroy()
        except: print("skip this window, as not open")
        try: 
            self.buttonOK_add.destroy()
        except: print("skip this window, as not open")

        
    def delete_position(self):
        
        self.action_cancel()
        
        self.lbframechange = tk.Label(master=self.viewer, text="Do you want to delete frame "+str(self.frames[self.frame_pos_to_change])+" ?", width=40, bg='white')
        self.lbframechange.grid(row=0, column=10, columnspan=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)              
        

        self.buttonOKdel= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_delete, width=10)
        self.buttonOKdel.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val)  
        
        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)     
        
    def action_apply_delete(self):

        del self.trace[self.frame_pos_to_change] 
        del self.frames[self.frame_pos_to_change] 
        # update visualisation
        self.show_list()  
        
        self.plot_image()
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
        self.action_cancel()
        self.intensity_calculation()
        
    def add_position(self): 
        
        self.action_cancel()   
        
        self.lbframechange = tk.Label(master=self.viewer, text=" Add frame: ", width=20, bg='white')
        self.lbframechange.grid(row=0, column=10, pady=self.pad_val, padx=self.pad_val)

        self.txt_frame = tk.Entry(self.viewer, width=10)
        self.txt_frame.grid(row=0, column=11)                
        

        self.lbpose = tk.Label(master=self.viewer, text=" new coordinates: (x,y) ", width=15, bg='white')
        self.lbpose.grid(row=1, column=10, pady=self.pad_val, padx=self.pad_val)  
        
        self.txt_position = tk.Entry(self.viewer, width=10)
        self.txt_position.grid(row=1, column=11, pady=self.pad_val, padx=self.pad_val)                
        

        self.buttonOK_add= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_add, width=10)
        self.buttonOK_add.grid(row=2, column=10, pady=self.pad_val, padx=self.pad_val)   

        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=2, column=11, pady=self.pad_val, padx=self.pad_val)     

        
    def action_apply_add(self):
        
        location_val=[int(self.txt_position.get().split(',')[0]), int(self.txt_position.get().split(',')[1])]
        frame_val=int(self.txt_frame.get())
        
        if frame_val<self.frames[0]:
            pos=0
        
        elif frame_val>self.frames[-1]:
            pos=len(self.frames)+1
        else:
            diff_array=np.asarray(self.frames)-frame_val
            diff_array_abs=abs(diff_array)
            val=min(abs(diff_array_abs))
            if min(diff_array)>0:
                pos=diff_array_abs.tolist().index(val)
            elif min(diff_array)<0:
                pos=diff_array_abs.tolist().inde, sticky=tk.Ex(val)+1
                

        self.trace.insert(pos,location_val)
        self.frames.insert(pos,frame_val)

        # update visualisation
        self.show_list()     
        
        self.plot_image()
        self.plot_displacement()
        self.intensity_calculation()
        self.show_parameters()
        # remove the widgets
        self.action_cancel()

        

    def move_to_previous(self):
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.plot_image()
        self.scale_movie.set(self.frame_pos)
        
    def move_to_next(self):
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.plot_image() 
        self.scale_movie.set(self.frame_pos)            
    
    
    def plot_image(self):
        
        # plot image
        

        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        fig.tight_layout()
        
        if self.membrane_switch==0:
            img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        elif self.membrane_switch==1:
            img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])+0.1*self.membrane_movie[self.frame_pos,:,:]
        else:
            img=self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        #calculate window position
        
        left_point_y=np.min(np.asarray(self.trace)[:,1])-self.pixN_basic
        right_point_y=np.max(np.asarray(self.trace)[:,1])+self.pixN_basic
        top_point_x=np.min(np.asarray(self.trace)[:,0])-self.pixN_basic
        bottom_point_x=np.max(np.asarray(self.trace)[:,0])+self.pixN_basic
        
        
        
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
            
        elif self.plot_switch==2: # plotting motion type
            #define colour
            red_c= (abs(np.array(self.motion)-1)).tolist()
            green_c= self.motion
            for pos in range(0, len(self.trace)-1):
                plt.plot(np.asarray(self.trace)[pos:pos+2,1]- y_min,np.asarray(self.trace)[pos:pos+2,0]-x_min,  color=(red_c[pos],green_c[pos],0))
            
#        plt.title("Displacement")
        
        #plot the border of the membrane if chosen
        if self.membrane_switch==2:
            #extract skeleton self.membrane_movie[self.frame_pos,:,:]
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
        
        def callback(event):
            print(event.x,"  " ,event.y)
        canvas.callbacks.connect('button_press_event', callback)

    def show_parameters(self): 

                # show the list of data with scroll bar
        
        
        listNodes_parameters = tk.Listbox(master=self.viewer, width=50,  font=("Times", 12), selectmode='single')
        listNodes_parameters.grid(row=6, column=1,  columnspan=4, sticky=tk.N+tk.S, pady=self.pad_val, padx=self.pad_val)
        
       # plot the track positions
             # add to the list
        listNodes_parameters.insert(tk.END, " Total distance travelled          "+str(np.round(self.total_distance*self.img_resolution,2))+" nm") 

        listNodes_parameters.insert(tk.END, " Net distance travelled            "+str(np.round(self.net_displacement*self.img_resolution,2))+" nm")  
#        listNodes_parameters.itemconfig(1, {'bg':'gray'})
        listNodes_parameters.insert(tk.END, " Maximum distance travelled        "+str(np.round(self.max_displacement*self.img_resolution,2))+" nm")
        
        listNodes_parameters.insert(tk.END, " Total trajectory time             "+str(np.round((self.frames[-1]-self.frames[0]+1)/self.frame_rate,5))+" sec")

        listNodes_parameters.insert(tk.END, " Final stop duration               "+str(np.round(FusionEvent.calculate_stand_length(self, self.trace, self.frames, self.max_movement_stay)/self.frame_rate, 5))+" sec")
#        listNodes_parameters.itemconfig(3, {'bg':'gray'})
        listNodes_parameters.insert(tk.END, " Net direction                     "+str(self.calculate_direction(self.trace))+ " degrees")

        listNodes_parameters.insert(tk.END, " Mean curvilinear speed: average   "+str(np.round(self.calculate_speed(self.trace, self.frames, "average")[0]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Mean straight-line speed: average "+str(np.round(self.calculate_speed(self.trace, self.frames, "average")[1]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Mean curvilinear speed: moving    "+str(np.round(self.calculate_speed(self.trace, self.frames, "movement")[0]*self.img_resolution*self.frame_rate,0))+" nm/sec")

        listNodes_parameters.insert(tk.END, " Mean straight-line speed: moving  "+str(np.round(self.calculate_speed(self.trace, self.frames, "movement")[1]*self.img_resolution*self.frame_rate,0))+" nm/sec")



#Mean curvilinear speed
#        listNodes_parameters.insert(tk.END, " Mean straight-line speed         "+str(0.00)+" px/frames")

#        listNodes_parameters.insert(tk.END, " Linearity of forward progression "+str(0.00))

#        listNodes_parameters.insert(tk.END, " Confinement ration               "+str(0.00))          

        
    def show_list(self): 
        
        def tracklist_on_select(even):
            self.frame_pos_to_change=listNodes.curselection()[0]

                # show the list of data with scroll bar
        lbend = tk.Label(master=self.viewer, text="LIST OF DETECTIONS:  ",  bg='white', font=("Times", 12))
        lbend.grid(row=1, column=5, columnspan=3, pady=self.pad_val, padx=self.pad_val)
        
        scrollbar = tk.Scrollbar(master=self.viewer, orient="vertical")
        scrollbar.grid(row=2, column=8, rowspan=5,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=self.viewer, width=30, font=("Times", 10), selectmode='single')
        listNodes.grid(row=2, column=5, columnspan=3, rowspan=5 , sticky=tk.N+tk.S, pady=self.pad_val)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<<ListboxSelect>>', tracklist_on_select)
        scrollbar.config(command=listNodes.yview)
        
       # plot the track positions
        for i in range(0, len(self.frames)):
             # add to the list
            listNodes.insert(tk.END, "frame: "+str(self.frames[i])+"  position: "+str(self.trace[i]))     





        
    def intensity_calculation(self):
        '''
        Calculates changes in intersity for the given the track
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
        #
        
        check_border=0 # variable for the
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
        #        fig1.tight_layout()
        plt.plot(frames, (intensity_array_1-np.min(intensity_array_1))/(np.max(intensity_array_1)-np.min(intensity_array_1)), "-b", label="segmented vesicle")
#        self.im =  plt.plot(frames, intensity_array_2/np.max(intensity_array_2), "-r", frames, intensity_array_1/np.max(intensity_array_1), "-g")
#        plt.plot(frames, intensity_array_1/np.max(intensity_array_1), "-g", label="segmented vesicle")
#plt.xlabel("frames", fontsize='small')
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
#        self.im = plt.plot(self.frames, disaplcement) 
        
        # plot dosplacement colour
        for i in range(0, len(self.motion)-1):
            if self.motion[i]==0:
                colourV='r'
            else:
                colourV='g'
            plt.plot((self.frames[i],self.frames[i+1]), (disaplcement[i],disaplcement[i+1]), colourV)
            
        #       plt.xlabel('frames', fontsize='small')
        plt.ylabel('displacement (nm)', fontsize='small')

        plt.title('Displacement per frame', fontsize='small')
        
        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=self.viewer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=9, columnspan=4, rowspan=2, pady=self.pad_val, padx=self.pad_val)   

   
        
class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("TrackHandler")
        parent.configure(background='white')
#        parent.geometry("1200x1000") #Width x Height
        self.main = MainVisual(parent)

        parent.protocol('WM_DELETE_WINDOW', self.close_app)

        tk.mainloop()

    def close_app(self):
        self.quit()
        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
