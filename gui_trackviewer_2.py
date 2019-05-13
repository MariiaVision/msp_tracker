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

from fusion_events import FusionEvent 

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
        self.movie=[] # matrix with data
        self._membrane_movie=[]
        self.track_data_original={}
        self.track_data={} # original tracking data
        self.track_data_filtered={}  # filtered tracking data  
        self.track_data_framed={}  # tracking data arranged by frames  
        self.filter_duration=[0, 1000]
        self.filter_length=[0, 10000]   
        self.frame_pos=0
        self.movie_length=0
        self.monitor_switch=0 # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        
        # 
        self.figsize_value=(4,4) # image sizeS
      
     # # # # # # menu to choose files and print data # # # # # #
        
        self.button1 = tk.Button(text="Select movie file", command=self.select_movie, width=30)
        self.button1.grid(row=0, column=2, pady=5)

        
        self.button2 = tk.Button(text="Select file with tracks", command=self.select_track, width=30)
        self.button2.grid(row=1, column=2, pady=5)

        self.buttonShow = tk.Button(text="Show tracks", command=self.show_tracks, width=30)
        self.buttonShow.grid(row=2, column=2, pady=5)  

#    # # # # # # filter choice # # # # # # #   
        var = tk.IntVar()
        
        def update_monitor_switch():            
            self.monitor_switch=var.get()
            self.show_tracks()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(root, text="track and IDs", variable=var, value=0, bg='white', command =update_monitor_switch )
        self.R1.grid(row=4, column=1)  
        
        self.R2 = tk.Radiobutton(root, text=" only tracks ", variable=var, value=1, bg='white',command = update_monitor_switch ) #  command=sel)
        self.R2.grid(row=4, column=2)
        
        self.R3 = tk.Radiobutton(root, text="    none    ", variable=var, value=2, bg='white',command=update_monitor_switch ) #  command=sel)
        self.R3.grid(row=4, column=3)
        
       # list switchL # 0 - all, 1 

        # duration
        lbl3 = tk.Label(master=root, text="Track ID: ", width=20, bg='white')
        lbl3.grid(row=0, column=5)
        self.txt_track_number = tk.Entry(root, width=10)
        self.txt_track_number.grid(row=0, column=6)



        # duration
        lbl3 = tk.Label(master=root, text="Duration (frames): from ", width=20, bg='white')
        lbl3.grid(row=1, column=5)
        self.txt_duration_from = tk.Entry(root, width=10)
        self.txt_duration_from.grid(row=1, column=6)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=1, column=7)
        self.txt_duration_to = tk.Entry(root, width=10)
        self.txt_duration_to.grid(row=1, column=8)


        # duration        
        
        
        lbl3 = tk.Label(master=root, text="Length (pixels): from ", width=20, bg='white')
        lbl3.grid(row=2, column=5)
        self.txt_length_from = tk.Entry(root, width=10)
        self.txt_length_from.grid(row=2, column=6)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=2, column=7)
        self.txt_length_to = tk.Entry(root, width=10)
        self.txt_length_to.grid(row=2, column=8)     
        
        # button to filter
        
        self.buttonFilter = tk.Button(text="Filter", command=self.filtering, width=10)
        self.buttonFilter.grid(row=3, column=4, columnspan=3,  pady=5)  
        
        # fusion events and statistics
        
        self.buttonFilter = tk.Button(text="Fusion events", command=self.find_fusion, width=10)
        self.buttonFilter.grid(row=3, column=6, columnspan=3,  pady=5)           
        
        
        # button to update changes
        
        button_save=tk.Button(master=root, text="update", command=self.update_data, width=10)
        button_save.grid(row=10, column=5)
        
        # button to update changes
        
        button_save=tk.Button(master=root, text="save movie", command=self.save_movie, width=10)
        button_save.grid(row=10, column=6)
        
        # save button
     
        button_save=tk.Button(master=root, text="save in file", command=self.save_in_file, width=10)
        button_save.grid(row=10, column=8)        
        
      # # # # # # movie  # # # # # # 

        # movie name 
        lbl1 = tk.Label(master=root, text="movie: "+self.movie_file, bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
        # plot bg
        bg_img=np.ones((200,200))*0.8
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(bg_img) # for later use self.im.set_data(new_data)


        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=1, columnspan=3,pady=5)
        
              
   #  #  # # # # next and previous buttons
        
        buttonbefore = tk.Button(text="previous", command=self.move_to_previous, width=10)
        buttonbefore.grid(row=10, column=1, pady=5, sticky=tk.W) 

        lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=15, bg='white')
        lbframe.grid(row=10, column=2, pady=5)
        
        buttonnext = tk.Button(text="next", command=self.move_to_next, width=10)
        buttonnext.grid(row=10, column=3, pady=5, sticky=tk.E)
        
        buttonnext = tk.Button(text="jumpt to ", command=self.jump_to, width=10)
        buttonnext.grid(row=11, column=2, pady=5)
        
        self.txt_jump_to = tk.Entry(root, width=10)
        self.txt_jump_to.grid(row=12, column=2)
        
    
    def find_fusion(self):
        '''
        fusion event detection based on last frame position
        '''
        print("fusion event detection ....")
        
        # open membrane mask
        filename = tk.filedialog.askopenfilename()
        root.update()
        self._membrane_movie=skimage.io.imread(filename)
        self._membrane_movie[self._membrane_movie>0]=1
        self._membrane_movie=sp.ndimage.binary_dilation(self._membrane_movie, iterations=1)
        
        # detect events
        
        event_count=FusionEvent(self.movie, self.movie, self._membrane_movie, self.track_data_filtered)
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
        
        
    def save_movie(self):
        length=self.movie.shape[0]
        final_img_set = np.zeros((length, self.movie.shape[1], self.movie.shape[2], 3))
    
        for frameN in range(0, length):
#            print("frame ", frameN)        
            plot_info=self.track_data_framed['frames'][frameN]['tracks']
            frame_img=self.movie[frameN,:,:]
            # Make a colour image frame
            orig_frame = np.zeros((self.movie.shape[1], self.movie.shape[2], 3))
    
            orig_frame [:,:,0] = frame_img/np.max(frame_img)*256
            orig_frame [:,:,1] = frame_img/np.max(frame_img)*256
            orig_frame [:,:,2] = frame_img/np.max(frame_img)*256
            
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

            ################### to save #################
            final_img_set[frameN,:,:,:]=orig_frame
            
        
                # save results
        save_file = tk.filedialog.asksaveasfilename()
        
        final_img_set=final_img_set/np.max(final_img_set)*255
        final_img_set=final_img_set.astype('uint8')
        if not(save_file.endswith(".tif") or save_file.endswith(".tiff")):
            save_file += ".tif"
        imageio.volwrite(save_file, final_img_set)
        cv2.destroyAllWindows()
        
    def jump_to(self):
        
        if self.txt_jump_to.get()!='':
            self.frame_pos=int(self.txt_jump_to.get())
            self.show_tracks()
            lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=15, bg='white')
            lbframe.grid(row=10, column=2, pady=5)    
            self.txt_jump_to.delete(0, 'end')
    
    def save_in_file(self):
        
        save_filename = tk.filedialog.asksaveasfilename()
        with open(save_filename+'.txt', 'w') as f:
            json.dump(self.track_data_filtered, f, ensure_ascii=False) 
        print("tracks are saved in  ", save_filename, " file")

    
    def update_data(self):
        
        self.list_update()
        self.track_to_frame()
        self.show_tracks()
          

    def move_to_previous(self):
        
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.show_tracks()
        lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=15, bg='white')
        lbframe.grid(row=10, column=2, pady=5)
        
    def move_to_next(self):
        
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.show_tracks()   
        lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=15, bg='white')
        lbframe.grid(row=10, column=2, pady=5)
        
        
    def show_tracks(self):    

        # plot image

        self.image = self.movie[self.frame_pos,:,:]
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)
        
        if  self.track_data_framed and self.monitor_switch<=1:
            # plot tracks
            plot_info=self.track_data_framed['frames'][self.frame_pos]['tracks']
            for p in plot_info:
                trace=p['trace']
                plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                if self.monitor_switch==0:
                    plt.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])

        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=8, column=1, columnspan=3, pady=5)
    

    def filtering(self):
        
        #read variables
        
        if self.txt_duration_from.get()=='':
            self.filter_duration[0]=0
        else:
            self.filter_duration[0]=int(self.txt_duration_from.get())

        if self.txt_duration_to.get()=='':
            self.filter_duration[1]=1000
        else:
            self.filter_duration[1]=int(self.txt_duration_to.get())                        

        if self.txt_length_from.get()=='':
            self.filter_length[0]=0
        else:
            self.filter_length[0]=int(self.txt_length_from.get())

        if self.txt_length_to.get()=='':
            self.filter_length[1]=10000
        else:
            self.filter_length[1]=int(self.txt_length_to.get())  
        
            
        print("filtering for length: ", self.filter_length, "  and for duration: ", self.filter_duration)

        # filtering 
        self.track_data_filtered={}
        self.track_data_filtered.update({'tracks':[]})
        
        # check through the 
        for p in self.track_data['tracks']:
            
            # check length
            if len(p['trace'])>0:
                point_start=p['trace'][0]
                # check length
                track_duration=p['frames'][-1]-p['frames'][0]+1
                #calculate maximum displacement between any two positions in track
                track_length=np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2))  
                
            else:
                track_duration=0
                track_length=0

                # variables to evaluate the trackS
            length_var=track_length>=self.filter_length[0] and track_length<=self.filter_length[1]
            duration_var=track_duration>=self.filter_duration[0] and track_duration<=self.filter_duration[1]
            if self.txt_track_number.get()=='':
                filterID=True 
            else:                
                filterID=p['trackID']== int(self.txt_track_number.get())

            if length_var==True and duration_var==True and filterID==True:
                    self.track_data_filtered['tracks'].append(p)
        self.track_to_frame()
        
        #update the list
        self.list_update()
    

    def list_update(self):
        
        def tracklist_on_select(even):
            position_in_list=listNodes.curselection()[0]
            
            # creating a new window with class TrackViewer
            self.new_window = tk.Toplevel(self.master)
            TrackViewer(self.new_window, self.track_data_filtered['tracks'][position_in_list], self.movie)
            
            
        def detele_track_question():
            
            self.qdeletetext = tk.Label(master=root, text="delete track "+str(self.track_data_filtered['tracks'][listNodes.curselection()[0]]['trackID'])+" ?",  bg='white', font=("Times", 10), width=20)
            self.qdeletetext.grid(row=6, column=6, columnspan=2, pady=5) 
            
            self.deletbutton = tk.Button(master=root, text=" OK ", command=detele_track, width=7,  bg='red')
            self.deletbutton.grid(row=6, column=8, columnspan=1, pady=5) 
            
        def detele_track():
            
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
            self.qnewtext.grid(row=7, column=6, columnspan=2, pady=5) 
            
            self.newbutton = tk.Button(master=root, text=" OK ", command=create_track, width=7,  bg='green')
            self.newbutton.grid(row=7, column=8, columnspan=1, pady=5) 
            
        def create_track():
            
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
            
            
            
        lbl2 = tk.Label(master=root, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 16, "bold"))
        lbl2.grid(row=5, column=5, columnspan=4, pady=5)
        
        # show the list of data with scroll bar
        
        scrollbar = tk.Scrollbar(master=root, orient="vertical")
        scrollbar.grid(row=8, column=9,  sticky=tk.N+tk.S)

        listNodes = tk.Listbox(master=root, width=40,  font=("Times", 12), selectmode='single')
        listNodes.grid(row=8, column=5, columnspan=4, sticky=tk.N+tk.S)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<Double-1>', tracklist_on_select)

        scrollbar.config(command=listNodes.yview)
        
        #delete button
        
        deletbutton = tk.Button(master=root, text="DELETE TRACK", command=detele_track_question, width=15,  bg='red')
        deletbutton.grid(row=6, column=5, columnspan=1, pady=5) 
        
        # add button

        deletbutton = tk.Button(master=root, text="ADD TRACK", command=new_track_question, width=15,  bg='green')
        deletbutton.grid(row=7, column=5, columnspan=1, pady=5) 

        
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

            
            
    def select_movie(self):
        
        filename = tk.filedialog.askopenfilename()
        self.movie_file=filename
        root.update()
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]  
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
        # plot image
        self.show_tracks()
        
    
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

############################################################

class TrackViewer(tk.Frame):
    '''
    class for the individual track viewer
    '''
    def __init__(self, master, track_data, movie):
        tk.Frame.__init__(self, master)

        master.configure(background='white')
        master.geometry("1200x800") #Width x Height
        
        self.viewer = master
        
        # save important data
        self.track_data=track_data
        self.movie=movie
        self.frames=track_data['frames']
        self.trace=track_data['trace']
        self.id=track_data['trackID']
        self.frame_pos=track_data['frames'][0]
        self.figsize_value=(5,5) # figure size
        self.frame_pos_to_change=0 # frame which can be changed
        self.movie_length=self.movie.shape[0] # movie length
        self.plot_switch=0 # switch between plotting/not plotting tracks
        
        self.max_movement_stay=1.5 # evaluate stopped vesicle - movement within the threshold
        self.frame_freq=4 # movie frame rate
        
        self.pixN_basic=100 # margin size 
        
        # change the name to add track ID
        master.title("TrackViewer: track ID "+str(self.id))
        
        
     # # # lay out of the frame
        self.show_list()     
        
        # movie control
        self.plot_image()
        
        # plot displacement
        
        self.plot_displacement()
        
        buttonbefore = tk.Button(master=self.viewer,text="previous", command=self.move_to_previous, width=10)
        buttonbefore.grid(row=8, column=2, pady=5, sticky=tk.W) 

        lbframe = tk.Label(master=self.viewer, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=8, column=3, pady=5)
        
        buttonnext = tk.Button(master=self.viewer,text="next", command=self.move_to_next, width=10)
        buttonnext.grid(row=8, column=4, pady=5, sticky=tk.E)
        
     # buttins to change the position
     
        buttonnext = tk.Button(master=self.viewer,text="change", command=self.change_position, width=10)
        buttonnext.grid(row=0, column=5, pady=5)     

        buttonnext = tk.Button(master=self.viewer,text="delete", command=self.delete_position, width=10)
        buttonnext.grid(row=0, column=6, pady=5)     
        
        buttonnext = tk.Button(master=self.viewer,text="add", command=self.add_position, width=10)
        buttonnext.grid(row=0, column=7, pady=5)    
        
        
    # information
        text1 = tk.Label(master=self.viewer, text=" duration : "+str(self.frames[-1]-self.frames[0]+1) +" frames", width=20, bg='white')
        text1.grid(row=0, column=9, columnspan=2, pady=5)    

        text1 = tk.Label(master=self.viewer, text=" final stop : "+str(self.calculate_stand_length(self.trace, self.frames))+ " frames", width=20, bg='white')
        text1.grid(row=0, column=11, columnspan=2, pady=5)    

        text1 = tk.Label(master=self.viewer, text=" speed : "+str(round(self.calculate_speed(self.trace, self.frames),2))+ " pix/sec", width=20, bg='white')
        text1.grid(row=1, column=9, columnspan=2, pady=5)    

        text1 = tk.Label(master=self.viewer, text=" direction : "+str(self.calculate_direction(self.trace))+ " degrees", width=20, bg='white')
        text1.grid(row=1 , column=11, columnspan=2, pady=5)  
          
    # plotting switch 
        var = tk.IntVar()
        def update_monitor_plot():            
            self.plot_switch=var.get()
            self.plot_image()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.R1 = tk.Radiobutton(master=self.viewer, text=" tracks on  ", variable=var, value=0, bg='white', command =update_monitor_plot )
        self.R1.grid(row=1, column=3)  
        
        self.R2 = tk.Radiobutton(master=self.viewer, text=" tracks off ", variable=var, value=1, bg='white',command = update_monitor_plot ) #  command=sel)
        self.R2.grid(row=1, column=4)
        
    
#       calculate parameters
    def calculate_stand_length(self, trajectory, frames):
        '''
        calculate length of the standing at the end
        '''
        
        # add missing frames
        pos=0
        new_frames=[]
        new_trace=[]
        for frame_pos in range(frames[0], frames[-1]+1):
            frame=frames[pos]
            
            if frame_pos==frame:
                new_frames.append(frame_pos)
                new_trace.append(trajectory[pos])
                pos=pos+1
            else:
                new_frames.append(frame_pos)
                new_trace.append(trajectory[pos])  
                
        # separated arrays for coordinates
        x=np.asarray(new_trace)[:,0]    
        y=np.asarray(new_trace)[:,1]    
        
        # end coordinates
        x_e=np.asarray(new_trace)[-1,0]
        y_e=np.asarray(new_trace)[-1,1]     
        
        sqr_disp_back=np.sqrt((x-x_e)**2+(y-y_e)**2)
        position=np.array(range(len(sqr_disp_back)))

        sqr_disp_back=sqr_disp_back[::-1]
        displacement_gaussian_3_end=gaussian_filter1d(sqr_disp_back, 3)

        #count for how long it doesn't exceed movement threshold
        movement_array=position[displacement_gaussian_3_end>self.max_movement_stay]
        
        if len(movement_array)>0:  
            stand_time=movement_array[0]
        else:
            stand_time=frames[-1]-frames[0]+1

        return stand_time

    def calculate_speed(self, trajectory, frames):
        '''
        calculate average vesicle speed
        '''
        # separated arrays for coordinates
        x_1=np.asarray(trajectory)[1:,0]    
        y_1=np.asarray(trajectory)[1:,1]   

        x_2=np.asarray(trajectory)[0:-1,0]    
        y_2=np.asarray(trajectory)[0:-1,1]           
        
        # sum of all the displacements       
        sqr_disp_back=np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)
        disp=np.sum(sqr_disp_back)
        
        # frames        
        time=(frames[-1]-frames[0]+1)/self.frame_freq
        
        #speed        
        speed=disp/time
    
        return speed
    
    def calculate_direction(self, trace):
        '''
        calculate average angle of the direction
        '''
        pointB=trace[-1]                        
        pointA=trace[0]
        changeInX = pointB[0] - pointA[0]
        changeInY = pointB[1] - pointA[1]
         
        return int(math.degrees(math.atan2(changeInY,changeInX)) )

        
    def plot_displacement(self):
        trajectory=self.trace
        
        #calculate the displacement
        x=np.asarray(trajectory)[:,0]    
        y=np.asarray(trajectory)[:,1]
        x_0=np.asarray(trajectory)[0,0]
        y_0=np.asarray(trajectory)[0,1]
        disp=np.sqrt((x-x_0)**2+(y-y_0)**2)

        fig = plt.figure(figsize=(3,4))

        fig.tight_layout()
        
        self.im = plt.plot(self.frames, disp)
        plt.xlabel('frames')
        plt.ylabel('displacement (px)')
        
        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=self.viewer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=5, column=9, columnspan=4, rowspan=3)   

        
    def change_position(self):
        
        self.action_cancel()

        self.lbframechange = tk.Label(master=self.viewer, text="Make changes in frame: "+str(self.frames[self.frame_pos_to_change]), width=40, bg='white')
        self.lbframechange.grid(row=2, column=10, columnspan=2, pady=5, sticky=tk.W)

        self.lbpose = tk.Label(master=self.viewer, text=" new coordinates: (x,y) ", width=15, bg='white')
        self.lbpose.grid(row=3, column=10, pady=5, sticky=tk.W)  
        
        self.txt_position = tk.Entry(self.viewer, width=15)
        self.txt_position.grid(row=3, column=11)                
        

        self.buttonOK= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_change, width=10)
        self.buttonOK.grid(row=4, column=10, pady=5)   
        
        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=4, column=11, pady=5)          
        
    def action_apply_change(self):
        
        self.trace[self.frame_pos_to_change]=[int(self.txt_position.get().split(',')[0]), int(self.txt_position.get().split(',')[1])]
        
        # update visualisation
        self.show_list()     
        self.plot_image()
        
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
        self.lbframechange.grid(row=2, column=10, columnspan=2,  pady=5, sticky=tk.W)              
        

        self.buttonOKdel= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_delete, width=10)
        self.buttonOKdel.grid(row=4, column=10, pady=5)  
        
        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=4, column=11, pady=5)     
        
    def action_apply_delete(self):

        del self.trace[self.frame_pos_to_change] 
        del self.frames[self.frame_pos_to_change] 
        # update visualisation
        self.show_list()     
        self.plot_image()
        
        self.action_cancel()

        
    def add_position(self): 
        
        self.action_cancel()   
        
        self.lbframechange = tk.Label(master=self.viewer, text=" Add frame: ", width=20, bg='white')
        self.lbframechange.grid(row=2, column=10, pady=5)

        self.txt_frame = tk.Entry(self.viewer, width=10)
        self.txt_frame.grid(row=2, column=11)                
        

        self.lbpose = tk.Label(master=self.viewer, text=" new coordinates: (x,y) ", width=15, bg='white')
        self.lbpose.grid(row=3, column=10, pady=5)  
        
        self.txt_position = tk.Entry(self.viewer, width=10)
        self.txt_position.grid(row=3, column=11)                
        

        self.buttonOK_add= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_add, width=10)
        self.buttonOK_add.grid(row=4, column=10,  pady=5)   

        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=4, column=11, pady=5)     

        
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
        
        # remove the widgets
        self.action_cancel()

        

    def move_to_previous(self):
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.plot_image()
        lbframe = tk.Label(master=self.viewer, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=8, column=3, pady=5)
        
    def move_to_next(self):
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.plot_image()   
        lbframe = tk.Label(master=self.viewer, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=8, column=3, pady=5)              
    
    
    def plot_image(self):
        
        # plot image

        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        fig.tight_layout()
        
        
        img=self.movie[self.frame_pos,:,:]
        
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
        if self.plot_switch==0:
            for pos in range(0, len(self.trace)-1):
                plt.plot(np.asarray(self.trace)[pos:pos+2,1]- y_min,np.asarray(self.trace)[pos:pos+2,0]-x_min,  color=(red_c[pos],0,blue_c[pos]))
        
            plt.text(np.asarray(self.trace)[-1,1]- y_min,np.asarray(self.trace)[-1,0]- x_min, "  END  ", fontsize=16, color="b")
            plt.plot(np.asarray(self.trace)[-1,1]- y_min,np.asarray(self.trace)[-1,0]- x_min,  "bo",)  
            
            plt.text(np.asarray(self.trace)[0,1]- y_min,np.asarray(self.trace)[0,0]- x_min, "  START  ", fontsize=16, color="r")
            
            plt.plot(np.asarray(self.trace)[0,1]- y_min,np.asarray(self.trace)[0,0]- x_min,  "ro",)  

        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=self.viewer)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=2, columnspan=3, rowspan=5)   
        
        def callback(event):
            print(event.x,"  " ,event.y)
        canvas.callbacks.connect('button_press_event', callback)
        

        
    def show_list(self): 
        
        def tracklist_on_select(even):
            self.frame_pos_to_change=listNodes.curselection()[0]

                # show the list of data with scroll bar
        lbend = tk.Label(master=self.viewer, text="LIST OF DETECTIONS:  ",  bg='white', font=("Times", 14))
        lbend.grid(row=1, column=5, columnspan=3)
        
        scrollbar = tk.Scrollbar(master=self.viewer, orient="vertical")
        scrollbar.grid(row=2, column=8, rowspan=5,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=self.viewer, width=30, font=("Times", 12), selectmode='single')
        listNodes.grid(row=2, column=5, columnspan=3, rowspan=5 , sticky=tk.N+tk.S)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<<ListboxSelect>>', tracklist_on_select)
        scrollbar.config(command=listNodes.yview)
        
       # plot the track positions
        for i in range(0, len(self.frames)):
             # add to the list
            listNodes.insert(tk.END, "frame: "+str(self.frames[i])+"  position: "+str(self.trace[i]))                        

class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("TrackHandler")
        parent.configure(background='white')
        parent.geometry("1200x1000") #Width x Height
        self.main = MainVisual(parent)

        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()
