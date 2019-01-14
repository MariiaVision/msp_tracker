#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:41:16 2019

@author: mariaa
"""
import numpy as np

import time
import copy
import tkinter as tk
from tkinter import filedialog

# for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import skimage
from skimage import io

import json        


class MainVisual(tk.Frame):
    # choose the files and visualise the tracks on the data
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        
        self.color_list=["#00FFFF", "#7FFFD4", "#0000FF", "#8A2BE2", "#7FFF00", "#D2691E", "#FF7F50", "#DC143C",
            "#008B8B", "#8B008B", "#FF8C00", "#E9967A", "#FF1493", "#9400D3", "#FF00FF", "#B22222",
            "#FFD700", "#ADFF2F", "#FF69B4", "#ADD8E6", "#F08080", "#90EE90", "#20B2AA", "#C71585", "#FF00FF"]
        
        self.movie_file=" " # path to the move file
        self.track_file=" "# path to the file with tracking data (json format)
        self.movie=[] # matrix with data
        self.track_data_original={}
        self.track_data={} # original tracking data
        self.track_data_filtered={}  # filtered tracking data  
        self.track_data_framed={}  # tracking data arranged by frames  
        self.filter_duration=[0, 1000]
        self.filter_length=[0, 10000]   
        self.frame_pos=0
        self.movie_length=0
        
        # 
        self.figsize_value=(6,6)
      
     # # # # # # menu to choose files and print data # # # # # #
        
        self.button1 = tk.Button(text="       Select movie file       ", command=self.select_movie, width=40)
        self.button1.grid(row=0, column=2, pady=5)

        
        self.button2 = tk.Button(text="       Select file with tracks      ", command=self.select_track, width=40)
        self.button2.grid(row=1, column=2, pady=5)

        self.buttonShow = tk.Button(text="      Show tracks      ", command=self.show_tracks, width=40)
        self.buttonShow.grid(row=2, column=2, pady=5)  

#    # # # # # # filter choice # # # # # # #        
        # filter / non-filter choice
#        R1 = tk.Radiobutton(root, text="all tracks", variable=self.filter_switch, value=False, bg='white')
#        R1.grid(row=1, column=5, padx=5)  
#        
#        R2 = tk.Radiobutton(root, text="filtering", variable=self.filter_switch, value=True, bg='white') #  command=sel)
#        R2.grid(row=1, column=6, padx=5)
        
       # 

        # duration
        lbl3 = tk.Label(master=root, text="Track duration (frames): from ", width=30, bg='white')
        lbl3.grid(row=2, column=5)
        self.txt_duration_from = tk.Entry(root, width=10)
        self.txt_duration_from.grid(row=2, column=6)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=2, column=7)
        self.txt_duration_to = tk.Entry(root, width=10)
        self.txt_duration_to.grid(row=2, column=8)
        #reader=txt.get()

        # duration        

        lbl3 = tk.Label(master=root, text="Track length (pixels): from ", width=30, bg='white')
        lbl3.grid(row=3, column=5)
        self.txt_length_from = tk.Entry(root, width=10)
        self.txt_length_from.grid(row=3, column=6)
        lbl3 = tk.Label(master=root, text="to", bg='white')
        lbl3.grid(row=3, column=7)
        self.txt_length_to = tk.Entry(root, width=10)
        self.txt_length_to.grid(row=3, column=8)     
        
        # button to filter
        
        self.buttonFilter = tk.Button(text="     Filter      ", command=self.filtering, width=40)
        self.buttonFilter.grid(row=4, column=5, columnspan=4,  pady=5)   
        
        
        # button to update changes
        
        button_save=tk.Button(master=root, text=" update changes ", command=self.update_data, width=15)
        button_save.grid(row=10, column=5)
        
        # save button
     
        button_save=tk.Button(master=root, text=" save in file ", command=self.save_in_file, width=15)
        button_save.grid(row=10, column=8)        
        
      # # # # # # movie  # # # # # # 

              # movie name 
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file, bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
        # plot bg
        bg_img=np.ones((400,400))*0.8
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(bg_img) # for later use self.im.set_data(new_data)


        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=1, columnspan=3,pady=5)
        # plot all tracks on top         
   #  #  # # # # next and previous buttons
        
        buttonbefore = tk.Button(text="previous", command=self.move_to_previous, width=10)
        buttonbefore.grid(row=10, column=1, pady=5, sticky=tk.W) 

        lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=10, column=2, pady=5)
        
        buttonnext = tk.Button(text="next", command=self.move_to_next, width=10)
        buttonnext.grid(row=10, column=3, pady=5, sticky=tk.E)

# # # #  play movie - not working - need another solultion
#        buttonnext = tk.Button(text="play", command=self.play_movie, width=20)
#        buttonnext.grid(row=10, column=2, padx=5)
#        
#        buttonnext = tk.Button(text="stop", command=self.play_stop, width=20)
#        buttonnext.grid(row=10, column=3, padx=5)
#        
#    def play_movie(self):
#        while self.frame_pos!=self.movie_length:
#            self.frame_pos+=1   
#            self.show_tracks()
#            lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=20, bg='white')
#            lbframe.grid(row=9, column=2, pady=5)
#            time.sleep(1)
        
        
    
    def save_in_file(self):
        save_filename = tk.filedialog.asksaveasfilename(filetypes = [("All files", "*.*")])
        with open(save_filename+'.txt', 'w') as f:
            json.dump(self.track_data, f, ensure_ascii=False)  
        print("origianal: ", self.track_data_original)
    
    def update_data(self):
        self.list_update()
        self.track_to_frame()
        self.show_tracks()
        print("data updated")            

    def move_to_previous(self):
        if self.frame_pos!=0:
            self.frame_pos-=1
        self.show_tracks()
        lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=10, column=2, pady=5)
        
    def move_to_next(self):
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.show_tracks()   
        lbframe = tk.Label(master=root, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=10, column=2, pady=5)
        
       

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
            point_start=p['trace'][0]
            track_duration=p['frames'][-1]-p['frames'][0]+1
            
            #calculate maximum displacement between any two positions in track
            track_length=np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2))  
            
            # variables to evaluate the trackS
            length_var=track_length>=self.filter_length[0] and track_length<=self.filter_length[1]
            duration_var=track_duration>=self.filter_duration[0] and track_duration<=self.filter_duration[1]
            
            if length_var==True and duration_var==True:
                self.track_data_filtered['tracks'].append(p)
        self.track_to_frame()
        
        #update the list
        self.list_update()
    

    def list_update(self):
        # updating the list 
        
        def tracklist_on_select(even):
            position_in_list=listNodes.curselection()[0]
            print('You selected: ', position_in_list)
            print('Track info:   ', self.track_data_filtered['tracks'][position_in_list]['trackID'])
            # creating a new window with class TrackViewer
            self.new_window = tk.Toplevel(self.master)
            TrackViewer(self.new_window, self.track_data_filtered['tracks'][position_in_list], self.movie)
            
        lbl2 = tk.Label(master=root, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 16, "bold"))
        lbl2.grid(row=5, column=5, columnspan=4, pady=5)
                # show the list of data with scroll bar
        lbend = tk.Label(master=root, text="LIST OF TRACKS:  ",  bg='white', font=("Times", 14))
        lbend.grid(row=7, column=5, columnspan=4)
        
        scrollbar = tk.Scrollbar(master=root, orient="vertical")
        scrollbar.grid(row=8, column=9,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=root, width=60,  font=("Times", 12), selectmode='single')
        listNodes.grid(row=8, column=5, columnspan=4, sticky=tk.N+tk.S)
        listNodes.config(yscrollcommand=scrollbar.set)
        listNodes.bind('<Double-1>', tracklist_on_select)
        scrollbar.config(command=listNodes.yview)
        
       # plot the tracks from filtered folder 
        for p in self.track_data_filtered['tracks']:
            #calculate length and duration
            point_start=p['trace'][0]
            track_duration=p['frames'][-1]-p['frames'][0]+1    
            track_length=round(np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2)),1)
            # add to the list
            listNodes.insert(tk.END, "ID: "+str(p['trackID'])+" duration: "+str(track_duration)+
                             " length: "+str(track_length)+" start frame: "+str((p['frames'][0])))        

            
            
    def select_movie(self):
        # Allow user to select movie
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.movie_file=filename
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        self.movie_length=self.movie.shape[0]  
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file.split("/")[-1], bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
                # plot image
        self.show_tracks()
        
    
    def select_track(self):
        # Allow user to select a file with tracking data
        global folder_path_output  
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.track_file=filename
        
        #read  the tracks data 
        with open(self.track_file) as json_file:  # 'tracking_original.txt'

            self.track_data_original = json.load(json_file)
            self.track_data=copy.deepcopy(self.track_data_original)
            self.track_data_filtered=self.track_data 
            self.track_to_frame()
            
        self.list_update()        
        
        
    def show_tracks(self):
        # read data from the selected filesa and show tracks        

        # plot image

        self.image = self.movie[self.frame_pos,:,:]
        plt.close()
        fig = plt.figure(figsize=self.figsize_value)
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)
        
        if  self.track_data_framed:
            # plot tracks
            plot_info=self.track_data_framed['frames'][self.frame_pos]['tracks']
            for p in plot_info:
                trace=p['trace']
                plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list[int(p['trackID'])%len(self.color_list)])      
    #
#        for p in self.track_data_filtered['tracks']:
#            trace=p['trace']
#            plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list[int(p['trackID'])%len(self.color_list)])


        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=8, column=1, columnspan=3, pady=5)
        
#        toolbar = NavigationToolbar2Tk(canvas, root)
#        toolbar.update()
#        canvas.get_tk_widget().grid(row=9, column=1, columnspan=3, pady=5)
#
#        def on_key_press(event):
#            print("you pressed {}".format(event.key))
#            key_press_handler(event, canvas, toolbar)
#        
#        
#        canvas.mpl_connect("key_press_event", on_key_press)
    
    def track_to_frame(self):
        # change data arrangment from tracks to frames
        self.track_data_framed={}
        self.track_data_framed.update({'frames':[]})
        
        for n_frame in range(0,self.movie_length):
            
            frame_dict={}
            frame_dict.update({'frame': n_frame})
            frame_dict.update({'tracks': []})
            
            #rearrange the data
            for p in self.track_data['tracks']:
                if n_frame in p['frames']: # if the frame is in the track
                    frame_index=p['frames'].index(n_frame) # find position in the track
                    
                    new_trace=p['trace'][0:frame_index+1] # copy all the traces before the frame
                    frame_dict['tracks'].append({'trackID': p['trackID'], 'trace': new_trace}) # add to the list
                    
                    
            self.track_data_framed['frames'].append(frame_dict) # add the dictionary

############################################################

class TrackViewer(tk.Frame):
    def __init__(self, master, track_data, movie):
        tk.Frame.__init__(self, master)
#        master.title("TrackViewer")
        master.configure(background='red')
        master.geometry("1500x1000") #Width x Height
        
        self.viewer = master
        
        # save important data
        self.track_data=track_data
        self.movie=movie
        self.frames=track_data['frames']
        self.trace=track_data['trace']
        self.id=track_data['trackID']
        self.frame_pos=track_data['frames'][0]
        self.figsize_value=(7,7) # figure size
        self.frame_pos_to_change=0 # frame which can be changed
        self.movie_length=self.movie.shape[0] # movie length
        # change the name to add track ID
        master.title("TrackViewer: track ID "+str(self.id))
        
        
     # # # lay out of the frame
        self.show_list()     
        
        # movie control
        self.plot_image()
        
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
        
#        buttonnext = tk.Button(master=self.viewer,text=" save ", command=self.add_position, width=10)
#        buttonnext.grid(row=9, column=6, pady=5)    
#    
#        
#    def save_to_main_frame(self):
#        print("saving to main frame")
#        
        
        
    def change_position(self):
        
        self.action_cancel()

        self.lbframechange = tk.Label(master=self.viewer, text="Make changes in frame: "+str(self.frames[self.frame_pos_to_change]), width=40, bg='white')
        self.lbframechange.grid(row=1, column=10, columnspan=2, pady=5, sticky=tk.W)

        self.lbpose = tk.Label(master=self.viewer, text=" new coordinates: (x,y) ", width=25, bg='white')
        self.lbpose.grid(row=2, column=10, pady=5, sticky=tk.W)  
        
        self.txt_position = tk.Entry(self.viewer, width=20)
        self.txt_position.grid(row=2, column=11)                
        

        self.buttonOK= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_change, width=10)
        self.buttonOK.grid(row=3, column=10, pady=5)   
        
        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=3, column=11, pady=5)          
        
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
        self.lbframechange.grid(row=1, column=10, columnspan=2,  pady=5, sticky=tk.W)              
        

        self.buttonOKdel= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_delete, width=10)
        self.buttonOKdel.grid(row=3, column=10, pady=5)  
        
        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=3, column=11, pady=5)     
        
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
        self.lbframechange.grid(row=1, column=10, pady=5)

        self.txt_frame = tk.Entry(self.viewer, width=10)
        self.txt_frame.grid(row=1, column=11)                
        

        self.lbpose = tk.Label(master=self.viewer, text=" new coordinates: (x,y) ", width=25, bg='white')
        self.lbpose.grid(row=2, column=10, pady=5)  
        
        self.txt_position = tk.Entry(self.viewer, width=20)
        self.txt_position.grid(row=2, column=11)                
        

        self.buttonOK_add= tk.Button(master=self.viewer,text=" apply ", command=self.action_apply_add, width=10)
        self.buttonOK_add.grid(row=3, column=10,  pady=5)   

        self.button_cancel= tk.Button(master=self.viewer,text=" cancel ", command=self.action_cancel, width=10)
        self.button_cancel.grid(row=3, column=11, pady=5)     

        
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
        
#        buttonbefore = tk.Button(master=self.viewer,text="previous", command=self.move_to_previous, width=10)
#        buttonbefore.grid(row=8, column=2, pady=5, sticky=tk.W) 
#        
#        buttonnext = tk.Button(master=self.viewer,text="next", command=self.move_to_next, width=10)
#        buttonnext.grid(row=8, column=4, pady=5, sticky=tk.E)
        
    def move_to_next(self):
        if self.frame_pos!=self.movie_length:
            self.frame_pos+=1
        self.plot_image()   
        lbframe = tk.Label(master=self.viewer, text=" frame: "+str(self.frame_pos), width=20, bg='white')
        lbframe.grid(row=8, column=3, pady=5)
#        
#        buttonbefore = tk.Button(master=self.viewer,text="previous", command=self.move_to_previous, width=10)
#        buttonbefore.grid(row=8, column=2, pady=5, sticky=tk.W) 
#        
#        buttonnext = tk.Button(master=self.viewer,text="next", command=self.move_to_next, width=10)
#        buttonnext.grid(row=8, column=4, pady=5, sticky=tk.E)                
    
    
    def plot_image(self):
        
        # plot image
        plt.close()
        fig = plt.figure(figsize=self.figsize_value)
#        plt.axis('off')
        fig.tight_layout()
        
        pixN_basic=200
        img=self.movie[self.frame_pos,:,:]
        pixN_min=np.min(([pixN_basic, np.min(np.asarray(self.trace)[:,1]), np.min(np.asarray(self.trace)[:,0])]))
        pixN=np.min(([pixN_min, img.shape[0]-np.max(np.asarray(self.trace)[:,1]),  img.shape[1]-np.max(np.asarray(self.trace)[:,0])]))
        
        y_min=np.min(np.asarray(self.trace)[:,1])-pixN
        y_max=np.max(np.asarray(self.trace)[:,1])+pixN
    
        x_min=np.min(np.asarray(self.trace)[:,0])-pixN
        x_max=np.max(np.asarray(self.trace)[:,0])+pixN
        
        region=img[x_min:x_max, y_min:y_max]
        
        blue_c=np.linspace(0., 1., len(self.trace))
        red_c=1-np.linspace(0., 1., len(self.trace))
    
        self.im = plt.imshow(region, cmap="gray")
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
            #self.frames[self.frame_to_change]
            print("selected "+str(self.frames[self.frame_pos_to_change]))

                # show the list of data with scroll bar
        lbend = tk.Label(master=self.viewer, text="LIST OF DETECTIONS:  ",  bg='white', font=("Times", 14))
        lbend.grid(row=1, column=5, columnspan=3)
        
        scrollbar = tk.Scrollbar(master=self.viewer, orient="vertical")
        scrollbar.grid(row=2, column=8, rowspan=5,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=self.viewer, width=50, font=("Times", 12), selectmode='single')
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
        parent.geometry("1100x900") #Width x Height
        self.main = MainVisual(parent)
#        self.main.pack(side="left")



        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()