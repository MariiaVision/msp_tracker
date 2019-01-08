#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:41:16 2019

@author: mariaa
"""
import numpy as np

import tkinter as tk
from tkinter import filedialog

# for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        self.track_data={} # original tracking data
        self.track_data_filtered=self.track_data.copy()  # filtered tracking data  
        self.filter_duration=[0, 1000]
        self.filter_length=[0, 10000]        
        
        
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
        self.buttonFilter.grid(row=4, column=5,columnspan=4,  pady=5)          
        
        
      # # # # # # movie  # # # # # # 

              # movie name 
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file, bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
        
        # plot bg
        bg_img=np.ones((400,400))*0.4
        fig = plt.figure(figsize=(10,10))
        plt.axis('off')
        self.im = plt.imshow(bg_img) # for later use self.im.set_data(new_data)


        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=8, column=1, columnspan=3,pady=5)
        # plot all tracks on top         
        

    def filtering(self):
        
        #read variables
        if self.txt_duration_from.get()=='':
            self.filter_duration[0]=0
            print("not filled")
        else:
            self.filter_duration[0]=int(self.txt_duration_from.get())

        if self.txt_duration_to.get()=='':
            self.filter_duration[1]=1000
            print("not filled")
        else:
            self.filter_duration[1]=int(self.txt_duration_to.get())            
            

        if self.txt_length_from.get()=='':
            self.filter_length[0]=0
            print("not filled")
        else:
            self.filter_length[0]=int(self.txt_length_from.get())

        if self.txt_length_to.get()=='':
            self.filter_length[1]=10000
            print("not filled")
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
            track_duration=p['frames'][-1]-p['frames'][0]
            
            #calculate maximum displacement between any two positions in track
            track_length=np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2))  
            
            length_var=track_length>=self.filter_length[0] and track_length<=self.filter_length[1]
            duration_var=track_duration>=self.filter_duration[0] and track_duration<=self.filter_duration[1]
            
            if length_var==True and duration_var==True:
                self.track_data_filtered['tracks'].append(p)
#                else:
#                    print("Track is not added: ", track_length, "  ", track_duration)


        
        #update the list
        lbl2 = tk.Label(master=root, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), width=30, bg='white',  font=("Times", 16, "bold"))
        lbl2.grid(row=5, column=5, columnspan=4, pady=5)
                # show the list of data with scroll bar
        lbend = tk.Label(master=root, text="LIST OF TRACKS:  ",  bg='white', font=("Times", 14))
        lbend.grid(row=7, column=5, columnspan=4)
        
        scrollbar = tk.Scrollbar(master=root, orient="vertical")
        scrollbar.grid(row=8, column=9,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=root, width=60,  font=("Helvetica", 12))
        listNodes.grid(row=8, column=5, columnspan=4, sticky=tk.N+tk.S)
        listNodes.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listNodes.yview)
        
       # plot the tracks from filtered folder 
        for p in self.track_data_filtered['tracks']:
            #calculate length and duration
            point_start=p['trace'][0]
            track_duration=p['frames'][-1]-p['frames'][0]            
            track_length=round(np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2)),1)
            # add to the list
            listNodes.insert(tk.END, "trackID: "+str(p['trackID'])+"   duration: "+str(track_duration)+
                             "  length: "+str(track_length)+"   start frame: "+str((p['frames'][0])))

    def select_movie(self):
        # Allow user to select movie
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.movie_file=filename
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file, bg='white')
        lbl1.grid(row=3, column=1, columnspan=3, pady=5)
                # plot image
        frame_N=0
        self.image = self.movie[frame_N,:,:]
        fig = plt.figure(figsize=(10,10))
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)

        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=8, column=1, columnspan=3, pady=5)
        
    
    def select_track(self):
        # Allow user to select a file with tracking data
        global folder_path_output  
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.track_file=filename
        
        #read  the tracks data 
        with open(self.track_file) as json_file:  # 'tracking_original.txt'

            self.track_data = json.load(json_file)
            self.track_data_filtered=self.track_data 
            
        lbl2 = tk.Label(master=root, text="Total number of tracks: "+str(len(self.track_data_filtered['tracks'])), bg='white', width=30,   font=("Times", 16, "bold"))
        lbl2.grid(row=5, column=5, columnspan=4, pady=5)
        
                # show the list of data with scroll bar
        lbend = tk.Label(master=root, text="LIST OF TRACKS:  ",  bg='white', font=("Times", 14))
        lbend.grid(row=7, column=5, columnspan=4)
        
        scrollbar = tk.Scrollbar(master=root, orient="vertical")
        scrollbar.grid(row=8, column=9,  sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=root, width=60,  font=("Helvetica", 12))
        listNodes.grid(row=8, column=5, columnspan=4, sticky=tk.N+tk.S)
        listNodes.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listNodes.yview)        

        
        for p in self.track_data['tracks']:
            #calculate length and duration
            point_start=p['trace'][0]
            track_duration=p['frames'][-1]-p['frames'][0]            
            track_length=round(np.max(np.sqrt((point_start[0]-np.asarray(p['trace'])[:,0])**2+(point_start[1]-np.asarray(p['trace'])[:,1])**2)),1)
            # add to the list
            listNodes.insert(tk.END, "trackID: "+str(p['trackID'])+"   duration: "+str(track_duration)+
                             "  length: "+str(track_length)+"   start frame: "+str((p['frames'][0])))
        
        
        
    def show_tracks(self):
        # read data from the selected filesa and show tracks        

        # plot image
        frame_N=0
        self.image = self.movie[frame_N,:,:]
        fig = plt.figure(figsize=(10,10))
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)
        
        # plot tracks
        

        for p in self.track_data_filtered['tracks']:
            trace=p['trace']
#            if len(trace)>20:
            plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list[int(p['trackID'])%len(self.color_list)])


        # DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().grid(row=8, column=1, columnspan=3, pady=5)
        
        



class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("TrackHandler")
        parent.configure(background='white')
        parent.geometry("1500x1200") #Width x Height
        self.main = MainVisual(parent)
#        self.main.pack(side="left")

        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()