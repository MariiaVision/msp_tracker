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
        
        self.movie_file=" "
        self.track_file=" "  
        self.movie=[]
        self.track_data={}
        
        # menu to choose files and print them
        
        self.button1 = tk.Button(text="       Select movie file       ", command=self.select_movie, width=40)
        self.button1.grid(row=2, column=2)

        
        self.button2 = tk.Button(text="       Select file with tracks      ", command=self.select_track, width=40)
        self.button2.grid(row=4, column=2)

        self.buttonShow = tk.Button(text="      Show tracks      ", command=self.show_tracks, width=40)
        self.buttonShow.grid(row=5, column=2)    
        
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file, bg='white')
        lbl1.grid(row=6, column=1, columnspan=3)
        # plot bg
        bg_img=np.ones((400,400))*0.4
        fig = plt.figure(figsize=(10,10))
        plt.axis('off')
        self.im = plt.imshow(bg_img) # for later use self.im.set_data(new_data)


        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9, column=1, columnspan=3)
        # plot all tracks on top         
        

        
    def select_movie(self):
        # Allow user to select movie
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.movie_file=filename
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        lbl1 = tk.Label(master=root, text="movie file: "+self.movie_file, bg='white')
        lbl1.grid(row=6, column=1, columnspan=3)
                # plot image
        frame_N=0
        self.image = self.movie[frame_N,:,:]
        fig = plt.figure(figsize=(10,10))
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)

        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9, column=1, columnspan=3)
        
    
    def select_track(self):
        # Allow user to select a file with tracking data
        global folder_path_output  
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.track_file=filename
        
        #read  the tracks data 
        with open(self.track_file) as json_file:  # 'tracking_original.txt'

            self.track_data = json.load(json_file)
            
        lbl2 = tk.Label(master=root, text="total number of tracks: "+str(len(self.track_data['tracks'])), bg='white')
        lbl2.grid(row=8, column=1, columnspan=3)
        
                # show the list of data with scroll bar
        lbend = tk.Label(master=root, text="LIST OF TRACKS:  ",  bg='white', font=("Helvetica", 14))
        lbend.grid(row=5, column=4)
        
        scrollbar = tk.Scrollbar(master=root, orient="vertical")
        scrollbar.grid(row=6, column=5, rowspan=2, sticky=tk.N+tk.S)
        
        listNodes = tk.Listbox(master=root, width=50, height=50, font=("Helvetica", 12))
        listNodes.grid(row=6, column=4, rowspan=2)
        listNodes.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listNodes.yview)        

        
        for p in self.track_data['tracks']:
#        for x in range(100):
            listNodes.insert(tk.END, "trackID: "+str(p['trackID'])+"   track length: "+str(len(p['trace']))+"   start frame: "+str((p['frames'][0])))
        
        
        
    def show_tracks(self):
        # read data from the selected filesa and show tracks        

        # plot image
        frame_N=0
        self.image = self.movie[frame_N,:,:]
        fig = plt.figure(figsize=(10,10))
        plt.axis('off')
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)
        
        # plot tracks
        

        for p in self.track_data['tracks']:
            trace=p['trace']
            if len(trace)>20:
                plt.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list[int(p['trackID'])%len(self.color_list)])


        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9, column=1, columnspan=3)
        



class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("TrackHandler")
        parent.configure(background='white')
        parent.geometry("1500x1000") #Width x Height
        self.main = MainVisual(parent)
#        self.main.pack(side="left")

        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()