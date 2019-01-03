#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:41:16 2019

@author: mariaa
"""

import tkinter as tk
from tkinter import filedialog

# for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import skimage
from skimage import io

#
#class EnterInfo(tk.Frame):
#    def __init__(self, master):
#        tk.Frame.__init__(self, master)
#        self.master = master
#        master.configure(background='green')
#
#        self.label = tk.Label(master, text="This is our first GUI!")
#        self.label.pack()
#
#        self.greet_button = tk.Button(master, text="Greet", command=self.greet)
#        self.greet_button.pack()
#
#    def greet(self):
#        print("EnterInfo!")
##         create the rest of your GUI here
        
class MainVisual(tk.Frame):
    # choose the files and visualise the tracks on the data
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.master = master
        
        self.movie_file=" "
        self.track_file=" "  
        self.movie=[]
        
        
        # menu to choose files and print them
        
        self.button1 = tk.Button(text="       Select movie file       ", command=self.select_movie, width=40)
        self.button1.grid(row=4, column=2)

        
        self.button2 = tk.Button(text="       Select file with tracks      ", command=self.select_track, width=40)
        self.button2.grid(row=6, column=2)

        self.buttonShow = tk.Button(text="      Show tracks      ", command=self.show_tracks, width=40)
        self.buttonShow.grid(row=7, column=2)    
        

        
        
    def select_movie(self):
        # Allow user to select movie
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.movie_file=filename
    
    def select_track(self):
        # Allow user to select a file with tracking data
        global folder_path_output  
        filename = tk.filedialog.askopenfilename(filetypes = [("All files", "*.*")])
        self.track_file=filename
    
    def show_tracks(self):
        # read data from the selected filesa and show tracks        
        print("track file: ", self.track_file)
        print("movie file: ", self.movie_file)
        
        # read files 
        self.movie=skimage.io.imread(self.movie_file)
        # plot image
        self.image = self.movie[0,:,:]
        fig = plt.figure(figsize=(5,5))
        self.im = plt.imshow(self.image) # for later use self.im.set_data(new_data)

        # DrawingArea
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=9, column=1, columnspan=3)
        # plot all tracks on top 
        
        # plot the list of data with scroll bar
        


class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("TrackHandler")
        parent.configure(background='white')
        parent.geometry("500x1000") #Width x Height
        self.main = MainVisual(parent)
#        self.main.pack(side="left")

        
if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()