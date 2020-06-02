#########################################################
#
#  main framework to run the GUI and tracking module 
#        
#########################################################

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # not to show warnings

import sys
sys.path.append('./tracking_lib/')

import numpy as np
import scipy as sp

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import csv

# for plotting
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import skimage
from skimage import io
import json 

from set_tracking import  TrackingSetUp


    
class MainApplication(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        parent.title("MSBNtracker")
        parent.configure(background='white')
        
        #set the window size        
        self.window_width = int(parent.winfo_screenwidth()/2.5) # part of the monitor width
        self.window_height = int(parent.winfo_screenheight()*0.7)  # 0.7 of the monitor height
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
            T = tk.Text(self.new_window_help, wrap=tk.WORD) #, height=int(self.window_height/4), width=int(self.window_width/4))
            S = tk.Scrollbar(self.new_window_help)
            S.config(command=T.yview)
            S.pack(side=tk.RIGHT, fill=tk.Y)
            T.config(yscrollcommand=S.set)
            T.configure(font=("Helvetica", 12))
            # widget in there with a text 
            text_help=""" Information about the software: 
                
The tracking software allows to set parameters and test them before running the tracker for the movie. 

You can test detection on a single frame and linking on a sequence of frames.   

Follow the steps to run the tracker: 
    
1) load the movie (button "select vesicle movie")
It should be single channel movie.

2) Set detection parameters (load from a file if set earlier): 
    a) Set parameter for candidate detection.  
    b) Set parameters for candidate pruning. 
    c) Run test and change parameters until satisfied with the results.  

3) Set Linking parameters (load from a file if set earlier): 
    a) Choose number of passes.  
    b) Set STEP 1. 
    c) Set one pass after another. 
    d) Set start and end frame for testing. 
    e) Run test and change parameters until satisfied with the results. 

4) Run the tracker:  
    a) Check the parameters 
    b) Choose start and end frames 
    c) Select location and name of the file with tracking results 
    d) Run the tracker with button “RUN TRACKING” 

You should not close the software until the tracking is complete. 
You can follow the tracking progress in the terminal. 
When finished the final tracks will appear in the linking window and also can be opened in the separate viewer software. 

The tracking algorithm is described in 
“Protein Tracking By CNN-Based Candidate Pruning And Two-Step Linking With Bayesian Network” 
Mariia Dmitrieva,  Helen L Zenner, Jennifer Richens, Daniel St Johnston, Jens Rittscher, 
2019 IEEE 29th International Workshop on Machine Learning for Signal Processing (MLSP) 


            """
            
#            lb_start = tk.Label(master=self.new_window_help, text=" Information about the software",  bg='white')
            T.insert(tk.END, text_help)
            T.pack(side=tk.LEFT, fill=tk.Y)
            
            
        def HelpDetection():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("Detection ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
            # widget in there with a text 
            T = tk.Text(self.new_window_help, wrap=tk.WORD) #, height=int(self.window_height/4), width=int(self.window_width/4))
            S = tk.Scrollbar(self.new_window_help)
            S.config(command=T.yview)
            S.pack(side=tk.RIGHT, fill=tk.Y)
            T.config(yscrollcommand=S.set)
            T.configure(font=("Helvetica", 12))
            # widget in there with a text 
            text_help=""" Detection:
    
    a) Set parameter for candidate detection. 
    
        - Firstly, set Multi Scale Spot Enhancing Filter (MSSEF) choosing Threshold coefficient and Sigma. You can use “Show MSSEF” button to check the  enhanced spots. 
The ideal image is when all the vesicles appear as separate spots without any holes inside and area outside the cell doesn't produce any spots.
Threshold coefficient - intensity of the spots (from 0 up to 2-5)
Sigma - range of blur for spot enhancement, influence the spot size and its shape.

        - Set Relevant peak height (from 0 to 1). Increase the value to detect only bright spots and decrease it if darker spots have to be detected as well.
        
    b) Set parameter for candidate pruning.
    
- Minimum distance between detections - number of pixels expected between centers of two vesicles
- Region of Interest size - region for classifier 8 or 16. At this stage it is 16 for all models unless states opposite.
- Threshold coefficient - threshold for the classifier (from 0 to 1)
- Load CNN model - there is a number of different trained models (folder "dl_weight"), if the classification results are not good you can try another model
- Gaussian fit - on/off 
-Region of Gaussian fit - region of interest to fit gaussian
- Expected particle radius - odd number of pixels
 
You can test the detection on the current frame – button “Run test”, save the parameters into a file – button “Save to file” and load saved parameters “Read from file”. 
            

"""
            
#            lb_start = tk.Label(master=self.new_window_help, text=" Information about the software",  bg='white')
            T.insert(tk.END, text_help)
            T.pack(side=tk.LEFT, fill=tk.Y)    
            
        def HelpLinking():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("Linking ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
            # widget in there with a text 
            T = tk.Text(self.new_window_help, wrap=tk.WORD) #, height=int(self.window_height/4), width=int(self.window_width/4))
            S = tk.Scrollbar(self.new_window_help)
            S.config(command=T.yview)
            S.pack(side=tk.RIGHT, fill=tk.Y)
            T.config(yscrollcommand=S.set)
            T.configure(font=("Helvetica", 12))
            
            text_help="""Linking:
                
    a) Choose number of passes. 
    
This is number of tracklinking passes. In most of the cases 1 pass is enough, but in case of dense vesicle population or difference in speed movement it can be beneficial to use few passes.

    b) Set STEP 1. 
    
At this step short tracklets are forms based on the distance. You need to specify three parameters: 

- maximum distance to link - number of pixels between to detection which still can be linked
- maximum skipped frames - how many frames can be skipped in the same tracklet between two detections
- maximum track length - number of frames in one tracklet. Could be 5-10 in general, but for a dense movement should be about 3-4.
   
    c) Set one pass after another. 

- Bayesian network topology - complete is the best option yet, which include all the nodes in the network
- connectivity threshold - final threshold which decide on tracklets connection (from 0 to 1)
- temporal gap - maximum number of frames which can be between two connections (when vesicle is not detected for a number of frames)
- distance limit - maximum expected distance between two detections
- orientation similarity - acceptable difference in orientation (from 0 to 180)
- speed similarity limit - acceptable proportional difference in speed (from 0 to 1)
- intensity similarity limit - acceptable difference in intensity (from 0 to 1)
- threshold of track length - the final tracks shorter than the number will be removed. Make it 0 for all the passes accept the last one.

When you have multiple passes the idea is first to connect slowly moving vesicles:
    - smaller values for speed, intensity and orientation - it is not important
    - small values for temporal gaps - the connections should be close to each other in time
    - smaller value for the distance limit - this value depends on the speed you want to take into account
    
And with second-third pass faster moving vesicles will be linked:
    - can increase values for speed, intensity and orientation, but not necessary 
    - increase the temporal gap 
    - increase the distance limit

    d) Set start and end frame for testing. 
It can be about 10-40 frames at the time when the most complex movement is happening

    e) Run test and change parameters until satisfied with results. 
    
    
            """
            
#            lb_start = tk.Label(master=self.new_window_help, text=" Information about the software",  bg='white')
            T.insert(tk.END, text_help)
            T.pack(side=tk.LEFT, fill=tk.Y)
                    
            
            
        def HelpTracking():
            # create a new window
            self.new_window_help = tk.Toplevel(self.master)
            self.new_window_help.title("Tracking ")
#            self.new_window_help.geometry(str(int(self.window_width/10))+"x"+str(int(self.window_height/10)))
            self.new_window_help.configure(background='white')
                        # widget in there with a text 
            T = tk.Text(self.new_window_help, wrap=tk.WORD) #, height=int(self.window_height/4), width=int(self.window_width/4))
            S = tk.Scrollbar(self.new_window_help)
            S.config(command=T.yview)
            S.pack(side=tk.RIGHT, fill=tk.Y)
            T.config(yscrollcommand=S.set)
            T.configure(font=("Helvetica", 12))
            text_help=""" Run the tracker: 
                
    a) Check the parameters. 
    b) Choose start and end frames 
    c) Select location and name of the file with tracking results 
    d) Run the tracker with button “RUN TRACKING” 

You should not close the software until the tracking is complete. 
You can see the progress in the terminal. 
When finished the final tracks will appear in the linking window and also can be opened in the separate viewer software. 


            """
            
#            lb_start = tk.Label(master=self.new_window_help, text=" Information about the software",  bg='white')
            T.insert(tk.END, text_help)
            T.pack(side=tk.LEFT, fill=tk.Y)
                 
            
            
        helpmenu.add_command(label="About...", command=HelpAbout)
        helpmenu.add_command(label="Detection", command=HelpDetection)
        helpmenu.add_command(label="Linking", command=HelpLinking)
        helpmenu.add_command(label="Run tracking", command=HelpTracking)
        
        # set movie and class for parameter settings
        self.movie=np.ones((1,200,200))
        self.detector=TrackingSetUp()
        self.Npass=1 # number of tracklinking pass
        

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
        self.detection_frame=0 # frame where the latest detection was made
        self.movie_length=1
        self.monitor_switch_detection=0
        self.pad_val=1
        self.dpi=100
        self.img_width=self.window_height*0.6
        self.figsize_value=(self.img_width/self.dpi, self.img_width/self.dpi)
        self.button_length=np.max((10,int(self.window_width/130)))
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
        self.button_mv = tk.Button(self.viewFrame_detection,text="   Select vesicle movie   ", command=self.select_vesicle_movie_detection, width=20, bg="#80818a")
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
        self.track_data_framed={} 
    
        
        self.color_list_plot=["#00FFFF", "#7FFFD4", "#0000FF", "#8A2BE2", "#7FFF00", "#D2691E", "#FF7F50", "#DC143C",
            "#008B8B", "#8B008B", "#FF8C00", "#E9967A", "#FF1493", "#9400D3", "#FF00FF", "#B22222",
            "#FFD700", "#ADFF2F", "#FF69B4", "#ADD8E6", "#F08080", "#90EE90", "#20B2AA", "#C71585", "#FF00FF"]        
        #############################################
    
    
        # Framework: place monitor and view point
        self.viewFrame_linking = tk.Frame(master=linkingFrame, width=int(self.window_width*0.6), height=self.window_height, bg="white")
        self.viewFrame_linking.grid(row=0, column=0, rowspan=2,  pady=self.pad_val, padx=self.pad_val)   


        # place number of paths choice
        self.parametersFrame_linking_path = tk.Frame(master=linkingFrame, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.parametersFrame_linking_path.grid(row=0, column=11, columnspan=1, rowspan=1, pady=self.pad_val, padx=self.pad_val)   
           
        # place parameters and buttons
        self.parametersFrame_linking = tk.Frame(master=linkingFrame, width=int(self.window_width*0.4), height=self.window_height, bg="white")
        self.parametersFrame_linking.grid(row=1, column=11, columnspan=1, rowspan=1, pady=self.pad_val, padx=self.pad_val)    

        
        
        # new detections or not
        
        var_detection_choice = tk.IntVar()
        
        def update_detection_switch():            
            self.detector.detection_choice=var_detection_choice.get()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.d1 = tk.Radiobutton(self.parametersFrame_linking_path, text=" update detections ", variable=var_detection_choice, value=0, bg='white', command =update_detection_switch )
        self.d1.grid(row=2, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)  
        
        self.d2 = tk.Radiobutton(self.parametersFrame_linking_path, text=" use previous detections ", variable=var_detection_choice, value=1, bg='white',command = update_detection_switch ) #  command=sel)
        self.d2.grid(row=2, column=4, columnspan=4,  pady=self.pad_val, padx=self.pad_val)
        

        
        # choice of the pass number 
        
        var_path_number = tk.IntVar()
        
        def update_detection_switch():            
            self.Npass=var_path_number.get()
            self.detector.tracklinking_Npass=var_path_number.get()

        # monitor switch: # 0- show tracks and track numbers, 1- only tracks, 2 - nothing
        self.M1 = tk.Radiobutton(self.parametersFrame_linking_path, text=" single passes ", variable=var_path_number, value=1, bg='white', command =update_detection_switch )
        self.M1.grid(row=3, column=0, columnspan=3, pady=self.pad_val, padx=self.pad_val)  
        
        self.M2 = tk.Radiobutton(self.parametersFrame_linking_path, text=" two passes ", variable=var_path_number, value=2, bg='white',command = update_detection_switch ) #  command=sel)
        self.M2.grid(row=3, column=3, columnspan=3,  pady=self.pad_val, padx=self.pad_val)
        
        self.M3 = tk.Radiobutton(self.parametersFrame_linking_path, text=" three passes ", variable=var_path_number, value=3, bg='white',command = update_detection_switch ) #  command=sel)
        self.M3.grid(row=3, column=6, columnspan=3, pady=self.pad_val, padx=self.pad_val)
        
        
        # TABs parameters
        
        #set the window colour
        
        self.tab_parametersFrame_linking = ttk.Notebook(self.parametersFrame_linking) # create tabs
        
        self.tab_parametersFrame_linking_step1 = ttk.Frame(self.tab_parametersFrame_linking, style="TNotebook")        
        self.tab_parametersFrame_linking.add(self.tab_parametersFrame_linking_step1, text=" main ")
        
        
        self.tab_parametersFrame_linking_step2 = ttk.Frame(self.parametersFrame_linking)
        self.tab_parametersFrame_linking.add(self.tab_parametersFrame_linking_step2, text=" 2nd pass ")

        
        self.tab_parametersFrame_linking_step3 = ttk.Frame(self.parametersFrame_linking)
        self.tab_parametersFrame_linking.add(self.tab_parametersFrame_linking_step3, text=" 3d pass ")        
        
        self.tab_parametersFrame_linking.pack(expand=1, fill='both')    
        
        # linking step 1
        self.parametersFrame_linking_step1 = tk.Frame(master=self.tab_parametersFrame_linking_step1)
        self.parametersFrame_linking_step1.pack(expand=1, fill='both')
        self.parametersFrame_linking_step1.configure(background='white')

        # linking step 2
        self.parametersFrame_linking_step2 = tk.Frame(master=self.tab_parametersFrame_linking_step2)
        self.parametersFrame_linking_step2.pack(expand=1, fill='both')
        self.parametersFrame_linking_step2.configure(background='white')
        
        # linking step 3
        self.parametersFrame_linking_step3 = tk.Frame(master=self.tab_parametersFrame_linking_step3)
        self.parametersFrame_linking_step3.pack(expand=1, fill='both')
        self.parametersFrame_linking_step3.configure(background='white')
        
        # Framework: place monitor 
        self.button_mv = tk.Button(self.viewFrame_linking,text="   Select vesicle movie   ", command=self.select_vesicle_movie_linking, width=20, bg="#80818a")
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

        lbl3 = tk.Label(master=self.gap_frame, text=" ",  bg='white', width=int(self.button_length), height=int(self.button_length/3))
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
        lbl3 = tk.Button(master=self.action_frame, text=" Where to save results ", command=define_result_path, width=int(self.button_length*1.5),bg="#80818a")
        lbl3.grid(row=4, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val)

        # button to set 
        lbl3 = tk.Button(master=self.action_frame, text=" Update the info ", command=update_info, width=int(self.button_length*1.5), bg="#80818a")
        lbl3.grid(row=5, column=0, columnspan=2, pady=self.pad_val, padx=self.pad_val)
          # empty space
        lbl3 = tk.Label(master=self.action_frame, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val)  
        
        # button to run tracking        
        lbl3 = tk.Button(master=self.action_frame, text=" RUN TRACKING  ", command=self.run_tracking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=7, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)

        # show parameters
        self.show_parameters()
        
        ####################################################
     
    # parameters 
    def show_parameters(self):
        #### show parameters : parametersFrame_linking
        
        lbl3 = tk.Label(master=self.information_frame, text=" - - - - - IMPORTANT PATHS: - - - - - ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=0, column=0, columnspan=4, pady=self.pad_val*2, padx=self.pad_val*2) 
        
        lbl3 = tk.Label(master=self.information_frame, text=" Original protein channel:  "+ self.movie_protein_path,  bg='white', wraplength=int(self.window_width*0.2), font=("Helvetica", 8))
        lbl3.grid(row=1, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
        lbl3 = tk.Label(master=self.information_frame, text=" Final result fill be saved to: "+ self.result_path,  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=2, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
        
          # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=3, column=0, columnspan=4,pady=self.pad_val, padx=self.pad_val)  
        
        
        lbl3 = tk.Label(master=self.information_frame, text=" - - - - - PARAMETERS - - - - - ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=4, column=0, columnspan=4, pady=self.pad_val*2, padx=self.pad_val*2) 
        
        
        lbl3 = tk.Label(master=self.information_frame, text=" CANDIDATES DETECTION ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=5, column=0, pady=self.pad_val, padx=self.pad_val) 
        
    # substract_bg_step background substraction step 

        lbl3 = tk.Label(master=self.information_frame, text=" Background subtraction based on  "+ str(self.detector.substract_bg_step)+" frames",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=6, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
    # threshold coef

        lbl3 = tk.Label(master=self.information_frame, text=" Threshold coefficient  "+ str(self.detector.c),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=7, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 

    # sigma
        lbl3 = tk.Label(master=self.information_frame, text=" Sigma from  "+ str(self.detector.sigma_min)+" to "+str(self.detector.sigma_max),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=8, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)

        
    # self.threshold_rel min pix value in relation to the image
    
        lbl3 = tk.Label(master=self.information_frame, text=" Relevant peak height "+str(self.detector.threshold_rel),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
            
          # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white')#, height=int(self.button_length/20))
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)        
                
        
        lbl3 = tk.Label(master=self.information_frame, text=" CANDIDATES PRUNING ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=11, column=0,  pady=self.pad_val, padx=self.pad_val) 

        
    # min_distance min distance minimum distance between two max after MSSEF

        lbl3 = tk.Label(master=self.information_frame, text=" Minimum distance between detections "+str(self.detector.min_distance)+" pix",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        

    #self.box_size # bounding box size for detection
        lbl3 = tk.Label(master=self.information_frame, text=" Region of Interest size "+str(self.detector.box_size)+" pix",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 


    # detection_threshold threshold for the CNN based classification
        lbl3 = tk.Label(master=self.information_frame, text=" Threshold coefficient "+str(self.detector.detection_threshold),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
    # gaussian_fit gaussian fit
        lbl3 = tk.Label(master=self.information_frame, text=" Gaussian fit:  "+str(self.detector.gaussian_fit),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 

    # expected_radius gaussian fit radius 
        lbl3 = tk.Label(master=self.information_frame, text=" Expected particle radius:  "+str(self.detector.expected_radius),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=16, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 

    #self.box_size_fit # bounding box size for detection
        lbl3 = tk.Label(master=self.information_frame, text=" Region for Gaussian fit "+str(self.detector.box_size_fit)+" pix",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=17, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
    # cnn_model cnn model 
        lbl3 = tk.Label(master=self.information_frame, text=" Loaded CNN model: "+self.detector.cnn_model_path.split("/")[-1],  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=18, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
  # # # # # #  # #

         # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white') #, height=int(self.button_length/20))
        lbl3.grid(row=19, column=0, pady=self.pad_val, padx=self.pad_val) 

        
        lbl3 = tk.Label(master=self.information_frame, text=" TRACKER: STEP 1 ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=20, column=0,pady=self.pad_val, padx=self.pad_val) 
        
    # Maximum distance to link 
    
        lbl3 = tk.Label(master=self.information_frame, text=" Maximum distance to link "+str(self.detector.tracker_distance_threshold)+" pix",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=21, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
    # Maximum skipped frames
    
        lbl3 = tk.Label(master=self.information_frame, text=" Maximum skipped frames  "+str(self.detector.tracker_max_skipped_frame),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=22, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
    
    # Maximum track length
        lbl3 = tk.Label(master=self.information_frame, text=" Maximum track length  "+str(self.detector.tracker_max_track_length)+" frames",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=23, column=0, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        
        ######## second colomn ########
              

        lbl3 = tk.Label(master=self.information_frame, text=" TRACKER: STEP 2 - tracklinking: First pass ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=6, column=1, pady=self.pad_val, padx=self.pad_val) 
        
    # Topology
    
        lbl3 = tk.Label(master=self.information_frame, text=" Bayesian network topology: "+self.detector.tracklinking_path1_topology,  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=7, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.information_frame, text=" Connectivity threshold "+str(self.detector.tracklinking_path1_connectivity_threshold),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)        
        
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.information_frame, text=" Temporal gap "+str(self.detector.tracklinking_path1_frame_gap_1)+" frames ", bg='white', font=("Helvetica", 8))
        lbl3.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Distance limit "+str(self.detector.tracklinking_path1_distance_limit)+" pix ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Orientation similarity limit "+str(self.detector.tracklinking_path1_direction_limit)+" degrees ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
    
    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Speed similarity limit "+str(self.detector.tracklinking_path1_speed_limit),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
    
    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Intensity similarity limit "+str(self.detector.tracklinking_path1_intensity_limit),  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=13, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 

        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.information_frame, text=" Threshold of track length "+str(self.detector.tracklinking_path1_track_duration_limit)+" frames ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=14, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 

         # empty space
        lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', font=("Helvetica", 8))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val)    
        
        if self.Npass>1:
            lbl3 = tk.Label(master=self.information_frame, text=" TRACKER: STEP 2 - tracklinking: Second pass ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=16, column=1, pady=self.pad_val, padx=self.pad_val) 
            
        # Topology
        
            lbl3 = tk.Label(master=self.information_frame, text=" Bayesian network topology: "+self.detector.tracklinking_path2_topology,  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=18, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        
        # tracklinking_path1_connectivity_threshold 
            lbl3 = tk.Label(master=self.information_frame, text=" Connectivity threshold "+str(self.detector.tracklinking_path2_connectivity_threshold),  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=19, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)        
             
            #tracklinking_path1_frame_gap_1
            lbl3 = tk.Label(master=self.information_frame, text=" Temporal gap "+str(self.detector.tracklinking_path2_frame_gap_1)+" frames ", bg='white', font=("Helvetica", 8))
            lbl3.grid(row=20, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        #  tracklinking_path1_distance_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Distance limit "+str(self.detector.tracklinking_path2_distance_limit)+" pix ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=21, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        # tracklinking_path1_direction_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Orientation similarity limit "+str(self.detector.tracklinking_path2_direction_limit)+" degrees ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=22, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        # tracklinking_path1_speed_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Speed similarity limit "+str(self.detector.tracklinking_path2_speed_limit),  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=23, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
        # tracklinking_path1_intensity_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Intensity similarity limit "+str(self.detector.tracklinking_path2_intensity_limit),  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=24, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
            
        # tracklinking_path1_track_duration_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Threshold of track length "+str(self.detector.tracklinking_path2_track_duration_limit)+" frames ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=25, column=1, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 

             # empty space
            lbl3 = tk.Label(master=self.information_frame, text=" ",  bg='white', font=("Helvetica", 8)) 
            lbl3.grid(row=26, column=0, pady=self.pad_val, padx=self.pad_val)          



     
        if self.Npass>2:
            lbl3 = tk.Label(master=self.information_frame, text=" TRACKER: STEP 2 - tracklinking: Third pass ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=6, column=2, pady=self.pad_val, padx=self.pad_val) 
            
        # Topology
        
            lbl3 = tk.Label(master=self.information_frame, text=" Bayesian network topology: "+self.detector.tracklinking_path3_topology,  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=7, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        
        # tracklinking_path1_connectivity_threshold 
            lbl3 = tk.Label(master=self.information_frame, text=" Connectivity threshold "+str(self.detector.tracklinking_path3_connectivity_threshold),  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=8, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
            
             
            #tracklinking_path1_frame_gap_1
            lbl3 = tk.Label(master=self.information_frame, text=" Temporal gap "+str(self.detector.tracklinking_path3_frame_gap_1)+" frames ", bg='white', font=("Helvetica", 8))
            lbl3.grid(row=9, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        #  tracklinking_path1_distance_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Distance limit "+str(self.detector.tracklinking_path3_distance_limit)+" pix ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=10, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        # tracklinking_path1_direction_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Orientation similarity limit "+str(self.detector.tracklinking_path3_direction_limit)+" degrees ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=11, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)
        
        # tracklinking_path1_speed_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Speed similarity limit "+str(self.detector.tracklinking_path3_speed_limit),  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=12, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
        # tracklinking_path1_intensity_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Intensity similarity limit "+str(self.detector.tracklinking_path3_intensity_limit),  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=13, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W) 
        
            
        # tracklinking_path1_track_duration_limit
            lbl3 = tk.Label(master=self.information_frame, text=" Threshold of track length "+str(self.detector.tracklinking_path3_track_duration_limit)+" frames ",  bg='white', font=("Helvetica", 8))
            lbl3.grid(row=14, column=2, pady=self.pad_val, padx=self.pad_val, sticky=tk.W)         
     
      # # # # # # # # # # # # # # # # # # # # # # # # # 
  
    def set_linking_parameters_frame(self):
        
        ##### step 1 #####

        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" TRACKER: STEP 1 ",  bg='white')
        lbl3.grid(row=0, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Maximum distance to link 
    
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Maximum distance to link, pix ",  bg='white')
        lbl3.grid(row=1, column=0) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracker_distance_threshold))
        self.l_tracker_distance_threshold = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracker_distance_threshold.grid(row=1, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # Maximum skipped frames
    
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Maximum skipped frames  ",  bg='white')
        lbl3.grid(row=2, column=0) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracker_max_skipped_frame))
        self.l_tracker_max_skipped_frame = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracker_max_skipped_frame.grid(row=2, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # Maximum track length
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Maximum track length  ",  bg='white')
        lbl3.grid(row=3, column=0)
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracker_max_track_length))
        self.l_tracker_max_track_length = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracker_max_track_length.grid(row=3, column=1, pady=self.pad_val, padx=self.pad_val)
        
        
        
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=4, column=0, pady=self.pad_val, padx=self.pad_val)        
        
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" TRACKER: first pass of tracklinking ",  bg='white')
        lbl3.grid(row=5, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Topology
    
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Bayesian network topology ",  bg='white')
        lbl3.grid(row=7, column=0)     
        self.comboTopology_1 = ttk.Combobox(master=self.parametersFrame_linking_step1, 
                            values=[
                                    "complete", 
                                    "no_intensity",
                                    "no_orientation",
                                    "no_motion",
                                    "no_gap"]) # comboTopology.get()
        self.comboTopology_1.grid(row=7, column=1) 
        self.comboTopology_1.current(0)
    
    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Connectivity threshold [0,1]  ",  bg='white')
        lbl3.grid(row=8, column=0)
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_connectivity_threshold))
        self.l_tracklinking_path1_connectivity_threshold = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_connectivity_threshold.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val) 
        
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Temporal gap, frames ", bg='white')
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_frame_gap_1))
        self.l_tracklinking_path1_frame_gap_1 = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_frame_gap_1.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Distance limit, pix ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_distance_limit))
        self.l_tracklinking_path1_distance_limit = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_distance_limit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Orientation similarity limit, degrees ",  bg='white')
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_direction_limit))
        self.l_tracklinking_path1_direction_limit = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_direction_limit.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Speed similarity limit ",  bg='white')
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_speed_limit))
        self.l_tracklinking_path1_speed_limit = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_speed_limit.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Intensity similarity limit, partition ",  bg='white')
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_intensity_limit))
        self.l_tracklinking_path1_intensity_limit = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_intensity_limit.grid(row=13, column=1, pady=self.pad_val, padx=self.pad_val)
    
    
        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Threshold of track length, frames ",  bg='white')
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.tracklinking_path1_track_duration_limit))
        self.l_tracklinking_path1_track_duration_limit = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.l_tracklinking_path1_track_duration_limit.grid(row=14, column=1, pady=self.pad_val, padx=self.pad_val)
     
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val) 
    
    # test range 
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" Testing from frame  ",  bg='white')
        lbl3.grid(row=16, column=0)
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.start_frame))
        self.start_frame = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.start_frame.grid(row=16, column=1, pady=self.pad_val, padx=self.pad_val)
    
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" to frame  ", bg='white')
        lbl3.grid(row=16, column=2, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_linking_step1, value=str(self.detector.end_frame))
        self.end_frame = tk.Entry(self.parametersFrame_linking_step1, width=self.button_length, text=v)
        self.end_frame.grid(row=16, column=3, pady=self.pad_val, padx=self.pad_val)
        
      # # # # # #  # #
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step1, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=18, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step1, text=" Run test ", command=self.run_test_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=19, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step1, text=" Save to file ", command=self.save_to_file_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=20, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step1, text=" Read from file ", command=self.read_from_file_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=21, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   


        
        ##### step 2 #####          
        
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=4, column=0, pady=self.pad_val, padx=self.pad_val)        
        
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" TRACKER: second pass of tracklinking ",  bg='white')
        lbl3.grid(row=5, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Topology
    
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Bayesian network topology ",  bg='white')
        lbl3.grid(row=7, column=0)     
        self.comboTopology_2 = ttk.Combobox(master=self.parametersFrame_linking_step2, 
                            values=[
                                    "complete", 
                                    "no_intensity",
                                    "no_orientation",
                                    "no_motion",
                                    "no_gap"]) # comboTopology.get()
        self.comboTopology_2.grid(row=7, column=1) 
        self.comboTopology_2.current(0)
    
    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Connectivity threshold [0,1]  ",  bg='white')
        lbl3.grid(row=8, column=0)
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_connectivity_threshold))
        self.l_tracklinking_path2_connectivity_threshold = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_connectivity_threshold.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val) 
        
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Temporal gap, frames ", bg='white')
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_frame_gap_1))
        self.l_tracklinking_path2_frame_gap_1 = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_frame_gap_1.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Distance limit, pix ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_distance_limit))
        self.l_tracklinking_path2_distance_limit = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_distance_limit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Orientation similarity limit, degrees ",  bg='white')
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_direction_limit))
        self.l_tracklinking_path2_direction_limit = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_direction_limit.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Speed similarity limit ",  bg='white')
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_speed_limit))
        self.l_tracklinking_path2_speed_limit = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_speed_limit.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Intensity similarity limit, partition ",  bg='white')
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_intensity_limit))
        self.l_tracklinking_path2_intensity_limit = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_intensity_limit.grid(row=13, column=1, pady=self.pad_val, padx=self.pad_val)
    
    
        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" Threshold of track length, frames ",  bg='white')
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step2, value=str(self.detector.tracklinking_path2_track_duration_limit))
        self.l_tracklinking_path2_track_duration_limit = tk.Entry(self.parametersFrame_linking_step2, width=self.button_length, text=v)
        self.l_tracklinking_path2_track_duration_limit.grid(row=14, column=1, pady=self.pad_val, padx=self.pad_val)
     
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val) 

        
      # # # # # #  # #
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step2, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=18, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step2, text=" Run test ", command=self.run_test_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=19, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step2, text=" Save to file ", command=self.save_to_file_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=20, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step2, text=" Read from file ", command=self.read_from_file_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=21, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   
            

        
        ##### step 3 #####
        
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=4, column=0, pady=self.pad_val, padx=self.pad_val)      
        
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" TRACKER: third pass of tracklinking ",  bg='white')
        lbl3.grid(row=5, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 
        
    # Topology
    
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Bayesian network topology ",  bg='white')
        lbl3.grid(row=7, column=0)     
        self.comboTopology_3 = ttk.Combobox(master=self.parametersFrame_linking_step3, 
                            values=[
                                    "complete", 
                                    "no_intensity",
                                    "no_orientation",
                                    "no_motion",
                                    "no_gap"]) # comboTopology.get()
        self.comboTopology_3.grid(row=7, column=1) 
        self.comboTopology_3.current(0)
    
    
    # tracklinking_path1_connectivity_threshold 
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Connectivity threshold [0,1]  ",  bg='white')
        lbl3.grid(row=8, column=0)
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_connectivity_threshold))
        self.l_tracklinking_path3_connectivity_threshold = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_connectivity_threshold.grid(row=8, column=1, pady=self.pad_val, padx=self.pad_val) 
         
        #tracklinking_path1_frame_gap_1
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Temporal gap, frames ", bg='white')
        lbl3.grid(row=9, column=0, pady=self.pad_val, padx=self.pad_val)
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_frame_gap_1))
        self.l_tracklinking_path3_frame_gap_1 = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_frame_gap_1.grid(row=9, column=1, pady=self.pad_val, padx=self.pad_val)
    
    #  tracklinking_path1_distance_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Distance limit, pix ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_distance_limit))
        self.l_tracklinking_path3_distance_limit = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_distance_limit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_direction_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Orientation similarity limit, degrees ",  bg='white')
        lbl3.grid(row=11, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_direction_limit))
        self.l_tracklinking_path3_direction_limit = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_direction_limit.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_speed_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Speed similarity limit ",  bg='white')
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_speed_limit))
        self.l_tracklinking_path3_speed_limit = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_speed_limit.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # tracklinking_path1_intensity_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Intensity similarity limit, partition ",  bg='white')
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_intensity_limit))
        self.l_tracklinking_path3_intensity_limit = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_intensity_limit.grid(row=13, column=1, pady=self.pad_val, padx=self.pad_val)
    
    
        
    # tracklinking_path1_track_duration_limit
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" Threshold of track length, frames ",  bg='white')
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_linking_step3, value=str(self.detector.tracklinking_path3_track_duration_limit))
        self.l_tracklinking_path3_track_duration_limit = tk.Entry(self.parametersFrame_linking_step3, width=self.button_length, text=v)
        self.l_tracklinking_path3_track_duration_limit.grid(row=14, column=1, pady=self.pad_val, padx=self.pad_val)
     
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=15, column=0, pady=self.pad_val, padx=self.pad_val) 
    
        
      # # # # # #  # #
    
         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_linking_step3, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=18, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step3, text=" Run test ", command=self.run_test_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=19, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step3, text=" Save to file ", command=self.save_to_file_linking, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=20, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_linking_step3, text=" Read from file ", command=self.read_from_file_linking, width=self.button_length*2, bg="#80818a")
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
        
        
    # self.threshold_rel min pix value in relation to the image
    
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Relevant peak height ",  bg='white')
        lbl3.grid(row=4, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.threshold_rel))
        self.d_threshold_rel = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_threshold_rel.grid(row=4, column=1, pady=self.pad_val, padx=self.pad_val)

    # button to show MSSEF
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Show MSSEF ", command=self.show_mssef, width=self.button_length)
        lbl3.grid(row=4, column=3, pady=self.pad_val, padx=self.pad_val)   


            
          # empty space
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=5, column=0, pady=self.pad_val, padx=self.pad_val)        
                
        
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" CANDIDATES PRUNING ",  bg='white')
        lbl3.grid(row=6, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val) 

    # min_distance min distance minimum distance between two max after MSSEF

        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Minimum distance between detections  ",  bg='white')
        lbl3.grid(row=7, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.min_distance))
        self.d_min_distance = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_min_distance.grid(row=7, column=1, pady=self.pad_val, padx=self.pad_val)
        
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
    
    # gaussian_fit 
        def clickgaussian_fit():
            self.detector.gaussian_fit=self.gaussianValue.get()
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Gaussian fit (True/False) ",  bg='white')
        lbl3.grid(row=10, column=0, pady=self.pad_val, padx=self.pad_val) 
        self.gaussianValue=tk.BooleanVar()
        self.gaussianValue.set(True)
        self.d_gaussian_fit = tk.Checkbutton(self.parametersFrame_detection, text='', var=self.gaussianValue, command=clickgaussian_fit)
        self.d_gaussian_fit.grid(row=10, column=1, pady=self.pad_val, padx=self.pad_val)

    #self.box_size_fit 

        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Region for Gaussian fit ",  bg='white')
        lbl3.grid(row=11, column=0) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.box_size_fit))
        self.d_box_size_fit = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_box_size_fit.grid(row=11, column=1, pady=self.pad_val, padx=self.pad_val)
        
    # expected_radius
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" Expected particle radius ",  bg='white')
        lbl3.grid(row=12, column=0, pady=self.pad_val, padx=self.pad_val) 
        v=tk.StringVar(self.parametersFrame_detection, value=str(self.detector.expected_radius))
        self.d_expected_radius = tk.Entry(self.parametersFrame_detection, width=self.button_length, text=v)
        self.d_expected_radius.grid(row=12, column=1, pady=self.pad_val, padx=self.pad_val)
    
    # cnn_model cnn model 
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Load CNN model ", command=self.load_cnn_model, width=self.button_length*2)
        lbl3.grid(row=13, column=0, pady=self.pad_val, padx=self.pad_val)  
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=self.detector.cnn_model_path.split("/")[-1],  bg='white')
        lbl3.grid(row=13, column=1, columnspan=3, pady=self.pad_val, padx=self.pad_val) 
    
  # # # # # #  # #

         # empty space
        lbl3 = tk.Label(master=self.parametersFrame_detection, text=" ",  bg='white', height=int(self.button_length/20))
        lbl3.grid(row=14, column=0, pady=self.pad_val, padx=self.pad_val) 
         # buttons   
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Run test ", command=self.run_test_detection, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=15, column=0,  columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Save to file ", command=self.save_to_file_detection, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=16, column=0, columnspan=4, pady=self.pad_val, padx=self.pad_val)   
        lbl3 = tk.Button(master=self.parametersFrame_detection, text=" Read from file ", command=self.read_from_file_detection, width=self.button_length*2, bg="#80818a")
        lbl3.grid(row=17, column=0,  columnspan=4,pady=self.pad_val, padx=self.pad_val)   
    

    def show_mssef(self):
        
        # read parameters
        self.detection_frame=self.frame_pos
        self.collect_detection_parameters()

        # movie 
        self.detector.movie=self.movie
        
        # generate MSSEF
        mssef=self.detector.get_mssef(self.frame_pos)
        
        # create new window
        fig_mssef=plt.figure()
        plt.imshow(mssef, cmap="gray")
        plt.axis('off')


        self.novi = tk.Toplevel()
        self.novi.title(" MSSEF image ")
        self.canvas_mssef = FigureCanvasTkAgg(fig_mssef, master=self.novi)
        self.canvas_mssef.get_tk_widget().pack(expand = tk.YES, fill = tk.BOTH)
        self.canvas_mssef.draw()
        
        def new_home( *args, **kwargs):
            # zoom out
            plt.xlim(0,self.movie.shape[2])
            plt.ylim(0,self.movie.shape[1])
            plt.imshow(mssef, cmap="gray")
            self.canvas_mssef.draw()
            
        NavigationToolbar2Tk.home = new_home
        # toolbar
        toolbar = NavigationToolbar2Tk(self.canvas_mssef, self.novi)
        toolbar.set_message=lambda x:"" # remove message with coordinates
        toolbar.update()        

        
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

        # parameters: gaussian fit    
        if self.d_box_size_fit.get()!='':
            self.detector.box_size_fit=int(self.d_box_size_fit.get())  
            
            
        if self.d_expected_radius.get()!='':
            self.detector.expected_radius=float(self.d_expected_radius.get())
            
            
            
        self.show_parameters()
        
    def run_test_detection(self):
        
        # read parameters from the buttons
        self.detection_frame=self.frame_pos
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
        print(" expected_radius", self.detector.expected_radius)
        print(" box_size_fit", self.detector.box_size_fit)
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
        
                # read limits
#        print(matplotlib.get_backend())
#        print (self.ax.dataLim)
        xlim_old=self.axd.get_xlim()
        ylim_old=self.axd.get_ylim()
        
        # plot image
        self.image = self.movie[self.frame_pos,:,:]/np.max(self.movie[self.frame_pos,:,:])
        
        self.axd.clear() # clean the plot 
        self.axd.imshow(self.image, cmap="gray")
        self.axd.axis('off')  
      
        
        # plot results        
   
        if self.monitor_switch_detection==1 and self.detection_frame==self.frame_pos: # candidates
            if len(self.detector.detection_candidates)>0:
                for i in range(0, len(np.asarray(self.detector.detection_candidates))):
                    circle=plt.Circle((np.asarray(self.detector.detection_candidates)[i,1], np.asarray(self.detector.detection_candidates)[i,0]), 3, color="b", fill=False)
                    self.axd.add_artist(circle)    
        elif self.monitor_switch_detection==2: # detection
            if len(self.detector.detection_vesicles)>0 and self.detection_frame==self.frame_pos:
                for i in range(0, len(np.asarray(self.detector.detection_vesicles))):
                    circle=plt.Circle((np.asarray(self.detector.detection_vesicles)[i,1], np.asarray(self.detector.detection_vesicles)[i,0]), 3, color="r", fill=False)
                    self.axd.add_artist(circle)    

        
        #set the same "zoom"
        self.axd.set_xlim(xlim_old[0],xlim_old[1])
        self.axd.set_ylim(ylim_old[0],ylim_old[1])  

        # inver y-axis as set_ylim change the orientation
        if ylim_old[0]<ylim_old[1]:
            self.axd.invert_yaxis()  
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.figd, master=self.viewFrame_detection)
        self.canvas.get_tk_widget().grid(row=5, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        toolbarFrame = tk.Frame(master=self.viewFrame_detection)
        toolbarFrame.grid(row=10, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        
        # update home button

        def new_home( *args, **kwargs):
            # zoom out
            self.axd.set_xlim(0,self.movie.shape[2])
            self.axd.set_ylim(0,self.movie.shape[1])

            self.show_frame_detection()
            
        NavigationToolbar2Tk.home = new_home
        
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
            
            #set the same "zoom"
            
            self.axl.set_xlim(0,self.movie.shape[2])
            self.axl.set_ylim(0,self.movie.shape[1])

            self.axd.set_xlim(0,self.movie.shape[2])
            self.axd.set_ylim(0,self.movie.shape[1])            
            
        
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
            
            #set the same "zoom"
            self.axd.set_xlim(0,self.movie.shape[2])
            self.axd.set_ylim(0,self.movie.shape[1])
            
            self.axl.set_xlim(0,self.movie.shape[2])
            self.axl.set_ylim(0,self.movie.shape[1])
        
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
        
        # parameters: TRACKER: create tracklets
        if self.l_tracker_distance_threshold.get()!='':
            self.detector.tracker_distance_threshold=int(self.l_tracker_distance_threshold.get())
            
        if self.l_tracker_max_skipped_frame.get()!='':
            self.detector.tracker_max_skipped_frame=int(self.l_tracker_max_skipped_frame.get())
            
        if self.l_tracker_max_track_length.get()!='':
            self.detector.tracker_max_track_length=int(self.l_tracker_max_track_length.get())
            
            
        # parameters: TRACKER: 1st pass of tracklinking    
        if self.comboTopology_1.get()!='':
            self.detector.tracklinking_path1_topology=str(self.comboTopology_1.get())
            
        if self.l_tracklinking_path1_connectivity_threshold.get()!='':
            self.detector.tracklinking_path1_connectivity_threshold=float(self.l_tracklinking_path1_connectivity_threshold.get())
            
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

        # parameters: TRACKER: 2nd pass of tracklinking    
        if self.comboTopology_2.get()!='':
            self.detector.tracklinking_path2_topology=str(self.comboTopology_2.get())
            
        if self.l_tracklinking_path2_connectivity_threshold.get()!='':
            self.detector.tracklinking_path2_connectivity_threshold=float(self.l_tracklinking_path2_connectivity_threshold.get())
        
        if self.l_tracklinking_path2_frame_gap_1.get()!='':
            self.detector.tracklinking_path2_frame_gap_1=int(self.l_tracklinking_path2_frame_gap_1.get())
            
        if self.l_tracklinking_path2_distance_limit.get()!='':
            self.detector.tracklinking_path2_distance_limit=float(self.l_tracklinking_path2_distance_limit.get())
            
        if self.l_tracklinking_path2_direction_limit.get()!='':
            self.detector.tracklinking_path2_direction_limit=float(self.l_tracklinking_path2_direction_limit.get())
            
        if self.l_tracklinking_path2_speed_limit.get()!='':
            self.detector.tracklinking_path2_speed_limit=float(self.l_tracklinking_path2_speed_limit.get())
            
        if self.l_tracklinking_path2_intensity_limit.get()!='':
            self.detector.tracklinking_path2_intensity_limit=float(self.l_tracklinking_path2_intensity_limit.get())


        # parameters: TRACKER: 3d pass of tracklinking    
        if self.comboTopology_3.get()!='':
            self.detector.tracklinking_path3_topology=str(self.comboTopology_3.get())
            
        if self.l_tracklinking_path3_connectivity_threshold.get()!='':
            self.detector.tracklinking_path3_connectivity_threshold=float(self.l_tracklinking_path3_connectivity_threshold.get())
    
        if self.l_tracklinking_path3_frame_gap_1.get()!='':
            self.detector.tracklinking_path3_frame_gap_1=int(self.l_tracklinking_path3_frame_gap_1.get())
            
        if self.l_tracklinking_path3_distance_limit.get()!='':
            self.detector.tracklinking_path3_distance_limit=float(self.l_tracklinking_path3_distance_limit.get())
            
        if self.l_tracklinking_path3_direction_limit.get()!='':
            self.detector.tracklinking_path3_direction_limit=float(self.l_tracklinking_path3_direction_limit.get())
            
        if self.l_tracklinking_path3_speed_limit.get()!='':
            self.detector.tracklinking_path3_speed_limit=float(self.l_tracklinking_path3_speed_limit.get())
            
        if self.l_tracklinking_path3_intensity_limit.get()!='':
            self.detector.tracklinking_path3_intensity_limit=float(self.l_tracklinking_path3_intensity_limit.get())            
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
        print("\n ----------------parameters -----------------")
        print(" tracker_distance_threshold", self.detector.tracker_distance_threshold)
        print(" tracker_max_skipped_frame", self.detector.tracker_max_skipped_frame)
        print(" tracker_max_track_length ", self.detector.tracker_max_track_length)
        
        print(" \n number of pass, ", self.detector.tracklinking_Npass)
        print(" topology", self.detector.tracklinking_path1_topology)
        print(" tracklinking_path1_connectivity_threshold", self.detector.tracklinking_path1_connectivity_threshold)
        print(" tracklinking_path1_frame_gap", self.detector.tracklinking_path1_frame_gap_1)
        print(" tracklinking_path1_distance_limit", self.detector.tracklinking_path1_distance_limit)
        print(" tracklinking_path1_direction_limit", self.detector.tracklinking_path1_direction_limit)
        print(" tracklinking_path1_speed_limit", self.detector.tracklinking_path1_speed_limit)
        print(" tracklinking_path1_intensity_limit", self.detector.tracklinking_path1_intensity_limit)
        
        if self.detector.tracklinking_Npass>1:
            print("\n topology 2", self.detector.tracklinking_path2_topology)
            print(" tracklinking_path2_connectivity_threshold", self.detector.tracklinking_path2_connectivity_threshold)
            print(" tracklinking_path2_frame_gap_1", self.detector.tracklinking_path2_frame_gap_1)
            print(" tracklinking_path2_distance_limit", self.detector.tracklinking_path2_distance_limit)
            print(" tracklinking_path2_direction_limit", self.detector.tracklinking_path2_direction_limit)
            print(" tracklinking_path2_speed_limit", self.detector.tracklinking_path2_speed_limit)
            print(" tracklinking_path2_intensity_limit", self.detector.tracklinking_path2_intensity_limit)

        if self.detector.tracklinking_Npass>2:
            print("\n topology 3", self.detector.tracklinking_path1_topology)
            print(" tracklinking_path3_connectivity_threshold", self.detector.tracklinking_path3_connectivity_threshold)
            print(" tracklinking_path3_frame_gap_1", self.detector.tracklinking_path3_frame_gap_1)
            print(" tracklinking_path3_distance_limit", self.detector.tracklinking_path3_distance_limit)
            print(" tracklinking_path3_direction_limit", self.detector.tracklinking_path3_direction_limit)
            print(" tracklinking_path3_speed_limit", self.detector.tracklinking_path3_speed_limit)
            print(" tracklinking_path3_intensity_limit", self.detector.tracklinking_path3_intensity_limit)
        
        print(" start_frame", self.detector.start_frame)
        print(" end_frame", self.detector.end_frame)
        
        print("\n running tracking ...")

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

        # read limits
        xlim_old=self.axl.get_xlim()
        ylim_old=self.axl.get_ylim()
        
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
                self.axl.plot(np.asarray(trace)[:,1],np.asarray(trace)[:,0],  self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])     
                self.axl.text(np.asarray(trace)[0,1],np.asarray(trace)[0,0], str(p['trackID']), fontsize=10, color=self.color_list_plot[int(p['trackID'])%len(self.color_list_plot)])

        
        #set the same "zoom"
        self.axl.set_xlim(xlim_old[0],xlim_old[1])
        self.axl.set_ylim(ylim_old[0],ylim_old[1])
        

        # inver y-axis as set_ylim change the orientation
        if ylim_old[0]<ylim_old[1]:
            self.axl.invert_yaxis()
            
        
        # DrawingArea
        self.canvas = FigureCanvasTkAgg(self.figl, master=self.viewFrame_linking)
        self.canvas.get_tk_widget().grid(row=5, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        self.canvas.draw()
        
        # toolbar
        toolbarFrame = tk.Frame(master=self.viewFrame_linking)
        toolbarFrame.grid(row=10, column=2, columnspan=5, pady=self.pad_val, padx=self.pad_val)
        
        # update home button

        def new_home( *args, **kwargs):
            # zoom out
        
            self.axl.set_xlim(0,self.movie.shape[2])
            self.axl.set_ylim(0,self.movie.shape[1])


            self.show_frame_linking()
            
        NavigationToolbar2Tk.home = new_home
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbarFrame)
        self.toolbar.set_message=lambda x:"" # remove message with coordinates
        self.toolbar.update()


 ############# Running the code  ##################

    def run_tracking(self):
        '''
        running the final tracking 
        '''
        
        # switch to the detection mode
        self.detector.detection_choice=0
        
        # read start and end frame
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
