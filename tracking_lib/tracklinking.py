#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import pylab as plt
import numpy as np
import math
from pgmpy.inference import VariableElimination
import json
import skimage
from skimage import io
from tqdm import tqdm
import scipy as sp


class GraphicalModelTracking(object):
    """
    Class to perform track linking with Bayesian network: 
    The code builds a BN based with chosen topology and a given conditional probabilities
    The connection between tracklets are  based on the probability of the connectivity variable.
    
    Attributes
    ----------

    topology : str (default is 'complete')
        type of the BN tolopogy: 'complete', 'no_gap', 
        'no_speed', 'no_orientation', 'no_intensity', 'no_motion' 
    movie : array  (default is []) 
        original array of the image sequency      
    tracklets: dict (default is{})
        input tracklets 
    tracks: dict  (default is{}) 
        final tracks 
    tracks_before_filter: dict  (default is{}) 
        output tracks before short and not-moving tracks are removed 
    data: dict (default is{}) 
        arrange tracks to savei n json file 
    bgm_tracklet (BayesianModel())
        Bayesian model for tracklingking 
    track1: dict (default is{}) 
        first tracklet of the pair for comparison 
    track2: dict (default is{}) 
        second tracklet of the pair for comparison 
    tracklets_connection: list (default is [])  
        list of tracjlet pairs for connection  
    track_pos: int (default is 0)
        track ID for the  connected tracks, iteratively increasing 
    track_data_framed: dict (default is{}) 
        tracklets rearranged in frame based order 
    track_data_framed_start: dict (default is{}) 
        tracklets rearranged in staring frame order base
    track_data_framed_end: dict (default is{}) 
        tracklets rearranged in ending frame order base

    # tracklinking parameters:
    
    frame_search_range: int (default is 6)
        frame range to search connection between tracks   
    distance_search_range: int (default is 12)
        distance range to search connection between tracks    
    frame_gap_tracklinking_0: int (default is 1)        
    frame_gap_tracklinking_1: int (default is 5)  
        upper limit for the frame gap      
    direction_limit_tracklinking: int (default 50)
        limit for orientation similarity
    distance_limit_tracklinking: int (deafault 10)
        limit of the distance in pix between two tracklets to be connected
    connectivity_threshold: float [0,1] (default 0.8)
        threshold for the tracklet connection 
    track_displacement_limit: float (default is 0) 
        minimum displacement of the final track 
    track_duration_limit: int(default is 1)  
        minimum number of frames per final track
    speed_limit_tracklinking: float (default 0.5)
        proportional speed difference between traklets to be connected 
    intensity_limit_tracklinking: float (default 0.4)
        intensity difference between tracklets to be connected
    roi_size: int (default 12)
        dimension of the roi with the object in it 
    
    
    Methods
    ---------
    tracklets_pgm()
    save_tracks(filename) 
    decision_tracklets_connection(track1, track2)
    decision_s(val_o, val_l)
    decision_p(val_s, val_g)
    decision_o()
    decision_g()
    decision_l()
    decision_c()
    calculate_intensity(point, frameN))
    decision_in()
    decision_sp()
    decision_d()
    calculate_speed(trace, frames)
    calculate_direction(trace)
    join_tracklets(connections)
    rearrange_track_to_frame_start_end(tracklets, movie)
    train_model(data)
    predic_connectivity_score(predict_data)
    connect_tracklet_time()
    
    """
    def __init__(self):
        """Initialize variables
        """
        
        self.topology='complete' # type of the BN topology: 
                  #complete, no_gap, no_speed, no_orientation, no_intensity, no_motion
        self.movie=[]
        self.tracklets={} # input tracklets
        self.tracks={} # output tracks
        self.tracks_before_filter={} # output tracks before short and not-moving tracks are removed
        self.data={}
        self.bgm_tracklet=BayesianModel() 
        self.track1={}
        self.track2={}
        
        self.tracklets_connection=[]
        self.track_pos=0 # ID for the new rearrange tracks
        self.track_data_framed={} # data rearrange by frames
        self.track_data_framed_start={}
        self.track_data_framed_end={}
        
        # parameters:
        
        self.frame_search_range=6 # frame range to search connection between tracks
        self.distance_search_range=12 # distance range to search connection between tracks
        
        self.frame_gap_tracklinking_0=1
        self.frame_gap_tracklinking_1=5        
        self.direction_limit_tracklinking=50
        self.distance_limit_tracklinking=10 # distance in pix between two tracklets to be connected
        self.connectivity_threshold=0.8
        
        self.track_displacement_limit=0 # minimum displacement of the final track
        self.track_duration_limit=1 # minimum number of frames per final track
        
        self.speed_limit_tracklinking=0.5 # proportional speed difference between traklets to be connected
        
        self.intensity_limit_tracklinking=0.4 # intensity difference between tracklets to be connected 
        self.roi_size=12 # dimension of the roi with the object in it 
        
    def tracklets_pgm(self):
        '''
        Define the BN topology
        '''
    
######### nodes:

# O  - order of the tracks (0-not in order, 1 -  in order) 
# G  - gap between tracks (0 - zero frames, 1 - from 1-5 frames, 2 - more than 5 frames)
# L  - overLap of the tracks (0 - no overlap, 1 - overlap)
# S  - sequence - join of overlap and order (0 - not affinity, 1- affinity)
# P  - position - join of sequence and gap(0 - not affinity, 1- affinity)
# D  - direction similarity  (0 - not similar, 1- similar)
# C  - coordinates similarity (0 - not near, 1 - near)
# SP - speed similarity  (0 - not similar, 1- similar)
# I  - intensity similarity  (0 - not similar, 1- similar)
# A  - connectivity score  (0 , 1)


        # Defining individual CPDs.
        cpd_o = TabularCPD(variable='O', variable_card=2, values=[[0.9, 0.1]])
        cpd_l = TabularCPD(variable='L', variable_card=2, values=[[0.9, 0.1]])
        cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.99, 1.0, 0.01, 0.9],
                                                                  [0.01, 0.0, 0.99, 0.1]], evidence=['O','L'], evidence_card=[2,2])
        
        cpd_g = TabularCPD(variable='G', variable_card=3, values=[[0.3, 0.3, 0.4]]) 
        
        cpd_p = TabularCPD(variable='P', variable_card=2, values=[[0.99, 0.99, 0.99, 0.01, 0.1, 0.9],
                                                                  [0.01, 0.01, 0.01, 0.99, 0.9, 0.1]], evidence=['S','G'], evidence_card=[2,3])     
        
        cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.4, 0.6]])   
        
        cpd_sp = TabularCPD(variable='SP', variable_card=2, values=[[0.5, 0.5]])    
        
        cpd_m = TabularCPD(variable='M', variable_card=2, values=[[0.99, 0.1, 0.1, 0.2],
                                                                  [0.01, 0.9, 0.9, 0.8]], evidence=['D','SP'], evidence_card=[2,2])
        
        cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.1, 0.9]])
        
        cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.5, 0.5]]) 
        
        
        # the score cpd:
        
        if self.topology=='complete':
        # all
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'P'), ('G', 'P'),('P', 'A'),('D', 'M'),('SP', 'M'),('M', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[1.0, 0.95, 0.99, 0.95, 0.95, 0.25, 0.95, 0.05,  1.0, 0.95, 0.99, 0.9, 0.9, 0.2, 0.9, 0.0],
                                                                      [0.0, 0.05, 0.01, 0.05, 0.05, 0.75, 0.05, 0.95,  0.0, 0.05, 0.01, 0.1, 0.1, 0.8, 0.1, 1.0]], 
                                                                    evidence=['I','P','M', 'C'], evidence_card=[2,2,2,2])     # not done    
            self.bgm_tracklet.add_cpds(cpd_o, cpd_g, cpd_s, cpd_l, cpd_p, cpd_d, cpd_c, cpd_a, cpd_sp, cpd_m, cpd_i) # associating the CPDs with the network
            
        elif self.topology=='no_intensity':
        # no intensity
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'P'), ('G', 'P'),('P', 'A'),('D', 'M'),('SP', 'M'),('M', 'A'),('C', 'A')]) # graphical model
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[1.0, 0.95, 0.99, 0.9, 0.9, 0.2, 0.9, 0.0],
                                                                      [0.0, 0.05, 0.01, 0.1, 0.1, 0.8, 0.1, 1.0]], 
                                                                    evidence=['P','M', 'C'], evidence_card=[2,2,2]) 
            self.bgm_tracklet.add_cpds(cpd_o, cpd_g, cpd_s, cpd_l, cpd_p, cpd_d, cpd_c, cpd_a, cpd_sp, cpd_m) # associating the CPDs with the network
        
        elif self.topology=='no_orientation':
#        # no orientation
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'P'), ('G', 'P'),('P', 'A'),('SP', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[1.0, 0.95, 0.99, 0.95, 0.95, 0.25, 0.95, 0.05,  1.0, 0.95, 0.99, 0.9, 0.9, 0.2, 0.9, 0.0],
                                                                      [0.0, 0.05, 0.01, 0.05, 0.05, 0.75, 0.05, 0.95,  0.0, 0.05, 0.01, 0.1, 0.1, 0.8, 0.1, 1.0]], 
                                                                    evidence=['I','P','SP', 'C'], evidence_card=[2,2,2,2])     # not done    
            self.bgm_tracklet.add_cpds(cpd_o, cpd_g, cpd_s, cpd_l, cpd_p, cpd_c, cpd_a, cpd_sp, cpd_i) # associating the CPDs with the network
            
        elif self.topology=='no_speed':
            # no speed
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'P'), ('G', 'P'),('P', 'A'),('D', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[1.0, 0.95, 0.99, 0.95, 0.95, 0.25, 0.95, 0.05,  1.0, 0.95, 0.99, 0.9, 0.9, 0.2, 0.9, 0.0],
                                                                      [0.0, 0.05, 0.01, 0.05, 0.05, 0.75, 0.05, 0.95,  0.0, 0.05, 0.01, 0.1, 0.1, 0.8, 0.1, 1.0]], 
                                                                    evidence=['I','P','D', 'C'], evidence_card=[2,2,2,2])     # not done    
            self.bgm_tracklet.add_cpds(cpd_o, cpd_g, cpd_s, cpd_l, cpd_p, cpd_d, cpd_c, cpd_a, cpd_i) # associating the CPDs with the network
           
        elif self.topology=='no_motion':
        # no movement
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'P'), ('G', 'P'),('P', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[ 0.99, 0.95,  0.95, 0.05,  0.99, 0.9, 0.9, 0.0],
                                                                      [ 0.01, 0.05,  0.05, 0.95,  0.01, 0.1, 0.1, 1.0]], 
                                                                    evidence=['I','P','C'], evidence_card=[2,2,2])     # not done    
            self.bgm_tracklet.add_cpds(cpd_o, cpd_g, cpd_s, cpd_l, cpd_p, cpd_c, cpd_a, cpd_i) # associating the CPDs with the network

        elif self.topology=='no_gap':
        # no gap
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'A'),('D', 'M'),('SP', 'M'),('M', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
    
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[1.0, 0.95, 0.99, 0.95, 0.95, 0.25, 0.95, 0.05,  1.0, 0.95, 0.99, 0.9, 0.9, 0.2, 0.9, 0.0],
                                                                      [0.0, 0.05, 0.01, 0.05, 0.05, 0.75, 0.05, 0.95,  0.0, 0.05, 0.01, 0.1, 0.1, 0.8, 0.1, 1.0]], 
                                                                    evidence=['I','S','M', 'C'], evidence_card=[2,2,2,2])     # not done    
            self.bgm_tracklet.add_cpds(cpd_o, cpd_s, cpd_l, cpd_d, cpd_c, cpd_a, cpd_sp, cpd_m, cpd_i)    # associating the CPDs with the network     
            
        elif self.topology=='no_sequence':
        # no gap
            self.bgm_tracklet.add_edges_from([ ('G', 'A'),('D', 'M'),('SP', 'M'),('M', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
    
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[1.0, 0.95, 0.99, 0.95, 0.98, 0.35, 0.9, 0.25, 0.95, 0.25, 0.95, 0.05,     1.0, 0.95, 0.99, 0.9, 0.99, 0.3, 0.85, 0.15, 0.9, 0.2, 0.9, 0.0],
                                                                      [0.0, 0.05, 0.01, 0.05, 0.02, 0.65, 0.1, 0.75, 0.05, 0.75, 0.05, 0.95,     0.0, 0.05, 0.01, 0.1, 0.01, 0.7, 0.15, 0.85, 0.1, 0.8, 0.1, 1.0]], 
                                                                    evidence=['I','G','M', 'C'], evidence_card=[2,3,2,2])     # not done    
            self.bgm_tracklet.add_cpds( cpd_g, cpd_d, cpd_c, cpd_a, cpd_sp, cpd_m, cpd_i)    # associating the CPDs with the network         
        
        else:
            print("the BN topology cannot be identify - check the self.topology parameter")
        print("Chosen topology: ", self.topology )
        
        # check model: defined and sum to 1.
        self.bgm_tracklet.check_model()
          


    def save_tracks(self, filename):
        '''
        save the final tracks
        '''
        with open(filename, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False)
        

            
    def decision_tracklets_connection(self, track1, track2):
        '''
        make a decision on two tracks to be connected
        '''

        self.track1=track1
        self.track2=track2
        
        # calculates values for the nodes 
        
        # order of the tracks (0-not in order, 1 -  in order)
        val_O=self.decision_o()

        # gap between tracks (0 - zero frames, 1 - from 1-5 frames, 2 - more than 5 frames)
        val_G=self.decision_g()

        # overLap of the tracks (0 - no overlap, 1 - overlap)
        val_L=self.decision_l()

        # coordinates similarity (0 - not near, 1 - near)
        val_C=self.decision_c()

        # orientation similarity  (0 - not similar, 1- similar)
        val_D=self.decision_d()

        # speed similarity  (0 - not similar, 1- similar)
        val_SP=self.decision_sp()

        # intensity similarity  (0 - not similar, 1- similar)
        val_I=self.decision_in()
        
        
        # make a query based on the calculated nodes
        infer = VariableElimination(self.bgm_tracklet) 
        
        if self.topology=='complete':
        #all
            q=infer.query(['A'], evidence={'O': val_O, 'G': val_G, 'L': val_L, 'C':val_C, 'D':val_D, 'SP': val_SP, 'I':val_I}, joint=False)['A'].values[1] 
        
        elif self.topology=='no_intensity':
        #no intensity
            q=infer.query(['A'], evidence={'O': val_O, 'G': val_G, 'L': val_L, 'C':val_C, 'D':val_D, 'SP': val_SP}, joint=False)['A'].values[1] 

        elif self.topology=='no_speed':
        #no speed
            q=infer.query(['A'], evidence={'O': val_O, 'G': val_G, 'L': val_L, 'C':val_C, 'D':val_D, 'I':val_I}, joint=False)['A'].values[1] 

        elif self.topology=='no_orientation':
        #no orientation
            q=infer.query(['A'], evidence={'O': val_O, 'G': val_G, 'L': val_L, 'C':val_C, 'SP': val_SP, 'I':val_I}, joint=False)['A'].values[1] 

        elif self.topology=='no_motion':
        #no motion
            q=infer.query(['A'], evidence={'O': val_O, 'G': val_G, 'L': val_L, 'C':val_C, 'I':val_I}, joint=False)['A'].values[1] 

        elif self.topology=='no_gap':
        #no gap
            q=infer.query(['A'], evidence={'O': val_O, 'L': val_L, 'C':val_C, 'D':val_D, 'SP': val_SP, 'I':val_I}, joint=False)['A'].values[1]  

        elif self.topology=='no_sequence':
        #no gap
            q=infer.query(['A'], evidence={'G':val_G, 'D':val_D, 'SP': val_SP, 'I':val_I}, joint=False)['A'].values[1] 
        else:
            q=0
            print("the BN topology cannot be identify - check the self.topology parameter")

#        print(" tracks ", self.track1["trackID"], self.track2["trackID"], " score: ", q)
#        print(" val_O", val_O," val_G", val_G,  " val_ L", val_L, " val_C", val_C, " val_D", val_D, " val_SP", val_SP, " val_I", val_I, " val_I", val_I, "\n") 
        
        return q 
    
    
    def decision_s(self, val_o, val_l):
        '''
        decision on sequence - join overlap and order(0 - not affinity, 1- affinity)
        '''
        if val_o==1 and val_l==0:
            val_s=1
        else:
            val_s=0
        return val_s
    
    
    def decision_p(self, val_s, val_g):
        '''
        decision on position - join of sequence and gap(0 - not affinity, 1- affinity)
        '''
        if val_s==0:
            val=0
        elif val_g==2:
            val=0
        else:
            val=1
        return val
        
    
    def decision_o(self):
        '''
        decision on order (0-not in order, 1 -  in order)
        '''
        pose_1_end=self.track1['frames'][-1]
        pos_2_start=self.track2['frames'][0]
        
        if pose_1_end<=pos_2_start:
            val=1 # there is an overlap
        else:
            val=0 # there is no overlap
        
#        print("O: ", val)
        return val
        
    def decision_g(self):
        '''
        decision on  gap (0 - zero frames, 1 - small gap, 2 - big gap)
        '''
        
        #calculate the gap
        gap=self.track2['frames'][0]-self.track1['frames'][-1]
        
        if gap==self.frame_gap_tracklinking_0:
            val=0 # no gap
        elif gap>self.frame_gap_tracklinking_0 and gap<=self.frame_gap_tracklinking_1:
            val=1
        else: # if the value is negative or mpre than 5
            val=2 # there is no overlap
        
#        print("G: ", val)
        return val

    def decision_l(self):
        '''
        decision on overlapping (0 - no overlap, 1 - overlap)
        '''
        gap=self.track2['frames'][0]-self.track1['frames'][-1]
        
        if gap<=0:
            val=1
        else:
            val=0
#        print("gap: ", gap)
#        print("L: ", val)
        return val

    def decision_c(self):
        '''
        decision on coordinates (0 - not near, 1 - near)
        '''
        #distance between tracks
        
        dist=np.sqrt((self.track2['trace'][0][0]-self.track1['trace'][-1][0])**2+(self.track2['trace'][0][1]-self.track1['trace'][-1][1])**2)
        N_frames_travelled=self.track2['frames'][0]-self.track1['frames'][-1]
#        print("frames ", N_frames_travelled)
#        print("track 2 ", self.track2['trace'])
#        print("distance: ", dist)
#        print("distance per frame : ", dist/N_frames_travelled)
        if dist/N_frames_travelled<=self.distance_limit_tracklinking:
            val = 1
        else:
            val = 0
#        print("distance: ", dist)
#        print("C: ", val)
        return val       
    
    def calculate_intensity(self, point, frameN):
        '''
        calculate intensity
        '''
        patch_size=self.roi_size
        x_min=int(point[0]-patch_size/2)
        x_max=int(point[0]+patch_size/2)
        y_min=int(point[1]-patch_size/2)
        y_max=int(point[1]+patch_size/2)
            
        if x_min>0 and y_min>0 and x_max<self.movie.shape[1] and y_max<self.movie.shape[2]:
                
            # create img
            track_img = self.movie[frameN, x_min:x_max, y_min:y_max]
            track_img = (track_img-np.min(track_img))/(np.max(track_img)-np.min(track_img))
            #calculate mean intensity inside the segment
            
            intensity=np.sum(track_img)/(patch_size*patch_size)            
        else:
            intensity=0
            
        return intensity

    def decision_in(self):
        '''
        decision on intensity (0 - not similar, 1- similar)
        '''
        
        #calculate intensity
        int1=self.calculate_intensity(self.track1['trace'][-1], self.track1['frames'][-1])
        int2=self.calculate_intensity(self.track2['trace'][0], self.track2['frames'][0])
        
        #comparison
        difference=abs(int1-int2)
        
        if difference>self.intensity_limit_tracklinking:
            val=0
        else:
            val=1
            
        return val      


    def decision_sp(self):
        '''
        decision on speed  (0 - not similar, 1- similar)
        '''

        if len(self.track1['trace'])==1 and len(self.track2['trace'])==1:
            speed1=0
            speed2=0
        elif len(self.track1['trace'])>1 and len(self.track2['trace'])>1:
        
            speed1= self.calculate_speed(self.track1['trace'], self.track1['frames'])
            speed2= self.calculate_speed(self.track2['trace'], self.track2['frames'])  

        elif len(self.track1['trace'])>1 and len(self.track2['trace'])==1:
            speed1= self.calculate_speed(self.track1['trace'], self.track1['frames'])
            speed2= self.calculate_speed(self.track1['trace']+self.track2['trace'], self.track1['frames']+self.track2['frames'])

        elif len(self.track1['trace'])==1 and len(self.track2['trace'])>1:
            speed1= self.calculate_speed(self.track1['trace']+self.track2['trace'], self.track1['frames']+self.track2['frames'])
            speed2= self.calculate_speed(self.track2['trace'], self.track2['frames'])
            
        difference=abs(speed2-speed1)/np.max((speed2,speed1))
            
        if difference>self.speed_limit_tracklinking:
            val = 0
        else:
            val = 1
#        print("D: ", difference, "  ", val)
        
        return val 
        
    def decision_d(self):
        '''
        decision on direction  (0 - not similar, 1- similar)
        '''
        
        if len(self.track1['trace'])==1 and len(self.track2['trace'])==1:
            direction1=0
            direction2=0
        elif len(self.track1['trace'])>1 and len(self.track2['trace'])>1:
        
            direction1= self.calculate_direction(self.track1['trace'])
            direction2= self.calculate_direction(self.track2['trace'])  

        elif len(self.track1['trace'])>1 and len(self.track2['trace'])==1:
            direction1= self.calculate_direction(self.track1['trace'])
            direction2= self.calculate_direction(self.track1['trace']+self.track2['trace'])

        elif len(self.track1['trace'])==1 and len(self.track2['trace'])>1:
            direction1= self.calculate_direction(self.track1['trace']+self.track2['trace'])
            direction2= self.calculate_direction(self.track2['trace'])
            
        difference=abs(direction2-direction1)
            
        if difference>self.direction_limit_tracklinking:
            val = 0
        else:
            val = 1
        
        return val  
    
    def calculate_speed(self, trace, frames):
        '''
        calculating the tracklet speed
        '''

        dist=np.sqrt((trace[-1][0]-trace[0][0])**2+(trace[-1][1]-trace[0][1])**2)
        time_frame=frames[-1]-frames[0]
        
        return dist/time_frame
        
    
    def calculate_direction(self, trace):
        '''
        calculating the tracklet direction
        '''        
        dist=np.sqrt((trace[-1][0]-trace[0][0])**2+(trace[-1][1]-trace[0][1])**2)
        if len(trace)>=2 and dist>=1:
            p1=trace[0]                        
            p2=trace[-1]
            xDiff = p2[0] - p1[0]
            yDiff = p2[1] - p1[1]
            return int(math.degrees(math.atan2(yDiff, xDiff)))
        else:
            return 0


            
    def join_tracklets(self,connections):
        '''
        join the tracklets in the final tracks based on the provided connections
        '''
        
        new_track={}
        frames=[]
        trace=[]

        
        for i in connections:

            tracklet=self.tracklets[str(int(i))]
            # join frames
            frames=frames+tracklet['frames']
            
            #join trace 
            trace=trace+tracklet['trace']
            
            # add missing frames
            pos=0
            new_frames=[]
            new_trace=[]
            for frame_pos in range(frames[0], frames[-1]+1):
                frame=frames[pos]
                
                if frame_pos==frame:
                    new_frames.append(frame_pos)
                    new_trace.append(trace[pos])
                    pos=pos+1
                else:
                    new_frames.append(frame_pos)
                    new_trace.append(trace[pos])             
        
        # create new Track 
        new_track.update({'trackID': self.track_pos, 'frames': new_frames, 'trace': new_trace}) 
        self.tracks_before_filter.update({self.track_pos:new_track}) 
        
        self.track_pos=self.track_pos+1 #trackID +1 for the next time
            

    def rearrange_track_to_frame_start_end(self, tracklets, movie):
        '''
        change data arrangment from tracks to frames
        '''
        
        self.tracklets=tracklets
        
        for n_frame in range(0, movie.shape[0]):
            #rearrange the data
            track_list_start=[]
            track_list_end=[]
            for p in self.tracklets:
                tracklet=self.tracklets[p]
                frames=tracklet['frames']
                if n_frame==frames[0]: # if the frame at the beginning
                    track_list_start.append(tracklet['trackID'])
           

                if n_frame==frames[-1]: # if the frame is at the end
                    track_list_end.append(tracklet['trackID'])

            if len(track_list_start)!=0:
                self.track_data_framed_start.update({n_frame:track_list_start})
                
            if len(track_list_end)!=0:
                self.track_data_framed_end.update({n_frame:track_list_end})
                
    def train_model(self, data):       
        ''' train the PGM '''
        self.bgm_tracklet.fit(data)
        self.bgm_tracklet.get_cpds()
#        print(self.bgm_tracklet.get_cpds(('A')))
    
    def predic_connectivity_score(self, predict_data):
        '''
        predict the connectivity score
        '''
        infer = VariableElimination(self.bgm_tracklet) 
        
        return infer.query(['A'], evidence=predict_data)['A'].values[1]


        
    def connect_tracklet_time(self):
        '''
        the main function for tracklinking: 
        connects tracklets (check connectivity and make a decision) 
        in the order of their temproal positioning
        '''
        checked_connections=[] # list of reviewed connections       
        possible_connections=[] # list of connection is a high connect. score
        
        #iterate over each frame with ending tracks
        print("scanning frames for possble connections ...")
        for n_frame in tqdm(self.track_data_framed_end):
#            print("frame ", n_frame)
            tracklets_to_check=[]
            
            #for the distance calculation
            possible_connection_ID_end=[]
            possible_connection_ID_start=[]
            possible_connection_end_x=[]
            possible_connection_end_y=[]
            possible_commection_start_x=[]
            possible_commection_start_y=[]
            
            # iterate over frame and form a list with possible connections
            # based on the start point
            for frame_pos in range(n_frame,n_frame+self.frame_search_range): #!!! start from n_frame+1
 
                if frame_pos in self.track_data_framed_start:
                    for trackID in self.track_data_framed_start[frame_pos]:
                         # get track ID
                        tracklets_to_check.append(trackID)
                        
                    #!!! calculate distance and check connectivity here ?
                        
                        # first position of the tracklet
                        track_one=self.tracklets.get(str(trackID))
                        
                        possible_commection_start_x.append(track_one['trace'][0][0]) 
                        possible_commection_start_y.append(track_one['trace'][0][1]) 
           
            # based on the end point           
            for t1 in self.track_data_framed_end[n_frame]:

                track1=self.tracklets.get(str(t1))
                possible_connection_end_x.append(track1['trace'][-1][0])
                possible_connection_end_y.append(track1['trace'][-1][1])

            
            #define arrays:
            possible_connection_ID_end=self.track_data_framed_end[n_frame]
            possible_connection_ID_start=tracklets_to_check
            
            possible_connection_ID=[] # list of all the possible connections
            
            for pos_end in possible_connection_ID_end:
                new_line=[]
                for pos_start in possible_connection_ID_start:
                    new_line.append([pos_end, pos_start])
                    
                possible_connection_ID.append(new_line)
            possible_connection_ID=np.asarray(possible_connection_ID)
            
            # Coordinates
            possible_connection_end_x=np.asarray(possible_connection_end_x).reshape((len(possible_connection_end_x),1))
            possible_connection_end_y=np.asarray(possible_connection_end_y).reshape((len(possible_connection_end_y),1))
            
            possible_commection_start_x=np.asarray(possible_commection_start_x).reshape((1, len(possible_commection_start_x)))
            possible_commection_start_y=np.asarray(possible_commection_start_y).reshape((1, len(possible_commection_start_y)))
            
            matrix_possible_connection_end_x=np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_connection_end_x            
            matrix_possible_commection_start_x = np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_commection_start_x

            matrix_possible_connection_end_y=np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_connection_end_y
            matrix_possible_commection_start_y = np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_commection_start_y

            # all distance between end and start
            matrix_dist=np.sqrt((matrix_possible_commection_start_x-matrix_possible_connection_end_x)**2+(matrix_possible_commection_start_y-matrix_possible_connection_end_y)**2)
            
            # filter the connections based on the spacial distance
            comparison_list=possible_connection_ID[matrix_dist<=self.distance_search_range]

            # look through comparison list and calculate connectivity score
            pos_check=0
            for ID_compare in comparison_list:
                if ID_compare[0]!=ID_compare[1] and [ID_compare[0], ID_compare[1]] not in checked_connections:

                    track1=self.tracklets.get(str(ID_compare[0]))
                    track2=self.tracklets.get(str(ID_compare[1]))

                    checked_connections.append([ID_compare[0], ID_compare[1]])
                    connectivity_score=self.decision_tracklets_connection(track1, track2)
#                    print("tracklets", [ID_compare[0], ID_compare[1]]," possible connection", connectivity_score)
                    
                    # remove pairs with small connectivity score
                    if connectivity_score>=self.connectivity_threshold:
                        
                        possible_connections.append([track1['trackID'],track2['trackID'], connectivity_score])
                        pos_check+=1
#                        print("tracklets", [ID_compare[0], ID_compare[1]]," possible connection", connectivity_score)
                else:

                    checked_connections.append([ID_compare[0], ID_compare[1]])   

            
      # # # compare results and connect the tracks
#        print("connecting tracklets ....")
        

        connected_tracklets=[] # new connections
        
        print("looking through possible connections  ... ")
        
        print("POSSIBLE CONNECTIONS", possible_connections)
        #iterste over the possible connections and make a decision
        for p in tqdm(range(0, len(possible_connections))):
            
            pair1=np.asarray(possible_connections)[p,:].tolist() # extract the connected tracks
            
            # check in existing connection:            
            if not connected_tracklets:
                start_connection_exist=False
                end_connection_exist=False
            else:
                start_connection_exist=pair1[0] in np.asarray(connected_tracklets)[:,0]
                end_connection_exist=pair1[1] in np.asarray(connected_tracklets)[:,1]
            
            if start_connection_exist==False and end_connection_exist==False:                              
                # check the first and choose highest
                
                # check another condidates for the beginning  tracklet
                comparison_list_start=[]
                first_tracklet=self.tracklets.get(str(int(pair1[0])))['frames'][-1]
                second_tracklet=self.tracklets.get(str(int(pair1[1])))['frames'][0]

                gap_list=[]#gap_frame] # list of the gap values  for the connections
                repeating_list_x, repeating_list_y=np.where((np.asarray(possible_connections)==pair1[0]))
                # repeating_list_x- is a position in the list of connection
                # repeating_list_y - is beginning (0) or ending (1) tracklet
                
                # iterate over all the possible connection with the tracklet
                for l in range(0, len(repeating_list_x)):
                    
                    if repeating_list_y[l]==0: # similar connection
                        comparison_list_start.append(possible_connections[repeating_list_x[l]])
                        
                        first_tracklet=self.tracklets.get(str(int(possible_connections[repeating_list_x[l]][0])))['frames'][-1]
                        second_tracklet=self.tracklets.get(str(int(possible_connections[repeating_list_x[l]][1])))['frames'][0]
                        
                        gap_list.append(second_tracklet-first_tracklet-1) # number of frames between the tracklets                   
                        
                
            # best choice for the beggining tracklet and delete all others
                best_choice_start_val=np.max(np.asarray(comparison_list_start)[:,2])
                
                #find all the option with max probability score
                best_choice_start_pos_array=np.where(np.asarray(comparison_list_start)[:,2]==best_choice_start_val)
                
                # it is the only one pick it
                if len(best_choice_start_pos_array)==1:                   
                    best_choice_start_pos=np.argmax(np.asarray(comparison_list_start)[:,2])
                    
                else: # if not - choose the one with the smallest time gap
                    best_choice_start_pos=np.argmin(np.asarray(gap_list))                
                        
                # check another condidates for the end tracklet
                comparison_list_end=[]
                
                repeating_list_x, repeating_list_y=np.where((np.asarray(possible_connections)==pair1[1]))  
                gap_list=[]  # list of the gap values  for the connections
                
                for l in range(0, len(repeating_list_x)):            
                    if repeating_list_y[l]==1: # similar connection
                        comparison_list_end.append(possible_connections[repeating_list_x[l]])
                        
                        first_tracklet=self.tracklets.get(str(int(possible_connections[repeating_list_x[l]][0])))['frames'][-1]
                        second_tracklet=self.tracklets.get(str(int(possible_connections[repeating_list_x[l]][1])))['frames'][0]
                        
                        gap_list.append(second_tracklet-first_tracklet-1) # number of frames between the tracklets          
                        
                
            # best choice for the beggining tracklet and delete all others
                best_choice_end_val=np.max(np.asarray(comparison_list_end)[:,2]) 
                best_choice_end_pos_array=np.where(np.asarray(comparison_list_end)[:,2]==best_choice_end_val)
                best_choice_end_pos=np.argmax(np.asarray(comparison_list_end)[:,2])
                
                # it is the only one pick it
                if len(best_choice_end_pos_array)==1:                   
                    best_choice_end_pos=np.argmax(np.asarray(comparison_list_end)[:,2])
                    
                else: # if not - choose the one with the smallest time gap
                    best_choice_end_pos=np.argmin(np.asarray(gap_list))  
                    
            #choose the right connection:
                if best_choice_start_val>=best_choice_end_val:
                    
                    #save the connection
                    connected_tracklets.append(comparison_list_start[best_choice_start_pos])
                    
                else:
                    connected_tracklets.append(comparison_list_end[best_choice_end_pos])
#                    print("connected by end: ", comparison_list_end[best_choice_end_pos])
 #               print("   connected: ", connected_tracklets[-1])

        # store connectivity score value
        new_connected_tracklets=[]
        for p in connected_tracklets:
            new_connected_tracklets.append(p[0:2])
            
        connected_tracklets=new_connected_tracklets

        print("\n \n connected_tracklets \n", connected_tracklets)
        # save not connected tracklets
        print("saving not connected tracklets ... ")
        for t3 in tqdm(self.tracklets, "saving not connected tracklets"):
            trackID=self.tracklets.get(t3)['trackID']
            if len(connected_tracklets)>0:
                c_var_x, c_var_y=np.where((np.asarray(connected_tracklets)==trackID))
                if len(c_var_x)==0:
                    self.join_tracklets([trackID]) 
            else:
                self.join_tracklets([trackID])

        
        # connect connected tracklets
        
        print("checking connected tracklets ... ")
        complete=False
        
        while complete==False:
            
            added_list_start=[]
            added_list_end=[]
            N_connections=0 # number of connections
#            print(N_connections)
            
            for track in connected_tracklets: #tqdm(connected_tracklets):
                if track[0] not in added_list_start and track[-1] not in added_list_end:
                    new_track=track
                    added_list_start.append(track[0])
                    added_list_end.append(track[1])
                    
                    pos_list=0
                
                    # find trackelts for connection
                    start_repeating_list_x=[] # position in the list
                    start_repeating_list_y=[] # is matching the start
                    for i in connected_tracklets:
                        
                        start_repeating_list_y.append(np.asarray(i)[-1]==track[0])
                        start_repeating_list_x.append(pos_list)
                        pos_list+=1
                    start_repeating_list_x=np.asarray(start_repeating_list_x)
                    start_repeating_list_y=np.asarray(start_repeating_list_y)
                        
                    
                    
                    # check the position at the start for connection
                    for l in range(0, len(start_repeating_list_x)):            
                            if start_repeating_list_y[l]==True: # connection before this track
                                # add connections 
                                track_to_add=connected_tracklets[start_repeating_list_x[l]]
                                if track_to_add[0] not in added_list_start:
                                    new_track=track_to_add[0:-1]+new_track # add it to the track
                                    added_list_start.append(track_to_add[0])
                                    added_list_end.append(track[0])
                                    N_connections=N_connections+1                                    
                                    
                    # check possible connection for end 
                    end_repeating_list_x=[]
                    end_repeating_list_y=[]
                    pos_list=0
                    for i in connected_tracklets:
                        
                        end_repeating_list_y.append(np.asarray(i)[0]==track[-1]) 
                        end_repeating_list_x.append(pos_list)
                        pos_list+=1
                    end_repeating_list_x=np.asarray(end_repeating_list_x)
                    end_repeating_list_y=np.asarray(end_repeating_list_y)
                    
                    # check the position at the end for connections    
                    for l in range(0, len(end_repeating_list_x)):       
                            if end_repeating_list_y[l]==True:  # connection after this track
                                track_to_add=connected_tracklets[end_repeating_list_x[l]] 
                                if track_to_add[-1] not in added_list_end:
                                   new_track=new_track+track_to_add[1:]
                                   added_list_end.append(track_to_add[-1])
                                   added_list_start.append(track[-1])
                                   N_connections=N_connections+1

                    # add the track in connections
                    self.tracklets_connection.append(new_track)
            print("N_connections ", N_connections)
#            print(len(self.tracklets_connection))
            if N_connections>0:
                connected_tracklets=self.tracklets_connection
                self.tracklets_connection=[]
            else:
                complete=True
                
        print("\n possible_connections \n", possible_connections)
        print("\n self.tracklets_connection \n", self.tracklets_connection)
        print("saving connected tracklets ... ")
        for tracklets_to_track in tqdm(self.tracklets_connection):
            
            self.join_tracklets(tracklets_to_track)
            
            
        # eliminate short and small displacement tracks:
        print("filtering tracks...")
        for t in tqdm(self.tracks_before_filter):
            
            track=self.tracks_before_filter.get(t)
#            print(track)
            if len(track['trace'])>0:
                dist=np.sqrt((track['trace'][0][0]-track['trace'][-1][0])**2+(track['trace'][0][1]-track['trace'][-1][1])**2)
                duration=track['frames'][-1]-track['frames'][0]
            else:
                dist=-1
                duration=-1
            if dist>self.track_displacement_limit and duration>self.track_duration_limit:
                self.tracks.update({t:track})
                
                # this is for transition to the distionary storage - to display in list yet
                self.data.update({t:track})

#        print("\n self.tracks \n", self.tracks)   
        print("ALL the tracks: ", len(self.tracks))            