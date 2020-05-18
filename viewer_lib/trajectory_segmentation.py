#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory segmentation based on the motion type
v0.1: segmnetation into 2 classes: directed motion and the rest 

"""

import numpy as np
import math
from scipy import stats
import pandas as pd
from viewer_lib.model_1dunet import unet

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class TrajectorySegment(object):
    """
    Class to perform trajectory segmentation based on the motion type
    
    Attributes
    ----------

    trace : list (default is empty)
        list of coordinates in the trajectory 
    frames : list (default is empty) 
        list of frames in the trajectory      

    
    
    Methods
    ---------
    msd_based_segmentation(self, trace)
    msd_slope(self, track)
    compute_msd(self, trajectory, t_step, coords=['x', 'y'])
    
    
    unet_based_segmentation(self, trace)

    from_cartesian_to_polar(self, x,y)

    
    """
    def __init__(self):
        """Initialize variables
        """
        
        self.trace=[]
        self.frames=[]
        self.window_length=8
        self.unet_threshold=0.5
        self.limit_segment_length=2
        
        
        
        # load the model
        
        self.model = unet( input_size = (self.window_length,2))
        self.model.load_weights('dl_weight/1Dunet_jointdata_val_acc-0.84.hdf5')
        
        
    def from_cartesian_to_polar(self, x,y):
        '''
        switching coordinate system
        '''
        r=math.sqrt(x**2+y**2)
        angle=math.degrees(math.atan2(y,x))
        return r, angle
     
    def unet_segmentation(self,trace):
        '''
        trajectory segmentation based on the 1D Unet architecture
        '''
        track_length=len(trace)
        Nch=2 # trained on two values (r and theta)
        
        # check the length of the track and extand if needed
        if track_length<self.window_length:
            new_track=np.zeros((self.window_length,Nch))+trace[-1]
            new_track[0:track_length, :]=trace
            new_track=new_track.tolist()                        
        else:
            new_track=trace
        
        new_segment=np.zeros((len(new_track),1))
        real_segment=np.zeros((len(new_track),1))
        
        # rolling window over the samples
        for pos in range(8, np.max((9,track_length)),4): 
    
            track=new_track[pos-8: pos]
            
            #calculate trajectory in radial coordinates
            trajectory=[]
            point_zero=track[0]
            for i in range(0,len(track)):
                point=track[i]
                r, angle=self.from_cartesian_to_polar(point[0]-point_zero[0],point[1]-point_zero[1])
    
                trajectory.append((r, angle))
    
            trajectory=np.asarray(trajectory)
            trajectory_test=np.zeros((self.window_length, Nch))
            trajectory_test[0:trajectory.shape[0],:]=np.asarray(trajectory)
    
            trajectory_test=trajectory_test.reshape((1, trajectory_test.shape[0], trajectory_test.shape[1]))
    
            # pass to the unet
            results = self.model.predict(trajectory_test,  steps=1)
            real_results=np.copy(results[0])
            semgentation_unet_result=np.zeros(real_results.shape)
            semgentation_unet_result[real_results>=self.unet_threshold]=1
            
            #pass to segmnetation array
            if pos==8:
                new_segment[0: pos]=semgentation_unet_result[:]
                real_segment[0: pos]=real_results[:]
            else:
                new_segment[pos-6: pos]=semgentation_unet_result[2:]
                real_segment[pos-6: pos]=real_results[2:]
            
            
        #save the last window to tge segmentation array
        if pos!=track_length and track_length>=8:
            track=new_track[track_length-8: track_length]
            
            
            #calculate trajectory in radial coordinates
            trajectory_r=[]
            point_zero=track[0]
            for i in range(0,len(track)):
                point=track[i]
                r, angle=self.from_cartesian_to_polar(point[0]-point_zero[0],point[1]-point_zero[1])
                trajectory_r.append((r, angle))
    
            trajectory_r=np.asarray(trajectory_r)
            trajectory_r_test=np.zeros((self.window_length, Nch))
            trajectory_r_test[0:trajectory_r.shape[0],:]=np.asarray(trajectory_r)
    
            trajectory_r_test=trajectory_r_test.reshape((1, trajectory_r_test.shape[0], trajectory_r_test.shape[1]))
    
          
            # pass to the unet
            results = self.model.predict(trajectory_r_test,  steps=1)
            real_results=np.copy(results[0])
            semgentation_unet_result=np.zeros(real_results.shape)

            semgentation_unet_result[real_results>=self.unet_threshold]=1
            #pass to segmnetation array
            new_segment[track_length-6:track_length]=semgentation_unet_result[2:]
            real_segment[track_length-6:track_length]=real_results[2:]
            
        # check on small segments:
        
        segmented_traj=self.remove_small_segments(new_segment.reshape((new_segment.shape[0])), self.limit_segment_length)
        
        # print segmentation results
        print("\n")
        for n in range(0,len(real_segment)):
            print(real_segment[n], " ->  ", new_segment[n], ' ->>  ', segmented_traj[n])
        
        return segmented_traj
    
    def msd_based_segmentation(self, trace):
        '''
        trajectory segmentation based on MSD and line fitting:
        1 - directed movement; 0 - other motion types
        The result is limited to the long trajectory length
        '''
      
        segmentation_msd_based=np.zeros(len(trace))
        for i in range(0, len(trace)):

            start_mini_track=np.max((0, int(i-self.window_length/2)))
            end_mini_track=np.min((int(i+self.window_length/2), len(trace)))
            mini_track=trace[start_mini_track:end_mini_track]
            slope=self.msd_slope(mini_track)
            
            if slope<=1:
                segmentation_msd_based[i]=0
            else:
                segmentation_msd_based[i]=1    
                
        
        # check on small segments:
        
        segmented_traj=self.remove_small_segments(segmentation_msd_based, self.limit_segment_length)
                
        return segmented_traj
        
    def msd_slope(self, track):
        '''
        evaluates msd slope based on the the provided part of the trajectory (track)
        '''
        t = np.linspace(0, len(track), len(track))
        dt=1 # sec
        traj = pd.DataFrame({'t': t, 'x': np.asarray(track)[:,0], 'y': np.asarray(track)[:,1]})
        msd, tau = self.compute_msd(traj, t_step=dt, coords=['x', 'y'])
            
        slope, intercept, r_value, p_value, std_err = stats.linregress(t[:5], msd[:5])
        
        return slope


    def compute_msd(self, trajectory, t_step, coords=['x', 'y']):
        '''
        compute MSD of the provided trajectory
        '''
    
        tau = trajectory['t'].copy()
        shifts = np.floor(tau / t_step).astype(np.int)
        msds = np.zeros(shifts.size)
        msds_std = np.zeros(shifts.size)
    
        for i, shift in enumerate(shifts):
            diffs = trajectory[coords] - trajectory[coords].shift(-shift)
            sqdist = np.square(diffs).sum(axis=1)
            msds[i] = sqdist.mean()
            msds_std[i] = sqdist.std()
    
        return msds, tau
    
    def remove_small_segments(self, segmented=[],segment_length=1):
        '''
        remove too short segment
        '''
        new_segment=np.copy(segmented)
        segment=np.diff(segmented)
        loc=np.where(np.abs(segment)==1)
        d=np.diff(loc)[0]
        
        for pos in range(0, len(d)):
            if (pos-1)>=0:
                v_before=d[pos-1]
            else:
                v_before=10
                    
            if (pos+1)<len(d):
                v_after=d[pos+1]
            else:
                v_after=10
                
            val=d[pos]
            # if the segment is between two big pieces
            if val<=segment_length and v_before>segment_length+1 and v_after>segment_length+1:
                actual_pos=loc[0][pos]+1 # position in the original array
                actual_new_pos=loc[0][pos]
                new_val=segmented[actual_new_pos]
                new_segment[actual_pos:actual_pos+val]=new_val
                
        
        return new_segment.tolist()
        #
#        
#
#if __name__ == "__main__":
#    segmented=[1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,]
#    print(np.asarray(segmented))
#    results=TrajectorySegment.remove_small_segments(segmented, segmented)
#    
#    print(results)
    