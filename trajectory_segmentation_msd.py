#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory segmentation based on the motion type
v0.1: segmnetation into 2 classes: directed motion and the rest 

@author: mariaa
"""

import numpy as np
import math
from scipy import stats
import pandas as pd
import skimage
from skimage import io

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

    msd_based_decision(self, traj)
    unet_based_decision(self, traj)

    from_cartesian_to_polar(self, x,y)

    
    """
    def __init__(self):
        """Initialize variables
        """
        
        self.trace=[]
        self.frames=[]
        self.window_length=8
        self.unet_threshold=0.3

        
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
                
        return segmentation_msd_based
        
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
    
        
        