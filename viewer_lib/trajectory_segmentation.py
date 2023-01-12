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
import skimage

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
        self.model.load_weights('viewer_lib/1Dunet_jointdata_val_acc-0.84.hdf5')
        
        
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
        
#        # print segmentation results
#        print("\n")
#        for n in range(0,len(real_segment)):
#            print(real_segment[n], " ->  ", new_segment[n], ' ->>  ', segmented_traj[n])
        
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
    
    
    def calculate_speed(self, track, mode="average"): # mode: "average"/"moving"
        '''
        calculate speed of vesicle movement
        
    output: 
        curvilinear_speed_mean
        straightline_speed
        curvilinear_speed_all
        
        '''
        
        trajectory=track['trace']
        frames=track['frames']
        motion=track['motion']

        # Mean curvilinear speed

        # separated arrays for coordinates
        if len(frames)>1:
            x_1=np.asarray(trajectory)[1:,0]    
            y_1=np.asarray(trajectory)[1:,1]   
    
            x_2=np.asarray(trajectory)[0:-1,0]    
            y_2=np.asarray(trajectory)[0:-1,1]  
        else:
            x_1=np.asarray(trajectory)[:,0]    
            y_1=np.asarray(trajectory)[:,1]   
    
            x_2=np.asarray(trajectory)[:,0]    
            y_2=np.asarray(trajectory)[:,1]              


        # calculate the discplacement

        sqr_disp_back=np.sqrt((x_1-x_2)**2+(y_1-y_2)**2)        

        if mode=="average":  
            
            # sum of all the displacements                   
            disp=np.sum(sqr_disp_back)
            
            # frames        
            time=(frames[-1]-frames[0])

        else: # movement mode
            
            disp=np.sum(np.asarray(motion[1:])*sqr_disp_back)
            frame_gap=np.asarray(frames[1:])-np.asarray(frames[:-1])
            time=np.max((1,np.sum(np.asarray(motion)[1:]*frame_gap)))
        
        # curvilinear speed        
        curvilinear_speed_mean=disp/time  
        
        # curvilinear all
        try:
            if mode=="average": 
                curvilinear_speed_all=sqr_disp_back  
            else: 
                frame_gap=np.asarray(frames[1:])-np.asarray(frames[:-1])
                curvilinear_speed_all=np.asarray(motion[1:])*sqr_disp_back/frame_gap
        except:
            curvilinear_speed_all=[]
        
        # straightline_speed
        if  mode=="average":
            
            straightline_dist=np.sqrt((x_2[0]-x_1[-1])**2+(y_2[0]-y_1[-1])**2)
            straightline_time=(frames[-1]-frames[0])
            straightline_speed=straightline_dist/straightline_time
            
        else:
            move_switch=0
            start=[0,0]
            end=[0,0]
            distance=0
            frame_n=0
            

            for pos in range(1, len(motion)):
                move_pos=motion[pos]

                if move_pos==1 and move_switch==0 and pos!=(len(motion)-1): # switching to the moving
                    
                    #for strightline speed                    
                    frame_n+=frames[pos]-frames[pos-1]
                    move_switch=1 # switch to moving mode
                    start=trajectory[pos-1]
                    
                    
                elif move_pos==1 and move_switch==0 and pos==(len(motion)-1): # switching to the moving at last frame
 
                    # for strightline speed
                    move_switch=1 # switch to moving mode
                    start=trajectory[pos-1]
                    end=trajectory[pos]

                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                    
                    
                elif move_pos==0 and move_switch==1: #  end of motion

                    # for strightline speed
                    move_switch=0 # switch off moving mode
                    end=trajectory[pos-1] 

                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
                
                elif move_pos==1 and move_switch==1 and pos!=(len(motion)-1): # continue moving
                    
                    # for strightline speed
                    frame_n+=frames[pos]-frames[pos-1]
                    
                    
                elif move_pos==1 and move_switch==1 and pos==(len(motion)-1): # end of movement at last frame

                    frame_n+=frames[pos]-frames[pos-1]
                    end=trajectory[pos]

                    distance=distance+np.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)


        
            frame_n=np.max((1, frame_n))
            straightline_speed=distance/frame_n

        
        # there is no movement 
        if curvilinear_speed_mean==0:
            curvilinear_speed_mean=None
            
        if straightline_speed==0:
            straightline_speed=None
            
        return curvilinear_speed_mean, straightline_speed, curvilinear_speed_all
    
    def max_speed_segment(self, track, width):
        '''
        calculate max speed within a moving window

        '''
               
        speed_max=0
        outcome={"frames":[], "speed": None}
        segment_list=skimage.measure.label(np.asarray(track['motion']))
        
        for seg_pos in range(1, np.max(segment_list)+1): # over segments
            
            positions=np.where(segment_list==seg_pos)
            
            # check that the segment is of moving trajectory
            if track['motion'][positions[0][0]]!=0:
                start_pos=np.max((0, np.min(positions[0])-1))
                end_pos=np.min((len(track['frames'])+1, np.max(positions[0])+1))            
                
                trace_segment=track['trace'][start_pos:end_pos]
                frame_segment=track['frames'][start_pos:end_pos]
                
                segment_length=frame_segment[-1]-frame_segment[0]+1
                
                
                # iterate ver the segment with sliding window
                if segment_length>=width and segment_length>1:
                
                    for pos in range(0, len(trace_segment)-1):
                        
                        if frame_segment[pos] + width<=frame_segment[-1]:
                            mini_track=np.asarray(trace_segment[pos:np.min((pos + width+1, len(trace_segment)))])
                            mini_frames=np.asarray(frame_segment[pos:np.min((pos + width+1, len(trace_segment)))])
                          
                            # if there is a gap -> adjust the range
                            if len(mini_frames)<(mini_frames[-1]-mini_frames[0]+1):
                         
                                new_width=width-1
                                while new_width>=1:
                                    mini_track=np.asarray(trace_segment[pos:np.min((pos + new_width+1, len(trace_segment)))])
                                    mini_frames=np.asarray(frame_segment[pos:np.min((pos + new_width+1, len(trace_segment)))])
        
                                    if width==(mini_frames[-1]-mini_frames[0]):
            
                                        break
                                    elif width>(mini_frames[-1]-mini_frames[0]): # too small width -> return to the previous result
                                        
                                        mini_track=np.asarray(trace_segment[pos:np.min((pos + new_width+2, len(trace_segment)))])
                                        mini_frames=np.asarray(frame_segment[pos:np.min((pos + new_width+2, len(trace_segment)))])
                                        break
                                    else:
                                        new_width=new_width-1
                                        
                            elif len(mini_frames)>(mini_frames[-1]-mini_frames[0]+1): 
                                val=abs(width-(mini_frames[-1]-mini_frames[0]))
                                mini_track=np.asarray(trace_segment[pos:np.min((pos + width+val+1, len(trace_segment)))])
                                mini_frames=np.asarray(frame_segment[pos:np.min((pos + width+val+1, len(trace_segment)))])
                                
                            
                            speed=(np.sum(np.sqrt((mini_track[1:, 0] - mini_track[:-1, 0])**2+((mini_track[1:, 1] - mini_track[:-1, 1])**2))))/(mini_frames[-1]-mini_frames[0])
                         
                            
                            
                            if speed>speed_max and width<=segment_length:
        
                                speed_max=speed
                                
                                outcome.update({"frames":[mini_frames[0], mini_frames[-1]], "speed":speed_max})
                      
        return outcome

    def num_orientation_change(self, trajectory, motion, check_dist):
        '''
        calculate how many times trajectory direction changes
        in:
            trajectory - list, trajectory
            motion - list, motion type for each point of the trajectory
            check_dist - int, number of frames to check the direction for
        out:
            change_direction_pos - list with the positions where the direction was changed
        '''

        # calculate the displacement
        x=np.asarray(trajectory)[:,0]    
        y=np.asarray(trajectory)[:,1]
        x_0=np.asarray(trajectory)[0,0]
        y_0=np.asarray(trajectory)[0,1]
        
        disaplcement=np.sqrt((x-x_0)**2+(y-y_0)**2)        
            
        change_direction_pos=[]
        
        
        for i in range(check_dist-1, len(disaplcement)-check_dist):
            before_list=[]
            after_list=[]
            for p in range(1,check_dist+1):
                before_list.append(disaplcement[i]-disaplcement[i-p])
                after_list.append(disaplcement[i+p]-disaplcement[i])
            
            b_1=(np.asarray(before_list)>0).tolist()
            b_0=(np.asarray(before_list)<0).tolist()
            
            a_1=(np.asarray(after_list)>0).tolist()
            a_0=(np.asarray(after_list)<0).tolist()
                
            # the same orientation in the same direction
            logic_0=all(b_1) or all(b_0)
            logic_1=all(a_1) or all(a_0)
            
            #different orientation from different sides of the point
            logic_2= b_1[0]!=a_1[0]
            
            # check that the change of the direction is at the moving segment
            motion_var=motion[i]>0 or motion[i+1]>0
            
            
            
            #if based on the displcement the change in orientation is detected
            if logic_0 and logic_1 and logic_2 and motion_var:
                # check orientation change is large enough
                point_start=trajectory[i-p]
                point_middle=trajectory[i]
                point_end=trajectory[i+p]
#                print("\n points: ", point_start, point_middle, point_end)
        
    
                # calculate orientation
                y=point_start[1]-point_middle[1]
                x=point_start[0]-point_middle[0]   
            

                orientation1=(math.degrees(math.atan2(y,x))+360-90)%360
                    
#                print("\n orientation1", orientation1)
    
                y=point_end[1]-point_middle[1]
                x=point_end[0]-point_middle[0]      

                orientation2=(math.degrees(math.atan2(y,x))+360-90)%360
      
#                print("orientation2", orientation2)
                
                
                dif=abs(orientation1-orientation2)
                
                if dif>180:
#                    print("dif-360")
                    dif=abs(dif-360)
                
#                dif=180-dif
#                print("-->", dif)
                #if the difference in orientation is more than 90 degrees
                if dif<=90:
                    change_direction_pos.append(i)  


             
                
        return change_direction_pos
        
        
    def moving_segments_data(self, track, width, framerate):
        '''
        calculate number of moving segments and average segment time (taking into account change in direction)
        
        in:
            track - dict, trajectory with trace, frames, motion information
            width - float, number of frames to consider for directionality change
            framerate - int, frame rate 
        out:
            change_direction_pos_updated - number of time the direction is changed
            num_moving_segment - number of moving segments (taking into account directionality change inside the moving segment)
            average_moving_time_per_segment - average duration of the moving segmnt
            total_moving_time - duration of movement in the trajectory
        '''
        
        segment_list=skimage.measure.label(np.asarray(track['motion'][1:]))
        
        
        # find the direction changes        
        change_direction_pos=self.num_orientation_change(track['trace'], track['motion'], width)
        change_direction_pos_updated=[]
        
        # total number of segment in the trajectory
        total_moving_segment_n=0
        
        # total time of the moving segments
#        motion_only=np.asarray(track['motion'])[np.asarray(track['motion'])>0]
#        print("motion only", len(motion_only), np.sum(np.asarray(track['motion'])), segment_list)
        total_moving_time=0
#        total_moving_time=np.sum(np.asarray(track['motion'])) # frames
        
        
        for seg_pos in range(0, np.max(segment_list)+1): # over segments
            
            positions=np.where(segment_list==seg_pos)
            
            positions_new=positions[0]+1

            
            if len(positions_new)>0: # if the array is not empty
    #            print("\n", positions)
                val_motion=np.asarray(track['motion'])[positions_new[0]]
       
                # check if it is moving segment
                if val_motion==1:
                    
                    # moving segment length
                    segment_length=track['frames'][np.max(positions_new)]-track['frames'][np.min(positions_new)]
                    
                    # if it is the first segment from the beginning take it into account
                    if np.min(positions_new)==1:
                        
                        segment_length+=track['frames'][1]-track['frames'][0]
                        
                    total_moving_time+=segment_length
#                    print("segment_length", segment_length, "->", track['frames'][np.max(positions_new)],"-", track['frames'][np.min(positions_new)])
                    
                    start_pos=np.min(positions_new)+2 # start position from the second point to avoid single frame segment
                    end_pos=np.max(positions_new)-1   # end position -second last to avoid single frame segment
                    frame_segment=track['frames'][start_pos:end_pos]
               
                    # find is any change of direction inside the moving segment
                    intersection = [value for value in change_direction_pos if track['frames'][value] in frame_segment]
#                    intersection = [track['frames'][value] for value in change_direction_pos if track['frames'][value] in frame_segment]
                    
                    # number of segments in thismoving segment
                    segment_n=1+len(intersection)
                    
                    # pruned  directionality change update
                    change_direction_pos_updated=change_direction_pos_updated+intersection
                    
                    # total number of segment in the trajectory
                    total_moving_segment_n+=segment_n
                    
        
        #  average_moving_time_per_segment
        
        if total_moving_segment_n!=0:
            average_moving_t=(total_moving_time/total_moving_segment_n)/framerate
        else:
            average_moving_t=0
   
#        print("total_moving_time ", total_moving_time, "vs", (track['frames'][-1]-track['frames'][0]), "  ->", total_moving_segment_n, average_moving_t)
        # out
#        out={'num_moving_segment':total_moving_segment_n, "average_moving_time_per_segment": average_moving_t}
                 
        return change_direction_pos_updated, total_moving_segment_n, average_moving_t, total_moving_time/framerate

            
    def refine_track_sequence(self, tracks, frames, motion):
        '''
        check the frame order in the track and save the positioning accordingly
        '''
        new_frames=[frames[0]]
        new_tracks=[tracks[0]]
        new_motion=[motion[0]]
        
        pos=frames[0]+1
        for f in range(1,len(frames)):
            
            if frames[f]==pos:
                new_frames.append(frames[f])
                new_tracks.append(tracks[f])
                new_motion.append(motion[f])
                pos+=1
                
            elif  frames[f]>pos: # there are skipped frames
                
                while frames[f]>pos:                
                    new_frames.append(pos)
                    new_tracks.append(new_tracks[-1])  
                    new_motion.append(new_motion[-1])          
                    pos+=1
                    
                #append the original f frame
                new_frames.append(pos)
                new_tracks.append(tracks[f]) 
                new_motion.append(motion[f])
                    
                pos+=1
                
        return new_tracks, new_frames, new_motion        
        
        
        #
#        
#
#if __name__ == "__main__":
#    segmented=[1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,]
#    print(np.asarray(segmented))
#    results=TrajectorySegment.remove_small_segments(segmented, segmented)
#    
#    print(results)
    