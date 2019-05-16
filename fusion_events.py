'''
   detect fusion events
'''

import numpy as np
import scipy as sp

import skimage
from skimage import io
import matplotlib.pyplot as plt
import json
import cv2
from scipy.ndimage import gaussian_filter1d
import math


class FusionEvent(object):
    
    """class for detection and description of the fusion events
    """
    
    def __init__(self, cargo_movie=np.zeros((10,10)), membrane_movie=np.zeros((10,10)), membrane_mask=np.zeros((10,10)), tracks=[]):
        """
        Initialise variables
        """
        self.tracks=tracks # output tracks
        self.membrane_movie=membrane_movie
        
        if len(membrane_mask.shape)!=3:
            self.membrane_mask=np.ones(cargo_movie.shape)*membrane_mask
        else:
            self.membrane_mask=membrane_mask
            
        self.cargo_movie=cargo_movie
        self.fusion_events=[]
        
        # parameters for detection
        self.track_length_min=2
        self.track_length_max=5000
        self.max_movement=1.5 # maximum movement which is counted as standing
        
        self.frame_freq=4
        self.distance_to_membrane=0 # minimum ditsnce to the membrane mask
        #parameters to estimate
        self.final_stop_length_array=[]
        self.max_displacement_array=[]
        self.duration_array=[]
        self.speed_array=[]
        self.angle_array=[]
        
        
    def direction_calculation(self, trace, plot_var=1):
        angle_array=[]
        if len(trace)>=2:
            for i in range(0, len(trace)-2):
                p1=trace[i]                        
                p2=trace[i+2]
                xDiff = p2[0] - p1[0]
                yDiff = p2[1] - p1[1]
                angle= int(math.degrees(math.atan2(yDiff, xDiff)))       
                angle_array.append(angle)
        
        
        #calculate generale angle:
            p1=trace[0]                        
            p2=trace[-1]
            xDiff = p2[0] - p1[0]
            yDiff = p2[1] - p1[1]        
            general_angle=int(math.degrees(math.atan2(yDiff, xDiff))) 
        
        #gaussian smoothing
            angle_array_gaussian=gaussian_filter1d(angle_array, 2)
            
            # plotting
            if plot_var==1:
                time=np.arange(0,len(angle_array))
                plt.figure()
                plt.plot(time, angle_array_gaussian, 'r-', label='gaussian ')
                plt.plot(time,angle_array, 'g-', label='angles')
                plt.xlabel('frames')
                plt.ylabel('angle')
                plt.title(general_angle)
                plt.legend()
                plt.show()
            
            
        return general_angle, angle_array
    
    
    def speed_calculation(self, trajectory, frames):
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
        time=(frames[-1]-frames[0])/self.frame_freq
        
        #speed        
        speed=disp/time
    
        return speed
    
    
    def calculate_stand_length(self, trajectory, plot_var=0):
        '''
        calculate length of the standing at the end
        '''
        # separated arrays for coordinates
        x=np.asarray(trajectory)[:,0]    
        y=np.asarray(trajectory)[:,1]    
        
        # end coordinates
        x_e=np.asarray(trajectory)[-1,0]
        y_e=np.asarray(trajectory)[-1,1]     
        
        sqr_disp_back=np.sqrt((x-x_e)**2+(y-y_e)**2)
        position=np.array(range(len(sqr_disp_back)))

        sqr_disp_back=sqr_disp_back[::-1]
        displacement_gaussian_3_end=gaussian_filter1d(sqr_disp_back, 3)

        #count for how long it doesn't exceed movement threshold
        movement_array=position[displacement_gaussian_3_end>self.max_movement]
        
        if len(movement_array)>0:           
            stand_time=movement_array[0]
        else:
            stand_time=0
        
        if plot_var==1:
            time=np.arange(0,len(trajectory))
            plt.figure()
            plt.plot(time, displacement_gaussian_3_end, 'r-', label='gaussian ')
            plt.plot(time,sqr_disp_back, 'g-', label='desplacement from the back')
            plt.plot(time, self.calculate_displacement(trajectory), 'b-', label=' original desplacement')
            plt.xlabel('frames')
            plt.ylabel('displacement')
            plt.title(str(stand_time))
            plt.legend()
            plt.show() 
        
        
        return stand_time
     
    def find_fusion(self):
        ''' 
        look thorugh all the track and
        find the tracks which ends at the membrane 
        '''
        print("distance_to_membrane -  ",self.distance_to_membrane)
        count=0
        for trackID in range(0, len(self.tracks['tracks'])):
            track=self.tracks['tracks'][trackID]
            
    #    calculate parameters
            if len(track['trace'])>self.track_length_min and len(track['trace'])<self.track_length_max: 
                point=track['trace'][-1]
                
                # calculate mask location and min distance                
                membrane_coordinates=np.argwhere(self.membrane_mask[track['frames'][-1],:,:]==1)
                distance_m=np.sqrt((np.asarray(membrane_coordinates)[:,0]-point[0])**2+(np.asarray(membrane_coordinates)[:,1]-point[1])**2)
                on_membrane_val=np.min(distance_m)<=self.distance_to_membrane

#                if self.membrane_mask[track['frames'][-1], point[0], point[1]]:
                if on_membrane_val==True:
                    count+=1
                    stand_time=self.calculate_stand_length(track['trace'],0)
                    vespeed=self.speed_calculation(track['trace'], track['frames'])
                    max_disp=np.max(self.calculate_displacement(track['trace'],0))
#                    print("\n", track['trackID'])
#                    print("duration:                                ", len(track['trace']))
#                    print("max displacement:                        ", max_disp)
#                    print("the last frame :                         ", track['frames'][-1])
#                    print("length of the final stop :               ", stand_time)
#                    print("vesicle speed            :               ", vespeed)
                    self.fusion_events.append(track)
                    
                    self.final_stop_length_array.append(stand_time)
                    self.speed_array.append(vespeed)
                    angle, angle_array=self.direction_calculation(track['trace'], plot_var=0)
                    self.angle_array.append(angle)
                    self.duration_array.append(len(track['trace']))
                    self.max_displacement_array.append(max_disp)
                    
        print(" Total number of fusion events: ", count, " \n ----------- \n")
        
        #plotting histograms
        fig, axes = plt.subplots(2,3) #, sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
#        plt.tight_layout()
        
        #length
        axes[0,0].hist(self.final_stop_length_array, bins=100)
        axes[0,0].set_title('stop duration')
        axes[0,0].set_ylabel('number of events')
        axes[0,0].set_xlabel('frames')
        
        #duration
        axes[0,1].hist(self.duration_array, bins=100)
        axes[0,1].set_title('track length')
        axes[0,1].set_ylabel('number of events')
        axes[0,1].set_xlabel('frames')
        
        # displacement
        axes[0,2].hist(self.max_displacement_array, bins=100)
        axes[0,2].set_title('final displacement')
        axes[0,2].set_ylabel('number of events')
        axes[0,2].set_xlabel('pix')
        # speed
        axes[1,0].hist(self.speed_array, bins=100)
        axes[1,0].set_title('average speed')
        axes[1,0].set_ylabel('number of events')
        axes[1,0].set_xlabel('pix/sec')

        #angle
        axes[1,1].hist(self.angle_array, bins=100)
        axes[1,1].set_title('average direction')
        axes[1,1].set_ylabel('number of events')
        axes[1,1].set_xlabel('degrees')

        axes[1,2].set_axis_off() 
        
        fig.savefig('subplot.png')
  

        return {'tracks': self.fusion_events}
    
    
    def calculate_displacement(self, trajectory, plot_var=0):
        # separated arrays for coordinates
        x=np.asarray(trajectory)[:,0]    
        y=np.asarray(trajectory)[:,1]    
        
        time=np.arange(0,len(trajectory))
        
        # start coordinates
        x_0=np.asarray(trajectory)[0,0]
        y_0=np.asarray(trajectory)[0,1]
        
        # end coordinates
        x_e=np.asarray(trajectory)[-1,0]
        y_e=np.asarray(trajectory)[-1,1]        

        #displacement in relation to the start
        sqr_disp=np.sqrt((x-x_0)**2+(y-y_0)**2)
        
        #displacement in relation to the end
        sqr_disp_back=np.sqrt((x-x_e)**2+(y-y_e)**2)
        
        displacement_gaussian_3=gaussian_filter1d(sqr_disp, 3)
        displacement_gaussian_3_end=gaussian_filter1d(sqr_disp_back, 3)
        
        
    
        if plot_var==1:
    
            plt.figure()
            plt.plot(time, sqr_disp, 'y-', label='displacement')
            plt.plot(time, displacement_gaussian_3, 'b-', label='gaussian 3 displ')
            plt.plot(time, displacement_gaussian_3_end, 'r-', label='gaussian 3 back')
            plt.plot(time,sqr_disp_back, 'g-', label='desplacement from the back')
            plt.xlabel('frames')
            plt.ylabel('displacement')
            plt.legend()
            plt.show() 
 
            
        return sqr_disp       
             
    def save_movie(self, save_file):
        '''
        save fusion events as a movie and tracks as fusion events
        '''
        
        def track_to_frame(track_data, movie):
            # change data arrangment from tracks to frames
            track_data_framed={}
            track_data_framed.update({'frames':[]})
            
            for n_frame in range(0, movie.shape[0]):
                
                frame_dict={}
                frame_dict.update({'frame': n_frame})
                frame_dict.update({'tracks': []})
                
                #rearrange the data
                for p in track_data:
                    if n_frame in p['frames']: # if the frame is in the track
                        frame_index=p['frames'].index(n_frame) # find position in the track
                        
                        new_trace=p['trace'][0:frame_index+1] # copy all the traces before the frame
                        frame_dict['tracks'].append({'trackID': p['trackID'], 'trace': new_trace}) # add to the list
                        
                        
                track_data_framed['frames'].append(frame_dict) # add the dictionary
             
            return track_data_framed
        
        # save tracks
        
        #remove the gaps
        new_track_list={'tracks':[]}
        for track in self.fusion_events:
            frames=track['frames']
            trace=track['trace']
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
            new_track_list['tracks'].append({'trackID': track['trackID'], 'frames': new_frames, 'trace': new_trace})    
        
        
        #save the tracks
        with open(save_file+'.txt', 'w') as f:
            json.dump(new_track_list, f, ensure_ascii=False) 
        print("tracks are saved in  ", save_file, " file")
        
        # save movie
        
        final_img_set = np.zeros(self.cargo_movie.shape)
        
        fusion_framed=track_to_frame(self.fusion_events, self.cargo_movie)

        for frameN in range(0, self.cargo_movie.shape[0]):      
            plot_info_fusions=fusion_framed['frames'][frameN]['tracks']
    
            # Make a colour image frame
            orig_frame = np.zeros((self.cargo_movie.shape[1], self.cargo_movie.shape[2]))
     
            #fusions
            for p in plot_info_fusions:
                trace=p['trace']
                
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
                                 (255, 0, 0), 2)
                    
            # Display the resulting tracking frame
            cv2.imshow('Tracking', orig_frame)
    
            ################### to save #################
            final_img_set[frameN,:,:]=orig_frame
    
                # save results
        
        final_img_set=final_img_set/np.max(final_img_set)*255
        final_img_set=final_img_set.astype('uint8')
        skimage.io.imsave(save_file+".tif", final_img_set)
        cv2.destroyAllWindows()
            
    def register_cam1(self, camera1_img):
        '''
        transfor the membrane img
        '''
        
        def rotate(image, angle, center = None, scale = 1.0):
            (h, w) = image.shape[:2]
        
            if center is None:
                center = (w / 2, h / 2)
        
            # Perform the rotation
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(image, M, (w, h))
        
            return rotated
        
        
        # flip vertically
        
        camera1_img=np.flip(camera1_img, 0)
        
        # translate 
        
        x_trans=13
        camera1_new=np.zeros(camera1_img.shape)
        camera1_new[0:-x_trans,:]=camera1_img[x_trans:,:]
        
        # rotate
        angle=-0.17 #-0.3
        center=(0,0)
        camera1_new=rotate(camera1_new, angle, center)
        
        return camera1_new        
        