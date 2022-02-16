#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils for the viewer
"""
import numpy as np
import scipy as sp

import skimage
from skimage.feature import peak_local_max

class SupportFunctions(object):
    
    def __init__(self):
        pass
    
    def intensity_calculation(self, movie, trace, frames, patch_size=16):
        '''
        Calculates changes in intersity for the given track
        
        in:
            movie - entire image sequence
            trace - trajectory coordinates
            frames - trajectory frames
            patch_size - patch size for intensity identification
        
        out: curretly it is different
            intensity_array_1 - array of intensity based on the segmented particles
            intensity_array_2 - array of intensity based on the patch_size
            intensity_mean_1 - mean intensity based on the patch
            intensity_mean_2 - mean intensity of the segmented vesicle
            
            intensity_array_1_norm - normalised in range[0,1]
            intensity_array_1_maxbased - normalised by max
            
            check_border - if there was any problem obtaining intensity due to the image boarder it will be 1
        '''
        
        def img_segmentation(img_segment, int_size, box_size):
            '''
            the function segments region based on the thresholding and watershed segmentation
            the only center part of the segmented part is taked into account.
            '''
    
        # calculate threshold based on the centre
            threshold=np.mean(img_segment[int(box_size/2-int_size):int(box_size/2+int_size), int(box_size/2-int_size):int(box_size/2+int_size)])
        #    thresholding to get the mask
            mask=np.zeros(np.shape(img_segment))
            mask[img_segment>threshold]=1
        
            # separate the objects in image
        ## Generate the markers as local maxima of the distance to the background
            distance = sp.ndimage.distance_transform_edt(mask)
            local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=mask)
            markers = sp.ndimage.label(local_maxi)[0]
        
            # segment the mask
            segment = skimage.morphology.watershed(-distance, markers, mask=mask)
           
        # save the segment which is only in the centre
            val=segment[int(box_size/2), int(box_size/2)]
            segment[segment!=val]=0
            segment[segment==val]=1
    
            return segment
        
        #extract images
        track_img=np.zeros((len(trace),patch_size,patch_size))
        
        intensity_array_1=[]
        intensity_array_2=[]
        
        check_border=0 
        for N in range(0,len(frames)):
            frameN=frames[N]
            point=trace[N]
            x_min=int(point[0]-patch_size/2)
            x_max=int(point[0]+patch_size/2)
            y_min=int(point[1]-patch_size/2)
            y_max=int(point[1]+patch_size/2)
            
            if x_min>0 and y_min>0 and x_max<movie.shape[1] and y_max<movie.shape[2]:
                
                # create img
                track_img[N,:,:]= movie[frameN, x_min:x_max, y_min:y_max]
                
                #segment img
                int_size=5
                segmented_vesicle=img_segmentation(track_img[N,:,:]/np.max(track_img[N,:,:]), int_size, patch_size)
                
                #calculate mean intensity inside the segment                
                intensity_1=np.sum(track_img[N,:,:]*segmented_vesicle)/np.sum(segmented_vesicle)
                intensity_2=np.sum(track_img[N,:,:])/(patch_size*patch_size)
                intensity_array_1.append(intensity_1)
                intensity_array_2.append(intensity_2)
            else:
                check_border=1
                intensity_array_1.append(0)
                intensity_array_2.append(0)
                
        max_val=np.max(movie)
        min_val=np.min(movie)
        
        non_zero_intensity_array_1=np.asarray(intensity_array_1)[np.asarray(intensity_array_1)!=0]
        non_zero_intensity_array_2=np.asarray(intensity_array_2)[np.asarray(intensity_array_1)!=0]
        
        intensity_mean_1=(np.mean(non_zero_intensity_array_1)-min_val)/(np.max((max_val, 0.00001))-min_val)
        intensity_mean_2=(np.mean(non_zero_intensity_array_2)-min_val)/(np.max((max_val, 0.00001))-min_val)
        
        intensity_array_1_norm=(intensity_array_1-min_val)/(np.max((max_val, 0.00001))-min_val)
        intensity_array_2_norm=(intensity_array_2-min_val)/(np.max((max_val, 0.00001))-min_val)
        

        intensity_mean_1_maxbased=np.mean(non_zero_intensity_array_1)/np.max((max_val, 0.00001))
        intensity_mean_2_maxbased=np.mean(non_zero_intensity_array_2)/np.max((max_val, 0.00001))
        
        intensity_array_1_maxbased=intensity_array_1/np.max((max_val, 0.00001))
        intensity_array_2_maxbased=intensity_array_2/np.max((max_val, 0.00001))


        
        
#        print("\n range: ", min_val, max_val, "\n -> ", intensity_array_1, "\n -> ", intensity_array_2)
#        print("intensities: ", intensity_mean_1, intensity_mean_2)
#        print(" --------range---- \n result: ", (np.mean(intensity_array_1)-min_val)/(max_val-min_val), (np.mean(intensity_array_2)-min_val)/(max_val-min_val))
        
#        return intensity_array_1, intensity_array_2, intensity_mean_1, intensity_mean_2, check_border
    
    
        return intensity_array_1_norm, intensity_array_1_maxbased, intensity_mean_1, intensity_mean_1_maxbased, check_border
