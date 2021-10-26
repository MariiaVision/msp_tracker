#########################################################
#
# particle detection 0.2
#        
#########################################################

import numpy as np
import scipy as sp
import math

import matplotlib.pyplot as plt
import skimage
from skimage import exposure, filters # to import file
from skimage.feature import peak_local_max # find local max on the image

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

import trackpy as tp
import json
import warnings
warnings.filterwarnings("ignore")

class Detectors(object):
    """
    Class to detect particles in a frame
    
    """
    def __init__(self):
        """Initialize variables
        """

        self.img_mssef=[]
        self.binary_mssef=[]
        self.img_set=[] # 
        
        # parameters for approach
        #MSSEF
        self.c=0.8 # coef for the thresholding
        self.k_max=20 # end of  the iteration
        self.k_min=1 # start of the iteration
        self.sigma_min=0.1 # min sigma for LOG
        self.sigma_max=3. # max sigma for LOG    
        self.intensity_min=0 # min relevant intensity
        self.intensity_max=1 # max relevant intensity
        
        #thresholding
        self.min_distance=4 # minimum distance between two max after MSSEF
        self.threshold_rel=0.1 # min pix value in relation to the image
        
        self.box_size=16 # bounding box size for detection
        self.box_size_fit= 16 # bounding box size for gaussian fit
        
        #CNN based classification        
        self.detection_threshold=0.99
        
        # result
        
        self.detected_vesicles=[]
        self.detected_candidates=[]
        
        #bg substration setting
        self.substract_bg_step=100
        
        #option for Gaussian fit
        self.gaussian_fit=False
        self.expected_radius=5 # expected radius of the paricle
        
        self.cnn_model_path=""
        
    def read_param_from_file(self, file_name):
        '''
        Read parameters from the given file
        '''
        
        with open(file_name) as json_file: 
            data = json.load(json_file)

        settings=data['parameters']
        print("\n reading data: \n", settings)
            
        #MSSEF
        self.c=settings['c']  # coef for the thresholding
        self.k_max=settings['k_max'] # end of  the iteration
        self.k_min=settings['k_min'] # start of the iteration
        self.sigma_min=settings['sigma_min'] # min sigma for LOG
        self.sigma_max=settings['sigma_max']# max sigma for LOG   
        self.intensity_min=settings['intensity_min'] # min relevant intensity
        self.intensity_max=settings['intensity_max'] # max relevant intensity     
        
        
        #thresholding
        self.min_distance=settings['min_distance'] # minimum distance between two max after MSSEF
        self.threshold_rel=settings['threshold_rel'] # min pix value in relation to the image        
        self.box_size=settings['box_size'] # bounding box size for detection
        
        #CNN based classification        
        self.detection_threshold=settings['detection_threshold']         
        self.substract_bg_step =settings['substract_bg_step']   
        
        #option for Gaussian fit
        self.gaussian_fit=settings['gaussian_fit']
        
        self.cnn_model_path=settings['cnn_model_path']        
        
        
    def img_enhancement(self, img, kernal_medfilter=3):
        '''
        image enhancement
        '''

        img_new=filters.median(img, selem=None, out=None)
        img_enh=exposure.equalize_adapthist(img_new, kernel_size=None, clip_limit=0.01, nbins=256)

        
        return img_enh
    
    def substract_bg_single(self, img_set, pos):
        '''
        substract image and in-situ background
        '''
        if self.substract_bg_step==0: # the background subtraction is off
            img_new=img_set[pos,:,:]
        else:
            start_i = pos-int(self.substract_bg_step/2) # start frame
            if start_i<0:
                start_i=0
            end_i = start_i+self.substract_bg_step # end frame
            if end_i>=img_set.shape[0]:
                end_i=img_set.shape[0]
                start_i=end_i-self.substract_bg_step
    
            insitu=np.min(img_set[start_i:end_i], axis=0) # insitu calculation    
            
            # removing background by substraction
            img_3ch=np.copy(img_set[pos])
            img_new= img_3ch-insitu  #calculate some percent of the insitu
            
        return img_new

    def sef(self, img, img_sef_bin_prev, sigma, c, print_val=0):   
        '''
        spot enhancing filter
        '''
    #function to calculate spot-enhancing filter for a single scale
        
        img_filtered=img*img_sef_bin_prev # multiply image with the binary from the pervious iteration
        img_sef1=sp.ndimage.gaussian_laplace(img_filtered, sigma) # calculate laplacian of gaussian
        img_sef1=abs(img_sef1-np.abs(np.max(img_sef1))) # remove negative values keeping the proportion b/w pixels

    # thresholding
        th=np.mean(img_sef1)+c*np.std(img_sef1) # calculate threshold value
        img_sef=np.copy(img_sef1) # copy the image
        img_sef[img_sef<th]=0 # thresholding 
    
        # create a binary mask for the next step
        img_sef_bin = np.copy(img_sef)
        img_sef_bin[img_sef_bin<th]=0
        img_sef_bin[img_sef_bin>=th]=1
    
        # plot the image if print_val==1
        if print_val==1:
            fig = plt.figure()
            plt.gray()
            ax1=fig.add_subplot(411)
            ax2=fig.add_subplot(412)
            ax3=fig.add_subplot(413)
            ax4=fig.add_subplot(414)
            ax1.imshow(img)
            ax2.imshow(img_sef1)
            ax3.imshow(img_sef_bin)
            ax4.imshow(img_sef)
            plt.show()
    
        return img_sef, img_sef_bin
    
    def mssef(self, img, c, k_max, k_min, sigma_min, sigma_max):
        '''
        Multi-scale spot enhancing filter
        '''
        
        img_bin=np.ones(img.shape) # original array
        N=k_max-k_min # number of steps
    
    # Multi-scale spot-enhancing filter loop
        for k in range(k_min, k_max):

            sigma=sigma_max-(k-1)*(sigma_max-sigma_min)/(N-1) #assign sigma
            result, img_sef_bin=self.sef(img, img_bin, sigma, c, print_val=0) # SEF for a single scale
            img_bin=img_sef_bin
            
        return result, img_sef_bin
    
    def classify_vesicle(self, centers, img, new_model, segment_size=16):
        '''
        CNN-based classifier
        '''
        updated_centers=[]

        img_segment_set=np.zeros((len(centers),segment_size,segment_size))
        i=0
        
        if len(centers)>0:
            for lm in centers:
                
                #roi coordinates
                x_st=int(lm[0]-segment_size/2)
                x_end=x_st+segment_size
                y_st=int(lm[1]-segment_size/2)
                y_end=y_st+segment_size
                
                # coordinates for roi stack
                x_st_im=0
                x_end_im=segment_size
                y_st_im=0
                y_end_im=segment_size
                
                img_roi=np.zeros((segment_size,segment_size))
                                              
                # check if the ROI fit
                
                # too close to left or top boarder
                if x_st<0: 
                    x_st_im=abs(x_st)
                    x_st=0

                if y_st<0: 
                    y_st_im=abs(y_st)
                    y_st=0
                    
                # too close to right or bottom
                if x_end > img.shape[0]:
                    x_end_im=segment_size - (x_end-img.shape[0])
                    x_end= img.shape[0]
                    
                if y_end > img.shape[1]:
                    y_end_im=segment_size - (y_end-img.shape[1])
                    y_end= img.shape[1]
                
                #create roi
                img_roi[x_st_im: x_end_im, y_st_im:y_end_im]= np.copy(img[x_st: x_end, y_st:y_end])   
                
                #mirror the image 
                img_segment=img_roi[::-1,::-1]
                #overlap with original
                img_segment[x_st_im: x_end_im, y_st_im:y_end_im]= np.copy(img[x_st: x_end, y_st:y_end])
                
                #preprocess the segment
                img_segment_set[i,:,:]=(img_segment-np.min(img_segment))/(np.max(img_segment)-np.min(img_segment))                
                i=i+1
            x_data=img_segment_set.reshape((len(centers), img_segment.shape[0],img_segment.shape[1], 1))
            score_set = new_model.predict(x_data) 

            for pos in range(0, len(centers)):
                
                if score_set[pos][1]>self.detection_threshold:
                    
                    updated_centers.append([float(centers[pos][0]), float(centers[pos][1])])

        return updated_centers


    def detect(self, frameN, new_model):
        """
        Main function for the particle detection
        """
        # calculate max and min intensity
        intensity_set_max=np.max(self.img_set)
        intensity_set_min=np.min(self.img_set)
        intensity_range=intensity_set_max-intensity_set_min
        
        
        #background substraction 
        
        if self.substract_bg_step==0:
            img=self.img_set[frameN,:,:]
        else:                
            img= self.substract_bg_single(self.img_set, frameN) 
        
        # enhance image
        gray = self.img_enhancement(img)
        
        
        # MSSEF
        self.img_mssef, self.binary_mssef=self.mssef(gray, self.c, self.k_max, self.k_min, self.sigma_min, self.sigma_max)
        
        # devide binaries into segments
        spots_labeled=sp.ndimage.label(self.binary_mssef)[0]
        
        local_max=[]
        self.detected_vesicles=[]
        self.detected_candidates=[]
        
        # iterate over segments 
        new_img=filters.median(img, selem=None, out=None)
        for i in range(1, np.max(spots_labeled)+1):
            mask_label=np.zeros(spots_labeled.shape)
            mask_label[spots_labeled==i]=1
            
            img_label=np.asarray(mask_label*new_img)

            local_peaks=peak_local_max(img_label, threshold_rel=self.threshold_rel,  min_distance=int(self.min_distance))
                                    
            for point in local_peaks:
                
                # check intensity 
                try:
                    point_avg_intensity=np.mean(self.img_set[frameN, point[0]-1:point[0]+2, point[1]-1:point[1]+2])
                except:
                    point_avg_intensity=self.img_set[frameN, point[0]-1:point[0]+2, point[1]-1:point[1]+2]


                if point_avg_intensity>=intensity_set_min+self.intensity_min*intensity_range and point_avg_intensity<=intensity_set_min+self.intensity_max*intensity_range:
                    point_new=[point[0], point[1]]
                    local_max.append(point_new) 
                    self.detected_candidates.append(point_new)   
    
        
        
        #remove detections which are too close to each other

        checked_detection=[]
        compared_spots=[-1]
        for pos in range(0, len(self.detected_candidates)):

            if (pos in compared_spots) == False: 
                res_distance=np.sqrt(np.sum((np.asarray(self.detected_candidates)-np.asarray(self.detected_candidates[pos]))**2, axis=1))

                near_by_spots=list(np.where(res_distance<self.min_distance)[0])

                if len(near_by_spots)>1: # there is an overlap
                    spots=list(np.asarray(self.detected_candidates)[near_by_spots])

                    centre=[(max(np.asarray(spots)[:,0])+min(np.asarray(spots)[:,0]))/2, (max(np.asarray(spots)[:,1])+min(np.asarray(spots)[:,1]))/2]

                    checked_detection.append(centre)

                    compared_spots=compared_spots+near_by_spots
                    
                else: # no overlap
                    checked_detection.append(self.detected_candidates[pos])
                    compared_spots=compared_spots+near_by_spots
                
        self.detected_candidates=checked_detection  
        
        #normalised frame
        frame_img_classify=self.img_set[frameN,:,:]
        
        if self.detection_threshold==0:
            
            updated_centers=self.detected_candidates
        else:
            
            updated_centers=self.classify_vesicle(self.detected_candidates, frame_img_classify, new_model, segment_size=self.box_size)    
        
        
        if self.gaussian_fit==True:

            for lm in updated_centers: # loop over all the found vesicles
                
                # ROI for the vesicle
                img=self.img_set[frameN,:,:]
                img_m=filters.median(img, selem=None, out=None)

                try: 
                    
                    img_roi_raw= np.copy(img[int(lm[0])-int(self.box_size_fit/2): int(lm[0])+int(self.box_size_fit/2), int(lm[1])-int(self.box_size_fit/2):int(lm[1])+int(self.box_size_fit/2)])  
                    img_roi_processed= np.copy(img_m[int(lm[0])-int(self.box_size_fit/2): int(lm[0])+int(self.box_size_fit/2), int(lm[1])-int(self.box_size_fit/2):int(lm[1])+int(self.box_size_fit/2)])  
        
                    # sub-pixel localisation
                    
                    coordinates=[[int(self.box_size_fit/2),int(self.box_size_fit/2)]]   
                    new_coor=tp.refine_com(raw_image=img_roi_raw, image=img_roi_processed, radius=int(self.expected_radius), coords=np.asarray(coordinates), shift_thresh=1)

                    x=new_coor['x'][0]
                    y=new_coor['y'][0]
                    
                    if math.isnan(x):
                        x=lm[0]
                        y=lm[1]
                    else:
                        x=lm[0]+x-int(self.box_size_fit/2)
                        y=lm[1]+y-int(self.box_size_fit/2)
                    point_new=[float(x), float(y)]            
                    self.detected_vesicles.append(point_new) 
                    
                except:

                    x=lm[0]
                    y=lm[1]
                    point_new=[float(x), float(y)]            
                    self.detected_vesicles.append(point_new) 

        else:
            self.detected_vesicles=updated_centers
            
  
        print("candidates: ", len(self.detected_candidates), ";   detections ", len(self.detected_vesicles), "\n ")

        return self.detected_vesicles