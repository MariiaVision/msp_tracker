#########################################################
#
# membrane segmentation
#        
#########################################################

import numpy as np
import skimage
import skimage.morphology
import matplotlib.pyplot as plt
from skimage import io, filters, exposure, morphology
import math
import scipy as sp
from skimage.draw import line
from tqdm import tqdm
from scipy.ndimage import maximum_filter, minimum_filter, correlate


class MultiscaleLineDetection():
    """
    Class to segment line-like structures and evaluate their orientation 
    
    """

    def __init__(self):
        
        
        self.image = []
        self.W = 11 
        self.W_start=11
        self.step = 2
        self.degree_step=15
        self.image_segmented=[]
        self.min_size=1

        # Preprocessing parameters
        self.use_normalization = True
        self.norm_saturation = 0.1
        self.norm_clip = True
        self.norm_use_clahe = True
        self.norm_clahe_args = {"kernel_size": None, "clip_limit": 0.01, "nbins": 256}

    def find_linemask(self, theta, masksize):
        '''
        calculate mask with single pixel line with theta orientation

        '''

        def find_basemask(theta, masksize):
            '''
            calculate line mask with given theta and masksize
            '''

            mask = np.zeros((masksize,masksize))

            halfsize = int((masksize)/2)

            if theta == 0:
                mask[halfsize,:] = 1
            elif theta == 90:
                mask[:,halfsize] = 1
            else:

                x0 = -halfsize
                y0 = round(x0*(math.sin(np.deg2rad(theta))/math.cos(np.deg2rad(theta))), 0)

                if y0 < -halfsize:
                    y0 = -halfsize
                    x0 = round(y0*(math.cos(np.deg2rad(theta))/math.sin(np.deg2rad(theta))), 0)

                x1 = halfsize
                y1 = round(x1*(math.sin(np.deg2rad(theta))/math.cos(np.deg2rad(theta))), 0)

                if y1 > halfsize:
                    y1 = halfsize
                    x1 = round(y1*(math.cos(np.deg2rad(theta))/math.sin(np.deg2rad(theta))), 0)

            # draw a line:
                rr, cc= line(int(halfsize-y0), int(halfsize+x0), int(halfsize-y1), int(halfsize+x1))
                mask[rr, cc] = 1

            return mask

        # calculate the line in range 0-90 and rotate if it's bigger angle
        if theta > 90:
            mask = find_basemask(180 - theta,masksize)

            #rotate the mask
            h,w = mask.shape
            linemask=np.zeros(mask.shape)
            for i in range(0,h):
                for j in range(0,w):
                    linemask[i,j] = mask[i,w-j-1]

        else:
           linemask = find_basemask(theta,masksize)

        return linemask

    def normalize(self, img=None, update_image=True):
        """ 
        Normalise the image intensity

        """
        # Select the image to process
        if img is None:
            img = np.copy(self.image)

        # Convert the image to float32
        img = img.astype(np.float32)

        # Apply intensity normalization
        imin = img.min()
        imax = np.percentile(img, 100 - self.norm_saturation)
        img = (img - imin) / (imax - imin)

        # Keep track of imin and imax to denormalise
        self.norm_imin = imin
        self.norm_imax = imax

        # Remove saturated intensities
        if self.norm_clip:
            img[img > 1] = 1

        # Perform local histogram equalization
        if self.norm_use_clahe:
            # change intensity -> must be between 0 and 1
            imin_clahe = img.min()
            imax_clahe = img.max()
            img = (img - imin_clahe) / (imax_clahe - imin_clahe)

            img = exposure.equalize_adapthist(img, **self.norm_clahe_args)

            # changing back to the original intensity range
            img = img * (imax_clahe - imin_clahe) + imin_clahe

        # Update the image
        if update_image:
            self.image = img

        return img


    def segmentation(self, img, threshold=1.2, roi_step=20, margen_step=5, img_threshold=0.01):
        '''
            image segmentation based on the correlation with line fo different orientation and length
            the image is proccessed by small regions
        '''

        def standardize_2(img):
            '''
            normalise the data
            '''
            img_norm = (img - np.min(img)) / np.max([(np.max(img)-np.min(img),1e-50)])

            return img_norm


        def find_lineresponse(L,img):
            '''
            calculate line response for a set of angles
            '''

            avgresponse = sp.ndimage.uniform_filter(img, self.W, mode="nearest")

            maxlinestrength = float('-inf')*np.ones(img.shape)


            for theta in range(0, 180, self.degree_step):
                # get line mask:
                linemask = self.find_linemask(theta,L)
                linemask = linemask/np.sum(linemask) # normilise the data
                imglinestrength = sp.ndimage.correlate(img,linemask, mode='constant')
                imglinestrength = imglinestrength - avgresponse
                maxlinestrength = np.maximum(maxlinestrength,imglinestrength)

            return maxlinestrength

        # Normalize the image intensity
        if self.use_normalization:
            self.image = self.normalize(img)
        else:
            self.image = img/np.max(img)

        #create empty segmented image
        self.image_segmented=np.zeros(self.image.shape)
        
        # processing the image region by region 
        for i in range(0, self.image.shape[0],int(roi_step/2)): 
            for j in range(0, self.image.shape[1],int(roi_step/2)):

              # find boundaries of the ROI
                x_st=np.max((i-int(roi_step/2),0))
                x_en=np.min((img.shape[0],i+int(roi_step/2)+1))
                y_st=np.max((j-int(roi_step/2),0))
                y_en=np.min((img.shape[1],j+int(roi_step/2)+1))
                ROI= standardize_2(img[x_st:x_en,y_st:y_en])

                #segmentation
                features=np.zeros(ROI.shape)

                #iterate over window size
                for L in range(self.W_start, self.W+1, self.step):
                    weight=find_lineresponse(L, ROI)
                    weight=standardize_2(weight)
                    features= features+weight

                feature_map=features/(L+1)
                ROI_segmented=features/(L+1)
                ROI_segmented=standardize_2(ROI_segmented)

                #threshold the features

                threshold_local=np.mean(ROI_segmented)*threshold

                ROI_segmented[ROI_segmented<threshold_local]=0
                ROI_segmented[ROI_segmented>=threshold_local]=1

                #save the segmented part

                # define margens
                if x_st==0:
                    margen_x_st=0
                else:
                    margen_x_st=margen_step
                    
                if y_st==0:
                    margen_y_st=0
                else:
                    margen_y_st=margen_step
                    
                if x_en==img.shape[0]:
                    margen_x_en=0
                else:
                    margen_x_en=margen_step
                    
                if y_en==img.shape[1]:
                    margen_y_en=0
                else:
                    margen_y_en=margen_step

                self.image_segmented[x_st+margen_x_st:x_en-margen_x_en,y_st+margen_y_st:y_en-margen_y_en]=ROI_segmented[margen_x_st:ROI_segmented.shape[0]-margen_x_en,margen_y_st:ROI_segmented.shape[1]-margen_y_en]
                

                
        #threshold based on image intensity
        self.image_segmented[self.image<=img_threshold]=0

        #remove small details
        self.image_segmented=self.remove_small_details(self.image_segmented, self.min_size)
        
        # fill holes if any
        self.image_segmented=morphology.closing(self.image_segmented, morphology.square(3))


        return self.image_segmented, feature_map


    def remove_small_details(self, img, size=1):
        '''
        remove small details from the segmented image

        '''

        # find separate parts
        parts=skimage.measure.label(img)

        # iterate over small parts
        for reg in range(0, np.max(parts)+1):
            region_img=np.zeros(parts.shape)
            region_img[parts==reg]=1
            # remove regions smaller than given size
            if np.sum(region_img)< size:
                img[parts==reg]=0
        
        return img
    