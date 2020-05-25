'''
    Cargo detector
    Python Version    : 3.6
'''

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

class Detectors(object):
    """
    Class to detect secretory protein vesicles in a frame
    
    Attributes
    ----------

    img_mssef: array
        image after multi-scale spot enhancing filter (MSSEF) (default is empty)  
        
    binary_mssef: array
        binary mask after mssef (default is empty)
        
    img_set: array
        original image sequence (default is empty)

    c: float
    MSSEF parameter - thresholding coefficient (default is 0.8)
    
    k_max: int
    MSSEF parameter - end of the iteration (default is 20)
    
    k_min: int 
    MSSEF parameter - start of the iteration (default is 1)
    
    sigma_min: float
    MSSEF parameter - min sigma for LOG  (default is 0.1)
    
    sigma_max: float
    MSSEF parameter - max sigma for LOG   (default is 3)  
    
    min_distance: float
    thresholding parameter - minimum distance between two max after MSSEF (default is 4.0)
    
    threshold_rel: float
    thresholding parameter - minimum pix value in relation to the image (default is 0.1)   
    
    box_size: int
    detection parameter -bounding box size (default is 32)  
    
    detection_threshold: float [0,1]
    classifier parameter - detection_threshold (default is 0.99)

    detected_vesicles: list
    list of the detected vesicles after pruning (default is empty)
    
    detected_candidates: list
    list of the detected vesicles before pruning(default is empty)
    
    substract_bg_step: int
    pre-processing parameter: number of frame to consider for the b/g substraction (default is 100)
    
    
    Methods
    ---------
    __init__(self)
    img_enhancement(img, kernal_medfilter=3)
    substract_bg_single(img_set, pos)
    sef(img, img_sef_bin_prev, sigma, c, print_val=0)
    mssef(img, c, k_max, k_min, sigma_min, sigma_max)
    classify_vesicle(centers, img, new_model, segment_size=16)
    detect(frameN, new_model)
    radialsym_centre(img)
    
    """
    def __init__(self):
        """Initialize variables
        """

        self.img_mssef=[]
        self.binary_mssef=[]
        self.img_set=[] # 
        
        # parameters for approach
        #MSSEF
        self.c=0.8 #0.01 # coef for the thresholding
        self.k_max=20 # end of  the iteration
        self.k_min=1 # start of the iteration
        self.sigma_min=0.1 # min sigma for LOG
        self.sigma_max=3. # max sigma for LOG     
        
        #thresholding
        self.min_distance=4 # minimum distance between two max after MSSEF
        self.threshold_rel=0.1 # min pix value in relation to the image
        
        self.box_size=16 # bounding box size for detection
        self.box_size_fit= 8 # bounding box size for gaussian fit
        
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
        Read parameters from the fiven file
        '''
        
        with open(file_name) as json_file: 
            data = json.load(json_file)

        settings=data['parameters']
        print("\n reading data: ", settings)
            
        #MSSEF
        self.c=settings['c'] #0.01 # coef for the thresholding
        self.k_max=settings['k_max'] # end of  the iteration
        self.k_min=settings['k_min'] # start of the iteration
        self.sigma_min=settings['sigma_min'] # min sigma for LOG
        self.sigma_max=settings['sigma_max']# max sigma for LOG     
        
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

        img_new=filters.median(img, selem=None, out=None, mask=None, shift_x=False, shift_y=False)
        img_enh=exposure.equalize_adapthist(img_new, kernel_size=None, clip_limit=0.01, nbins=256)

        
        return img_enh
    
    def substract_bg_single(self, img_set, pos):
        '''
        substract image and in-situ background
        '''
        start_i = pos-self.substract_bg_step # start frame
        if start_i<0:
            start_i=0
        end_i = start_i+2*self.substract_bg_step # end frame
        if end_i>=img_set.shape[0]:
            end_i=img_set.shape[0]
            start_i=end_i-2*self.substract_bg_step

        insitu=np.min(img_set[start_i:end_i], axis=0) # insitu calculation    
        
        # removing background by substraction
        img_3ch=np.copy(img_set[pos])
        img_new= img_3ch-insitu  #calculate some percent of the insitu
            
        return img_new

    def detect_bg(self, img_set, pos, frameN):
        '''
        find in-situ background
        
        input:
        
        img_set: array
            sequence of images
        pos: int
            frame number in the sequence
        frameN: int
            number of frames taken into account
        
        '''
        start_i = pos-int(frameN/2) # start frame
        if start_i<0:
            start_i=0
        end_i = start_i+frameN # end frame
        if end_i>=img_set.shape[0]:
            end_i=img_set.shape[0]
            start_i=end_i-frameN
    
        insitu=np.min(img_set[start_i:end_i], axis=0) # insitu calculation    
    
            
        return insitu

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
        Multi-scale spot enhancing filter:
the code is based on the paper "Tracking virus particles in fluorescence microscopy images 
using two-step multi-frame association," Jaiswal,Godinez, Eils, Lehmann, Rohr 2015
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

        check_centres=[[-10,-10]]
        img_segment_set=np.zeros((len(centers),segment_size,segment_size))
        i=0
        
        if len(centers)>0:
            for lm in centers:
        
                x_st=int(lm[0]-segment_size/2)
                x_end=x_st+segment_size
                y_st=int(lm[1]-segment_size/2)
                y_end=y_st+segment_size
                img_segment= np.copy(img[x_st: x_end, y_st:y_end])
                #preprocess the segment
                img_segment_set[i,:,:]=(img_segment-np.min(img_segment))/(np.max(img_segment)-np.min(img_segment))
                i=i+1
            x_data=img_segment_set.reshape((len(centers), img_segment.shape[0],img_segment.shape[1], 1))
            score_set = new_model.predict(x_data) 
            
            for pos in range(0, len(centers)):
                res_distance=np.sqrt(np.sum((np.asarray(check_centres)-np.asarray(centers[pos]))**2, axis=1))
                if score_set[pos][1]>self.detection_threshold and all(res_distance>self.min_distance):
        
                    updated_centers.append(centers[pos])
                    check_centres.append(centers[pos])

        return updated_centers


    def detect(self, frameN, new_model):
        """
        Main function for the particle detection
        """
#        print("----------- detector settings: ")
#        print("self.c =             ", self.c)
#        print("self.k_max =         ", self.k_max)
#        print("self.k_min =         ", self.k_min)
#        print("self.sigma_min =     ", self.sigma_min)
#        print("self.sigma_max =     ", self.sigma_max)
#        print("self.min_distance =  ", self.min_distance)
#        print("self.threshold_rel = ", self.threshold_rel)
#        print("self.box_size =      ", self.box_size)

        #background substraction 
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
        new_img=filters.median(img, selem=None, out=None, mask=None, shift_x=False, shift_y=False)
        for i in range(1, np.max(spots_labeled)+1):
            mask_label=np.zeros(spots_labeled.shape)
            mask_label[spots_labeled==i]=1
            img_label=skimage.img_as_float(np.asarray(mask_label*new_img).astype(np.int))
            local_peaks=peak_local_max(img_label,  threshold_rel=self.threshold_rel, min_distance=int(self.min_distance))
            
            for point in local_peaks:
                check_pos_var= (point[1]-self.box_size/2)>0 and  (gray.shape[1]-point[1]-self.box_size/2)>0 and (point[0]-self.box_size/2)>0 and  (gray.shape[0]-point[0]-self.box_size/2)>0
                if check_pos_var==True: # remove spots close to the boarder
                    point_new=[point[0], point[1]]
                    local_max.append(point_new) 
                    self.detected_candidates.append(point_new)   
    
    # # # # segmentation using watershed new_img
        updated_centers=self.classify_vesicle(local_max, self.img_set[frameN,:,:], new_model, segment_size=self.box_size)  
#        updated_centers=self.classify_vesicle(local_max, new_img, new_model)  
        
        if self.gaussian_fit==True:

            for lm in updated_centers: # loop over all the found vesicles
                # ROI for the vesicle
                img=self.img_set[frameN,:,:]
                img=filters.median(img, selem=None, out=None, mask=None, shift_x=False, shift_y=False)
                img_roi= np.copy(img[lm[0]-int(self.box_size_fit/2): lm[0]+int(self.box_size_fit/2), lm[1]-int(self.box_size_fit/2):lm[1]+int(self.box_size_fit/2)])  
                
                # radial fit
                
                x,y=self.new_fit( img_roi, self.expected_radius)
#                x,y=self.radialsym_centre(img_roi)
                if math.isnan(x):
                    x=lm[0]
                    y=lm[1]
                else:
                    x=lm[0]+x-int(self.box_size_fit/2)
                    y=lm[1]+y-int(self.box_size_fit/2)
                point_new=[x, y]            
                self.detected_vesicles.append(point_new) 
        else:
            self.detected_vesicles=updated_centers
#        print("detections: \n", self.detected_vesicles)
            
        print("candidates: ", len(self.detected_candidates), ";   detections ", len(self.detected_vesicles), "\n ")

        return self.detected_vesicles

    def radialsym_centre(self, img):
        '''
         Calculates the center of a 2D intensity distribution.
         
     Method: Considers lines passing through each half-pixel point with slope
     parallel to the gradient of the intensity at that point.  Considers the
     distance of closest approach between these lines and the coordinate
     origin, and determines (analytically) the origin that minimizes the
     weighted sum of these distances-squared.
     
     code is based on the paper: Raghuveer Parthasarathy 
    "Rapid, accurate particle tracking by calculation of radial symmetry centers"   
    
    input: 
        img: array
        image itself. Image dimensions should be even(not odd) number for both axis
        
    output:
        x: float
        x coordinate of the radial symmetry
        y: float
        y coordinate of the radial symmetry
        '''
        
        def lsradialcenterfit(m, b, w):
            '''
            least squares solution to determine the radial symmetry center
            inputs m, b, w are defined on a grid
            w are the weights for each point
            '''
            wm2p1=np.divide(w,(np.multiply(m,m)+1))
            sw=np.sum(wm2p1)
            smmw = np.sum(np.multiply(np.multiply(m,m),wm2p1))
            smw  = np.sum(np.multiply(m,wm2p1))
            smbw = np.sum(np.multiply(np.multiply(m,b),wm2p1))
            sbw  = np.sum(np.multiply(b,wm2p1))
            det = smw*smw - smmw*sw
            xc = (smbw*sw - smw*sbw)/det # relative to image center
            yc = (smbw*smw - smmw*sbw)/det # relative to image center
                
            return xc, yc
    
        # GRID
        #  number of grid points
        Ny, Nx = img.shape
        
        # for x
        val=int((Nx-1)/2.0-0.5)
        xm_onerow = np.asarray(range(-val,val+1))
        xm = np.ones((Nx-1,Nx-1))*xm_onerow
        
        # for y
        val=int((Ny-1)/2.0-0.5)
        ym_onerow = np.asarray(range(-val,val+1))
        ym = (np.ones((Ny-1,Ny-1))*ym_onerow).transpose()
    
        # derivate along 45-degree shidted coordinates
    
        dIdu = np.subtract(img[0:Nx-1, 1:Ny].astype(float),img[1:Nx, 0:Ny-1].astype(float))
        dIdv = np.subtract(img[0:Nx-1, 0:Ny-1].astype(float),img[1:Nx, 1:Ny].astype(float))
        
        
        #smoothing
        filter_core=np.ones((3,3))/9
        fdu=sp.signal.convolve2d(dIdu,filter_core,  mode='same', boundary='fill', fillvalue=0)
        fdv=sp.signal.convolve2d(dIdv,filter_core,  mode='same', boundary='fill', fillvalue=0)
    
        dImag2=np.multiply(fdu,fdu)+np.multiply(fdv,fdv)
    
        #slope of the gradient
        m = np.divide(-(fdv + fdu), (fdu-fdv))
        
        # if some of values in m is NaN 
        m[np.isnan(m)]=np.divide(dIdv+dIdu, dIdu-dIdv)[np.isnan(m)]
        
        # if some of values in m is still NaN
        m[np.isnan(m)]=0 
        
        
        # if some of values in m  are inifinite
        
        m[np.isinf(m)]=10*np.max(m)
        
        #shortband b
        b = ym - m*xm
        
        #weighting
        sdI2=np.sum(dImag2)
        
        xcentroid = np.sum(np.multiply(dImag2, xm))/sdI2
        ycentroid = np.sum(np.multiply(dImag2, ym))/sdI2
        w=np.divide(dImag2, np.sqrt(np.multiply((xm-xcentroid),(xm-xcentroid))+np.multiply((ym-ycentroid),(ym-ycentroid))))
        
        # least square minimisation
        xc,yc=lsradialcenterfit(m, b, w)
        
        # output replated to upper left coordinate
        x=xc + (Nx+1)/2
        y=yc + (Ny+1)/2
        
        return x, y
    
    def new_fit(self, img, rad):
        
        
        features = tp.locate(img, rad, topn=1, engine='python')
        if features.empty:
            pos=(float(img.shape[0]/2), float(img.shape[1]/2))
        else:
            pos=features[['x', 'y']].iloc[0].values
        
#        print("!!!!!!!!",pos)
#        plt.figure()
#        plt.imshow(img)
#        plt.plot(pos[0], pos[1], 'r*')
#        plt.show()
        
        return pos[0], pos[1]
