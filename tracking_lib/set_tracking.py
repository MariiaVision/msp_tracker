#########################################################
#
# class to set parameters for the tracking
#           author: Mariia Dmitrieva, University of Oxford
#########################################################

#from skimage import io
import numpy as np

from detectors_dl_new import Detectors
from gm_tracking_new import GraphicalModelTracking
from tracker import Tracker
#from detectors_dl_spinning import Detectors
from keras.models import  load_model
import json 


class TrackingSetUp(object):
    '''
    Class to set parameters for detection, linking and membrane evaluation
        

    Methods
    ---------
    __init__(self)
    detection(self, frameN)
    detection_parameter_to_file(self)
    detection_parameters_from_file(self)
    '''
    
    
    def __init__(self):
        """Initialize variables
        """
        
        
        self.movie=[] # the movie for detection and tracking
        self.movie_membrane=[] # the membrane movie
        self.start_frame=0
        self.end_frame=10
        
        # testing results
        
        self.detection_vesicles=[]
        self.detection_candidates=[]
        self.tracks=[]
        self.tracklets=[]
        
        #
        self.detection_parameter_path="detection_temp.txt"
        self.linking_parameter_path="linking_temp.txt"
        
    # # # # # DETECTION parameters # # # # #
        
        #MSSEF
        self.c=1.1 #0.01 # coef for the thresholding
        self.k_max=5 # end of  the iteration
        self.k_min=1 # start of the iteration
        self.sigma_min=1 # min sigma for LOG
        self.sigma_max=2# max sigma for LOG     
        
        #thresholding
        self.min_distance=5 # minimum distance between two max after MSSEF
        self.threshold_rel=0.3 # min pix value in relation to the image
        self.box_size=16 # bounding box size for detection
        
        #CNN based classification        
        self.detection_threshold=0.8         
        self.substract_bg_step =100   
        
        #option for Gaussian fit
        self.gaussian_fit=True        
        self.cnn_model_path="dl_weight/cnn-weights-spiral_disk-39-0.95.hdf5"
        
    # # # # # LINKING parameters # # # # #
    
        # tracking: short tracks         
        self.tracker_distance_threshold=2
        self.tracker_max_skipped_frame=1
        self.tracker_max_track_length=3
        
        # tracking: tracklinking
        self.tracklinking_Npass=1 # number of tracklinking passes
        self.tracklinking_path1_topology='complete' # topology type      
        self.tracklinking_path1_frame_gap_0=1
        self.tracklinking_path1_frame_gap_1=4        
        self.tracklinking_path1_direction_limit=50
        self.tracklinking_path1_distance_limit=1.5 # distance in pix between two tracklets to be connected 
        self.tracklinking_path1_connectivity_threshold=0.7
        self.tracklinking_path1_speed_limit=0.5
        self.tracklinking_path1_intensity_limit=0.4
        
        # filter final tracks
        self.tracklinking_path1_track_displacement_limit=0 # minimum displacement of the final track
        self.tracklinking_path1_track_duration_limit=3 # minimum number of frames per final track
                
        
        

    def detection(self, frameN):
        '''
         run detection with given parameters
        '''
        
        #set up parameters
        detector=Detectors()
        
        #MSSEF
        detector.c=self.c #0.01 # coef for the thresholding
        detector.k_max=self.k_max # end of  the iteration
        detector.k_min=self.k_min # start of the iteration
        detector.sigma_min=self.sigma_min # min sigma for LOG
        detector.sigma_max=self.sigma_max# max sigma for LOG     
        #thresholding
        detector.min_distance=self.min_distance # minimum distance between two max after MSSEF
        detector.threshold_rel=self.threshold_rel # min pix value in relation to the image
        
        detector.box_size=self.box_size # bounding box size for detection
        detector.img_set=self.movie
        #CNN based classification        
        detector.detection_threshold=self.detection_threshold
         
        detector.substract_bg_step =self.substract_bg_step   
        
        #option for Gaussian fit
        detector.gaussian_fit=self.gaussian_fit

        # loading CNN        
        cnn_model=load_model(self.cnn_model_path) 
        #######################################################################

            
#        frame=self.movie[frameN,:,:]
#        frame=(frame-np.min(frame))/(np.max(frame)-np.min(frame))
        
        self.detection_vesicles=detector.detect(frameN, cnn_model)    

        self.detection_candidates=detector.detected_candidates
            
        return self.detection_vesicles
    

    def detection_parameter_to_file(self):
        '''
        save new detection parameters
        '''
        
        
        parameters={'c':self.c, 'k_max':self.k_max , 'k_min':self.k_min,
                    'sigma_min':self.sigma_min,  'sigma_max':self.sigma_max,  'min_distance':self.min_distance,
                    'threshold_rel':self.threshold_rel,'box_size':self.box_size,'detection_threshold':self.detection_threshold, 
                    'substract_bg_step':self.substract_bg_step, 'gaussian_fit': self.gaussian_fit, 'cnn_model_path':self.cnn_model_path}
        data={'parameters':parameters}
        
        # save the parameters       
        with open(self.detection_parameter_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False) 
        print("\n saving data: ", data['parameters'])
            
    def detection_parameters_from_file(self, param_path=''):
        '''
        read settings from the file
        '''

        # load detection    
        if param_path=='':
            param_path=self.detection_parameter_path
            
        with open(param_path) as json_file: 
            data = json.load(json_file)
        settings=data['parameters']
            
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
            
        
    def linking(self):
        '''
        run linking with given parameters
        '''
        
        # set up parameters 
        
        # detection
        
        #set up parameters
        detector=Detectors()
        
        #MSSEF
        detector.c=self.c #0.01 # coef for the thresholding
        detector.k_max=self.k_max # end of  the iteration
        detector.k_min=self.k_min # start of the iteration
        detector.sigma_min=self.sigma_min # min sigma for LOG
        detector.sigma_max=self.sigma_max# max sigma for LOG     
        #thresholding
        detector.min_distance=self.min_distance # minimum distance between two max after MSSEF
        detector.threshold_rel=self.threshold_rel # min pix value in relation to the image
        
        detector.box_size=self.box_size # bounding box size for detection
        detector.img_set=self.movie
        #CNN based classification        
        detector.detection_threshold=self.detection_threshold
         
        detector.substract_bg_step =self.substract_bg_step   
        
        #option for Gaussian fit
        detector.gaussian_fit=self.gaussian_fit

        # loading CNN        
        cnn_model=load_model(self.cnn_model_path) 
  
        ###########
        
        # first linking
        tracker = Tracker(self.tracker_distance_threshold, self.tracker_max_skipped_frame, self.tracker_max_track_length, 0)
        
        #tracklinking pass 1
        tracklink=GraphicalModelTracking()
        
        tracklink.topology=self.tracklinking_path1_topology
        tracklink.tracklets_pgm()
        
        tracklink.frame_gap_tracklinking_0=self.tracklinking_path1_frame_gap_0
        tracklink.frame_gap_tracklinking_1=self.tracklinking_path1_frame_gap_1        
        tracklink.direction_limit_tracklinking=self.tracklinking_path1_direction_limit
        tracklink.distance_limit_tracklinking=self.tracklinking_path1_distance_limit
        tracklink.connectivity_threshold=self.tracklinking_path1_connectivity_threshold
        tracklink.speed_limit_tracklinking=self.tracklinking_path1_speed_limit
        tracklink.intensity_limit_tracklinking=self.tracklinking_path1_intensity_limit

        tracklink.frame_search_range=tracklink.frame_gap_tracklinking_1+2 
        tracklink.distance_search_range=tracklink.distance_limit_tracklinking+1
        
        # filter final tracks
        tracklink.track_displacement_limit=self.tracklinking_path1_track_displacement_limit
        tracklink.track_duration_limit=self.tracklinking_path1_track_duration_limit 
        tracklink.movie=self.movie
        
        # run tracking
        # step 1

        for frameN in range(self.start_frame,self.end_frame):
            #detection
            vesicles=detector.detect(frameN, cnn_model)
            #tracking
            tracker.update(vesicles, frameN)            
        
        # current tracks to save in complete
        for trackN in range(0, len(tracker.tracks)):
            tracker.completeTracks.append(tracker.tracks[trackN])
        # rearrange the data into disctionary
        data={}
        for trackN in range(0, len(tracker.completeTracks)):
            data.update({str(tracker.completeTracks[trackN].track_id):{
                    'trackID':tracker.completeTracks[trackN].track_id,
                    'trace': np.asarray(tracker.completeTracks[trackN].trace).astype(int).tolist(),
                    'frames':tracker.completeTracks[trackN].trace_frame,
                    'skipped_frames': tracker.completeTracks[trackN].skipped_frames
                    }})
            
        self.tracklets=data
        # step 2 tracklinking pass 1
        
        # set tracklets
        tracklink.rearrange_track_to_frame_start_end(data, self.movie)
            
        #connect tracklets       
        tracklink.connect_tracklet_time()
        
        #final tracks
        self.tracks=tracklink.tracks
        
        return self.tracks
    
    
    def linking_parameter_to_file(self):
        '''
        save new detection parameters
        '''
        
        
        parameters={'tracker_distance_threshold':self.tracker_distance_threshold, 'tracker_max_skipped_frame':self.tracker_max_skipped_frame , 'tracker_max_track_length':self.tracker_max_track_length,
                    'tracklinking_Npass':self.tracklinking_Npass,  'tracklinking_path1_topology':self.tracklinking_path1_topology,  'tracklinking_path1_frame_gap_0':self.tracklinking_path1_frame_gap_0,
                    'tracklinking_path1_frame_gap_1':self.tracklinking_path1_frame_gap_1,'tracklinking_path1_direction_limit':self.tracklinking_path1_direction_limit,'tracklinking_path1_distance_limit':self.tracklinking_path1_distance_limit, 
                    'tracklinking_path1_connectivity_threshold':self.tracklinking_path1_connectivity_threshold, 'tracklinking_path1_speed_limit': self.tracklinking_path1_speed_limit, 'tracklinking_path1_intensity_limit':self.tracklinking_path1_intensity_limit,
                    'tracklinking_path1_track_displacement_limit':self.tracklinking_path1_track_displacement_limit, 'tracklinking_path1_track_duration_limit':self.tracklinking_path1_track_duration_limit}
        data={'parameters':parameters}
        
        # save the parameters       
        with open(self.linking_parameter_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False) 
        print("\n saving data: ", data['parameters'])
                
    def linking_parameters_from_file(self, param_path=''):
        '''
        read settings from the file
        '''

        # load detection    
        if param_path=='':
            param_path=self.linking_parameter_path
            
        with open(param_path) as json_file: 
            data = json.load(json_file)
        settings=data['parameters']
            
        # tracker step 1
        self.tracker_distance_threshold=settings['tracker_distance_threshold']
        self.tracker_max_skipped_frame=settings['tracker_max_skipped_frame']
        self.tracker_max_track_length=settings['tracker_max_track_length']

        # tracking: tracklinking
        self.tracklinking_Npass=settings['tracklinking_Npass'] 
        self.tracklinking_path1_topology=settings['tracklinking_path1_topology']      
        self.tracklinking_path1_frame_gap_0=settings['tracklinking_path1_frame_gap_0']
        self.tracklinking_path1_frame_gap_1=settings['tracklinking_path1_frame_gap_1']       
        self.tracklinking_path1_direction_limit=settings['tracklinking_path1_direction_limit']
        self.tracklinking_path1_distance_limit=settings['tracklinking_path1_distance_limit']
        self.tracklinking_path1_connectivity_threshold=settings['tracklinking_path1_connectivity_threshold']
        self.tracklinking_path1_speed_limit=settings['tracklinking_path1_speed_limit']
        self.tracklinking_path1_intensity_limit=settings['tracklinking_path1_intensity_limit']
        
        # filter final tracks
        self.tracklinking_path1_track_displacement_limit=settings['tracklinking_path1_track_displacement_limit']
        self.tracklinking_path1_track_duration_limit=settings['tracklinking_path1_track_duration_limit']
            
                