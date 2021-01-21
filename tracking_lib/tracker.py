#########################################################
#
# tracklet formation
#        
#########################################################


import numpy as np
from scipy.optimize import linear_sum_assignment

class Track(object):
    '''
    class of the track object
     '''

    def __init__(self, first_point, first_frame, trackIdCount):

        self.track_id = trackIdCount  # identification of each track object
        self.trace_frame = [first_frame]  # list of frames
        self.skipped_frames = 0  # number of skipped frames (in sequence)
        self.trace = [first_point]  # trace, list of particle coordinates

class Tracker(object):
    '''
    class for linking with the Hungarian algorithm
    '''

    def __init__(self, dist_thresh=30, max_frames_to_skip=5, max_trace_length=100,
                 trackIdCount=0):

        self.dist_thresh = dist_thresh # maximum distance for the detection and track to be linked
        self.max_frames_to_skip = max_frames_to_skip # maximum number of skipped frames
        self.max_trace_length = max_trace_length # maximum trajectory length
        self.tracks = [] # list of tracks
        self.trackIdCount = trackIdCount # track ID to start with
        self.completeTracks=[] # final output
        
        # plot main parameters        
        print(" - - - - - trackelt formation: - - - - - - - ")
        print("Maximum distance to link: ", self.dist_thresh)
        print("Maximum skupped frames:  ", self.max_frames_to_skip)
        print("Maximum track length:  ", self.max_trace_length)
        
    def cost_calculation(self, detections):
        '''
        calculate distance based cost mastrix
        '''
        N = len(self.tracks)
        M = len(detections)
        cost = np.zeros((N, M))
        for i in range(len(self.tracks)):
            for j in range(len(detections)):
                try:
                    diff = np.array(self.tracks[i].trace[len(self.tracks[i].trace)-1]) - np.array(detections[j])
                    distance = np.sqrt((diff[0])**2 +(diff[1])**2 )

                    cost[i][j] = distance
                except:
                    pass
                
        # replace cost of the far distance with a very huge number
        cost_array=np.asarray(cost)
        cost_array[cost_array>self.dist_thresh]=10000
        cost=cost_array.tolist()        
        return cost
    
    def assignDetectionToTracks(self, cost):
        '''
        detection assignment based on the Hungerian algorithm
        '''
        
        N = len(self.tracks)
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]
        
        return assignment
    
    
    def update(self, detections, frameN):
        '''
        linking new detection with tracks: the results are in the self.completeTracks     
        ! after the last frame self.completeTracks should be appended !
        !      with the remaining tracks which were not completed     !
        '''

        # Create tracks if there is none yet
        if (len(self.tracks) == 0):
            for i in range(len(detections)):
                track = Track(detections[i], frameN, self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

            # tracking the targets if there were tracks before
        else:
            # Calculate cost matrix
            cost=self.cost_calculation(detections)
    
            # Hungarian Algorithm assignment:
            assignment=self.assignDetectionToTracks(cost)
 
            # add the position to the assigned tracks and detect not assigned tracks

            for i in range(len(assignment)):
                if (assignment[i] != -1):
                    # check with the cost distance threshold and remove if cost is high
                    if (cost[i][assignment[i]] > self.dist_thresh):
                        assignment[i] = -1
                        self.tracks[i].skipped_frames += 1
                        
                    else: # add the detection to the track
                        self.tracks[i].trace.append(detections[assignment[i]])
                        self.tracks[i].trace_frame.append(frameN)
                        self.tracks[i].skipped_frames = 0
                else: # add skipped frame to the track without assigned detection
                    self.tracks[i].skipped_frames += 1                

                        
            # not assigned detections
            un_assigned_detects = []
            for i_det in range(len(detections)):
                    if i_det not in assignment:
                        un_assigned_detects.append(i_det)
    
            # start new tracks
            if(len(un_assigned_detects) != 0):
                for i in range(len(un_assigned_detects)):
                    track = Track(detections[un_assigned_detects[i]], frameN,
                                  self.trackIdCount)              

                    self.trackIdCount += 1
                    self.tracks.append(track)
                        
     
            # removing tracks

            del_tracks = []
            
            #remove tracks which are longer than the max_length
            for i in range(len(self.tracks)):
                
                if ((self.tracks[i].trace_frame[-1]-self.tracks[i].trace_frame[0]+1) >= self.max_trace_length):
                    del_tracks.append(i)
                    
                #  remove tracks which has high number of skipped frames    
                if (self.tracks[i].skipped_frames >= self.max_frames_to_skip) and i not in del_tracks:
                    del_tracks.append(i)
        
        
     # when there are some tracks to delete:    
            if len(del_tracks) > 0:   

                val_compensate_for_del=0
                for id in del_tracks:
                    new_id=id-val_compensate_for_del
                    
                    self.completeTracks.append(self.tracks[new_id])
                    del self.tracks[new_id]
                    val_compensate_for_del+=1