#########################################################
#
# tracklinking: connecting tracklets (short tracks)
#        
#########################################################

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
import numpy as np
import math
from pgmpy.inference import VariableElimination
import json
import skimage
from skimage import io
from tqdm import tqdm
import scipy as sp
from scipy.optimize import linear_sum_assignment 

import networkx as nx
import warnings
warnings.filterwarnings("ignore")

class GraphicalModelTracking(object):
    """
    Class to perform track linking with Bayesian network: 
    The code builds a BN with a given conditional probabilities
    The connection between tracklets are  based on the probability of the connectivity variable.
    
    """
    def __init__(self):
        """Initialize variables
        """
        
        self.topology='complete' # type of the BN topology: currently only "complete" option is available
        self.movie=[]
        self.tracklets={} # input tracklets
        self.tracks={} # output tracks
        self.tracks_before_filter={} # output tracks before short and not-moving tracks are removed
        self.data={}
        self.bgm_tracklet=BayesianModel() 
        self.track1={}
        self.track2={}
        
        self.tracklets_connection=[]
        self.track_pos=0 # ID for the new rearrange tracks
        self.track_data_framed={} # data rearrange by frames
        self.track_data_framed_start={}
        self.track_data_framed_end={}
        
        # parameters:
        
        self.frame_search_range=6 # frame range to search connection between tracks
        self.distance_search_range=12 # distance range to search connection between tracks
        
        self.frame_gap_tracklinking_1=5        
        self.direction_limit_tracklinking=180
        self.distance_limit_tracklinking=10 # distance in pix between two tracklets to be connected
        self.connectivity_threshold=0.7
        
        self.track_displacement_limit=0 # minimum displacement of the final track
        self.track_duration_limit=1 # minimum number of frames per final track
        
        self.speed_limit_tracklinking=0.5 # proportional speed difference between traklets to be connected
        
        self.intensity_limit_tracklinking=0.4 # intensity difference between tracklets to be connected 
        self.roi_size=12 # dimension of the roi with the object in it 
        
    def tracklets_pgm(self):
        '''
        Define the BN topology
        '''
    
######### nodes:

# O  - order of the tracks (0-not in order, 1 -  in order) 
# G  - gap between tracks (0 - zero frames, 1 - from 1-5 frames, 2 - more than 5 frames)
# L  - overLap of the tracks (0 - no overlap, 1 - overlap)
# S  - sequence - join of overlap and order (0 - not affinity, 1- affinity)
# P  - position - join of sequence and gap(0 - not affinity, 1- affinity)
# D  - direction similarity  (0 - not similar, 1- similar)
# C  - coordinates similarity (0 - 3 - near enough to connect;  4 - too far)
# SP - speed similarity  (0 - not similar, 1- similar)
# I  - intensity similarity  (0 - not similar, 1- similar)
# A  - connectivity score  (0 , 1)


        # Defining individual CPDs.
        cpd_o = TabularCPD(variable='O', variable_card=2, values=[[0.9, 0.1]])
        cpd_l = TabularCPD(variable='L', variable_card=2, values=[[0.9, 0.1]])
        cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.99, 1.0, 0.01, 0.9],
                                                                  [0.01, 0.0, 0.99, 0.1]], evidence=['O','L'], evidence_card=[2,2])
        
        cpd_g = TabularCPD(variable='G', variable_card=10, values=[[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]) 
        
        cpd_p = TabularCPD(variable='P', variable_card=2, values=[[0.99, 0.99, 0.99, 0.99, 0.99, 0.99,0.99, 0.99, 0.99, 0.99,   0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.99],
                                                                  [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01, 0.01, 0.01, 0.01,   0.99, 0.9, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.01]], evidence=['S','G'], evidence_card=[2,10])     
        
        cpd_d = TabularCPD(variable='D', variable_card=2, values=[[0.4, 0.6]])   
        
        cpd_sp = TabularCPD(variable='SP', variable_card=2, values=[[0.5, 0.5]])    
        
        cpd_m = TabularCPD(variable='M', variable_card=2, values=[[0.99, 0.1, 0.1, 0.2],
                                                                  [0.01, 0.9, 0.9, 0.8]], evidence=['D','SP'], evidence_card=[2,2])
        
        cpd_c = TabularCPD(variable='C', variable_card=5, values=[[0.2, 0.2, 0.2, 0.2, 0.2]])
        
        cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.5, 0.5]]) 
        
        
        # the score cpd:
        
        if self.topology=='complete':
        # all
            self.bgm_tracklet.add_edges_from([('O', 'S'),('L', 'S'),('S', 'P'), ('G', 'P'),('P', 'A'),('D', 'M'),('SP', 'M'),('M', 'A'),('C', 'A'), ('I', 'A')]) # graphical model
            cpd_a = TabularCPD(variable='A', variable_card=2, values=[[0.95, 0.96, 0.97, 0.98, 1.0,     0.95,0.96, 0.97, 0.98, 0.99,      0.25,0.28,0.31,0.35,0.95,     0.05, 0.10, 0.15, 0.20, 0.95,       0.95, 0.96, 0.97, 0.98, 1.0,     0.9, 0.92, 0.94, 0.95, 0.99,      0.2, 0.24, 0.28, 0.3, 0.9,       0.0, 0.04, 0.07, 0.1, 0.9],
                                                                      [0.05, 0.04, 0.03, 0.02, 0.0,     0.05,0.04, 0.03, 0.02, 0.01,      0.75,0.72,0.69,0.65,0.05,     0.95, 0.90, 0.85, 0.80, 0.05,       0.05, 0.04, 0.03, 0.02, 0.0,     0.1, 0.08, 0.06, 0.05, 0.01,      0.8, 0.76, 0.72, 0.7, 0.1,       1.0, 0.96, 0.93, 0.9, 0.1]], 
                                                                    evidence=['I','P','M', 'C'], evidence_card=[2,2,2,5])  

            self.bgm_tracklet.add_cpds(cpd_o, cpd_g, cpd_s, cpd_l, cpd_p, cpd_d, cpd_c, cpd_a, cpd_sp, cpd_m, cpd_i) # associating the CPDs with the network
        else:
            print("the BN topology cannot be identify - check the self.topology parameter")
        
        # check model: defined and sum to 1.
        self.bgm_tracklet.check_model()
          


    def save_tracks(self, filename):
        '''
        save the final tracks
        '''
        with open(filename, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False)
        

            
    def decision_tracklets_connection(self, track1, track2):
        '''
        make a decision on two tracks to be connected
        '''

        self.track1=track1
        self.track2=track2
        
        # calculates values for the nodes 
        
        # order of the tracks (0-not in order, 1 -  in order)
        val_O=self.decision_o()

        # gap between tracks (0 - zero frames, 1...8 - medium gap, 9 - large gap)
        val_G=self.decision_g()

        # overLap of the tracks (0 - no overlap, 1 - overlap)
        val_L=self.decision_l()

        # coordinates similarity (0 - 3 - near enough to connect;  4 - too far)
        val_C=self.decision_c()

        # orientation similarity  (0 - not similar, 1- similar)
        val_D=self.decision_d()

        # speed similarity  (0 - not similar, 1- similar)
        val_SP=self.decision_sp()

        # intensity similarity  (0 - not similar, 1- similar)
        val_I=self.decision_in()
        
        
        # make a query based on the calculated nodes
        infer = VariableElimination(self.bgm_tracklet) 
        
        if self.topology=='complete':
        #all
            q=infer.query(['A'], evidence={'O': val_O, 'G': val_G, 'L': val_L, 'C':val_C, 'D':val_D, 'SP': val_SP, 'I':val_I}, joint=False, show_progress=False)['A'].values[1] 
#            print('O', val_O, '   G', val_G, '   L', val_L, '   C',val_C, '    D',val_D, '    SP',  val_SP, '    I',val_I)
#            print("q ", q)
        else:
            q=0
            print("the BN topology cannot be identify - check the self.topology parameter")
     
        return q 
    
    
    def decision_s(self, val_o, val_l):
        '''
        decision on sequence - join overlap and order (0 - not affinity, 1- affinity)
        '''
        if val_o==1 and val_l==0:
            val_s=1
        else:
            val_s=0
        return val_s
    
        
    
    def decision_o(self):
        '''
        decision on order (0-not in order, 1 -  in order)
        '''
        pose_1_end=self.track1['frames'][-1]
        pos_2_start=self.track2['frames'][0]
        
        if pose_1_end<=pos_2_start:
            val=1 # there is an overlap
        else:
            val=0 # there is no overlap
        
        return val
        
    def decision_g(self):
        '''
        decision on  gap (0 - zero frames, 1...8 - medium gap, 9 - large gap)
        '''
        # define step for each of 10 positions
        if self.frame_gap_tracklinking_1>10:    
            gap_step=10/self.frame_gap_tracklinking_1
        else: 
            gap_step=1
            
        #calculate the gap
        gap=self.track2['frames'][0]-self.track1['frames'][-1]
        #define output
        if gap<=0: # maximum possible distance
            gap=9        
        elif gap>0 and gap<=self.frame_gap_tracklinking_1:
            val=int((gap-1)*gap_step) # no gap or medium gap
            
        else: # more than maximum gap
            val=9
        
        return val

    def decision_l(self):
        '''
        decision on overlapping (0 - no overlap, 1 - overlap)
        '''
        gap=self.track2['frames'][0]-self.track1['frames'][-1]
        
        if gap<=0:
            val=1
        else:
            val=0
            
        return val
    
    def decision_c(self):
        '''
        decision on coordinates (0 - 3 - near enough to connect;  4 - too far)
        '''
        #distance between tracks
        
        dist=np.sqrt((self.track2['trace'][0][0]-self.track1['trace'][-1][0])**2+(self.track2['trace'][0][1]-self.track1['trace'][-1][1])**2)
        N_frames_travelled=self.track2['frames'][0]-self.track1['frames'][-1]
        
        dist_per_frame=dist/N_frames_travelled
        
        step=self.distance_limit_tracklinking/4 # where 4 is number of steps to consider
        
        # the distance is to far
        if dist_per_frame>=self.distance_limit_tracklinking:
            val = 4
        else: # near vesicle grading
            val=int(dist_per_frame/step) # no gap or medium gap

        return val       
    
    def calculate_intensity(self, point, frameN):
        '''
        calculate intensity
        '''
        patch_size=self.roi_size
        x_min=int(point[0]-patch_size/2)
        x_max=int(point[0]+patch_size/2)
        y_min=int(point[1]-patch_size/2)
        y_max=int(point[1]+patch_size/2)
            
        if x_min>0 and y_min>0 and x_max<self.movie.shape[1] and y_max<self.movie.shape[2]:
                
            # create img
            track_img = self.movie[frameN, x_min:x_max, y_min:y_max]
            track_img = (track_img-np.min(track_img))/(np.max(track_img)-np.min(track_img))
            #calculate mean intensity inside the segment
            
            intensity=np.sum(track_img)/(patch_size*patch_size)            
        else:
            intensity=0
            
        return intensity

    def decision_in(self):
        '''
        decision on intensity (0 - not similar, 1- similar)
        '''
        
        #calculate intensity
        int1=self.calculate_intensity(self.track1['trace'][-1], self.track1['frames'][-1])
        int2=self.calculate_intensity(self.track2['trace'][0], self.track2['frames'][0])
        
        #comparison
        difference=abs(int1-int2)
        
        if difference>self.intensity_limit_tracklinking:
            val=0
        else:
            val=1
            
        return val      


    def decision_sp(self):
        '''
        decision on speed  (0 - not similar, 1- similar)
        '''

        if len(self.track1['trace'])==1 and len(self.track2['trace'])==1:
            speed1=0
            speed2=0
        elif len(self.track1['trace'])>1 and len(self.track2['trace'])>1:
        
            speed1= self.calculate_speed(self.track1['trace'], self.track1['frames'])
            speed2= self.calculate_speed(self.track2['trace'], self.track2['frames'])  

        elif len(self.track1['trace'])>1 and len(self.track2['trace'])==1:
            speed1= self.calculate_speed(self.track1['trace'], self.track1['frames'])
            speed2= self.calculate_speed(self.track1['trace']+self.track2['trace'], self.track1['frames']+self.track2['frames'])

        elif len(self.track1['trace'])==1 and len(self.track2['trace'])>1:
            speed1= self.calculate_speed(self.track1['trace']+self.track2['trace'], self.track1['frames']+self.track2['frames'])
            speed2= self.calculate_speed(self.track2['trace'], self.track2['frames'])
            
        difference=abs(speed2-speed1)/np.max((speed2,speed1))
            
        if difference>self.speed_limit_tracklinking:
            val = 0
        else:
            val = 1
            
        
        return val 
        
    def decision_d(self):
        '''
        decision on direction  (0 - not similar, 1- similar)
        '''
        
        if len(self.track1['trace'])==1 and len(self.track2['trace'])==1:
            direction1=0
            direction2=0
        elif len(self.track1['trace'])>1 and len(self.track2['trace'])>1:
        
            direction1= self.calculate_direction(self.track1['trace'])
            direction2= self.calculate_direction(self.track2['trace'])  

        elif len(self.track1['trace'])>1 and len(self.track2['trace'])==1:
            direction1= self.calculate_direction(self.track1['trace'])
            direction2= self.calculate_direction(self.track1['trace']+self.track2['trace'])

        elif len(self.track1['trace'])==1 and len(self.track2['trace'])>1:
            direction1= self.calculate_direction(self.track1['trace']+self.track2['trace'])
            direction2= self.calculate_direction(self.track2['trace'])
            
        
        difference=(abs(direction2-direction1))

        if difference>180:
            difference=360-difference

        if difference>self.direction_limit_tracklinking:
            val = 0
        else:
            val = 1

        return val  
    
    def calculate_speed(self, trace, frames):
        '''
        calculating the tracklet speed
        '''

        dist=np.sqrt((trace[-1][0]-trace[0][0])**2+(trace[-1][1]-trace[0][1])**2)
        time_frame=frames[-1]-frames[0]
        
        return dist/time_frame
        
    
    def calculate_direction(self, trace):
        '''
        calculating the tracklet direction
        '''        
        dist=np.sqrt((trace[-1][0]-trace[0][0])**2+(trace[-1][1]-trace[0][1])**2)
        if len(trace)>=2 and dist>=1:
            p1=trace[0]                        
            p2=trace[-1]
            xDiff = p2[0] - p1[0]
            yDiff = p2[1] - p1[1]
            return int(math.degrees(math.atan2(yDiff, xDiff)))
        else:
            return 0


            
    def join_tracklets(self,connections):
        '''
        join the tracklets in the final tracks based on the provided connections
        '''
        
        new_track={}
        frames=[]
        trace=[]
#        print("connections ID ", self.track_pos, "  : ", connections)

        #sort order
        
        frame_pos=[]
        for i in connections:
            tracklet=self.tracklets[str(int(i))]
            frame_pos.append(tracklet['frames'][0])
        
        frame_pos_order=(np.asarray(frame_pos)).argsort()
        
        connections_ordered=list(np.asarray(connections)[frame_pos_order])
        
#        print(connections_ordered)
        
        for i in connections_ordered:

            tracklet=self.tracklets[str(int(i))]
            # join frames
            frames=frames+tracklet['frames']
            
            #join trace 
            trace=trace+tracklet['trace']
            
            # add missing frames
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

        # create  a new Track 
        
        new_track.update({'trackID': self.track_pos, 'frames': new_frames, 'trace': new_trace}) 
        self.tracks_before_filter.update({self.track_pos:new_track}) 
        
        self.track_pos=self.track_pos+1 #trackID +1 for the next time
            

    def rearrange_track_to_frame_start_end(self, tracklets, movie):
        '''
        change data arrangment from orginised in terns of tracks to frames
        '''
        
        self.tracklets=tracklets
        self.track_data_framed_start={}
        self.track_data_framed_end={}
        for n_frame in range(0, movie.shape[0]):
            #rearrange the data
            track_list_start=[]
            track_list_end=[]
            for p in self.tracklets:
                tracklet=self.tracklets[p]
                frames=tracklet['frames']
                if n_frame==frames[0]: # if the frame at the beginning
                    track_list_start.append(tracklet['trackID'])
           

                if n_frame==frames[-1]: # if the frame is at the end
                    track_list_end.append(tracklet['trackID'])

            if len(track_list_start)!=0:
                self.track_data_framed_start.update({n_frame:track_list_start})
                
            if len(track_list_end)!=0:
                self.track_data_framed_end.update({n_frame:track_list_end})
                
    def train_model(self, data):       
        ''' train the BN '''
        self.bgm_tracklet.fit(data)
        self.bgm_tracklet.get_cpds()
        
    
    def predic_connectivity_score(self, predict_data):
        '''
        predict the connectivity score
        '''
        infer = VariableElimination(self.bgm_tracklet) 
        
        return infer.query(['A'], evidence=predict_data)['A'].values[1]


        
    def connect_tracklet_time(self):
        '''
        the main function for tracklinking: 
        connects tracklets (check connectivity and make a decision) 
        in the order of their temproal positioning
        '''
        checked_connections=[] # list of reviewed connections       
        possible_connections=[] # list of connection is a high connectivity score
        
        #iterate over each frame with ending tracks
        print("scanning frames for possble connections ...")
        for n_frame in tqdm(self.track_data_framed_end):
#
            tracklets_to_check=[]
            
            #for the distance calculation
            possible_connection_ID_end=[]
            possible_connection_ID_start=[]
            possible_connection_end_x=[]
            possible_connection_end_y=[]
            possible_commection_start_x=[]
            possible_commection_start_y=[]
            
            # iterate over frame and form a list with possible connections
            # based on the start point
            for frame_pos in range(n_frame+1,n_frame+self.frame_search_range): 
 
                if frame_pos in self.track_data_framed_start:
                    for trackID in self.track_data_framed_start[frame_pos]:
                         # get track ID
                        tracklets_to_check.append(trackID)
                        
                        # first position of the tracklet
                        track_one=self.tracklets.get(str(trackID))
                        
                        possible_commection_start_x.append(track_one['trace'][0][0]) 
                        possible_commection_start_y.append(track_one['trace'][0][1]) 
           
            # based on the end point           
            for t1 in self.track_data_framed_end[n_frame]:

                track1=self.tracklets.get(str(t1))
                possible_connection_end_x.append(track1['trace'][-1][0])
                possible_connection_end_y.append(track1['trace'][-1][1])

            
            #define arrays:
            possible_connection_ID_end=self.track_data_framed_end[n_frame]
            possible_connection_ID_start=tracklets_to_check
            
            possible_connection_ID=[] # list of all the possible connections
            
            for pos_end in possible_connection_ID_end:
                new_line=[]
                for pos_start in possible_connection_ID_start:
                    new_line.append([pos_end, pos_start])
                    
                possible_connection_ID.append(new_line)
            possible_connection_ID=np.asarray(possible_connection_ID)
            
            # Coordinates
            possible_connection_end_x=np.asarray(possible_connection_end_x).reshape((len(possible_connection_end_x),1))
            possible_connection_end_y=np.asarray(possible_connection_end_y).reshape((len(possible_connection_end_y),1))
            
            possible_commection_start_x=np.asarray(possible_commection_start_x).reshape((1, len(possible_commection_start_x)))
            possible_commection_start_y=np.asarray(possible_commection_start_y).reshape((1, len(possible_commection_start_y)))
            
            matrix_possible_connection_end_x=np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_connection_end_x            
            matrix_possible_commection_start_x = np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_commection_start_x

            matrix_possible_connection_end_y=np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_connection_end_y
            matrix_possible_commection_start_y = np.ones((len(possible_connection_ID_end), len(possible_connection_ID_start)))*possible_commection_start_y

            # all distance between end and start
            matrix_dist=np.sqrt((matrix_possible_commection_start_x-matrix_possible_connection_end_x)**2+(matrix_possible_commection_start_y-matrix_possible_connection_end_y)**2)
            
            # filter the connections based on the spacial distance
            comparison_list=possible_connection_ID[matrix_dist<=self.distance_search_range]

            # look through comparison list and calculate connectivity score
            pos_check=0
            for ID_compare in comparison_list:
                if ID_compare[0]!=ID_compare[1] and [ID_compare[0], ID_compare[1]] not in checked_connections:

                    track1=self.tracklets.get(str(ID_compare[0]))
                    track2=self.tracklets.get(str(ID_compare[1]))

                    checked_connections.append([ID_compare[0], ID_compare[1]])
                    connectivity_score=self.decision_tracklets_connection(track1, track2)
#                    print("\n tracklets", [ID_compare[0], ID_compare[1]]," possible connection", connectivity_score, " possible connection:", connectivity_score>=self.connectivity_threshold)
                    
                    # remove pairs with small connectivity score
                    if connectivity_score>=self.connectivity_threshold:
                        
                        possible_connections.append([track1['trackID'],track2['trackID'], connectivity_score])
                        pos_check+=1
                        
                else:

                    checked_connections.append([ID_compare[0], ID_compare[1]])   


        connected_tracklets=[] # new connections
        
        print("looking through possible connections  ... ")


        # iterate over small batches of tracklets
        
        N_tracklets=10000 # number of tracklets per batch

        # if there possible connections
        if len(possible_connections)!=0: 
            Nsteps=int(len(possible_connections)/N_tracklets)+1
            
            for step in tqdm(range(0, Nsteps)):
                # create matrix
                connections_start=step*N_tracklets
                connection_end=np.min((connections_start+N_tracklets, len(possible_connections)))
                possible_connections_part=possible_connections[connections_start:connection_end]
                               
                # list of the first tracklets in the range
                first_tracklet_set=np.asarray(possible_connections_part)[:,0].tolist()
                first_tracklet_set=list(dict.fromkeys(first_tracklet_set))
                second_tracklet_set=np.asarray(possible_connections_part)[:,1].tolist()
                second_tracklet_set=list(dict.fromkeys(second_tracklet_set))
                
                # cost matrix of connection
        
                prob_matrix=np.ones((len(first_tracklet_set),len(second_tracklet_set)))*200
    
                
                for p in tqdm(range(0, len(possible_connections_part))):
                    
                    pair1=np.asarray(possible_connections_part)[p,:].tolist() # extract the connected tracks
                    first_tracklet=int(pair1[0])
                    second_tracklet=int(pair1[1])
                    score=pair1[2]
                    pos_first=first_tracklet_set.index(first_tracklet)
                    pos_second=second_tracklet_set.index(second_tracklet)
                    prob_matrix[pos_first, pos_second]=1-score # score as opposite value
                
    
                # Hungerian to assign the  connections
                
                assignment=self.assignDetectionToTracks(prob_matrix) 
            
                # check the assignment
        
                for i in range(len(assignment)):
                    if (assignment[i] != -1):
                        # check with the cost distance threshold and remove if cost is high
                        # i - first tracklet position
                        #assignment[i] - second tracklet position
                        
                        if (prob_matrix[i][assignment[i]] > 1):
                            assignment[i] = -1
                            
                        else: # add the detection to the track
                            # check that the connection is unique
                            if connected_tracklets:
                                check_start_var=first_tracklet_set[i] not in np.asarray(connected_tracklets)[:,0]
                                check_end_var=second_tracklet_set[assignment[i]] not in np.asarray(connected_tracklets)[:,1]
                            else:
                                check_start_var=True
                                check_end_var=True
                            if check_start_var and check_end_var:
                                connected_tracklets.append([first_tracklet_set[i], second_tracklet_set[assignment[i]], 1-prob_matrix[i][assignment[i]]])
                            else: #compare scores if their are the same tracklets
                                current_score=1-prob_matrix[i][assignment[i]]
                                if check_start_var==False: # the first tracklet connection is the same
                                    
                                    first_pos_list=np.asarray(connected_tracklets)[:,0].tolist()
                                    pos_first=first_pos_list.index(first_tracklet_set[i])
                                    
                                    previous_score=connected_tracklets[pos_first][2]
                                    
                                    if current_score>previous_score:
                                        del connected_tracklets[pos_first]
                                        connected_tracklets.append([first_tracklet_set[i], second_tracklet_set[assignment[i]], 1-prob_matrix[i][assignment[i]]])
                                    
                                if  check_end_var==False:# the second tracklet connection is the same
                                    
                                    second_pos_list=np.asarray(connected_tracklets)[:,1].tolist()
                                    pos_second=second_pos_list.index(second_tracklet_set[assignment[i]])
                                    
                                    previous_score=connected_tracklets[pos_second][2]
                                    
                                    if current_score>previous_score:
                                        del connected_tracklets[pos_second]
                                        connected_tracklets.append([first_tracklet_set[i], second_tracklet_set[assignment[i]], 1-prob_matrix[i][assignment[i]]])
                                    

            connected_tracklets.sort()
            
                           
            connected_tracklets=np.asarray(connected_tracklets)[:,0:-1].tolist()


            # find connections
            
            G=nx.DiGraph()
            G.add_edges_from(connected_tracklets)
            self.tracklets_connection=list(nx.weakly_connected_components(G))
            
            for tracklets_to_track in tqdm(self.tracklets_connection, "saving connected tracklets"):
    
                tracklets_to_track=list(tracklets_to_track)
                tracklets_to_track.sort()           
                self.join_tracklets(tracklets_to_track)

      
        # save not connected tracklets
        
        for t3 in tqdm(self.tracklets, "saving not connected tracklets"):
            trackID=self.tracklets.get(t3)['trackID']
            if len(connected_tracklets)>0:
                c_var_x, c_var_y=np.where((np.asarray(connected_tracklets)==trackID))
                if len(c_var_x)==0:
                    self.join_tracklets([trackID]) 
            else:
                self.join_tracklets([trackID])
                
               
        # eliminate short  tracks:
        
        for t in tqdm(self.tracks_before_filter, "filtering tracks"):
            
            track=self.tracks_before_filter.get(t)
#            print(track)
            if len(track['trace'])>0:
                duration=track['frames'][-1]-track['frames'][0]+1
            else:
                duration=-1
            if  duration>=self.track_duration_limit:
                self.tracks.update({t:track})
                
                # this is for transition to the distionary storage - to display in list yet
                self.data.update({t:track})

 
        print("\n The MSP-tracker found ", len(self.tracks), " tracks")            
    
    def assignDetectionToTracks(self, cost):
        '''
        tracklet assignment based on the Hungerian algorithm
        '''
        N = cost.shape[0]
        assignment = []
        for _ in range(N):
            assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        return assignment
    