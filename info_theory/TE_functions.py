import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import digamma, factorial
from .MI_functions import angle_distance


#--------------------TE SCALAR------------------------------
def compute_TE_scalar(Xf, Xp, Yp):
    '''
    computes transfer entropy using a KNN method
    takes a sample of X and Y from a sample of fixed past times- Xp, Yp, and a sample of X from a future time, Xf
    data type is array
    in Zhu, et al., which this routine is derived from, the notation is Xp, X-, Y-, standing for X predicted, and X and Y past, 
    respectively
    '''

    K = 2
    # ---- COMPUTE NEAREST NEIGHBOR -----

    L = len(data)

    ni_Xp = []
    ni_Xp_Yp = []
    ni_Xf_Xp = []
    
    for i in range(0, L):

        # compute subspace dist between ref point and every other point to find ei via max norm
        d = []
        dist = np.zeros((L,3)) 
        
        for j in range(0,L):
            
            temp_xf = np.abs(Xf[i] - Xf[j]) 
            
            temp_xp = np.abs(Xp[i] - Xp[j])

            temp_yp = np.abs(Yp[i] - Yp[j])
            
            d.append(max(temp_xf, temp_xp, temp_yp)) 

            dist[j,:] = [temp_xf,temp_xp, temp_yp] 

        d = [x for x in d if x > 0] 

        # record distance to nearest neighbor
        ei = np.min(d) 
        
        # count neighbors in xp subspace
        dist_xp = dist[:,1] 
        num_Xp = len(dist_xp[(0 < dist_xp) & (dist_xp < ei)]) #-1
        ni_Xp = np.append(ni_Xp,num_Xp)
               
        # count neighbors in xp, yp subspace
        dist_xp_yp = np.amax(np.c_[np.abs(Xp[i]-Xp[:]),np.abs(Yp[i]-Yp[:])],1)
        num_Xp_Yp = len(np.array(dist_xp_yp[(0 < dist_xp_yp) &  (dist_xp_yp < ei)]))
        ni_Xp_Yp = np.append(ni_Xp_Yp,num_Xp_Yp)
        
        # count neighbors in xp, xf subspace
        dist_xf_xp = np.amax(np.c_[np.abs(Xf[i]-Xf[:]),np.abs(Xp[i]-Xp[:])],1)
        num_Xf_Xp = len(np.array(dist_xf_xp[(0<dist_xf_xp) & (dist_xf_xp < ei)]))
        ni_Xf_Xp = np.append(ni_Xf_Xp,num_Xf_Xp)

    # ---- COMPUTE MUTUAL INFORMATION -----
        # only add 1 to zeros and don't subtract 1
    for i in range(0, len(ni_Xp)):
        if(ni_Xp[i] == 0 ):
            ni_Xp[i]=1

    for i in range(0, len(ni_Xp_Yp)):
        if(ni_Xp_Yp[i] == 0 ):
            ni_Xp_Yp[i]=1
            
    for i in range(0, len(ni_Xf_Xp)):
        if(ni_Xf_Xp[i] == 0 ):
            ni_Xf_Xp[i]=1

    # use k-1 since we had to K=2 to find nearest neighbor that wasn't self
    I = digamma(K-1) + np.mean(digamma(ni_Xp) - digamma(ni_Xp_Yp) - digamma(ni_Xf_Xp)) 
    return(I)



#--------------------TE ANGULAR-------------------------
def compute_TE_angular(Xf, Xp, Yp):
    
    K = 2
    # ---- COMPUTE NEAREST NEIGHBOR -----

    L = len(Xp)
    
    cos_Xf = np.cos(Xf)
    sin_Xf = np.sin(Xf)
    cos_Xp = np.cos(Xp)
    sin_Xp = np.sin(Xp)
    cos_Yp = np.cos(Yp)
    sin_Yp = np.sin(Yp)

    ni_Xp = []
    ni_Xp_Yp = []
    ni_Xf_Xp = []
    
    for i in range(0, L):

        # compute subspace dist between ref point and every other point to find ei via max norm
        d = []
        dist = np.zeros((L,3)) 
        
        for j in range(0,L):
            
            temp_xf = angle_distance(cos_Xf[i], cos_Xf[j], sin_Xf[i], sin_Xf[j]) 
            
            temp_xp = angle_distance(cos_Xp[i], cos_Xp[j], sin_Xp[i], sin_Xp[j])

            temp_yp = angle_distance(cos_Yp[i], cos_Yp[j], sin_Yp[i], sin_Yp[j])
            
            d.append(max(temp_xf, temp_xp, temp_yp)) 

            dist[j,:] = [temp_xf,temp_xp, temp_yp] 

        d = [x for x in d if x > 0] 

        # record distance to nearest neighbor
        ei = np.min(d) 
        
        # count neighbors in xp subspace
        dist_xp = dist[:,1] 
        num_Xp = len(dist_xp[(0 < dist_xp) & (dist_xp < ei)]) #-1
        ni_Xp = np.append(ni_Xp,num_Xp)
               
        # count neighbors in xp, yp subspace
        dist_xp_yp = np.amax(np.c_[angle_distance(cos_Xp[i], cos_Xp, sin_Xp[i], sin_Xp),
                                   angle_distance(cos_Yp[i], cos_Yp, sin_Yp[i], sin_Yp)],1)
        num_Xp_Yp = len(np.array(dist_xp_yp[(0 < dist_xp_yp) &  (dist_xp_yp < ei)]))
        ni_Xp_Yp = np.append(ni_Xp_Yp,num_Xp_Yp)
        
        # count neighbors in xp, xf subspace
        dist_xf_xp = np.amax(np.c_[angle_distance(cos_Xf[i], cos_Xf, sin_Xf[i], sin_Xf),
                                   angle_distance(cos_Xp[i], cos_Xp, sin_Xp[i], sin_Xp)],1)
        num_Xf_Xp = len(np.array(dist_xf_xp[(0<dist_xf_xp) & (dist_xf_xp < ei)]))
        ni_Xf_Xp = np.append(ni_Xf_Xp,num_Xf_Xp)

    # ---- COMPUTE MUTUAL INFORMATION -----
        # only add 1 to zeros and don't subtract 1
    for i in range(0, len(ni_Xp)):
        if(ni_Xp[i] == 0 ):
            ni_Xp[i]=1

    for i in range(0, len(ni_Xp_Yp)):
        if(ni_Xp_Yp[i] == 0 ):
            ni_Xp_Yp[i]=1
            
    for i in range(0, len(ni_Xf_Xp)):
        if(ni_Xf_Xp[i] == 0 ):
            ni_Xf_Xp[i]=1

    # use k-1 since we had to K=2 to find nearest neighbor that wasn't self
    I = digamma(K-1) + np.mean(digamma(ni_Xp) - digamma(ni_Xp_Yp) - digamma(ni_Xf_Xp)) 
    return(I)

