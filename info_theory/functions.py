import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma, factorial

def fun(x):
    '''
    squares argument and adds log(3)
    '''

    return(x**2 + np.log(3))



def compute_MI_scalar(data, K):
    '''
    computes mutual information between two 1-D random variables using the max norm as the distance metric
    uses a binary tree to rapidly search for nearest neighbors (to implement KNN-algorithm 1 from Kraskov, et al.)
    data must be array of x,y positions, K must be 2 or greater
    '''
    

    # ---- COMPUTE NEAREST NEIGHBOR -----
    L = len(data)

    tree = cKDTree(data)

    temp_list = []
    for i in range(0, L):
        ref_point = data[i,:]
        KNN = tree.query(ref_point, k = [K], p=np.inf, workers = 4)
        d = {'k':K,
            'X ref':ref_point[0],
              'Y ref':ref_point[1], #include stationary point
              'X KNN':data[int(KNN[1]),0], #include neighbor
              'Y KNN':data[int(KNN[1]),1],
                'ei2_X': np.abs(ref_point[0]-data[int(KNN[1]),0]),
                'ei2_Y': np.abs(ref_point[1]-data[int(KNN[1]),1]),
                'ei2': max( np.abs(ref_point[0]-data[int(KNN[1]),0]), np.abs(ref_point[1]-data[int(KNN[1]),1])),
                 'KNN_dist':KNN[0] }
        temp_list.append(d)

    neighbors = pd.DataFrame(temp_list)  
  
    
    # ---- COUNT CLOSE POINTS -----
    ni_X = []
    ni_Y = []

    for i in range(0,len(neighbors)):
        point_x = neighbors.iloc[i]['X ref']
        point_y = neighbors.iloc[i]['Y ref']

        xdiff = np.abs(point_x-neighbors['X ref'])
        num_X = len(xdiff[xdiff < neighbors.iloc[i]['ei2']]) #-1  #subtract 1 here (to exclude self)


        ydiff = np.abs(point_y-neighbors['Y ref'])
        num_Y = len(ydiff[ydiff < neighbors.iloc[i]['ei2']]) #-1

        ni_X = np.append(ni_X,num_X)
        ni_Y = np.append(ni_Y,num_Y)

        
    # ---- COMPUTE MUTUAL INFORMATION -----
    
    for i in range(0, len(ni_X)):
        if(ni_X[i] == 0 ):
            ni_X[i]=1
            
    for i in range(0, len(ni_Y)):
        if(ni_Y[i] == 0 ):
            ni_Y[i]=1
            
    nx = ni_X #+ 1 # add 1 because digamma(0) = -inf
    ny = ni_Y #+ 1
    
    
    # use k-1 since we had to K=2 to find nearest neighbor that wasn't self 
    I = digamma(K-1) - np.mean(digamma(nx) + digamma(ny)) + digamma(L)

    
    return(I)