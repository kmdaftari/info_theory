import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, factorial

def fun(x):
    '''
    squares argument and adds log(3)
    '''

    return(x**2 + np.log(3))


#----------------MI SCALAR WITH BINARY TREE-------------------------------------------

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
    for i in range(0, L): # iterate over all points as reference points
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


#---------------------MI SCALAR WITHOUT BINARY TREE---------------------------------------------

def compute_MI_scalar_2(X,Y):
    '''
    computes MI of between two 1-D RVs using the max norm as distance metric
    uses loop to search for nearest neighbors instead of binary tree (to implement KNN-algorithm 1 from Kraskov, et al.)
    data must two arrays for X and Y
    K fixed at 1, so can only deploy KNN algorithm on first nearest neighbor
    '''

    K = 2
        # ---- COMPUTE NEAREST NEIGHBOR -----

    L = len(X)
    ni_X = []
    ni_Y = []
    for i in range(0, L): #iterate over reference points (each point is reference point)

        # compute subspace dist between ref point and every other point
        d = []
        dist = np.zeros((L,2)) 
        for j in range(0,L):
            temp_x = np.abs(X[i]- X[j])

            temp_y = np.abs(Y[i]- Y[j])

            d.append(max(temp_x,temp_y)) 

            dist[j,:] = [temp_x, temp_y] 

        d = [x for x in d if x > 0] 

        # record distance to nearest neighbor
        ei = np.min(d) 

        dist_x = dist[:,0] 
        dist_y = dist[:,1] 

        # count neighbors in subspaces
        num_X = len(dist_x[dist_x < ei]) #-1
        num_Y = len(dist_y[dist_y < ei]) #-1 

        ni_X = np.append(ni_X,num_X)
        ni_Y = np.append(ni_Y,num_Y)


    # ---- COMPUTE MUTUAL INFORMATION -----
        # only add 1 to zeros and don't subtract 1
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
    return(I )




#----------------------------MI ANGLES WITHOUT BINARY TREE--------------------
def compute_MI_scalar_angles(X,Y):
    '''
    compute MI between two RVs that are angles, Z = (X, Y), where X and Y are angles
    data input is transformed to (cos(X), cos(Y), sin(X), sin(Y))
    does not use binary tree, so K fixed at 2
    '''
    
    K = 2
    # ---- COMPUTE NEAREST NEIGHBOR -----

    L = len(X)
    cos1 = np.array(np.cos(X))
    cos2 = np.array(np.cos(Y))
    sin1 = np.array(np.sin(X))
    sin2 = np.array(np.sin(Y))


    ni_X = []
    ni_Y = []
    for i in range(0, L):

        # compute subspace dist between ref point and every other point
        d = []
        dist = np.zeros((L,2)) 
        for j in range(0,L):
            temp_x = angle_distance(cos1[i],cos1[j],
                                    sin1[i], sin1[j])

            temp_y = angle_distance(cos2[i],cos2[j],
                                    sin2[i], sin2[j])

            d.append(max(temp_x,temp_y)) 

            dist[j,:] = [temp_x, temp_y] 

        d = [x for x in d if x > 0] 

        # record distance to nearest neighbor
        ei = np.min(d) 

        dist_x = dist[:,0] 
        dist_y = dist[:,1] 

        # count neighbors in subspaces
        num_X = len(dist_x[dist_x < ei]) #-1
        num_Y = len(dist_y[dist_y < ei]) #-1 

        ni_X = np.append(ni_X,num_X)
        ni_Y = np.append(ni_Y,num_Y)


    # ---- COMPUTE MUTUAL INFORMATION -----
        # only add 1 to zeros and don't subtract 1
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



#-------------------MI ANGLE AND SCALAR WITHOUT BINARY TREE-----------

def compute_MI_scalar_mixed(angle_data, linear_data):
    '''
    computes MI between two RVs, one which is an angle and another which is a scalar
    input type is array
    does not use binary tree
    '''
    
    K = 2
    # ---- COMPUTE NEAREST NEIGHBOR -----

    L = len(angle_data)
    cos1 = np.array(np.cos(angle_data))
    sin1 = np.array(np.sin(angle_data))

    ni_X = []
    ni_Y = []
    for i in range(0, L):

        # compute subspace dist between ref point and every other point
        d = []
        dist = np.zeros((L,2)) 
        for j in range(0,L):
            temp_x = angle_distance(cos1[i],cos1[j],
                                    sin1[i], sin1[j])

            temp_y = np.abs(linear_data[i]- linear_data[j])

            d.append(max(temp_x,temp_y)) 

            dist[j,:] = [temp_x, temp_y] 

        d = [x for x in d if x > 0] 

        # record distance to nearest neighbor
        ei = np.min(d) 

        dist_x = dist[:,0] 
        dist_y = dist[:,1] 

        # count neighbors in subspaces
        num_X = len(dist_x[dist_x < ei]) #-1
        num_Y = len(dist_y[dist_y < ei]) #-1 

        ni_X = np.append(ni_X,num_X)
        ni_Y = np.append(ni_Y,num_Y)


    # ---- COMPUTE MUTUAL INFORMATION -----
        # only add 1 to zeros and don't subtract 1
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



#--------------------MI SCALAR 4D WITHOUT BINARY TREE---------------------
def compute_MI_vector(dataX, dataY): 
    '''
    computes MI between two 2D RVs, i.e., Z1 = (X1,Y1), Z2 = (X2, Y2)
    inpute type is array for both X and Y values: [X1, X2], [Y1, Y2]
    does not use binary tree, so K fixed at 2
    '''

    K = 2
    # ---- COMPUTE NEAREST NEIGHBOR -----
    L = len(dataX)

    ni_X = []
    ni_Y = []
    for i in range(0, L): # go through all Zi
        
        d = []
        dist = np.zeros((L,2))
        
        for j in range(0,L): # compute max norm between all Zj and Zi by computing subspace norms and taking max
            
            temp_x = euclidian_dist(dataX[i,0],dataX[i,1],dataX[j,0],dataX[j,1])
            temp_y = euclidian_dist(dataY[i,0],dataY[i,1],dataY[j,0],dataY[j,1])

            d.append(max(temp_x,temp_y)) # pick out max norm
            
            dist[j,:] = [temp_x,temp_y] # record distance

        # record distance between ref_point and every other point 
        d = [x for x in d if x > 0] #remove distance from self, which is 0

        # record distance to nearest neighbor
        ei = np.min(d) 

        # count neighbors in subspaces
        dist_x = dist[:,0]
        dist_y = dist[:,1]
        
        num_X = len(dist_x[dist_x < ei]) #-1
        num_Y = len(dist_y[dist_y < ei]) #-1 
    
        ni_X = np.append(ni_X,num_X)
        ni_Y = np.append(ni_Y,num_Y)

        
    # ---- COMPUTE MUTUAL INFORMATION -----
        # only add 1 to zeros and don't subtract 1
    for i in range(0, len(ni_X)):
        if(ni_X[i] == 0 ):
            ni_X[i]=1
            
    for i in range(0, len(ni_Y)):
        if(ni_Y[i] == 0 ):
            ni_Y[i]=1
            
    nx = ni_X #+ 1 # add 1 because digamma(0) = -inf
    ny = ni_Y #+ 1
#     print('nx, ny -------')
#     print(nx,ny)

    # use k-1 since we had to K=2 to find nearest neighbor that wasn't self
    I = digamma(K-1) - np.mean(digamma(nx) + digamma(ny)) + digamma(L)

    
    return(I )




#--------ANGLE DISTANCE FUNCTION----------

def angle_distance(cos1, cos2, sin1, sin2):
    '''
    computes min distance between two angles
    '''

    return(np.arccos(cos1*cos2 + sin1*sin2))

#----------EUCLIDIAN DISTANCE FUNCTION-------------

def euclidian_dist(x1,y1,x2,y2):
    '''
    computes distance between two x,y pairs
    '''

    return(np.sqrt((x1-x2)**2 + (y1-y2)**2))