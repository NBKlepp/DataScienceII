'''
MultiRankWalk

Author : Nicholas Klepp

This is an implementation of the multi-rank walk algorithm. 
@see https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf

'''
import argparse
import numpy as np

import sklearn.metrics.pairwise as pairwise
import numpy.linalg as la
from matplotlib import pyplot

np.set_printoptions(threshold=np.nan)

DEBUG = False

'''
Read in the data from a file. 
@param filepath  the filepath to find the data
@return [X,y]        the training data, X, and corresponding labels, y 
'''
def read_data(filepath):
    Z = np.loadtxt(filepath)
    y = np.array(Z[:, 0], dtype = np.int)  # labels are in the first column
    X = np.array(Z[:, 1:], dtype = np.float)  # data is in all the others
    return [X, y]
'''
Save the numpy array to a file. 
@param filepath  The path of the file to save to
@param Y         The ndarray to save to a file
'''
def save_data(filepath, Y):
    np.savetxt(filepath, Y, fmt = "%d")

'''
Get the data as an ndarray of int values.
@param filepath The path to find the data at
@return An ndarray of int values from the path
'''
def get_truth(filepath):
    truth = np.loadtxt(filepath)
    return np.array(truth,dtype=np.int)
'''
Get a one-hot vector denoting the labeled instances of the data. 
Labels are in the range 0 to C for some C.
@param y  a vector of labels (-1 for unlabeled instances, 0-C for labeled instances)
@return   a one-hot vector representing the labeled data instances
'''
def get_labeled_instances(y):
    labeled_instances = []
    C = np.unique(y).size -1 
    for i in range(0,C):
        labeled_instances=np.append([labeled_instances],[np.where(y==i,1,0)],axis=0)
    return labeled_instances
'''
Get a hone-hot vector denoting the instances of the data with label c in 0 to C.
@param c  the label of interest
@param y  the vector of labels (-1 for unlabeled instances, 0-C for labeled instances)
@return   a one-hot vector representing the data instances with the specified label
'''
def get_labeled_instances(c,y):
    return np.where(y==c,1,0)

'''
Get the "degree" vector of an affinity matrix of a graph. 
@param a  an affinity matrix, where A(i,j)=1 iff v_i is adjacent to v_j
@return   the "degree" vector, where d_i is the sum of edge weights incident to v_i
'''
def get_D_Mat(A):                  
    return np.sum(A,axis=1)

'''
Get the top k seed instances of the data according to the degree vector for a particular label c. 
@param c  the label of interest
@param A  the affinity matrix for the data 
@param k  the number of seeds to retrieve 
@param y  the label vector
'''
def get_top_k_seeds(c,A,k,y):
    D = get_D_Mat(A)
    c_instancess = get_labeled_instances(c,y)
    DD= D*c_instances
    return np.argsort(-1*DD)[:k]

'''
Return a random set of seeds for a particular label c. 
@param c  the label of interest
@param A  the affinity matrix 
@param k  the number of seeds to return
@param y  the vector of labels. 
'''
def get_random_k_seeds(c,A,k,y):
    D = get_D_Mat(A)
    c_instances = get_labeled_instances(c,y)
    DD= np.where(D*c_instances!=0)[0] #the indices of the D values corresponding to label c
    u=np.sort(np.random.choice(DD,k,replace=False)) #a random selection of the indices of interest
    uu=np.array([])
    '''
    We need to generate a one-hot vector, uu, where uu[i]=1 iff i is one of the indices in u
    '''
    for index in u:
        uu=np.concatenate([uu,[0]*(index-uu.size),[1]])
    uu=np.concatenate([uu,[0]*(A.shape[0]-uu.size)])
    if DEBUG:
        print("getting random seeds...")
        print("c:",c)
        print("y:",y)
        print("c_instances:",c_instances)
        print("DD:",DD)
        print("u:",u)
        print("uu:",uu)
    return uu

'''
The multirank random walk algorithm. 
@param W       the weighted probability transition matrix
@param u       the seed vector
@param d       the damping value
@param epsilon the similarity limit
@return        the ranking vector  
'''
def random_walk(W,u,d,epsilon):
    r=np.zeros(W.shape[0]).T
    diff=2*epsilon
    x=0
    if DEBUG :
        print("From randomWalk")
        print("W.shape:",W.shape)
        print("u.shape:",u.shape)
        print("r.shape:",r.shape)
    while diff>epsilon:
        x+=1
        old_r = r
        r = (1-d)*u+d*np.dot(W,r)
        diff=la.norm(r-old_r)
    if DEBUG :
        print("number of random walk interations:",x)
    return r
'''
Get the weighted transition probability matrix for the data X. 
First find the "affinity" matrix for the data according to the pairwise 
rbf_kernel for affinity values. Then, weight the affinity matrix 
according to the degree of each node. 
@param X      the data
@param gamma  the gamma value to pass to the rbf_kernel method
@return       The weighted transition probability matrix for the data X
'''
def get_W(X,gamma):
    A = pairwise.rbf_kernel(X,gamma=gamma)
    D = get_D_Mat(A)
    return A/D

'''
Normalize a vector, u. 
@paramter u  the vector to normalize
@return      the normalized vector
'''
def normalize(u):
    return u/la.norm(u)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 4",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment4.py -i <input-data> -o <output-file> [optional args]")

    # Required args.
    parser.add_argument("-i", "--infile", required = True,
        help = "Path to an input text file containing the data.")
    parser.add_argument("-l", "--labels", required = True,
        help = "Path to an input text file containing the labels.")
    parser.add_argument("-o", "--outfile", required = True,
        help = "Path to the output file where the class predictions are written.")

    # Optional args.
    parser.add_argument("-d", "--damping", default = 0.95, type = float,
        help = "Damping factor in the MRW random walks. [DEFAULT: 0.95]")
    parser.add_argument("-k", "--seeds", default = 1, type = int,
        help = "Number of labeled seeds per class to use in initializing MRW. [DEFAULT: 1]")
    parser.add_argument("-t", "--type", choices = ["random", "degree"], default = "random",
        help = "Whether to choose labeled seeds randomly or by largest degree. [DEFAULT: random]")
    parser.add_argument("-g", "--gamma", default = 0.5, type = float,
        help = "Value of gamma for the RBF kernel in computing affinities. [DEFAULT: 0.5]")
    parser.add_argument("-e", "--epsilon", default = 0.01, type = float,
        help = "Threshold of convergence in the rank vector. [DEFAULT: 0.01]")

    args = vars(parser.parse_args())

    # Read in the variables needed.
    labels  = args['labels']    # File where the labels can be found. 
    outfile = args['outfile']   # File where output (predictions) will be written. 
    d = args['damping']         # Damping factor d in the MRW equation.
    k = args['seeds']           # Number of (labeled) seeds to use per class.
    t = args['type']            # Strategy for choosing seeds.
    gamma = args['gamma']       # Gamma parameter in the RBF kernel
    epsilon = args['epsilon']   # Convergence threshold in the MRW iteration.

    '''
    NOTE: 
    For RBF, see: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html#sklearn.metrics.pairwise.rbf_kernel
    '''

    # Read in the data.
    X, y = read_data(args['infile'])

    # Extract the relevant parameters. 
    C = np.unique(y)[1:]                       # The class labels
    m = X.shape[0]                             # The size of the input data (i.e. - number of points) 
    W = get_W(X,gamma)                         # The weighted probability transition matrix
    R = np.ones(m).reshape(1,m)                # The matrix to hold the ranking vectors in.
    A = pairwise.rbf_kernel(X,gamma=gamma)     # The "affinity" matrix
    truth = get_truth(labels)                  # The ground truth labels

    def get_seeds(c,A,k,y):
        if t == "random" : return get_random_k_seeds(c,A,k,y)
        else : return get_top_k_seeds(c,A,k,y)

    '''
    For each class,c, we need to run the multirank walk algorithm to find the ranking vector, r
    The ranking vector, r, gives us a prediction "score" for each observation for the class c. 
    The result is a c by m matrix R, where each row R_i is the ranking vector for class i.
    The max value in each column of R gives us the predicted class for that observation. 
    '''
    for c in C:
        u = get_seeds(c,A,k,y)
        u = normalize(u)
        r = random_walk(W,u,d,epsilon).reshape(1,m)    
        R = np.concatenate((R,r))
        if(DEBUG):
            print("u:",u)
            print("r:",r)
            print("R:",R)

    R=R[1:]
    if DEBUG : print("R:\n",R)
    prediction = np.array([])
    for i in range(0,m):
        if DEBUG : print(np.argmax(R[:,i]))
        prediction = np.concatenate((prediction,[np.argmax(R[:,i])]))
    if DEBUG : print(prediction)

    save_data(outfile,prediction)
    accuracy = np.sum(np.where(prediction==truth,1,0))/prediction.size
    if DEBUG : print("accuracy:",accuracy)
    
    

    
