'''
This is an implementation of stochastic SVD, comparing the results of this implementation with the "stock" sklearn implementation.
We have implemented power iterations as well as oversampling to improve performance. 
The results of the stochastic SVD solver implemented here are compared with a deterministic SVD solver implemented by sci-kit learn,
as well as the "stock" stochastic SVD solver implemented in sklearn. The results are plotted to the screen.
'''
import numpy as np
import argparse
from sklearn import datasets
import numpy.linalg
from matplotlib import pyplot as plt
from collections import OrderedDict
from sklearn.decomposition import PCA

#Constants used later to generate different plots
DEBUG = False
OVERSAMPLE = True
N_OVERSAMPLE = False
POWER = True
N_POWER = False

'''
The stochastic SVD solver. 
First compute the preconditioning matrix.
    Given a matrix of data, A, compute Y = A*Omega, where Omega ~ N(0,1)
    Perform QR decomposition on Y to computer Y=QR, where Q is our preconditioning matrix. 
    If we are interested in using power itereations, they may then be performed. 
Next, precondition the system by finding B = Q^t * A. 
    The singular values of the preconditioned matrix are much easier to solve for, and a 
    good approxiation of the singular values of our original system after we project back 
    them back into the original space.
@param A  the data 
@param k  the number of singular values we want to find
@param oversample  whether to use oversampling
@param power       whether to use power iterations
@param q           the number of power iterations to perform (if performing)
@return            a tuple (l,s,r) of left_singular_vectors, the singular values, and right singular vectors           
'''
def SSVD(A, k, oversample,power,q):
    #the size of the data
    m = A.shape[1]
    #find Y=A*Omega
    OMEGA = np.random.normal(0,1,m*k).reshape(m,k) if oversample == False else np.random.normal(0,1,m*2*k).reshape(m,2*k)
    Y = np.dot(A,OMEGA)
    #find Q using QR decomposition on Y
    Q,R = np.linalg.qr(Y)
    #use power iterations if necessary
    if power : Q = powerIter(Q,q,A)
    #precondition the system.
    B = np.dot(Q.T,A)
    #extract the singular values. 
    U_hat,s,V_hat_T = np.linalg.svd(B)
    V_hat = V_hat_T.T
    #project back into the original space (i.e. - pre-pre-conditioned...)
    lsv = np.dot(Q,U_hat)
    rsv = V_hat[:,0:k]
    ret = (lsv,s,rsv) if not oversample else (lsv,s[0:k],rsv)
    
    if DEBUG :
        print("s:",s)
        print("U_hat.shape:", U_hat.shape, "\ns.size:", s.size, "\nV_hat_t.shape:", V_hat_T.shape)
        print("ret:",ret)
    return ret
'''
A method to perform power iterations on a preconditioning matrix. 
    Let Q_0 be the original preconditioning matrix. 
    For i in 1 to q :
        Y = A * A^T * Q_i-1
        Q_i, R = QR_decomp(Y)

@param Q  the preconditioning matrix
@param q  the number of power iterations to perform
@param A  the original data matrix
'''
def powerIter(Q,q,A):
    for i in np.arange(q):
        Y = np.dot(A,np.dot(A.T,Q))
        Q = np.linalg.qr(Y)[0]
    return Q

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Assignment 5 Question 3",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment5_q3.py -k <basis> -p <power-iterations>") 

    # args.
    
    parser.add_argument("-k", "--basis", default = 10, type = int,
        help = "The number of basis vectors.")
    parser.add_argument("-p", "--power", default = 3, type = int,
        help = "The number of basis vectors.")

    args = vars(parser.parse_args())

    k = args['basis']
    q = args['power']

    #run tests on the MNIST dataset
    A = datasets.load_digits().data
    #the deterministic SVD solution (for comparison's sake)
    U,s,V_t = np.linalg.svd(A)
    deterministicSingularValues = s[0:k]

    #space for results
    stochasticRes = np.array([]*k).reshape(0,k)
    overSampleRes = np.array([]*k).reshape(0,k)
    q_vals = np.arange(1,q+1)
    power1 = np.array([]*k).reshape(0,k)
    power2 = np.array([]*k).reshape(0,k)
    power3 = np.array([]*k).reshape(0,k)
    powers = [power1,power2,power3]

    #some parameters for plotting
    q_colors = ['c','m','y']

    #run the SSVD solver 10 times without oversampling and without power iterations, storing the results
    for i in np.arange(10):
        if DEBUG : print("stochasticRes,SSVD(A,k,N_OVERSAMPLE)[1]:", SSVD(A,k,N_OVERSAMPLE,N_POWER,0)[1])
        s = SSVD(A,k,N_OVERSAMPLE,N_POWER,0)[1]
        stochasticRes = np.concatenate([stochasticRes, np.array([s])])
    #run the SSVD solver 10 times with oversampling but without power iterations, storing the results
    for i in np.arange(10):
        s = SSVD(A,k,OVERSAMPLE,N_POWER,0)[1]
        if DEBUG : print("s.size from OVERSAMPLE results:",s.size)
        overSampleRes = np.concatenate([overSampleRes, np.array([s])])
    #run the SSVD solver 10 times with oversampling and power iterations, storing the results
    for q in q_vals:
        for i in np.arange(10):
            s = SSVD(A,k,OVERSAMPLE,POWER,q)[1]
            if DEBUG : print("s.size from OVERSAMPLE results:",s.size)
            powers[q-1] = np.concatenate([powers[q-1], np.array([s])])
            
    #finding the SSVD results using the stock sklearn package
    pca = PCA(n_components=k,svd_solver = 'randomized')
    pca.fit(A)
    pca2 = PCA(n_components=k,svd_solver = 'randomized',iterated_power = 3)
    pca2.fit(A)

    #for use as the x-axis in our plots
    x = np.arange(1,k+1)
    if DEBUG : print("x.size:",x.size)

    #the difference is so small, it can be hard to see! 
    if DEBUG : print("Diff:",pca.singular_values_ - pca2.singular_values_)
   
    plots = []

    plots = plots + [plt.scatter(x, deterministicSingularValues,marker="x",label = "deterministic",c='r')]
    
    for s_val in stochasticRes :
        if DEBUG: print("s_val.size:",s_val.size)
        plots = plots + [plt.scatter(x,s_val,marker = "+",label = "no_oversample",c='b')]

    for s_val in overSampleRes :
        plots = plots + [plt.scatter(x,s_val,marker = "^",label = "oversampled",c='g')]

    for q in q_vals:
        for s_val in powers[q-1] :
            plots = plots + [plt.scatter(x,s_val,marker = "*",label = "power" + str(q) ,c=q_colors[q-1])]

    plots = plots + [plt.scatter(x,pca.singular_values_,marker = "v",label = "Halko0Power",c='k')]
    plots = plots + [plt.scatter(x,pca2.singular_values_,marker = "v",label = "Halko3Power",c='xkcd:sky blue')]
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels,handles))
    plt.legend(by_label.values(),by_label.keys())
    plt.title("Comparing the SVD Solvers")
    plt.show()
