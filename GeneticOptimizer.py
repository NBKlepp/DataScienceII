'''
GeneticOptimizer

Author: Nicholas Klepp

The genetic optimizer is an implementation of a binary logistic regression document 
classifier the weights of which are learned via genetic programming. 

Given a set of training documents, X, and the corresponding training labels, y, the 
binary classification logistic regression model
                    y(X;Theta) = exp(-Theta*X) / ( 1 + exp(-Theta *X))
is learned by via a simple genetic programming algorithm with respect to parameter Theta.
The predictions for an unlabeled set of documents, z, is given by
                    y_hat(z;Theta) = floor(exp(-Theta*z) / ( 1 + exp(-Theta *z) ) + .5)
In other words, the prediction y_hat(z;Theta) is 0 if the logistic regression function is less
.5 and 1 if the logistic regression function is greater than or equal to .5.

The genetic programming:
A initial population of possible Theta parameters, P, of size n is drawn from a random 
distribution.  For some predetermined number of iterations (generations) g, the prediction 
function y_hat(X;Theta) is calculated for each p in P and the population evolves.  Each p in 
P is judged for its fitness for  prediction, the fitness function being the average number of 
documents for which p generates an accurate prediction. The top s percent of performers according to 
the fitness criteria are selected, and the remaining members of the population are culled. The 
remaining members of the population are then randomly "mated" to produce the next generation 
of P, whereby two p_i and p_j combine to form p', the average of the two weight vectors. The 
offspring then undergo random mutation. If Theta=[Theta_0, Theta_1, ... , Theta_n] is the vector 
of weights used in the logistic classifier, then P describes a set of n_p many possible weights 
for each parameter Theta_i. To mutate the population, for each Theta_i we calculate the mean, mu, 
and variance, sigma^2, for w_i over the entire population. Then, for each p=[w_p0, w_p1, ... , w_pn] 
in P, with probability m each w_pi is reassigned a new value pulled from a standard normal distribution 
centered at mu with standard deviation sigma^2. The next iteration of evolution then begins. 

Required Input: 
        input training documents, X
        input training labels,    y
        unlabeled documents,      z

Optional Input:
        population size,  n_p [DEFAULT: 100] 
        mutation rate,    m   [DEFAULT: .01] 
        survival rate,    s   [DEFAULT: .3] 
        # of generations, g   [DEFAULT: 100]
'''

import argparse
import numpy as np

def _load_X(path):
    # Load the data.
    mat = np.loadtxt(path, dtype = int)
    max_doc_id = mat[:, 0].max()
    max_word_id = mat[:, 1].max()
    X = np.zeros(shape = (max_doc_id, max_word_id))
    for (docid, wordid, count) in mat:
        X[docid - 1, wordid - 1] = count
    return X

def _load_Z(path):
    mat = np.loadtxt(path,dtype = int)
    max_doc_id = mat[:,0].max()
    max_word_id = mat[:,1].max()
    Z = np.zeros(shape = (max_doc_id,max_word_id))
    Z = np.zeros(shape=(max_doc_id,X.shape[1]))
    for (docid,wordid,count) in mat:
        Z[docid-1,wordid-1]=count
    
    return Z[:,:X.shape[1]]

def _load_train(data, labels):
    # Load the labels.
    y = np.loadtxt(labels, dtype = int)
    X = _load_X(data)

    # Return.
    return [X, y]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 2",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
        prog = "python assignment2.py [train-data] [train-label] [test-data] <optional args>")
    parser.add_argument("paths", nargs = 3)
    parser.add_argument("-n", "--population", default = 100, type = int,
        help = "Population size [DEFAULT: 100].")
    parser.add_argument("-s", "--survival", default = 0.3, type = float,
        help = "Per-generation survival rate [DEFAULT: 0.3].")
    parser.add_argument("-m", "--mutation", default = 0.01, type = float,
        help = "Point mutation rate [DEFAULT: 0.01].")
    parser.add_argument("-g", "--generations", default = 100, type = int,
        help = "Number of generations to run [DEFAULT: 100].")
    parser.add_argument("-r", "--random", default = -1, type = int,
        help = "Random seed for debugging [DEFAULT: -1].")
    args = vars(parser.parse_args())

    # Do we set a random seed?
    if args['random'] > -1:
        np.random.seed(args['random'])

    # Read in the training data.
    X, y = _load_train(args["paths"][0], args["paths"][1])
    
    ### FINISH ME

    DEBUG = False
    
    #declare useful variables
    n=args['population']                    #the size of the population
    s=args['survival']                      #the survival rate
    m=args['mutation']                      #the mutation rate
    g=args['generations']                   #the number of generations to run the algorithm for
    d=int(X.shape[0])                            #the number of documents in the training set
    v=int(X.shape[1])                            #the size of the vocabulary

    if DEBUG : print("args:",args)
    test_data = args['paths'][2]
    
    #initialize a random population.
    pop = np.random.normal(loc=0,scale = 1,size = (n,v))    #an initial population drawn from a normal distribution in an nXv matrix
    if DEBUG : print("pop:",pop.shape)
    
    def copy_doc(docs):                                 #stack n many mXv arrays on top of each other for one nXmXv array
        if DEBUG: print("copy_doc")
        docs = np.array([docs]*n)                     
        return np.transpose(docs,[1,0,2])
    
    def combine_cubes(population,doc_copy):             #return an nXm matrix where M[i,j] is p_i DOT d_j
        if DEBUG: print("combining cubes")
        return np.sum(population*doc_copy,axis=2).T

    def h(z):                                         #the sigmoid function
        if DEBUG: print("h(z)")
        return 1.0 / (1.0 + np.exp(-1.0*z))           

    def predict(hz):                                  #the prediction rule
        if DEBUG: print("predict")
        return np.where(hz>1/2,1,0)
    
    def make_predictions(combo_cube):                 #return an nXm matrix where M[i,j] is the prediction from p_i for document d_j
        if DEBUG: print("make_predictions")  
        return predict(h(combo_cube))

    def cull_flock(population, preds,y):                        #assess the fitness of the population. 
        if DEBUG: print("cull_flock")
        accurates = np.array(preds==y)                          #a boolean array of where each prediction was correct              
        accuracies = np.where(accurates,1,0)                    #a binary array (1=>correct,0=>incorrect)
        if DEBUG: print("accuracies:",accuracies.shape)
        accuracies = np.sum(accuracies,axis=1)                  #sum of correct predictions for each population member
        accuracies = accuracies/accuracies.shape[0]             #the average correctness for each population member
        survivors = population[np.argsort(-1*accuracies)]       #sorting the population by their average accuracy descending
        return survivors[:int(np.floor(s*survivors.shape[0]))]

    def mate(survivors):                                                 #create a new population of size=n from survivors of size sn
        if DEBUG: print("mate")
        mate1=survivors[np.random.randint(survivors.shape[0],size=n)]
        mate2=survivors[np.random.randint(survivors.shape[0],size=n)]
        return (mate1+mate2)/2

    def mutate(offspring):                                                         #mutate the offspring
        if DEBUG: print("mutate:",m)
        pop_mean = np.mean(pop,axis=0)                                             #mean of each w_i over the entire population
        pop_var = np.var(pop,axis=0)                                               #variance of each w_i over the entire population
        pop_sd = np.sqrt(pop_var)                                                  #the sd ...
        mutations = np.random.normal(loc=pop_mean,scale=pop_sd,size=pop.shape)     #an entirely random population pulled from a normal dist with above params
        coin = np.random.binomial(1,m,pop.shape)                                   #a population of weighted coin flips (mostly 0s)
        return coin*mutations + (1-coin)*offspring

    for i in range(0,g):                                                           #run the generations
        if DEBUG : print("generation",i)
        doc_copy   = copy_doc(X)
        combo_cube = combine_cubes(pop,doc_copy)
        preds      = make_predictions(combo_cube)
        survivors  = cull_flock(pop,preds,y)
        offspring  = mate(survivors)
        mutants    = mutate(offspring)
        pop        = mutants

    doc_copy = copy_doc(X)                                          #one final generation
    combo_cube = combine_cubes(pop,doc_copy)
    preds = make_predictions(combo_cube)
    survivors = cull_flock(pop,preds,y)
    highlander = survivors[0]                                       #the most fit of the final generation...THERE CAN BE ONLY ONE! 

    Z = _load_Z(test_data)                                          #the test data
    
    predictions = predict(h(np.sum(highlander*Z,axis=1)))           #make predictions using the Highlander

    for p in predictions:
        print(p)
