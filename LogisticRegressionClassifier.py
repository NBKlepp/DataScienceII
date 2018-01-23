'''
Author: Nicholas Klepp
Date: Fall 2017

This file is an implementation of a logistic regression document classifier. 
The expected training data for the classifier is a file of line separated word count data and a file of document labels.
Each line in the word count training file should have the following format (white space separated): 

<document_id> <word_id> <count>

meaning that the word associated with the word ID <word_id> appears <count> many times in the document associated with the document ID <document_id>. 
The document IDs and word IDs should be sequential integers. 

The number, <row_number>, of each line in the document label training file should contain a single integer, which is the clas label for the document associated 
with the document ID <row_number>.

Use 'python LogistRegressionClassifier.py -help' for more information on how to run the document classifier. 
'''

import sys,numpy as np
from numpy import linalg as la
import argparse
import pandas as pd

STEP_SIZE = 0.0001

'''
A method to read the word count training file and return a matrix representing the word count training data.
See above note for specifications on the trainingDataFile format specifications. 
'''
def getData(trainingDataFile):
    data  = pd.read_table(trainingDataFile,header=None, names=["doc_id","word_id","count"],sep=" ") 
    words = data.word_id.max()
    docs  = data.doc_id.max()
    matrix = np.array([1]*docs*(words+1)).reshape((docs,(words+1)))
    for row in data.itertuples():
        matrix[row[1]-1,row[2]]=row[3]
    return matrix

'''
A method to read the training label file and return an array of labels.
See above note for specifications on the trainingLabelFile format specifications.
'''
def getLabels(trainingLabelFile):
    labels = pd.read_table(trainingLabelFile,header=None,names=["label"])
    return labels.label
'''
The sigmoid function. 
'''
def sigmoid(z) :
    return 1 / (1+np.exp(-z))

'''
The hypothesis function (a.k.a. - the prediction)
'''
def h(x,theta) :
    z = np.dot(x,theta)
    return sigmoid(z)

'''
The cost function. 
'''
def cost(X,y,theta):
    a = h(X,theta)
    print(a)
    cost1 = -y * np.log(a)
    cost2 = (1-y)*np.log(1-a)
    cost = cost1 - cost2
    return cost.sum()/y.size

'''
The true prediction function. .5 is the decision boundary. 
'''
def pred(y_hat):
    pred = 0
    if y_hat >= .5 : pred = 1
    return pred

'''
Logistic gradient decent. 
'''
def train(X,y):
    m,n = X.shape
    theta = np.random.uniform(1,-1,size=n)
    epsilon = .00001
    converged = False
    i = 0
    while not converged :
        if i%1000 == 0 : print("i = ", i, ", cost = ", cost(X,y,theta))
        i+=1
        X_t    = np.transpose(X)
        a      = h(X,theta)
        grad   = np.dot(X_t,a-y) * STEP_SIZE/m
        theta2 = theta - grad
        if la.norm(theta-theta2) < .00003:
            converged = True
        else:
            theta = theta2
    return theta

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 1",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
                    prog = "python LogisticRegressionClassifier.py [train-data] [train-label] [test-data] [test-labels]")
    parser.add_argument("paths", nargs = 4)
    args = vars(parser.parse_args())
    
    training_data_file = args["paths"][0]
    training_label_file = args["paths"][1]
    testing_data_file = args["paths"][2]
    testing_label_file = args["paths"][3]

    X_train = getData(training_data_file)
    X_test  = getData(testing_data_file)
    y_train = getLabels(training_label_file)
    y_test  = getLabels(testing_label_file)
    
    theta = train(X_train,y_train)

    y_preds = pred(h(X_test,theta))

    accuracy = np.sum(np.where(y_preds==y_test,1,0))/y_preds.size 
    
    print("total accuracy: ",accuracy)
