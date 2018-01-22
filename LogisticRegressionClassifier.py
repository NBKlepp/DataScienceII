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
The hypothesis function (a.k.a. - the prediction)
'''
def h(x,theta) :
    z = np.dot(x,theta)
    return sigmoid(z)

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
    i=0
    while not converged :
        if i%1000 == 0 : print("i = ", i, ", cost = ", cost(X,y,theta))
        i+=1
        grad   = np.zeros(n)
        X_t    = np.transpose(X)
        a      = h(X,theta)
        grad   = np.dot(X_t,y-a) * STEP_SIZE/m
        theta2 = theta - grad
        if la.norm(theta-theta2) < .00003:
            converged = True
        else:
            theta = theta2
    return theta

'''
wordCounts = sys.argv[1]
docLabels  = sys.argv[2]
testData   = sys.argv[3]


with open(wordCounts) as f:
    wordLines = f.readlines()

with open(docLabels) as f:
    docLabs = f.readlines()

with open(testData) as f:
    testLines = f.readlines()

DEBUG =False
DEBUG2=False
DEBUG3=False
wCount = 0                #increment the counter each time a new word is found
dCount = 0                #increment the counter each time a new document is found
wIndex = {}               #a dictionary of {wid : wCount} pairs 
dIndex = {}               #a dictionary of {did : dCount} pairs
X1 = []                    #the training set of instance features, a design matrix [numDocs,numWords] dimension
y = []                    #the training set of instance labels, a vector [numDocs] dimension

for lab in docLabs:
    y.append(float(lab))
    X1.append([1.0])

if DEBUG2 : print("len(y):",len(y),"len(X):",len(X1))

#for index, wordLine in enumerate(wordLines):
for wordLine in wordLines:
    didd, wid, freqq = wordLine.split()          #each line in the training set file
    freq = float(freqq)                           #X^i_j = the value of feature x_j (i.e.-wid) in instance x^i (i.e.-did)
    did  = int(didd)
    #if DEBUG : print("did:",did)
    if wid in wIndex:                           #if this IS NOT a new word
        wix = wIndex[wid]                       #get the "j" value for this wid
        X1[did-1][wix] = freq
    else:                                       #this IS a new word
        wCount+=1                               #increment the word counter
        wix = wCount                            #get the new j-value
        wIndex[wid]=wix                         #record the j-value for this wid
        for x in X1: x.append(0)                 #create a new feature for each document
        X1[did-1][wix]=freq                      #record this new feature value
#end for


if DEBUG2:
    print("wordCount:",wCount)
    for x in X1: assert(x[0]==1)
    print("X:",X1)
    #for i,x in enumerate(X):
     #   print("X[i]: ",x)
     #   print("*"*25)

X=np.array(X1)

for i,x in enumerate(X1):
    if DEBUG2 : print("i",i)
    x.append(y[i])

if DEBUG2 : print("X1 NOW:",X1)
XX=np.array(X1)
y=np.array(y)

theta = np.array([1.0] * X[0].size)            #the parameter vector with size wCount+1



def h (x) :                                   #the hypothesis function
    if DEBUG3:
        print("h(x)")
        print("x:",x)
        print("theta:",theta)
        print("np.dot(theta,x)",np.dot(theta,x))
        print("np.exp(-(np.dot(theta,x))",np.exp(-(np.dot(theta,x))))
        print("(1+np.exp(-(np.dot(theta,x))))",1+np.exp(-(np.dot(theta,x))))
        print("1 / (1+np.exp(-(np.dot(theta,x))))",1 / (1+np.exp(-(np.dot(theta,x)))))
    return 1 / (1+np.exp(-(np.dot(theta,x))))

def f(x):
    if DEBUG2:
        print("x",x)
        print("x[-1]:",x[-1])
        print("x[:x.size-1]: ",x[:x.size-1])
    return (x[-1]-h(x[:x.size-1]))*x[:x.size-1]  

if DEBUG2:
    print("theta:")
    print(theta)
    print("X:")
    print(X)
    print("y:")
    print(y)
    
#train the model
converged = False
alpha = .0001
n = 0
while not converged:
#while n < 100000:
    theta2=theta+alpha*sum(np.apply_along_axis(f,axis=1,arr=XX))
    if la.norm(theta-theta2) < .00003:
            converged = True
    else:
            theta = theta2
    n+=1        
    #theta=theta2
    if DEBUG2 : print(theta)
    
    for i,x in enumerate(X):
        if DEBUG2: print("i:",i) 
        for j,t in enumerate(theta):
            if DEBUG2: print("j:",j)
            if False:
                print("j:",j)
                print("theta[j]:",theta[j])
                print("y[i]:",y[i])
                print("X[i]:")
                print(X[i])
                print("X[i][j]:",X[i][j])
            theta2[j]=theta[j]+alpha*(y[i]-h(X[i]))*X[i][j]  #stochastic update rule
            if DEBUG2: print("theta2[j]:",theta2[j],"| theta[j]:",theta[j])
            if DEBUG2: print(theta)
        theta = theta2
        
        
        if converged : break
        
    #n+=1
#end train

if DEBUG : print("Convergence")
n = theta.size

xOut = []
outDocIndex = {}
outDocId = {}
m = 0
max = 0
#collect the test statistics
for testLine in testLines:
    did, wid, freqq = testLine.split()          #each line in the training set file
    freq = float(freqq)
    dix = 0
    if did in outDocIndex :                     #if this is NOT a new document
        dix = outDocIndex[did]       
    else:                                       #if this IS a new document
        outDocIndex[did]=m
        outDocId[m]=did
        dix = outDocIndex[did]
        xOut.append([0.0]*n)
        xOut[m][0]=1.0
        m+=1
        if int(did) > max :
            max = int(did)
    if wid in wIndex:                       #if this is a word in the corpus vocabulary
        wix = wIndex[wid]
        xOut[dix][wix]=freq
#end for
DEBUG4 = False
y=[-1]*max                                     #the output vector
if DEBUG4 :
    print("y:",y)
    print("m:",m)
    print("max:",max)
DEBUG3=False
for i,row in enumerate(xOut):
    x = np.array(row)
    did = int(outDocId[i])-1                  #i'th row of xOut stores data about did outDocId[i]
    y[did] = 1 if h(x) > 1/2 else 0
for x in y:
    print(x)
  
    x=np.array([0]*n)
    x[0]=1.0
    words = doc.split()
    for word in words:
        if word in wIndex: #wIndex.has_key(word):

            x[wIndex[word]]+=1                  #words were indexed starting at 0, but x_0=1 for all x, meaning x_1 is first word in any document...
    if h(x)>1/2:
        print(1)
    else:
        print(0)
'''    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Assignment 1",
        epilog = "CSCI 4360/6360 Data Science II: Fall 2017",
        add_help = "How to use",
                    prog = "python assignment1.py [train-data] [train-label] [test-data] [test-labels]")
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

    print("X_train: ",X_train)
    print("X_test: ",X_test)
    print("y_train: ",y_train)
    print("y_test: ",y_test)
    
    theta = train(X_train,y_train)

    y_preds = pred(h(X_test,theta))

    accuracy = np.sum(np.where(y_preds==y_test,1,0))/y_preds.size 
    
    print("total accuracy: ",accuracy)
