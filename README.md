# DataScience **II**

This repo is a set of programs reflecting some of the data science methods and topics I learned during the fall of 2017.

Included are:
1. a logistic regression document classifier with weights learned via gradient decent
2. a logistic regression document classifier with weights learned via a simple genetic programming algorithm
3. an implementation of the [MultiRankWalk algorithm](https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf)
4. an implementation of a stochastic Singular Value Decomposition solver.

## Logistic Regression Document Classifier (GD)

Given a set of training documents, X, and their corresponding labels, y, the parameter, &theta;=[&theta;<sub>1</sub>,&theta;<sub>2</sub>,...,&theta;<sub>n</sub>]<sup>T</sup> is learned via gradient decent. If a set of unlabeled documents is provided as a command line argument, then the predicted classes for the documents is printed. If the ground truth for the set of unlabeled documents is provided, then the accuracy of the predictions is reported as well.

`python LogisticRegressionClassifer.py --help`

OR

`python LogisticRegressionClassifer.py -h`

for more information.  
## Logistic Regression Binary Document Classifier (GP)

Given a set of training documents, X, and their corresponding labels, y, the parameter, &theta;=[&theta;<sub>1</sub>,&theta;<sub>2</sub>,...,&theta;<sub>n</sub>]<sup>T</sup> is learned via genetic programming. If a set of unlabeled documents is provided as a command line argument, then the predicted classes for the documents is printed. If the ground truth for the set of unlabeled documents is provided, then the accuracy of the predictions is reported as well.

`python GeneticOptimizer.py --help`

OR

`python GeneticOptimizer.py -h`
## MultiRankWalk
The MulRankWalk algorithm is a semi-supervised learning algorithm. Given a small set of training instances - i.e. points in n-dimensional space - the algorithm will classify a set of unknown instances in the same space. The details of the algorithm may be found [here](https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf). This particular implementation is designed to test the effect of a number of the algorithm's hyperparameters, including the size of the "seed" set (i.e. - labeled instances), the method of choosing seeds from the seed set (random or ranked according to a fitness criteria), the effect of varying the damping parameter, and the effect of varying the gamma parameter in the RBF kernel for measuring instance similarity.

`python MultiRankWalk.py --help`

OR

`python MultiRankWalk.py -h`
## StochasticSVD

This is an implementation of a stochastic singular value decomposition solver which compares its results to the results of both the deterministic and stochastic SVD solvers implemented in the scikit learn package. Interestingly, this implementation far outperforms the stochastic SVD solver implemented in the scikit learn package in every tested environment. No data need be provided to the program; the program will run on the MNIST dataset by default. The performance of the implemented SSVD solver versus the stock SSVD solver as well as the deterministic solver will be plotted to the screen. 
