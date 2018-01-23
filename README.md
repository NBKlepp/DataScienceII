# DataScience **II**

This repo is a set of programs reflecting some of the data science methods and topics I learned during the fall of 2017.

Included are:
1. a logistic regression document classifier with weights learned via gradient decent
2. a logistic regression document classifier with weights learned via a simple genetic programming algorithm
3. an implementation of the [MultiRankWalk algorithm](https://lti.cs.cmu.edu/sites/default/files/research/reports/2009/cmulti09017.pdf)
4. an implementation of a stochastic Singular Value Decomposition solver.

## Logistic Regression Document Classifier (GD)

Given a set of training documents, X, and their corresponding labels, y, the parameter, &theta;=[&theta;<sub>1</sub>,&theta;<sub>2</sub>,...,&theta;<sub>n</sub>]<sup>T</sup> is learned via gradient decent. The predicted classes for a set of unlabeled documents is generated according to the learned model.

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
## StochasticSVD
