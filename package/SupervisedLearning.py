from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np


class SupervisedAlgorithm:
    def fit(self, X, y):
        m,n = X.shape
        if self.pac(n)>m:
            raise ValueError("Not enough observations to fit model")
        self.clf = self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def score(self, X, y):
        return self.clf.score(X, y)


class DecisionTree(SupervisedAlgorithm):
    epsilon = 0.1
    delta = 0.1

    def __init__(self):
        self.clf = DecisionTreeClassifier()

    def pac(self, n):
        '''
            k: depth of tree
            n: number of features
            
        '''
        epsilon = self.__class__.epsilon
        delta = self.__class__.delta

        k = self.clf.max_depth
        # Returns minimum number of observations
        return np.log(2)/(2*epsilon**2)*\
            ( (2^k-1)* (1+np.log2(n)) + 1 + np.log(delta**-1) )

