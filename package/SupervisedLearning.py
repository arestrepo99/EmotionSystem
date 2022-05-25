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

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

class ModelCompare():
    def __init__(self, models):
        self.models = models

    def plotROC(self, X_test, y_test):
        # plotting roc curves
        plt.figure(figsize=(10, 10))
        for ind, model in enumerate(self.models):
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
            plt.plot(fpr, tpr, label='model ' + str(ind))

class DecisionTree():
    epsilon = 0.1
    delta = 0.1

    def __init__(self, *args,  **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.clf = DecisionTreeClassifier(*args, **kwargs)

    def pac(self, n):
        '''
            k: depth of tree
            n: number of features
        '''
        epsilon = self.__class__.epsilon
        delta = self.__class__.delta

        if 'max_depth' in self.kwargs:
            k = self.kwargs['max_depth']
            # Returns minimum number of observations
            return np.log(2)/(2*epsilon**2)*\
                ( (2^k-1)* (1+np.log2(n)) + 1 + np.log(delta**-1) )
        else:
            return 0
        