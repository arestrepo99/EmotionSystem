from keyword import kwlist
from unittest.mock import NonCallableMagicMock
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.tree import DecisionTreeClassifier
from package.Data import Dataset
import pandas as pd
from scipy import stats

def early_stopping(n, verbose = True):
    def early_stopping(*args, **kwargs):
        # Getting the list of losses
        loss = kwargs['loss']
        # Stopping if loss does not improve for n iterations
        if len(loss) > n and loss[-n] > loss[-n-1]:
            if verbose:
                print(f'Early stopping, iteration {len(loss)}, loss did not improve for {n} iterations.')
            return 'early_stopping'
    return early_stopping

class plot_loss:
    def __init__(self, figsize = (8,5)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.suptitle('Forward Feature Selection Progress')

    def __call__(self, *args, **kwargs):
        loss = kwargs['loss']
        self.ax.clear()
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.plot(loss)
        self.fig.show()
        plt.pause(0.1)

class ForwardFeatureSelection():
    def __init__(self, model= DecisionTreeClassifier(), callbacks = None, maxIter = 100):
        '''
            model : object with scikit learn API (fit and score) - default DecisionTree
            callbacks : [callables]] (optional) - List of callables to be called after each iteration.
            maxIter : int (optional) - maximum number of iterations
        '''
        self.model = model
        if callbacks is None:
            callbacks = [early_stopping(5), plot_loss()]
        self.callbacks = callbacks
        self.maxIter = maxIter
    
    def fit(self, dataset):
        X_train, X_test, y_train, y_test = dataset.split()

        self.selection = []
        self.loss = []
        for i in range(self.maxIter):
            feature_loss = {}
            for feature in X_train.columns:
                self.model.fit(X_train[self.selection+[feature]], y_train)
                feature_loss[feature] = -self.model.score(X_test[self.selection+[feature]], y_test)
            bestFeature, minLoss = min(feature_loss.items(), key = lambda x: x[1]) 
            self.selection.append(bestFeature)
            self.loss.append(minLoss)
            for callback in self.callbacks:
                output = callback(selection = self.selection, loss = self.loss)
                if output == 'early_stopping':
                    self.selection = self.selection[:np.argmax(self.loss)]
                    return Dataset(dataset.X[self.selection], dataset.y, dataset.classes)
        return  Dataset(dataset.X[self.selection], dataset.y, dataset.classes)
    
    def transform(self, dataset):
        return Dataset(dataset.X[self.selection], dataset.y, dataset.classes)



class tStudent():
    def __init__(self, threshold = 0.9):
        self.threshold = threshold
        
    def fit(self, dataset):
        labels = dataset.y.iloc[:,0].unique()
        self.biClassTest = pd.DataFrame(columns = dataset.X.columns)
        for label1 in labels:
            for label2 in labels:
                if label1 != label2:
                    testResultsPerFeature = []
                    for feature in dataset.X.columns:
                        testResultsPerFeature.append(
                            stats.ttest_ind(
                                dataset.X[feature][dataset.y.iloc[:,0] == label1],
                                dataset.X[feature][dataset.y.iloc[:,0] == label2],
                                equal_var = False
                                )[1] < 0.05)
                    self.biClassTest.loc[str(label1)+'/'+str(label2)] = testResultsPerFeature
        
        mean = self.biClassTest.mean(axis=0)
        self.selection = list(mean.loc[mean >= self.threshold].index)
        return Dataset(dataset.X[self.selection], dataset.y, dataset.classes)

    def transform(self, dataset, threshold= None):
        if threshold is None:
            threshold = self.threshold
        mean = self.biClassTest.mean(axis=0)
        self.selection = list(mean.loc[mean >= threshold].index)
        return Dataset(dataset.X[self.selection], dataset.y, dataset.classes)