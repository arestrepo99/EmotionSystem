from keyword import kwlist
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from package.SupervisedLearning import DecisionTree
from package.data import Dataset
   
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
        self.ax.set_xlabel('Iteration (And number of features added)')
        self.ax.set_ylabel('Loss')
        self.ax.plot(loss)
        self.fig.show()
        plt.pause(0.1)

class ForwardFeatureSelection():
    
    def __init__(self, model= DecisionTree(), callbacks = [early_stopping(5), plot_loss()], maxIter = 100):
        '''
            model : object with scikit learn API (fit and score) - default DecisionTree
            callbacks : [callables]] (optional) - List of callables to be called after each iteration.
            maxIter : int (optional) - maximum number of iterations
        '''
        self.model = model
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
                    return Dataset(dataset.X[self.selection], dataset.y, dataset.classes)
        return  Dataset(dataset.X[self.selection], dataset.y, dataset.classes)
    
    def transform(self, dataset):
        return Dataset(dataset.X[self.selection], dataset.y, dataset.classes)







