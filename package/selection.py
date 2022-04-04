from keyword import kwlist
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval


def forwardFeatureSelection(function, features, callbacks = [], iters = 100):
    '''
    Parameters
    ----------
        func : callable 
            Function to minimize. Should take a single list of features
            and return the objective value.
        features: list
            List of features to be selected from.
        callbacks : [callables]] (optional)
            List of callables to be called after each iteration.
    '''
    selection = []
    loss = []
    for i in range(iters):
        feature_loss = {}
        for feature in features:
            feature_loss[feature] = function(selection+[feature])
        bestFeature, minLoss = min(feature_loss.items(), key = lambda x: x[1]) 
        selection.append(bestFeature)
        loss.append(minLoss)
        for callback in callbacks:
            output = callback(selection = selection, loss = loss)
            if output == 'early_stopping':
                return {'selection': selection, 'value': loss[-1]}
    
    return {'selection': selection, 'value': loss[-1], 'loss': loss}


class plot_loss:
    def __init__(self, figsize = (10,5)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.suptitle('Loss')

    def __call__(self, *args, **kwargs):
        loss = kwargs['loss']
        self.ax.clear()
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Loss')
        self.ax.plot(loss)
        self.fig.show()
        plt.pause(2)
        
        
def early_stopping(n, verbose = False):
    def early_stopping(*args, **kwargs):
        # Getting the list of losses
        loss = kwargs['loss']
        # Stopping if loss does not improve for n iterations
        if len(loss) > n and loss[-n] > loss[-n-1]:
            if verbose:
                print(f'Early stopping, iteration {len(loss)}, loss did not improve for {n} iterations.')
            return 'early_stopping'
    return early_stopping



