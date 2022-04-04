from matplotlib.pyplot import cla
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

class Dataset(pd.DataFrame):


    def __init__(self, temporalData, window, 
            overlap = 0.6, 
            classes = None, 
            label = None,
            processors = [], 
            verbose = False):
        ''' 
            temporalData (list): list of temporal observations 

            window (int) : length of window to sample

            overlap (float) : percentage of overlap - must be between (0-1)

            classes (DataFrame) : of classes with names and values

            label (str) : name of column to use as label

            procesors (list): list of objects ( object(DataFrame) -> DataFrame) 
                to be called on the data in order (after windowing).  
                objects include kernels (objects) and processors (functions)

        '''
        # Exceptions
        if overlap < 0 or overlap > 1:
            raise ValueError( "Overlap (float) : must be between (0-1)")

        # Windowing data
        windows = []
        classes_ = []
        for obsIndex, observation in enumerate(temporalData):
            for windowStart in range(0,len(observation)-window,int((1-overlap)*window)):
                windows.append(observation[windowStart:windowStart+window])
                if classes is not None:
                    classes_.append(classes.iloc[obsIndex])

        # Applying processors 
        features = pd.DataFrame(windows)
        if classes is not None:
            classes = pd.DataFrame(classes_).reset_index()
        transformations = []
        for procesor in processors:
            features = procesor(features)
            if verbose:
                print(procesor)
            transformations.append(procesor)

        # Initialize dataframe with freatures and classes
        features.columns = pd.MultiIndex.from_product([features.columns, ['features']])
        if classes is not None:
            classes.columns = pd.MultiIndex.from_product([classes.columns, ['classes']])
            super().__init__(pd.concat((features, classes),axis=1))
        else:
            super().__init__(features)
        self.columns = self.columns.reorder_levels([1, 0])
        self.sort_index(axis=1, inplace= True)

        self.label = label

    def get_train_test(self, test_size = 0.2, random_state = 42, one_hot_encoding = False):
        '''
            Returns train and test dataframes (X_train, X_test, y_train, y_test )
        '''
        return train_test_split(self.features(), self.labels(one_hot_encoding=one_hot_encoding), test_size=test_size, random_state=random_state)

    def select(self, selection):
        '''
            Sets a new selection for features() method to use with only the selected columns
        '''
        self.selection = selection

    def labels(self, one_hot_encoding = False):
        if one_hot_encoding:
            return pd.get_dummies(self.classes[self.label])
        return self.classes[self.label]

    def features(self):
        if self.selection is None:
            return self.features
        return self.features[self.selection]

    def save(self, filename = 'dataset.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(filename = 'dataset.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)