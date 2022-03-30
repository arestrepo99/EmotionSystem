from matplotlib.pyplot import cla
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

class Dataset(pd.DataFrame):
    def __init__(self, temporalData, window, 
            overlap = 0.6, 
            classes = None, 
            processors = [], 
            verbose = False):
        ''' 
            temporalData (list): list of temporal observations 
            window (int) : length of window to sample
            overlap (float) : percentage of overlap - must be between (0-1)
            classes (DataFrame) : of classes with names and values
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
            labels = pd.DataFrame(classes_).reset_index()
        transformations = []
        for procesor in processors:
            features = procesor(features, verbose)
            if verbose:
                print(procesor)
            transformations.append(procesor)

        # Initialize dataframe with freatures and labels
        features.columns = pd.MultiIndex.from_product([features.columns, ['features']])
        if classes is not None:
            labels.columns = pd.MultiIndex.from_product([labels.columns, ['labels']])
            super().__init__(pd.concat((features, labels),axis=1))

        else:
            super().__init__(features)
        self.columns = self.columns.reorder_levels([1, 0])
        self.sort_index(axis=1, inplace= True)

    def get_train_test(self, test_size = 0.2, random_state = 42):
        '''
            Returns train and test dataframes (X_train, X_test, y_train, y_test )
        '''
        return train_test_split(self.features, self.labels['emotion'], test_size=test_size, random_state=random_state)

    def save(self, filename = 'dataset.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(filename = 'dataset.pkl'):
        with open(filename, 'rb') as f:
            return pickle.load(f)