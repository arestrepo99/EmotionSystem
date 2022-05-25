from sklearn.model_selection import train_test_split
from scipy.io import wavfile
import pandas as pd
import os

class TemporalData():
    def __init__(self, data, samplerate):
        ''' 
            samplerate (int): sample rate of the data
            data: numpy array of shape (n_samples)
            classes: dict {class_name: class_id}
        '''
        self.samplerate = samplerate
        self.data = data
    
    def window(self, window, overlap = 0.6):
        '''
            window: window size in samples
            overlap: overlap in %
        '''
        window_size = int(window)
        overlap_size = int(overlap*window_size)
        n_windows = (len(self.data) - window_size)//overlap_size +1
        windows = []
        for i in range(n_windows):
            windows.append(self.data[i*overlap_size:i*overlap_size+window_size])
        return windows, n_windows

class Audio(TemporalData):
    def __init__(self, filename):
        samplerate, data = \
                wavfile.read(filename)
        super().__init__(data, samplerate)


class Dataset():
    def __init__(self, X, y= None, classes = None):
        ''' )
            X = dataframe of shape (n_samples, n_features)
            y = dataframe of shape(1,n_features)
            classes = dataframe of shape (n_classes, n_features) 
                (for visualization and comparison with other clases)

        '''
        self.X = pd.DataFrame(X)
        self.y = pd.DataFrame(y)
        self.classes = pd.DataFrame(classes)

    def split(self, test_size = 0.2):
        '''
            Returns train and test dataframes (X_train, X_test, y_train, y_test)
        '''
        return train_test_split(self.X, self.y, test_size=test_size)
    
    def save(self, name='data'):
        path = '.dataset/'
        # Check if path exists if not make directory
        if not os.path.exists(path):
            os.makedirs(path)
        # Save dataframes
        path = '.dataset/'+name+'/'
        # Check if path exists if not make directory
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.X.to_csv(path+'X.csv')
        if self.X is not None:
            self.y.to_csv(path+'y.csv')
        if self.classes is not None:
            self.classes.to_csv(path+'classes.csv')

    def load(name='data'):
        path = '.dataset/'+name+'/'
        X = pd.read_csv(path+'X.csv', index_col=0)
        if os.path.exists(path+'y.csv'):
            y = pd.read_csv(path+'y.csv', index_col=0)
        if os.path.exists(path+'classes.csv'):
            classes = pd.read_csv(path+'classes.csv', index_col=0)
        return Dataset(X, y, classes)
    
    def __str__(self):
        out = f'''
            Dataset
            X: {self.X.shape}
            y: {self.y.shape}
            classes: {self.classes.shape}
        ''' 
        return out
    
    def __repr__(self):
        return self.__str__()
        

# class Dataset(pd.DataFrame):
#     def __init__(self, data, window, 
#             overlap = 0.6, 
#             classes = None, 
#             label = None,
#             processors = [], 
#             verbose = False):
#         ''' 
#             classes (DataFrame) : of classes with names and values

#             label (str) : name of column of classes to use as label, if it is not set dataset
#                 will be will be considered unlabeled

#             procesors (list): list of objects ( object(DataFrame) -> DataFrame) 
#                 to be called on the data in order (after windowing).  
#                 objects include kernels (objects) and processors (functions)

#         '''

#         # Exceptions
#         if overlap < 0 or overlap > 1:
#             raise ValueError( "Overlap (float) : must be between (0-1)")

#         # Windowing data
#         windows = []
#         classes_ = []
#         for obsIndex, observation in enumerate(temporalData):
#             for windowStart in range(0,len(observation)-window,int((1-overlap)*window)):
#                 windows.append(observation[windowStart:windowStart+window])
#                 if classes is not None:
#                     classes_.append(classes.iloc[obsIndex])

#         # Applying processors 
#         features = pd.DataFrame(windows)
#         if classes is not None:
#             classes = pd.DataFrame(classes_).reset_index()
#         transformations = []
#         for procesor in processors:
#             features = procesor(features)
#             if verbose:
#                 print(procesor)
#             transformations.append(procesor)


#         # Initialize dataframe with freatures and classes
#         features.columns = pd.MultiIndex.from_product([features.columns, ['features']])
#         if classes is not None:
#             classes.columns = pd.MultiIndex.from_product([classes.columns, ['classes']])
#             super().__init__(pd.concat((features, classes),axis=1))
#         else:
#             super().__init__(features)
#         self.columns = self.columns.reorder_levels([1, 0])
#         self.sort_index(axis=1, inplace= True)
#         self.index.set_names('observation', inplace=True)

#         # Initializing Variables
#         if 'label' in self.__dict__:
#             raise Exception("overriding 'label' varaible in super class")
#         self.label = label

        

#     def get_train_test(self, test_size = 0.2, random_state = None, one_hot_encoding = False):
#         '''
#             Returns train and test dataframes (X_train, X_test, y_train, y_test )
#         '''
#         return train_test_split(
#             self.get_features(), 
#             self.get_labels(one_hot_encoding=one_hot_encoding), 
#             test_size=test_size, 
#             random_state=random_state)

#     def select(self, selection):
#         '''
#             Sets a new selection for features() method to use with only the selected columns
#         '''
#         self.selection = selection

#     def get_labels(self, one_hot_encoding = False):
#         if one_hot_encoding:
#             return pd.get_dummies(self.classes[self.label])
#         return self.classes[self.label]

#     def get_features(self):
#         if 'selection' not in self.__dict__:
#             return self.features
#         return self.features[self.selection]

#     def save(self, filename = 'dataset.pkl'):
#         with open(filename, 'wb') as f:
#             pickle.dump(self, f)
    
#     def load(filename = 'dataset.pkl'):
#         with open(filename, 'rb') as f:
#             return pickle.load(f)