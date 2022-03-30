from ast import Return
import pandas as pd
import numpy as np
import pywt as wt
import scipy.stats as stats
from tqdm import tqdm


class Processor:
    def __init__():
        pass

class DWT():
    daubechies = ['db1','db2','db3','db4','db5','db6','db7','db8','db9','db10']

    features = {
        'mean' : np.mean,
        'var' :  np.var,
        'skew': stats.skew,
        'kurt': stats.kurtosis,
        'rms' : lambda x: np.sqrt(np.mean(x**2))
    }

    def __init__(self, wavelets, levels, featureFunctions):
        '''
            wavelets (list) : List of wavelet names. Example: ['db1','db2']
            levels (int) : Number of levels to decompose 
            features (dict) : Dictionary of {name: function} to be applied to DWT coefficient sets
        '''
        self.wavelets = wavelets
        self.levels = levels
        self.featureFunctions = featureFunctions
        self.featureNames = []
        for wavelet in wavelets:
            for level in range(levels):
                for featureFunction in featureFunctions:
                    self.featureNames.append(f'{wavelet}-{level}-cA-{featureFunction}')
                    self.featureNames.append(f'{wavelet}-{level}-cD-{featureFunction}')


    def __call__(self, data):
        ''' 
            data : array of shape (Number Of Observations, Segment Length)
        '''
        features = []
        #Iterating Over Observations
        for observation in tqdm(data.values):
            observationFeatures = []
            for wavelet in self.wavelets:
                cA = observation
                for level in range(self.levels):
                    for featureFunctionName,featureFunction in self.featureFunctions.items():
                        #Wavelet decomposition for current level
                        (cA, cD) = wt.dwt(cA, wavelet)
                        #Appending Features to Dictionary
                        observationFeatures.append(featureFunction(cA))
                        observationFeatures.append(featureFunction(cD))
            features.append(observationFeatures)            
        return pd.DataFrame(features, columns=self.featureNames)
    
    def __str__(self):
        out = f'''
            Wavelet Decomposition
            Wavelets: {self.wavelets}
            Max Level: {self.levels}
            Functions: {list(self.featureFunctions.keys())}
        ''' 
        return out

class Normalize:
    def __call__(self, features):
        self.min = features.min(axis=0)
        self.max = features.max(axis=0)
        return (features-self.min)/(self.max-self.min)

    def __str__(self):
        return f'Features Normalized'

class Dropna:
    def __call__(self, features):
        self.isna = np.any(features.isna(),axis=0)
        return features[self.isna[~self.isna].index]
    def __str__(self):
        return f'Nan dropped features: {list(self.isna[self.isna].index)}'
