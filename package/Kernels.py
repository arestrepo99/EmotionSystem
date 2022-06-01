from ast import Return
import pandas as pd
import numpy as np
import pywt as wt
import scipy.stats as stats
from tqdm import tqdm
from package.Data import Dataset


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

    def __init__(self, wavelets =  None, levels =  None, featureFunctions = None):
        '''
            wavelets (list) : List of wavelet names. Example: ['db1','db2']
            levels (int) : Number of levels to decompose 
            featureFunctions (dict) : Dictionary of {name: function} to be applied to DWT coefficient sets
        '''
        if wavelets is None:
            self.wavelets = DWT.daubechies
        else:
            self.wavelets = wavelets
            
        if featureFunctions is None:
            self.featureFunctions = DWT.features
        else:
            self.featureFunctions = featureFunctions
        
        self.levels = levels

    def featureNames(self):
        featureNames = []
        for wavelet in self.wavelets:
            for level in range(self.levels):
                for featureFunction in self.featureFunctions:
                    featureNames.append(f'{wavelet}-{level}-cA-{featureFunction}')
                    featureNames.append(f'{wavelet}-{level}-cD-{featureFunction}')
        return featureNames

    def decompose(self, observation):
        assert self.levels is not None, 'number of levels to decompose not set'
        observationFeatures = []
        for wavelet in self.wavelets:
            cA = observation
            for level in range(self.levels):
                for _,featureFunction in self.featureFunctions.items():
                    #Wavelet decomposition for current level
                    (cA, cD) = wt.dwt(cA, wavelet)
                    #Appending Features to Dictionary
                    observationFeatures.append(featureFunction(cA))
                    observationFeatures.append(featureFunction(cD))
        return observationFeatures

    def fit(self, dataset):
        '''
            dataset (Dataset) : Dataset to be processed
        '''
        data = dataset.X

        n,m = data.shape
        if self.levels is None:
            # Setting levels to maximum level posible
            self.levels = int(np.log2(m))
        
        features = []
        #Iterating Over Observations
        # Can be paralelize for improved speed in future
        for observation in tqdm(data.values):
            features.append(self.decompose(observation))        

        return Dataset(pd.DataFrame(features, columns=self.featureNames()), y= dataset.y, classes = dataset.classes)
    
    def transform(self, dataset):
        return self.fit(dataset)

    def __str__(self):
        out = f'''
            Wavelet Decomposition
            Wavelets: {self.wavelets}
            Max Level: {self.levels}
            Functions: {list(self.featureFunctions.keys())}
        ''' 
        return out

class Normalize:
    def fit(self, dataset):
        self.min = dataset.X.min(axis=0)
        self.max = dataset.X.max(axis=0)

        return Dataset((dataset.X-self.min)/(self.max-self.min), y=dataset.y, classes = dataset.classes)

    def transform(self, dataset):
        return Dataset((dataset.X-self.min)/(self.max-self.min), y=dataset.y, classes = dataset.classes)
    def __str__(self):
        return f'Features Normalized'

class Dropna:
    def fit(self, dataset):
        self.isna = np.any(dataset.X.isna(),axis=0)
        return Dataset(dataset.X[self.isna[~self.isna].index], y=dataset.y, classes = dataset.classes)

    def transform(self, dataset):
        return Dataset(dataset.X[self.isna[~self.isna].index], y=dataset.y, classes = dataset.classes)

    def __str__(self):
        return f'Nan dropped features: {list(self.isna[self.isna].index)}'


from r_functions import create
import pandas as pd
analysisECG = create('package/r/analysisECG.R', 'analysisECG')

class AnalysisECG():
    def __init__(self, 
            fs = 2000,
            t= None, 
            gr_r = 0.8,
            gr2 = 200,
            gr3 = 640,
            gr4 = 100,
            gr5 = 400,
            gr6 = 160,
            gr7 = 130,
            gr8 = 200,
            gr9 = 200,
            gr10 = 80,
        ):
        self.fs = fs
        self.t = t
        self.gr_r = gr_r
        self.gr2 = gr2 # max number of samples for RS distance
        self.gr3 = gr3 # max number of samples for ST distance = 0.32[s]*fs
        self.gr4 = gr4 # max QR distance
        self.gr5 = gr5 # max PQ distance
        self.gr6 = gr6 # max PP1 distance (P1 - beginning of P wave)
        self.gr7 = gr7 # max PP2 distance (P2 - end of P wave)
        self.gr8 = gr8 # max TT1 distance (T1 - beginning of T wave)
        self.gr9 = gr9 # max of TT2 distance (T2 - end of T wave)
        self.gr10 = gr10 # max of SS2 distance (S2 - end of QRS complex)
        


    def fit(self, dataset):
        if self.t is None:
            self.t = list(np.arange(0, dataset.X.shape[1]/self.fs, 1/self.fs))
        features = [] 
        index = []
        for obs in tqdm(dataset.X.index):
            try:
                out = analysisECG(
                    list(dataset.X.loc[obs]), 
                    self.fs,
                    self.t,
                    self.gr_r,
                    self.gr2,
                    self.gr3,
                    self.gr4,
                    self.gr5,
                    self.gr6,
                    self.gr7,
                    self.gr8,
                    self.gr9,
                    self.gr10
                )
                features.append([feature['vrednost'] for feature in out])
                index.append(obs)
            except:
                print(f'Warning: observation {obs} failed')
        X = pd.DataFrame(features, columns = [feature['obelezje'] for feature in out], index = index)
        return Dataset(X, dataset.y, dataset.classes)
