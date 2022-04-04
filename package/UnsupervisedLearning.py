from sklearn import manifold
import umap
from Data import Dataset

class Cluster():
    def __init__(self, algorithm, dataset):
        self.algorithm = algorithm
        self.dataset = dataset



class UMAP():
    parameters = {
        'min_dist': 0.8,
        'n_components': 2,
        'n_neighbors': 30,
        'a': None, 
        'angular_rp_forest': False, 
        'b': None,
        'force_approximation_algorithm': False, 
        'init': 'spectral', 
        'learning_rate': 1.0,
        'local_connectivity': 1.0,
        'low_memory': False,
        'metric': 'euclidean',
        'metric_kwds': None, 
        'n_epochs': None,
        'negative_sample_rate': 5,
        'output_metric': 'euclidean',
        'output_metric_kwds': None,
        'random_state': 42, 
        'repulsion_strength': 1.0,
        'set_op_mix_ratio': 1.0,
        'spread': 1.0,
        'target_metric': 'categorical',
        'target_metric_kwds': None,
        'target_n_neighbors': -1, 
        'target_weight': 0.5,
        'transform_queue_size': 4.0,
        'transform_seed': 42, 
        'unique': False, 
        'verbose': False
    }

    def __init__(self, **kwargs):
        '''
            Initializes reducer with class parameters, and overwrites with kwargs
            The reducer can be fit with a dataset, either supervised or unsupervised
            This trained reducer can be then used to transform new data
        '''
        self.reducer = umap.UMAP(**(UMAP.parameters.update(kwargs)))
    
    def fit(self, dataset, supervised = False):
        if supervised:
            self.embedding = self.reducer.fit_transform(dataset.features, dataset.labels)
        else:
            self.embedding = self.reducer.fit_transform(dataset.features)
        
    def transform(self, dataset):
        return self.reducer.transform(dataset.features)