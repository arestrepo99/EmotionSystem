from package.Data import Dataset
import pandas as pd
from tqdm import tqdm

class Cluster():
    def __init__(self, algorithms):
        self.algorithms = algorithms
    
    def fit(self, dataset):
        classes = {}
        for algorithm in tqdm(self.algorithms):
            res = algorithm.fit(dataset.X)
            # Get algorithm name
            classes[algorithm.__class__.__name__] = res.labels_
        return Dataset(dataset.X, dataset.y, pd.DataFrame(classes))