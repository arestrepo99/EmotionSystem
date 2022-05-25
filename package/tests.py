from package.Data import Dataset
from package.Clustering import KMeans, AffinityPropagation
from package.DimensionalityReduction import UMAP
from package.FeatureSelection import ForwardFeatureSelection
from package.Kernels import DWT, Normalize, AnalysisECG, Dropna

import pandas as pd
import numpy as np
# Create a dataset

X = pd.DataFrame(np.random.rand(100,10), columns=['a','b','c','d','e','f','g','h','i','j'])
y = [['on','off'][i] for i in (2*np.random.rand(100)).astype(int)]
classes = (3*np.random.rand(100,10)).astype(int)
datasetLabelClasses = Dataset(X, y=y, classes=classes)
datasetLabel = Dataset(X, y=y)
dataset = Dataset(X)

def test(dataset):
    out = DWT().fit(dataset)
    assert type(out) == Dataset

    out = Normalize().fit(dataset)
    assert type(out) == Dataset

    out = AnalysisECG().fit(dataset)
    assert type(out) == Dataset

    out = Dropna().fit(dataset)
    assert type(out) == Dataset

    out = ForwardFeatureSelection().fit(dataset)
    assert type(out) == Dataset

    out = UMAP().fit(dataset)
    assert type(out) == Dataset

    out = KMeans().fit(dataset)
    assert type(out) == Dataset

    out = AffinityPropagation().fit(dataset)
    assert type(out) == Dataset



