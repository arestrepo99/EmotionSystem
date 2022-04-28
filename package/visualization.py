from package.DimensionalityReduction import UMAP
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

class Visualization():
    def __init__(self, reductionAlgoritm = UMAP(n_components=2)):
        self.reductionAlgoritm = reductionAlgoritm

    def fit(self, dataset):
        self.reductionAlgoritm.fit(dataset)

    def transform(self, dataset):
        return self.reductionAlgoritm.transform(dataset)

    def plot(self, dataset):
        embedding = self.transform(dataset)
        df = pd.DataFrame(embedding)
        # combining dataset.classes and df
        df = pd.concat([dataset.classes, df], axis=1)
        
        # Computing optimal shape of subplot 
        num_plots = len(dataset.classes.columns)
        minProd = num_plots**2
        dims = (0,0)
        for i in range(num_plots):
            for j in range(num_plots):
                if i*j>=num_plots:
                    if i*j<minProd:
                        minProd = i*j
                        dims = (i,j)
        
        # Plotting
        fig, axes = plt.subplots(*dims, sharex=True, figsize=(16,8))
        fig.suptitle('Visualization of embedded dataset')
        for ind, col in enumerate(dataset.classes.columns):
            axes[ind//dims[1], ind%dims[1]].set_title(col)
            sns.scatterplot(ax= axes[ind//dims[1], ind%dims[1]] , x=0, y=1, hue=col, data=df)
    