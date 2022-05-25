import seaborn as sns
import pandas as pd
from umap import UMAP
import numpy as np
import matplotlib.pyplot as plt
from babyplots import Babyplot

class Visualization():
    def __init__(self, reducer = UMAP(n_components=2)):
        assert reducer.n_components == 2, 'reducer must have 2 components'
        self.reducer = reducer

    def plot(self, dataset, classes = None):
        '''
            plot(dataset, classes = None) -> plots a scatter plot of the data with classes as colors
            dataset: Dataset object
            classes: dataframe of classes for colors
        '''
        if classes is None:
            classes = dataset.classes

        embedding = self.reducer.fit_transform(dataset.X)

        df = pd.DataFrame(embedding)
        # combining classes and df
        df = pd.concat([classes, df], axis=1)
        
        # Computing optimal shape of subplot 
        num_plots = len(classes.columns)
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
        for ind, col in enumerate(classes.columns):
            axes[ind//dims[1], ind%dims[1]].set_title(col)
            sns.scatterplot(ax= axes[ind//dims[1], ind%dims[1]] , x=0, y=1, hue=col, data=df)
    

class Visualization3D():
    def __init__(self, reducer = UMAP(n_components=3)):
        assert reducer.n_components == 3, 'reducer must have 3 components'
        self.reducer = reducer


    def plot(self, dataset, classes):
        '''
            plot(dataset, classes = None) -> plots a scatter plot of the data with classes as colors
            dataset: Dataset object
            classes: dataframe of classes for colors
        '''
        if classes is None:
            classes = dataset.y
            
        embedding = self.reducer.fit_transform(dataset.X)

        # create the babyplots visualization
        bp = Babyplot()
        bp.add_plot(embedding.tolist(), "pointCloud", "categories", np.array(classes).squeeze().tolist(), 
            {
                "size": 3,
                'shape': 'circle',
                "colorScale": "Set2",
                "showAxes": [True, True, True],
                "axisLabels": ["E1", "E2", "E3"],
                'showLegend': True,
            })
        # show the visualization
        return bp