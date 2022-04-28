from sklearn.cluster import KMeans as skKMeans
from sklearn.cluster import AffinityPropagation as skAffinityPropagation


class Cluster():
    pass

class KMeans(Cluster):

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.clf = skKMeans(*args, **kwargs)

    def fit(self, X):
        if 'n_clusters' not in self.kwargs:
            self.kwargs['n_clusters'] = self.number_of_clusters(X)
            self.clf.n_clusters = self.kwargs['n_clusters']
        self.clf.fit(X)
        self.labels = self.clf.labels_

    def number_of_clusters(self, X):
        return 2
    
class AffinityPropagation(Cluster):
    pass
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.clf = skAffinityPropagation(*args, **kwargs)
    
    def fit(self, X):
        self.clf.fit(X)
        self.labels = self.clf.labels_
