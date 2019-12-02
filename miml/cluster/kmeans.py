
from smile.clustering import KMeans as JKMeans

import mipylib.numeric as np
from .cluster import Cluster

class KMeans(Cluster):
    '''
    K-Means clustering.
    
    The algorithm partitions n observations into k clusters in which each observation belongs to 
    the cluster with the nearest mean. Although finding an exact solution to the k-means problem 
    for arbitrary input is NP-hard, the standard approach to finding an approximate solution (often 
    called Lloyd's algorithm or the k-means algorithm) is used widely and frequently finds 
    reasonable solutions quickly.

    :param k: (*int*) Number of clusters.
    :param max_iter: (*int*) The maximum number of iterations for each running.
    :param runs: (*int*) The number of runs of K-Means algorithm.
    '''
    
    def __init__(self, k, max_iter=100, runs=1):
        super(KMeans, self).__init__()
        
        self._k = k
        self._max_iter = max_iter
        self._runs = runs
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = JKMeans(x.tojarray('double'), self._k, self._max_iter, self._runs)
        return self
    
    def fit_predict(self, x):
        """
        Fitting and cluster data.

        :param x: (*array*) Input data.
        
        :returns: (*array*) The cluster labels.
        """
        self.fit(x)
        
        r = self._model.getClusterLabel()
        return np.array(r)
        
        
##############################################