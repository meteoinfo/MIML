
from smile.clustering import KMeans as JKMeans
from smile.clustering import PartitionClustering
from java.util.function import Supplier

import mipylib.numeric as np
from .cluster import Cluster

class supF(Supplier):
    def __init__(self, x, k, max_iter, tol):
        self.x = x
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def get(self):
        return JKMeans.fit(self.x, self.k, self.max_iter, self.tol)

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
    :param tol: (*float*) the tolerance of convergence test.
    :param runs: (*int*) The number of runs of K-Means algorithm.
    '''
    
    def __init__(self, k, max_iter=100, tol=1e-4, runs=10):
        super(KMeans, self).__init__()
        
        self._k = k
        self._max_iter = max_iter
        self._tol = tol
        self._runs = runs
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = PartitionClustering.run(self._runs, supF(x.tojarray('double'), self._k, self._max_iter, self._tol))
        #self._model = JKMeans.fit(x.tojarray('double'), self._k, self._max_iter, self._tol)
        return self
    
    def fit_predict(self, x):
        """
        Fitting and cluster data.

        :param x: (*array*) Input data.
        
        :returns: (*array*) The cluster labels.
        """
        self.fit(x)
        return np.array(self._model.y)
        
        
##############################################