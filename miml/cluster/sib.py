
from smile.clustering import SIB as JSIB

import mipylib.numeric as np
from .cluster import Cluster

class SIB(Cluster):
    '''
    Sequential Information Bottleneck clustering.
    
    SIB clusters co-occurrence data such as text documents vs words. SIB is guaranteed to converge to 
    a local maximum of the information. Moreover, the time and space complexity are significantly 
    improved in contrast to the agglomerative IB algorithm.
    
    :param k: (*int*) Number of clusters.
    :param max_iter: (*int*) The maximum number of iterations for each running.
    :param runs: (*int*) The number of runs of K-Means algorithm.
    '''
    
    def __init__(self, k, max_iter=100, runs=1):
        super(SIB, self).__init__()
        
        self._k = k
        self._max_iter = max_iter
        self._runs = runs
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = JSIB(x.tojarray('double'), self._k, self._max_iter, self._runs)
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