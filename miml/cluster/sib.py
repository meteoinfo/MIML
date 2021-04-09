
from smile.clustering import SIB as JSIB
from smile.clustering import PartitionClustering
from java.util.function import Supplier

import mipylib.numeric as np
from .cluster import Cluster

class supF(Supplier):
    def __init__(self, x, k, max_iter):
        self.x = x
        self.k = k
        self.max_iter = max_iter

    def get(self):
        return JSIB.fit(self.x, self.k, self.max_iter)

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
    
    def __init__(self, k, max_iter=100, runs=8):
        super(SIB, self).__init__()
        
        self.k = k
        self.max_iter = max_iter
        self.runs = runs
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = PartitionClustering.run(self.runs, supF(x.tojarray('double'), self.k,
            self.max_iter))
        self.labels_ = np.array(self._model.y)
        return self
        
        
##############################################