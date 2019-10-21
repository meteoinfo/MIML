
from smile.clustering import SIB as JSIB

import mipylib.numeric as np
from .cluster import Cluster

class SIB(Cluster):
    '''
    Sequential Information Bottleneck clustering.
    
    SIB clusters co-occurrence data such as text documents vs words. SIB is guaranteed to converge to 
    a local maximum of the information. Moreover, the time and space complexity are significantly 
    improved in contrast to the agglomerative IB algorithm.
    
    :param data: (*array*) The data set.
    :param k: (*int*) Number of clusters.
    :param max_iter: (*int*) The maximum number of iterations for each running.
    :param runs: (*int*) The number of runs of K-Means algorithm.
    '''
    
    def __init__(self, data, k, max_iter=100, runs=1):
        self._data = data
        self._k = k
        self._max_iter = max_iter
        self._runs = runs
        self._model = JSIB(self._data.tojarray('double'), self._k, self._max_iter, self._runs)

    def get_cluster_label(self):
        '''
        Returns the cluster labels of data.
        
        :returns: (*array*) The cluster labels of data.
        '''
        r = self._model.getClusterLabel()
        return np.array(r)
        
        
##############################################