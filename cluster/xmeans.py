
from smile.clustering import XMeans as JXMeans

import mipylib.numeric as np
from .cluster import Cluster

class XMeans(Cluster):
    '''
    X-Means clustering.
    
    An extended K-Means which tries to automatically determine the number of clusters based on BIC 
    scores. Starting with only one cluster, the X-Means algorithm goes into action after each run 
    of K-Means, making local decisions about which subset of the current centroids should split 
    themselves in order to better fit the data. The splitting decision is done by computing the 
    Bayesian Information Criterion (BIC).
    
    :param data: (*array*) The data set.
    :param k_max: (*int*) The maximum number of clusters.
    '''
    
    def __init__(self, data, k_max=100):
        self._data = data
        self._k_max = k_max
        self._model = JXMeans(self._data.tojarray('double'), self._k_max)

    def get_cluster_label(self):
        '''
        Returns the cluster labels of data.
        
        :returns: (*array*) The cluster labels of data.
        '''
        r = self._model.getClusterLabel()
        return np.array(r)
        
        
##############################################