
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
    
    :param k_max: (*int*) The maximum number of clusters.
    '''
    
    def __init__(self, k_max=100):
        super(XMeans, self).__init__()
        
        self.k_max = k_max
        self.cluster_centers_ = None
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = JXMeans.fit(x.tojarray('double'), self.k_max)
        self.labels_ = np.array(self._model.y)
        self.cluster_centers_ = np.array(self._model.centroids)
        return self
        
        
##############################################