
from smile.clustering import GMeans as JGMeans

import mipylib.numeric as np
from .cluster import Cluster

class GMeans(Cluster):
    '''
    G-Means clustering.
    
    An extended K-Means which tries to automatically determine the number of clusters by normality 
    test. The G-means algorithm is based on a statistical test for the hypothesis that a subset of 
    data follows a Gaussian distribution. G-means runs k-means with increasing k in a hierarchical 
    fashion until the test accepts the hypothesis that the data assigned to each k-means center are 
    Gaussian.

    :param k_max: (*int*) The maximum number of clusters.
    '''
    
    def __init__(self, k_max=100):
        super(GMeans, self).__init__()
        
        self.k_max = k_max
        self.cluster_centers_ = None
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = JGMeans.fit(x.tojarray('double'), self.k_max)
        self.labels_ = np.array(self._model.y)
        self.cluster_centers_ = np.array(self._model.centroids)
        return self
        
        
##############################################