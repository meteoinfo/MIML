
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
    
    :param data: (*array*) The data set.
    :param k_max: (*int*) The maximum number of clusters.
    '''
    
    def __init__(self, data, k_max=100):
        self._data = data
        self._k_max = k_max
        self._model = JGMeans(self._data.tojarray('double'), self._k_max)

    def get_cluster_label(self):
        '''
        Returns the cluster labels of data.
        
        :returns: (*array*) The cluster labels of data.
        '''
        r = self._model.getClusterLabel()
        return np.array(r)
        
        
##############################################