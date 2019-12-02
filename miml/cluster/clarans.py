
from smile.clustering import CLARANS as JCLARANS

import mipylib.numeric as np
from .cluster import Cluster
from ..utils import smile_util

class CLARANS(Cluster):
    '''
    Clustering Large Applications based upon RANdomized Search.
    
    CLARANS is an efficient medoid-based clustering algorithm. The k-medoids algorithm is an 
    adaptation of the k-means algorithm. Rather than calculate the mean of the items in each cluster, 
    a representative item, or medoid, is chosen for each cluster at each iteration. In CLARANS, the 
    process of finding k medoids from n objects is viewed abstractly as searching through a certain 
    graph. In the graph, a node is represented by a set of k objects as selected medoids. Two nodes 
    are neighbors if their sets differ by only one object. In each iteration, CLARANS considers a set 
    of randomly chosen neighbor nodes as candidate of new medoids. We will move to the neighbor node 
    if the neighbor is a better choice for medoids. Otherwise, a local optima is discovered. The 
    entire process is repeated multiple time to find better.

    :param k: (*int*) Number of clusters.
    :param distance: (*string*) the distance/dissimilarity measure.
    :param max_neighbor: (*int*) the maximum number of neighbors examined during a random search of 
        local minima.
    :param nlocal: (*int*) the number of local minima to search for.
    '''
    
    def __init__(self, k, distance='euclidean', max_neighbor=None, nlocal=None):
        super(CLARANS, self).__init__()
        
        self._k = k
        self._distance = distance
        self._max_neighbor = max_neighbor
        self._nlocal = nlocal        
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        distance = smile_util.get_distance(self._distance)
        self._model = JCLARANS(x.tojarray('double'), distance, self._k, self._max_neighbor,
            self._nlocal)
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