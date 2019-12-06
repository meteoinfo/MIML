
from smile.clustering import CLARANS as JCLARANS
from smile.math import MathEx
from java.util.function import ToDoubleBiFunction

import mipylib.numeric as np
from .cluster import Cluster

class dbiF(ToDoubleBiFunction):
    def applyAsDouble(self, x, y):
        return MathEx.squaredDistance(x, y)

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
    :param max_neighbor: (*int*) the maximum number of neighbors examined during a random search of 
        local minima.
    '''
    
    def __init__(self, k, max_neighbor=None):
        super(CLARANS, self).__init__()
        
        self._k = k
        self._max_neighbor = max_neighbor
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = JCLARANS.fit(x.tojarray('double'), self._k, self._max_neighbor,
            dbiF())
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