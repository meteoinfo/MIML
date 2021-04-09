
from smile.clustering import MEC as JMEC

import mipylib.numeric as np
from .cluster import Cluster
from ..utils import smile_util

class MEC(Cluster):
    '''
    Nonparametric Minimum Conditional Entropy Clustering.
    
    DBSCAN finds a number of clusters starting from the estimated density
    distribution of corresponding nodes.
    
    DBSCAN requires two parameters: radius (i.e. neighborhood radius) and the
    number of minimum points required to form a cluster (minPts). It starts
    with an arbitrary starting point that has not been visited. This point's
    neighborhood is retrieved, and if it contains sufficient number of points,
    a cluster is started. Otherwise, the point is labeled as noise. Note that
    this point might later be found in a sufficiently sized radius-environment
    of a different point and hence be made part of a cluster.

    :param distance: (*string*) The distance.
    :param k: (*int*) the number of clusters. Note that this is just a hint. 
        The final number of clusters may be less.
    :param radius: (*float*) the neighborhood radius.
    '''
    
    def __init__(self, distance='euclidean', k=None, radius=None):
        super(MEC, self).__init__()
        
        self.distance = distance
        self.k = k
        self.radius = radius
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        distance = smile_util.get_distance(self.distance)
        self._model = JMEC.fit(x.tojarray('double'), distance, self.k, self.radius)
        self.labels_ = np.array(self._model.y)
        return self
        
        
##############################################