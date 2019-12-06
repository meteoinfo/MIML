
from smile.clustering import DBSCAN as JDBSCAN

import mipylib.numeric as np
from .cluster import Cluster
from ..utils import smile_util

class DBSCAN(Cluster):
    '''
    Density-Based Spatial Clustering of Applications with Noise.
    
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
    :param min_pts: (*int*) the minimum number of neighbors for a core data point.
    :param radius: (*float*) the neighborhood radius.
    '''
    
    def __init__(self, distance='euclidean', min_pts=None, radius=None):
        super(DBSCAN, self).__init__()
        
        self._distance = distance
        self._min_pts = min_pts
        self._radius = radius 
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        distance = smile_util.get_distance(self._distance) 
        self._model = JDBSCAN.fit(x.tojarray('double'), distance,
            self._min_pts, self._radius)
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