
from smile.clustering import BIRCH as JBIRCH

import mipylib.numeric as np
from .cluster import Cluster

class BIRCH(Cluster):
    '''
    Balanced Iterative Reducing and Clustering using Hierarchies.
    
    BIRCH performs hierarchical clustering over particularly large datasets. An advantage of BIRCH is 
    its ability to incrementally and dynamically cluster incoming, multi-dimensional metric data 
    points in an attempt to produce the high quality clustering for a given set of resources (memory 
    and time constraints).

    :param k: (*int*) Number of clusters.
    :param min_pts: (*int*) a CF leaf will be treated as outlier if the number of its
        points is less than minPts.
    :param branch: (*int*) the branching factor. Maximum number of children nodes.
    :param radius: (*float*) the maximum radius of a sub-cluster.
    '''
    
    def __init__(self, k, min_pts=None, branch=None, radius=None):
        super(BIRCH, self).__init__()
        
        self.k = k
        self.min_pts = min_pts
        self.branch = branch
        self.radius = radius
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        n = x.shape[0]
        self._model = JBIRCH(x.shape[1], self.branch, self.radius)
        for i in range(n):
            self._model.add(x[i,:])
        self._model.partition(self.k, self.min_pts)

        if x.ndim == 1:
            y = self._model.predict(x)
        else:
            y = np.zeros(len(x), dtype='int')
            for i in range(len(x)):
                y[i] = self._model.predict(x[i])

        self.labels_ = np.array(y)
        return self
        
        
##############################################