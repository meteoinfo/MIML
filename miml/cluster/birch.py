
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
        
        self._k = k
        self._min_pts = min_pts
        self._branch = branch
        self._radius = radius        
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        n = x.shape[0]
        self._model = JBIRCH(x.shape[1], self._branch, self._radius)
        for i in range(n):
            self._model.add(x[i,:])
        self._model.partition(self._k, self._min_pts)
        return self

    def fit_predict(self, x):
        '''
        Cluster a new instance to the nearest CF leaf. After building the CF tree, the user should 
        call partition(int) method first to clustering leaves. Then they call this method to 
        clustering new data.
        
        :param x: (*array*) A new instance.
        
        :returns: (*array*) the cluster label, which is the label of nearest CF leaf. Note that it 
            may be Clustering Outler.
        '''
        self.fit(x)
        
        if x.ndim == 1:
            return self._model.predict(x)
        else:
            y = np.zeros(len(x), dtype='int')
            for i in range(len(x)):
                y[i] = self._model.predict(x[i])
            return y
        
        
##############################################