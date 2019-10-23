
from smile.clustering import DENCLUE as JDENCLUE

import mipylib.numeric as np
from .cluster import Cluster

class DENCLUE(Cluster):
    '''
    DENsity CLUstering.
    
    The DENCLUE algorithm employs a cluster model based on kernel density estimation. A cluster is 
    defined by a local maximum of the estimated density function. Data points going to the same local 
    maximum are put into the same cluster.
    
    Clearly, DENCLUE doesn't work on data with uniform distribution. In high dimensional space, the 
    data always look like uniformly distributed because of the curse of dimensionality. Therefore, 
    DENCLUDE doesn't work well on high-dimensional data in general.

    :param sigma: (*float*) the smooth parameter in the Gaussian kernel. The user can choose sigma 
        such that number of density attractors is constant for a long interval of sigma.
    :param m: (*int*) the number of selected samples used in the iteration. This number should be 
        much smaller than the number of data points to speed up the algorithm. It should also be 
        large enough to capture the sufficient information of underlying distribution.
    '''
    
    def __init__(self, sigma=None, m=None):
        super(DENCLUE, self).__init__()
        
        self._sigma = sigma
        self._m = m
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        self._model = JDENCLUE(x.tojarray('double'), self._sigma, self._m)
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