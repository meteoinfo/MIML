
from smile.clustering import SpectralClustering as JSpectralClustering

import mipylib.numeric as np
from .cluster import Cluster

class SpectralClustering(Cluster):
    '''
    Spectral Clustering.
    
    Given a set of data points, the similarity matrix may be defined as a matrix S where Sij 
    represents a measure of the similarity between points. Spectral clustering techniques make use 
    of the spectrum of the similarity matrix of the data to perform dimensionality reduction for 
    clustering in fewer dimensions. Then the clustering will be performed in the dimension-reduce 
    space, in which clusters of non-convex shape may become tight. There are some intriguing 
    similarities between spectral clustering methods and kernel PCA, which has been empirically 
    observed to perform clustering.

    :param k: (*int*) The number of cluster.
    :param l: (*int*) the number of random samples for Nystrom approximation.
    :param sigma: (*float*) the smooth/width parameter of Gaussian kernel, which is a somewhat 
        sensitive parameter. To search for the best setting, one may pick the value that gives the 
        tightest clusters (smallest distortion) in feature space.
    '''
    
    def __init__(self, k, l=None, sigma=None):
        super(SpectralClustering, self).__init__()
        
        self.k = k
        self.l = l
        self.sigma = sigma
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        if self.l is None:
            self._model = JSpectralClustering.fit(x.tojarray('double'), self.k, self.sigma)
        else:
            self._model = JSpectralClustering.fit(x.tojarray('double'), self.k, self.l, self.sigma)

        self.labels_ = np.array(self._model.y)
        return self
        
        
##############################################