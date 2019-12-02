# -*- coding: utf-8 -*-

from smile.classification import RBFNetwork as JRBFNetwork
from smile.math.distance import EuclideanDistance
from smile.math.rbf import GaussianRadialBasis
from smile.util import SmileUtils

import mipylib.numeric as np
from .classifer import Classifer
from ..utils.smile_util import get_distance

class RBFNetwork(Classifer):
    '''
    Radial basis function networks.

    A radial basis function network is an artificial neural network that uses radial basis 
    functions as activation functions. It is a linear combination of radial basis functions. 
    They are used in function approximation, time series prediction, and control.

    :param distance: (*string*) The distance metric functor.
    :param rbf: (*string*) The radial basis functions.
    :param ncenters: (*int*) The number of centers of RBF functions.
    :param normalized: (*boolean*) True for the normalized RBF network.
    '''
    
    def __init__(self, distance='euclidean', rbf='gaussian', ncenters=50,
        normalized=False):
        super(RBFNetwork, self).__init__()

        self._distance = distance
        self._rbf = rbf
        self._ncenters = ncenters
        self._normalized = normalized

    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        distance = get_distance(self._distance)      
        centers = np.zeros((50, x.shape[1]), dtype='double').tojarray('double')
        x = x.tojarray('double')
        rbf = SmileUtils.learnGaussianRadialBasis(x, centers)
        self._model = JRBFNetwork(x, y.tojarray('int'), 
            distance, rbf, centers, self._normalized)
        
        
##################################################