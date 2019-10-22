# -*- coding: utf-8 -*-

from smile.classification import RBFNetwork as JRBFNetwork
from smile.math.distance import EuclideanDistance
from smile.math.rbf import GaussianRadialBasis
from smile.util import SmileUtils

import mipylib.numeric as np
from .classifer import Classifer

class RBFNetwork(Classifer):
    '''
    Radial basis function networks.

    A radial basis function network is an artificial neural network that uses radial basis 
    functions as activation functions. It is a linear combination of radial basis functions. 
    They are used in function approximation, time series prediction, and control.

    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param distance: (*string*) The distance metric functor.
    :param rbf: (*string*) The radial basis functions.
    :param ncenters: (*int*) The number of centers of RBF functions.
    :param normalized: (*boolean*) True for the normalized RBF network.
    '''
    
    def __init__(self, x=None, y=None, distance='euclidean', rbf='gaussian', ncenters=50,
        normalized=False):
        self._x = x
        self._y = y
        self._distance = distance
        self._rbf = rbf
        self._ncenters = ncenters
        self._normalized = normalized
        if x is None or y is None:
            self._model = None
        else:
            self._learn()
        
    def _learn(self):
        distance = EuclideanDistance()        
        centers = np.zeros((50, self._x.shape[1]), dtype='double').tojarray('double')
        x = self._x.tojarray('double')
        rbf = SmileUtils.learnGaussianRadialBasis(x, centers)
        self._model = JRBFNetwork(x, self._y.tojarray('int'), 
            distance, rbf, centers, self._normalized)
    
    def learn(self, x=None, y=None, distance=None, rbf=None, centers=None,
        normalized=None):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param distance: (*string*) The distance metric functor.
        :param rbf: (*string*) The radial basis functions.
        :param centers: (*array*) The centers of RBF functions.
        :param normalized: (*boolean*) True for the normalized RBF network.
        """ 
        if not x is None:
            self._x = x
        if not y is None:
            self._y = y
        if not distance is None:
            self._distance = distance
        if not rbf is None:
            self._rbf = rbf
        if not centers is None:
            self._centers = centers
        if not normalized is None:
            self._normalized = normalized
        self._learn()        
        
        
##################################################