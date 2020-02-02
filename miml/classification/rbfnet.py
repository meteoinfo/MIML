# -*- coding: utf-8 -*-

from smile.classification import RBFNetwork as JRBFNetwork
from smile.base.rbf import RBF

from .classifer import Classifer

class RBFNetwork(Classifer):
    '''
    Radial basis function networks.

    A radial basis function network is an artificial neural network that uses radial basis 
    functions as activation functions. It is a linear combination of radial basis functions. 
    They are used in function approximation, time series prediction, and control.

    :param k: (*int*) The number of RBF neurons to learn.
    :param normalized: (*boolean*) True for the normalized RBF network.
    '''
    
    def __init__(self, k=50, normalized=False):
        super(RBFNetwork, self).__init__()

        self._k = k
        self._normalized = normalized

    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        super(RBFNetwork, self).fit(x, y)
        x = x.tojarray('double')
        neurons = RBF.fit(x, self._k)
        self._model = JRBFNetwork.fit(x, y.tojarray('int'), neurons, self._normalized)
        
        
##################################################