# -*- coding: utf-8 -*-

from smile.regression import RBFNetwork as JRBFNetwork
from smile.base.rbf import RBF

from .regressor import Regressor

class RBFNetwork(Regressor):
    '''
    Radial basis function networks.

    A radial basis function network is an artificial neural network that uses radial basis 
    functions as activation functions. It is a linear combination of radial basis functions. 
    They are used in function approximation, time series prediction, and control.

    :param k: (*int*) Trains a Gaussian RBF network with k-means.
    :param normalized: (*boolean*) True for the normalized RBF network.
    '''
    
    def __init__(self, k=10,
            normalized=False):
        super(RBFNetwork, self).__init__()

        self._k = k
        self._normalized = normalized
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        x = x.tojarray('double')
        neurons = RBF.fit(x, self._k)
        self._model = JRBFNetwork.fit(x, y.tojarray('double'), neurons, self._normalized)
        
        
##################################################