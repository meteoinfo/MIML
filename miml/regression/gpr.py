# -*- coding: utf-8 -*-

from smile.regression import GaussianProcessRegression as JGaussianProcessRegression

from ..utils.smile_util import get_kernel

import mipylib.numeric as np
from .regressor import Regressor

class GaussianProcessRegression(Regressor):
    '''
    Gaussian Process for Regression. 

    A Gaussian process is a stochastic process whose realizations consist of random values 
    associated with every point in a range of times (or of space) such that each such random 
    variable has a normal distribution. Moreover, every finite collection of those random variables 
    has a multivariate normal distribution.

    :param t: (*array*) The inducing input, which are pre-selected or inducing samples
        acting as active set of regressors. In simple case, these can be chosen
        randomly from the training set or as the centers of k-means clustering.
    :param kernel: (*string*) The Mercer kernel.
    :param L: (*float*) The shrinkage/regularization parameter.
    :param nystrom: (*boolean*) Set it true for Nystrom approximation of kernel matrix.
    '''
    
    def __init__(self, t=None, kernel='gaussian', L=0.01, 
            nystrom=False, **kwargs):
        super(GaussianProcessRegression, self).__init__()
        
        self._t = t
        self._kernel = get_kernel(kernel, **kwargs)
        self._L = L        
        self._nystrom = nystrom
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        if self._t is None:
            self._model = JGaussianProcessRegression(x.tojarray('double'),
                y.tojarray('double'), self._kernel, self._L)
        else:
            if self._nystrom:
                self._model = JGaussianProcessRegression(x.tojarray('double'),
                    y.tojarray('double'), self._t.tojarray('double'), self._kernel, 
                    self._L, True)
            else:
                self._model = JGaussianProcessRegression(x.tojarray('double'),
                    y.tojarray('double'), self._t.tojarray('double'), self._kernel, 
                    self._L)
        
        
##################################################