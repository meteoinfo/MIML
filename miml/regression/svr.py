# -*- coding: utf-8 -*-

from smile.regression import SVR as JSVR
from smile.util import SmileUtils

from ..utils.smile_util import get_kernel

import mipylib.numeric as np
from .regressor import Regressor

class SVR(Regressor):
    '''
    Support vector machines for classification.

    The basic support vector machine is a binary linear classifier which chooses the hyperplane 
    that represents the largest separation, or margin, between the two classes. If such a 
    hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier 
    it defines is known as a maximum margin classifier.

    :param kernel: (*string*) Mercer kernel.
    :param eps: (*float*) The loss function error threshold.
    :param C: (*float*) The soft margin penalty parameter.
    :param weight: (*array*) positive instance weight. The soft margin penalty
         parameter for instance i will be weight[i] * C.
    :param tol: (*float*) The tolerance of convergence test.
    '''
    
    def __init__(self, kernel='gaussian', eps=None, C=10, weight=None,
            tol=1e-3, **kwargs):
        super(SVR, self).__init__()
        
        self._kernal = get_kernel(kernel, **kwargs)
        self._eps = eps
        self._C = C
        self._weight = weight
        self._tol = tol        
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        x = x.tojarray('double')
        y = y.tojarray('double')
        if self._weight is None:
            self._model = JSVR(x, y, self._kernal, self._eps, self._C, self._tol)
        else:
            self._model = JSVR(x, y, self._weight, self._kernal, self._eps, self._C, self._tol)
        
        
##################################################