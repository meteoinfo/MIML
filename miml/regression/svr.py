# -*- coding: utf-8 -*-

from smile.regression import SVR as JSVR

from ..utils.smile_util import get_kernel
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
    :param tol: (*float*) The tolerance of convergence test.
    '''
    
    def __init__(self, kernel='gaussian', eps=None, C=10, weight=None,
            tol=1e-3, **kwargs):
        super(SVR, self).__init__()
        
        self.kernal = get_kernel(kernel, **kwargs)
        self.eps = eps
        self.C = C
        self.tol = tol
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        x = x.tojarray('double')
        y = y.tojarray('double')
        self._model = JSVR.fit(x, y, self.kernal, self.eps, self.C, self.tol)
        
        
##################################################