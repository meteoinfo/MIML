# -*- coding: utf-8 -*-

from smile.regression import LASSO as JLASSO

from .regressor import Regressor
import mipylib.numeric as np

class LASSO(Regressor):
    '''
    Lasso (least absolute shrinkage and selection operator) regression. 

    The Lasso is a shrinkage and selection method for linear regression. It minimizes the usual sum 
    of squared errors, with a bound on the sum of the absolute values of the coefficients (i.e. 
    L1-regularized). It has connections to soft-thresholding of wavelet coefficients, forward 
    stage-wise regression, and boosting methods.

    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param L: (*float*) The shrinkage/regularization parameter. Large lambda means more shrinkage. 
        Choosing an appropriate value of lambda is important, and also difficult.
    :param tol: (*float*) The tolerance for stopping iterations (relative target duality gap).
    :param max_iter: (*int*) The maximum number of IPM (Newton) iterations.
    '''
    
    def __init__(self, x=None, y=None, L=None, tol=1e-3, max_iter=5000):
        self._x = x
        self._y = y        
        self._L = L
        self._tol = tol
        self._max_iter = max_iter
        if x is None or y is None or L is None:
            self._model = None
        else:
            self._learn()
        
    def _learn(self):
        self._model = JLASSO(self._x.tojarray('double'), self._y.tojarray('double'), 
            self._L, self._tol, self._max_iter)
    
    def learn(self, x=None, y=None):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        if not x is None:
            self._x = x
        if not y is None:
            self._y = y
        self._learn()
        
        
##################################################