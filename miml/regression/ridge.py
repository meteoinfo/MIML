# -*- coding: utf-8 -*-

from smile.regression import RidgeRegression as JRidgeRegression

from .regressor import Regressor
import mipylib.numeric as np

class RidgeRegression(Regressor):
    '''
    Ridge Regression. 

    Coefficient estimates for multiple linear regression models rely on the independence of the 
    model terms. When terms are correlated and the columns of the design matrix X have an 
    approximate linear dependence, the matrix X'X becomes close to singular. As a result, the 
    least-squares estimate becomes highly sensitive to random errors in the observed response Y, 
    producing a large variance.

    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param L: (*float*) The shrinkage/regularization parameter. Large lambda means more shrinkage. 
        Choosing an appropriate value of lambda is important, and also difficult.
    '''
    
    def __init__(self, x=None, y=None, L=None):
        self._x = x
        self._y = y        
        self._L = L
        if x is None or y is None or L is None:
            self._model = None
        else:
            self._learn()
        
    def _learn(self):
        self._model = JRidgeRegression(self._x.tojarray('double'), self._y.tojarray('double'), 
            self._L)
    
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