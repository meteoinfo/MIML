# -*- coding: utf-8 -*-

from smile.regression import OLS as JOLS

from .regressor import Regressor
import mipylib.numeric as np

class OLS(Regressor):
    '''
    Ordinary least squares. 

    In linear regression, the model specification is that the dependent variable is a linear 
    combination of the parameters (but need not be linear in the independent variables). The 
    residual is the difference between the value of the dependent variable predicted by the model, 
    and the true value of the dependent variable. Ordinary least squares obtains parameter 
    estimates that minimize the sum of squared residuals, SSE (also denoted RSS).

    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param method: (*string*) ['qr' | 'svd'].
    '''
    
    def __init__(self, x=None, y=None, method='qr'):
        self._x = x
        self._y = y        
        self._method = method
        if x is None or y is None:
            self._model = None
        else:
            self._learn()
        
    def _learn(self):
        self._model = JOLS(self._x.tojarray('double'), self._y.tojarray('double'), 
            self._method=='svd')
    
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