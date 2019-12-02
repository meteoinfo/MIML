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

    :param method: (*string*) ['qr' | 'svd'].
    '''
    
    def __init__(self, method='qr'):
        super(OLS, self).__init__()
        
        self._method = method
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        self._model = JOLS(x.tojarray('double'), y.tojarray('double'), 
            self._method=='svd')
        
        
##################################################