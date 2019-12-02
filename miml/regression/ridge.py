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

    :param L: (*float*) The shrinkage/regularization parameter. Large lambda means more shrinkage. 
        Choosing an appropriate value of lambda is important, and also difficult.
    '''
    
    def __init__(self, L=None):
        super(RidgeRegression, self).__init__()
        
        self._L = L
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        self._model = JRidgeRegression(x.tojarray('double'), y.tojarray('double'), 
            self._L)
        
        
##################################################