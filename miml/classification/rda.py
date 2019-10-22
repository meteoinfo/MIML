# -*- coding: utf-8 -*-

from smile.classification import RDA

import mipylib.numeric as np
from .classifer import Classifer

class RegularizedDiscriminantAnalysis(Classifer):
    '''
    Regularized Discriminant Analysis

    RDA is a compromise between LDA and QDA, which allows one to shrink the separate covariances 
    of QDA toward a common variance as in LDA. This method is very similar in flavor to ridge 
    regression. The regularized covariance matrices of each class is Σk(α) = α Σk + (1 - α) Σ. 
    The quadratic discriminant function is defined using the shrunken covariance matrices Σk(α). 
    The parameter α in [0, 1] controls the complexity of the model. When α is one, RDA becomes QDA. 
    While α is zero, RDA is equivalent to LDA. Therefore, the regularization factor α allows a 
    continuum of models between LDA and QDA.
    
    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param alpha: (*float*) Regularization factor in [0, 1] allows a continuum of models between 
        LDA and QDA.
    :param priori: (*array*) The priori probability of each class.
    :param tol: (*float*) a tolerance to decide if a covariance matrix is singular; it will reject 
        variables whose variance is less than tol^2.
    
    :returns: 
    '''
    
    def __init__(self, x=None, y=None, alpha=None, priori=None, tol=0.0001):
        self._x = x
        self._y = y
        self._alpha = alpha
        self._priori = priori
        self._tol = tol
        if x is None or y is None or alpha is None:
            self._model = None
        else:
            self._learn()
    
    def _learn(self):
        if self._priori is None:
            self._model = RDA(self._x.tojarray('double'), self._y.tojarray('int'), 
                self._priori, self._alpha, self._tol)
        else:
            self._model = RDA(self._x.tojarray('double'), self._y.tojarray('int'), 
                self._priori.tojarray('double'), self._alpha, self._tol)
    
    def learn(self, x=None, y=None, alpha=None, priori=None, tol=None):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param alpha: (*float*) Regularization factor in [0, 1] allows a continuum of models between 
            LDA and QDA.
        :param priori: (*array*) The priori probability of each class.
        :param tol: (*float*) a tolerance to decide if a covariance matrix is singular; it will reject 
            variables whose variance is less than tol^2.
        """
        if not x is None:
            self._x = x
        if not y is None:
            self._y = y
        if not alpha is None:
            self._alpha = alpha
        if not priori is None:
            self._priori = priori
        if not tol is None:
            self._tol = tol
        self._learn()
        
        
##################################################