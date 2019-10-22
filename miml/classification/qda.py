# -*- coding: utf-8 -*-

from smile.classification import QDA

import mipylib.numeric as np
from .classifer import Classifer

class QuadraticDiscriminantAnalysis(Classifer):
    '''
    Quadratic Discriminant Analysis

    QDA is closely related to linear discriminant analysis (LDA). Like LDA, QDA models the 
    conditional probability density functions as a Gaussian distribution, then uses the posterior 
    distributions to estimate the class for a given test data. Unlike LDA, however, in QDA there 
    is no assumption that the covariance of each of the classes is identical. Therefore, the 
    resulting separating surface between the classes is quadratic.
    
    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param priori: (*array*) The priori probability of each class.
    :param tol: (*float*) a tolerance to decide if a covariance matrix is singular; it will reject 
        variables whose variance is less than tol^2.
    
    :returns: 
    '''
    
    def __init__(self, x=None, y=None, priori=None, tol=0.0001):
        self._x = x
        self._y = y
        self._priori = priori
        self._tol = tol
        if x is None or y is None:
            self._model = None
        else:
            self._learn()
    
    def _learn(self):
        if self._priori is None:
            self._model = QDA(self._x.tojarray('double'), self._y.tojarray('int'), self._tol)
        else:
            self._model = QDA(self._x.tojarray('double'), self._y.tojarray('int'), 
                self._priori.tojarray('double'), self._tol)
    
    def learn(self, x=None, y=None, priori=None, tol=None):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param priori: (*array*) The priori probability of each class.
        :param tol: (*float*) a tolerance to decide if a covariance matrix is singular; it will reject 
            variables whose variance is less than tol^2.
        """
        if not x is None:
            self._x = x
        if not y is None:
            self._y = y
        if not priori is None:
            self._priori = priori
        if not tol is None:
            self._tol = tol
        self._learn()
        
        
##################################################