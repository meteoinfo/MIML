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

    :param priori: (*array*) The priori probability of each class.
    :param tol: (*float*) a tolerance to decide if a covariance matrix is singular; it will reject 
        variables whose variance is less than tol^2.    
    '''
    
    def __init__(self, priori=None, tol=0.0001):
        super(QuadraticDiscriminantAnalysis, self).__init__()
        
        self._priori = priori
        self._tol = tol
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        priori = None if self._priori is None else self._priori.tojarray('double')
        self._model = QDA.fit(x.tojarray('double'), y.tojarray('int'), priori, self._tol)
        
        
##################################################