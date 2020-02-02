# -*- coding: utf-8 -*-

from smile.classification import FLD

import mipylib.numeric as np
from .classifer import Classifer

class FisherLinearDiscriminant(Classifer):
    '''
    Fisher's linear discriminant

    Fisher defined the separation between two distributions to be the ratio of the variance between 
    the classes to the variance within the classes, which is, in some sense, a measure of the 
    signal-to-noise ratio for the class labeling. FLD finds a linear combination of features which 
    maximizes the separation after the projection. The resulting combination may be used for 
    dimensionality reduction before later classification.
    
    :param L: (*int*) the dimensionality of mapped space. The default value is the number of 
        classes - 1.
    :param tol: (*float*) a tolerance to decide if a covariance matrix is singular; it will reject 
        variables whose variance is less than tol^2.
    
    :returns: 
    '''
    
    def __init__(self, L=-1, tol=0.0001):
        super(FisherLinearDiscriminant, self).__init__()
        
        self._L = L
        self._tol = tol
         
    def fit(self, x, y):
        '''
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        '''
        super(FisherLinearDiscriminant, self).fit(x, y)
        self._model = FLD.fit(x.tojarray('double'), y.tojarray('int'),
            self._L, self._tol) 
        
##################################################