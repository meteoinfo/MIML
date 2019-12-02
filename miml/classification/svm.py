# -*- coding: utf-8 -*-

from smile.classification import SVM as JSVM

from ..utils.smile_util import get_kernel
from .classifer import Classifer

class SVM(Classifer):
    '''
    Support vector machines for classification.

    The basic support vector machine is a binary linear classifier which chooses the hyperplane 
    that represents the largest separation, or margin, between the two classes. If such a 
    hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier 
    it defines is known as a maximum margin classifier.

    :param kernel: (*string*) Mercer kernel.
    :param C: (*int*) Regularization parameter.
    :param tol: (*float*) the tolerance of convergence test.
    '''
    
    def __init__(self, kernel='gaussian', C=10, tol=1E-3,
        **kwargs):
        super(SVM, self).__init__()
        
        self._kernel = get_kernel(kernel, **kwargs)
        self._C = C
        self._tol = tol
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        y = (y * 2 - 1).copy()
        x = x.tojarray('double')
        y = y.tojarray('int')
        self._model = JSVM.fit(x, y, self._kernel, self._C, self._tol)

    def predict(self, x):
        r = super(SVM, self).predict(x)
        return (r + 1) / 2
        
##################################################