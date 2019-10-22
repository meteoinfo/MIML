# -*- coding: utf-8 -*-

from smile.classification import SVM as JSVM
from smile.util import SmileUtils

from ..utils.smile_util import get_kernel

import mipylib.numeric as np
from .classifer import Classifer

class SVM(Classifer):
    '''
    Support vector machines for classification.

    The basic support vector machine is a binary linear classifier which chooses the hyperplane 
    that represents the largest separation, or margin, between the two classes. If such a 
    hyperplane exists, it is known as the maximum-margin hyperplane and the linear classifier 
    it defines is known as a maximum margin classifier.

    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param kernel: (*string*) Mercer kernel.
    :param C: (*int*) Regularization parameter.
    :param strategy: (*string*) Multi-class classification strategy, one vs all or one vs one. 
        Ignored for binary classification..
    :param epochs: (*int*) The number of training epochs.
    '''
    
    def __init__(self, x=None, y=None, kernel='gaussian', C=10, strategy='one_vs_one',
        epochs=1, **kwargs):
        self._x = x
        self._y = y
        self._kernal = get_kernel(kernel, **kwargs)
        self._C = C
        self._strategy = strategy
        self._epochs = epochs        
        if x is None or y is None:
            self._model = None
        else:
            self._learn()
        
    def _learn(self):
        k = int(self._y.max()) + 1
        if k == 2:
            self._model = JSVM(self._kernal, self._C)
        else:
            self._model = JSVM(self._kernal, self._C, k, JSVM.Multiclass.valueOf(self._strategy.upper()))
        x = self._x.tojarray('double')
        y = self._y.tojarray('int')
        for i in range(self._epochs):
            self._model.learn(x, y)
            self._model.finish()
    
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