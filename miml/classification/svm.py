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

    :param kernel: (*string*) Mercer kernel.
    :param C: (*int*) Regularization parameter.
    :param strategy: (*string*) Multi-class classification strategy, one vs all or one vs one. 
        Ignored for binary classification..
    :param epochs: (*int*) The number of training epochs.
    '''
    
    def __init__(self, kernel='gaussian', C=10, strategy='one_vs_one',
        epochs=1, **kwargs):
        super(SVM, self).__init__()
        
        self._kernel = get_kernel(kernel, **kwargs)
        self._C = C
        self._strategy = strategy
        self._epochs = epochs        
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        k = int(y.max()) + 1
        if k == 2:
            self._model = JSVM(self._kernel, self._C)
        else:
            self._model = JSVM(self._kernel, self._C, k, JSVM.Multiclass.valueOf(self._strategy.upper()))
        x = x.tojarray('double')
        y = y.tojarray('int')
        for i in range(self._epochs):
            print('Epoch %i' % (i+1))
            self._model.learn(x, y)
            self._model.finish()
        
        
##################################################