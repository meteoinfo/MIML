# -*- coding: utf-8 -*-

from smile.classification import SVM as JSVM
from smile.classification import OneVersusOne, OneVersusRest
from java.util.function import BiFunction

from ..utils.smile_util import get_kernel
from .classifer import Classifer

class biF(BiFunction):
    def __init__(self, kernel, C, tol):
        self.kernel = kernel
        self.C = C
        self.tol = tol

    def apply(self, x, y):
        return JSVM.fit(x, y, self.kernel, self.C, self.tol)


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
    :param strategy: (*string*) Multi-class classification strategy ['one_vs_one' | 'one_vs_rest'].
        Ignored for binary classification.
    '''
    
    def __init__(self, kernel='gaussian', C=10, tol=1E-3, strategy='one_vs_one',
        **kwargs):
        super(SVM, self).__init__()
        
        self.kernel = get_kernel(kernel, **kwargs)
        self.C = C
        self.tol = tol
        self.strategy = strategy

    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        super(SVM, self).fit(x, y)
        k = int(y.max()) + 1
        if k == 2:
            self.multiclass = False
            y = (y * 2 - 1).copy()
            x = x.tojarray('double')
            y = y.tojarray('int')
            self._model = JSVM.fit(x, y, self.kernel, self.C, self.tol)
        else:
            self.multiclass = True
            x = x.tojarray('double')
            y = y.tojarray('int')
            if self.strategy == 'one_vs_one':
                self._model = OneVersusOne.fit(x, y, biF(self.kernel, self.C, self.tol))
            else:
                self._model = OneVersusRest.fit(x, y, biF(self.kernel, self.C, self.tol))

    def predict(self, x):
        if self.multiclass:
            return super(SVM, self).predict(x)
        else:
            r = super(SVM, self).predict(x)
            return (r + 1) / 2
        
##################################################