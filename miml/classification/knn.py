from smile.classification import KNN

import mipylib.numeric as np
from .classifer import Classifer

class KNearestNeighbor(Classifer):
    '''
    K-nearest neighbor classifier.
    
    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param k: (*int*) The number of neighbors for classification.
    
    :returns: 
    '''
    
    def __init__(self, x=None, y=None, k=1):
        self._x = x
        self._y = y
        self._k = k
        if x is None or y is None:
            self._model = None
        else:
            self._learn()
    
    def _learn(self):
        self._model = KNN.learn(self._x.tojarray('double'), self._y.tojarray('int'), self._k)
    
    def learn(self, x=None, y=None, k=None):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param k: (*int*) The number of neighbors for classification.
        """
        if not k is None:
            self._k = k
        if not x is None:
            self._x = x
        if not y is None:
            self._y = y
        self._learn()
        
        
##################################################