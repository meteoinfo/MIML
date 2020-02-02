from smile.classification import KNN

import mipylib.numeric as np
from .classifer import Classifer

class KNearestNeighbor(Classifer):
    '''
    K-nearest neighbor classifier.

    :param k: (*int*) The number of neighbors for classification.
    
    :returns: 
    '''
    
    def __init__(self, k=1):
        super(KNearestNeighbor, self).__init__()

        self._k = k
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        super(KNearestNeighbor, self).fit(x, y)
        self._model = KNN.fit(x.tojarray('double'), y.tojarray('int'), self._k)
        
        
##################################################