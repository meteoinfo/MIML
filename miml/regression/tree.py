# -*- coding: utf-8 -*-

from smile.regression import RegressionTree as JRegressionTree

from ..utils.smile_util import numeric_attributes

import mipylib.numeric as np
from .regressor import Regressor

class RegressionTree(Regressor):
    '''
    Decision tree for regression. 

    A decision tree can be learned by splitting the training set into subsets based on an attribute 
    value test. This process is repeated on each derived subset in a recursive manner called 
    recursive partitioning. The recursion is completed when the subset at a node all has the same 
    value of the target variable, or when splitting no longer adds value to the predictions.

    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param attributes: (*array*) Attribute properties.
    '''
    
    def __init__(self, max_nodes=200, attributes=None):
        super(RegressionTree, self).__init__()
        
        self._max_nodes = max_nodes
        self._attributes = attributes       
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        if self._attributes is None:
            self._attributes = numeric_attributes(x.shape[1])
        self._model = JRegressionTree(self._attributes, x.tojarray('double'),
            y.tojarray('double'), self._max_nodes)
        
        
##################################################