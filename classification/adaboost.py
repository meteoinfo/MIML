# -*- coding: utf-8 -*-

from smile.classification import AdaBoost as JAdaBoost

from ..utils.smile_util import numeric_attributes

import mipylib.numeric as np
import math
from .classifer import Classifer

class AdaBoost(Classifer):
    '''
    AdaBoost (Adaptive Boosting) classifier with decision trees. 

    In principle, AdaBoost is a meta-algorithm, and can be used in conjunction with many other 
    learning algorithms to improve their performance. In practice, AdaBoost with decision trees 
    is probably the most popular combination. AdaBoost is adaptive in the sense that subsequent 
    classifiers built are tweaked in favor of those instances misclassified by previous classifiers. 
    AdaBoost is sensitive to noisy data and outliers. However in some problems it can be less 
    susceptible to the over-fitting problem than most learning algorithms.

    :param x: (*array*) Training samples. 2D array.
    :param y: (*array*) Training labels in [0, c), where c is the number of classes.
    :param attributes: (*array*) Attribute properties.
    :param ntrees: (*int*) The number of trees.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    '''
    
    def __init__(self, x=None, y=None, attributes=None, ntrees=500, max_nodes=2):
        self._x = x
        self._y = y        
        self._attributes = attributes
        self._ntrees = ntrees        
        self._max_nodes = max_nodes
        if x is None or y is None:
            self._model = None
        else:
            self._learn()
        
    def _learn(self):
        p = self._x.shape[1]
        if self._attributes is None:
            self._attributes = numeric_attributes(p)
        self._model = JAdaBoost(self._attributes, self._x.tojarray('double'),
            self._y.tojarray('int'), self._ntrees, self._max_nodes)
    
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