# -*- coding: utf-8 -*-

from smile.classification import GradientTreeBoost as JGradientTreeBoost

from ..utils.smile_util import numeric_attributes

import mipylib.numeric as np
import math
from .classifer import Classifer

class GradientTreeBoost(Classifer):
    '''
    Gradient boosting for classification. 

    Gradient boosting is typically used with decision trees (especially CART regression trees) of 
    a fixed size as base learners. For this special case Friedman proposes a modification to 
    gradient boosting method which improves the quality of fit of each base learner.

    :param attributes: (*array*) Attribute properties.
    :param ntrees: (*int*) The number of trees.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param shrinkage: (*float*) The shrinkage parameter in (0, 1] controls the learning rate of 
        procedure.
    :param sub_sample: (*float*) The sampling rate for training tree. 1.0 means sampling with 
        replacement. < 1.0 means sampling without replacement.
    '''
    
    def __init__(self, attributes=None, ntrees=500, max_nodes=6, shrinkage=0.05,
            mtry=-1, sub_sample=1.0):  
        super(GradientTreeBoost, self).__init__()
        
        self._attributes = attributes
        self._ntrees = ntrees        
        self._max_nodes = max_nodes
        self._shrinkage = shrinkage
        self._sub_sample = sub_sample
    
    def fit(self, x, y):
        '''
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        '''
        p = x.shape[1]
        if self._attributes is None:
            self._attributes = numeric_attributes(p)
        self._model = JGradientTreeBoost(self._attributes, x.tojarray('double'),
            y.tojarray('int'), self._ntrees, self._max_nodes, self._shrinkage, 
            self._sub_sample)
        
        
##################################################