# -*- coding: utf-8 -*-

from smile.classification import DecisionTree as JDecisionTree
from smile.base.cart import SplitRule

from ..utils.smile_util import numeric_attributes

import mipylib.numeric as np
from .classifer import Classifer

class DecisionTree(Classifer):
    '''
    Decision tree for classification. 

    A decision tree can be learned by splitting the training set into subsets based on an attribute 
    value test. This process is repeated on each derived subset in a recursive manner called 
    recursive partitioning. The recursion is completed when the subset at a node all has the same 
    value of the target variable, or when splitting no longer adds value to the predictions.

    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param attributes: (*array*) Attribute properties.
    :param split_rule: (*string*) The splitting rule. 
    '''
    
    def __init__(self, max_nodes=200, attributes=None, split_rule='gini'):
        super(DecisionTree, self).__init__()
        
        self._max_nodes = max_nodes
        self._attributes = attributes
        self._split_rule = split_rule       
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        if self._attributes is None:
            self._attributes = numeric_attributes(x.shape[1])
        self._model = JDecisionTree(self._attributes, x.tojarray('double'),
            y.tojarray('int'), self._max_nodes, 
            SplitRule.valueOf(self._split_rule.upper()))

    @property
    def feature_importances_(self):
        """Return the feature importances.

        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.

        Returns
        -------
        feature_importances_ : array, shape = [n_features]
        """
        if self._model is None:
            return None
        else:
            importances = np.array(self._model.importance())
            normalizer = np.sum(importances)
            if normalizer > 0.0:
                importances /= normalizer
            return importances
        
##################################################