# -*- coding: utf-8 -*-

from smile.classification import DecisionTree as JDecisionTree
from smile.base.cart import SplitRule
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

import mipylib.numeric as np
from .classifer import Classifer

class DecisionTree(Classifer):
    '''
    Decision tree for classification. 

    A decision tree can be learned by splitting the training set into subsets based on an attribute 
    value test. This process is repeated on each derived subset in a recursive manner called 
    recursive partitioning. The recursion is completed when the subset at a node all has the same 
    value of the target variable, or when splitting no longer adds value to the predictions.

    :param split_rule: (*string*) The splitting rule.
    :param max_depth: (*int*) the maximum depth of the tree.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param node_size: (*int*) the minimum size of leaf nodes.
    '''
    
    def __init__(self, split_rule='gini', max_depth=20, max_nodes=0, node_size=5):
        super(DecisionTree, self).__init__()

        self._split_rule = SplitRule.valueOf(split_rule.upper())
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._node_size = node_size
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
        formula = Formula.lhs("class")
        if self._max_nodes == 0:
            self._max_nodes = df.size() / 5
        self._model = JDecisionTree.fit(formula, df, self._split_rule, self._max_depth,
            self._max_nodes, self._node_size)

    def predict(self, x):
        df = SmileUtil.toDataFrame(x.asarray())
        r = self._model.predict(df)
        return np.array(r)

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