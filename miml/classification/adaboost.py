# -*- coding: utf-8 -*-

from smile.classification import AdaBoost as JAdaBoost
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

import mipylib.numeric as np
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

    :param ntrees: (*int*) The number of trees.
    :param max_depth: (*int*) the maximum depth of the tree.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param node_size: (*int*) Number of instances in a node below which the tree will not split.
    '''
    
    def __init__(self, ntrees=500, max_depth=20, max_nodes=6, node_size=1):
        super(AdaBoost, self).__init__()

        self._ntrees = ntrees
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._node_size = node_size
        
    def fit(self, x, y):
        '''
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        '''
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
        formula = Formula.lhs("class")
        if self._max_nodes == 0:
            self._max_nodes = df.size() / 5
        self._model = JAdaBoost.fit(formula, df, self._ntrees, self._max_depth,
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