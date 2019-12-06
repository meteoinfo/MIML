# -*- coding: utf-8 -*-

from smile.classification import GradientTreeBoost as JGradientTreeBoost
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

import mipylib.numeric as np
from .classifer import Classifer

class GradientTreeBoost(Classifer):
    '''
    Gradient boosting for classification. 

    Gradient boosting is typically used with decision trees (especially CART regression trees) of 
    a fixed size as base learners. For this special case Friedman proposes a modification to 
    gradient boosting method which improves the quality of fit of each base learner.

    :param ntrees: (*int*) The number of trees.
    :param max_depth: (*int*) the maximum depth of the tree.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param node_size: (*int*) Number of instances in a node below which the tree will not split.
    :param shrinkage: (*float*) The shrinkage parameter in (0, 1] controls the learning rate of 
        procedure.
    :param sub_sample: (*float*) The sampling rate for training tree. 1.0 means sampling with 
        replacement. < 1.0 means sampling without replacement.
    '''
    
    def __init__(self, ntrees=500, max_depth=20, max_nodes=6, node_size=5, shrinkage=0.05,
            sub_sample=0.7):
        super(GradientTreeBoost, self).__init__()

        self._ntrees = ntrees
        self._max_depth = max_depth
        self._max_nodes = max_nodes
        self._node_size = node_size
        self._shrinkage = shrinkage
        self._sub_sample = sub_sample
    
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
        self._model = JGradientTreeBoost.fit(formula, df, self._ntrees, self._max_depth,
            self._max_nodes, self._node_size, self._shrinkage, self._sub_sample)

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