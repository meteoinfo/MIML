# -*- coding: utf-8 -*-

from smile.regression import GradientTreeBoost as JGradientTreeBoost
from smile.base.cart import Loss
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

import mipylib.numeric as np
from .regressor import Regressor

class GradientTreeBoost(Regressor):
    '''
    Gradient boosting for regression. 

    Gradient boosting is typically used with decision trees (especially CART regression trees) of 
    a fixed size as base learners. For this special case Friedman proposes a modification to 
    gradient boosting method which improves the quality of fit of each base learner.

    :param loss: (*string*) Loss function for regression. By default, least absolute deviation is 
        employed for robust regression.
    :param ntrees: (*int*) The number of trees.
    :param max_depth: (*int*) the maximum depth of the tree.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param node_size: (*int*) Number of instances in a node below which the tree will not split.
    :param shrinkage: (*float*) The shrinkage parameter in (0, 1] controls the learning rate of 
        procedure.
    :param sub_sample: (*float*) The sampling rate for training tree. 1.0 means sampling with 
        replacement. < 1.0 means sampling without replacement.
    '''
    
    def __init__(self, loss=Loss.ls(), ntrees=500, max_depth=20,
            max_nodes=6, node_size=5, shrinkage=0.05, sub_sample=0.7):
        super(GradientTreeBoost, self).__init__()

        self.loss = loss
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.node_size = node_size
        self.shrinkage = shrinkage
        self.sub_sample = sub_sample
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
        formula = Formula.lhs("class")
        if isinstance(self.loss, basestring):
            self.loss = Loss.valueOf(self.loss)
        if self.max_nodes == 0:
            self.max_nodes = df.size() / 5
        self._model = JGradientTreeBoost.fit(formula, df, self.loss, self.ntrees,
            self.max_depth, self.max_nodes, self.node_size, self.shrinkage, self.sub_sample)

    def predict(self, x):
        x = np.atleast_2d(x)
        y = np.zeros(len(x))
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
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