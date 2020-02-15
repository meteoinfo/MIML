# -*- coding: utf-8 -*-

from smile.classification import RandomForest as JRandomForest
from smile.base.cart import SplitRule
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

import mipylib.numeric as np
import math
from .classifer import Classifer

class RandomForest(Classifer):
    '''
    Random forest for classification. 

    Random forest is an ensemble classifier that consists of many decision trees and outputs the 
    majority vote of individual trees. The method combines bagging idea and the random selection 
    of features.

    :param ntrees: (*int*) The number of trees.
    :param max_depth: (*int*) the maximum depth of the tree.
    :param max_nodes: (*int*) The maximum number of leaf nodes in the tree.
    :param node_size: (*int*) Number of instances in a node below which the tree will not split.
    :param mtry: (*int*) the number of random selected features to be used to determine the 
        decision at a node of the tree. floor(sqrt(dim)) seems to give generally good performance, 
        where dim is the number of variables.
    :param sub_sample: (*float*) The sampling rate for training tree. 1.0 means sampling with 
        replacement. < 1.0 means sampling without replacement.
    :param split_rule: (*string*) The splitting rule. 
    :param class_weight: (*array*) Priors of the classes.
    '''
    
    def __init__(self, ntrees=500, max_depth=20, max_nodes=0, node_size=5,
            mtry=0, sub_sample=1.0, split_rule='gini', class_weight=None):
        super(RandomForest, self).__init__()

        self.ntrees = ntrees
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.node_size = node_size
        self.mtry = mtry
        self.sub_sample = sub_sample
        self.split_rule = SplitRule.valueOf(split_rule.upper())
        self.class_weight = class_weight
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        super(RandomForest, self).fit(x, y)
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
        formula = Formula.lhs("class")
        if self.max_nodes == 0:
            self.max_nodes = df.size() / 5
        if self.mtry == 0:
            self.mtry = int(math.floor(math.sqrt(df.ncols() - 1)))
        self._model = JRandomForest.fit(formula, df, self.ntrees, self.mtry, self.split_rule,
            self.max_depth, self.max_nodes, self.node_size, self.sub_sample)

    def predict(self, x):
        x = np.atleast_2d(x)
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