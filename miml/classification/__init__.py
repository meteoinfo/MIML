"""
The :mod:`miml.classification` module includes utilities classification`.
"""

from .knn import KNearestNeighbor
from .lda import LinearDiscriminantAnalysis
from .fld import FisherLinearDiscriminant
from .qda import QuadraticDiscriminantAnalysis
from .rda import RegularizedDiscriminantAnalysis
from .logistic import LogisticRegression
from .mlp import MLPClassifer
from .rbfnet import RBFNetwork
from .svm import SVM
from .tree import DecisionTree
from .rforest import RandomForest
from .gbt import GradientTreeBoost
from .adaboost import AdaBoost