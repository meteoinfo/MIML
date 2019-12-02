"""
The :mod:`miml.regression` module includes utilities regression`.
"""

from .ols import OLS
from .ridge import RidgeRegression
from .lasso import LASSO
from .rbfnet import RBFNetwork
from .svr import SVR
from .tree import RegressionTree
from .rforest import RandomForest
from .gbt import GradientTreeBoost
from .gpr import GaussianProcessRegression