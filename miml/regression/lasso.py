# -*- coding: utf-8 -*-

from smile.regression import LASSO as JLASSO
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

from .regressor import Regressor

class LASSO(Regressor):
    '''
    Lasso (least absolute shrinkage and selection operator) regression. 

    The Lasso is a shrinkage and selection method for linear regression. It minimizes the usual sum 
    of squared errors, with a bound on the sum of the absolute values of the coefficients (i.e. 
    L1-regularized). It has connections to soft-thresholding of wavelet coefficients, forward 
    stage-wise regression, and boosting methods.

    :param L: (*float*) The shrinkage/regularization parameter. Large lambda means more shrinkage. 
        Choosing an appropriate value of lambda is important, and also difficult.
    :param tol: (*float*) The tolerance for stopping iterations (relative target duality gap).
    :param max_iter: (*int*) The maximum number of IPM (Newton) iterations.
    '''
    
    def __init__(self, L=None, tol=1e-3, max_iter=5000):
        super(LASSO, self).__init__()
        
        self._L = L
        self._tol = tol
        self._max_iter = max_iter
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
        formula = Formula.lhs("class")
        self._model = JLASSO.fit(formula, df, self._L, self._tol, self._max_iter)
        
        
##################################################