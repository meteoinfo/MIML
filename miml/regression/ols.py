# -*- coding: utf-8 -*-

from smile.regression import OLS as JOLS
from smile.data.formula import Formula
from org.meteothink.miml.util import SmileUtil

from .regressor import Regressor

class OLS(Regressor):
    '''
    Ordinary least squares. 

    In linear regression, the model specification is that the dependent variable is a linear 
    combination of the parameters (but need not be linear in the independent variables). The 
    residual is the difference between the value of the dependent variable predicted by the model, 
    and the true value of the dependent variable. Ordinary least squares obtains parameter 
    estimates that minimize the sum of squared residuals, SSE (also denoted RSS).

    :param method: (*string*) the fitting method ['qr' | 'svd'].
    :param stderr: (*bool*) if true, compute the estimated standard errors of the estimate of parameters.
    :param recursive: (*bool*) if true, the return model supports recursive least squares.
    '''
    
    def __init__(self, method='qr', stderr=True, recursive=True):
        super(OLS, self).__init__()
        
        self.method = method
        self.stderr = stderr
        self.recursive = recursive
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        df = SmileUtil.toDataFrame(x.asarray(), y.asarray())
        formula = Formula.lhs("class")
        self._model = JOLS.fit(formula, df, self.method, self.stderr, self.recursive)
        
        
##################################################