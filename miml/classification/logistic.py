# -*- coding: utf-8 -*-

from smile.classification import LogisticRegression as JLogisticRegression

from .classifer import Classifer

class LogisticRegression(Classifer):
    '''
    Logistic Regression

    Logistic regression (logit model) is a generalized linear model used for binomial regression. 
    Logistic regression applies maximum likelihood estimation after transforming the dependent into 
    a logit variable. A logit is the natural log of the odds of the dependent equaling a certain 
    value or not (usually 1 in binary logistic models, the highest value in multinomial models). 
    In this way, logistic regression estimates the odds of a certain event (value) occurring.

    :param L: (*float*) Lambda - λ > 0 gives a "regularized" estimate of linear weights which often 
        has superior generalization performance, especially when the dimensionality is high.
    :param tol: (*float*) The tolerance for stopping iterations.
    :param max_iter: (*int*) Maximum number of iterations taken for the solvers to converge.   
    '''
    
    def __init__(self, L=0., tol=1e-5, max_iter=500):
        super(LogisticRegression, self).__init__()
        
        self.L = L
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        super(LogisticRegression, self).fit(x, y)
        self._model = JLogisticRegression.fit(x.tojarray('double'), y.tojarray('int'),
            self.L, self.tol, self.max_iter)
        
        
##################################################