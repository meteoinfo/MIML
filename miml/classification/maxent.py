# -*- coding: utf-8 -*-

from smile.classification import Maxent as JMaxent

from .classifer import Classifer

class Maxent(Classifer):
    '''
    Maximum Entropy Classifier.

    Maximum entropy is a technique for learning probability distributions from data. In maximum
    entropy models, the observed data itself is assumed to be the testable information. Maximum
    entropy models don't assume anything about the probability distribution other than what have
    been observed and always choose the most uniform distribution subject to the observed
    constraints.

    Basically, maximum entropy classifier is another name of multinomial logistic regression applied
    to categorical independent variables, which are converted to binary dummy variables. Maximum
    entropy models are widely used in natural language processing. Here, we provide an implementation
    which assumes that binary features are stored in a sparse array, of which entries are the indices
    of nonzero features.

    :param p: (*int*) the dimension of feature space.
    :param L: (*float*) Lambda - λ > 0 gives a "regularized" estimate of linear weights which often
        has superior generalization performance, especially when the dimensionality is high.
    :param tol: (*float*) The tolerance for stopping iterations.
    :param max_iter: (*int*) Maximum number of iterations taken for the solvers to converge.
    '''

    def __init__(self, p, L=0., tol=1e-5, max_iter=500):
        super(Maxent, self).__init__()

        self._p = p
        self._L = L
        self._tol = tol
        self._max_iter = max_iter

    def fit(self, x, y):
        """
        Learn from input data and labels.

        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        self._model = JMaxent.fit(self._p, x.tojarray('int'), y.tojarray('int'),
                                              self._L, self._tol, self._max_iter)


##################################################