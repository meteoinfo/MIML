from abc import abstractmethod
import mipylib.numeric as np
from ..metrics import accuracy_score
from ..base import BaseEstimator

from org.meteothink.miml.util import SmileUtil

class Classifer(BaseEstimator):
    '''
    Classification model base class.
    '''
    
    def __init__(self):
        self.estimator_type = 'classifier'
        self._model = None
        self.nclass = 2    #the number of classes
    
    def __str__(self):
        _str = self.__class__.__name__
        if not self._model is None:
            _str = _str + '\n' + self._model.toString()
        return _str
        
    def __repr__(self):
        return self.__str__()
        
    @abstractmethod
    def fit(self, x, y):
        '''
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        '''
        self.nclass = int(y.max()) + 1
    
    def predict(self, x):
        """
        Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        x = np.atleast_2d(x)
        r = self._model.predict(x.tojarray('double'))
        return np.array(r)

    def predict_proba(self, x):
        """
        Probability estimates.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : the probability of the sample for each class in the model.
        """
        x = np.atleast_2d(x)
        r = SmileUtil.predictProbability(self._model, x.tojarray('double'), self.nclass)
        return np.array(r)

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)