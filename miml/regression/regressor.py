from abc import abstractmethod
import mipylib.numeric as np
from ..metrics import r2_score

class Regressor(object):
    '''
    Regression model base class.
    '''
    
    def __init__(self):
        self.estimator_type = 'regressor'
        self._model = None
    
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
        pass        
    
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

    def score(self, X, y, sample_weight=None):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples. For some estimators this may be a
            precomputed kernel matrix instead, shape = (n_samples,
            n_samples_fitted], where n_samples_fitted is the number of
            samples used in the fitting for the estimator.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True values for X.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        return r2_score(y, self.predict(X), sample_weight=sample_weight,
                        multioutput='variance_weighted')