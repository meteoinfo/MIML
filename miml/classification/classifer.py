from abc import abstractmethod
import mipylib.numeric as np

class Classifer(object):
    '''
    Classification model base class.
    '''
    
    def __init__(self):
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
        r = self._model.predict(x.tojarray('double'))
        return np.array(r)