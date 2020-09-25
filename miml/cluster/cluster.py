from ..base import BaseEstimator

class Cluster(BaseEstimator):
    '''
    Clustering model base class.
    '''
    
    def __init__(self):
        self.estimator_type = 'cluster'
        self._model = None
    
    def __str__(self):
        _str = self.__class__.__name__
        if not self._model is None:
            _str = _str + '\n' + self._model.toString()
        return _str
        
    def __repr__(self):
        return self.__str__()
        
    def fit(self, x):
        """
        Fitting data.
        
        :param x: (*array*) Input data.
        
        :returns: self.
        """
        pass
    
    def fit_predict(self, x):
        """
        Fitting and cluster data.

        :param x: (*array*) Input data.
        
        :returns: (*array*) The cluster labels.
        """
        pass