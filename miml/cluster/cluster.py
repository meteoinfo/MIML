from ..base import BaseEstimator
import mipylib.numeric as np
from org.meteothink.miml.util import SmileUtil

class Cluster(BaseEstimator):
    '''
    Clustering model base class.
    '''
    
    def __init__(self):
        self.estimator_type = 'cluster'
        self._model = None
        self.labels_ = None
    
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
        return self.fit(x).labels_

    def predict(self, x):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        x : array-like, new data to predict.

        Returns
        -------
        y : array, index of the cluster each sample belongs to.
        """
        x = np.array(x)
        if x.ndim == 2:
            r = SmileUtil.clusterPredict(self._model, x.tojarray('double'))
        else:
            r = self._model.predict(x.tojarray('double'))
        return np.array(r)