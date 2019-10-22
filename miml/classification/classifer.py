import mipylib.numeric as np

class Classifer(object):
    '''
    Classification model base class.
    '''
    
    def __init__(self):
        self._model = None
    
    def __str__(self):
        return self._model.toString()
        
    def __repr__(self):
        return self._model.toString()
    
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