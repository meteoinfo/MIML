
class Cluster(object):
    '''
    Clustering model base class.
    '''
    
    def __init__(self):
        self._model = None
    
    def __str__(self):
        return self._model.toString()
        
    def __repr__(self):
        return self._model.toString()
    
    def predict(self, x):
        """
        Cluster a new instance.

        :param x: (*array*) A new instance.
        
        :returns: (*int*) The cluster label.
        """
        r = self._model.predict(x.tojarray('double'))
        return r