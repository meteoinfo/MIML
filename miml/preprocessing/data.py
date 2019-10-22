import mipylib.numeric as np
import numbers

def _handle_zeros_in_scale(scale):
    ''' Makes sure that whenever scale is zero, we handle it correctly.
    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if isinstance(scale, np.NDArray):
        scale[scale == 0.0] = 1.0
        return scale
    else:
        if scale == .0:
            scale = 1.
        return scale

# MinMaxScaler class        
class MinMaxScaler(object):
    '''
    Transforms features by scaling each feature to a given range.
    
    The transformation is given by::
    
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        
        X_scaled = X_std * (max - min) + min
        
    where min, max = feature_range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1).
        Desired range of transformed data.
    '''
    
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
    
    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_
    
    def fit(self, X):
        '''
        Compute the minimum and maximum to be used for later scaling.
        
        :param X: (*array_like*) shape [n_samples, n_features]. The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        '''
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)
    
    def partial_fit(self, X):
        '''
        Compute the minimum and maximum to be used for later scaling.
        
        :param X: (*array_like*) shape [n_samples, n_features]. The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        '''
        if isinstance(X, (list, tuple)):
            X = np.array(X)
            
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError("Minimum of desired feature range must be smaller"
                             " than maximum. Got %s." % str(feature_range))
                             
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)
        
        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next steps
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = ((feature_range[1] - feature_range[0]) /
                       _handle_zeros_in_scale(data_range))
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self
        
    def transform(self, X):
        """
        Fit and scaling features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        if isinstance(X, numbers.Number):
            X = np.array([X])
        elif isinstance(X, (list, tuple)):
            X = np.array(X)
            
        X *= self.scale_
        X += self.min_
        return X
        
    def fit_transform(self, X):
        """Scaling features of X according to feature_range.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        return self.fit(X).transform(X)        

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Input data that will be transformed. It cannot be sparse.
        """
        X -= self.min_
        X /= self.scale_
        return X