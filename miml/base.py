"""Base classes for all estimators."""

from org.python.util import PythonObjectInputStream
from java import io

class BaseEstimator(object):
    """
    Base class for all estimators
    """
    def __init__(self):
        self.estimator_type = 'None'
        self._model = None

    def dump(self, fn):
        """
        Save model to file.

        Parameters
        ----------
        fn : string
             Output file name.
        """
        outs = io.ObjectOutputStream(io.FileOutputStream(fn))
        outs.writeObject(self)
        outs.close()

    @staticmethod
    def load(fn):
        """
        Load model from file.

        :param fn: (*string*) Input file name.

        :return: The loaded model.
        """
        ins = PythonObjectInputStream(io.FileInputStream(fn))
        x = ins.readObject()
        ins.close()
        return x

class TransformerMixin:
    """Mixin class for all transformers in MIML."""

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.
        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y :  array-like of shape (n_samples,) or (n_samples, n_outputs), \
                default=None
            Target values (None for unsupervised transformations).
        **fit_params : dict
            Additional fit parameters.
        Returns
        -------
        X_new : ndarray array of shape (n_samples, n_features_new)
            Transformed array.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X)

def is_classifier(estimator):
    """Returns True if the given estimator is (probably) a classifier.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a classifier and False otherwise.
    """
    return getattr(estimator, "estimator_type", None) == "classifier"


def is_regressor(estimator):
    """Returns True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.
    """
    return getattr(estimator, "estimator_type", None) == "regressor"

def is_cluster(estimator):
    """Returns True if the given estimator is (probably) a cluster.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a cluster and False otherwise.
    """
    return getattr(estimator, "estimator_type", None) == "cluster"

def is_outlier_detector(estimator):
    """Returns True if the given estimator is (probably) an outlier detector.

    Parameters
    ----------
    estimator : object
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an outlier detector and False otherwise.
    """
    return getattr(estimator, "estimator_type", None) == "outlier_detector"