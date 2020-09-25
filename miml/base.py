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