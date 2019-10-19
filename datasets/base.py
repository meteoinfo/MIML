import mipylib.numeric as np
import csv
import os
from ..utils import Bunch

def get_data_home(data_home=None):
    """Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data dir is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str | None
        The path to scikit-learn data dir.
    """
    module_path = os.path.dirname(__file__)
    data_home = os.path.join(module_path, 'data')
    return data_home

def load_data(module_path, data_file_name):
    """Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    module_path : string
        The module path.

    data_file_name : string
        Name of csv file to be loaded from
        module_path/data/data_file_name. For example 'wine_data.csv'.

    Returns
    -------
    data : Numpy array
        A 2D array with each row representing one sample and each column
        representing the features of a given sample.

    target : Numpy array
        A 1D array holding target variables for all the samples in `data.
        For example target[0] is the target varible for data[0].

    target_names : Numpy array
        A 1D array containing the names of the classifications. For example
        target_names[0] is the name of the target[0] class.
    """
    with open(os.path.join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.dtype.int)

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.dtype.float64)
            target[i] = np.asarray(ir[-1], dtype=np.dtype.int)

    return data, target, target_names
    
def load_iris(return_X_y=False):
    """Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification
    dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Read more in the :ref:`User Guide <iris_dataset>`.

    Parameters
    ----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification labels,
        'target_names', the meaning of the labels, 'feature_names', the
        meaning of the features, 'DESCR', the full description of
        the dataset, 'filename', the physical location of
        iris csv dataset (added in version `0.20`).

    (data, target) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.18

    Notes
    -----
        .. versionchanged:: 0.20
            Fixed two wrong data points according to Fisher's paper.
            The new version is the same as in R, but not as in the UCI
            Machine Learning Repository.

    Examples
    --------
    Let's say you are interested in the samples 10, 25, and 50, and want to
    know their class name.

    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> data.target[[10, 25, 50]]
    array([0, 0, 1])
    >>> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    """
    module_path = os.path.dirname(__file__)
    data, target, target_names = load_data(module_path, 'iris.csv')
    iris_csv_filename = os.path.join(module_path, 'data', 'iris.csv')

    with open(os.path.join(module_path, 'descr', 'iris.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['sepal length (cm)', 'sepal width (cm)',
                                'petal length (cm)', 'petal width (cm)'],
                 filename=iris_csv_filename)
                 
###########################################