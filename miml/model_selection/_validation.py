"""
The :mod:`sklearn.model_selection._validation` module includes classes and
functions to validate the model.

Ported from scikit-learn
"""

from ._split import KFold
from ..base import is_classifier, is_regressor
from ..metrics import accuracy_score, r2_score
import mipylib.numeric as np

__all__ = ['cross_val_score']

def cross_val_score(estimator, X, y=None, cv=3):
    '''
    Evaluate metric(s) by cross-validation

    :param estimator: estimator object implementing 'fit'
        The object to use to fit the data.
    :param X: (*array_like*) The data to fit.
    :param y: (*array_like*) The target variable to try to predict in the case of
        supervised learning.
    :param cv: (*int*) cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross validation,
        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

    :return: dict of float arrays of shape=(n_splits,) Array of scores of the estimator for each
        run of the cross validation.
    '''
    kf = KFold(n_splits=cv)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        if is_classifier(estimator):
            scores.append(accuracy_score(y_test, y_pred))
        elif is_regressor(estimator):
            scores.append(r2_score(y_test, y_pred))

    return np.array(scores)