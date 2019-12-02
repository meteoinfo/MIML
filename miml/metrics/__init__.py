"""
The :mod:`miml.metrics` module includes score functions, performance metrics
and pairwise metrics and distance computations.

Forked from scikit-learn
"""

from .classification import accuracy_score

from .regression import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error

__all__ = [
    'accuracy_score',
    'r2_score',
    'mean_absolute_error',
    'mean_squared_error',
    'mean_squared_log_error',
    'median_absolute_error'
]