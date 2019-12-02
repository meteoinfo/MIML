"""
The :mod:`miml.datasets` module includes utilities to load datasets,
including methods to load and fetch popular reference datasets. It also
features some artificial data generators.
"""
from .base import load_iris, get_data_home, load_data, load_digits, load_boston
from .arff import load as load_arff

__all__ = [
    'load_arff',
    'load_boston',
    'load_data',
    'load_digits',
    'load_iris',
    'get_data_home',
]