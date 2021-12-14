import sys
import os

# base_dir = os.path.dirname(os.path.abspath(__file__))
# lib_dir = os.path.join(base_dir, 'lib')
#
# fns = os.listdir(lib_dir)
# for fn in fns:
#     fpath = os.path.join(lib_dir, fn)
#     if os.path.isfile(fpath) and fn.endswith('.jar'):
#         if not fpath in sys.path:
#             sys.path.append(fpath)
    
__version__ = '1.0'

from .base import BaseEstimator

__all__ = ['__version__','BaseEstimator','classification','cluster','datasets','neural_network','preprocessing',
           'regression','utils','deep_learning','metrics','model_selection','math','decomposition']