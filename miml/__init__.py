import sys
import os
from org.meteoinfo.util import PathUtil

base_dir = os.path.dirname(os.path.abspath(__file__))
native_lib_dir = os.path.join(base_dir, 'native_lib')
if os.path.exists(native_lib_dir):
    PathUtil.addLibraryPath(native_lib_dir)

lib_dir = os.path.join(base_dir, 'lib')
fns = os.listdir(lib_dir)
for fn in fns:
    fpath = os.path.join(lib_dir, fn)
    if os.path.isfile(fpath) and fn.endswith('.jar'):
        if not fpath in sys.path:
            sys.path.append(fpath)
    
__version__ = '0.7'

from .base import BaseEstimator

__all__ = ['BaseEstimator','classification','cluster','datasets','neural_network','preprocessing',
           'regression','utils','deep_learning','metrics','model_selection']