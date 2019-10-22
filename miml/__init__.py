import sys
import os

libdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
fns = os.listdir(libdir)
for fn in fns:
    fpath = os.path.join(libdir, fn)
    if os.path.isfile(fpath) and fn.endswith('.jar'):
        if not fpath in sys.path:
            sys.path.append(fpath)
    
version = '0.2'