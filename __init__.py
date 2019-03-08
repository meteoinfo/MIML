import sys
import os

encogpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encog-core-3.4.jar')
if not encogpath in sys.path:
    sys.path.append(encogpath)
    
version = '0.1'

from miml.ffnetwork import FeedforwardNetwork
from miml.mlutil import MinMaxScaler