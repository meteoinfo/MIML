
from org.meteothink.miml.nd4j import Nd4jUtil
import mipylib.numeric as np

def to_matrix(a, n):
    '''
    Convert one dimension labels to matrix labels.
    :param a: Input one dimension labels
    :param n: Number of output
    :return: Matrix labels
    '''
    r = Nd4jUtil.toMatrix(a._array, n)
    return np.array(r)