
from smile.math.kernel import GaussianKernel
from smile.math.distance import EuclideanDistance, ChebyshevDistance, EditDistance, HammingDistance, \
    JensenShannonDistance, LeeDistance, MahalanobisDistance, ManhattanDistance, MinkowskiDistance, \
    SparseChebyshevDistance, SparseEuclideanDistance, SparseManhattanDistance, \
    SparseMinkowskiDistance
from smile.data import NumericAttribute
from org.meteoinfo.math.distance import EuclideanDistance as MIEuclideanDistance
import mipylib.numeric as np

def get_kernel(key, **kwargs):
    '''
    Get kernel object.
    
    :param key: (*string*) Kernel key.
    
    :returns: Kernel object.
    '''
    key = key.lower()
    if key == 'gaussian':
        sigma = kwargs.pop('sigma', 1.0)
        return GaussianKernel(sigma)
        
def get_distance(key):
    '''
    Get distance object.
    
    :param key: (*string*) Distance key.
    
    :returns: Distance object.
    '''
    key = key.lower()
    if key == 'euclidean':
        return EuclideanDistance()
    elif key == 'chebyshev':
        return ChebyshevDistance()
    elif key == 'edit':
        return EditDistance()
    elif key == 'hamming':
        return HammingDistance()
    elif key == 'jensen_shannon':
        return JensenShannonDistance()
    elif key == 'lee':
        return LeeDistance()
    elif key == 'mahalanobis':
        return MahalanobisDistance()
    elif key == 'manhattan':
        return ManhattanDistance()
    elif key == 'minkowski':
        return MinkowskiDistance()
    elif key == 'sparse_chebyshev':
        return SparseChebyshevDistance()
    elif key == 'sparse_euclidean':
        return SparseEuclideanDistance()
    elif key == 'sparse_manhattan':
        return SparseManhattanDistance()
    elif key == 'sparse_minkowski':
        return SparseMinkowskiDistance()
    else:
        return EuclideanDistance()
        
def numeric_attributes(n):
    '''
    Get numeric attributes
    
    :param n: (*int*) Attribute number.
    
    :returns: Numeric attributes
    '''
    attrs = []
    for i in range(n):
        attrs.append(NumericAttribute(str(i)))
    return attrs
    
def proximity(data, dist, half=True):
    '''
    Returns the proximity matrix of a dataset for given distance function.
    
    :param data: (*array*) The data set.
    :param dist: (*string*) The distance function.
    :param half: (*boolean*) If true, only the lower half of matrix is allocated to save space.
    
    :returns: (*array*) The lower half of proximity matrix.
    '''
    dist = get_distance(dist)
    n = len(data)
    if half:
        d = []
        for i in range(n):
            d.append(np.full(i + 1, np.nan))
            for j in range(i):
                d[i][j] = dist.d(data[i], data[j])
    else:
        d = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i):
                d[i,j] = dist.d(data[i], data[j])
                d[j,i] = d[i,j]
    return np.array(d)
    
def pdist(data, half=False):
    '''
    Returns the pairwise Euclidean distance matrix.
    
    :param data: (*array*) The data set.
    :param half: (*boolean*) If true, only the lower half of matrix is allocated to save space.
    
    :returns: (*array*) The lower half of proximity matrix.
    '''
    eudis = MIEuclideanDistance()
    r = eudis.proximity(data, False)
    return np.array(r)