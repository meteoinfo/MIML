
from smile.math.kernel import GaussianKernel
from smile.data import NumericAttribute

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