from org.meteothink.miml.data import AttributeDataset as JAttributeDataset
import mipylib.numeric as np

class AttributeDataset(object):
    '''
    Attribute dataset including x, y and some attributes.
    
    :param dataset: (*smile.data.AttributeDataset*) Backend java AttributeDataset object.
    '''

    def __init__(self, dataset):
        self._dataset = dataset
        
    def __str__(self):
        return self._dataset.toString()
        
    def __repr__(self):
        return self._dataset.toString()
        
    def data(self):
        return np.array(self._dataset.data)
        
    @property
    def x(self):
        '''
        Get x data array.
        '''
        return np.array(self._dataset.x())
        
    @property
    def y(self):
        '''
        Get y data array.
        '''
        return np.array(self._dataset.y())