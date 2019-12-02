from org.meteothink.miml.data.parser import ArffParser

from .dataset import AttributeDataset

def load(fn, ridx=-1):
    '''
    Read Weka ARFF (attribute relation file format) data file.
    
    :param fn: (*string*) ARFF data file name.
    :param ridx: (*string*) The column index (starting at 0) of dependent/response variable.
    
    :returns: (*AttributeDataset*) AttributeDataset.
    '''
    r = ArffParser().setResponseIndex(ridx).parse(fn)
    return AttributeDataset(r)