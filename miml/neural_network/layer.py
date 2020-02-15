from org.encog.neural.networks.layers import BasicLayer
import mlutil

class Layer(object):

    def __init__(self, size, actname='sigmoid', bias_neuron=True, dropout=0):
        '''
        Layer initialize
        
        :param size: (*int*) Neuron number.
        :param actname: (*string*) Activation function name ['sigmoid' | 'bipolar' | 'bipolarss' | 
            'clippedlinear' | 'competitive' | 'elliott' | 'elliotts' | 'gaussian' | 'log' |
            'linear' | 'ramp' | 'relu' | 'sin' | 'softmax' | 'ssigmoid' | 'step' | 'tanh'].
        :param bias_neuron: (*bool*) Using bias neuron or not. Default is True.
        :param dropout: (*float*) Dropout ratio
        '''
        self.actname = actname
        activation = mlutil.activation_function(actname)
        self.activation = activation
        self.bias_neuron = bias_neuron
        self.dropout = dropout
        self._layer = BasicLayer(activation, bias_neuron, size, dropout)
        
    def __str__(self):
        return '(Layer: size=%i, activation=%s)' % (self._layer.getCount(), self.actname)
        
    def __repr__(self):
        return self.__str__()
            
    def get_size(self):
        '''
        Get neural count
        '''
        return self._layer.getCount()
        
    def set_size(self, size):
        '''
        Set neural count.
        
        :param size: (*int*) Neuron count.
        '''
        self._layer = BasicLayer(self.activation, self.bias_neuron, size, self.dropout)
        
    def set_activation(self, actname):
        '''
        Set activation function.
        
        :param actname: (*string*) Activation function name.
        '''
        activation = mlutil.activation_function(actname)
        self._layer.setActivation(activation)
        self.activation = activation
        self.actname = actname