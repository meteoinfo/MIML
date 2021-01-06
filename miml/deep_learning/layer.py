
from org.deeplearning4j.nn.conf.layers import DenseLayer as JDenseLayer
from org.deeplearning4j.nn.conf.layers import OutputLayer as JOutputLayer
from org.deeplearning4j.nn.conf.layers import ConvolutionLayer as JConvolutionLayer
from org.deeplearning4j.nn.conf.layers import SubsamplingLayer as JSubsamplingLayer
from org.nd4j.linalg.activations import Activation

import network_util
from .loss_function import LossFunction

class DenseLayer(object):
    '''
    Dense layer.

    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    :param weight_init: (*float*) Init weight.
    :param dropout: (*float*) Dropout connection.
    :param bias_init: (*int*) Initialize the bias.
    '''

    def __init__(self, nin=2, nout=10, activation=Activation.RELU, weight_init=None, dropout=None,
                 bias_init=None):
        self.nin = nin
        self.nout = nout
        self.activation = activation
        self.weight_init = weight_init
        self.dropout = dropout
        self.bias_init = bias_init
        conf = JDenseLayer.Builder().nIn(nin).nOut(nout) \
                .activation(self.activation)
        if not self.weight_init is None:
            conf.weightInit(self.weight_init)
        if not self.dropout is None:
            conf.dropOut(self.dropout)
        if not self.bias_init is None:
            conf.biasInit(self.bias_init)
        self._layer = conf.build()

class OutputLayer(object):
    '''
    Output layer.

    :param loss: (*string*) Loss function name.
    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    '''

    def __init__(self, loss=LossFunction.NEGATIVELOGLIKELIHOOD, nin=None, nout=2, activation=Activation.SOFTMAX,
                 weight_init=None, **kwargs):
        self.loss = loss
        self.nin = nin
        self.nout = nout
        self.activation = activation
        self.weight_init = weight_init
        conf = JOutputLayer.Builder(self.loss)
        if not self.nin is None:
            conf.nIn(self.nin)
        conf.nOut(self.nout).activation(self.activation)
        if not self.weight_init is None:
            conf.weightInit(self.weight_init)
        self._layer = conf.build()

class ConvolutionLayer(object):
    '''
    Convolution layer.

    :param kernel_size: (*tuple*) Kernel size.
    :param stride: (*tuple*) Stride.
    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    '''

    def __init__(self, kernel_size=(3,3), stride=(1,1), nin=None, nout=2, activation=Activation.IDENTITY):
        self.kernel_size = kernel_size
        self.stride = stride
        self.nin = nin
        self.nout = nout
        self.activation = activation
        conf = JConvolutionLayer.Builder(self.kernel_size)
        if not self.nin is None:
            conf.nIn(self.nin)
        conf.stride(self.stride)
        if not self.nout is None:
            conf.nOut(self.nout)
        conf.activation(self.activation)
        self._layer = conf.build()

class SubsamplingLayer(object):
    '''
    Subsampling layer also referred to as pooling in convolution neural nets.

    Supports the following pooling types: MAX, AVG, SUM, PNORM.

    :param pooling_type: (*string*) Pooling type.
    :param kernel_size: (*tuple*) Kernel size.
    :param stride: (*tuple*) Stride.
    '''

    def __init__(self, pooling_type='max', kernel_size=(3,3), stride=(1,1)):
        self.pooling_type = JSubsamplingLayer.PoolingType.valueOf(pooling_type.upper())
        self.kernel_size = kernel_size
        self.stride = stride
        conf = JSubsamplingLayer.Builder(self.pooling_type)
        conf.kernelSize(self.kernel_size)
        conf.stride(self.stride)
        self._layer = conf.build()