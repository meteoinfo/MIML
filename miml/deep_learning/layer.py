
from org.deeplearning4j.nn.conf.layers import DenseLayer
from org.deeplearning4j.nn.conf.layers import OutputLayer
from org.nd4j.linalg.activations import Activation

import network_util

class Dense(object):
    '''
    Dense layer.

    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    '''

    def __init__(self, nin=2, nout=10, activation='relu', weight_init=None):
        self.nin = nin
        self.nout = nout
        self.activation = Activation.valueOf(activation.upper())
        self.weight_init = weight_init
        conf = DenseLayer.Builder().nIn(nin).nOut(nout) \
                .activation(self.activation)
        if not self.weight_init is None:
            conf.weightInit(network_util.get_weight_init(self.weight_init))
        self._layer = conf.build()

class Output(object):
    '''
    Output layer.

    :param loss: (*string*) Loss function name.
    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    '''

    def __init__(self, loss='negativeloglikelihood', nin=None, nout=2, activation='softmax',
                 weight_init=None, **kwargs):
        self.loss = network_util.get_loss_function(loss, **kwargs)
        self.nin = nin
        self.nout = nout
        self.activation = Activation.valueOf(activation.upper())
        self.weight_init = weight_init
        conf = OutputLayer.Builder(self.loss)
        if not self.nin is None:
            conf.nIn(self.nin)
        conf.nOut(self.nout).activation(self.activation)
        if not self.weight_init is None:
            conf.weightInit(network_util.get_weight_init(self.weight_init))
        self._layer = conf.build()