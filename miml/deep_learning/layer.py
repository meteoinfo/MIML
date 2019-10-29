
from org.deeplearning4j.nn.conf.layers import DenseLayer
from org.deeplearning4j.nn.conf.layers import OutputLayer
from org.nd4j.linalg.activations import Activation
from org.nd4j.linalg.lossfunctions import LossFunctions

class Dense(object):
    '''
    Dense layer.

    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    '''

    def __init__(self, nin=2, nout=10, activation='relu'):
        self.nin = nin
        self.nout = nout
        self.activation = Activation.valueOf(activation.upper())
        self._layer = DenseLayer.Builder().nIn(nin).nOut(nout) \
                      .activation(self.activation).build()

    def _get_activation(self, name):
        '''
        Get activation.

        :param name: (*string*) Activation name
        '''
        return Activation.valueOf(name.lower())

class Output(object):
    '''
    Output layer.

    :param loss: (*string*) Loss function name.
    :param nin: (*int*) In node number.
    :param nout: (*int*) Out node number.
    :param activation: (*string*) Activation name.
    '''

    def __init__(self, loss='negativeloglikelihood', nin=10, nout=2, activation='softmax'):
        self.loss = LossFunctions.LossFunction.valueOf(loss.upper())
        self.nin = nin
        self.nout = nout
        self.activation = Activation.valueOf(activation.upper())
        self._layer = OutputLayer.Builder().nIn(nin).nOut(nout) \
                      .activation(self.activation).build()