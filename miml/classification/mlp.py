# -*- coding: utf-8 -*-

from smile.classification import MLP
from smile.base.mlp import Layer, OutputFunction

from .classifer import Classifer

class MLPClassifer(Classifer):
    '''
    Multilayer perceptron neural network for classification. 

    An MLP consists of several layers of nodes, interconnected through weighted acyclic arcs from 
    each preceding layer to the following, without lateral or feedback connections. Each node 
    calculates a transformed weighted linear combination of its inputs (output activations from 
    the preceding layer), with one of the weights acting as a trainable bias connected to a 
    constant input. The transformation, called activation function, is a bounded non-decreasing 
    (non-linear) function, such as the sigmoid functions (ranges from 0 to 1). Another popular 
    activation function is hyperbolic tangent which is actually equivalent to the sigmoid function 
    in shape but ranges from -1 to 1. More specialized activation functions include radial basis 
    functions which are used in RBF networks.

    :param hidden_layer_sizes: (*list of int*) Length = n_layers - 2, default [100]
        The ith element represents the number of neurons in the ith
        hidden layer.
    :param error: (*string*) The error function ['least_mean_squares' | 'cross_entropy'].
    :param activation: (*string*) The activation function of output layer ['linear' | 
        'logistic_sigmoid' | 'softmax']. 
    :param epochs: (*int*) the number of epochs of stochastic learning.
    :param eta: (*float*) the learning rate.
    :param alpha: (*float*) the momentum factor.
    :param L: (*float*) the weight decay for regularization.
    '''
    
    def __init__(self, hidden_layer_sizes=[100], error='cross_entropy', activation='logistic_sigmoid',
            epochs=10, eta=0.1, alpha=0., L=0.):
        super(MLPClassifer, self).__init__()
        
        self._hidden_layer_sizes = hidden_layer_sizes
        self._error = error
        self._activation = activation
        self._epochs = epochs
        self._eta = eta
        self._alpha = alpha
        self._L = L
    
    def fit(self, x, y):
        """
        Learn from input data and labels.
        
        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """ 
        k = int(y.max()) + 1
        layers = []
        for num_unit in self._hidden_layer_sizes:
            layers.append(Layer.sigmoid(num_unit))
        if k == 2:
            layers.append(Layer.mle(1, OutputFunction.SIGMOID))
        else:
            layers.append(Layer.mle(k, OutputFunction.SOFTMAX))
        p = x.shape[1]
        self._model = MLP(p, layers)
        self._model.setLearningRate(self._eta)
        self._model.setMomentum(self._alpha)
        self._model.setWeightDecay(self._L)
        
        x = x.tojarray('double')
        y = y.tojarray('int')
        for i in range(self._epochs):
            self._model.update(x, y)
        
        
##################################################