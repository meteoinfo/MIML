# -*- coding: utf-8 -*-
import os
from org.encog import Encog
from org.encog.ml.data import MLData
from org.encog.ml.data import MLDataPair
from org.encog.ml.data import MLDataSet
from org.encog.ml.data.basic import BasicMLDataSet, BasicMLData
from org.encog.neural.networks import BasicNetwork
from org.encog.neural.networks.training.lma import LevenbergMarquardtTraining
from org.encog.neural.networks.training.propagation.resilient import ResilientPropagation
from org.encog.neural.networks.training.propagation.scg import ScaledConjugateGradient
import mipylib.numeric.minum as np
import mlutil
from layer import Layer
import numbers

class FeedforwardNetwork(object):
    
    def __init__(self, hidden_sizes=10, train_fcn='trainlm'):
        '''
        FeedforwardNetwork initialize.
        
        :param hidden_sizes: (*int or list*) Row vector of one or more hidden layer sizes (default = None).
        :param train_fcn: (*string*) Training function (default = 'trainlm').
        '''
        if not isinstance(hidden_sizes, (list, tuple)):
            hidden_sizes = [hidden_sizes]
        self._layers = []
        for hs in hidden_sizes:
            layer = Layer(hs)
            self._layers.append(layer)    
        self.train_fcn = train_fcn
        self.train_epochs = 1000
        self.train_goal = 0.001
        self.train_lr = 0.01
        self.train_mc = 0.9  
        self.network = None
        self.out_activation = 'tanh'
     
    #---- layers property
    def get_layers(self):
        return self._layers
        
    layers = property(get_layers)    
    
    def add_layer(self, size, actname='sigmoid', bias_neuron=True, dropout=0):
        '''
        Add a layer.
        
        :param size: (*int*) Neuron number.
        :param actname: (*string*) Activation function name ['sigmoid' | 'bipolar' | 'bipolarss' | 
            'clippedlinear' | 'competitive' | 'elliott' | 'elliotts' | 'gaussian' | 'log' |
            'linear' | 'ramp' | 'relu' | 'sin' | 'softmax' | 'ssigmoid' | 'step' | 'tanh'].
        :param bias_neuron: (*bool*) Using bias neuron or not. Default is True.
        :param dropout: (*float*) Dropout ratio
        
        :returns: The added layer.
        '''
        layer = Layer(size, actname, bias_neuron, dropout)
        self._layers.append(layer)
    
    def train(self, indata, ideal, isprint=False):
        '''
        Train the neural network.
        
        :param indata: (*array_like*) Input data.
        :param ideal: (*array_like*) Ideal data.
        :param isprint: (*bool*) Print tain step or not. Default is False.
        '''                
        if isinstance(indata, (list, tuple)):
            indata = np.array(indata)
        if isinstance(ideal, (list, tuple)):
            ideal = np.array(ideal)
        if indata.ndim == 1:
            indata = indata.reshape(indata.shape[0], 1)
        if ideal.ndim == 1:
            ideal = ideal.reshape(ideal.shape[0], 1)
        trainset = BasicMLDataSet(indata.tojarray('double'), ideal.tojarray('double'))
        
        self.n_in = indata.shape[1]
        self.n_out = ideal.shape[1]
        network = BasicNetwork()
        network.addLayer(Layer(self.n_in, None)._layer)
        for layer in self._layers:
            network.addLayer(layer._layer)
        network.addLayer(Layer(self.n_out, actname=self.out_activation, bias_neuron=False)._layer)
        network.getStructure().finalizeStructure()
        network.reset()
        self.network = network
        
        trainf = None
        if self.train_fcn == 'trainrp':
            trainf = ResilientPropagation(network, trainset)
        elif self.train_fcn == 'trainlm':
            trainf = LevenbergMarquardtTraining(network, trainset)
        elif self.train_fcn == 'trainscg':
            trainf = ScaledConjugateGradient(network, trainset)
        else:
            raise ValueError('Training function not exist: %s' % self.train_fcn)
        
        if not trainf is None:
            trainf.setThreadCount(0)
            for i in range(self.train_epochs):
                trainf.iteration()
                if isprint:
                    print 'Epochs %i: Error=%.3f' % (i+1, trainf.getError())
                if trainf.getError() < self.train_goal:
                    break
                    
            trainf.finishTraining()
        
    def predict(self, indata):
        '''
        Predict
        
        :param indata: (*array_like*) Input data.
        
        :returns: (*array_like*) Output data.
        '''
        if isinstance(indata, (list, tuple)):
            indata = np.array(indata)
        if indata.ndim == 2:
            n = indata.shape[0]
            if self.n_out == 1:
                r = np.full(n, np.nan)
                for i in range(n):
                    data = indata[i,:]
                    if isinstance(data, numbers.Number):
                        data = np.array([data])
                    data = BasicMLData(data.tojarray('double'))
                    rr = self.network.compute(data)
                    r[i] = rr.getData()[0]
            else:
                r = np.full((n, self.n_out), np.nan)
                for i in range(n):
                    data = indata[i,:]
                    if isinstance(data, numbers.Number):
                        data = np.array([data])
                    data = BasicMLData(data.tojarray('double'))
                    rr = self.network.compute(data)
                    r[i,:] = np.array(rr.getData())
            return r
        else:
            indata = BasicMLData(indata.tojarray('double'))
            r = self.network.compute(indata)
            return np.array(r.getData())           

################################################