# -*- coding: utf-8 -*-
from org.encog.engine.network.activation import ActivationSigmoid, ActivationBiPolar, \
    ActivationBipolarSteepenedSigmoid, ActivationClippedLinear, ActivationCompetitive, \
    ActivationElliott, ActivationElliottSymmetric, ActivationGaussian, ActivationLOG, \
    ActivationLinear, ActivationRamp, ActivationReLU, ActivationSIN, ActivationSoftMax, \
    ActivationSteepenedSigmoid, ActivationStep, ActivationTANH
    
def activation_function(act):
    '''
    Get activation function.
    
    :param act: (*string*) Activation function name. ['sigmoid' | 'bipolar' | 'bipolarss' | 
        'clippedlinear' | 'competitive' | 'elliott' | 'elliotts' | 'gaussian' | 'log' |
        'linear' | 'ramp' | 'relu' | 'sin' | 'softmax' | 'ssigmoid' | 'step' | 'tanh'].
    
    :returns: Activation function
    '''
    if act is None:
        return None
        
    act = act.lower()
    if act == 'sigmoid':
        return ActivationSigmoid()
    elif act == 'bipolar':
        return ActivationBiPolar()
    elif act == 'bipolarss':
        return ActivationBipolarSteepenedSigmoid()
    elif act == 'clippedlinear':
        return ActivationClippedLinear()
    elif act == 'competitive':
        return ActivationCompetitive()
    elif act == 'elliott':
        return ActivationElliott()
    elif act == 'elliotts':
        return ActivationElliottSymmetric()
    elif act == 'gaussian':
        return ActivationGaussian()
    elif act == 'log':
        return ActivationLOG()
    elif act == 'linear':
        return ActivationLinear()
    elif act == 'ramp':
        return ActivationRamp()
    elif act == 'relu':
        return ActivationReLU()
    elif act == 'sin':
        return ActivationSIN()
    elif act == 'softmax':
        return ActivationSoftMax()
    elif act == 'ssigmoid':
        return ActivationSteepenedSigmoid()
    elif act == 'step':
        return ActivationStep()
    elif act == 'tanh':
        return ActivationTANH()
    else:
        raise ValueError('Not valid activation function name: ' + act)

#################################################################