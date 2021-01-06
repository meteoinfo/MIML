"""
The :mod:`miml.deep_learning` module includes utilities deep learning`.
"""

from .network import Network, InputType
from .computation_graph import ComputationGraph
from .layer import DenseLayer, OutputLayer, ConvolutionLayer, SubsamplingLayer
from .loss_function import LossFunction
from .updater import Adam, Nadam, AdaGrad, Nesterovs, Sgd
import datasets_

from org.nd4j.linalg.activations import Activation
from org.deeplearning4j.nn.weights import WeightInit
from org.deeplearning4j.nn.conf.distribution import UniformDistribution
from org.deeplearning4j.nn.api import OptimizationAlgorithm as Optimizer