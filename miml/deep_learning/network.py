from org.deeplearning4j.nn.conf import NeuralNetConfiguration
from org.deeplearning4j.nn.multilayer import MultiLayerNetwork
from org.deeplearning4j.nn.conf.inputs import InputType
from org.deeplearning4j.optimize.listeners import ScoreIterationListener
from org.nd4j.linalg.api.ndarray import INDArray
from org.nd4j.linalg.indexing import NDArrayIndex
from org.nd4j.evaluation.classification import Evaluation
from org.nd4j.linalg.util import FeatureUtil
from org.nd4j.linalg.activations import Activation
from org.meteothink.miml.nd4j import Nd4jUtil
import mipylib.numeric as np
from .layer import OutputLayer
import network_util

class Network(object):
    '''
    Multiple Layer Neural Network.


    '''

    def __init__(self, seed=123, activation=None, weight_init=None, learning_rate=None, optimizer=None, updater=None,
            bias_init=None, l1=None, l2=None, mini_batch=None, layers=None, **kwargs):
        self.seed = seed
        self.activation = activation
        self.weight_init = weight_init
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.updater = updater
        self.bias_init = bias_init
        self.l1 = l1
        self.l2 = l2
        self.mini_batch = mini_batch
        self.layers = layers
        self.nout = None
        self._model = None
        self.score_iter = None
        self.input_type = kwargs.pop('input_type', None)

    def __str__(self):
        return self._model.summary()

    def __repr__(self):
        return self.__str__()

    def add(self, layer):
        '''
        Add a layer

        :param layer: (*Layer*) The layer.
        '''
        if self.layers is None:
            self.layers = []
        self.layers.append(layer)
        if isinstance(layer, OutputLayer):
            self.nout = layer.nout

    def compile(self, score_iter=10):
        '''
        Build the network.

        :param score_iter: (*int*) Number of parameter updates for printing score
        '''
        confb = NeuralNetConfiguration.Builder() \
            .seed(self.seed)
        if not self.activation is None:
            confb.activation(self.activation)
        if not self.learning_rate is None:
            confb.learningRate(self.learning_rate)
        if not self.optimizer is None:
            confb.optimizationAlgo(self.optimizer)
        if not self.updater is None:
            confb.updater(self.updater)
        if not self.weight_init is None:
            confb.weightInit(self.weight_init)
        if not self.bias_init is None:
            confb.biasInit(self.bias_init)
        if not self.l1 is None:
            confb.l1(self.l1)
        if not self.l2 is None:
            confb.l2(self.l2)
        if not self.mini_batch is None:
            confb.miniBatch(self.mini_batch)
        confb = confb.list()
        for layer in self.layers:
            confb.layer(layer._layer)
        if not self.input_type is None:
            confb.setInputType(self.input_type)
        conf = confb.build()
        self._model = MultiLayerNetwork(conf)
        self._model.init()
        self.score_iter = score_iter
        self._model.setListeners(ScoreIterationListener(self.score_iter))

    def fit(self, x, y, epochs=1, batchsize=None, print_stride=1):
        """
        Learn from input data and labels.

        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param epochs: (*int*) Number of fit epochs.
        :param batchsize: (*int*) Batch size of training data for each fit process.
        :param print_stride: (*int*) Epochs print stride. Default is 1.
        """
        n = len(x)
        x = Nd4jUtil.toNDArray(x._array)
        if y.ndim == 1:
            #y = y.tojarray('int')
            y = Nd4jUtil.toNDMatrix(y._array, self.nout)
        else:
            y = Nd4jUtil.toNDArray(y._array)
        ppi = 0
        for i in range(epochs):
            if i == ppi or i == epochs - 1:
                print('Epoch %i' % (i + 1))
                ppi += print_stride
            if batchsize is None:
                self._model.fit(x, y)
            else:
                si = 0
                while si < n:
                    ei = si + batchsize if si + batchsize <= n else n
                    xx = x.get(NDArrayIndex.interval(si, ei))
                    if isinstance(y, INDArray):
                        yy = y.get(NDArrayIndex.interval(si, ei))
                    else:
                        yy = y[si:ei]
                    self._model.fit(xx, yy)
                    si += batchsize

    def predict(self, x):
        """
        Predict the class labels for the provided data

        Parameters
        ----------
        X : array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        x = Nd4jUtil.toNDArray(x._array)
        r = self._model.output(x)
        r = Nd4jUtil.toArray(r)
        return np.array(r)

    def eval(self, x, y, batchsize=None):
        """
        Evaluation from input data and labels.

        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param batchsize: (*int*) Batch size of training data for each fit process.
        """
        _eval = Evaluation(self.nout)
        n = len(x)
        x = Nd4jUtil.toNDArray(x._array)
        if batchsize is None:
            y_pred = self._model.output(x)
            y = FeatureUtil.toOutcomeMatrix(y.tojarray('int'), self.nout)
            _eval.eval(y, y_pred)
        else:
            si = 0
            while si < n:
                ei = si + batchsize if si + batchsize <= n else n
                xx = x.get(NDArrayIndex.interval(si, ei))
                yy = y[si:ei]
                yy = FeatureUtil.toOutcomeMatrix(yy.tojarray('int'), self.nout)
                y_pred = self._model.output(xx)
                _eval.eval(yy, y_pred)
                si += batchsize

        return _eval