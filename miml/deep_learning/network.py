from org.deeplearning4j.nn.conf import NeuralNetConfiguration
from org.deeplearning4j.nn.multilayer import MultiLayerNetwork
from org.deeplearning4j.optimize.listeners import ScoreIterationListener
from org.nd4j.linalg.api.ndarray import INDArray
from org.nd4j.linalg.indexing import NDArrayIndex
from org.nd4j.evaluation.classification import Evaluation
from org.nd4j.linalg.util import FeatureUtil
from org.meteothink.miml.nd4j import Nd4jUtil
import mipylib.numeric as np
from .layer import Dense, Output
import network_util

class Network(object):
    '''
    Multiple Layer Neural Network.


    '''

    def __init__(self, seed=123, weight_init=None, updater=None,
            bias_init=None, mini_batch=None, layers=None, **kwargs):
        self._seed = seed
        self._weight_init = None if weight_init is None else network_util.get_weight_init(weight_init)
        self._updater = None if updater is None else network_util.get_updater(updater)
        self._bias_init = bias_init
        self._mini_batch = mini_batch
        self.layers = layers

    def add(self, layer):
        '''
        Add a layer

        :param layer: (*Layer*) The layer.
        '''
        if self.layers is None:
            self.layers = []
        self.layers.append(layer)
        if isinstance(layer, Output):
            self._nout = layer.nout

    def compile(self, score_iter=10):
        '''
        Build the network.

        :param score_iter: (*int*) Number of parameter updates for printing score
        '''
        confb = NeuralNetConfiguration.Builder() \
            .seed(self._seed)
        if not self._updater is None:
            confb.updater(self._updater)
        if not self._weight_init is None:
            confb.weightInit(self._weight_init)
        if not self._bias_init is None:
            confb.biasInit(self._bias_init)
        if not self._mini_batch is None:
            confb.miniBatch(self._mini_batch)
        confb = confb.list()
        for layer in self.layers:
            confb.layer(layer._layer)
        conf = confb.build()
        self._model = MultiLayerNetwork(conf)
        self._model.init()
        self._score_iter = score_iter
        self._model.setListeners(ScoreIterationListener(self._score_iter))

    def fit(self, x, y, epochs=1, batchsize=None):
        """
        Learn from input data and labels.

        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        :param epochs: (*int*) Number of fit epochs.
        :param batchsize: (*int*) Batch size of training data for each fit process.
        """
        n = len(x)
        x = Nd4jUtil.toNDArray(x._array)
        if y.ndim == 1:
            #y = y.tojarray('int')
            y = Nd4jUtil.toNDMatrix(y._array, self._nout)
        else:
            y = Nd4jUtil.toNDArray(y._array)
        stride = epochs / 10;
        ppi = 0
        for i in range(epochs):
            if epochs >= 10:
                if i == ppi or i == epochs - 1:
                    print 'Epoch %i' % (i + 1)
                    ppi += stride
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
        _eval = Evaluation(self._nout)
        n = len(x)
        x = Nd4jUtil.toNDArray(x._array)
        if batchsize is None:
            y_pred = self._model.output(x)
            y = FeatureUtil.toOutcomeMatrix(y.tojarray('int'), 2)
            _eval.eval(y, y_pred)
        else:
            si = 0
            while si < n:
                ei = si + batchsize if si + batchsize <= n else n
                xx = x.get(NDArrayIndex.interval(si, ei))
                yy = y[si:ei]
                yy = FeatureUtil.toOutcomeMatrix(yy.tojarray('int'), 2)
                y_pred = self._model.output(xx)
                _eval.eval(yy, y_pred)
                si += batchsize

        return _eval.stats()