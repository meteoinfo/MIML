from org.deeplearning4j.nn.conf import MultiLayerConfiguration, NeuralNetConfiguration
from org.deeplearning4j.nn.multilayer import MultiLayerNetwork
from org.deeplearning4j.nn.weights import WeightInit
from org.deeplearning4j.optimize.listeners import ScoreIterationListener
from org.nd4j.linalg.learning.config import Nesterovs
from org.meteothink.miml.nd4j import ArrayUtil
import mipylib.numeric as np

class Network(object):
    '''
    Multiple Layer Neural Network.


    '''

    def __init__(self, seed=123, weight_init='xavier', learn_rate=0.01, momentum=0.9,
            layers=None):
        self._seed = seed
        self._weight_init = WeightInit.valueOf(weight_init.upper())
        self._learn_rate = learn_rate
        self._momentum = momentum
        self.layers = layers

    def add(self, layer):
        '''
        Add a layer

        :param layer: (*Layer*) The layer.
        '''
        if self.layers is None:
            self.layers = []
        self.layers.append(layer)

    def compile(self):
        '''
        Build the network.
        '''
        confb = NeuralNetConfiguration.Builder().seed(self._seed) \
            .weightInit(self._weight_init) \
            .updater(Nesterovs(self._learn_rate, self._momentum)) \
            .list()
        for layer in self.layers:
            confb.layer(layer._layer)
        conf = confb.build()
        self._model = MultiLayerNetwork(conf)
        self._model.init()
        #Print score every 10 parameter updates
        self._model.setListeners(ScoreIterationListener(10))

    def fit(self, x, y, epochs=1):
        """
        Learn from input data and labels.

        :param x: (*array*) Training samples. 2D array.
        :param y: (*array*) Training labels in [0, c), where c is the number of classes.
        """
        x = ArrayUtil.toNDArray(x._array)
        if y.ndim == 1:
            y = y.tojarray('int')
        else:
            y = ArrayUtil.toNDArray(y._array)
        for i in range(epochs):
            #print 'Epoch %i' % (i + 1)
            self._model.fit(x, y)

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
        x = ArrayUtil.toNDArray(x._array)
        r = self._model.output(x)
        r = ArrayUtil.toArray(r)
        return np.array(r)