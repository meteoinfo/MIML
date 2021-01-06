from org.nd4j.linalg.learning.config import Nesterovs as JNesterovs
from org.nd4j.linalg.learning.config import Sgd as JSgd
from org.nd4j.linalg.learning.config import Adam as JAdam
from org.nd4j.linalg.learning.config import Nadam as JNadam
from org.nd4j.linalg.learning.config import AdaGrad as JAdaGrad

__all__ = ['Nesterovs','Sgd','Adam','Nadam','AdaGrad']

class Nesterovs(JNesterovs):

    def __init__(self, learn_rate=0.01, momentum=0.9):
        super(Nesterovs, self).__init__(learn_rate, momentum)

class Sgd(JSgd):

    def __init__(self, learn_rate=0.1):
        super(Sgd, self).__init__(learn_rate)

class Adam(JAdam):

    def __init__(self, learn_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(learn_rate, beta1, beta2, epsilon)

class Nadam(JNadam):

    def __init__(self, learn_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Nadam, self).__init__(learn_rate, beta1, beta2, epsilon)

class AdaGrad(JAdaGrad):

    def __init__(self, learn_rate=1e-1, epsilon=1e-6):
        super(AdaGrad, self).__init__(learn_rate, epsilon)