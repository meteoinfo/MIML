from org.nd4j.linalg.lossfunctions import LossFunctions

class LossFunction(object):
    MSE = LossFunctions.LossFunction.MSE
    L1 = LossFunctions.LossFunction.L1
    XENT = LossFunctions.LossFunction.XENT
    MCXENT = LossFunctions.LossFunction.MCXENT
    SPARSE_MCXENT = LossFunctions.LossFunction.SPARSE_MCXENT
    SQUARED_LOSS = LossFunctions.LossFunction.SQUARED_LOSS
    RECONSTRUCTION_CROSSENTROPY = LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY
    NEGATIVELOGLIKELIHOOD = LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD
    COSINE_PROXIMITY = LossFunctions.LossFunction.COSINE_PROXIMITY
    HINGE = LossFunctions.LossFunction.HINGE
    SQUARED_HINGE = LossFunctions.LossFunction.SQUARED_HINGE
    KL_DIVERGENCE = LossFunctions.LossFunction.KL_DIVERGENCE
    MEAN_ABSOLUTE_ERROR = LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR
    L2 = LossFunctions.LossFunction.L2
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = LossFunctions.LossFunction.MEAN_ABSOLUTE_PERCENTAGE_ERROR
    MEAN_SQUARED_LOGARITHMIC_ERROR = LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR
    POISSON = LossFunctions.LossFunction.POISSON
    WASSERSTEIN = LossFunctions.LossFunction.WASSERSTEIN