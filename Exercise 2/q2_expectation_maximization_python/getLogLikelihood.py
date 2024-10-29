import numpy as np


def getLogLikelihood(means, weights, covariances, X):
    # Log Likelihood estimation
    #
    # INPUT:
    # means          : Mean for each Gaussian KxD
    # weights        : Weight vector for K Gaussians
    # covariances    : Covariance matrices for each gaussian DxDxK
    # X              : Input data NxD
    # where N is number of data points
    # D is the dimension of the data points
    # K is number of gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar)

    #####Insert your code here for subtask 6a#####

    K, D = means.shape
    N = X.shape[0]

    def gauss(x, mean, invCovariance):
        return np.exp(/ np.sqrt(2* np.pi * covariance)

    logLikelihood = 0.0

    for n in N:
        x = X[n]
        likelihood = 0.0
  
        for k in K:
            mean = means[k]
            weight = weights[k]
            covariances[:, :, k]





    return logLikelihood

