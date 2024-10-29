import numpy as np
from getLogLikelihood import getLogLikelihood


def MStep(gamma, X):
    # Maximization step of the EM Algorithm
    #
    # INPUT:
    # gamma          : NxK matrix of responsibilities for N datapoints and K Gaussians.
    # X              : Input data (NxD matrix for N datapoints of dimension D).
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood  : Log-likelihood (a scalar).
    # means          : Mean for each gaussian (KxD).
    # weights        : Vector of weights of each gaussian (1xK).
    # covariances    : Covariance matrices for each component(DxDxK).

    #####Insert your code here for subtask 6c#####
    gamma = np.array(gamma)
    X = np.array(X)

    N, K = gamma.shape
    D = X.shape[1]

    N_k = np.sum(gamma, axis=0)
    weights = N_k / N
    means = np.dot(gamma.T, X) / N_k[:, np.newaxis]

    covariances = np.zeros((D, D, K))
    for k in range(K):
        # diffrence between each datepoint and the current mean
        diff = X - means[k]  # Shape: (N, D)
        
        # 
        weighted_diff = gamma[:, k][:, np.newaxis] * diff  # Shape: (N, D)
        cov_k = np.dot(weighted_diff.T, diff) / N_k[k]  # Shape: (D, D)
        
        
        covariances[:, :, k] = cov_k  # Shape: (D, D)
        
    # calculate new likelihood
    logLikelihood = getLogLikelihood(means, weights, covariances, X)

    return weights, means, covariances, logLikelihood
