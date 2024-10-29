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
    
    means = np.array(means)
    weights = np.array(weights)
    covariances = np.array(covariances)
    X = np.array(X)

    N, D = X.shape
    K = means.shape[0]

    def gauss(x, mean, inv, det):
        normalization = 1 / (((2*np.pi) ** (D/2)) * (det ** (1/2)))
        exponent = -0.5 * np.dot(np.dot((x-mean).T, inv), (x-mean))

        return np.exp(exponent) * normalization

    logLikelihood = 0.0

    for n in range(N):
        x = X[n]
        likelihood = 0.0
  
        for k in range(K):
            mean = means[k]
            weight = weights[k]
            covariance = covariances[:, :, k]

            try:
                invCov = np.linalg.inv(covariance)
                detCov = np.linalg.det(covariance)
            except np.linalg.LinAlgError:
                continue

            likelihood += weight * gauss(x, mean, invCov, detCov)
        
        logLikelihood += np.log(likelihood)

    return logLikelihood

