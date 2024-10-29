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


    # calculate likelihood for all K Gaussians, parallel for all N data points
    likelihoods = np.zeros(N)
    for k in range(K):
        mean = means[k]
        weight = weights[k]
        covariance = covariances[:, :, k]

        try:
            invCov = np.linalg.inv(covariance)
            detCov = np.linalg.det(covariance)
        except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError(f"covariance matrice for {k} is singular.")

        # diffrence between each datapoint and the mean
        diff = X - mean # Shape: (N, D)

        # Compute exponent for all datapoints and normalization
        exponent = -0.5 * np.sum(diff @ invCov * diff, axis=1)
        normalization = 1 / (((2 * np.pi) ** (D / 2)) * (detCov ** 0.5))

        gauss_pdf = normalization * np.exp(exponent)
    
        # add to likelihoods
        likelihoods += weight * gauss_pdf

    logLikelihood = np.sum(np.log(likelihoods))

    return logLikelihood

