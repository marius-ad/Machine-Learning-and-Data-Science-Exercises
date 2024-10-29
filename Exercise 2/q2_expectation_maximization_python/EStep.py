import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    # Expectation step of the EM Algorithm
    #
    # INPUT:
    # means : Mean for each Gaussian KxD
    # weights : Weight vector 1xK for K Gaussians
    # covariances : Covariance matrices for each Gaussian DxDxK
    # X : Input data NxD
    #
    # N is number of data points
    # D is the dimension of the data points
    # K is number of Gaussians
    #
    # OUTPUT:
    # logLikelihood : Log-likelihood (a scalar).
    # gamma : NxK matrix of responsibilities for N datapoints and K Gaussians

    means = np.array(means)             # Shape: (K, D)
    covariances = np.array(covariances) # Shape: (D, D, K)
    weights = np.array(weights)         # Shape: (K,)
    X = np.array(X)                     # Shape: (N, D)
    
    N, D = X.shape
    K = means.shape[0]
    gamma = np.zeros((N, K))
    
    for k in range(K):
        mean = means[k]                   # Shape: (D,)
        covariance = covariances[:, :, k] # Shape: (D, D)
        weight = weights[k]               # Skalar
        
        try:
            inv_cov = np.linalg.inv(covariance)  
            det_cov = np.linalg.det(covariance)  
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(f"Die Kovarianzmatrix für Komponente {k} ist singulär.")
        
        diff = X - mean  # Shape: (N, D)
        normalization = 1 / (((2 * np.pi) ** (D / 2)) * (det_cov ** 0.5))

        # Compute exponent for all datapoints
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)  # Shape: (N,)
        
        pdf = normalization * np.exp(exponent)  # Shape: (N,)
        
        gamma[:, k] = weight * pdf  # Shape: (N,)
    
    
    total_prob = np.sum(gamma, axis=1, keepdims=True)  # Shape: (N, 1)
    gamma /= total_prob  # Shape: (N, K)

    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    return logLikelihood, gamma

