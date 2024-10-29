import numpy as np
from getLogLikelihood import getLogLikelihood


def EStep(means, covariances, weights, X):
    """
    Expectation step of the EM Algorithm for Gaussian Mixture Models.
    
    INPUT:
    means         : Mean for each Gaussian, shape (K, D)
    covariances   : Covariance matrices for each Gaussian, shape (D, D, K)
    weights       : Weight vector for K Gaussians, shape (K,)
    X             : Input data, shape (N, D)
    
    OUTPUT:
    logLikelihood : Log-likelihood (a scalar)
    gamma         : NxK matrix of responsibilities for N datapoints and K Gaussians
    """

    # Konvertiere Eingaben in NumPy-Arrays, falls sie es noch nicht sind
    means = np.array(means)             # Shape: (K, D)
    covariances = np.array(covariances) # Shape: (D, D, K)
    weights = np.array(weights)         # Shape: (K,)
    X = np.array(X)                     # Shape: (N, D)
    
    N, D = X.shape
    K = means.shape[0]
    
    # Initialisiere die Gamma-Matrix
    gamma = np.zeros((N, K))
    
    # Berechne die Wahrscheinlichkeit p(xn | j) für jede Komponente und jeden Datenpunkt
    for k in range(K):
        mean = means[k]                   # Shape: (D,)
        covariance = covariances[:, :, k] # Shape: (D, D)
        weight = weights[k]               # Skalar
        
        try:
            inv_cov = np.linalg.inv(covariance)  # Inverse der Kovarianzmatrix
            det_cov = np.linalg.det(covariance)  # Determinante der Kovarianzmatrix
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(f"Die Kovarianzmatrix für Komponente {k} ist singulär.")
        
        # Berechne die Differenz zwischen Datenpunkten und Mittelwert
        diff = X - mean  # Shape: (N, D)
        
        # Berechne den Exponenten der Gaußschen PDF
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)  # Shape: (N,)
        
        # Berechne die Normalisierungskonstante
        norm_const = 1 / (((2 * np.pi) ** (D / 2)) * (det_cov ** 0.5))
        
        # Berechne die Wahrscheinlichkeit p(xn | j)
        pdf = norm_const * np.exp(exponent)  # Shape: (N,)
        
        # Berechne die gewichtete Wahrscheinlichkeit
        gamma[:, k] = weight * pdf  # Shape: (N,)
    
    # Berechne die Gesamtheit der Wahrscheinlichkeiten für jede Datenpunkt
    total_prob = np.sum(gamma, axis=1, keepdims=True)  # Shape: (N, 1)
    
    # Normalisiere die Gamma-Matrix, um die Verantwortlichkeiten zu erhalten
    gamma /= total_prob  # Shape: (N, K)
    
    # Berechne die Log-Likelihood
    # Verwende die vorhandene getLogLikelihood-Funktion
    logLikelihood = getLogLikelihood(means, weights, covariances, X)
    
    return logLikelihood, gamma

