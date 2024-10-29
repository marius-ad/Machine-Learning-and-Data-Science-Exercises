import numpy as np


def knn(samples, k):
    """
    compute density estimation from samples with k-NN
    Input
     samples    : (N,) vector of data points
     k          : number of neighbors
    Output
     estimatedDensity : (200, 2) estimated density in the range of [-5, 5]
    """

    #####Start Subtask 5b#####

    # Get the number of the samples x_n
    N = len(samples)

    # Create a linearly spaced vector of 200 points between -5 and 5
    pos = np.arange(-5.0, 5.0, 0.05)  # "x"

    # Sort the distances so that we can choose the k-th point
    distance_matrix = pos[np.newaxis, :] - samples[:, np.newaxis]
    distances = np.sort(np.abs(distance_matrix), axis=0)

    # Estimate the probability density using the k-NN density estimation
    # "1D sphere" volume = 2 * distance to the k-th farthest neighbor
    V = 2 * distances[k - 1, :]
    estimatedDensity = k / (N * V)

    # Form the output variable
    estimatedDensity = np.stack((pos, estimatedDensity), axis=1)

    #####End Subtask#####


    return estimatedDensity
