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

    #####Insert your code here for subtask 5b#####
    N = len(samples)
    pos = np.arange(-5.0, 5.0, 0.05)
    samples.sort()

    def pdf(V):
        return k / (N * V)

    # calc distance between positions and each sample and sort
    distance = np.abs(pos.reshape(200, 1) - samples.reshape(1, 100))
    distance = np.sort(distance, axis=1)

    # take for each pos the kth nearest neighbor
    kth_neighbor_distance = distance[:, k-1]

    estimate = pdf(2 * kth_neighbor_distance)

    # normalize the density function -> integral must be 1
    delta_x = 0.05
    integral = np.sum(estimate) * delta_x
    estimate /= integral

    # Form the output variable
    estimatedDensity = np.stack((pos, estimate), axis=1)

    return estimatedDensity
