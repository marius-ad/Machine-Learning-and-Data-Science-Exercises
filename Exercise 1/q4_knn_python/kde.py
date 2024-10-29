import numpy as np


def kde(samples, h):
    """
    compute density estimation from samples with KDE and a Gaussian kernel
    Input
     samples    : (N,) vector of data points
     h          : standard deviation of the Gaussian kernel
    Output
     estimatedDensity : (200, 2) matrix of estimated density in the range of [-5, 5]
    """

    # Get the number of the samples x_n
    N = len(samples)

    # Create a linearly spaced vector of 200 points between -5 and 5
    pos = np.arange(-5.0, 5.0, 0.05)  # "x"

    #####Insert your code here for subtask 5a#####
    # Estimate the density from the samples

    # define kernel method, here Gaussian-dist.
    def kernel(x) -> float:
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / 2)

    # define pdf
    def pdf(x) -> float:
        kernelSum = 0 
        for xSample in samples:
            kernelSum += kernel((x - xSample) / h)

        return kernelSum / (N * h)  # normalize

    # estimate density for each position
    estimate = np.zeros(200)
    for i in range(len(pos)):
        estimate[i] = pdf(pos[i])

    """
        ### faster way with numpy vector operations ###
    
    # calculate difference between pos and each sample (with numpy broadcasting)
    diff = (pos.reshape(200, 1) - samples.reshape(1, 100)) / h
    # calculate kernel values
    kernel_values = kernel(diff)
    # build the sum of kernel values for each position and normilize
    estimate = np.sum(kernel_values, axis=1) / (N * h)
    """

    # Form the output variable
    estimatedDensity = np.stack((pos, estimate), axis=1)

    return estimatedDensity



