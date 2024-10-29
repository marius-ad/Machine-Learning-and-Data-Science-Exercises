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


    #####Start Subtask 5a#####
    # Estimate the density from the samples
    distance_matrix = pos[np.newaxis, :] - samples[:, np.newaxis]
    norm = np.sqrt(2 * np.pi) * h * N
    estimatedDensity = (
        np.sum(
            np.exp(-(distance_matrix**2) / (2 * h**2)),
            axis=0,
        )
        / norm
    )
    #####End Subtask#####


    # Form the output variable
    estimatedDensity = np.stack((pos, estimatedDensity), axis=1)

    return estimatedDensity
