import numpy as np


def F_EstimateProb(integer_array):
    """
    Estimate the probability distribution of symbolic integers.

    Args:
        integer_array (np.ndarray): 1D array of integers (symbols)

    Returns:
        np.ndarray: Array of probabilities, same shape as input
    """
    integer_array = np.asarray(integer_array)
    n = len(integer_array)
    prob = np.zeros(n)

    unique_vals = np.unique(integer_array)
    for val in unique_vals:
        indices = np.where(integer_array == val)[0]
        prob[indices] = len(indices) / n

    return prob
