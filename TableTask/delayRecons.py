import numpy as np


def delayRecons(data, v, m):
    """
    Create embedded data matrix based on embedding dimension m and lag v.

    Parameters:
        data : np.ndarray shape (n_samples, n_channels)
        v : int, lag
        m : int, embedding dimension

    Returns:
        y : np.ndarray shape (n_samples - v*(m-1), m, n_channels)
    """
    MaxEpoch = data.shape[0]
    ch = data.shape[1]

    y = np.zeros((MaxEpoch - v * (m - 1), m, ch))

    for c in range(ch):
        for j in range(m):
            start_idx = j * v
            end_idx = MaxEpoch - v * (m - 1 - j)
            y[:, j, c] = data[start_idx:end_idx, c]

    return y
