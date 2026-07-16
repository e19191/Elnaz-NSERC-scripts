import numpy as np

def F_Int2Symb(E):
    """
    Convert embedded data rows to symbolic integer using ranks.

    Parameters:
        E : np.ndarray shape (n_samples, dim)

    Returns:
        Symbol : np.ndarray shape (n_samples,)
    """
    sorted_idx = np.argsort(E, axis=1)
    n_samples, dim = E.shape

    # Initialize ranks matrix (same shape as E)
    ranks = np.zeros_like(E, dtype=int)

    # Create rank values 1 to dim
    rank_values = np.arange(1, dim + 1)

    # Assign ranks based on sorted indices for each row
    for i in range(n_samples):
        ranks[i, sorted_idx[i]] = rank_values

    # Convert ranks vector into single integer symbol using base-10 polynomial
    # Example: for dim=3 and ranks=[3,1,2], symbol = 3*10^2 + 1*10 + 2 = 312
    Symbol = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        Symbol[i] = 0
        for j in range(dim):
            Symbol[i] += ranks[i, j] * (10 ** (dim - j - 1))

    return Symbol
