from F_EstimateProb import F_EstimateProb

def f_Integer2prob(Integer1, Integer2, Integer3):
    """
    Calculate probability distributions for combinations of integers.

    Parameters:
        Integer1, Integer2, Integer3 : np.ndarray (1D)

    Returns:
        P1, P2, P3, P4 : np.ndarray - probabilities of joint states
    """
    import numpy as np

    # Determine number of digits (symbol length)
    def symbol_len(arr):
        return np.max(np.ceil(np.log10(arr + 0.1)).astype(int))

    SymbolLen = max(symbol_len(Integer1), symbol_len(Integer2), symbol_len(Integer3))

    INT1 = Integer1 * 10 ** (SymbolLen * 2) + Integer2 * 10 ** (SymbolLen) + Integer3
    INT2 = Integer2 * 10 ** (SymbolLen) + Integer3
    INT3 = Integer1 * 10 ** (SymbolLen) + Integer2
    INT4 = Integer2

    P1 = F_EstimateProb(INT1)
    P2 = F_EstimateProb(INT2)
    P3 = F_EstimateProb(INT3)
    P4 = F_EstimateProb(INT4)

    # Find unique indices of INT1 (preserve order)
    _, U_Index = np.unique(INT1, return_index=True)

    # Keep only unique probabilities at these indices
    P1 = P1[U_Index]
    P2 = P2[U_Index]
    P3 = P3[U_Index]
    P4 = P4[U_Index]

    return P1, P2, P3, P4
