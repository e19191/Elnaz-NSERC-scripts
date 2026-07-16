import numpy as np
# from calculate_STE import calculate_STE
from f_predictiontime import f_predictiontime
# from f_nste import f_nste
from delayRecons import delayRecons
from F_Int2Symb import F_Int2Symb
from F_Integer2prob import f_Integer2prob
from F_EstimateProb import F_EstimateProb


def f_nste(data, dim, lag, delta):
    """
    Compute STE and NSTE for bivariate data.

    Parameters:
        data: np.ndarray of shape (n_samples, 2) - two time series
        dim: int - embedding dimension
        lag: int - embedding lag
        delta: int - prediction time lag

    Returns:
        STE: np.ndarray, shape (2,) [STE_YX, STE_XY]
        NSTE: np.ndarray, shape (2,) [NSTE_YX, NSTE_XY]
    """
    ch = data.shape[1]  # number of signals, expected 2

    # Part 1: STE of original data

    Ddata = delayRecons(data, lag, dim)  # shape (samples, dim, ch)

    INT = np.zeros_like(Ddata[:, :, 0], dtype=int)  # preallocate

    for c in range(ch):
        INT[:, c] = F_Int2Symb(Ddata[:, :, c])

    Int_future = INT[delta:, :]
    Int_past = INT[:-delta, :]

    # Compute STE_YX (target = X = col0, source = Y = col1)
    P1, P2, P3, P4 = f_Integer2prob(Int_future[:, 0], Int_past[:, 0], Int_past[:, 1])
    STE_YX = np.sum(P1 * (np.log2(P1 * P4) - np.log2(P2 * P3)))
    H_YX = -np.sum(P3 * (np.log2(P3) - np.log2(P4)))

    # Compute STE_XY (target = Y = col1, source = X = col0)
    P1, P2, P3, P4 = f_Integer2prob(Int_future[:, 1], Int_past[:, 1], Int_past[:, 0])
    STE_XY = np.sum(P1 * (np.log2(P1 * P4) - np.log2(P2 * P3)))
    H_XY = -np.sum(P3 * (np.log2(P3) - np.log2(P4)))

    STE = np.array([STE_YX, STE_XY])
    H = np.array([H_YX, H_XY])

    # Part 2: STE of shuffled data - for NSTE_YX
    data2 = data[:, 1]  # source (Y)
    num_trials = 20

    STE_shuffled_all = np.zeros(num_trials)
    for i in range(num_trials):
        shuffled_data2 = np.random.permutation(data2)
        Ddata = delayRecons(np.column_stack((data[:, 0], shuffled_data2)), lag, dim)
        INT = np.zeros_like(Ddata[:, :, 0], dtype=int)
        for c in range(ch):
            INT[:, c] = F_Int2Symb(Ddata[:, :, c])
        Int_future = INT[delta:, :]
        Int_past = INT[:-delta, :]
        P1, P2, P3, P4 = f_Integer2prob(Int_future[:, 0], Int_past[:, 0], Int_past[:, 1])
        STE_shuffled_all[i] = np.sum(P1 * (np.log2(P1 * P4) - np.log2(P2 * P3)))

    STE_shuffled_ave = np.mean(STE_shuffled_all)

    NSTE_YX_num = STE[0] - STE_shuffled_ave
    NSTE_YX = NSTE_YX_num / H[0] if H[0] != 0 else 0

    # Part 3: STE of shuffled data - for NSTE_XY
    data1 = data[:, 0]  # source (X)

    STE_shuffled_all = np.zeros(num_trials)
    for i in range(num_trials):
        shuffled_data1 = np.random.permutation(data1)
        Ddata = delayRecons(np.column_stack((shuffled_data1, data[:, 1])), lag, dim)
        INT = np.zeros_like(Ddata[:, :, 0], dtype=int)
        for c in range(ch):
            INT[:, c] = F_Int2Symb(Ddata[:, :, c])
        Int_future = INT[delta:, :]
        Int_past = INT[:-delta, :]
        P1, P2, P3, P4 = f_Integer2prob(Int_future[:, 1], Int_past[:, 1], Int_past[:, 0])
        STE_shuffled_all[i] = np.sum(P1 * (np.log2(P1 * P4) - np.log2(P2 * P3)))

    STE_shuffled_ave = np.mean(STE_shuffled_all)

    NSTE_XY_num = STE[1] - STE_shuffled_ave
    NSTE_XY = NSTE_XY_num / H[1] if H[1] != 0 else 0

    NSTE = np.array([NSTE_YX, NSTE_XY])

    return STE, NSTE
