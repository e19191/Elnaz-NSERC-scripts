import numpy as np


def f_predictiontime(x, y, maxlag):
    """
    Select prediction time as time lag with maximum cross-correlation.
    If lag is <= 1, set predtime=1, else use lag.

    Parameters:
        x, y : np.ndarray (1D)
        maxlag : int, max lag for cross-correlation

    Returns:
        predtime : int
    """
    # Compute cross-correlation with maxlag
    res = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    # len(res) = 2*len(x)-1, get lags centered at len(x)-1
    mid = len(x) - 1
    lags = np.arange(-maxlag, maxlag + 1)
    # Extract cross-corr values only at lags within maxlag
    # Align indexes: lag 0 corresponds to index mid
    res_maxlag = res[(mid - maxlag):(mid + maxlag + 1)]

    # Find lag at max correlation
    ik = np.argmax(res_maxlag)
    tlag_max_cc = lags[ik]

    if tlag_max_cc <= 1:
        predtime = 1
    else:
        predtime = tlag_max_cc

    return predtime
