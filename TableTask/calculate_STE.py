import numpy as np
from f_predictiontime import f_predictiontime
from f_nste import f_nste

def calculate_STE(X, Y, dim, tau_array, n_permutations=0,
                  file_name=None, window_idx=None, total_windows=None, w_sec = None):
    """
    Returns:
        STE           (T,2)
        NSTE          (T,2)
        mean_perm     (T,2)
        std_perm      (T,2)
        Z             (T,2)
        pvals         (T,2)
        asymmetry     (T,)

        asym > 0	R → D dominates
        asym < 0	D → R dominates
        asym ≈ 0	symmetric / no directionality

    """

    tau_array = np.array(tau_array)
    T = len(tau_array)

    STE = np.full((T, 2), np.nan)
    NSTE = np.full((T, 2), np.nan)
    mean_perm = np.full((T, 2), np.nan)
    std_perm = np.full((T, 2), np.nan)
    Z = np.full((T, 2), np.nan)
    pvals = np.full((T, 2), np.nan)
    asym = np.full(T, np.nan)

    # For progress
    if file_name is not None:
        if window_idx is not None and total_windows is not None:
            print(f"\n➡️ File: {file_name}  |  Window {window_idx+1}/{total_windows}  |  Window_sec {w_sec}")

    # Prediction offset
    delta = f_predictiontime(X, Y, 50)

    # --- REAL STE/NSTE ---
    for i, tau_val in enumerate(tau_array):
        # print(f"    • Real STE: tau {tau_val} ({i+1}/{T})")

        ste_vals, nste_vals = f_nste(
            np.column_stack((X, Y)), dim, tau_val, delta
        )
        STE[i, :] = ste_vals
        NSTE[i, :] = nste_vals

    # --- ASYMMETRY ---
    asym = STE[:, 0] - STE[:, 1]

    # --- PERMUTATIONS ---
    if n_permutations > 0:
        perm_results = np.zeros((n_permutations, T, 2))

        for p in range(n_permutations):
            print(f"    🔁 Permutation {p+1}/{n_permutations}")

            Yp = np.random.permutation(Y)

            for i, tau_val in enumerate(tau_array):
                #print(f"       - tau {tau_val} ({i+1}/{T})")

                ste_p, _ = f_nste(
                    np.column_stack((X, Yp)), dim, tau_val, delta
                )
                perm_results[p, i, :] = ste_p

        mean_perm = perm_results.mean(axis=0)
        std_perm = perm_results.std(axis=0)

        Z = (STE - mean_perm) / (std_perm + 1e-12)

        pvals = np.mean(perm_results >= STE[None, :, :], axis=0)

    return STE, NSTE, mean_perm, std_perm, Z, pvals, asym
