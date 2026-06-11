import numpy as np


def run_linear_cusum(r, k=1.0):
    """
    Run linear cumulative sum (CUSUM) on residuals.
    Particularly useful for detecting positive deviations (such as magnification peaks).

    Parameters
    ----------
    r : np.ndarray
        Standardized residuals (e.g., (y - y_model) / y_err)
    k : float, default 1.0
        Slack parameter (drain) to suppress baseline fluctuations.

    Returns
    -------
    S : np.ndarray
        Accumulated CUSUM statistic over time.
    """
    S = np.zeros(len(r))
    for i in range(1, len(r)):
        val = S[i - 1] + r[i] - k
        if val > 0.0:
            S[i] = val
    return S


def run_quadratic_cusum(r, k=2.0):
    """
    Run quadratic cumulative sum (CUSUM) on residuals.
    Accumulates variance excess above the expected baseline noise level.

    Parameters
    ----------
    r : np.ndarray
        Standardized residuals (e.g., (y - y_model) / y_err)
    k : float, default 2.0
        Slack parameter (drain) to suppress baseline fluctuations.

    Returns
    -------
    S : np.ndarray
        Accumulated CUSUM statistic over time.
    """
    S = np.zeros(len(r))
    for i in range(1, len(r)):
        val = S[i - 1] + r[i] ** 2 - 1.0 - k
        if val > 0.0:
            S[i] = val
    return S


def seed_by_flat_cusum(t, y, y_err, method="linear", k=1.0, threshold=10.0):
    """
    Estimate rough lensing parameters t0 and tE from a flat baseline fit using CUSUM.

    Parameters
    ----------
    t : np.ndarray
        Time array (BJD). Must be sorted in ascending order.
    y : np.ndarray
        Flux or magnitude array.
    y_err : np.ndarray
        Observation errors.
    method : {'linear', 'quadratic'}, default 'linear'
        CUSUM formulation to use.
    k : float, default 1.0
        Slack parameter for CUSUM.
    threshold : float, default 10.0
        Significance threshold (H) for CUSUM triggering.

    Returns
    -------
    t0_seed : float
        Seed value for peak time t0.
    tE_seed : float
        Seed value for Einstein time tE.
    triggered : bool
        True if CUSUM exceeded the significance threshold.
    """
    # Estimate robust baseline using median (flux excess assumption)
    y_base = np.median(y)
    r = (y - y_base) / y_err

    if method == "linear":
        S = run_linear_cusum(r, k=k)
    elif method == "quadratic":
        S = run_quadratic_cusum(r, k=k)
    else:
        raise ValueError("Method must be 'linear' or 'quadratic'")

    max_score = np.max(S)
    if max_score < threshold:
        # Fallback: simple peak flux finding
        peak_idx = np.argmax(y)
        return float(t[peak_idx]), 20.0, False

    max_idx = np.argmax(S)
    onset_idx = max_idx
    while onset_idx > 0 and S[onset_idx] > 0.0:
        onset_idx -= 1

    t_onset = t[onset_idx]
    t_end = t[max_idx]

    # Peak is the maximum absolute residual within the CUSUM window
    window_r = r[onset_idx : max_idx + 1]
    peak_offset = np.argmax(np.abs(window_r))
    t0_seed = float(t[onset_idx + peak_offset])

    # Einstein time estimate based on CUSUM window duration
    duration = t_end - t_onset
    tE_seed = float(max(duration / 4.0, 2.0))

    return t0_seed, tE_seed, True


def find_anomalies_cusum(t, residuals_sigma, threshold=25.0, k=2.0):
    """
    Detect anomalies in residuals using quadratic CUSUM and extract their properties.

    Parameters
    ----------
    t : np.ndarray
        Time array (BJD).
    residuals_sigma : np.ndarray
        Standardized residuals: (y_data - y_model) / y_err
    threshold : float, default 25.0
        Significance threshold (H) for CUSUM.
    k : float, default 2.0
        CUSUM slack parameter.

    Returns
    -------
    dict
        Properties of the detected anomaly:
        {
            'triggered': bool,
            'score': float,
            't0': float or None,
            'onset': float or None,
            'end': float or None,
            'duration': float,
            'residuals_std': float,
            'cusum_statistic': np.ndarray
        }
    """
    S = run_quadratic_cusum(residuals_sigma, k=k)
    max_score = float(np.max(S))
    max_idx = np.argmax(S)

    if max_score < threshold:
        return {
            "triggered": False,
            "score": max_score,
            "t0": None,
            "onset": None,
            "end": None,
            "duration": 0.0,
            "residuals_std": 1.0,
            "cusum_statistic": S,
        }

    # Trace backward to find the onset
    onset_idx = max_idx
    while onset_idx > 0 and S[onset_idx] > 0.0:
        onset_idx -= 1

    t_onset = float(t[onset_idx])
    t_end = float(t[max_idx])
    duration = t_end - t_onset

    # Peak is the maximum absolute residual inside the window
    window_res = np.abs(residuals_sigma[onset_idx : max_idx + 1])
    peak_offset = np.argmax(window_res)
    t0_anomaly = float(t[onset_idx + peak_offset])

    # Standard deviation of the residuals inside the physical window
    residuals_std = float(np.std(residuals_sigma[onset_idx : max_idx + 1]))

    return {
        "triggered": True,
        "score": max_score,
        "t0": t0_anomaly,
        "onset": t_onset,
        "end": t_end,
        "duration": duration,
        "residuals_std": residuals_std,
        "cusum_statistic": S,
    }
