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


def seed_by_flat_cusum(t, y, y_err, method="linear", k=1.0, threshold=10.0, return_all=False):
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
    return_all : bool, default False
        If True, return all triggering candidate seeds as a list of (t0, tE) tuples.
        If False, return the single most significant seed as (t0, tE, triggered).

    Returns
    -------
    If return_all is False:
        t0_seed : float
        tE_seed : float
        triggered : bool
    If return_all is True:
        list of (t0_seed, tE_seed) tuples
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

    # Find contiguous triggered regions where S > 0
    is_triggered = S > 0.0
    changes = np.diff(is_triggered.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    if is_triggered[0]:
        starts = np.insert(starts, 0, 0)
    if is_triggered[-1]:
        ends = np.append(ends, len(S))

    candidates = []
    for start_idx, end_idx in zip(starts, ends):
        sub_S = S[start_idx:end_idx]
        max_val = np.max(sub_S)
        if max_val >= threshold:
            max_offset = np.argmax(sub_S)
            max_idx = start_idx + max_offset
            onset_idx = max(0, start_idx - 1)
            
            t_onset = t[onset_idx]
            t_end = t[max_idx]
            duration = t_end - t_onset
            tE_seed = float(max(duration / 4.0, 2.0))
            
            # 1. Primary candidate: peak absolute residual in the triggered region
            peak_offset = np.argmax(np.abs(r[onset_idx : max_idx + 1]))
            t0_seed = float(t[onset_idx + peak_offset])
            candidates.append((t0_seed, tE_seed, float(max_val)))
            
            # 2. If the CUSUM window is wide, add grid checkpoints across the window to prevent local min trapping
            if duration > 10.0:
                for frac in [0.25, 0.50, 0.75]:
                    t_frac = t_onset + frac * duration
                    candidates.append((float(t_frac), tE_seed, float(max_val) - 1e-5))

    # Sort candidates by CUSUM score descending
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    if return_all:
        if len(candidates) > 0:
            return [(c[0], c[1]) for c in candidates]
        else:
            peak_idx = np.argmax(y)
            return [(float(t[peak_idx]), 20.0)]
    else:
        if len(candidates) > 0:
            return candidates[0][0], candidates[0][1], True
        else:
            # Fallback: simple peak flux finding
            peak_idx = np.argmax(y)
            return float(t[peak_idx]), 20.0, False


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
