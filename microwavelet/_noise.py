"""
microwavelet._noise
===================
Utilities for robust noise characterisation in time-series data.
Identifies white noise floor, red (correlated) noise excess, and autocorrelation.
"""

import numpy as np

def characterize_noise(t, y, y_err=None, bin_sizes=None):
    """
    Robustly characterises white and red (correlated) noise in a light curve.

    Uses three complementary diagnostic methods:
    1. Robust first-difference scatter (white noise floor estimation).
    2. Pont-style binning scaling factor (red noise excess quantification).
    3. Autocorrelation coefficients at short lags.

    Parameters
    ----------
    t : np.ndarray
        Observed times (days).
    y : np.ndarray
        Observed flux or magnitude residuals.
    y_err : np.ndarray, optional
        Reported observational errors.
    bin_sizes : list[int], optional
        Bin block sizes to test for the Pont red noise scaling.
        Defaults to [5, 10, 20, 50] points.

    Returns
    -------
    dict
        Detailed noise metrics including:
        - "sigma_white": Robust white noise scatter.
        - "sigma_total": Total standard deviation.
        - "pont_excess": Dict mapping bin sizes to red noise excess factor (1.0 = pure white).
        - "autocorr_lag1": Autocorrelation at lag 1.
        - "autocorr_lag2": Autocorrelation at lag 2.
        - "has_red_noise": Boolean flag indicating significant correlated noise.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    
    # Sort by time to ensure physical lag calculations
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]
    
    # 1. White Noise Estimation (Robust against long-term physical trends / events)
    # Standard deviation of first differences divided by sqrt(2)
    dy = np.diff(y)
    med_dy = np.median(dy)
    mad_dy = np.median(np.abs(dy - med_dy)) / 0.6745
    sigma_white = float(mad_dy / np.sqrt(2.0))
    
    # 2. Total Variance
    med_y = np.median(y)
    mad_y = np.median(np.abs(y - med_y)) / 0.6745
    sigma_total = float(mad_y)
    
    # 3. Autocorrelation Function (ACF) at lag 1 and 2
    n = len(y)
    autocorr_lag1 = 0.0
    autocorr_lag2 = 0.0
    
    if n > 5:
        y_centered = y - np.mean(y)
        var = np.var(y)
        if var > 1e-12:
            autocorr_lag1 = float(np.sum(y_centered[:-1] * y_centered[1:]) / ((n - 1) * var))
            autocorr_lag2 = float(np.sum(y_centered[:-2] * y_centered[2:]) / ((n - 2) * var))

    # 4. Pont-Style Binning Scaling Factor (Red Noise Excess)
    if bin_sizes is None:
        bin_sizes = [5, 10, 20, 50]
        
    pont_excess = {}
    for M in bin_sizes:
        if n >= 2 * M:
            # Reshape into non-overlapping bins of size M
            n_bins = n // M
            binned_y = np.mean(y[:n_bins * M].reshape(n_bins, M), axis=1)
            
            # Robust scatter of binned residuals
            med_b = np.median(binned_y)
            mad_b = np.median(np.abs(binned_y - med_b)) / 0.6745
            
            # Expected scatter if pure white noise
            expected_scatter = sigma_white / np.sqrt(M)
            
            # Pont excess ratio: actual binned scatter / expected white scatter
            ratio = float(mad_b / (expected_scatter + 1e-12))
            pont_excess[M] = ratio
        else:
            pont_excess[M] = 1.0

    # Classify whether significant red noise exists:
    # 1. ACF at lag 1 is high (> 0.2)
    # 2. Pont excess factor at block size 10 is > 1.3
    has_red_noise = bool(autocorr_lag1 > 0.2 or (10 in pont_excess and pont_excess[10] > 1.3))

    return {
        "sigma_white": float(sigma_white),
        "sigma_total": float(sigma_total),
        "autocorr_lag1": float(autocorr_lag1),
        "autocorr_lag2": float(autocorr_lag2),
        "pont_excess": pont_excess,
        "has_red_noise": has_red_noise,
    }
