"""
microwavelet._detrending
=========================
Periodic GP baseline detrending for multi-filter light curves.

Pipeline
--------
1. find_shared_period        – Lomb-Scargle period search on the primary band
2. fit_periodic_gp_robust    – Iterative sigma-clipping GPR on phase-folded data
3. resolve_fundamental_period – Bayesian Occam's Razor to find fundamental period
4. fine_tune_period          – GP-based period optimization
5. detrend_light_curve_periodic – Orchestrates 1-4 across all bands
"""

import numpy as np
import warnings
from scipy import optimize
from scipy.stats import median_abs_deviation
from astropy.timeseries import LombScargle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning


# ---------------------------------------------------------------------------
# 1. Period search
# ---------------------------------------------------------------------------

def find_shared_period(t, y, dy, min_period=1.0, max_period=10.0):
    """
    Finds the dominant periodic baseline modulation using Lomb-Scargle
    plus a robust harmonic (PDM) check to avoid transient bias and aliases.

    Returns
    -------
    best_period : float
    frequency : np.ndarray
    power : np.ndarray
    clean_mask : np.ndarray[bool]
        Mask where True means data was kept (not clipped as transient).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy)

    # 1. Aggressive pre-clipping for period search (2.5 sigma MAD)
    # This prevents high-amplitude ML events from biasing the initial search.
    mad = median_abs_deviation(y)
    sigma = 1.4826 * mad if mad > 0 else np.std(y)
    clean_mask = y < np.median(y) + 2.5 * sigma
    
    t_c, y_c, dy_c = t[clean_mask], y[clean_mask], dy[clean_mask]
    
    if len(t_c) < 20:
        t_c, y_c, dy_c = t, y, dy
        clean_mask = np.ones_like(y, dtype=bool)

    # 2. Initial LS search
    frequency, power = LombScargle(t_c, y_c, dy_c).autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=10,
    )
    p_ls = 1.0 / frequency[np.argmax(power)]

    # 3. Robust Harmonic Check (PDM scoring)
    # We test P/2, P, and 2P to find the fundamental that folds cleanest.
    def get_pdm_score(p):
        if p < min_period or p > max_period:
            return 1e10
        phase = (t_c % p) / p
        # Simple binned MAD score: lower is better
        bins = np.linspace(0.0, 1.0, 20)
        scatters = []
        for i in range(len(bins)-1):
            m = (phase >= bins[i]) & (phase < bins[i+1])
            if np.sum(m) > 2:
                scatters.append(median_abs_deviation(y_c[m]))
        return np.mean(scatters) if scatters else 1e10

    candidates = [p_ls, 0.5 * p_ls, 2.0 * p_ls]
    scores = [get_pdm_score(c) for c in candidates]
    best_period = candidates[np.argmin(scores)]

    return best_period, frequency, power, clean_mask


# ---------------------------------------------------------------------------
# 2. GPR phase-folded fit
# ---------------------------------------------------------------------------

def fit_periodic_gp_robust(t, y, y_err, period, n_bins=120, max_iter=4, sigma_clip=3.0):
    """
    Fits a smooth periodic Gaussian Process to the phase-folded light curve.

    Uses iterative sigma-clipping (positive outliers only) to identify and
    protect transient microlensing brightening from being fitted as baseline.

    Returns
    -------
    gp : fitted GaussianProcessRegressor
    phase : np.ndarray  (same length as t)
    valid_mask : np.ndarray[bool]
        True where the data was kept (not clipped as a transient).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    y_err = np.asarray(y_err)
    phase = (t % period) / period
    valid_mask = np.ones_like(y, dtype=bool)
    gp = None

    # ExpSineSquared periodic kernel with wide length-scale bounds allows fitting
    # extremely sharp CV eclipses and smooth pulsations alike.
    kernel = C(1.0, (1e-3, 1e3)) * ExpSineSquared(
        length_scale=0.1,
        periodicity=1.0,
        length_scale_bounds=(0.02, 2.0),
        periodicity_bounds="fixed",
    )

    for _ in range(max_iter):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        bin_values = []
        bin_errors = []

        for k in range(n_bins):
            in_bin = (phase >= bins[k]) & (phase < bins[k + 1]) & valid_mask
            n_pts = np.sum(in_bin)
            if n_pts > 0:
                median_val = np.median(y[in_bin])
                bin_values.append(median_val)
                std_val = np.std(y[in_bin])
                se_val = 1.2533 * std_val / np.sqrt(n_pts) if n_pts > 1 else y_err[in_bin][0]
                bin_errors.append(max(se_val, 1e-4))
            else:
                bin_values.append(np.nan)
                bin_errors.append(np.nan)

        bin_values = np.array(bin_values)
        bin_errors = np.array(bin_errors)

        # Interpolate empty bins
        nan_mask = np.isnan(bin_values)
        if np.any(nan_mask):
            non_nan_x = bin_centers[~nan_mask]
            non_nan_y = bin_values[~nan_mask]
            non_nan_err = bin_errors[~nan_mask]
            if len(non_nan_x) > 1:
                x_pad = np.concatenate([non_nan_x - 1.0, non_nan_x, non_nan_x + 1.0])
                y_pad = np.tile(non_nan_y, 3)
                err_pad = np.tile(non_nan_err, 3)
                bin_values[nan_mask] = np.interp(bin_centers[nan_mask], x_pad, y_pad)
                bin_errors[nan_mask] = np.interp(bin_centers[nan_mask], x_pad, err_pad)
            else:
                bin_values[nan_mask] = np.median(y[valid_mask])
                bin_errors[nan_mask] = 0.05

        # Smoothly wrap boundary for perfect phase continuity
        boundary_val = 0.5 * (bin_values[0] + bin_values[-1])
        x_spline = np.concatenate([[0.0], bin_centers, [1.0]])
        y_spline = np.concatenate([[boundary_val], bin_values, [boundary_val]])
        err_spline = np.concatenate([[bin_errors[0]], bin_errors, [bin_errors[0]]])

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=err_spline ** 2 + 1e-6,
            n_restarts_optimizer=5,
            random_state=42,
        )
        
        # Suppress convergence warnings during repeated fits
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x_spline[:, np.newaxis], y_spline)

        y_model = gp.predict(phase[:, np.newaxis])
        residuals = y - y_model

        mad = median_abs_deviation(residuals[valid_mask])
        sigma = 1.4826 * mad if mad > 0 else np.std(residuals[valid_mask])
        sigma = max(sigma, 1e-6)

        # Asymmetric clipping: only remove positive outliers (protect microlensing peaks)
        outliers = residuals > sigma_clip * sigma
        if np.sum(outliers & valid_mask) == 0:
            break
        valid_mask = valid_mask & (~outliers)

    return gp, phase, valid_mask


# ---------------------------------------------------------------------------
# 3. Alias resolution
# ---------------------------------------------------------------------------

def resolve_fundamental_period(t, y, y_err, period, mask, min_period=1.0, max_period=10.0):
    """
    Finds the fundamental period by iteratively doubling and halving 
    candidates using Bayesian Log-Marginal Likelihood (LML).

    1. Doubling: Try to find the full orbital cycle where complex shapes
       (e.g., ellipsoidal + eclipses) are fully resolved.
    2. Halving: Apply Aggressive Occam's Razor to find the simplest fundamental.
    """
    t_f, y_f, ye_f = np.asarray(t)[mask], np.asarray(y)[mask], np.asarray(y_err)[mask]
    
    def get_gp_evidence(p):
        if p < min_period or p > max_period:
            return -1e20
        try:
            # We use max_iter=2 for a reasonably converged fit
            gp, phase, m = fit_periodic_gp_robust(t_f, y_f, ye_f, p, max_iter=2)
            return gp.log_marginal_likelihood(gp.kernel_.theta)
        except:
            return -1e20

    current_p = period
    current_lml = get_gp_evidence(current_p)

    # 1. Iterative Doubling
    # We double as long as it's significantly better (Delta LML > 5).
    for _ in range(3):
        double_p = 2.0 * current_p
        if double_p > max_period:
            break
        double_lml = get_gp_evidence(double_p)
        if double_lml > (current_lml + 5.0):
            current_p = double_p
            current_lml = double_lml
        else:
            break

    # 2. Iterative Halving (Aggressive Occam's Razor)
    # We prefer the shorter period unless it's much worse (Delta LML < -10).
    for _ in range(3):
        half_p = 0.5 * current_p
        if half_p < min_period:
            break
        half_lml = get_gp_evidence(half_p)
        if half_lml > (current_lml - 10.0):
            current_p = half_p
            current_lml = half_lml
        else:
            break
            
    return current_p


# ---------------------------------------------------------------------------
# 4. Period fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_period(t, y, y_err, initial_period, mask, baseline_func=None):
    """
    Fine-tunes the period by minimizing the residual RMS of a full 
    robust periodic GP fit to the phase-folded data.
    """
    t_fit = np.asarray(t)
    y_fit = np.asarray(y)
    ye_fit = np.asarray(y_err)
    
    if len(t_fit[mask]) < 20:
        return initial_period

    def objective(p):
        if p <= 0: return 1e10
        try:
            gp, phase, m = fit_periodic_gp_robust(t_fit, y_fit, ye_fit, p, max_iter=2)
            y_pred = gp.predict(phase[:, np.newaxis])
            return np.sqrt(np.mean((y_fit[m] - y_pred[m])**2))
        except:
            return 1e10

    try:
        res = optimize.minimize_scalar(
            objective, 
            bounds=(initial_period * 0.90, initial_period * 1.10), 
            method='bounded',
            options={'xatol': 1e-6}
        )
        if res.success:
            return res.x
    except:
        pass

    return initial_period


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------

def detrend_light_curve_periodic(band_data, min_period=1.0, max_period=10.0, baseline_func=None):
    """
    Full periodic baseline detrending pipeline for multi-filter observations.
    """
    # Primary band = most data points
    primary_band = max(band_data, key=lambda b: len(band_data[b]["t"]))
    p = band_data[primary_band]

    # 1. Lomb-Scargle period search
    ls_period, _, _, search_mask = find_shared_period(
        p["t"], p["y"], p["y_err"],
        min_period=min_period, max_period=max_period,
    )

    # 2. Rough mask
    _, _, rough_mask = fit_periodic_gp_robust(p["t"], p["y"], p["y_err"], ls_period, max_iter=3)

    # 3. Alias resolution
    resolved_period = resolve_fundamental_period(p["t"], p["y"], p["y_err"], ls_period, rough_mask)

    # 4. Fine-tune
    optimized_period = fine_tune_period(
        p["t"], p["y"], p["y_err"], resolved_period, rough_mask, baseline_func
    )

    # 5. Per-band GPR detrending
    detrended_bands = {}
    for b, data in band_data.items():
        t = np.asarray(data["t"])
        y = np.asarray(data["y"])
        y_err = np.asarray(data["y_err"])

        gp, phase, mask = fit_periodic_gp_robust(t, y, y_err, optimized_period)
        baseline_model = gp.predict(phase[:, np.newaxis])

        y_detrended = y / baseline_model
        y_err_scaled = y_err * (y_detrended / np.where(y > 0, y, 1.0))

        detrended_bands[b] = {
            "t": t,
            "y_raw": y,
            "y_detrended": y_detrended,
            "y_err": y_err_scaled,
            "baseline_model": baseline_model,
            "phase": phase,
            "outlier_mask": mask,
        }

    detrending_info = {
        "period_days": optimized_period,
        "search_mask": search_mask,
        "period_search": {
            "initial_ls": ls_period,
            "alias_resolved": resolved_period,
            "optimized": optimized_period,
        },
    }

    return detrended_bands, detrending_info
