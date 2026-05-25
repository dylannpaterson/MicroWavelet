"""
microlens_cwt._detrending
=========================
Periodic GP baseline detrending for multi-filter light curves.

Pipeline
--------
1. find_shared_period        – Lomb-Scargle period search on the primary band
2. fit_periodic_gp_robust    – Iterative sigma-clipping GPR on phase-folded data
3. resolve_half_period_alias – χ² comparison P vs 2P to catch EB/CV aliases
4. fine_tune_period          – Non-linear period fine-tuning via scipy.optimize
5. detrend_light_curve_periodic – Orchestrates 1-4 across all bands
"""

import numpy as np
from scipy import optimize
from scipy.stats import median_abs_deviation
from astropy.timeseries import LombScargle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, ConstantKernel as C


# ---------------------------------------------------------------------------
# 1. Period search
# ---------------------------------------------------------------------------

def find_shared_period(t, y, dy, min_period=1.0, max_period=10.0):
    """
    Finds the dominant periodic baseline modulation using Lomb-Scargle.

    Parameters
    ----------
    t, y, dy : array-like
        Times, fluxes, and flux uncertainties for the primary band.
    min_period, max_period : float
        Search bounds in days.

    Returns
    -------
    best_period : float
    frequency : np.ndarray
    power : np.ndarray
    """
    t = np.asarray(t)
    y = np.asarray(y)
    dy = np.asarray(dy)

    frequency, power = LombScargle(t, y, dy).autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=10,
    )
    best_freq = frequency[np.argmax(power)]
    best_period = 1.0 / best_freq
    return best_period, frequency, power


# ---------------------------------------------------------------------------
# 2. GPR phase-folded fit
# ---------------------------------------------------------------------------

def fit_periodic_gp_robust(t, y, y_err, period, n_bins=120, max_iter=4, sigma_clip=3.0):
    """
    Fits a smooth periodic Gaussian Process to the phase-folded light curve.

    Uses iterative sigma-clipping (positive outliers only) to identify and
    protect transient microlensing brightening from being fitted as baseline.

    Parameters
    ----------
    t, y, y_err : array-like
    period : float
    n_bins : int
        Number of phase bins for initial GP training data.
    max_iter : int
        Maximum sigma-clipping iterations.
    sigma_clip : float
        Sigma threshold for clipping (asymmetric: only positive outliers).

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

def resolve_half_period_alias(t, y, y_err, period, mask):
    """
    Resolves the eclipsing binary / CV half-period alias via χ² comparison.

    Restricted to the first 300 days to eliminate phase drift from long
    baseline observations.

    Returns
    -------
    float : The resolved period (either ``period`` or ``2 * period``).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    y_err = np.asarray(y_err)
    mask = np.asarray(mask)

    local_mask = (t < t.min() + 300.0) & mask
    if np.sum(local_mask) < 20:
        return period

    t_l, y_l, ye_l = t[local_mask], y[local_mask], y_err[local_mask]

    try:
        gp_P, phase_P, _ = fit_periodic_gp_robust(t_l, y_l, ye_l, period, max_iter=2)
        chi2_P = np.sum(((y_l - gp_P.predict(phase_P[:, np.newaxis])) / ye_l) ** 2)

        double_period = 2.0 * period
        gp_2P, phase_2P, _ = fit_periodic_gp_robust(t_l, y_l, ye_l, double_period, max_iter=2)
        chi2_2P = np.sum(((y_l - gp_2P.predict(phase_2P[:, np.newaxis])) / ye_l) ** 2)

        if chi2_P - chi2_2P > 20.0:
            return double_period
    except Exception:
        pass

    return period


# ---------------------------------------------------------------------------
# 4. Period fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_period(t, y, y_err, initial_period, mask, baseline_func=None):
    """
    Fine-tunes the period via non-linear optimisation on the unmasked baseline.

    Parameters
    ----------
    baseline_func : callable, optional
        Custom baseline model ``f(t, period, t0_offset) -> y``.
        If None, assumes a simple harmonic sine.

    Returns
    -------
    float : Optimised period (falls back to ``initial_period`` on failure).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    y_err = np.asarray(y_err)
    mask = np.asarray(mask)

    t_fit, y_fit, ye_fit = t[mask], y[mask], y_err[mask]
    if len(t_fit) < 20:
        return initial_period

    if baseline_func is None:
        def loss(params):
            period, amp, phase_shift, offset = params
            if period < 0.1:
                return 1e10
            model = offset + amp * np.sin(2.0 * np.pi * t_fit / period + phase_shift)
            return np.sum(((y_fit - model) / ye_fit) ** 2)

        p_init = [initial_period, np.std(y_fit) * np.sqrt(2), 0.0, 1.0]
        bounds = [
            (initial_period * 0.95, initial_period * 1.05),
            (1e-4, 0.5),
            (-2.0 * np.pi, 2.0 * np.pi),
            (0.8, 1.2),
        ]
    else:
        def loss(params):
            period, t0_offset = params
            if period < 0.1:
                return 1e10
            return np.sum(((y_fit - baseline_func(t_fit, period, t0_offset)) / ye_fit) ** 2)

        p_init = [initial_period, 0.0]
        bounds = [
            (initial_period * 0.98, initial_period * 1.02),
            (-initial_period, initial_period),
        ]

    try:
        res = optimize.minimize(loss, p_init, method="L-BFGS-B", bounds=bounds)
        if res.success:
            return res.x[0]
    except Exception:
        pass

    return initial_period


# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------

def detrend_light_curve_periodic(band_data, min_period=1.0, max_period=10.0, baseline_func=None):
    """
    Full periodic baseline detrending pipeline for multi-filter observations.

    Steps
    -----
    1. Joint Lomb-Scargle period search on the highest-cadence band.
    2. Rough mask via sigma-clipping GPR to protect microlensing transients.
    3. Orbital alias resolution (P vs 2P).
    4. Non-linear period fine-tuning.
    5. Per-band robust GPR detrending and division.

    Parameters
    ----------
    band_data : dict
        ``{band_name: {"t": ..., "y": ..., "y_err": ...}}``
    min_period, max_period : float
        Period search bounds (days).
    baseline_func : callable, optional
        Custom baseline model for period fine-tuning.

    Returns
    -------
    detrended_bands : dict
        Per-band dict with keys:
        ``t, y_raw, y_detrended, y_err, baseline_model, phase, outlier_mask``
    detrending_info : dict
        ``{period_days, period_search: {initial_ls, alias_resolved, optimized}}``
    """
    # Primary band = most data points
    primary_band = max(band_data, key=lambda b: len(band_data[b]["t"]))
    p = band_data[primary_band]

    # 1. Lomb-Scargle period search
    ls_period, _, _ = find_shared_period(
        p["t"], p["y"], p["y_err"],
        min_period=min_period, max_period=max_period,
    )

    # 2. Rough mask
    _, _, rough_mask = fit_periodic_gp_robust(p["t"], p["y"], p["y_err"], ls_period, max_iter=3)

    # 3. Alias resolution
    resolved_period = resolve_half_period_alias(p["t"], p["y"], p["y_err"], ls_period, rough_mask)

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
            "outlier_mask": mask,   # True = kept (not sigma-clipped as transient)
        }

    detrending_info = {
        "period_days": optimized_period,
        "period_search": {
            "initial_ls": ls_period,
            "alias_resolved": resolved_period,
            "optimized": optimized_period,
        },
    }

    return detrended_bands, detrending_info
