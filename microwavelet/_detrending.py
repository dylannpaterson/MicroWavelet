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
from astropy.timeseries import LombScargle
from scipy import optimize
from scipy.interpolate import CubicSpline
from scipy.stats import median_abs_deviation

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
        maximum_frequency=1.0 / min_period,
        minimum_frequency=1.0 / max_period,
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
        for i in range(len(bins) - 1):
            m = (phase >= bins[i]) & (phase < bins[i + 1])
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


class DummyKernel:
    def __init__(self):
        self.theta = np.array([1.0])


class WhittakerSmoother:
    """
    A 100% backward-compatible wrapper around the Whittaker-Eilers smoother
    that mimics the sklearn GaussianProcessRegressor interface.
    """

    def __init__(
        self,
        bin_centers,
        bin_values,
        bin_errors,
        W_diag,
        D_TD,
        lam,
        tr_H,
        rss,
        N_valid,
        sum_log_var,
    ):
        self.bin_centers = bin_centers
        self.bin_values = bin_values
        self.bin_errors = bin_errors
        self.W_diag = W_diag
        self.D_TD = D_TD
        self.lam = lam
        self.tr_H = tr_H
        self.rss = rss
        self.N_valid = N_valid
        self.sum_log_var = sum_log_var
        self.kernel_ = DummyKernel()

        # Set up periodic cubic spline for evaluation
        # We append a wrapped first element at the end of the period
        x_cs = np.concatenate([bin_centers, [bin_centers[0] + 1.0]])
        y_cs = np.concatenate([bin_values, [bin_values[0]]])
        self.cs = CubicSpline(x_cs, y_cs, bc_type="periodic")

    def predict(self, X, return_std=False):
        X_arr = np.asarray(X)
        if X_arr.ndim == 2:
            phase = X_arr[:, 0]
        else:
            phase = X_arr

        # Map phase to [bin_centers[0], bin_centers[0] + 1.0)
        phase_mapped = (phase - self.bin_centers[0]) % 1.0 + self.bin_centers[0]
        y_pred = self.cs(phase_mapped)

        if return_std:
            return y_pred, np.zeros_like(y_pred)
        return y_pred

    def log_marginal_likelihood(self, *args, **kwargs):
        # Return the binned Gaussian log-likelihood as a robust proxy for period folding quality.
        # This prevents complexity-penalty bias when comparing different periods.
        if self.N_valid < 4:
            return -1e20
        log_like = -0.5 * (self.rss + self.sum_log_var)
        return log_like


def fit_periodic_gp_robust(t, y, y_err, period, n_bins=None, max_iter=4, sigma_clip=5.0):
    """
    Fits a smooth periodic Whittaker-Eilers smoother (Smoothing Spline)
    to the phase-folded light curve.

    Uses iterative sigma-clipping (positive outliers only) to identify and
    protect transient microlensing brightening from being fitted as baseline.

    Returns
    -------
    gp : WhittakerSmoother
        An object wrapping the fitted Whittaker smoother and mimicking the
        GaussianProcessRegressor interface.
    phase : np.ndarray  (same length as t)
    valid_mask : np.ndarray[bool]
        True where the data was kept (not clipped as a transient).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    y_err = np.asarray(y_err)
    phase = (t % period) / period
    # Initialize valid_mask using robust positive 3-sigma MAD clipping on the quiescent baseline
    # to protect the baseline from massive transients right away.
    y_low = y[y <= np.median(y)]
    if len(y_low) > 5:
        base_level = np.median(y_low)
        mad = median_abs_deviation(y_low)
        sigma = 1.4826 * mad if mad > 0 else np.std(y_low)
    else:
        base_level = np.median(y)
        mad = median_abs_deviation(y)
        sigma = 1.4826 * mad if mad > 0 else np.std(y)

    valid_mask = y < base_level + 3.0 * sigma
    gp = None

    # Dynamically set n_bins based on the number of active points
    # to ensure robust bin occupancy and prevent empty-bin GCV issues.
    if n_bins is None:
        n_bins = int(np.clip(np.sum(valid_mask) // 8, 40, 120))

    # Construct the periodic second-difference Laplacian matrix D of size n_bins x n_bins
    D = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        D[i, (i - 1) % n_bins] = 1.0
        D[i, i] = -2.0
        D[i, (i + 1) % n_bins] = 1.0
    D_TD = D.T @ D

    for _ in range(max_iter):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        bin_values = np.zeros(n_bins)
        bin_errors = np.zeros(n_bins)
        W_diag = np.zeros(n_bins)

        for k in range(n_bins):
            in_bin = (phase >= bins[k]) & (phase < bins[k + 1]) & valid_mask
            n_pts = np.sum(in_bin)
            if n_pts > 0:
                median_val = np.median(y[in_bin])
                bin_values[k] = median_val
                std_val = np.std(y[in_bin])
                se_val = 1.2533 * std_val / np.sqrt(n_pts) if n_pts > 1 else y_err[in_bin][0]

                # Stabilized bin error estimation using data point uncertainties
                min_err = np.mean(y_err[in_bin]) / np.sqrt(n_pts)
                bin_errors[k] = max(se_val, min_err, 1e-4)
                W_diag[k] = 1.0 / bin_errors[k] ** 2
            else:
                bin_values[k] = 0.0
                bin_errors[k] = 1e-4
                W_diag[k] = 0.0

        N_valid = np.sum(W_diag > 0)
        if N_valid < 4:
            W_diag = np.ones(n_bins)
            bin_values = np.ones(n_bins) * np.median(y[valid_mask])
            bin_errors = np.ones(n_bins) * 0.05
            N_valid = n_bins

        # Optimize lambda using GCV
        def gcv_score(
            log_lam,
            W_diag=W_diag,
            bin_values=bin_values,
            N_valid=N_valid,
        ):
            lam = 10**log_lam
            A = np.diag(W_diag) + lam * D_TD
            try:
                y_hat = np.linalg.solve(A, W_diag * bin_values)
                H = np.linalg.solve(A, np.diag(W_diag))
                tr_H = np.trace(H)
                rss = np.sum(W_diag * (bin_values - y_hat) ** 2)
                denom = (1.0 - tr_H / N_valid) ** 2
                if denom < 1e-6:
                    return 1e10
                return (rss / N_valid) / denom
            except np.linalg.LinAlgError:
                return 1e10

        # Grid search (lower bound of 0.0 / lambda=1.0 protects empty bins from ill-conditioned ringing)
        log_lams = np.linspace(0.0, 6.0, 30)
        scores = [gcv_score(log_lam) for log_lam in log_lams]
        best_idx = np.argmin(scores)
        best_log_lam = log_lams[best_idx]

        # Refinement
        lam = 10**best_log_lam
        try:
            bounds = (max(0.0, best_log_lam - 1.0), min(6.0, best_log_lam + 1.0))
            res = optimize.minimize_scalar(
                gcv_score, bounds=bounds, method="bounded", options={"xatol": 1e-3}
            )
            if res.success:
                lam = 10**res.x
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass

        # Perform the final solve with optimized lambda
        A = np.diag(W_diag) + lam * D_TD
        try:
            y_hat = np.linalg.solve(A, W_diag * bin_values)
            H = np.linalg.solve(A, np.diag(W_diag))
            tr_H = np.trace(H)
            rss = np.sum(W_diag * (bin_values - y_hat) ** 2)
        except np.linalg.LinAlgError:
            y_hat = bin_values
            tr_H = n_bins
            rss = 0.0

        valid_bins = W_diag > 0
        sum_log_var = np.sum(np.log(2.0 * np.pi * bin_errors[valid_bins] ** 2))

        gp = WhittakerSmoother(
            bin_centers=bin_centers,
            bin_values=y_hat,
            bin_errors=bin_errors,
            W_diag=W_diag,
            D_TD=D_TD,
            lam=lam,
            tr_H=tr_H,
            rss=rss,
            N_valid=N_valid,
            sum_log_var=sum_log_var,
        )

        y_model = gp.predict(phase)
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
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
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
    # We prefer the shorter period unless it's much worse (Delta LML < -2.0).
    for _ in range(3):
        half_p = 0.5 * current_p
        if half_p < min_period:
            break
        half_lml = get_gp_evidence(half_p)
        if half_lml > (current_lml - 2.0):
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
        if p <= 0:
            return 1e10
        try:
            gp, phase, m = fit_periodic_gp_robust(t_fit, y_fit, ye_fit, p, max_iter=2)
            y_pred = gp.predict(phase[:, np.newaxis])
            return np.sqrt(np.mean((y_fit[m] - y_pred[m]) ** 2))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return 1e10

    try:
        res = optimize.minimize_scalar(
            objective,
            bounds=(initial_period * 0.90, initial_period * 1.10),
            method="bounded",
            options={"xatol": 1e-6},
        )
        if res.success:
            return res.x
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
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
        p["t"],
        p["y"],
        p["y_err"],
        min_period=min_period,
        max_period=max_period,
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
