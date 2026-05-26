"""
microwavelet._wavelet
======================
Scale-space CWT peak/dip finder using Paczynski wavelet kernels.

Key design choices
------------------
- Kernels are L1-normalised (sum(|k|) = 1).  This makes the CWT coefficient
  scale-independent, allowing the consensus Z-score (SNR) to be compared
  across very different timescales.
- u0 is estimated from the **peak flux** (f_interp[peak_idx]), NOT from the
  wavelet coefficient.  The L1-normalised even kernel (ke) is built with a
  fixed template u0=0.05, so the coefficient encodes a shape-mismatch factor
  that would bias any amplitude-derived u0 estimate.  Reading the flux
  directly avoids this.
- tE is bias-corrected for u0 mismatch.  The CWT scale at maximum response is
  NOT the true tE when the event u0 differs from the template u0=0.05.  A
  broader (larger u0) event has a wider 2nd-derivative structure, so the CWT
  inflates the scale to compensate.  We correct by the ratio of 2nd-derivative
  RMS widths: tE_true ≈ tE_cwt × σ(u0_template) / σ(u0_event).
- Edge and seasonal gap suppression is done via a unified gap-contamination
  mask (eta), which down-weights CWT coefficients by (1 - eta)^2.
"""

import numpy as np
from scipy import interpolate
from scipy.signal import fftconvolve, find_peaks, windows


# ---------------------------------------------------------------------------
# Paczynski kernel generator
# ---------------------------------------------------------------------------

def get_kernels(tk, tE, u0=0.05):
    """
    Generates the even (2nd-derivative) and odd (1st-derivative)
    Paczynski wavelet kernels at a given Einstein timescale.

    Both kernels are L1-normalised (``sum(|k|) = 1``).

    Parameters
    ----------
    tk : np.ndarray
        Symmetric time array centred on zero (e.g. ``np.arange(-N, N+1)*dt``).
    tE : float
        Einstein crossing time (same units as tk).
    u0 : float
        Template impact parameter.  Default 0.05 (compact, high-SNR template).

    Returns
    -------
    ke, ko : np.ndarray, np.ndarray
        Even (symmetric) and odd (anti-symmetric) kernels.
    """
    u = np.sqrt(u0 ** 2 + (tk / tE) ** 2)
    A = (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))
    dt = tk[1] - tk[0]

    first_deriv = np.gradient(A, dt)
    second_deriv = np.gradient(first_deriv, dt)

    ke = -second_deriv
    ke -= np.mean(ke)
    ke /= (np.sum(np.abs(ke)) + 1e-12)

    ko = first_deriv
    ko -= np.mean(ko)
    ko /= (np.sum(np.abs(ko)) + 1e-12)

    return ke, ko


# ---------------------------------------------------------------------------
# Symmetry metric
# ---------------------------------------------------------------------------

def calculate_symmetry(t, y, t0, tE_val):
    """
    Quantifies peak/dip symmetry as RMS(y - y_flipped) / peak-to-trough height
    over a ±2·tE window centred on t0.

    Returns
    -------
    float : 0.0 = perfectly symmetric, >0.3 = likely asymmetric noise excursion.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    mask = (t >= t0 - 2 * tE_val) & (t <= t0 + 2 * tE_val)
    t_win, y_win = t[mask] - t0, y[mask]
    if len(y_win) < 5:
        return 0.0

    t_grid = np.linspace(-2 * tE_val, 2 * tE_val, 200)
    y_grid = np.interp(t_grid, t_win, y_win)
    y_flipped = y_grid[::-1]

    rms_diff = np.sqrt(np.mean((y_grid - y_flipped) ** 2))
    peak_h = np.max(y_grid) - np.min(y_grid)
    return rms_diff / (peak_h + 1e-12)


# ---------------------------------------------------------------------------
# Analytical u0 estimator
# ---------------------------------------------------------------------------

def _u0_from_amplification(A):
    """
    Analytically inverts the Paczynski amplification formula to recover u0.

    A = (u² + 2) / (u · √(u² + 4))

    Rearranges to a quadratic in x = u²:
      (A²-1)·x² + 4·(A²-1)·x - 4 = 0
    Positive root: x = 2·(A / √(A²-1) - 1)

    For A >> 1 (small u0): u0 ≈ 1/A.

    Returns
    -------
    float : Estimated u0, or np.nan for A <= 1.
    """
    if A <= 1.001:
        return np.nan
    A2m1 = A ** 2 - 1.0
    x = 2.0 * (A / np.sqrt(A2m1) - 1.0)
    return np.sqrt(max(x, 1e-6))


# ---------------------------------------------------------------------------
# tE bias correction
# ---------------------------------------------------------------------------

def _tE_correction_factor(u0_event, u0_template=0.05):
    """
    Compute the multiplicative factor to correct the CWT tE bias due to
    template u0 mismatch.

    Background
    ----------
    The even kernel is built at ``u0_template=0.05`` (a narrow, high-mag
    template).  For events with larger u0, the Paczynski 2nd-derivative
    structure is *wider* in time.  The CWT maximum occurs at a scale that
    makes the template width match the data width:

        tE_scan = tE_true * r_peak(u0_event)
        → tE_true = tE_scan / r_peak(u0_event)

    where r_peak(u0) is the exact numerical peak ratio computed via 
    scale-invariant CWT.

    Parameters
    ----------
    u0_event : float
        Estimated u0 of the real event (from flux inversion).
    u0_template : float
        u0 used to build the CWT kernel (default 0.05).

    Returns
    -------
    float : correction factor f such that ``tE_true ≈ tE_cwt * f``.
            Returns 1.0 if u0_event is invalid.
    """
    if u0_event is None or np.isnan(u0_event) or u0_event <= 0:
        return 1.0

    # Clip u0 to the fitted interval [0.01, 1.0] to prevent polynomial divergence
    u0_clipped = np.clip(u0_event, 0.01, 1.0)

    # Weighted degree-5 polynomial coefficients from numerical peak fitting:
    coeffs = [12.07074638, -29.26124165, 28.15495458, -18.91458072, 22.28322211, -0.06346686]
    poly = np.poly1d(coeffs)
    r_val = poly(u0_clipped)
    
    # Avoid division by zero or negative values
    if r_val < 0.01:
        return 1.0
        
    return 1.0 / r_val


# ---------------------------------------------------------------------------
# Gamma sweep tE estimator
# ---------------------------------------------------------------------------

def detect_cwt_peaks(
    t_obs,
    y_obs,
    y_obs_err=None,
    tE_scales=None,
    dt=0.02,
    cwt_threshold=12.0,
    t_grid=None,
    baseline_flux=None,
    interpolator="linear",
    min_dchi2=None,
):
    """
    Scale-space CWT peak and dip finder using Paczynski wavelet kernels.

    Parameters
    ----------
    t_obs : array-like
        Observed times (days).
    y_obs : array-like
        Baseline-subtracted flux (so quiescent ≈ 0).
    y_obs_err : array-like, optional
        Observational uncertainties on y_obs. Used for weighted regression and dchi2.
    tE_scales : np.ndarray, optional
        Log-spaced grid of Einstein timescale scales to probe (days).
        Defaults to 120 scales from 0.02 to 200 days.
    dt : float
        Regular grid spacing for CWT interpolation (days).
    cwt_threshold : float
        Minimum robust Z-score (MAD-normalised) for a detection.
    t_grid : np.ndarray, optional
        Pre-computed regular time grid.  Auto-generated if None.
    baseline_flux : float, optional
        Quiescent flux level *before* baseline subtraction.  Needed to
        compute magnification A and hence u0.  Pass None to skip u0 estimation.
    interpolator : str
        Choice of interpolator: "linear" or "weighted".
        "weighted" performs a Nadaraya-Watson local Gaussian kernel regression
        weighted by 1/y_obs_err^2.
    min_dchi2 : float, optional
        Minimum dchi2 (chi2_null - chi2_lens) required for a detection.

    Returns
    -------
    dict with keys:
        even_peaks, odd_peaks : list[dict]
            Detected anomalies from even/odd kernel morphologies.
        t_grid, f_interp, grid_mask : np.ndarray
            Interpolated grid and coverage mask.
        consensus_1d_even, consensus_1d_odd : np.ndarray
        norm_map_even, norm_map_odd : np.ndarray
        tE_scales : np.ndarray
    """
    t_obs = np.asarray(t_obs)
    y_obs = np.asarray(y_obs)

    # Sort observations by time to ensure searchsorted and interpolation work correctly
    sort_idx = np.argsort(t_obs)
    t_obs = t_obs[sort_idx]
    y_obs = y_obs[sort_idx]
    if y_obs_err is not None:
        y_obs_err = np.asarray(y_obs_err)[sort_idx]

    # Pre-calculate weights for weighted fitting and weighted interpolation
    if y_obs_err is None:
        w = np.ones_like(y_obs)
    else:
        err_clean = np.where((y_obs_err > 1e-12) & np.isfinite(y_obs_err), y_obs_err, 1e-12)
        w = 1.0 / (err_clean ** 2)

    # Calculate median observation spacing (cadence) at the top
    diffs = np.diff(t_obs)
    median_spacing = np.median(diffs) if len(diffs) > 0 else dt

    if tE_scales is None:
        tE_scales = np.logspace(np.log10(0.02), np.log10(200.0), 120)

    t_start, t_end = t_obs[0] - 10.0, t_obs[-1] + 10.0
    if t_grid is None:
        t_grid = np.arange(t_start, t_end, dt)

    # Mark grid cells near actual observations
    grid_mask = np.zeros(len(t_grid), dtype=bool)
    for to in t_obs:
        idx_m = int((to - t_grid[0]) / dt)
        grid_mask[max(0, idx_m - 10): min(len(t_grid), idx_m + 10)] = True

    # Interpolate onto regular grid
    f_interp = interpolate.interp1d(
        t_obs, y_obs, bounds_error=False, fill_value=(y_obs[0], y_obs[-1])
    )(t_grid)

    if interpolator == "weighted":
        # Adaptive bandwidth based on the median observation spacing (cadence)
        h = max(2.0 * dt, 1.5 * median_spacing)
        active_indices = np.where(grid_mask)[0]
        left_idx = np.searchsorted(t_obs, t_grid[active_indices] - 4.0 * h, side="left")
        right_idx = np.searchsorted(t_obs, t_grid[active_indices] + 4.0 * h, side="right")

        for idx_in_active, g_idx in enumerate(active_indices):
            l_idx = left_idx[idx_in_active]
            r_idx = right_idx[idx_in_active]
            if r_idx - l_idx > 0:
                t_sub = t_obs[l_idx:r_idx]
                y_sub = y_obs[l_idx:r_idx]
                w_sub = w[l_idx:r_idx]

                dts = t_sub - t_grid[g_idx]
                kernel = np.exp(-0.5 * (dts / h) ** 2)
                weighted_kernel = w_sub * kernel
                sum_wk = np.sum(weighted_kernel)
                if sum_wk > 1e-12:
                    f_interp[g_idx] = np.sum(weighted_kernel * y_sub) / sum_wk

    n_scales = len(tE_scales)
    n_time = len(t_grid)
    pe_e = np.zeros((n_scales, n_time))
    pe_o = np.zeros((n_scales, n_time))

    # Gap/edge suppression via gap-contamination fraction eta.
    # gap_mask = 1 where there is NO observation, 0 where there IS data.
    # Convolving |kernel| over this mask gives the fraction of the kernel
    # support that falls over gaps or boundaries → eta ∈ [0, 1].
    # CWT coefficient is then down-weighted by (1 - eta)^2.
    gap_mask = 1.0 - grid_mask.astype(float)

    for j, tE in enumerate(tE_scales):
        half = int(5.0 * tE / dt)
        # Skip scales that are physically larger than the entire observation duration
        if tE > (t_obs[-1] - t_obs[0]):
            continue
        tk = np.arange(-half, half + 1) * dt
        ke, ko = get_kernels(tk, tE)

        abs_ke = np.abs(ke)
        abs_ko = np.abs(ko)

        # Pad gap mask with 1 (= gap) outside boundaries
        gap_padded = np.pad(gap_mask, half, mode="constant", constant_values=1.0)
        eta_e = np.clip(fftconvolve(gap_padded, abs_ke, mode="valid"), 0.0, 1.0)
        eta_o = np.clip(fftconvolve(gap_padded, abs_ko, mode="valid"), 0.0, 1.0)

        # Reflection pad data to smooth the edge transition
        f_padded = np.pad(f_interp, (half, half), mode="reflect")

        pe_e[j, :] = np.abs(fftconvolve(f_padded, ke, mode="valid")) * (1.0 - eta_e) ** 2
        pe_o[j, :] = np.abs(fftconvolve(f_padded, ko, mode="valid")) * (1.0 - eta_o) ** 2

    def process_morphology(pe_map):
        """Convert a CWT power map into a list of validated anomaly candidates."""

        # ------------------------------------------------------------------
        # Step 0: Estimate the noise-colour gamma from the CWT noise floor.
        #
        # For noise with PSD ∝ f^(-β), the CWT power at scale s scales as
        # ∝ s^β.  The optimal detection penalty is gamma = -β/2 so that the
        # MAD-normalised consensus score is uniform across scales.
        #
        # We estimate β by fitting a power law to the 20th-percentile of the
        # CWT power across well-observed time positions at each scale.
        # This runs at zero extra cost — the pe_map is already computed.
        # ------------------------------------------------------------------
        noise_floor = []
        for j in range(pe_map.shape[0]):
            active = pe_map[j, grid_mask]
            if len(active) > 20:
                noise_floor.append(np.percentile(active, 20))
            else:
                noise_floor.append(np.nan)
        noise_floor = np.array(noise_floor)
        valid_nf = np.isfinite(noise_floor) & (noise_floor > 1e-12)
        if valid_nf.sum() >= 10:
            beta_fit = np.polyfit(
                np.log(tE_scales[valid_nf]),
                np.log(noise_floor[valid_nf]),
                1,
            )[0]
            # Clip to physically plausible range: white noise (0) to steep red noise (2)
            gamma_global = float(np.clip(-beta_fit / 2.0, -1.5, 0.0))
        else:
            gamma_global = -0.5   # fallback: assume 1/f noise

        norm_map = np.zeros_like(pe_map)
        for i in range(pe_map.shape[0]):
            row = pe_map[i, :]
            active = row[grid_mask]
            if len(active) > 20:
                med = np.median(active)
                mad = np.median(np.abs(active - med)) / 0.6745
                if mad > 1e-6:
                    norm_map[i, :] = (row - med) / mad

        # Exclude sub-cadence scales from the consensus Z-score map
        min_physical_scale = max(0.05, 0.25 * median_spacing)
        valid_scale_mask = tE_scales >= min_physical_scale
        if not np.any(valid_scale_mask):
            valid_scale_mask = np.ones_like(tE_scales, dtype=bool)

        consensus_1d = np.max(norm_map[valid_scale_mask, :], axis=0)
        peaks, _ = find_peaks(
            consensus_1d * grid_mask,
            height=cwt_threshold,
            distance=int(2.5 / dt),
            prominence=cwt_threshold * 0.4,
            width=2,
        )

        found = []
        for p_idx in peaks:
            col_raw = pe_map[:, p_idx]

            # Initial tE estimate from the scan pass restricted to valid scales
            corrected = col_raw * (tE_scales ** -0.5)
            valid_indices = np.where(valid_scale_mask)[0]
            if len(valid_indices) == 0:
                valid_indices = np.arange(len(tE_scales))
            row_idx = valid_indices[np.argmax(corrected[valid_indices])]
            dlog_tE = np.log10(tE_scales[1]) - np.log10(tE_scales[0])
            if 0 < row_idx < len(tE_scales) - 1:
                a = np.log(corrected[row_idx - 1] + 1e-12)
                b = np.log(corrected[row_idx]     + 1e-12)
                c = np.log(corrected[row_idx + 1] + 1e-12)
                denom = a - 2 * b + c
                offset = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
                tE_scan = 10 ** (np.log10(tE_scales[row_idx]) + offset * dlog_tE)
            else:
                tE_scan = float(tE_scales[row_idx])

            # Local extremum verification: must be a genuine local max or min
            # (not a CWT side-lobe ghost or edge slope).
            w_size = max(2, int(0.1 * tE_scan / dt))
            v_c = f_interp[p_idx]
            v_l = f_interp[max(0, p_idx - w_size)]
            v_r = f_interp[min(n_time - 1, p_idx + w_size)]

            is_max = (v_c > v_l) and (v_c > v_r)
            is_min = (v_c < v_l) and (v_c < v_r)
            if not (is_max or is_min):
                continue

            sym = calculate_symmetry(t_grid, f_interp, t_grid[p_idx], tE_scan)

            # u0 estimation from peak flux
            f_excess = f_interp[p_idx]
            if baseline_flux is not None and baseline_flux > 0:
                A_peak = 1.0 + abs(f_excess) / baseline_flux
                u0_est = _u0_from_amplification(A_peak)
            else:
                A_peak = np.nan
                u0_est = np.nan

            # Filter out physically unresolved sub-cadence detections (interpolation/noise artifacts).
            # A physically resolvable event must have a scan timescale of at least 1.0 * median_spacing (minimum 0.1 days).
            if tE_scan < max(0.1, 1.0 * median_spacing):
                continue

            # Refine tE using analytical bias correction (fast and robust).
            exact_tE = tE_scan * _tE_correction_factor(u0_est)

            # Compute dchi2 and dbic analytically using weighted linear least-squares fit
            u0_fit = u0_est if (u0_est is not None and not np.isnan(u0_est) and u0_est > 0) else 0.05
            win_mask = (t_obs >= t_grid[p_idx] - 5.0 * exact_tE) & (t_obs <= t_grid[p_idx] + 5.0 * exact_tE)
            t_win = t_obs[win_mask]
            y_win = y_obs[win_mask]
            w_win = w[win_mask]

            if len(t_win) < 5:
                # fall back to entire light curve
                t_win = t_obs
                y_win = y_obs
                w_win = w

            u_win = np.sqrt(u0_fit**2 + ((t_win - t_grid[p_idx]) / exact_tE)**2)
            u_win = np.where(u_win > 1e-6, u_win, 1e-6)
            A_win = (u_win**2 + 2.0) / (u_win * np.sqrt(u_win**2 + 4.0))
            S_win = A_win - 1.0

            sum_w = np.sum(w_win)
            sum_wS = np.sum(w_win * S_win)
            sum_wS2 = np.sum(w_win * S_win**2)
            sum_wy = np.sum(w_win * y_win)
            sum_wSy = np.sum(w_win * S_win * y_win)

            det = sum_wS2 * sum_w - sum_wS**2
            if det > 1e-12:
                Fs = (sum_wSy * sum_w - sum_wS * sum_wy) / det
                Fb = (sum_wS2 * sum_wy - sum_wS * sum_wSy) / det
            else:
                Fs = 0.0
                Fb = sum_wy / (sum_w + 1e-12)

            # Model chi2
            y_model = Fs * S_win + Fb
            chi2_lens = np.sum(w_win * (y_win - y_model)**2)

            # Null chi2
            Fb_null = sum_wy / (sum_w + 1e-12)
            chi2_null = np.sum(w_win * (y_win - Fb_null)**2)

            dchi2 = chi2_null - chi2_lens
            N_pts = len(t_win)
            dbic = chi2_lens - chi2_null + 2.0 * np.log(N_pts)

            # Boundary proximity edge flag
            t0 = float(t_grid[p_idx])
            edge_flag = bool((t0 < t_obs[0] + 0.5 * exact_tE) or (t0 > t_obs[-1] - 0.5 * exact_tE))

            # Apply min_dchi2 filtering if specified
            if min_dchi2 is not None and dchi2 < min_dchi2:
                continue

            found.append({
                "t0":       float(t_grid[p_idx]),
                "tE":       float(exact_tE),
                "tE_scan":  float(tE_scan),           # u0=0.05 scan estimate
                "u0":       float(u0_est) if not np.isnan(u0_est) else None,
                "A_peak":   float(A_peak) if not np.isnan(A_peak) else None,
                "snr":      float(consensus_1d[p_idx]),
                "symmetry": float(sym),
                "type":     "dip" if is_min else "peak",
                "gamma":    float(gamma_global),      # noise-floor estimated gamma
                "dchi2":    float(dchi2),
                "dbic":     float(dbic),
                "edge_flag": edge_flag,
            })

        return found, consensus_1d, norm_map

    even_peaks, c1d_e, n_map_e = process_morphology(pe_e)
    odd_peaks,  c1d_o, n_map_o = process_morphology(pe_o)

    return {
        "even_peaks": even_peaks,
        "odd_peaks":  odd_peaks,
        "t_grid":     t_grid,
        "f_interp":   f_interp,
        "grid_mask":  grid_mask,
        "consensus_1d_even": c1d_e,
        "consensus_1d_odd":  c1d_o,
        "norm_map_even":     n_map_e,
        "norm_map_odd":      n_map_o,
        "tE_scales":  tE_scales,
    }
