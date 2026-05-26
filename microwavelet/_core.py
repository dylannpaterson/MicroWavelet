"""
microwavelet._core
===================
High-level ``analyze_lightcurve`` entry point.

Output schema
-------------
{
    "anomalies": [
        {
            "t0":       float,  # peak/dip centre time (same units as input t)
            "tE":       float,  # Einstein crossing time estimate (days)
            "u0":       float | None,  # impact parameter (Paczynski inversion)
            "A_peak":   float | None,  # peak magnification (flux ratio)
            "snr":      float,  # CWT consensus Z-score
            "symmetry": float,  # 0 = perfectly symmetric, >0.3 = asymmetric
            "type":     str,    # "peak" | "dip"
            "band":     str,    # primary detection band
            "dchi2":    float,  # delta chi2 (chi2_null - chi2_lens)
            "dbic":     float,  # delta BIC (chi2_lens - chi2_null + 2 ln(N))
            "edge_flag": bool,  # True if close to temporal boundaries (artifact)
            "chromaticity_ratio": float, # ratio of other band scale to primary band
            "chromatic_flag": bool, # True if wildly chromatic (stellar flare / CV)
        },
        ...
    ],
    "detrending": None  |  {   # populated only when detrend_periodic=True
        "period_days": float,
        "period_search": {
            "initial_ls":      float,
            "alias_resolved":  float,
            "optimized":       float,
        },
        "bands": {
            band_name: {
                "t":              np.ndarray,
                "y_raw":          np.ndarray,
                "y_detrended":    np.ndarray,
                "y_err":          np.ndarray,
                "baseline_model": np.ndarray,
                "phase":          np.ndarray,
                "outlier_mask":   np.ndarray,  # True = kept (not sigma-clipped)
            },
            ...
        },
    },
    "diagnostics": {
        "primary_band": str,
        "cwt": {
            "t_grid":            np.ndarray,
            "f_interp":          np.ndarray,
            "grid_mask":         np.ndarray,
            "consensus_1d_even": np.ndarray,
            "consensus_1d_odd":  np.ndarray,
            "norm_map_even":     np.ndarray,
            "norm_map_odd":      np.ndarray,
            "tE_scales":         np.ndarray,
        },
    },
}
"""

import numpy as np
import pandas as pd
from scipy.signal import windows

from ._detrending import detrend_light_curve_periodic
from ._wavelet import detect_cwt_peaks


def analyze_lightcurve(
    time_series,
    detrend_periodic=False,
    min_period=1.0,
    max_period=10.0,
    dt=0.02,
    cwt_threshold=12.0,
    tE_scales=None,
    baseline_func=None,
    interpolator="weighted",
    min_dchi2=None,
    stamp_dir=None,
):
    """
    Robust, standalone CWT-based anomaly detector for multi-filter light curves.

    Parameters
    ----------
    time_series : dict or pd.DataFrame
        **dict** format (preferred)::

            {
                "F146": {"t": t_arr, "y": flux_arr, "y_err": flux_err_arr},
                "F213": {"t": t_arr, "y": flux_arr, "y_err": flux_err_arr},
                ...
            }

        ``y`` and ``y_err`` must be in relative flux units.
        If magnitudes, convert first: ``flux = 10**(-0.4*(mag - mag_median))``.

        **DataFrame** format (auto-converted)::

            Columns: ["filt", "bjd", "mag", "mag_err"]

        Median-based mag→flux conversion is applied per filter.

    detrend_periodic : bool
        If True, runs Lomb-Scargle + iterative GPR detrending to remove
        periodic baseline (CVs, pulsators, EBs) before CWT detection.
    min_period, max_period : float
        Bounds for the period search (days).  Only used when
        ``detrend_periodic=True``.
    dt : float
        CWT interpolation grid spacing (days).  Smaller = more precise t0
        at the cost of memory and compute.
    cwt_threshold : float
        Minimum MAD-normalised consensus Z-score for a detection.
        Higher values = fewer, more confident detections.
    tE_scales : np.ndarray, optional
        Custom log-spaced grid of Einstein timescale scales (days).
        Defaults to 120 scales from 0.02 to 200 days.
    baseline_func : callable, optional
        Custom baseline model ``f(t, period, t0_offset) -> y`` for period
        fine-tuning.  Only relevant when ``detrend_periodic=True``.
    interpolator : str
        Choice of interpolator: "linear" or "weighted".
        "weighted" performs a Nadaraya-Watson local Gaussian kernel regression.
    min_dchi2 : float, optional
        Minimum dchi2 threshold required for anomaly peaks.

    Returns
    -------
    dict
        See module docstring for the full output schema.

    Raises
    ------
    ValueError
        If the input dict is missing required keys, or the DataFrame is missing
        required columns.
    TypeError
        If ``time_series`` is neither a dict nor a DataFrame.
    """
    # ------------------------------------------------------------------
    # 1. Normalise input to internal band_data format
    # ------------------------------------------------------------------
    band_data = {}

    if isinstance(time_series, pd.DataFrame):
        required = {"filt", "bjd", "mag", "mag_err"}
        if not required.issubset(time_series.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")
        for b in sorted(time_series["filt"].unique()):
            lc = time_series[time_series["filt"] == b].sort_values("bjd")
            if len(lc) < 5:
                continue
            med = lc["mag"].median()
            flux = 10 ** (-0.4 * (lc["mag"] - med))
            flux_err = lc["mag_err"] * flux * (np.log(10) / 2.5)
            band_data[b] = {
                "t": lc["bjd"].values,
                "y": flux.values,
                "y_err": flux_err.values,
            }

    elif isinstance(time_series, dict):
        for b, data in time_series.items():
            if not {"t", "y", "y_err"}.issubset(data.keys()):
                raise ValueError(f"Band '{b}' dict must contain keys: 't', 'y', 'y_err'")
            band_data[b] = {
                "t": np.asarray(data["t"]),
                "y": np.asarray(data["y"]),
                "y_err": np.asarray(data["y_err"]),
            }
    else:
        raise TypeError("time_series must be a dict or a pandas DataFrame")

    if not band_data:
        raise ValueError("No valid time-series data found in input.")

    # ------------------------------------------------------------------
    # 1.5. Apply High-Precision Time Scaling (MJD/BJD Protection)
    # Shift all time coordinates internally so they start at 0.0,
    # completely shielding the GPR, CWT, and GCV steps from float64 precision limits.
    # ------------------------------------------------------------------
    t_all = np.concatenate([data["t"] for data in band_data.values()])
    t_min = float(np.min(t_all)) if len(t_all) > 0 else 0.0

    for b in band_data:
        band_data[b]["t"] = band_data[b]["t"] - t_min

    # ------------------------------------------------------------------
    # 2. Optional periodic GPR detrending
    # ------------------------------------------------------------------
    detrending_out = None

    if detrend_periodic:
        detrended_bands, detrending_info = detrend_light_curve_periodic(
            band_data,
            min_period=min_period,
            max_period=max_period,
            baseline_func=baseline_func,
        )
        # Re-point band_data at the detrended versions for the CWT step
        band_data = detrended_bands

        detrending_out = {
            "period_days": detrending_info["period_days"],
            "search_mask": detrending_info.get("search_mask"),
            "period_search": detrending_info["period_search"],
            "bands": {
                b: {
                    "t": d["t"],
                    "y_raw": d["y_raw"],
                    "y_detrended": d["y_detrended"],
                    "y_err": d["y_err"],
                    "baseline_model": d["baseline_model"],
                    "phase": d["phase"],
                    "outlier_mask": d["outlier_mask"],
                }
                for b, d in detrended_bands.items()
            },
        }

    # ------------------------------------------------------------------
    # 3. Select primary band for CWT (highest cadence / most data points)
    # ------------------------------------------------------------------
    primary_band = max(band_data, key=lambda b: len(band_data[b]["t"]))
    p_data = band_data[primary_band]

    # ------------------------------------------------------------------
    # 4. Prepare CWT input flux
    # ------------------------------------------------------------------
    if detrend_periodic:
        # Detrended flux is y / baseline_model, so quiescent ≈ 1.0.
        # Shift to 0-centred and apply a Tukey taper to neutralise
        # seasonal boundary ringing in the CWT.
        y_cwt = p_data["y_detrended"] - 1.0
        taper = windows.tukey(len(y_cwt), alpha=0.05)
        y_cwt = y_cwt * taper
        # For detrended data, the effective baseline (the quiescent level
        # of the *shifted* signal before conversion to magnification) is 1.0.
        cwt_baseline = 1.0
    else:
        # 20th-percentile of raw flux ≈ quiescent baseline
        cwt_baseline = float(np.percentile(p_data["y"], 20))
        y_cwt = p_data["y"] - cwt_baseline

    # ------------------------------------------------------------------
    # 5. CWT peak/dip detection
    # ------------------------------------------------------------------
    cwt_results = detect_cwt_peaks(
        p_data["t"],
        y_cwt,
        y_obs_err=p_data["y_err"],
        tE_scales=tE_scales,
        dt=dt,
        cwt_threshold=cwt_threshold,
        baseline_flux=cwt_baseline,
        interpolator=interpolator,
        min_dchi2=min_dchi2,
    )

    # ------------------------------------------------------------------
    # 6. Merge even + odd detections; deduplicate; annotate band
    # ------------------------------------------------------------------
    all_anomalies = list(cwt_results["even_peaks"])  # copy
    for p_odd in cwt_results["odd_peaks"]:
        # Add odd detections that are not already covered by an even detection
        if not any(abs(p_odd["t0"] - p_e["t0"]) < 1.5 for p_e in all_anomalies):
            all_anomalies.append(p_odd)

    # 6.1. Robust scale-based anomaly deduplication (keeps the highest-SNR peak)
    # to avoid double-counting or splitting a single event.
    deduped = []
    # Sort by SNR descending to process the most significant detections first
    sorted_by_snr = sorted(all_anomalies, key=lambda x: x["snr"], reverse=True)
    for a in sorted_by_snr:
        is_duplicate = False
        for accepted in deduped:
            separation = abs(a["t0"] - accepted["t0"])
            max_tE = max(a["tE"], accepted["tE"])
            # If peaks are within 1.5 * max_tE, they are part of the same event
            if separation < 1.5 * max_tE:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(a)

    # Chronological order; stamp the detection band and compute chromaticity flags
    all_anomalies = sorted(deduped, key=lambda x: x["t0"])

    def get_paczynski_template(t, t0, tE, u0):
        u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
        u = np.where(u > 1e-6, u, 1e-6)
        A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
        return A - 1.0

    for a in all_anomalies:
        a["band"] = primary_band
        a["chromatic_flag"] = False
        a["chromaticity_ratio"] = 1.0

        t0 = a["t0"]
        tE = a["tE"]
        u0 = a["u0"] if (a["u0"] is not None and not np.isnan(a["u0"])) else 0.05

        # Fit template on primary band to get base amplitude
        p_t = p_data["t"]
        p_y = y_cwt
        p_err = p_data["y_err"]
        p_w = 1.0 / (np.where(p_err > 1e-12, p_err, 1e-12) ** 2)

        p_win = (p_t >= t0 - 5.0 * tE) & (p_t <= t0 + 5.0 * tE)
        t_p = p_t[p_win]
        y_p = p_y[p_win]
        w_p = p_w[p_win]

        if len(t_p) < 5:
            t_p, y_p, w_p = p_t, p_y, p_w

        S_p = get_paczynski_template(t_p, t0, tE, u0)
        sum_wp = np.sum(w_p)
        sum_wpS = np.sum(w_p * S_p)
        sum_wpS2 = np.sum(w_p * S_p**2)
        sum_wpy = np.sum(w_p * y_p)
        sum_wpSy = np.sum(w_p * S_p * y_p)

        det_p = sum_wpS2 * sum_wp - sum_wpS**2
        Fs_primary = 0.0
        if det_p > 1e-12:
            Fs_primary = (sum_wpSy * sum_wp - sum_wpS * sum_wpy) / det_p
        else:
            Fs_primary = sum_wpSy / (sum_wpS2 + 1e-12)

        Fs_primary_clean = (
            Fs_primary if abs(Fs_primary) > 1e-6 else (1e-6 if Fs_primary >= 0 else -1e-6)
        )

        # Project onto other bands
        for b, b_data in band_data.items():
            if b == primary_band:
                continue

            b_t = b_data["t"]
            if detrend_periodic:
                b_y = b_data["y_detrended"] - 1.0 if "y_detrended" in b_data else b_data["y"] - 1.0
            else:
                b_baseline = float(np.percentile(b_data["y"], 20))
                b_y = b_data["y"] - b_baseline

            b_err = b_data["y_err"]
            b_w = 1.0 / (np.where(b_err > 1e-12, b_err, 1e-12) ** 2)

            b_win = (b_t >= t0 - 5.0 * tE) & (b_t <= t0 + 5.0 * tE)
            t_b = b_t[b_win]
            y_b = b_y[b_win]
            w_b = b_w[b_win]

            if len(t_b) >= 3:
                S_b = get_paczynski_template(t_b, t0, tE, u0)
                sum_wb = np.sum(w_b)
                sum_wbS = np.sum(w_b * S_b)
                sum_wbS2 = np.sum(w_b * S_b**2)
                sum_wby = np.sum(w_b * y_b)
                sum_wbSy = np.sum(w_b * S_b * y_b)

                det_b = sum_wbS2 * sum_wb - sum_wbS**2
                if det_b > 1e-12:
                    Fs_b = (sum_wbSy * sum_wb - sum_wbS * sum_wby) / det_b
                    Fb_b = (sum_wbS2 * sum_wby - sum_wbS * sum_wbSy) / det_b
                else:
                    Fs_b = sum_wbSy / (sum_wbS2 + 1e-12)
                    Fb_b = sum_wby / (sum_wb + 1e-12)

                y_model_b = Fs_b * S_b + Fb_b
                chi2_lens_b = np.sum(w_b * (y_b - y_model_b) ** 2)
                Fb_null_b = sum_wby / (sum_wb + 1e-12)
                chi2_null_b = np.sum(w_b * (y_b - Fb_null_b) ** 2)
                dchi2_b = chi2_null_b - chi2_lens_b

                if dchi2_b >= 10.0:
                    ratio = Fs_b / Fs_primary_clean
                    a["chromaticity_ratio"] = float(ratio)
                    if ratio < -0.1 or ratio > 3.0:
                        a["chromatic_flag"] = True
                else:
                    if len(t_b) >= 5 and a.get("dchi2", 0.0) >= 50.0:
                        a["chromaticity_ratio"] = 0.0
                        a["chromatic_flag"] = True

    # ------------------------------------------------------------------
    # 6.4. Unshift all times back to original unscaled BJD/MJD coordinates
    # to guarantee user-facing coordinates are correct and stamp plots show original times
    # ------------------------------------------------------------------
    for a in all_anomalies:
        a["t0"] = float(a["t0"] + t_min)

    for b in band_data:
        band_data[b]["t"] = band_data[b]["t"] + t_min

    if detrending_out is not None:
        for b in detrending_out["bands"]:
            detrending_out["bands"][b]["t"] = detrending_out["bands"][b]["t"] + t_min

    cwt_results["t_grid"] = cwt_results["t_grid"] + t_min

    # ------------------------------------------------------------------
    # 6.5. Generate Stamp Plot if stamp_dir is supplied and peaks detected
    # ------------------------------------------------------------------
    if stamp_dir is not None and len(all_anomalies) > 0:
        try:
            import os

            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
            import matplotlib.pyplot as plt

            # 1. Determine clip boundaries
            t_starts = [a["t0"] - 5.0 * a["tE"] for a in all_anomalies]
            t_ends = [a["t0"] + 5.0 * a["tE"] for a in all_anomalies]
            t_start = min(t_starts)
            t_end = max(t_ends)

            # 2. Setup Figure
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Inter"]

            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            fig.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#f8f9fa")

            # Styling spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")
            ax.grid(True, linestyle=":", alpha=0.5)

            # Beautiful color palette
            palette = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22"]
            markers = ["o", "s", "^", "D", "v", "p", "*"]

            # 3. Plot each filter/band
            for idx, (b_name, b_data) in enumerate(sorted(band_data.items())):
                color = palette[idx % len(palette)]
                marker = markers[idx % len(markers)]

                t_arr = np.asarray(b_data["t"])
                if "y" in b_data:
                    y_arr = np.asarray(b_data["y"])
                elif "y_detrended" in b_data:
                    y_arr = np.asarray(b_data["y_detrended"])
                else:
                    y_arr = np.asarray(b_data["y_raw"])
                y_err_arr = np.asarray(b_data["y_err"])

                # Clip data to window [t_start, t_end]
                mask = (t_arr >= t_start) & (t_arr <= t_end)
                if np.sum(mask) == 0:
                    continue

                ax.errorbar(
                    t_arr[mask],
                    y_arr[mask],
                    yerr=y_err_arr[mask],
                    fmt=marker,
                    color=color,
                    markersize=4,
                    alpha=0.7,
                    elinewidth=0.6,
                    capsize=0,
                    label=b_name,
                )

            # 4. Draw peak indicators
            for a in all_anomalies:
                ax.axvline(a["t0"], color="#2c3e50", linestyle="--", linewidth=1.2, alpha=0.8)
                label_text = f"Peak: $t_0$={a['t0']:.2f}\n$t_E$={a['tE']:.1f}d\nSNR={a['snr']:.1f}σ"
                # Place label neatly
                ax.text(
                    a["t0"] + 0.05 * (t_end - t_start),
                    ax.get_ylim()[0] + 0.7 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    label_text,
                    fontsize=9,
                    color="#2c3e50",
                    bbox=dict(
                        facecolor="#ffffff",
                        alpha=0.8,
                        edgecolor="#cccccc",
                        boxstyle="round,pad=0.3",
                    ),
                    zorder=5,
                )

            ax.set_xlabel("Time (BJD / Days)", fontsize=11, fontweight="bold", labelpad=10)
            ax.set_ylabel("Relative Flux", fontsize=11, fontweight="bold", labelpad=10)
            ax.set_xlim(t_start, t_end)
            ax.set_title(
                "MicroWavelet: Detailed Anomaly Detections",
                fontsize=13,
                fontweight="bold",
                pad=15,
                loc="left",
            )
            ax.legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc")

            # Save file
            os.makedirs(stamp_dir, exist_ok=True)
            out_path = os.path.join(stamp_dir, "stamp_peaks.png")
            plt.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
            plt.close(fig)
            print(f"✅ Anomaly stamp plot successfully saved to {out_path}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to generate stamp plot: {e}")

    # ------------------------------------------------------------------
    # 7. Assemble output
    # ------------------------------------------------------------------
    return {
        "anomalies": all_anomalies,
        "detrending": detrending_out,
        "diagnostics": {
            "primary_band": primary_band,
            "cwt": {
                "t_grid": cwt_results["t_grid"],
                "f_interp": cwt_results["f_interp"],
                "grid_mask": cwt_results["grid_mask"],
                "consensus_1d_even": cwt_results["consensus_1d_even"],
                "consensus_1d_odd": cwt_results["consensus_1d_odd"],
                "norm_map_even": cwt_results["norm_map_even"],
                "norm_map_odd": cwt_results["norm_map_odd"],
                "tE_scales": cwt_results["tE_scales"],
            },
        },
    }
