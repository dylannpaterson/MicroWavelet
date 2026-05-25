"""
microlens_cwt._core
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
    cwt_threshold=25.0,
    tE_scales=None,
    baseline_func=None,
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
                raise ValueError(
                    f"Band '{b}' dict must contain keys: 't', 'y', 'y_err'"
                )
            band_data[b] = {
                "t":     np.asarray(data["t"]),
                "y":     np.asarray(data["y"]),
                "y_err": np.asarray(data["y_err"]),
            }
    else:
        raise TypeError("time_series must be a dict or a pandas DataFrame")

    if not band_data:
        raise ValueError("No valid time-series data found in input.")

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
            "period_search": detrending_info["period_search"],
            "bands": {
                b: {
                    "t":              d["t"],
                    "y_raw":          d["y_raw"],
                    "y_detrended":    d["y_detrended"],
                    "y_err":          d["y_err"],
                    "baseline_model": d["baseline_model"],
                    "phase":          d["phase"],
                    "outlier_mask":   d["outlier_mask"],
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
        tE_scales=tE_scales,
        dt=dt,
        cwt_threshold=cwt_threshold,
        baseline_flux=cwt_baseline,
    )

    # ------------------------------------------------------------------
    # 6. Merge even + odd detections; deduplicate; annotate band
    # ------------------------------------------------------------------
    all_anomalies = list(cwt_results["even_peaks"])  # copy
    for p_odd in cwt_results["odd_peaks"]:
        # Add odd detections that are not already covered by an even detection
        if not any(abs(p_odd["t0"] - p_e["t0"]) < 1.5 for p_e in all_anomalies):
            all_anomalies.append(p_odd)

    # Chronological order; stamp the detection band
    all_anomalies = sorted(all_anomalies, key=lambda x: x["t0"])
    for a in all_anomalies:
        a["band"] = primary_band

    # ------------------------------------------------------------------
    # 7. Assemble output
    # ------------------------------------------------------------------
    return {
        "anomalies": all_anomalies,
        "detrending": detrending_out,
        "diagnostics": {
            "primary_band": primary_band,
            "cwt": {
                "t_grid":            cwt_results["t_grid"],
                "f_interp":          cwt_results["f_interp"],
                "grid_mask":         cwt_results["grid_mask"],
                "consensus_1d_even": cwt_results["consensus_1d_even"],
                "consensus_1d_odd":  cwt_results["consensus_1d_odd"],
                "norm_map_even":     cwt_results["norm_map_even"],
                "norm_map_odd":      cwt_results["norm_map_odd"],
                "tE_scales":         cwt_results["tE_scales"],
            },
        },
    }
