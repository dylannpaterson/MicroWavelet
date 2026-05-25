"""
microlens_cwt
=============

Lightweight, standalone CWT-based anomaly detector for multi-filter
microlensing light curves.

Public API
----------
analyze_lightcurve(time_series, ...)
    Main entry point.  Takes a dict of {band: {t, y, y_err}} or a raw
    DataFrame and returns a structured result dict.

Output schema
-------------
{
    "anomalies": [
        {
            "t0":      float,   # peak centre time (same units as input t)
            "tE":      float,   # Einstein crossing time estimate (days)
            "u0":      float,   # impact parameter (Paczynski inversion)
            "A_peak":  float,   # peak magnification (flux ratio)
            "snr":     float,   # CWT consensus Z-score
            "symmetry":float,   # 0 = perfectly symmetric, >0.3 = asymmetric
            "type":    str,     # "peak" | "dip"
            "band":    str,     # primary band used for detection
        },
        ...
    ],
    "detrending": None | {       # only populated when detrend_periodic=True
        "period_days": float,
        "period_search": {
            "initial_ls":  float,
            "alias_resolved": float,
            "optimized":   float,
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
            "t_grid":           np.ndarray,
            "f_interp":         np.ndarray,
            "grid_mask":        np.ndarray,
            "consensus_1d_even":np.ndarray,
            "consensus_1d_odd": np.ndarray,
            "norm_map_even":    np.ndarray,
            "norm_map_odd":     np.ndarray,
            "tE_scales":        np.ndarray,
        },
    },
}
"""

from ._core import analyze_lightcurve

__all__ = ["analyze_lightcurve"]
__version__ = "0.1.0"
