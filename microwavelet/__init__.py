"""
microwavelet
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
            "dchi2":   float,   # delta chi2 (chi2_null - chi2_lens)
            "dbic":    float,   # delta BIC (chi2_lens - chi2_null + 2 ln(N))
            "edge_flag": bool,  # True if close to temporal boundaries (artifact)
            "chromaticity_ratio": float, # ratio of other band scale to primary band
            "chromatic_flag": bool, # True if wildly chromatic (stellar flare / CV)
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
from ._cusum import find_anomalies_cusum, run_linear_cusum, run_quadratic_cusum, seed_by_flat_cusum
from ._fit import detect_anomalies_with_fit, get_paczynski
from ._noise import characterize_noise

__all__ = [
    "analyze_lightcurve",
    "characterize_noise",
    "run_linear_cusum",
    "run_quadratic_cusum",
    "seed_by_flat_cusum",
    "find_anomalies_cusum",
    "detect_anomalies_with_fit",
    "get_paczynski",
]
__version__ = "26.1.3"
