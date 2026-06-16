"""
microwavelet
=============

Lightweight, standalone CWT-based anomaly detector for multi-filter
microlensing light curves.
"""

from ._core import analyze_lightcurve
from ._cusum import (
    find_anomalies_cusum,
    run_backward_quadratic_cusum,
    run_linear_cusum,
    run_quadratic_cusum,
    seed_by_flat_cusum,
)
from ._fit import detect_anomalies_with_fit, get_paczynski
from ._noise import characterize_multiband_noise, characterize_noise

__all__ = [
    "analyze_lightcurve",
    "characterize_noise",
    "characterize_multiband_noise",
    "run_linear_cusum",
    "run_quadratic_cusum",
    "run_backward_quadratic_cusum",
    "seed_by_flat_cusum",
    "find_anomalies_cusum",
    "detect_anomalies_with_fit",
    "get_paczynski",
]

__version__ = "26.1.7"
