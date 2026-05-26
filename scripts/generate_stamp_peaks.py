import os
import sys

import numpy as np

# Ensure microwavelet is in path
sys.path.insert(0, os.path.abspath("."))
from microwavelet import analyze_lightcurve


def generate_stamp():
    # 1. Generate synthetic data: CV-like periodic variable + Microlensing event
    np.random.seed(42)
    t = np.arange(0, 100, 0.2)  # 0.2 days cadence

    # CV-like baseline: Narrow eclipses + double-hump ellipsoidal modulation
    period_true = 4.21
    phase_true = (t % period_true) / period_true
    ellipsoidal = 0.05 * np.sin(4 * np.pi * phase_true)
    eclipses = -0.3 * np.exp(-0.5 * ((np.mod(phase_true + 0.1, 1.0) - 0.5) / 0.02) ** 2)
    baseline_flux = 1.0 + ellipsoidal + eclipses

    # Microlensing event (tE=12.0, u0=0.2 -> A_peak ~ 5)
    t0, tE, u0 = 55.0, 12.0, 0.2
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

    # Combined flux with noise
    y_raw = baseline_flux * A + np.random.normal(0, 0.015, size=len(t))
    y_err = np.ones_like(t) * 0.015

    # We will also add a second filter/band to make it chromaticity-rich and show multi-band plotting!
    # Achromatic magnification in both Band1 and Band2
    y_raw2 = baseline_flux * A + np.random.normal(0, 0.02, size=len(t))
    y_err2 = np.ones_like(t) * 0.02

    data = {
        "F146 (Primary)": {"t": t, "y": y_raw, "y_err": y_err},
        "F213 (Secondary)": {"t": t, "y": y_raw2, "y_err": y_err2},
    }

    # 2. Run the pipeline with stamp plot output enabled
    print("--- Running MicroWavelet Anomaly Detection Pipeline ---")
    results = analyze_lightcurve(data, detrend_periodic=True, stamp_dir="docs")

    print(f"\nNumber of anomalies found: {len(results['anomalies'])}")
    for a in results["anomalies"]:
        print(f"t0: {a['t0']:.2f}, tE: {a['tE']:.2f}, snr: {a['snr']:.2f}, dchi2: {a['dchi2']:.2f}")


if __name__ == "__main__":
    generate_stamp()
