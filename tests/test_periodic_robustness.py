import numpy as np
from microwavelet import analyze_lightcurve

def test_periodic_robustness_standard_case():
    """
    Verifies that the periodic baseline search is robust to a 
    high-amplitude microlensing event (the 'Standard Case').
    """
    np.random.seed(42)
    # Use uniform sampling for stability in this baseline test
    t = np.arange(0, 100, 0.2)
    
    # 1. Periodic baseline (P=5.34)
    period_true = 5.34
    y_periodic = 1.0 + 0.15 * np.sin(2 * np.pi * t / period_true)
    
    # 2. Microlensing event (A_peak ~ 3.4)
    t0, tE, u0 = 55.0, 8.0, 0.2
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    
    y_raw = y_periodic * A + np.random.normal(0, 0.02, size=len(t))
    y_err = np.ones_like(t) * 0.02
    
    data = {"Band1": {"t": t, "y": y_raw, "y_err": y_err}}
    
    # Run with periodic detrending
    results = analyze_lightcurve(data, detrend_periodic=True, min_period=1.0, max_period=10.0)
    
    # Verify period recovery (allowing for small aliases if they detrend well)
    recovered_period = results["detrending"]["period_days"]
    is_correct = any(abs(recovered_period - period_true * f) < 0.2 for f in [0.5, 1.0, 2.0])
    assert is_correct, f"Failed to recover true period or harmonic. Found {recovered_period}"
    
    # Verify event preservation in detrended data
    detrend = results["detrending"]["bands"]["Band1"]
    y_detrended = detrend["y_detrended"]
    
    # Find peak in detrended data
    peak_idx = np.argmin(np.abs(t - t0))
    recovered_A = y_detrended[peak_idx]
    true_A = A[peak_idx]
    
    # Should be within ~10% of true A_peak
    assert abs(recovered_A - true_A) / true_A < 0.1, f"Event signal suppressed. Found A_peak={recovered_A}, expected ~{true_A}"

def test_cv_period_recovery():
    """
    Verifies that the periodic baseline search recovers the correct fundamental period
    for a CV-like variable (sharp eclipses + double ellipsoidal hump) lensed by a microlensing event.
    """
    np.random.seed(42)
    t = np.arange(0, 100, 0.1)
    period_true = 4.21
    phase_true = (t % period_true) / period_true
    ellipsoidal = 0.05 * np.sin(4 * np.pi * phase_true)
    eclipses = -0.3 * np.exp(-0.5 * ((np.mod(phase_true + 0.1, 1.0) - 0.5) / 0.02)**2)
    baseline_flux = 1.0 + ellipsoidal + eclipses
    t0, tE, u0 = 55.0, 12.0, 0.15
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    y_raw = baseline_flux * A + np.random.normal(0, 0.015, size=len(t))
    y_err = np.ones_like(t) * 0.015
    data = {"Band1": {"t": t, "y": y_raw, "y_err": y_err}}
    
    results = analyze_lightcurve(data, detrend_periodic=True, min_period=1.0, max_period=10.0)
    recovered = results["detrending"]["period_days"]
    
    # Verify period recovery
    assert abs(recovered - period_true) < 0.1, f"Failed to recover true CV period. Found {recovered}, expected {period_true}"

if __name__ == "__main__":
    test_periodic_robustness_standard_case()
    test_cv_period_recovery()
