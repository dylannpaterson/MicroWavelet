import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import curve_fit

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath("../.."))
from microwavelet import find_anomalies_cusum

def paczynski_mag(u):
    """Standard PSPL magnification formula: A(u) -> 1 as u -> infinity."""
    return (u**2 + 2) / (u * np.sqrt(u**2 + 3))

def paczynski_model(t, t0, u0, tE):
    """PSPL magnification model."""
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    u = np.maximum(u, 1e-9)
    return paczynski_mag(u)

def generate_microlensing_data(t, t0, u0, tE, baseline=1.0, anomaly_t=None, anomaly_amp=0.0, anomaly_width=None, noise_std=0.01):
    """Generates a microlensing light curve with a flat baseline and an optional anomaly."""
    # Flux = baseline * magnification
    y = baseline * paczynski_model(t, t0, u0, tE)
    
    anomaly_signal = 0.0
    if anomaly_t is not None:
        # A Gaussian anomaly added to the flux
        width = anomaly_width if anomaly_width is not None else (0.1 * tE)
        anomaly_signal = anomaly_amp * np.exp(-((t - anomaly_t)**2) / (2 * width**2))
        y += anomaly_signal
    
    noise = np.random.normal(0, noise_std, len(t))
    return y + noise, y, anomaly_signal

def run_demo():
    # Parameters
    t_start, t_end = 0, 200
    n_points = 3000
    t = np.linspace(t_start, t_end, n_points)
    
    # Event centered in the middle of the new range
    t0, u0, tE = 100, 0.8, 15
    baseline = 1.0
    
    # Anomaly: smaller and narrower than the event
    anomaly_t = 110
    anomaly_amp = 0.2 
    anomaly_width = 0.5 
    
    noise_std = 0.02

    # 1. Generate Data
    y_data, y_true, anomaly_signal = generate_microlensing_data(
        t, t0, u0, tE, baseline, anomaly_t, anomaly_amp, anomaly_width, noise_std
    )

    # --- STAGE 1: Event Detection (vs Baseline) ---
    # Residuals relative to a flat baseline, STANDARDIZED
    residuals_baseline_raw = y_data - baseline
    residuals_baseline_sigma = residuals_baseline_raw / noise_std
    
    # Use bidirectional CUSUM for the event
    cusum_event = find_anomalies_cusum(t, residuals_baseline_sigma, threshold=25.0, k=1.0, bidirectional=True)

    # --- STAGE 2: PSPL Fitting ---
    def fit_func(t, baseline_fit, t0_fit, u0_fit, tE_fit):
        return baseline_fit * paczynski_model(t, t0_fit, u0_fit, tE_fit)

    p0 = [baseline, t0, u0, tE]
    try:
        popt, pcov = curve_fit(fit_func, t, y_data, p0=p0)
        baseline_fit, t0_fit, u0_fit, tE_fit = popt
        print(f"Fit successful: t0={t0_fit:.2f}, u0={u0_fit:.2f}, tE={tE_fit:.2f}, baseline={baseline_fit:.2f}")
    except Exception as e:
        print(f"Fit failed: {e}")
        popt = p0
        baseline_fit, t0_fit, u0_fit, tE_fit = popt

    y_pspl = fit_func(t, *popt)

    # --- STAGE 3: Anomaly Detection (vs PSPL) ---
    # Residuals relative to the fitted PSPL model, STANDARDIZED
    residuals_pspl_raw = y_data - y_pspl
    residuals_pspl_sigma = residuals_pspl_raw / noise_std
    
    # Use bidirectional quadratic CUSUM for the anomaly
    cusum_anomaly = find_anomalies_cusum(t, residuals_pspl_sigma, threshold=12.0, k=2.0, bidirectional=True)

    # --- Plotting (2 Panels) ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), constrained_layout=True)

    # Panel 1: Light Curve + Event CUSUM Score
    ax = axes[0]
    ax_cusum = ax.twinx()
    ax.scatter(t, y_data, s=10, color='black', alpha=0.4, label='Observed Data')
    ax.plot(t, y_pspl, 'r-', linewidth=2, label='Fitted PSPL Model')
    ax.axhline(baseline, color='gray', linestyle='--', label='Baseline')
    ax_cusum.plot(t, cusum_event['cusum_statistic'], 'g-', alpha=0.7, label='Event CUSUM Score')
    
    if cusum_event['triggered']:
        ax.axvspan(cusum_event['onset'], cusum_event['end'], color='green', alpha=0.1)
        
    ax.set_title("1. Microlensing Event Detection (Light Curve & Event CUSUM)")
    ax.set_ylabel("Flux")
    ax_cusum.set_ylabel("Event CUSUM Score")
    ax.legend(loc='upper left')
    ax_cusum.legend(loc='upper right')

    # Panel 2: PSPL Residuals + Anomaly CUSUM Score
    ax = axes[1]
    ax_cusum = ax.twinx()
    ax.scatter(t, residuals_pspl_raw, s=10, color='black', alpha=0.3, label='PSPL Residuals')
    ax_cusum.plot(t, cusum_anomaly['cusum_statistic'], 'b-', alpha=0.7, label='Anomaly CUSUM Score')
    
    if cusum_anomaly['triggered']:
        ax.axvspan(cusum_anomaly['onset'], cusum_anomaly['end'], color='blue', alpha=0.1)
        ax.axvline(cusum_anomaly['t0'], color='blue', linestyle='--', alpha=0.5)

    ax.set_title("2. Anomaly Detection (PSPL Residuals & Anomaly CUSUM)")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Residuals")
    ax_cusum.set_ylabel("Anomaly CUSUM Score")
    ax.legend(loc='upper left')
    ax_cusum.legend(loc='upper right')

    plt.savefig("docs/microlensing_cusum_workflow_v7.png", dpi=150)
    print("Saved docs/microlensing_cusum_workflow_v7.png")

if __name__ == "__main__":
    run_demo()
