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
    n_points = 1500
    t = np.linspace(0, 100, n_points)
    t0, u0, tE = 50, 0.5, 15
    baseline = 1.0
    
    # Anomaly: smaller and narrower than the event
    anomaly_t = 60
    anomaly_amp = 0.3
    anomaly_width = 0.5 
    
    noise_std = 0.01

    # 1. Generate Data
    y_data, y_true, anomaly_signal = generate_microlensing_data(
        t, t0, u0, tE, baseline, anomaly_t, anomaly_amp, anomaly_width, noise_std
    )

    # --- STAGE 1: Event Detection (vs Baseline) ---
    # Residuals relative to a flat baseline
    residuals_baseline = y_data - baseline
    cusum_event = find_anomalies_cusum(t, residuals_baseline, threshold=20.0, bidirectional=False)

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
    residuals_pspl = y_data - y_pspl
    cusum_anomaly = find_anomalies_cusum(t, residuals_pspl, threshold=10.0, bidirectional=True)

    # --- Plotting (4 Panels: Data, Event Score, Anomaly Score, Comparison) ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 20), constrained_layout=True)

    # Panel 1: Light Curve
    ax = axes[0]
    ax.scatter(t, y_data, s=10, color='black', alpha=0.4, label='Observed Data')
    ax.plot(t, y_pspl, 'r-', linewidth=2, label='Fitted PSPL Model')
    ax.axhline(baseline, color='gray', linestyle='--', label='Baseline')
    ax.set_title("1. Microlensing Light Curve & PSPL Fit")
    ax.set_ylabel("Flux")
    ax.legend()

    # Panel 2: Event CUSUM Score
    ax = axes[1]
    ax.plot(t, cusum_event['cusum_statistic'], 'g-', label='Event CUSUM Score')
    if cusum_event['triggered']:
        ax.axvspan(cusum_event['onset'], cusum_event['end'], color='green', alpha=0.1)
    ax.set_title("2. Event Detection CUSUM Score (vs Baseline)")
    ax.set_ylabel("Score")
    ax.legend()

    # Panel 3: Anomaly CUSUM Score
    ax = axes[2]
    ax.plot(t, cusum_anomaly['cusum_statistic'], 'b-', label='Anomaly CUSUM Score')
    if cusum_anomaly['triggered']:
        ax.axvspan(cusum_anomaly['onset'], cusum_anomaly['end'], color='blue', alpha=0.1)
    ax.set_title("3. Anomaly Detection CUSUM Score (vs PSPL)")
    ax.set_ylabel("Score")
    ax.legend()

    # Panel 4: Comparison
    ax = axes[3]
    ax.plot(t, cusum_event['cusum_statistic'], 'g-', alpha=0.6, label='Event CUSUM Score')
    ax.plot(t, cusum_anomaly['cusum_statistic'], 'b-', alpha=0.6, label='Anomaly CUSUM Score')
    ax.set_title("4. CUSUM Score Comparison")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Score")
    ax.legend()

    plt.savefig("docs/microlensing_cusum_workflow_v3.png", dpi=150)
    print("Saved docs/microlensing_cusum_workflow_v3.png")

if __name__ == "__main__":
    run_demo()
