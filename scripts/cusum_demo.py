import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath(".."))
from microwavelet import find_anomalies_cusum, run_quadratic_cusum, run_backward_quadratic_cusum

def paczynski(t, t0, u0, tE):
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    return (1 + u**2) / (u * np.sqrt(2))

def generate_signal(t, t0, u0, tE, anomaly_t=None, anomaly_amp=0.0):
    y = paczynski(t, t0, u0, tE)
    if anomaly_t is not None:
        # A simple Gaussian anomaly
        anomaly = anomaly_amp * np.exp(-((t - anomaly_t)**2) / (2 * (0.1 * tE)**2))
        y += anomaly
    return y

def run_demo():
    n_points = 2000
    t = np.linspace(0, 100, n_points)
    
    # --- CASE 1: Standard Microlensing Event (Event Detection) ---
    # Smooth Paczynski curve
    t0, u0, tE = 50, 0.5, 15
    y_event = generate_signal(t, t0, u0, tE)
    noise = np.random.normal(0, 0.05, n_points)
    y_event_noisy = y_event + noise
    residuals_event = (y_event_noisy - y_event) / 0.05
    
    # --- CASE 2: Microlensing with Planetary Anomaly (Anomaly Detection) ---
    # Smooth event + a sudden deviation
    anomaly_t = 65
    anomaly_amp = 0.8
    y_anomaly = generate_signal(t, t0, u0, tE, anomaly_t=anomaly_t, anomaly_amp=anomaly_amp)
    y_anomaly_noisy = y_anomaly + noise
    residuals_anomaly = (y_anomaly_noisy - y_anomaly) / 0.05

    # --- CUSUM Analysis ---
    # 1. Forward CUSUM (on event)
    anom_f = find_anomalies_cusum(t, residuals_event, threshold=15.0, bidirectional=False)
    
    # 2. Bidirectional CUSUM (on anomaly)
    anom_bi = find_anomalies_cusum(t, residuals_anomaly, threshold=15.0, bidirectional=True)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 16), constrained_layout=True)

    # Plot 1: Event Detection (Forward CUSUM)
    ax = axes[0]
    ax.plot(t, y_event_noisy, 'k.', alpha=0.3, label='Data')
    ax.plot(t, y_event, 'r-', label='Model')
    if anom_f['triggered']:
        ax.axvspan(anom_f['onset'], anom_f['end'], color='red', alpha=0.2, label='CUSUM Trigger')
        ax.axvline(anom_f['t0'], color='red', linestyle='--', label=f"Peak: {anom_f['t0']:.1f}")
    ax.set_title("Case 1: Standard Event Detection (Forward CUSUM)")
    ax.legend()

    # Plot 2: Anomaly Detection (Bidirectional CUSUM)
    ax = axes[1]
    ax.plot(t, y_anomaly_noisy, 'k.', alpha=0.3, label='Data')
    ax.plot(t, y_anomaly, 'r-', label='Model')
    if anom_bi['triggered']:
        ax.axvspan(anom_bi['onset'], anom_bi['end'], color='blue', alpha=0.2, label='Bidirectional Trigger')
        ax.axvline(anom_bi['t0'], color='blue', linestyle='--', label=f"Anomaly Peak: {anom_bi['t0']:.1f}")
    ax.set_title("Case 2: Planetary Anomaly Detection (Bidirectional CUSUM)")
    ax.legend()

    # Plot 3: CUSUM Statistics Comparison
    ax = axes[2]
    ax.plot(t, residuals_event, 'gray', alpha=0.5, label='Event Residuals')
    ax.plot(t, residuals_anomaly, 'gray', alpha=0.5, label='Anomaly Residuals')
    
    # Plot the CUSUM statistic for the anomaly
    if anom_bi['triggered']:
        ax.plot(t, anom_bi['cusum_statistic'], 'b-', label='Bidirectional CUSUM (Anomaly)')
    
    ax.set_title("CUSUM Statistics on Residuals")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Residuals / CUSUM Score")
    ax.legend()

    plt.savefig("docs/cusum_demo.png", dpi=150)
    print("Saved docs/cusum_demo.png")

if __name__ == "__main__":
    run_demo()
