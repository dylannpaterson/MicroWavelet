import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath("../.."))
from microwavelet import find_anomalies_cusum


def paczynski(t, t0, u0, tE):
    """Point Source Point Lens (PSPL) model."""
    # Avoid division by zero
    u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
    u = np.maximum(u, 1e-9)
    return (1 + u**2) / (u * np.sqrt(2))


def generate_microlensing_data(
    t, t0, u0, tE, baseline=1.0, anomaly_t=None, anomaly_amp=0.0, noise_std=0.02
):
    """Generates a microlensing light curve with an optional anomaly."""
    y = baseline + paczynski(t, t0, u0, tE)
    anomaly_signal = 0.0
    if anomaly_t is not None:
        # A Gaussian anomaly
        anomaly_signal = anomaly_amp * np.exp(-((t - anomaly_t) ** 2) / (2 * (0.1 * tE) ** 2))
        y += anomaly_signal

    noise = np.random.normal(0, noise_std, len(t))
    return y + noise, y, anomaly_signal


def run_demo():
    # Parameters
    n_points = 1000
    t = np.linspace(0, 100, n_points)
    t0, u0, tE = 50, 0.5, 15
    baseline = 1.0
    anomaly_t = 65
    anomaly_amp = 0.5
    noise_std = 0.01

    # 1. Generate Data
    y_data, y_true, anomaly_signal = generate_microlensing_data(
        t, t0, u0, tE, baseline, anomaly_t, anomaly_amp, noise_std
    )

    # --- STAGE 1: Event Detection (vs Baseline) ---
    # Residuals relative to a flat baseline
    residuals_baseline = y_data - baseline
    # We use a threshold for the CUSUM to detect the onset of the event
    # Note: residuals_sigma is the parameter name in the library
    cusum_event = find_anomalies_cusum(t, residuals_baseline, threshold=20.0, bidirectional=False)

    # --- STAGE 2: PSPL Fitting ---
    # Fit the Paczynski model to the data (including the anomaly)
    # We need to account for the baseline in the fit
    def fit_func(t, baseline_fit, t0_fit, u0_fit, tE_fit):
        return baseline_fit + paczynski(t, t0_fit, u0_fit, tE_fit)

    # Initial guesses
    p0 = [baseline, t0, u0, tE]
    try:
        popt, pcov = curve_fit(fit_func, t, y_data, p0=p0)
        baseline_fit, t0_fit, u0_fit, tE_fit = popt
        print(
            f"Fit successful: t0={t0_fit:.2f}, u0={u0_fit:.2f}, tE={tE_fit:.2f}, baseline={baseline_fit:.2f}"
        )
    except Exception as e:
        print(f"Fit failed: {e}")
        popt = p0
        baseline_fit, t0_fit, u0_fit, tE_fit = popt

    y_pspl = fit_func(t, *popt)

    # --- STAGE 3: Anomaly Detection (vs PSPL) ---
    # Residuals relative to the fitted PSPL model
    residuals_pspl = y_data - y_pspl
    # CUSUM on these residuals to find deviations from the model
    cusum_anomaly = find_anomalies_cusum(t, residuals_pspl, threshold=10.0, bidirectional=True)

    # --- Plotting ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), constrained_layout=True)

    # Plot 1: Light Curve & Event Detection
    ax = axes[0]
    ax.scatter(t, y_data, s=10, color="black", alpha=0.4, label="Observed Data")
    ax.axhline(baseline, color="gray", linestyle="--", label="Baseline")
    if cusum_event["triggered"]:
        ax.axvspan(
            cusum_event["onset"],
            cusum_event["end"],
            color="green",
            alpha=0.2,
            label="Event Detected (CUSUM)",
        )
    ax.set_title("Stage 1: Event Detection (Residuals vs Baseline)")
    ax.set_ylabel("Flux")
    ax.legend()

    # Plot 2: PSPL Fit
    ax = axes[1]
    ax.scatter(t, y_data, s=10, color="black", alpha=0.4, label="Observed Data")
    ax.plot(t, y_pspl, "r-", linewidth=2, label="Fitted PSPL Model")
    ax.set_title("Stage 2: PSPL Model Fitting")
    ax.set_ylabel("Flux")
    ax.legend()

    # Plot 3: Anomaly Detection (Residuals vs PSPL)
    ax = axes[2]
    ax.plot(t, residuals_pspl, "k.", alpha=0.3, label="PSPL Residuals")
    if cusum_anomaly["triggered"]:
        ax.axvspan(
            cusum_anomaly["onset"],
            cusum_anomaly["end"],
            color="red",
            alpha=0.2,
            label="Anomaly Detected (CUSUM)",
        )
        ax.axvline(
            cusum_anomaly["t0"],
            color="red",
            linestyle="--",
            label=f"Anomaly Peak: {cusum_anomaly['t0']:.1f}",
        )

    # Plot the CUSUM statistic for the anomaly
    ax.twinx().plot(
        t, cusum_anomaly["cusum_statistic"], color="blue", alpha=0.5, label="CUSUM Statistic"
    )

    ax.set_title("Stage 3: Anomaly Detection (Residuals vs PSPL)")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Residuals")
    ax.legend(loc="upper left")
    ax.twinx().set_ylabel("CUSUM Score")

    plt.savefig("docs/microlensing_cusum_workflow.png", dpi=150)
    print("Saved docs/microlensing_cusum_workflow.png")


if __name__ == "__main__":
    run_demo()
