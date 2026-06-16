import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process.kernels import ConstantKernel as C

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="The optimal value found for dimension 0")

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath(".."))


def simulate_microlensing_data(n_points=200):
    """Simulates a light curve with red noise, a Paczynski peak, and Gaussian noise."""
    np.random.seed(42)
    t = np.linspace(0, 100, n_points)

    # 1. Red Noise Baseline (Slowly varying sine wave + random walk)
    baseline = (
        1.0 + 0.05 * np.sin(t / 10.0) + 0.02 * np.cumsum(np.random.normal(0, 0.005, n_points))
    )

    # 2. Microlensing Anomaly (Paczynski peak)
    t0 = 50.0
    u0 = 0.1
    tE = 5.0

    def paczynski(t, t0, u0, tE):
        u = np.sqrt(u0**2 + ((t - t0) / tE) ** 2)
        return (u**2 + 2) / (u * np.sqrt(u**2 + 4))

    magnification = paczynski(t, t0, u0, tE)
    signal = baseline * magnification

    # 3. Gaussian Noise
    y_err = 0.02 * baseline
    y = signal + np.random.normal(0, y_err)

    return t, y, y_err, baseline, signal


def standard_gp_detrend(t, y, y_err):
    """Standard GP detrending (prone to signal absorption)."""
    t_reshaped = t.reshape(-1, 1)
    kernel = C(1.0) * RBF(length_scale=20.0, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(
        noise_level=0.01
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0)
    gp.fit(t_reshaped, y)
    return gp.predict(t_reshaped)


def robust_masked_gp_detrend(t, y, y_err, threshold=3.0):
    """Robust Masked GP detrending (prevents signal absorption)."""
    t_reshaped = t.reshape(-1, 1)

    # Step 1: Initial Fit with bounds to prevent collapse
    kernel = C(1.0) * RBF(length_scale=20.0, length_scale_bounds=(1.0, 100.0)) + WhiteKernel(
        noise_level=0.01
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0)
    gp.fit(t_reshaped, y)
    y_pred_initial = gp.predict(t_reshaped)

    # Step 2: Identify Outliers (Standardized Residuals)
    residuals = (y - y_pred_initial) / y_err
    mask = np.abs(residuals) < threshold

    # Step 3: Re-fit using only masked points
    if np.any(mask):
        t_masked = t[mask].reshape(-1, 1)
        y_masked = y[mask]

        gp_robust = GaussianProcessRegressor(kernel=kernel, alpha=0)
        gp_robust.fit(t_masked, y_masked)
        y_baseline = gp_robust.predict(t_reshaped)
    else:
        y_baseline = y_pred_initial

    return y_baseline


def generate_plot():
    # Set up styling for publication-quality visual excellence
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Inter"]
    plt.rcParams["text.color"] = "#1a1a1a"
    plt.rcParams["axes.labelcolor"] = "#1a1a1a"
    plt.rcParams["xtick.color"] = "#4a4a4a"
    plt.rcParams["ytick.color"] = "#4a4a4a"

    # 1. Generate synthetic data
    t, y, y_err, true_baseline, true_signal = simulate_microlensing_data()

    # 2. Run both methods
    y_std_gp = standard_gp_detrend(t, y, y_err)
    y_robust_gp = robust_masked_gp_detrend(t, y, y_err)

    # Calculate Residuals (The "Whitened" signal)
    res_std = (y - y_std_gp) / y_err
    res_robust = (y - y_robust_gp) / y_err

    # 3. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), dpi=200, sharex=True)
    fig.patch.set_facecolor("#ffffff")

    for ax in axes:
        ax.set_facecolor("#f8f9fa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(True, linestyle=":", alpha=0.5)

    # Panel 1: Original Data and Baselines
    axes[0].errorbar(
        t, y, yerr=y_err, fmt=".", color="gray", alpha=0.4, label="Observed Data", zorder=1
    )
    axes[0].plot(t, true_baseline, "k--", label="True Baseline", alpha=0.7, zorder=2)
    axes[0].plot(t, y_std_gp, "r-", label="Naive GP Baseline", linewidth=2, zorder=3)
    axes[0].plot(t, y_robust_gp, "g-", label="Robust GP Baseline", linewidth=2, zorder=4)
    axes[0].set_ylabel("Relative Flux", fontweight="bold", fontsize=12)
    axes[0].set_title("Baseline Recovery Comparison", fontsize=14, fontweight="bold", loc="left")
    axes[0].legend(loc="upper right", frameon=True, facecolor="#ffffff", fontsize=10)

    # Panel 2: Naive Residuals
    axes[1].plot(t, res_std, "r-", alpha=0.6, label="Naive Residuals", linewidth=1.5)
    axes[1].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_ylabel(r"Standardized Residuals ($\sigma$)", fontweight="bold", fontsize=12)
    axes[1].set_title(
        "Naive Residuals (Signal is 'Flattened')", fontsize=14, fontweight="bold", loc="left"
    )
    axes[1].legend(loc="upper right", frameon=True, facecolor="#ffffff", fontsize=10)

    # Panel 3: Robust Residuals
    axes[2].plot(t, res_robust, "g-", alpha=0.8, label="Robust Residuals", linewidth=1.5)
    axes[2].axhline(0, color="black", linestyle="--", alpha=0.5)
    axes[2].axhline(3.0, color="red", linestyle=":", alpha=0.5, label=r"3$\sigma$ Threshold")
    axes[2].axhline(-3.0, color="red", linestyle=":", alpha=0.5)
    axes[2].set_ylabel(r"Standardized Residuals ($\sigma$)", fontweight="bold", fontsize=12)
    axes[2].set_xlabel("Time (Days)", fontweight="bold", fontsize=12)
    axes[2].set_title(
        "Robust Residuals (Signal is 'Preserved')", fontsize=14, fontweight="bold", loc="left"
    )
    axes[2].legend(loc="upper right", frameon=True, facecolor="#ffffff", fontsize=10)

    plt.tight_layout()

    out_path = "microwavelet/docs/gp_comparison.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    print(f"✅ Comparison plot saved to {out_path}")


if __name__ == "__main__":
    generate_plot()
