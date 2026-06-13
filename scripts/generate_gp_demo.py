import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from scipy.optimize import minimize

def simulate_microlensing_data(n_points=200):
    """Simulates a light curve with red noise, a Paczynski peak, and Gaussian noise."""
    np.random.seed(42)
    t = np.linspace(0, 100, n_points)
    
    # 1. Red Noise Baseline (Slowly varying sine wave + random walk)
    baseline = 1.0 + 0.05 * np.sin(t / 10.0) + 0.02 * np.cumsum(np.random.normal(0, 0.005, n_points))
    
    # 2. Microlensing Anomaly (Paczynski peak)
    t0 = 50.0
    u0 = 0.1
    tE = 5.0
    def paczynski(t, t0, u0, tE):
        u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
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
    kernel = C(1.0) * RBF(length_scale=20.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2)
    gp.fit(t_reshaped, y)
    return gp.predict(t_reshaped)

def robust_masked_gp_detrend(t, y, y_err, threshold=3.0):
    """Robust Masked GP detrending (prevents signal absorption)."""
    t_reshaped = t.reshape(-1, 1)
    
    # Step 1: Initial Fit
    kernel = C(1.0) * RBF(length_scale=20.0) + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2)
    gp.fit(t_reshaped, y)
    y_pred_initial = gp.predict(t_reshaped)
    
    # Step 2: Identify Outliers (Standardized Residuals)
    residuals = (y - y_pred_initial) / y_err
    mask = np.abs(residuals) < threshold
    
    # Step 3: Re-fit using only masked points
    if np.any(mask):
        t_masked = t[mask].reshape(-1, 1)
        y_masked = y[mask]
        y_err_masked = y_err[mask]
        
        gp_robust = GaussianProcessRegressor(kernel=kernel, alpha=y_err_masked**2)
        gp_robust.fit(t_masked, y_masked)
        y_baseline = gp_robust.predict(t_reshaped)
    else:
        y_baseline = y_pred_initial
        
    return y_baseline

if __name__ == "__main__":
    # --- Execution ---
    t, y, y_err, true_baseline, true_signal = simulate_microlensing_data()

    # Run both methods
    y_std_gp = standard_gp_detrend(t, y, y_err)
    y_robust_gp = robust_masked_gp_detrend(t, y, y_err)

    # Calculate Residuals (The "Whitened" signal)
    res_std = (y - y_std_gp) / y_err
    res_robust = (y - y_robust_gp) / y_err

    # Plotting results
    plt.figure(figsize=(12, 12))

    # Plot 1: Original Data and Baselines
    plt.subplot(3, 1, 1)
    plt.errorbar(t, y, yerr=y_err, fmt='.', color='gray', alpha=0.5, label='Data')
    plt.plot(t, true_baseline, 'k--', label='True Baseline', alpha=0.8)
    plt.plot(t, y_std_gp, 'r-', label='Naive GP Baseline')
    plt.plot(t, y_robust_gp, 'g-', label='Robust GP Baseline')
    plt.title("Baseline Recovery Comparison")
    plt.legend()

    # Plot 2: Naive Residuals
    plt.subplot(3, 1, 2)
    plt.plot(t, res_std, 'r-', alpha=0.6, label='Naive Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Naive Residuals (Suppressed)")
    plt.legend()

    # Plot 3: Robust Residuals
    plt.subplot(3, 1, 3)
    plt.plot(t, res_robust, 'g-', alpha=0.8, label='Robust Residuals')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Robust Residuals (Preserved)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("microwavelet/docs/gp_comparison.png")
    print("Comparison plot saved to 'microwavelet/docs/gp_comparison.png'")
