import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath('.'))
from microwavelet import analyze_lightcurve

def generate_plot():
    # Set up styling for publication-quality visual excellence
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Inter']
    plt.rcParams['text.color'] = '#1a1a1a'
    plt.rcParams['axes.labelcolor'] = '#1a1a1a'
    plt.rcParams['xtick.color'] = '#4a4a4a'
    plt.rcParams['ytick.color'] = '#4a4a4a'
    
    # 1. Generate synthetic achromatic microlensing data
    t = np.arange(0, 100, 0.1)
    t0 = 50.0
    tE = 10.0
    u0 = 0.1
    
    # Paczynski amplification excess
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
    S = A - 1.0
    
    np.random.seed(42)
    
    # Primary Band (F146) with lensed peak and small noise
    y_f146 = S + np.random.normal(0, 0.008, size=len(t))
    y_err_f146 = np.ones_like(t) * 0.008
    
    # Secondary Band (F087) with lensed peak and larger cadence
    t_f087 = np.arange(0, 100, 0.5)
    u_f087 = np.sqrt(u0**2 + ((t_f087 - t0) / tE)**2)
    S_f087 = (u_f087**2 + 2.0) / (u_f087 * np.sqrt(u_f087**2 + 4.0)) - 1.0
    y_f087 = S_f087 + np.random.normal(0, 0.012, size=len(t_f087))
    y_err_f087 = np.ones_like(t_f087) * 0.012
    
    # 2. Run the CWT Pipeline
    data = {
        "F146": {"t": t, "y": y_f146 + 1.0, "y_err": y_err_f146}, # shift quiescent to 1.0
        "F087": {"t": t_f087, "y": y_f087 + 1.0, "y_err": y_err_f087}
    }
    results = analyze_lightcurve(data, interpolator="weighted", cwt_threshold=12.0)
    
    # Extract diagnostics for plotting
    diag = results["diagnostics"]["cwt"]
    t_grid = diag["t_grid"]
    f_interp = diag["f_interp"]
    c1d_even = diag["consensus_1d_even"]
    c1d_odd = diag["consensus_1d_odd"]
    
    # Create the beautiful dual-panel visual layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=200)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#f8f9fa')
    ax2.set_facecolor('#f8f9fa')
    
    # Panel 1: Multi-band Light Curve & Fitted Templates
    ax1.errorbar(t, y_f146 + 1.0, yerr=y_err_f146, fmt='o', color='#3498db', alpha=0.5, label='F146 (Primary Band)', markersize=3, elinewidth=0.5)
    ax1.errorbar(t_f087, y_f087 + 1.0, yerr=y_err_f087, fmt='s', color='#e74c3c', alpha=0.7, label='F087 (Secondary Band)', markersize=3.5, elinewidth=0.7)
    
    # Overlay interpolator grid
    ax1.plot(t_grid, f_interp + 1.0, color='#2c3e50', linestyle='--', alpha=0.4, label='Local Gaussian Interpolation', linewidth=1.2)
    
    # Plot the analytical best-fit model for primary band
    if results["anomalies"]:
        anom = results["anomalies"][0]
        # Reconstruct fitted curve
        fit_t = np.linspace(20, 80, 500)
        fit_u = np.sqrt(anom["u0"]**2 + ((fit_t - anom["t0"]) / anom["tE"])**2)
        fit_A = (fit_u**2 + 2.0) / (fit_u * np.sqrt(fit_u**2 + 4.0))
        ax1.plot(fit_t, fit_A, color='#2ecc71', label=f'Best-Fit Paczynski ($t_E={anom["tE"]:.2f}$ d)', linewidth=2.0)
        ax1.axvline(anom["t0"], color='#2ecc71', linestyle=':', alpha=0.8, label=f'Peak Centre $t_0={anom["t0"]:.2f}$')
        
    ax1.set_ylabel("Relative Flux", fontsize=11, fontweight='bold')
    ax1.set_title("MicroWavelet: CWT Detection & Parametric Fit Demonstration", fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc="upper right", frameon=True, facecolor='#ffffff', edgecolor='#e1e4e6')
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_ylim(0.9, 2.2)
    
    # Panel 2: CWT Scale-Space Consensus SNR
    ax2.plot(t_grid, c1d_even, color='#9b59b6', label='Even (Symmetric) Consensus SNR', linewidth=1.8)
    ax2.plot(t_grid, c1d_odd, color='#f1c40f', label='Odd (Asymmetric) Consensus SNR', linewidth=1.2, alpha=0.7)
    ax2.axhline(12.0, color='#e74c3c', linestyle='--', label='Default Detection Threshold (12.0σ)', linewidth=1.0)
    
    if results["anomalies"]:
        ax2.scatter([anom["t0"]], [anom["snr"]], color='#e74c3c', s=80, zorder=5, edgecolor='#ffffff', linewidth=1.5, label=f'Detection Peak ({anom["snr"]:.1f}σ)')
        
    ax2.set_xlabel("Time (BJD - Offset)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("CWT Consensus Z-Score (SNR)", fontsize=11, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True, facecolor='#ffffff', edgecolor='#e1e4e6')
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_ylim(-3, 35)
    
    # Beautiful styling details
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        
    plt.tight_layout()
    
    os.makedirs('docs', exist_ok=True)
    out_path = 'docs/cwt_demo.png'
    plt.savefig(out_path, bbox_inches='tight', facecolor='#ffffff')
    print(f"✅ Diagnostic demonstration plot saved successfully to {out_path}!")

if __name__ == "__main__":
    generate_plot()
