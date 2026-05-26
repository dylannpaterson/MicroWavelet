import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import median_abs_deviation

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath('.'))
from microwavelet import analyze_lightcurve

def generate_detrend_plot():
    # Set up styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Inter']
    plt.rcParams['text.color'] = '#1a1a1a'
    plt.rcParams['axes.labelcolor'] = '#1a1a1a'
    plt.rcParams['xtick.color'] = '#4a4a4a'
    plt.rcParams['ytick.color'] = '#4a4a4a'

    # 1. Generate synthetic data: CV-like periodic variable + Microlensing event
    np.random.seed(42)
    t = np.arange(0, 100, 0.1)

    # CV-like baseline: Narrow eclipses + double-hump ellipsoidal modulation
    period_true = 4.21
    phase_true = (t % period_true) / period_true
    # Ellipsoidal modulation (sin 2phi)
    ellipsoidal = 0.05 * np.sin(4 * np.pi * phase_true)
    # Sharp Gaussian eclipses
    eclipses = -0.3 * np.exp(-0.5 * ((np.mod(phase_true + 0.1, 1.0) - 0.5) / 0.02)**2)
    baseline_flux = 1.0 + ellipsoidal + eclipses

    # Microlensing event (tE=12.0, u0=0.15 -> A_peak ~ 6.7)
    t0, tE, u0 = 55.0, 12.0, 0.15
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))

    # Combined signal with noise
    noise_lvl = 0.015
    y_raw = baseline_flux * A + np.random.normal(0, noise_lvl, size=len(t))
    y_err = np.ones_like(t) * noise_lvl

    
    data = {"Band1": {"t": t, "y": y_raw, "y_err": y_err}}
    
    # 2. Run the pipeline
    results = analyze_lightcurve(data, detrend_periodic=True, min_period=1.0, max_period=10.0)
    
    detrend = results["detrending"]["bands"]["Band1"]
    y_detrended = detrend["y_detrended"]
    baseline_model = detrend["baseline_model"]
    phase = detrend["phase"]
    search_mask = results["detrending"]["search_mask"]
    outlier_mask = detrend["outlier_mask"]
    recovered_period = results["detrending"]["period_days"]

    # 3. Plotting with Robust Dynamic Scaling
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), dpi=200)
    fig.patch.set_facecolor('#ffffff')
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Dynamic limits for panels with the full ML event
    y_min, y_max = np.min(y_raw), np.max(y_raw)
    y_range = y_max - y_min
    full_ylims = (y_min - 0.1 * y_range, y_max + 0.2 * y_range)

    # Panel 1: Step 1 - Robust Search
    ax1.scatter(t[~search_mask], y_raw[~search_mask], color='#e74c3c', alpha=0.4, s=25, label='Clipped Transients', zorder=3)
    ax1.errorbar(t[search_mask], y_raw[search_mask], yerr=y_err[search_mask], fmt='o', color='#7f8c8d', alpha=0.6, markersize=3, label='Baseline Points', zorder=2)
    ax1.set_ylabel("Relative Flux", fontweight='bold', fontsize=11)
    ax1.set_title("Step 1: Robust Period Search (Masking Transients)", loc='left', fontsize=13, fontweight='bold')
    ax1.legend(loc="upper right", frameon=True, facecolor='#ffffff', fontsize=10)
    ax1.set_ylim(full_ylims)

    # Panel 2: Step 2 - GP Model (Focus on baseline periodicity)
    sort_idx = np.argsort(phase)
    # Background all data lightly to show the 'smear'
    ax2.scatter(phase, y_raw, color='#7f8c8d', alpha=0.1, s=5)
    # Highlight points used in the converged model
    ax2.scatter(phase[outlier_mask], y_raw[outlier_mask], color='#7f8c8d', alpha=0.5, s=10, label='Folded Baseline')
    ax2.plot(phase[sort_idx], baseline_model[sort_idx], color='#e74c3c', linewidth=3, label=f'GP Model (P={recovered_period:.2f}d)', zorder=4)
    ax2.set_ylabel("Relative Flux", fontweight='bold', fontsize=11)
    ax2.set_xlabel("Phase", fontweight='bold', fontsize=11)
    ax2.set_title("Step 2: Iterative GP Baseline Modeling", loc='left', fontsize=13, fontweight='bold')
    ax2.legend(loc="upper right", frameon=True, facecolor='#ffffff', fontsize=10)
    # Focus on the actual periodic baseline modulation (ignore smeared event)
    ax2.set_ylim(0.7, 1.4)

    # Panel 3: Step 3 - Final Recovered Signal
    ax3.errorbar(t, y_detrended, yerr=detrend["y_err"], fmt='o', color='#3498db', markersize=3, alpha=0.6, label='Detrended Data', zorder=2)
    ax3.plot(t, A, color='#2ecc71', linewidth=2.5, label='True Microlensing Signal', alpha=0.9, zorder=3)
    ax3.set_xlabel("Time (Days)", fontweight='bold', fontsize=11)
    ax3.set_ylabel("Normalized Flux", fontweight='bold', fontsize=11)
    ax3.set_title("Step 3: Isolated Microlensing Event", loc='left', fontsize=13, fontweight='bold')
    ax3.legend(loc="upper right", frameon=True, facecolor='#ffffff', fontsize=10)
    # Scaling matches raw data panel to show the 'unfolding' of the event
    det_min, det_max = np.min(y_detrended), np.max(y_detrended)
    det_range = det_max - det_min
    ax3.set_ylim(det_min - 0.1 * det_range, det_max + 0.2 * det_range)

    plt.suptitle("MicroWavelet: Periodic Baseline Detrending Pipeline", fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_path = 'docs/detrend_demo.png'
    os.makedirs('docs', exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', facecolor='#ffffff')
    print(f"✅ Polished stepwise demonstration plot saved to {out_path}")

if __name__ == "__main__":
    generate_detrend_plot()
