import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath(".."))
from microwavelet import characterize_multiband_noise

def generate_synthetic_multiband_noise(n_points=500):
    t = np.linspace(0, 100, n_points)
    
    # Band 1: Red noise (f^-2) + white noise
    # We can simulate red noise by integrating white noise
    white_noise = np.random.normal(0, 0.05, n_points)
    red_noise = np.cumsum(white_noise)
    red_noise = (red_noise - np.mean(red_noise)) / np.std(red_noise) * 0.1
    y1 = 1.0 + red_noise + np.random.normal(0, 0.02, n_points)
    
    # Band 2: Similar red noise but slightly different
    y2 = 1.0 + red_noise * 0.8 + np.random.normal(0, 0.02, n_points)
    
    # Band 3: Mostly white noise
    y3 = 1.0 + np.random.normal(0, 0.02, n_points)
    
    bands_data = {
        "W146": {"t": t, "y": y1, "y_err": np.ones(n_points) * 0.02},
        "W184": {"t": t, "y": y2, "y_err": np.ones(n_points) * 0.02},
        "F087": {"t": t, "y": y3, "y_err": np.ones(n_points) * 0.02}
    }
    return bands_data, t

def main():
    bands_data, t = generate_synthetic_multiband_noise()
    
    # Run analysis
    widths = np.geomspace(1.0, 20.0, 20)
    results = characterize_multiband_noise(bands_data, widths=widths)
    
    # 3. Plotting
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Inter"]
    plt.rcParams["text.color"] = "#1a1a1a"
    plt.rcParams["axes.labelcolor"] = "#1a1a1a"
    plt.rcParams["xtick.color"] = "#4a4a4a"
    plt.rcParams["ytick.color"] = "#4a4a4a"

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    # Flat UI Palette from _core.py
    palette = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22"]

    for ax in axes:
        ax.set_facecolor("#f8f9fa")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cccccc")
        ax.spines["bottom"].set_color("#cccccc")
        ax.grid(True, linestyle=":", alpha=0.5)

    # 1. Spectral Indices
    band_names = list(results["individual_metrics"].keys())
    betas = [results["individual_metrics"][name]["beta"] for name in band_names]
    axes[0].bar(band_names, betas, color=palette[:len(band_names)], alpha=0.8, edgecolor="#ffffff", linewidth=1)
    axes[0].set_ylabel("Spectral Index ($\\beta$)", fontweight="bold", fontsize=12)
    axes[0].set_title("Estimated Spectral Indices ($P(f) \\sim f^{-\\beta}$)", fontsize=14, fontweight="bold", loc="left")
    axes[0].axhline(2.0, color="#2c3e50", linestyle="--", alpha=0.5, label="Red Noise ($\\beta=2$)")
    axes[0].legend(loc="upper right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc")

    # 2. Wavelet Coherence Map (W146 vs W184)
    pair_key = "W146_W184"
    if pair_key in results["coherence_maps"]:
        coh_map = results["coherence_maps"][pair_key]
        im = axes[1].imshow(coh_map, aspect='auto', origin='lower', 
                            extent=[t[0], t[-1], widths[0], widths[-1]],
                            cmap='viridis')
        fig.colorbar(im, ax=axes[1], label='Coherence')
        axes[1].set_ylabel("Scale (Width)", fontweight="bold", fontsize=12)
        axes[1].set_xlabel("Time (Days)", fontweight="bold", fontsize=12)
        axes[1].set_title(f"Wavelet Coherence: {pair_key}", fontsize=14, fontweight="bold", loc="left")
    else:
        axes[1].text(0.5, 0.5, "Coherence map not found", ha='center')

    plt.tight_layout()
    out_path = "../docs/noise_demo.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="#ffffff")
    print(f"✅ Noise demo plot saved to {out_path}")

if __name__ == "__main__":
    main()
