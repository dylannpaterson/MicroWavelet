import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure microwavelet is in PYTHONPATH
sys.path.insert(0, os.path.abspath(".."))
from microwavelet import characterize_multiband_noise


def generate_stationary_noise(n_points, beta, correlation_factor=1.0):
    """Generates stationary noise with a specific spectral index beta."""
    n_fft = 2 ** int(np.ceil(np.log2(n_points)))
    freqs = np.fft.rfftfreq(n_fft)

    # Power law spectrum
    psd = np.zeros_like(freqs)
    psd[1:] = freqs[1:] ** (-beta)
    psd[0] = psd[1]

    # Random phases
    phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(freqs)))
    noise_fft = phases * np.sqrt(psd)
    noise = np.fft.irfft(noise_fft, n=n_fft)[:n_points]

    # Normalize
    noise = (noise - np.mean(noise)) / np.std(noise)
    return noise


def run_comparison():
    n_points = 2000
    t = np.linspace(0, 100, n_points)

    # --- CASE 1: PURE RED NOISE (Beta=2, Highly Correlated) ---
    # We create two bands that are identical (perfect correlation)
    red_noise = generate_stationary_noise(n_points, 2.0)
    bands_red = {"Band A": {"t": t, "y": red_noise}, "Band B": {"t": t, "y": red_noise}}

    # --- CASE 2: PURE WHITE NOISE (Beta=0, Uncorrelated) ---
    white_noise_a = generate_stationary_noise(n_points, 0.0)
    white_noise_b = generate_stationary_noise(n_points, 0.0)
    bands_white = {"Band A": {"t": t, "y": white_noise_a}, "Band B": {"t": t, "y": white_noise_b}}

    # --- CASE 3: MIXED NOISE (Beta=1, Partially Correlated) ---
    # Shared red component + independent white components
    shared_red = generate_stationary_noise(n_points, 1.0) * 0.8
    white_a = generate_stationary_noise(n_points, 0.0) * 0.2
    white_b = generate_stationary_noise(n_points, 0.0) * 0.2
    bands_mixed = {
        "Band A": {"t": t, "y": shared_red + white_a},
        "Band B": {"t": t, "y": shared_red + white_b},
    }

    cases = [
        ("Pure Red (Beta=2.0, Correlated)", bands_red),
        ("Pure White (Beta=0.0, Uncorrelated)", bands_white),
        ("Mixed (Beta=1.0, Partially Correlated)", bands_mixed),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15), constrained_layout=True)

    for idx, (title, bands) in enumerate(cases):
        # Run analysis
        results = characterize_multiband_noise(bands)

        # 1. Plot Spectral Indices (Left Column)
        ax_spec = axes[idx, 0]
        names = list(bands.keys())
        betas = [results["individual_metrics"][n]["beta"] for n in names]
        colors = ["#4477AA", "#EE6677"]  # Paul Tol-ish

        bars = ax_spec.bar(names, betas, color=colors, alpha=0.8)
        ax_spec.axhline(2.0, color="black", linestyle="--", alpha=0.5, label="$\\beta=2$ (Red)")
        ax_spec.axhline(0.0, color="black", linestyle=":", alpha=0.5, label="$\\beta=0$ (White)")
        ax_spec.set_ylim(-0.5, 2.5)
        ax_spec.set_ylabel("Spectral Index $\\beta$")
        ax_spec.set_title(f"{title}\nSpectral Indices")
        ax_spec.legend(loc="upper right", fontsize="small")

        # Add text labels on bars
        for bar in bars:
            height = bar.get_height()
            ax_spec.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Print the results to stdout for verification
        print(f"[{title}] Fitted Betas: {betas}")

        # 2. Plot Coherence (Right Column)
        ax_coh = axes[idx, 1]
        pair_key = f"{names[0]}_{names[1]}"
        coh_map = results["coherence_maps"][pair_key]

        im = ax_coh.imshow(
            coh_map, aspect="auto", origin="lower", extent=[0, 100, 1, 50], cmap="viridis"
        )
        ax_coh.set_title(f"Wavelet Coherence\n({pair_key})")
        ax_coh.set_xlabel("Time [days]")
        ax_coh.set_ylabel("Scale [days]")
        fig.colorbar(im, ax=ax_coh, label="Coherence $R^2$")

    plt.savefig("docs/noise_demo.png", dpi=150)
    print("Saved docs/noise_demo.png")


if __name__ == "__main__":
    run_comparison()
