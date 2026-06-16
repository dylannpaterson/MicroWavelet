import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.timeseries import LombScargle

# Add current dir to path to import microwavelet
sys.path.insert(0, os.path.abspath("."))

from microwavelet._noise import estimate_spectral_index


def test_red_noise():
    n_points = 2000
    t = np.linspace(0, 100, n_points)

    # Generate stationary red noise (beta=2)
    n_fft = 2 ** int(np.ceil(np.log2(n_points)))
    freqs = np.fft.rfftfreq(n_fft)
    psd_true = np.zeros_like(freqs)
    psd_true[1:] = freqs[1:] ** (-2.0)
    psd_true[0] = psd_true[1]

    phases = np.exp(1j * np.random.uniform(0, 2 * np.pi, len(freqs)))
    noise_fft = phases * np.sqrt(psd_true)
    noise = np.fft.irfft(noise_fft, n=n_fft)[:n_points]
    noise = (noise - np.mean(noise)) / np.std(noise) * 0.1

    # Add a tiny white noise floor
    noise += np.random.normal(0, 0.001, n_points)

    print("--- Testing Pure Red Noise ---")
    res = estimate_spectral_index(t, noise)
    print(f"Result: {res}")

    if "error" in res:
        print(f"Error: {res['error']}")
        return

    # Plotting for inspection
    plt.figure(figsize=(10, 6))
    ls = LombScargle(t, noise)
    f, p = ls.autopower()

    plt.loglog(f, p, label="Lomb-Scargle Power", alpha=0.5)

    # Plot the fit
    f_fit = res["frequency"]
    p_fit = res["power"]
    plt.loglog(f_fit, p_fit, "r--", label="Fit")

    # Reconstruct model for plotting
    A = res["A"]
    beta = res["beta"]
    C = res["C"]
    f_model = np.logspace(np.log10(f[1]), np.log10(f[-1]), 100)
    p_model = A * f_model ** (-beta) + C
    plt.loglog(f_model, p_model, "k-", label="Model")

    plt.xlabel("Frequency")
    plt.ylabel("Power")
    plt.legend()
    plt.title(f"PSD Fit (Beta={res.get('beta', 'N/A')})")
    plt.savefig("debug_red_noise.png")
    print("Saved debug_red_noise.png")


if __name__ == "__main__":
    test_red_noise()
