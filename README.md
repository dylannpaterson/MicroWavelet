# MicroWavelet

`MicroWavelet` is a lightweight, standalone Continuous Wavelet Transform (CWT) peak and dip anomaly detector designed for multi-filter microlensing light curves. 

It leverages Paczynski wavelet templates to identify transient signals (such as microlensing peaks or planetary dips) across multi-filter datasets, performing robust period fine-tuning, GPR detrending, and scale-space CWT anomaly searches.

---

## Key Mathematical Features

1. **Scale-Invariant $s^{-0.5}$ Wavelet Scaling**:
   - Discrete L1-normalization of CWT kernels forces an implicit continuous CWT scale normalization of $p=1$ ($\psi_s(t) = \frac{1}{s^p} \psi(t/s)$).
   - We analytically proved that the optimal scale matching for Paczynski templates requires a scale normalization of $p=1.5$.
   - By scaling the raw CWT amplitude coefficients by $s^{-0.5}$, we ensure that the peak scale $t_{E,\text{scan}}$ aligns perfectly with the true physical timescale of the event, independent of the background noise color.

2. **Degree-5 Mismatch Bias Correction**:
   - The CWT kernel is built using a compact $u_0 = 0.05$ template. For wider events (larger $u_0$), CWT inflates the scanned scale $t_{E,\text{scan}}$ to compensate for the shape mismatch.
   - We numerically swept the exact peak ratio $r_{\text{peak}}(u_0) = t_{E,\text{scan}} / t_{E,\text{true}}$ on a dense grid of $u_0 \in [0.01, 1.0]$ and fit it with a weighted 5th-degree polynomial:
     $$r_{\text{peak}}(u_0) \approx 12.0707 u_0^5 - 29.2612 u_0^4 + 28.1550 u_0^3 - 18.9146 u_0^2 + 22.2832 u_0 - 0.0635$$
   - The final corrected crossing time is computed instantly and robustly as:
     $$t_{E,\text{true}} = t_{E,\text{scan}} / r_{\text{peak}}(u_{0,\text{event}})$$
   - This keeps estimation errors **under 2%** across the entire physical range of $u_0$.

3. **Analytical $\Delta\chi^2$ and $\Delta\text{BIC}$ Solver**:
   - Performs a closed-form, unconstrained $2 \times 2$ weighted linear least-squares Paczynski fit ($y = F_s S + F_b$) locally in a $t_0 \pm 5 t_E$ window on the raw observed points.
   - Computes parametric test statistics:
     - $\Delta\chi^2 = \chi^2_{\text{null}} - \chi^2_{\text{lens}}$
     - $\Delta\text{BIC} = \chi^2_{\text{lens}} - \chi^2_{\text{null}} + 2 \ln(N)$
   - Allows high-throughput triage filtering using input parameter `min_dchi2`.

4. **Boundary Proximity Artifact Suppression (`edge_flag`)**:
   - Automatically identifies and flags potential rising edge/boundary artifacts by setting `edge_flag = True` if the peak time $t_0$ lies within $0.5 \cdot t_E$ of the start or end of the observation window.

5. **Weighted Local Gaussian Kernel Regression (`interpolator="weighted"`)**:
   - Provides a Nadaraya-Watson-like local Gaussian kernel interpolator weighted by $1/y_{\text{err}}^2$ as a robust alternative to standard linear interpolation.
   - Vectorized using `np.searchsorted` to only evaluate a local $\pm 4h$ sliding window (where $h = 2\cdot dt$) on active grid points, running in milliseconds and resisting high-uncertainty outliers.

6. **Robust White & Red Noise Characterisation (`characterize_noise`)**:
   - Analyzes time-series noise by computing robust white noise scatter (via first-difference MAD), lag-1 and lag-2 autocorrelation, and Pont-style binning scaling excess factors.
   - Automatically flags segments with heavy correlated red noise to enable dynamic threshold scaling and error corrections.

7. **Multi-Band Chromaticity Flagging (`chromatic_flag`, `chromaticity_ratio`)**:
   - Projects the fixed Paczynski template from the primary band onto the observations of all other filter bands using a fast local 1D weighted linear least-squares fit.
   - Computes a robust `chromaticity_ratio` ($\mathcal{R} = F_{s,\text{other}} / F_{s,\text{primary}}$). If the ratio is negative (opposite directions) or wildly large ($>3.0$), or if a band with sufficient data shows absolutely no signal (detecting a chromatic stellar flare or systematic anomaly), sets `chromatic_flag = True`.
   - Protects against uneven cadence gaps by only evaluating other filters that have sufficient active observations inside the event window.

---

## Installation

```bash
pip install .
```

---

## Quick Start Example

```python
import numpy as np
from microwavelet import analyze_lightcurve, characterize_noise

# 1. Prepare multi-band relative flux data dict
data = {
    "F146": { # Primary high-cadence band
        "t": np.arange(0, 100, 0.1),
        "y": np.random.normal(1.0, 0.01, 1000),      # Relative Flux (quiescent ≈ 1.0)
        "y_err": np.ones(1000) * 0.01
    },
    "F087": { # Secondary band
        "t": np.arange(0, 100, 0.5),
        "y": np.random.normal(1.0, 0.01, 200),
        "y_err": np.ones(200) * 0.01
    }
}

# Add a synthetic achromatic microlensing peak at t0 = 50.0
# (lens amplification Paczynski model excess)
u = np.sqrt(0.1**2 + ((data["F146"]["t"] - 50.0) / 10.0)**2)
A = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
data["F146"]["y"] += (A - 1.0)

u2 = np.sqrt(0.1**2 + ((data["F087"]["t"] - 50.0) / 10.0)**2)
A2 = (u2**2 + 2.0) / (u2 * np.sqrt(u2**2 + 4.0))
data["F087"]["y"] += (A2 - 1.0)

# 2. Run the robust white and red noise characterisation utility
noise_report = characterize_noise(data["F146"]["t"], data["F146"]["y"])
print(f"White noise scatter: {noise_report['sigma_white']:.4f}")
print(f"Has red (correlated) noise: {noise_report['has_red_noise']}")

# 3. Run the CWT detector with custom options
results = analyze_lightcurve(
    data,
    interpolator="weighted",  # Use robust local error-weighted Gaussian interpolator
    cwt_threshold=12.0,       # Custom Z-score threshold
    min_dchi2=25.0            # Custom delta chi2 threshold
)

# 4. Print detected anomalies with new physical diagnostics
for anomaly in results["anomalies"]:
    print(f"Detected {anomaly['type']} at t0 = {anomaly['t0']:.3f} days")
    print(f"  Crossing time tE = {anomaly['tE']:.2f} days (u0 = {anomaly['u0']:.3f})")
    print(f"  CWT SNR = {anomaly['snr']:.2f} (band = {anomaly['band']})")
    print(f"  Analytical dchi2 = {anomaly['dchi2']:.1f} (dbic = {anomaly['dbic']:.1f})")
    print(f"  Edge Flag = {anomaly['edge_flag']} (boundary artifact suppression)")
    print(f"  Chromatic Flag = {anomaly['chromatic_flag']} (ratio = {anomaly['chromaticity_ratio']:.3f})")
```

---

## Running Verification Tests

To verify the logic and accuracy of the CWT timescale estimation:

```bash
# Verify analytic peak matching logic
python3 test_cwt_logic.py

# Verify end-to-end tE estimation accuracy
python3 test_te_accuracy.py
```

---

## Requirements

- Python >= 3.9
- numpy
- scipy
- pandas
- astropy
- scikit-learn
