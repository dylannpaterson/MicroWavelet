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

---

## Installation

```bash
pip install .
```

---

## Quick Start Example

```python
import numpy as np
from microwavelet import analyze_lightcurve

# Prepare relative flux light curve data dict
data = {
    "F146": {
        "t": np.array([50.1, 50.2, 50.3, ...]),      # Times (days)
        "y": np.array([1.01, 1.05, 1.12, ...]),      # Relative Flux (quiescent ≈ 1.0)
        "y_err": np.array([0.01, 0.01, 0.01, ...])   # Errors
    }
}

# Run the anomaly detector
results = analyze_lightcurve(data)

# Print detected anomalies
for anomaly in results["anomalies"]:
    print(f"Detected {anomaly['type']} at t0 = {anomaly['t0']:.3f} days")
    print(f"Estimated crossing time tE = {anomaly['tE']:.2f} days")
    print(f"Estimated impact parameter u0 = {anomaly['u0']:.3f}")
    print(f"CWT Consensus SNR = {anomaly['snr']:.2f}")
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
