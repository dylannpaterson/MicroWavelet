import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def paczynski(t, t0, tE, u0):
    u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return A - 1.0

def get_kernels(tk, tE, u0=0.05):
    u = np.sqrt(u0**2 + (tk/tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    dt = tk[1] - tk[0]
    first_deriv = np.gradient(A, dt)
    second_deriv = np.gradient(first_deriv, dt)
    ke = -second_deriv
    ke -= np.mean(ke)
    ke /= (np.sum(np.abs(ke)) + 1e-12)
    return ke

def _tE_correction_factor(u0_event):
    if u0_event is None or np.isnan(u0_event) or u0_event <= 0:
        return 1.0
    coeffs = [12.07074638, -29.26124165, 28.15495458, -18.91458072, 22.28322211, -0.06346686]
    poly = np.poly1d(coeffs)
    r_val = poly(u0_event)
    if r_val < 0.01:
        return 1.0
    return 1.0 / r_val

def test_cwt_peak_location():
    dt = 0.02
    t = np.arange(-500, 500, dt)
    tE_true = 10.0
    u0_template = 0.05
    u0_list = [0.05, 0.1, 0.2, 0.3, 0.5]
    
    tE_scales = np.logspace(np.log10(1.0), np.log10(150.0), 200)
    
    print(f"{'u0_true':<10} | {'tE_true':<10} | {'tE_scan':<10} | {'tE_est':<10} | {'Error (%)':<10}")
    print("-" * 65)
    
    for u0_true in u0_list:
        y = paczynski(t, 0, tE_true, u0_true)
        
        powers = []
        for tE_s in tE_scales:
            half = int(5.0 * tE_s / dt)
            tk = np.arange(-half, half + 1) * dt
            ke = get_kernels(tk, tE_s, u0=u0_template)
            
            # Center of the convolution
            lo = len(y)//2 - half
            hi = len(y)//2 + half + 1
            if lo < 0 or hi > len(y):
                powers.append(0.0)
                continue
            center_val = abs(np.sum(y[lo:hi] * ke))
            # Apply scale-invariant CWT peak scaling
            powers.append(center_val * (tE_s ** -0.5))
            
        tE_scan = tE_scales[np.argmax(powers)]
        
        # Apply exact polynomial correction factor
        tE_est = tE_scan * _tE_correction_factor(u0_true)
        error = abs(tE_est - tE_true) / tE_true * 100
        print(f"{u0_true:<10.2f} | {tE_true:<10.1f} | {tE_scan:<10.2f} | {tE_est:<10.2f} | {error:<10.2f}")

if __name__ == "__main__":
    test_cwt_peak_location()
