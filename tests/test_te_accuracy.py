import numpy as np
import pandas as pd
from microwavelet import analyze_lightcurve

def paczynski(t, t0, tE, u0, baseline=1.0):
    u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return baseline * A

def test_te_accuracy():
    t_obs = np.arange(0, 100, 0.1)
    t0_true = 50.0
    tE_true = 10.0
    u0_list = [0.05, 0.1, 0.3, 0.5]
    
    print(f"{'u0_true':<10} | {'tE_true':<10} | {'tE_scan':<10} | {'tE_est':<10} | {'Error (%)':<10}")
    print("-" * 70)
    
    for u0_true in u0_list:
        y_flux = paczynski(t_obs, t0_true, tE_true, u0_true)
        # Add tiny noise to avoid numerical issues but keep it clean for accuracy check
        y_err = np.ones_like(y_flux) * 0.001
        y_obs = y_flux + np.random.normal(0, 0.001, size=len(y_flux))
        
        data = {
            "test_band": {
                "t": t_obs,
                "y": y_obs,
                "y_err": y_err
            }
        }
        
        results = analyze_lightcurve(data)
        anomalies = results["anomalies"]
        
        if anomalies:
            # Find the anomaly closest to t0_true
            best_anomaly = min(anomalies, key=lambda x: abs(x["t0"] - t0_true))
            tE_est = best_anomaly["tE"]
            tE_scan = best_anomaly["tE_scan"]
            error = abs(tE_est - tE_true) / tE_true * 100
            print(f"{u0_true:<10.2f} | {tE_true:<10.1f} | {tE_scan:<10.2f} | {tE_est:<10.2f} | {error:<10.2f}")
        else:
            print(f"{u0_true:<10.2f} | {tE_true:<10.1f} | {'FAILED':<10} | {'FAILED':<10} | {'N/A':<10}")

if __name__ == "__main__":
    test_te_accuracy()
