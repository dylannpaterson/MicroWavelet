import numpy as np
from microwavelet import analyze_lightcurve

def paczynski(t, t0, tE, u0, baseline=1.0):
    u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return baseline * A

def test_new_features():
    # 1. Create a synthetic light curve with an event in the middle and an event near the edge
    t_obs = np.arange(0, 100, 0.1)
    
    # Event 1: Clean event in the middle (t0 = 50.0)
    y_flux_mid = paczynski(t_obs, 50.0, 10.0, 0.1)
    # Event 2: Event near the left edge (t0 = 1.0)
    y_flux_edge = paczynski(t_obs, 1.0, 5.0, 0.05) - 1.0
    
    y_flux = y_flux_mid + y_flux_edge
    y_err = np.ones_like(y_flux) * 0.01
    y_obs = y_flux + np.random.normal(0, 0.01, size=len(y_flux))
    
    data = {
        "band1": {
            "t": t_obs,
            "y": y_obs,
            "y_err": y_err
        }
    }
    
    # Test linear interpolator (default)
    res_linear = analyze_lightcurve(data)
    print("Linear results:")
    for a in res_linear["anomalies"]:
        print(f"t0: {a['t0']:.2f}, tE: {a['tE']:.2f}, snr: {a['snr']:.2f}, dchi2: {a['dchi2']:.2f}, dbic: {a['dbic']:.2f}, edge: {a['edge_flag']}")
        # Event near t0=50 should have edge_flag = False
        if abs(a["t0"] - 50.0) < 5.0:
            assert not a["edge_flag"], "Event in the middle should not be flagged as edge"
        # Event near t0=1 should have edge_flag = True
        if abs(a["t0"] - 1.0) < 3.0:
            assert a["edge_flag"], "Event near the edge should be flagged as edge"
            
    # Test weighted interpolator
    res_weighted = analyze_lightcurve(data, interpolator="weighted")
    print("\nWeighted results:")
    for a in res_weighted["anomalies"]:
        print(f"t0: {a['t0']:.2f}, tE: {a['tE']:.2f}, snr: {a['snr']:.2f}, dchi2: {a['dchi2']:.2f}, dbic: {a['dbic']:.2f}, edge: {a['edge_flag']}")
        
    # Test min_dchi2 filtering
    # Detections should have huge dchi2 (e.g. >1000). Let's filter with min_dchi2 = 1000000 to verify no anomalies found,
    # or min_dchi2 = 100 to verify they are kept.
    res_filtered_none = analyze_lightcurve(data, min_dchi2=1e8)
    assert len(res_filtered_none["anomalies"]) == 0, "High min_dchi2 should filter out all anomalies"
    
    res_filtered_keep = analyze_lightcurve(data, min_dchi2=10.0)
    assert len(res_filtered_keep["anomalies"]) > 0, "Low min_dchi2 should keep the anomalies"
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    test_new_features()
