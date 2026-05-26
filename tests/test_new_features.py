import numpy as np
from microwavelet import analyze_lightcurve, characterize_noise

def paczynski(t, t0, tE, u0, baseline=1.0):
    u = np.sqrt(u0**2 + ((t - t0)/tE)**2)
    A = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    return baseline * A

def test_new_features():
    np.random.seed(42)
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

    # Test multi-band chromaticity logic
    # Achromatic Event (Clean lensed signal in both band1 and band2)
    t_multi = np.arange(0, 100, 0.1)
    y_flux1 = paczynski(t_multi, 50.0, 10.0, 0.1)
    y_flux2 = paczynski(t_multi, 50.0, 10.0, 0.1)  # identical achromatic amplitude
    
    data_achromatic = {
        "band1": {"t": t_multi, "y": y_flux1 + np.random.normal(0, 0.005, size=len(t_multi)), "y_err": np.ones_like(t_multi) * 0.005},
        "band2": {"t": t_multi, "y": y_flux2 + np.random.normal(0, 0.005, size=len(t_multi)), "y_err": np.ones_like(t_multi) * 0.005}
    }
    res_achromatic = analyze_lightcurve(data_achromatic)
    print("\nAchromatic Event Results:")
    for a in res_achromatic["anomalies"]:
        print(f"t0: {a['t0']:.2f}, chromatic: {a['chromatic_flag']}, ratio: {a['chromaticity_ratio']:.3f}")
        assert not a["chromatic_flag"], "Achromatic event should have chromatic_flag = False"
        assert abs(a["chromaticity_ratio"] - 1.0) < 0.2, "Achromatic ratio should be close to 1.0"

    # Chromatic Event (Stellar Flare / CV / Systematic excursion only in band1, flat in band2)
    data_chromatic = {
        "band1": {"t": t_multi, "y": y_flux1 + np.random.normal(0, 0.005, size=len(t_multi)), "y_err": np.ones_like(t_multi) * 0.005},
        "band2": {"t": t_multi, "y": 1.0 + np.random.normal(0, 0.005, size=len(t_multi)), "y_err": np.ones_like(t_multi) * 0.005} # completely flat
    }
    res_chromatic = analyze_lightcurve(data_chromatic)
    print("\nChromatic Event Results:")
    for a in res_chromatic["anomalies"]:
        print(f"t0: {a['t0']:.2f}, chromatic: {a['chromatic_flag']}, ratio: {a['chromaticity_ratio']:.3f}")
        assert a["chromatic_flag"], "Chromatic event should have chromatic_flag = True"

    # Test noise characterisation utility
    # Pure white noise test
    t_noise = np.arange(0, 50, 0.1)
    y_noise_white = np.random.normal(0, 0.05, size=len(t_noise))
    metrics_white = characterize_noise(t_noise, y_noise_white)
    print("\nWhite Noise Characterisation:")
    print(f"Sigma White: {metrics_white['sigma_white']:.4f}, Total: {metrics_white['sigma_total']:.4f}")
    print(f"Autocorr Lag 1: {metrics_white['autocorr_lag1']:.4f}, Has Red Noise: {metrics_white['has_red_noise']}")
    assert abs(metrics_white["sigma_white"] - 0.05) < 0.015
    assert not metrics_white["has_red_noise"], "White noise should not be flagged as red noise"

    # Red noise test (correlated random walk / AR(1) process)
    y_noise_red = np.zeros_like(t_noise)
    for i in range(1, len(t_noise)):
        y_noise_red[i] = 0.8 * y_noise_red[i-1] + np.random.normal(0, 0.03)
    metrics_red = characterize_noise(t_noise, y_noise_red)
    print("\nRed Noise Characterisation:")
    print(f"Sigma White: {metrics_red['sigma_white']:.4f}, Total: {metrics_red['sigma_total']:.4f}")
    print(f"Autocorr Lag 1: {metrics_red['autocorr_lag1']:.4f}, Has Red Noise: {metrics_red['has_red_noise']}")
    print(f"Pont Excess (Bin 10): {metrics_red['pont_excess'].get(10, 1.0):.4f}")
    assert metrics_red["has_red_noise"], "AR(1) correlated noise should be flagged as red noise"

    print("\nAll tests completed successfully!")

def test_stamp_plot_generation():
    import tempfile
    import os
    import shutil

    # Create dummy data with a peak
    t_obs = np.arange(0, 100, 0.1)
    y_flux = paczynski(t_obs, 50.0, 10.0, 0.1)
    y_err = np.ones_like(y_flux) * 0.01
    y_obs = y_flux + np.random.normal(0, 0.01, size=len(y_flux))
    data = {"band1": {"t": t_obs, "y": y_obs, "y_err": y_err}}

    temp_dir = tempfile.mkdtemp()
    try:
        # Run with stamp plot output enabled
        analyze_lightcurve(data, stamp_dir=temp_dir)
        
        # Verify that stamp_peaks.png was generated
        plot_path = os.path.join(temp_dir, "stamp_peaks.png")
        assert os.path.exists(plot_path), "Stamp plot was not created"
        assert os.path.getsize(plot_path) > 0, "Stamp plot is empty"
        print(f"✅ Stamp plot test passed! Saved successfully in: {plot_path}")
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def test_bjd_scaling_recovery():
    np.random.seed(42)
    # Simulate a microlensing event
    t_obs = np.arange(0, 100, 0.1)
    y_flux = paczynski(t_obs, 50.0, 10.0, 0.1)
    y_err = np.ones_like(y_flux) * 0.01
    y_obs = y_flux + np.random.normal(0, 0.01, size=len(y_flux))

    # Base case (no offset)
    data_base = {"band1": {"t": t_obs.copy(), "y": y_obs.copy(), "y_err": y_err.copy()}}
    res_base = analyze_lightcurve(data_base)
    assert len(res_base["anomalies"]) > 0
    t0_base = res_base["anomalies"][0]["t0"]
    snr_base = res_base["anomalies"][0]["snr"]

    # Massive offset case (e.g. BJD offset)
    offset = 2459000.0
    t_obs_offset = t_obs + offset
    data_offset = {"band1": {"t": t_obs_offset, "y": y_obs.copy(), "y_err": y_err.copy()}}
    res_offset = analyze_lightcurve(data_offset)

    assert len(res_offset["anomalies"]) > 0
    t0_offset = res_offset["anomalies"][0]["t0"]
    snr_offset = res_offset["anomalies"][0]["snr"]

    print("\nBJD Scaling Recovery:")
    print(f"Base t0: {t0_base:.6f}, Offset t0: {t0_offset:.6f} (Expected close to {t0_base + offset:.6f})")
    print(f"Base SNR: {snr_base:.4f}, Offset SNR: {snr_offset:.4f}")

    # Assert unscaled coordinates are recovered perfectly
    assert abs(t0_offset - (t0_base + offset)) < 1e-8, "Precision loss or incorrect offset recovery in peak t0"
    assert np.isclose(snr_offset, snr_base, rtol=1e-3), f"Numerical instability affected peak SNR: {snr_offset} vs {snr_base}"
    print("✅ BJD Scaling Recovery test passed!")


if __name__ == "__main__":
    test_new_features()
    test_stamp_plot_generation()
    test_bjd_scaling_recovery()

