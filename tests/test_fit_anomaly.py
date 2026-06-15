import numpy as np
from microwavelet import detect_anomalies_with_fit

def test_detect_anomalies_with_fit_quiet():
    # Generate a quiet PSPL event
    t = np.linspace(-20, 20, 400)
    # True parameters: t0=0.0, u0=0.3, tE=5.0
    u = np.sqrt(0.3**2 + (t / 5.0)**2)
    # Paczynski magnification A
    u_abs = np.maximum(np.abs(u), 1e-8)
    A = (u_abs**2 + 2) / (u_abs * np.sqrt(u_abs**2 + 4))
    
    fs = 0.2
    fb = 0.8
    y_true = fs * A + fb
    
    # Add small Gaussian noise
    np.random.seed(42)
    y_err = np.ones_like(t) * 0.005
    y = y_true + np.random.normal(0, 0.005, len(t))
    
    res = detect_anomalies_with_fit(t, y, y_err, threshold=25.0, k=2.0)
    
    # Check schema
    assert 'pspl_fit' in res
    assert 'anomaly' in res
    
    fit = res['pspl_fit']
    assert abs(fit['t0'] - 0.0) < 0.2
    assert abs(fit['u0'] - 0.3) < 0.1
    assert abs(fit['tE'] - 5.0) < 1.0
    assert abs(fit['fs'] - 0.2) < 0.05
    assert abs(fit['fb'] - 0.8) < 0.05
    
    anom = res['anomaly']
    assert not anom['triggered']
    assert anom['score'] < 25.0

def test_detect_anomalies_with_fit_with_anomaly():
    # Generate a PSPL event with a sharp localized anomaly (planetary spike)
    t = np.linspace(-20, 20, 400)
    # True parameters: t0=0.0, u0=0.3, tE=5.0
    u = np.sqrt(0.3**2 + (t / 5.0)**2)
    u_abs = np.maximum(np.abs(u), 1e-8)
    A = (u_abs**2 + 2) / (u_abs * np.sqrt(u_abs**2 + 4))
    
    fs = 0.2
    fb = 0.8
    y_true = fs * A + fb
    
    # Add a massive anomaly at t=5.0 (index 250)
    # Spike height = 0.1, duration = 0.5 days
    anomaly_mask = (t >= 4.75) & (t <= 5.25)
    y_true[anomaly_mask] += 0.08
    
    np.random.seed(42)
    y_err = np.ones_like(t) * 0.002
    y = y_true + np.random.normal(0, 0.002, len(t))
    
    res = detect_anomalies_with_fit(t, y, y_err, threshold=15.0, k=1.0)
    
    fit = res['pspl_fit']
    # The fitter should still find a reasonable overall fit
    assert abs(fit['t0'] - 0.0) < 1.0
    
    anom = res['anomaly']
    assert anom['triggered']
    assert anom['score'] > 15.0
    assert abs(anom['t0'] - 5.0) < 0.5
    assert anom['duration'] > 0.0

def test_detect_anomalies_dual_channel():
    t = np.linspace(-20, 20, 400)
    u = np.sqrt(0.3**2 + (t / 5.0)**2)
    u_abs = np.maximum(np.abs(u), 1e-8)
    A = (u_abs**2 + 2) / (u_abs * np.sqrt(u_abs**2 + 4))
    
    fs = 0.2
    fb = 0.8
    y_true = fs * A + fb
    
    # Add a gentle anomaly at t >= 10.0
    anomaly_mask = (t >= 10.0) & (t <= 15.0)
    y_true[anomaly_mask] += 0.003  # subtle, gentle deviation
    
    np.random.seed(42)
    y_err = np.ones_like(t) * 0.002
    y = y_true + np.random.normal(0, 0.002, len(t))
    
    res = detect_anomalies_with_fit(
        t, y, y_err, 
        threshold=25.0, k=2.0,            # Fast channel (should NOT trigger)
        threshold_slow=15.0, k_slow=0.5   # Slow channel (should trigger)
    )
    
    assert 'anomaly_fast' in res
    assert 'anomaly_slow' in res
    
    assert not res['anomaly_fast']['triggered']
    assert res['anomaly_slow']['triggered']

def test_detect_anomalies_bidirectional():
    t = np.linspace(-20, 20, 400)
    u = np.sqrt(0.3**2 + (t / 5.0)**2)
    u_abs = np.maximum(np.abs(u), 1e-8)
    A = (u_abs**2 + 2) / (u_abs * np.sqrt(u_abs**2 + 4))
    
    fs = 0.2
    fb = 0.8
    y_true = fs * A + fb
    
    # Add a massive spike at t=5.0
    anomaly_mask = (t >= 4.75) & (t <= 5.25)
    y_true[anomaly_mask] += 0.08
    
    np.random.seed(42)
    y_err = np.ones_like(t) * 0.002
    y = y_true + np.random.normal(0, 0.002, len(t))
    
    # Run in bidirectional mode
    res_bi = detect_anomalies_with_fit(t, y, y_err, threshold=12.5, k=2.0, bidirectional=True)
    assert res_bi['anomaly_fast']['triggered']
    
    # Run in standard (forward) mode
    res_std = detect_anomalies_with_fit(t, y, y_err, threshold=25.0, k=2.0, bidirectional=False)
    assert res_std['anomaly_fast']['triggered']
    
    # Check that bidirectional score is smaller than standard forward score
    assert res_bi['anomaly_fast']['score'] < res_std['anomaly_fast']['score']
    # Check that the onset of bidirectional mode is before the peak (5.0)
    assert res_bi['anomaly_fast']['onset'] <= 5.0
    # Check that the end of bidirectional mode is after the peak (5.0)
    assert res_bi['anomaly_fast']['end'] >= 5.0
