import numpy as np

from microwavelet import run_linear_cusum, run_quadratic_cusum, seed_by_flat_cusum


def test_linear_cusum():
    # Test linear CUSUM with positive residuals
    r = np.array([0.0, 0.5, 1.5, 2.0, 0.0, -1.0])
    # k = 1.0
    # S[0] = 0.0
    # S[1] = max(0, S[0] + 0.5 - 1.0) = 0.0
    # S[2] = max(0, S[1] + 1.5 - 1.0) = 0.5
    # S[3] = max(0, S[2] + 2.0 - 1.0) = 1.5
    # S[4] = max(0, S[3] + 0.0 - 1.0) = 0.5
    # S[5] = max(0, S[4] - 1.0 - 1.0) = 0.0
    S = run_linear_cusum(r, k=1.0)
    expected = np.array([0.0, 0.0, 0.5, 1.5, 0.5, 0.0])
    assert np.allclose(S, expected)


def test_quadratic_cusum():
    # Test quadratic CUSUM with residuals
    r = np.array([0.0, 2.0, 2.0, 0.0])
    # k = 1.0
    # S[0] = 0.0
    # S[1] = max(0, S[0] + 2.0**2 - 1.0 - 1.0) = max(0, 0 + 4.0 - 2.0) = 2.0
    # S[2] = max(0, S[1] + 2.0**2 - 1.0 - 1.0) = max(0, 2.0 + 4.0 - 2.0) = 4.0
    # S[3] = max(0, S[2] + 0.0 - 1.0 - 1.0) = max(0, 4.0 - 2.0) = 2.0
    S = run_quadratic_cusum(r, k=1.0)
    expected = np.array([0.0, 2.0, 4.0, 2.0])
    assert np.allclose(S, expected)


def test_seed_by_flat_cusum():
    # Generate flat data with a peak
    t = np.arange(0, 50, 1.0)
    y = np.ones_like(t) * 10.0
    # Add a clear peak at t=25 (index 25)
    y[23:28] = [12.0, 15.0, 20.0, 15.0, 12.0]
    y_err = np.ones_like(t) * 0.1

    t0, tE, triggered = seed_by_flat_cusum(t, y, y_err, method="linear", k=1.0, threshold=5.0)
    assert triggered
    assert abs(t0 - 25.0) < 1.0
    assert tE > 0.0

    # Test non-triggered case (pure noise)
    np.random.seed(42)
    y_noise = np.random.normal(10.0, 0.1, len(t))
    t0_n, tE_n, triggered_n = seed_by_flat_cusum(
        t, y_noise, y_err, method="linear", k=1.0, threshold=50.0
    )
    assert not triggered_n
