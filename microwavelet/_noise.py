import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy.ndimage import uniform_filter1d

def _psd_model(f, A, beta, C):
    """Power law model with a white noise floor."""
    return A * (f**(-beta)) + C

def estimate_spectral_index(t, y, min_freq=None, max_freq=None):
    """
    Estimates the spectral index beta from the Lomb-Scargle periodogram.
    Models P(f) = A * f^-beta + C.

    Parameters
    ----------
    t : np.ndarray
        Observed times.
    y : np.ndarray
        Observed flux or magnitude residuals.
    min_freq : float, optional
        Minimum frequency to include in the fit.
    max_freq : float, optional
        Maximum frequency to include in the fit.

    Returns
    -------
    dict
        Dictionary containing:
        - 'beta': Estimated spectral index.
        - 'beta_err': Uncertainty in beta.
        - 'A': Amplitude parameter.
        - 'C': White noise floor.
        - 'r_squared': Goodness of fit.
        - 'frequency': The frequencies used in the fit.
        - 'power': The power at those frequencies.
        - 'error': Error message if the fit fails.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    
    ls = LombScargle(t, y)
    frequency, power = ls.autopower()

    # Filter frequencies
    mask = (frequency > 0)
    if min_freq is not None:
        mask &= (frequency >= min_freq)
    if max_freq is not None:
        mask &= (frequency <= max_freq)
    mask &= (power > 0)

    f_fit = frequency[mask]
    p_fit = power[mask]

    if len(f_fit) < 4:
        return {"error": "Not enough frequency points for fitting."}

    # Initial guesses: A=max, beta=1, C=min
    p0 = [np.max(p_fit), 1.0, np.min(p_fit)]
    
    try:
        popt, pcov = curve_fit(_psd_model, f_fit, p_fit, p0=p0)
        A, beta, C = popt
        perr = np.sqrt(np.diag(pcov))
        beta_err = perr[1]
        
        # R-squared
        residuals = p_fit - _psd_model(f_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((p_fit - np.mean(p_fit))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            "beta": float(beta),
            "beta_err": float(beta_err),
            "A": float(A),
            "C": float(C),
            "r_squared": float(r_squared),
            "frequency": frequency[mask],
            "power": power[mask],
        }
    except Exception as e:
        return {"error": str(e)}

class WaveletCoherenceAnalyzer:
    """
    Analyzes coherence between two time-series using Continuous Wavelet Transform (CWT).
    Designed for unevenly sampled data via interpolation.
    """
    def __init__(self, widths=None, wavelet_type='ricker'):
        """
        Parameters
        ----------
        widths : np.ndarray, optional
            The scales (widths) to use for the CWT.
        wavelet_type : str
            'ricker' (Mexican Hat) or 'gaussian'.
        """
        self.widths = widths
        self.wavelet_type = wavelet_type

    def _get_kernel(self, scale):
        """Generates a normalized wavelet kernel for a given scale."""
        # Create a local time grid for the kernel
        # We use a width of 5*scale to ensure the kernel decays to zero
        tk = np.linspace(-5 * scale, 5 * scale, int(10 * scale) + 1)
        if len(tk) < 5: tk = np.linspace(-5, 5, 10)
        
        if self.wavelet_type == 'ricker':
            # Mexican Hat: (1 - x^2) * exp(-x^2 / 2)
            # We use x = tk / scale
            x = tk / scale
            kernel = (1.0 - x**2) * np.exp(-0.5 * x**2)
        elif self.wavelet_type == 'gaussian':
            x = tk / scale
            kernel = np.exp(-0.5 * x**2)
        else:
            raise ValueError(f"Unknown wavelet type: {self.wavelet_type}")
            
        # L1 Normalization
        kernel /= (np.sum(np.abs(kernel)) + 1e-12)
        return tk, kernel

    def compute_cwt(self, t_grid, y):
        """Computes CWT using fftconvolve for speed."""
        n_time = len(t_grid)
        n_scales = len(self.widths)
        cwt_map = np.zeros((n_scales, n_time))
        
        for i, scale in enumerate(self.widths):
            tk, kernel = self._get_kernel(scale)
            # Use fftconvolve for speed. 
            # Mode='same' ensures the output matches t_grid length.
            cwt_map[i, :] = fftconvolve(y, kernel, mode='same')
            
        return cwt_map

    def compute_coherence(self, t_grid, y1, y2):
        """
        Computes the wavelet coherence between two signals on a regular grid.
        
        Returns
        -------
        coherence : 2D array (scales x time)
        cross_wavelet : 2D array (scales x time)
        """
        # 1. Compute CWT for both signals
        cwt1 = self.compute_cwt(t_grid, y1)
        cwt2 = self.compute_cwt(t_grid, y2)
        
        # 2. Compute Cross-Wavelet Transform (XWT)
        # XWT(s, t) = W1(s, t) * conj(W2(s, t))
        xwt = cwt1 * np.conj(cwt2)
        
        # 3. Compute Coherence
        # R^2 = |S(s^-1 * XWT)|^2 / (S(s^-1 * |W1|^2) * S(s^-1 * |W2|^2))
        # We use a moving average for smoothing in both time and scale.
        
        # Magnitude squared
        abs_xwt_sq = np.abs(xwt)**2
        abs_cwt1_sq = np.abs(cwt1)**2
        abs_cwt2_sq = np.abs(cwt2)**2
        
        # Smoothing in time (axis 1) then scale (axis 0)
        def smooth_both(arr):
            res = uniform_filter1d(arr, size=5, axis=1)
            res = uniform_filter1d(res, size=3, axis=0)
            return res

        S_xwt = smooth_both(abs_xwt_sq)
        S_cwt1 = smooth_both(abs_cwt1_sq)
        S_cwt2 = smooth_both(abs_cwt2_sq)
        
        coherence = S_xwt / (S_cwt1 * S_cwt2 + 1e-12)
        coherence = np.clip(coherence, 0, 1)
        
        return coherence, xwt

def characterize_noise(t, y, y_err=None, bin_sizes=None):
    """
    Robustly characterises white and red (correlated) noise in a light curve.
    (Existing implementation preserved)
    """
    t = np.asarray(t)
    y = np.asarray(y)

    # Sort by time to ensure physical lag calculations
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    y = y[sort_idx]

    # 1. White Noise Estimation (Robust against long-term physical trends / events)
    # Standard deviation of first differences divided by sqrt(2)
    dy = np.diff(y)
    med_dy = np.median(dy)
    mad_dy = np.median(np.abs(dy - med_dy)) / 0.6745
    sigma_white = float(mad_dy / np.sqrt(2.0))

    # 2. Total Variance
    med_y = np.median(y)
    mad_y = np.median(np.abs(y - med_y)) / 0.6745
    sigma_total = float(mad_y)

    # 3. Autocorrelation Function (ACF) at lag 1 and 2
    n = len(y)
    autocorr_lag1 = 0.0
    autocorr_lag2 = 0.0

    if n > 5:
        y_centered = y - np.mean(y)
        var = np.var(y)
        if var > 1e-12:
            autocorr_lag1 = float(np.sum(y_centered[:-1] * y_centered[1:]) / ((n - 1) * var))
            autocorr_lag2 = float(np.sum(y_centered[:-2] * y_centered[2:]) / ((n - 2) * var))

    # 4. Pont-Style Binning Scaling Factor (Red Noise Excess)
    if bin_sizes is None:
        bin_sizes = [5, 10, 20, 50]

    pont_excess = {}
    for M in bin_sizes:
        if n >= 2 * M:
            # Reshape into non-overlapping bins of size M
            n_bins = n // M
            binned_y = np.mean(y[: n_bins * M].reshape(n_bins, M), axis=1)

            # Robust scatter of binned residuals
            med_b = np.median(binned_y)
            mad_b = np.median(np.abs(binned_y - med_b)) / 0.6745

            # Expected scatter if pure white noise
            expected_scatter = sigma_white / np.sqrt(M)

            # Pont excess ratio: actual binned scatter / expected white scatter
            ratio = float(mad_b / (expected_scatter + 1e-12))
            pont_excess[M] = ratio
        else:
            pont_excess[M] = 1.0

    # Classify whether significant red noise exists:
    # 1. ACF at lag 1 is high (> 0.2)
    # 2. Pont excess factor at block size 10 is > 1.3
    has_red_noise = bool(autocorr_lag1 > 0.2 or (10 in pont_excess and pont_excess[10] > 1.3))

    return {
        "sigma_white": float(sigma_white),
        "sigma_total": float(sigma_total),
        "autocorr_lag1": float(autocorr_lag1),
        "autocorr_lag2": float(autocorr_lag2),
        "pont_excess": pont_excess,
        "has_red_noise": has_red_noise,
    }

def characterize_multiband_noise(bands_data, widths=None):
    """
    Analyzes noise correlations across multiple photometric bands.

    Parameters
    ----------
    bands_data : dict
        A dictionary where keys are band names (e.g., 'W146', 'W184') 
        and values are dicts containing:
            't': np.ndarray (times)
            'y': np.ndarray (residuals)
            'y_err': np.ndarray (errors, optional)
    widths : np.ndarray, optional
        Scales for wavelet analysis.
    """
    from scipy.interpolate import interp1d
    
    band_names = list(bands_data.keys())
    n_bands = len(band_names)
    
    # 1. Find common time grid
    all_t = []
    for name in band_names:
        all_t.append(bands_data[name]['t'])
    
    t_min = max(np.min(t) for t in all_t)
    t_max = min(np.max(t) for t in all_t)
    
    # Use a dense grid for wavelet analysis
    # We'll use the median cadence of all bands as a guide
    cadences = []
    for name in band_names:
        t_sorted = np.sort(bands_data[name]['t'])
        cadences.append(np.median(np.diff(t_sorted)))
    avg_cadence = np.mean(cadences)
    
    t_grid = np.arange(t_min, t_max, avg_cadence)
    
    # 2. Interpolate all bands to the common grid
    interpolated_y = {}
    for name in band_names:
        t = bands_data[name]['t']
        y = bands_data[name]['y']
        f = interp1d(t, y, bounds_error=False, fill_value="extrapolate")
        interpolated_y[name] = f(t_grid)
        
    # 3. Individual Spectral Analysis
    individual_metrics = {}
    for name in band_names:
        individual_metrics[name] = estimate_spectral_index(bands_data[name]['t'], bands_data[name]['y'])
        
    # 4. Wavelet Coherence Analysis
    if widths is None:
        widths = np.geomspace(1.0, 50.0, 30)
        
    analyzer = WaveletCoherenceAnalyzer(widths=widths)
    
    coherence_matrix = {}
    coherence_maps = {}
    
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            name_i = band_names[i]
            name_j = band_names[j]
            
            coh_map, _ = analyzer.compute_coherence(t_grid, interpolated_y[name_i], interpolated_y[name_j])
            
            # Integrated coherence (mean over all scales and time)
            integrated_coh = np.mean(coh_map)
            
            pair_key = f"{name_i}_{name_j}"
            coherence_matrix[pair_key] = float(integrated_coh)
            coherence_maps[pair_key] = coh_map
            
    return {
        "individual_metrics": individual_metrics,
        "coherence_matrix": coherence_matrix,
        "coherence_maps": coherence_maps,
        "t_grid": t_grid
    }
