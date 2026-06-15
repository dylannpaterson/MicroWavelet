import numpy as np
from scipy.optimize import minimize

from ._cusum import find_anomalies_cusum, seed_by_flat_cusum


def get_paczynski(u):
    """
    Evaluates standard point-source point-lens magnification.
    """
    u = np.abs(u)
    u = np.maximum(u, 1e-8)
    return (u**2 + 2) / (u * np.sqrt(u**2 + 4))

def pspl_linear_fit(t, y, y_err, t0, u0, tE):
    """
    Analytical weighted least squares solver for source and baseline fluxes.
    """
    u = np.sqrt(u0**2 + ((t - t0) / tE)**2)
    A = get_paczynski(u)
    X = np.column_stack((A, np.ones_like(A)))
    w = 1.0 / (y_err**2)
    XTW = X.T * w
    XTWX = XTW @ X
    XTWy = XTW @ y
    try:
        coeffs = np.linalg.solve(XTWX, XTWy)
        fs, fb = coeffs
        if fs <= 0:
            return np.inf, 0.0, 0.0
        y_model = fs * A + fb
        chi2 = np.sum(((y - y_model) / y_err)**2)
        return chi2, fs, fb
    except np.linalg.LinAlgError:
        return np.inf, 0.0, 0.0

def _objective(params, t, y, y_err):
    t0, log_u0, log_tE = params
    u0 = 10**log_u0
    tE = 10**log_tE
    chi2, fs, fb = pspl_linear_fit(t, y, y_err, t0, u0, tE)
    return chi2

def detect_anomalies_with_fit(t, y, y_err, threshold=12.5, k=2.0, threshold_slow=17.5, k_slow=0.5, bidirectional=True):
    """
    Unified microlensing pipeline: performs CUSUM-seeded multi-start PSPL fitting,
    calculates residuals, and runs the CUSUM change-point anomaly detector on them.

    Parameters
    ----------
    t : np.ndarray
        Time array (BJD). Must be sorted in ascending order.
    y : np.ndarray
        Flux array.
    y_err : np.ndarray
        Observation errors.
    threshold : float, default 25.0
        Significance threshold (H) for CUSUM anomaly detection.
    k : float, default 2.0
        CUSUM slack parameter.
    threshold_slow : float, default 35.0
        Significance threshold (H) for slow CUSUM anomaly detection.
    k_slow : float, default 0.5
        CUSUM slack parameter for slow channel.

    Returns
    -------
    dict
        Nested dictionary of fitting results and anomaly properties:
        {
            'pspl_fit': {
                't0': float,
                'u0': float,
                'tE': float,
                'fs': float,
                'fb': float,
                'chi2': float,
                'dchi2_excess': float
            },
            'anomaly': {
                'triggered': bool,
                'score': float,
                't0': float,
                'onset': float,
                'end': float,
                'duration': float,
                'residuals_std': float,
                'cusum_statistic': np.ndarray
            }
        }
    """
    # 1. Seed parameters using CUSUM on the robust median baseline flat-fit
    # Combine both fast (k=2.0) and slow (k=0.5) linear CUSUM configurations to find seed candidates
    seeds_fast = seed_by_flat_cusum(t, y, y_err, method='linear', k=2.0, threshold=10.0, return_all=True)
    seeds_slow = seed_by_flat_cusum(t, y, y_err, method='linear', k=0.5, threshold=10.0, return_all=True)
    
    # Merge seeds and remove duplicates within 5 days
    all_seeds = []
    for t0_s, tE_s in seeds_fast + seeds_slow:
        if not any(abs(t0_s - existing_t0) < 5.0 for existing_t0, _ in all_seeds):
            all_seeds.append((t0_s, tE_s))
            
    if len(all_seeds) == 0:
        peak_idx = np.argmax(y)
        all_seeds = [(float(t[peak_idx]), 20.0)]
        
    seeds = all_seeds
    
    best_overall_chi2 = np.inf
    best_overall_fit = None
    
    for t0_seed, tE_seed in seeds:
        # Perform 1D search over u0 at this seed's t0, tE
        u0_grid = np.logspace(-2, 0, 10)
        best_seed_chi2 = np.inf
        best_u0 = 0.1
        for u0 in u0_grid:
            chi2, fs, fb = pspl_linear_fit(t, y, y_err, t0_seed, u0, tE_seed)
            if chi2 < best_seed_chi2:
                best_seed_chi2, best_u0 = chi2, u0
                
        init_params = [t0_seed, np.log10(best_u0), np.log10(tE_seed)]
        
        # Restrict t0 to remain close to the CUSUM trigger peak time, preventing it from drifting to adjacent seasons/gaps.
        half_width = max(2.5 * tE_seed, 15.0)
        bounds = [
            (t0_seed - half_width, t0_seed + half_width),
            (-3.0, 1.0),
            (0.0, 3.0)
        ]
        
        res = minimize(_objective, init_params, args=(t, y, y_err), method='Nelder-Mead', bounds=bounds, tol=1e-3)
        fit_t0, fit_log_u0, fit_log_tE = res.x
        fit_u0 = 10**fit_log_u0
        fit_tE = 10**fit_log_tE
        final_chi2, fs, fb = pspl_linear_fit(t, y, y_err, fit_t0, fit_u0, fit_tE)
        
        if final_chi2 < best_overall_chi2:
            best_overall_chi2 = final_chi2
            best_overall_fit = (fit_t0, fit_u0, fit_tE, fs, fb)
            
    if best_overall_fit is None:
        # Fallback if no fit completed
        fit_t0 = t[np.argmax(y)]
        fit_u0 = 0.1
        fit_tE = 20.0
        fs = 0.0
        fb = np.median(y)
        best_overall_chi2 = np.sum(((y - fb) / y_err)**2)
    else:
        fit_t0, fit_u0, fit_tE, fs, fb = best_overall_fit
        
    # 2. Compute residuals
    u_pts = np.sqrt(fit_u0**2 + ((t - fit_t0) / fit_tE)**2)
    y_model_pts = fs * get_paczynski(u_pts) + fb
    residuals_sigma = (y - y_model_pts) / y_err
    
    # 3. Detect anomalies in residuals (both fast/narrow and slow/gentle configurations)
    anom_fast = find_anomalies_cusum(t, residuals_sigma, threshold=threshold, k=k, bidirectional=bidirectional)
    anom_slow = find_anomalies_cusum(t, residuals_sigma, threshold=threshold_slow, k=k_slow, bidirectional=bidirectional)
    
    # 4. Format outputs
    N = len(t)
    dchi2_excess = best_overall_chi2 - (N - 3)
    
    master_triggered = bool(anom_fast['triggered'] or anom_slow['triggered'])
    if anom_fast['triggered']:
        master_anom = anom_fast
    elif anom_slow['triggered']:
        master_anom = anom_slow
    else:
        master_anom = anom_fast
        
    return {
        'pspl_fit': {
            't0': float(fit_t0),
            'u0': float(fit_u0),
            'tE': float(fit_tE),
            'fs': float(fs),
            'fb': float(fb),
            'chi2': float(best_overall_chi2),
            'dchi2_excess': float(dchi2_excess)
        },
        'anomaly': {
            'triggered': master_triggered,
            'score': float(master_anom['score']),
            't0': float(master_anom['t0']) if master_triggered else float(fit_t0),
            'onset': float(master_anom['onset']) if master_triggered else float(fit_t0),
            'end': float(master_anom['end']) if master_triggered else float(fit_t0),
            'duration': float(master_anom['duration']),
            'residuals_std': float(master_anom['residuals_std']),
            'cusum_statistic': master_anom['cusum_statistic']
        },
        'anomaly_fast': {
            'triggered': bool(anom_fast['triggered']),
            'score': float(anom_fast['score']),
            't0': float(anom_fast['t0']) if anom_fast['triggered'] else float(fit_t0),
            'onset': float(anom_fast['onset']) if anom_fast['triggered'] else float(fit_t0),
            'end': float(anom_fast['end']) if anom_fast['triggered'] else float(fit_t0),
            'duration': float(anom_fast['duration']),
            'residuals_std': float(anom_fast['residuals_std']),
            'cusum_statistic': anom_fast['cusum_statistic']
        },
        'anomaly_slow': {
            'triggered': bool(anom_slow['triggered']),
            'score': float(anom_slow['score']),
            't0': float(anom_slow['t0']) if anom_slow['triggered'] else float(fit_t0),
            'onset': float(anom_slow['onset']) if anom_slow['triggered'] else float(fit_t0),
            'end': float(anom_slow['end']) if anom_slow['triggered'] else float(fit_t0),
            'duration': float(anom_slow['duration']),
            'residuals_std': float(anom_slow['residuals_std']),
            'cusum_statistic': anom_slow['cusum_statistic']
        }
    }
