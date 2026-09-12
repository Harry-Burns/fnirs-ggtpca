import numpy as np

from src.processing import temporal_derivative, split_bands


def _tukey_biweight_weights(res: np.ndarray, sigma: float) -> np.ndarray:
    d = res / (4.685 * sigma)
    w = np.zeros_like(d, dtype=float)
    mask = np.abs(d) < 1.0
    w[mask] = (1.0 - d[mask]**2)**2
    return w

def _tddr_loop(s: np.ndarray, w: np.ndarray):
    w_sum = np.sum(w)
    if w_sum == 0: return w, np.nan

    mean = np.sum(s * w) / w_sum
    res = np.abs(s - mean)

    sigma = 1.4826 * np.median(res)
    sigma = max(sigma, 1e-12)

    w_ = _tukey_biweight_weights(res, sigma)

    return w_, mean

def _tddr_single(s: np.ndarray, tol: float, max_iter: int, verbose: bool) -> tuple[np.ndarray, np.ndarray]:
    if s.size == 0: raise ValueError("s must contain at least one sample")

    s_td = temporal_derivative(s, addzero=False)

    if s_td.size == 0:
        return s.copy(), np.zeros_like(s, dtype=float)

    w = np.ones_like(s_td, dtype=float)
    curr_mean = np.mean(s_td); new_mean = curr_mean
    
    i = 0
    for i in range(max_iter):
        w, new_mean = _tddr_loop(s_td, w)
        if np.isnan(new_mean):
            raise RuntimeError("Variable new_mean ran to np.nan. Stopping process.")
        if np.abs(new_mean - curr_mean) < tol:
            break
        curr_mean = new_mean

    if verbose: print(f"Final mean: {new_mean}\nNumber of iterations: {i}\nNumber of 0s in w: {np.sum(w == 0)}")

    # transform s using weights
    s_td_corr = w * (s_td - new_mean)
    s_corr = np.cumsum(np.insert(s_td_corr, 0, 0.0))
    s_corr = s_corr - np.mean(s_corr) + np.mean(s)
    w_out = np.insert(w, 0, 0.0)
    return s_corr, w_out


def tddr(s: np.ndarray, tol: float=1e-6, max_iter: int=50, verbose: bool=False) -> tuple[np.ndarray, np.ndarray]:
    if s.ndim == 1: 
        return _tddr_single(s, tol=tol, max_iter=max_iter, verbose=verbose)
    
    if s.ndim == 2:
        corrected = np.zeros_like(s, dtype=float)
        weights = np.zeros_like(s, dtype=float)
        for ch in range(s.shape[0]):
            s_ch, w_ch = _tddr_single(s[ch], tol=tol, max_iter=max_iter, verbose=verbose)
            corrected[ch] = s_ch
            weights[ch] = w_ch
        return corrected, weights

    raise ValueError("Signal variable s must be 1D (time,) or 2D (channels, time)")

def tddr_pipeline(s: np.ndarray, lowpass: float=0.5, trim_samples: int=60, fs: float=6.25, tol: float=1e-6, max_iter: int=50, readd_trim=False, verbose: bool=False) -> tuple[np.ndarray, np.ndarray]:
    s_low,_,s_high = split_bands(s, fs=fs, lowpass=lowpass)

    if trim_samples and s.shape[1] <= 2 * trim_samples:
        raise ValueError("trim_samples is too large, would remove entire signal.")

    s_low_trim = s_low[:,trim_samples:-trim_samples] if trim_samples else s_low
    s_high_trim = s_high[:,trim_samples:-trim_samples] if trim_samples else s_high

    s_tddr, weights = tddr(s_low_trim,tol,max_iter,verbose)

    if readd_trim:   # This leaves a spike at the end. Only use if removing the trimmed area elsewhere
        s_out = s.copy()
        s_out[:,trim_samples:-trim_samples] = s_tddr + s_high_trim
    else:
        s_out = s_tddr + s_high_trim

    return s_out, weights