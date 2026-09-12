import numpy as np

from src.processing import temporal_derivative, split_bands


# "Noisy measurements were empirically defined as those with greater than 7.5% temporal standard
#   deviation in the least noisy (lowest mean GVTD) 60 seconds of each run [17], and were excluded
#   from further processing."
def gvtd_channel_filter(s: np.ndarray, gvtd: np.ndarray, fs: float=6.25, thresh=0.075, window_s=20):
    W = int(window_s * fs)

    gvtd_mean = np.convolve(gvtd, np.ones(W)/W, mode="valid")
    i0 = np.argmin(gvtd_mean)

    seg = s[:, i0:i0+W]

    sd  = np.std(seg, axis=1, ddof=1)
    return sd <= thresh

def gvtd(s: np.ndarray) -> np.ndarray:
    td = temporal_derivative(s,addzero=False)
    gvtd = np.sqrt(np.mean(td * td, axis=0))
    return gvtd

def gvtd_pipeline(s: np.ndarray, lowpass=0.5, highpass: float=0.02, trim_samples: int=60, filter: bool=True, filter_window: float=20.0, fs: float=6.25, c: float=3.0, bin_padding: float=1.0, bin_ref: str="transformed") -> tuple[np.ndarray,np.ndarray,float,list[tuple[int,int]]]:
    '''
    s: Signal matrix to perform gvtd on. Should be num_channels x num_samples
    lowpass: Frequency of low-pass filter
    highpass: Frequency of high-pass filter
    trim_samples: Number of samples to trim off after performing lowpass & highpass filter 
    filter: On = Perform additional channel-filter and recalculate gvtd
    fs: Sampling frequency of the signal
    c: GVTD threshold multiplier. higher = higher motion threshold
    bin_padding: Seconds of padding to add to each captured bin
    bin_ref: What index the bins should reference. 'original' means bin[0] is s[0], 'transformed' (or default) means bin[0] is s_gvtd[0] or s[trim_samples]
    '''

    if filter:
        filter_gvtd = gvtd(s)
        channel_mask = gvtd_channel_filter(s, filter_gvtd, fs=fs, window_s=filter_window)

        if not np.any(channel_mask): # Case for when all channels fail the filter
            return {"gvtd": np.zeros_like(filter_gvtd), "channel_mask": channel_mask, "g_thresh": 0, "threshold_bins": []}
        
        s_clean = s[channel_mask]
    else:
        channel_mask = np.ones(s.shape[0], dtype=bool)
        s_clean = s
    
    _,s_mid,_ = split_bands(s_clean,fs=fs,highpass=highpass,lowpass=lowpass)
    
    if trim_samples and s.shape[1] <= 2 * trim_samples:
        raise ValueError("trim_samples is too large, would remove entire signal.")
        
    s_trim = s_mid[:,trim_samples:-trim_samples] if trim_samples else s_mid

    s_gvtd = gvtd(s_trim)

    # thresholding
    g_threshold = gvtd_threshold(s_gvtd, c=c, bins="fd")
    threshold_mask = s_gvtd > g_threshold 
    threshold_bins = calculate_bins(threshold_mask, fs=fs, padding=bin_padding)

    if bin_ref == "original" and trim_samples:
        threshold_bins = [(s+trim_samples,e+trim_samples) for s,e in threshold_bins]

    return {"gvtd": s_gvtd, "channel_mask": channel_mask, "g_thresh": g_threshold, "threshold_bins": threshold_bins}
    

def gvtd_threshold(g: np.ndarray, c: float, bins="fd", min_samples: int=10) -> float:
    g = g[np.isfinite(g)]
    
    if g.size < min_samples:
        raise ValueError(f"Not enough GVTD samples to estimate a threshold. Expected minimum {min_samples}, recieved {g.size}")
    
    counts, edges = np.histogram(g, bins=bins)

    max_i = np.argmax(counts)
    mode = 0.5 * (edges[max_i] + edges[max_i+1])

    left_of_mode = g[g < mode]
    if left_of_mode.size == 0: sigma_L = np.std(g, ddof=0) # Rare case if bin is small & mode is the minimum.
    else: sigma_L = np.sqrt(np.mean((left_of_mode - mode) ** 2))

    gthresh = mode + c * sigma_L
    return gthresh


def calculate_bins(mask: np.ndarray, fs: float, padding: float=1) -> list[tuple[int,int]]:
    pad = int(round(padding * fs)) # 'padding' parameter is in seconds
    bins = []

    i = 0; n = len(mask)

    while i < n:
        if not mask[i]: i += 1; continue

        start = i
        while i < n and mask[i]: i += 1
        end = i

        a = max(0, start - pad)
        b = min(n, end + pad)

        if bins and a <= bins[-1][1]:
            bins[-1][1] = max(bins[-1][1], b)
        else:
            bins.append([a, b])
    return [tuple(x) for x in bins]








