import numpy as np

# Default Params
MIN_SAMPLES = 6
MIN_CHANNELS = 8
VAR_TARGET = 0.9
MAX_PCS = None
ALIGN_WIN = 4
TAPER = 2
# ------------


def tpca(s: np.ndarray, bins: list[tuple[int,int]], min_samples=MIN_SAMPLES, var_target=VAR_TARGET, max_pcs=MAX_PCS, align_win=ALIGN_WIN, taper=TAPER, verbose=False) -> np.ndarray:   
    num_ch, num_t = s.shape

    s_out = np.copy(s)

    if num_ch < MIN_CHANNELS:
        if verbose:
            print(f"tPCA didn't have enough channels to run on. Recieved {num_ch} expected minimum {MIN_CHANNELS}")
        return s_out

    bins = [
        (max(0,st),min(num_t,end)) 
        for st,end in bins 
        if (end > 0 and st < num_t) and (end-st >= min_samples)
    ]

    if not bins:
        return s_out

    seg_lengths = [end-st for st,end in bins]
    seg_concat = np.concatenate([s_out[:,st:end] for st,end in bins], axis=1)
    seg_concat_pca = pca_removal(seg_concat, var_target=var_target, max_pcs=max_pcs)


    for i,(b_start,b_end) in enumerate(bins):
        b_len = seg_lengths[i]
        b_concat_offset = sum(seg_lengths[:i])

        seg = seg_concat[:,b_concat_offset:b_concat_offset+b_len]
        seg_pca = seg_concat_pca[:,b_concat_offset:b_concat_offset+b_len]

        # --- Bin Alignment

        # Align to average of 'align_win' samples before and after the bin.
        win_left_slice = s_out[:,max(0, b_start - align_win):b_start]
        win_right_slice = s_out[:,b_end:min(num_t, b_end + align_win)]

        win_left_mean = win_left_slice.mean(axis=1) if win_left_slice.shape[1] > 0 else seg[:, 0]
        win_right_mean = win_right_slice.mean(axis=1) if win_right_slice.shape[1] > 0 else seg[:, -1]
        
        seg_mean_win = max(1, min(align_win, b_len // 2))
        seg_left_mean = seg_pca[:,:seg_mean_win].mean(axis=1)
        seg_right_mean = seg_pca[:,-seg_mean_win:].mean(axis=1)

        d0 = (win_left_mean - seg_left_mean)[:, None]
        d1 = (win_right_mean - seg_right_mean)[:, None]
        r = np.linspace(0.0, 1.0, b_len, dtype=float)[None, :]
        
        # Add a linear ramp so the corrected bin matches surrounding data
        seg_pca = seg_pca + (1.0 - r) * d0 + r * d1


        # Replace bin in s_out
        s_out[:,b_start:b_end] = seg_pca


        # Small edge taper
        taper_n = min(int(taper), b_len // 2) # As we taper on each side of the bin
        if taper_n > 0:
            a = (np.arange(1, taper_n + 1, dtype=float) / (taper_n + 1.0))[None, :] 
            s_out[:, b_start:b_start+taper_n] = win_left_mean[:, None] * (1.0 - a) + s_out[:, b_start:b_start+taper_n] * a

            a2 = a[:, ::-1]
            s_out[:, b_end-taper_n:b_end] = win_right_mean[:, None] * (1.0 - a2) + s_out[:, b_end-taper_n:b_end] * a2
    
    return s_out

def pca_removal(s: np.ndarray, var_target: float, max_pcs: int) -> np.ndarray:
    # s: (N, T) : N channels, T Samples

    # 1. Centre data
    mean = np.mean(s, axis=1)[:,None]
    A = s - mean # (N, T)
    
    # 2. Apply SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False) 
    # A = U @ diag(S) @ Vt : (N, T) = (N, P) @ (P, P) @ (P, T) : P = min(N,T)

    # 3. Get explained variance (singular values ^ 2)
    var = S**2
    var_sum = var.sum()

    if var_sum == 0: return s

    frac = var / var_sum # fraction of variance explained per component, frac[0] is largest component
    cfrac = np.cumsum(frac) # cfrac[i] = variance explained by the first i+1 components

    # 4. Select number of principal components to remove
    if max_pcs is None:
        max_k = U.shape[1] - 1
    else:
        max_k = min(int(max_pcs), U.shape[1] - 1)
    
    k = int(np.searchsorted(cfrac, var_target) + 1)
    k = min(k, max_k)

    if k <= 0: return s

    # 5. Reconstruct signal from first k components
    A_art = (U[:, :k] * S[:k]) @ Vt[:k, :]

    # 6. Remove reconstruction from original signal, re-add mean.
    return (A - A_art) + mean

