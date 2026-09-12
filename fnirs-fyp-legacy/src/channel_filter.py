import numpy as np
import pandas as pd

from src.processing import split_bands

def SCI(s: np.ndarray, fs: float, lowpass: float, highpass: float):
    s_mid = np.zeros_like(s, dtype=float)

    _, s_mid[0], _ = split_bands(s[0], highpass=highpass, lowpass=lowpass, fs=fs)
    _, s_mid[1], _ = split_bands(s[1], highpass=highpass, lowpass=lowpass, fs=fs)

    wv1 = s_mid[0]
    wv2 = s_mid[1]

    std1 = np.std(wv1, axis=1, keepdims=True)
    std2 = np.std(wv2, axis=1, keepdims=True)

    zero_mask = (std1[:, 0] == 0) | (std2[:, 0] == 0)
    valid_mask = ~zero_mask

    sci = np.zeros(s.shape[1], dtype=float)

    if np.any(valid_mask):
        v1 = wv1[valid_mask]
        v2 = wv2[valid_mask]

        v1 = (v1 - v1.mean(axis=1, keepdims=True)) / std1[valid_mask]
        v2 = (v2 - v2.mean(axis=1, keepdims=True)) / std2[valid_mask]

        sci[valid_mask] = np.mean(v1 * v2, axis=1)
    return sci


def filter_distance(df: pd.DataFrame,  min: float = 25, max: float = 45, key: str = 'distance'):
    mask = (df[key] >= min) & (df[key] <= max)
    return mask.to_numpy()

def filter_nodes(df: pd.DataFrame, keep_nodes:list[str]=[]): # Use keep_nodes="None" to allow all nodes.
    if keep_nodes is None:
        return np.ones(len(df), dtype=bool)
    
    keep = set(keep_nodes)
    valid_mask = df['sourceNode'].isin(keep) & df['detectorNode'].isin(keep)
    return valid_mask.to_numpy()



def apply_filter(s: np.ndarray, df: pd.DataFrame, mask: np.ndarray):
    if len(df) != len(mask) or len(s) != len(df):
        raise ValueError("s and df must have the same number of rows.   ")

    s_out = s[mask].copy()
    df_out = df.iloc[mask].reset_index(drop=True).copy()

    return s_out, df_out



# Allows for the use of multiple baselines to evaluate the overall noise
def noisy_channels_avg(s_list: list[np.ndarray], couple_signals_fn: callable, fs: float, threshold: float, verbose=False) -> np.ndarray:
    if isinstance(s_list, np.ndarray):
        s_list = [s_list]

    N = s_list[0].shape[0]
    assert all(s.shape[0] == N for s in s_list)

    coupled_mask = couple_signals_fn()

    if not np.any(coupled_mask):
        if verbose: print("No channels left to process on after coupling. Returning empty mask.")
        return np.zeros(N, dtype=bool), np.zeros((0, len(s_list)))

    n_pairs = int(np.sum(coupled_mask))
    n_baselines = len(s_list)

    pair_scores = np.zeros((n_pairs,n_baselines))

    for i,s in enumerate(s_list):
        s_coupled, _ = couple_signals_fn(s=s)
        pair_scores[:,i] = SCI(s_coupled, fs, lowpass=1.5, highpass=0.7) # Fixed params for cardiac frequency - independent from task

    pair_score_avg = np.mean(pair_scores, axis=1)
    pair_mask = pair_score_avg >= threshold
    
    raw_mask = couple_signals_fn(pair_mask=pair_mask)

    if verbose:
        # Chatgpt generated to help analysis
        baseline_masks = pair_scores >= threshold
        pass_counts = np.sum(baseline_masks, axis=1)

        print(f"SCI mean: {pair_score_avg.mean():.4f}")
        print(f"SCI median: {np.median(pair_score_avg):.4f}")
        print(f"SCI min/max: {pair_score_avg.min():.4f}, {pair_score_avg.max():.4f}")

        print("SCI mean per baseline:", np.mean(pair_scores, axis=0))
        print("SCI median per baseline::", np.median(pair_scores, axis=0))
        print("SCI min per baseline::", np.min(pair_scores, axis=0))
        print("SCI max per baseline:", np.max(pair_scores, axis=0))
        print("Channels kept per baseline:", np.sum(baseline_masks, axis=0) * 2, "/", pair_scores.shape[0] * 2)

        print("Pairs passing exactly k baselines:")
        for k in range(n_baselines + 1):
            print(f"  k={k}: {np.sum(pass_counts == k)}")

        print("Pairs passing at least k baselines:")
        for k in range(1, n_baselines + 1):
            print(f"  >={k}: {np.sum(pass_counts >= k)}")

        print("Pairs passing >=1 baseline but rejected by avg:", np.sum((pass_counts >= 1) & (~pair_mask)))
        print("Pairs passing >=2 baselines but rejected by avg:", np.sum((pass_counts >= 2) & (~pair_mask)))
        print("Pairs passing all baselines but rejected by avg:", np.sum((pass_counts == n_baselines) & (~pair_mask)))

        print("Pairs kept by avg and also passing >=2 baselines:", np.sum(pair_mask & (pass_counts >= 2)))
        print("Pairs kept by avg but passing <2 baselines:", np.sum(pair_mask & (pass_counts < 2)))

        print("Per-pair SCI std mean/median/max:",
            np.mean(np.std(pair_scores, axis=1)),
            np.median(np.std(pair_scores, axis=1)),
            np.max(np.std(pair_scores, axis=1)))

        print("Per-pair SCI min mean/median:", np.mean(np.min(pair_scores, axis=1)), np.median(np.min(pair_scores, axis=1)))
        print("Per-pair SCI max mean/median:", np.mean(np.max(pair_scores, axis=1)), np.median(np.max(pair_scores, axis=1)))

        print("\nBaseline overlap matrix (intersection counts):")
        inter = baseline_masks.astype(int).T @ baseline_masks.astype(int)
        print(inter.astype(int))

        print("\nBaseline overlap matrix (Jaccard):")
        jacc = np.zeros((n_baselines, n_baselines), dtype=float)
        for i in range(n_baselines):
            for j in range(n_baselines):
                union = np.sum(baseline_masks[:, i] | baseline_masks[:, j])
                jacc[i, j] = np.sum(baseline_masks[:, i] & baseline_masks[:, j]) / union if union > 0 else 0.0
        print(np.round(jacc, 3))

        unstable_idx = np.where((pass_counts > 0) & (pass_counts < n_baselines))[0]
        print("\nNum unstable pairs (pass some baselines but not all):", len(unstable_idx))

        if len(unstable_idx) > 0:
            show = unstable_idx[:10]
            print("Example unstable pair indices:", show)
            print("Their SCI scores:")
            for idx in show:
                print(f"  pair {idx}: scores={np.round(pair_scores[idx], 3)}, avg={pair_score_avg[idx]:.3f}, pass_count={pass_counts[idx]}")

        rejected_despite_good = np.where((pass_counts >= 2) & (~pair_mask))[0]
        if len(rejected_despite_good) > 0:
            show = rejected_despite_good[:10]
            print("\nExample pairs passing >=2 baselines but rejected by avg:")
            for idx in show:
                print(f"  pair {idx}: scores={np.round(pair_scores[idx], 3)}, avg={pair_score_avg[idx]:.3f}, pass_count={pass_counts[idx]}")
        # ---------
    return raw_mask, pair_scores

# Use when performing the noisy channel analysis on just a single baseline (rather than averaging over a list of baselines)
def noisy_channels_start(s_bp: list[np.ndarray], couple_signals_fn: callable, fs: float, threshold: float, verbose: bool = False) -> np.ndarray:
    return noisy_channels_avg(s_list=[s_bp], couple_signals_fn=couple_signals_fn, fs=fs, threshold=threshold, verbose=verbose)
