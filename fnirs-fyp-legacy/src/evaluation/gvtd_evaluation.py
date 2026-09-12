import numpy as np

from src.stim_labels import *
from src.gvtd import gvtd
def eval_motion_locked_gvtd(eval_input: dict, window_pre: float = 1.5, window_break: float = 0.5, window_post: float = 3.0, sci_threshold: float = 0.5) -> dict:
    s_od_bp = eval_input["s_od_bp"]
    df_w1 = eval_input["df_w1"]
    stim_offset = eval_input["stim_offset"]
    stim_data = eval_input["stim_data"]
    fs = eval_input["fs"]
    T = eval_input["trial_name"]

    if T == "NHM":
        return None

    if s_od_bp.ndim != 3:
        raise ValueError(f"s_od_bp must have shape (2, N, T), got {s_od_bp.shape}")

    # Reduce to the higher-SCI metrics
    channel_filter_mask = (df_w1['pair_score'] > sci_threshold).to_numpy()
    s_od_bp_filtered = s_od_bp[:,channel_filter_mask,:]

    g = gvtd(s_od_bp_filtered.reshape(-1, s_od_bp_filtered.shape[-1]))

    pre_samps = int(round(window_pre * fs))
    break_samps = int(round(window_break * fs))
    post_samps = int(round(window_post * fs))
    window_len = pre_samps + break_samps + post_samps

    t = (np.arange(window_len) - (pre_samps + break_samps)) / fs

    baseline_mask = (t >= -(window_pre + window_break)) & (t < -window_break)
    score_mask = (t >= 0.0) & (t <= window_post)

    if not np.any(baseline_mask):
        raise ValueError("Baseline mask is empty")
    if np.count_nonzero(score_mask) < 2:
        raise ValueError("Score mask must contain at least 2 samples")

    intervals = get_motion_intervals(stim_data=stim_data, stim_offset=stim_offset, T=T, window_pre=pre_samps + break_samps, window_post=post_samps, total_len=g.shape[-1])

    if len(intervals) == 0:
        return None

    windows = np.stack([g[w_start:w_end] for (w_start, w_end) in intervals], axis=0)

    # Main statistic
    event_baselines = np.median(windows[:, baseline_mask], axis=1)

    excess_windows = np.clip(windows - event_baselines[:, None], 0.0, None)
    event_scores = np.trapezoid(excess_windows[:, score_mask], x=t[score_mask], axis=1)

    # Extra stats
    event_peaks = excess_windows[:, score_mask].max(axis=1)
    avg = windows.mean(axis=0)
    std = windows.std(axis=0, ddof=1) if windows.shape[0] > 1 else np.zeros_like(avg)
    sem = std / np.sqrt(windows.shape[0])

    avg_excess = excess_windows.mean(axis=0)
    avg_excess_auc = float(np.trapezoid(avg_excess[score_mask], x=t[score_mask]))
    avg_peak_excess = float(avg_excess[score_mask].max())

    return {
        "gvtd": g,
        "t": t,
        "intervals": intervals,
        "windows": windows,                     # raw GVTD windows
        "excess_windows": excess_windows,       # baseline-subtracted, clipped >= 0
        "event_baselines": event_baselines,
        "event_scores": event_scores,           # PRIMARY per-event metric
        "event_peaks": event_peaks,
        "median_score": float(np.median(event_scores)),   # PRIMARY summary metric
        "mean_score": float(np.mean(event_scores)),
        "std_score": float(np.std(event_scores, ddof=1)) if len(event_scores) > 1 else 0.0,
        "n_events": int(len(event_scores)),
        "avg": avg,
        "std": std,
        "sem": sem,
        "avg_excess": avg_excess,
        "avg_excess_auc": avg_excess_auc,
        "avg_peak_excess": avg_peak_excess,
        "baseline_mask": baseline_mask,
        "score_mask": score_mask,
        "channel_filter_mask": channel_filter_mask,
    }


def get_motion_intervals(stim_data: dict, stim_offset: int, T: str, window_pre: int, window_post: int, total_len: int):
    intervals = []

    for i in range(NUM_BLOCKS):
        key = MOTION_START_IT.format(i=i, T=T)

        if key not in stim_data:
            raise KeyError(f"Missing stim label: {key}")

        motion_start_stim = stim_data[key]

        if len(motion_start_stim) != 1:
            raise ValueError(f"Expected exactly 1 onset for {key}, got {len(motion_start_stim)}")

        m_start = int(motion_start_stim[0]) - stim_offset
        w_start = m_start - window_pre
        w_end = m_start + window_post

        if w_end <= w_start:
            raise ValueError("Window length too small (w_end <= w_start)")

        intervals.append((w_start, w_end))

    return intervals


def gvtd_attenuation_percent(result: dict, no_correction_result: dict) -> float:
    ref = no_correction_result["median_score"]
    cur = result["median_score"]

    if np.isclose(ref, 0.0):
        return np.nan

    return 100.0 * (1.0 - (cur / ref))

