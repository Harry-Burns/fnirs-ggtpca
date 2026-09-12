import numpy as np
from scipy.stats import pearsonr

from src.stim_labels import *


def eval_hbo_hbr_correlation(eval_input: dict, channel_index_to_od_correlation: dict, sci_threshold: float, od_correlation_threshold: float) -> dict:
    s_mbll = eval_input["s_mbll"]
    df_w1 = eval_input["df_w1"]
    stim_offset = eval_input["stim_offset"]
    stim_data = eval_input["stim_data"]
    T = eval_input["trial_name"]

    df_w1["od_corrcoef"] = df_w1["channelIndex"].map(channel_index_to_od_correlation)

    channel_filter_mask = (df_w1['pair_score'] > sci_threshold) & (df_w1['od_corrcoef'] < od_correlation_threshold)

    channel_indices = np.where(channel_filter_mask)[0]
    if channel_indices.size == 0:
        return None

    # ---- Extract intervals directly from stim markers ----
    task_intervals, rest_intervals = get_stim_intervals(stim_data, stim_offset, T)

    # ---- Build concatenated segments ----
    task_concat = concatenate_intervals(s_mbll, task_intervals)   # (2, n_ch, task_total)
    rest_concat = concatenate_intervals(s_mbll, rest_intervals)   # (2, n_ch, rest_total)

    # ---- Compute correlations on filtered channels ----
    full_corrs = _per_channel_correlation(
        s_mbll[0, channel_indices, :],
        s_mbll[1, channel_indices, :],
    )
    task_corrs = _per_channel_correlation(
        task_concat[0, channel_indices, :],
        task_concat[1, channel_indices, :],
    )
    rest_corrs = _per_channel_correlation(
        rest_concat[0, channel_indices, :],
        rest_concat[1, channel_indices, :],
    )

    return {
        "full_corrs": full_corrs,
        "task_corrs": task_corrs,
        "rest_corrs": rest_corrs,

        "full_summary": _summarise(full_corrs),
        "task_summary": _summarise(task_corrs),
        "rest_summary": _summarise(rest_corrs),

        "channel_filter_mask": channel_filter_mask,
        "n_valid_channels": sum(channel_filter_mask),

        "n_task_periods": len(task_intervals),
        "n_rest_periods": len(rest_intervals),

        "valid_channel_indexes": df_w1.loc[channel_filter_mask, "channelIndex"].to_numpy(),
    }



def get_stim_intervals(stim_data: dict, stim_offset: int, T: str):
    task_starts = []
    rest_starts = []

    for i in range(NUM_BLOCKS):
        ts = stim_data[TASK_START_IT.format(i=i, T=T)]
        rs = stim_data[REST_IT.format(i=i, T=T)]
        assert len(ts) == 1 and len(rs) == 1
        task_starts.append(int(ts[0]) - stim_offset)
        rest_starts.append(int(rs[0]) - stim_offset)

    # End of the final rest period: start of the next trial's baseline
    trial_idx = TRIALS.index(T)
    last_rest_end_key = BASELINE_PERIODS[trial_idx + 1][0]
    last_rest_end_stim = stim_data[last_rest_end_key]
    assert len(last_rest_end_stim) == 1
    last_rest_end = int(last_rest_end_stim[0]) - stim_offset

    # Task periods: Task_Start_i -> Rest_i
    task_intervals = [(task_starts[i], rest_starts[i])for i in range(NUM_BLOCKS)]

    # Rest periods
    rest_intervals = [(rest_starts[i], task_starts[i + 1])for i in range(NUM_BLOCKS - 1)]
    rest_intervals.append((rest_starts[NUM_BLOCKS - 1], last_rest_end))

    return task_intervals, rest_intervals


def _per_channel_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_z = x - x.mean(axis=1, keepdims=True)
    y_z = y - y.mean(axis=1, keepdims=True)
    num = (x_z * y_z).sum(axis=1)
    den = np.sqrt((x_z**2).sum(axis=1) * (y_z**2).sum(axis=1))
    return num / den


def concatenate_intervals(s: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    segments = [s[:, :, st:e] for st, e in intervals]
    return np.concatenate(segments, axis=2)


def _summarise(corrs: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(corrs)),
        "median": float(np.median(corrs)),
        "std": float(np.std(corrs, ddof=1)) if len(corrs) > 1 else 0.0,
        "q25": float(np.percentile(corrs, 25)),
        "q75": float(np.percentile(corrs, 75)),
        "frac_negative": float(np.mean(corrs < 0)),
        "n": int(len(corrs)),
    }