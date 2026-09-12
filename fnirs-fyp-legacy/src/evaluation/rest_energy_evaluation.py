import numpy as np

from src.stim_labels import *
from src.evaluation.correlation_evaluation import get_stim_intervals, concatenate_intervals


def eval_rest_energy(eval_input: dict, sci_threshold: float = 0.5) -> dict:
    s_mbll = eval_input["s_mbll"]           # (2, n_channels, n_samples)
    df_w1 = eval_input["df_w1"]
    stim_offset = eval_input["stim_offset"]
    stim_data = eval_input["stim_data"]
    T = eval_input["trial_name"]


    channel_filter_mask = (df_w1['pair_score'] > sci_threshold).to_numpy()
    channel_indices = np.where(channel_filter_mask)[0]
    
    channel_ids = df_w1.iloc[channel_indices]["channelIndex"].to_numpy()

    if channel_indices.size == 0:
        return None

    _, rest_intervals = get_stim_intervals(stim_data, stim_offset, T)
    rest_concat = concatenate_intervals(s_mbll, rest_intervals)  # (2, n_ch, rest_total)

    hbo_rest = rest_concat[0, channel_indices, :]  # (n_valid, rest_total)
    hbr_rest = rest_concat[1, channel_indices, :]

    hbo_std = np.std(hbo_rest, axis=1)  # (n_valid,)
    hbr_std = np.std(hbr_rest, axis=1)

    return {
        "hbo_std": hbo_std,
        "hbr_std": hbr_std,

        "hbo_summary": _summarise(hbo_std),
        "hbr_summary": _summarise(hbr_std),

        "channel_filter_mask": channel_filter_mask,
        "channel_indices": channel_indices,
        "n_valid_channels": int(channel_indices.size),
        "n_rest_periods": len(rest_intervals),
        
        "channel_ids": channel_ids,
    }


def rest_energy_change_percent(result: dict, no_correction_result: dict) -> dict:
    ref_ids = no_correction_result["channel_ids"]
    cur_ids = result["channel_ids"]

    shared, ref_idx, cur_idx = np.intersect1d(ref_ids, cur_ids, return_indices=True)

    out = {}
    for ch in ("hbo", "hbr"):
        ref = no_correction_result[f"{ch}_std"][ref_idx]
        cur = result[f"{ch}_std"][cur_idx]
        pct = 100.0 * (cur - ref) / ref
        out[f"{ch}_median_pct"] = float(np.median(pct))
        out[f"{ch}_mean_pct"] = float(np.mean(pct))
        out[f"{ch}_per_channel_pct"] = pct
    out["n_shared"] = len(shared)
    return out


def _summarise(vals: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
        "n": int(len(vals)),
    }