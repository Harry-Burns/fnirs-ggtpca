import numpy as np

from src.stim_labels import *

def eval_block_avg(eval_input: dict, rest_window: float, task_window: float, break_delay: float):
    s = eval_input['s_mbll']
    stim_offset = eval_input['stim_offset'] # offset from stim_data 0 reference point to s_mbll[0]
    T = eval_input["trial_name"]
    stim_data = eval_input["stim_data"]
    fs = eval_input["fs"]

    rest_window_samp = int(rest_window * fs)
    task_window_samp = int(task_window * fs)
    break_delay_samp = int(break_delay * fs)

    rest_intervals, task_intervals = get_intervals(stim_data, stim_offset, T, rest_window_samp, task_window_samp, break_delay_samp)

    out = block_average(s, rest_intervals, task_intervals)

    return out

def block_average(s: np.ndarray, rest_intervals: list[tuple[int,int]], task_intervals: list[tuple[int,int]]):
    # s is shape (2,n_channels,n_samples)

    rest_lengths = [e-st for st,e in rest_intervals]
    task_lengths = [e-st for st,e in task_intervals]

    assert max(rest_lengths) == min(rest_lengths)
    assert max(task_lengths) == min(task_lengths)
    assert len(rest_intervals) == len(task_intervals)   

    n_channels = s.shape[1];  n_blocks = len(rest_intervals); 

    # rest_blocks is shape (2,n_channels,n_blocks,n_samples)
    rest_blocks = np.zeros(shape=(s.shape[0],n_channels,n_blocks,rest_lengths[0],))
    task_blocks = np.zeros(shape=(s.shape[0],n_channels,n_blocks,task_lengths[0],))

    for i,(st,e) in enumerate(rest_intervals):
        rest_blocks[:,:,i,:] = s[:,:,st:e]
    for i,(st,e) in enumerate(task_intervals):
        task_blocks[:,:,i,:] = s[:,:,st:e]

    baseline = np.mean(rest_blocks, axis=3, keepdims=True)
    rest_blocks_corrected = rest_blocks - baseline   # rest will now end near 0
    task_blocks_corrected = task_blocks - baseline   # task starts from same 0 reference

    def block_stats(blocks): #(2,n_channels,n_blocks,n_samples)
        n_blocks = blocks.shape[2]
        avg = blocks.mean(axis=2)
        std = blocks.std(axis=2, ddof=1)
        sem = std / np.sqrt(n_blocks)
        mx  = blocks.max(axis=2)
        mn  = blocks.min(axis=2)
        return avg, std, sem, mx, mn

    rest_avg, rest_std, rest_sem, rest_max, rest_min = block_stats(rest_blocks_corrected)
    task_avg, task_std, task_sem, task_max, task_min = block_stats(task_blocks_corrected)


    return {
        "rest": {
            "blocks_raw":       rest_blocks,
            "blocks_corrected": rest_blocks_corrected,
            "avg":              rest_avg,
            "sem":              rest_sem,
            "std":              rest_std,
            "max":              rest_max,
            "min":              rest_min,
            "n_blocks":         n_blocks,
        },
        "task": {
            "blocks_raw":       task_blocks,
            "blocks_corrected": task_blocks_corrected,
            "avg":              task_avg,
            "sem":              task_sem,
            "std":              task_std,
            "max":              task_max,
            "min":              task_min,
            "n_blocks":         n_blocks,
        },
        "baseline": baseline,  # (2, n_ch, n_blocks, 1)
        "n_blocks": n_blocks
    }

def get_intervals(stim_data: dict, stim_offset: int, T: str, rest_window: int, task_window: int, break_delay: int):
    # windows should be passed in numbers of samples, not secondss
    rest_intervals = []; task_intervals = []

    for i in range(NUM_BLOCKS):
        rest_stim = stim_data[REST_IT.format(i=i,T=T)]
        task_stim = stim_data[TASK_START_IT.format(i=i,T=T)]
        
        assert len(rest_stim) == 1
        assert len(task_stim) == 1

        rest_start = rest_stim[0] - stim_offset
        task_start = task_stim[0] - stim_offset

        rest_int = (task_start - rest_window, task_start)
        task_int = (task_start + break_delay, task_start + break_delay + task_window)

        assert rest_int[0] >= 0

        rest_intervals.append(rest_int)
        task_intervals.append(task_int)

    return rest_intervals, task_intervals

def channel_subset_block_average(block_avg_out: dict, channel_indices: list[int]) -> dict:
    rest_blocks = block_avg_out["rest"]["blocks_corrected"]  # (2, n_ch, n_blocks, rest_window)
    task_blocks = block_avg_out["task"]["blocks_corrected"]  # (2, n_ch, n_blocks, task_window)

    rest_ch = rest_blocks[:, channel_indices, :, :].mean(axis=1)  # (2, n_blocks, rest_window)
    task_ch = task_blocks[:, channel_indices, :, :].mean(axis=1)  # (2, n_blocks, task_window)

    n_blocks = task_ch.shape[1]

    def block_stats(blocks): # (2,n_blocks,n_samples)
        avg = blocks.mean(axis=1)
        std = blocks.std(axis=1, ddof=1)
        sem = std / np.sqrt(n_blocks)
        mx  = blocks.max(axis=1)
        mn  = blocks.min(axis=1)
        return {"blocks": blocks, "avg": avg, "sem": sem, "std": std, "max": mx, "min": mn}

    return {
        "rest": block_stats(rest_ch),
        "task": block_stats(task_ch),
        "n_blocks": n_blocks
    }
     




























