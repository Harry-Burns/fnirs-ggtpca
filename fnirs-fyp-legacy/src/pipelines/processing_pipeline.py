import numpy as np
import pandas as pd

from src.stim_labels import *

from src.snirf_processing import generate_variables, couple_signals_fn_builder
from src.processing import raw_to_od, od_to_mbll, split_bands
from src.global_regression import global_regression_mbll, ss_regression_mbll

from src.channel_filter import noisy_channels_start, filter_distance, filter_nodes, apply_filter

from src.pipelines.correction_pipelines import CorrectionPipelineOut, CorrectionPipelineIn
from src.pipelines.correction_pipelines import run_correction_pipelines


def process_recording(file: str, params: dict, correction_pipelines: list, verbose_stages=True, verbose=False) -> dict:
    if verbose_stages or verbose:
        print(f"Starting processing pipeline...")

    # ----------------- INIT
    snirf_data = generate_variables(file_path=file)

    s_raw = snirf_data['s_raw']
    df_m = snirf_data['df_m']
    stim_data = snirf_data['stim_data']
    fs = snirf_data['fs']
    wavelengths = snirf_data['wavelengths']

    bp_stim = BASELINE_PERIODS[0]
    bp = (stim_data[bp_stim[0]][0], stim_data[bp_stim[1]][0]) # Baseline period interval
    
    if verbose_stages or verbose:
        print("Loaded in variables.")
    # -----------------



    # ----------------- CHANNEL FILTERING
    if verbose_stages or verbose:
        print("\nPruning unused channels...")

    mask_node = filter_nodes(df=df_m, keep_nodes=params['keep_nodes'])
    mask_dist = filter_distance(df=df_m, min=params['min_dist'], max=params['max_dist'])
    mask = (mask_node & mask_dist)

    s_filtered, df_filtered = apply_filter(s=s_raw, df=df_m, mask=mask)

    snirf_data['s_raw_filtered'] = s_filtered
    snirf_data['df_filtered'] = df_filtered

    if len(s_filtered) == 0:
        if verbose_stages or verbose:
            print("No channels remaining after pruning. Exiting evaluation...")
        return None, None, None
    
    if verbose:
        print(f"{len(s_filtered)}/{len(s_raw)} Channels Remaining")
    # -----------------



    # ----------------- OD CONVERSION
    if verbose_stages or verbose:
        print("\nConverting to OD signal...")

    od_bp = bp if params['use_baseline'] else None
    s_od = raw_to_od(s_filtered, od_bp)

    snirf_data['s_od_filtered'] = s_od

    if verbose_stages or verbose:
        print("Signal converted!")
    # -----------------


    # ----------------- CHANNEL QUALITY EVALUATION - Scalp Coupling Index (SCI)
    if verbose_stages or verbose:
        print("\nEvaluating channel noise...")

    couple_signals_fn = couple_signals_fn_builder(df_filtered)
    mask, pair_scores = noisy_channels_start(s_od[:, bp[0]:bp[1]], couple_signals_fn, fs, threshold=0.5, verbose=verbose) # Remove noise channels 
    ##s_raw = s_raw[mask]; df_m = df_m[mask].reset_index(drop=True) # Toggle this if removing bad channels. This is too limiting for this projects recordings

    # Storing the channel quality
    df_w1 = df_filtered.iloc[couple_signals_fn()].reset_index(drop=True)
    pair_score_by_channel_index = dict(zip(df_w1['channelIndex'].to_numpy(), pair_scores.reshape(-1)))
    df_filtered["pair_score"] = df_filtered["channelIndex"].map(pair_score_by_channel_index)

    snirf_data['noisy_mask'] = mask
    snirf_data['noisy_pair_score_by_channel_index'] = pair_score_by_channel_index

    if verbose:
        print(f"{mask.sum()}/{len(s_od)} Over SCI threshold")
    # -----------------


    # This is safe as all channel filtering so far has been done in pairs
    df = df_filtered
    couple_signals_fn = couple_signals_fn_builder(df_filtered)

    snirf_data['couple_signals_fn'] = couple_signals_fn
    snirf_data['df_w1'] = df_filtered.iloc[couple_signals_fn()].reset_index(drop=True)


    # ----------------- RUN CORRECTION PIPELINES
    # p_outputs are dict of dataclass like {"NoCorrection": CorrectionPipelineOut{ s_od_before, s_od, df_m, bad_channels, couple_signals_fn, name }, "TDDR": ...}
    p_outputs = run_correction_pipelines(correction_pipelines, s_od=s_od, df_m=df, couple_signals_fn=couple_signals_fn, fs=fs, params=params, verbose=verbose)
    snirf_data['correction_outputs'] = p_outputs
    # -----------------


    # ----------------- FINISH PROCESSING (for each pipeline)
    # p_processed are dict of dicts like {"NoCorrection": {"s_mbll", "s_mbll_globally_regressed", "s_od_bp", "df_w1", "df_m", "stim_offset"}, ...}
    p_processed = {name:finish_processing(p_out, fs=fs, wavelengths=wavelengths, params=params) for name,p_out in p_outputs.items()}
    snirf_data['processing_outputs'] = p_processed
    # -----------------


    # Another CHANNEL QUALITY EVALUATION - Bandpassed OD Correlation in NoCorrection
    nocorr_pro = p_processed.get('NoCorrection',None)
    if nocorr_pro is not None:
        s_od_bp = nocorr_pro['s_od_bp']; df_w1_proc = nocorr_pro['df_w1']
        od_corr_map = {df_w1_proc.iloc[ch]['channelIndex']: np.corrcoef(s_od_bp[0, ch], s_od_bp[1, ch])[0, 1] for ch in range(s_od_bp.shape[1])}
        snirf_data['od_bp_correlation_by_channel_index'] = od_corr_map
    else:
        snirf_data['od_bp_correlation_by_channel_index'] = {}
    # -----------------



    # ----------------- Slice trials 
    # p_sliced are dict of dicts like {"NoCorrection":  {"NHM": {"slice": slice, "stim_offset": offset}, "SHM": ...}, ...}
    p_sliced = {name:slice_trials(p_pro, stim_data, p_pro['stim_offset']) for name,p_pro in p_processed.items()}
    snirf_data['slice_outputs'] = p_sliced

    if verbose_stages or verbose:
        print("\nSliced data into seperate trials.")

    if verbose:
        sample = p_sliced[next(iter(p_sliced))]
        nhm_off, shm_off, lhm_off = sample['NHM']['stim_offset'], sample['SHM']['stim_offset'], sample['LHM']['stim_offset']
        nhm_len, shm_len, lhm_len = sample['NHM']['s_mbll'].shape[-1], sample['SHM']['s_mbll'].shape[-1], sample['LHM']['s_mbll'].shape[-1]
        print(f"Trial lengths (in samples): \n\tNHM = {nhm_len} | offset = {nhm_off}\n\tSHM = {shm_len} | offset = {shm_off}\n\tLHM = {lhm_len} | offset = {lhm_off}\n")       # ------------------
    # -----------------

    if verbose_stages or verbose:
        print(f"\nFinished processing \"{file}\"")
    return snirf_data





# Helper function for processing OD -> bandpass -> trim -> MBLL
def finish_processing(p_out: CorrectionPipelineOut, fs: float, wavelengths: np.ndarray, params: dict) -> dict:
    _, s_mid, _ = split_bands(p_out.s_od, fs, highpass=params['highpass'], lowpass=params['lowpass'])

    s_bp_coupled, w1_mask = p_out.couple_signals_fn(s=s_mid)
    df_w1 = p_out.df_m[w1_mask].reset_index(drop=True)

    # Trim 10s from each size edge to remove any filtering artefacts
    trim_samples = params['trim_samples']
    s_bp_coupled_trim = s_bp_coupled[...,trim_samples:-trim_samples]
    s_mbll = od_to_mbll(s_od=s_bp_coupled_trim, wavelengths=wavelengths, df_m=df_w1)

    # Also coupling and trimming original od for later storage.
    s_od_coupled, _ = p_out.couple_signals_fn(s=p_out.s_od)
    s_od_trimmed = s_od_coupled[...,trim_samples:-trim_samples]

    # These are not used in this project, but could be given cleaner data. These were used when exploring the pipeline.
    s_mbll_globall_regressed = global_regression_mbll(s_mbll)

    short = df_w1[df_w1['distance'] < 13].index
    s_mbll_ss_regressed = ss_regression_mbll(s_mbll, short)

    return {"s_mbll": s_mbll, "s_mbll_ss_regressed": s_mbll_ss_regressed, "s_mbll_globally_regressed": s_mbll_globall_regressed, "s_od_bp": s_bp_coupled_trim, "s_od": s_od_trimmed, "df_w1": df_w1, "df_m":  p_out.df_m, "stim_offset": trim_samples}





# region    ===> Recording slicing and stitching (CUSTOM TO MY RECORDING)
def get_baseline_periods(s: np.ndarray, stim_data: np.ndarray) -> np.ndarray:
    s_list = []
    for start_label, end_label in BASELINE_PERIODS:
        s_base = s[:,stim_data[start_label][0]:stim_data[end_label][0]]
        s_list.append(s_base)
    return s_list

def slice_trials(processed_input: dict, stim_data: dict, offset: int) -> dict:
    # Should all be the same shape
    s_mbll = processed_input["s_mbll"].copy()
    s_od_bp = processed_input["s_od_bp"].copy()
    s_od = processed_input["s_od"].copy()

    assert s_mbll.shape == s_od_bp.shape

    out_data = {}
    for i,T in enumerate(TRIALS):
        start_label = BASELINE_PERIODS[i][0]
        end_label = BASELINE_PERIODS[i+1][1]

        assert len(stim_data[start_label]) == len(stim_data[end_label]) == 1

        start_sample = stim_data[start_label][0] - offset
        end_sample = stim_data[end_label][0] - offset

        start = min(s_mbll.shape[-1], max(0,start_sample))
        end = min(s_mbll.shape[-1], max(0,end_sample))

        assert end >= start >= 0

        out_data[T] = {"s_mbll":  s_mbll[...,start:end], "s_od_bp": s_od_bp[...,start:end], "s_od": s_od[...,start:end], "stim_offset": start + offset}

    return out_data # {"NHM": {"s_mbll": s_mbll slice, "s_od_bp": bandpassed od slice, "s_od": od slice, "stim_offset": offset}, "SHM": ...}
# endregion
