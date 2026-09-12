import numpy as np
import pandas as pd

from src.stim_labels import *

from src.pipelines.processing_pipeline import process_recording
from src.pipelines.correction_pipelines import NoCorrection, TDDRCorrection, GGTPCACorrection, FGGTPCACorrection

from src.evaluation.block_avg import eval_block_avg
from src.evaluation.gvtd_evaluation import eval_motion_locked_gvtd
from src.evaluation.correlation_evaluation import eval_hbo_hbr_correlation
from src.evaluation.rest_energy_evaluation import eval_rest_energy

# file paths
ROOT_PATH = "data/subject-recordings/"
FILES = ["Subj-1", "Subj-2"]

# pipelines being used
CORRECTION_PIPELINES  =  [NoCorrection, TDDRCorrection, GGTPCACorrection, FGGTPCACorrection]


def evaluate_all(files: list[str], params: dict, correction_pipelines=None, verbose=False):
    # To avoid loading in all of the data at once, process each data file individually
    evaluations = []
    for file in files:
        evaluation = evaluate_recording(ROOT_PATH + file, params=params, correction_pipelines=correction_pipelines, verbose=verbose)
        evaluations.append(evaluation)
    return evaluations


def prepare_eval_inputs(snirf_data):
    p_sliced = snirf_data['slice_outputs']; p_processed = snirf_data['processing_outputs']
    eval_inputs = { p_name: { T: {
        "s_mbll": p_sliced[p_name][T]["s_mbll"],
        "s_od_bp": p_sliced[p_name][T]["s_od_bp"],
        "df_w1": p_processed[p_name]["df_w1"],
        "stim_data": snirf_data["stim_data"],
        "stim_offset": p_sliced[p_name][T]["stim_offset"],
        "fs": snirf_data["fs"],
        "trial_name": T,
    } for T in TRIALS } for p_name in p_processed.keys() }

    return eval_inputs



def evaluate_recording(file: str, params: dict, correction_pipelines=None, verbose_stages=True, verbose=False):
    if verbose_stages or verbose:
        print("\n--------------------------------------------------------")
        print(f"Starting evaluation pipeline for file \"{file}\"...")

    correction_pipelines = correction_pipelines or CORRECTION_PIPELINES

    if NoCorrection not in correction_pipelines:
        correction_pipelines = [NoCorrection] + list(correction_pipelines)

    snirf_data = process_recording(file=file, params=params, correction_pipelines=correction_pipelines, verbose_stages=verbose_stages, verbose=verbose)

    # ------------------------------------ EVALUATION 

    print("\nBeginning Evaluation...")

    # Seperating everything into clean per-correction per-trial data
    # {"NoCorrection": {"NHM": {"s_mbll", "s_od", "s_od_bp", "df_m", "stim_data", "stim_offset", "fs"}, "SHM": ...}, "TDDR":}
    eval_inputs = prepare_eval_inputs(snirf_data)    


    eval_out = {}
    for C,pipe_data in eval_inputs.items():
        pipe_eval = {}

        # -- Per Trial Evaluations
        for T,_ in pipe_data.items():
            trial_eval = {}

            # Block Average -----------------
            ba_eval = eval_block_avg(
                eval_input=eval_inputs[C][T],
                rest_window=params['ba_rest_window'],
                task_window=params['ba_task_window'],
                break_delay=params['ba_break_delay'] # break delay between task-start and task-window
            )

            trial_eval['block_average'] = ba_eval
            #---------------------------------


            # GVTD Analysis -----------------
            gvtd_eval = eval_motion_locked_gvtd(
                eval_input=eval_inputs[C][T],
                window_pre=params['gvtd_eval_window_pre'],
                window_break=params['gvtd_eval_window_break'], # break between window-pre and motion-onset
                window_post=params['gvtd_eval_window_post'],
                sci_threshold=params['gvtd_eval_sci_threshold']
            )
            
            trial_eval['gvtd_eval'] = gvtd_eval
            #---------------------------------


            # HbO HbR Correlation Analysis -----------------
            correlation_eval = eval_hbo_hbr_correlation(
                eval_input=eval_inputs[C][T],
                channel_index_to_od_correlation=snirf_data['od_bp_correlation_by_channel_index'],
                sci_threshold=params['correlation_eval_sci_threshold'],
                od_correlation_threshold=params['correlation_eval_od_corr_threshold']
            )
            
            trial_eval['correlation_eval'] = correlation_eval
            #---------------------------------

            # Rest Energy Analysis -----------------
            rest_energy_eval = eval_rest_energy(
                eval_input=eval_inputs[C][T],
                sci_threshold=params['rest_energy_eval_sci_threshold'],
            )
            
            trial_eval['rest_energy_eval'] = rest_energy_eval
            #---------------------------------


            pipe_eval[T] = trial_eval
        eval_out[C] = pipe_eval
        
    # ----------------------------------------------

    print("Evaluation Complete! Returning results...")

    # Returning {"NHM": evaluation, ...}, {"NoCorrection": (NHM, SHM, LHM), "TDDR": ...}
    return {"evaluation": eval_out, "data": snirf_data}

