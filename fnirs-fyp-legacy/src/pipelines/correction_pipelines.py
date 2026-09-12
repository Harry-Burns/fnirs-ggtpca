from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from src.snirf_processing import couple_signals_fn_builder

def run_correction_pipelines(correction_pipelines, s_od: np.ndarray, df_m: pd.DataFrame, couple_signals_fn: callable, fs: float, params: dict, verbose: bool=False):
    cps = [
        correction_pipe(
            CorrectionPipelineIn(s_od=s_od, df_m=df_m, couple_signals_fn=couple_signals_fn, fs=fs, params=params), 
            verbose=verbose
        ) for correction_pipe in correction_pipelines
    ]
    
    outputs = {}
    for pipe in cps:
        out: CorrectionPipelineOut = pipe.pipeline()
        outputs[out.name] = out
    
    # returns {'NoCorrection': CorrectionPipelineOut, 'TDDR': ...}
    return outputs


@dataclass
class CorrectionPipelineOut:
    s_od_before: np.ndarray
    s_od: np.ndarray
    df_m: pd.DataFrame
    bad_channels: np.ndarray
    couple_signals_fn: callable
    name: str


@dataclass
class CorrectionPipelineIn:
    s_od: np.ndarray
    df_m: np.ndarray
    couple_signals_fn: callable
    fs: float
    params: dict[str,Any]


# --- TEMPLATE PIPELINE --- 
class CorrectionPipeline:
    name = "Template"
    def __init__(self, pipe_in: CorrectionPipelineIn, verbose=False):
        self.s_od = pipe_in.s_od.copy()
        self.df_m = pipe_in.df_m.copy()
        self.couple_signals_fn = pipe_in.couple_signals_fn   
        self.fs = pipe_in.fs 

        self.verbose = verbose

        self.setup_params(pipe_in.params)

    def setup_params(self, params):
        pass

    def pipeline(self) -> CorrectionPipelineOut:
        raise NotImplementedError
# -------------------------


from src.tddr import tddr_pipeline
from src.clustering import cluster
from src.gvtd import gvtd_pipeline
from src.tpca import tpca

class NoCorrection(CorrectionPipeline):
    name = "NoCorrection"

    def pipeline(self) -> CorrectionPipelineOut:
        return CorrectionPipelineOut(
            s_od_before=self.s_od,
            s_od=self.s_od.copy(),
            df_m=self.df_m.copy(),
            bad_channels=np.zeros(self.s_od.shape[0], dtype=bool),
            couple_signals_fn=self.couple_signals_fn,
            name=self.name
        )


class TDDRCorrection(CorrectionPipeline):
    name = "TDDR"

    def setup_params(self, params):
        self.max_iter = params['tddr_max_iter']
        self.lowpass = params['tddr_lowpass']
        self.trim_samples = params['tddr_trim_samples']

    def pipeline(self) -> CorrectionPipelineOut:
        s_tddr, _ = tddr_pipeline(
            s=self.s_od.copy(), 
            lowpass=self.lowpass, 
            trim_samples=self.trim_samples, 
            fs=self.fs,
            max_iter=self.max_iter,
            readd_trim=True
        )
        
        return CorrectionPipelineOut(
            s_od_before=self.s_od,
            s_od=s_tddr,
            df_m=self.df_m,
            bad_channels=np.zeros(self.s_od.shape[0], dtype=bool),
            couple_signals_fn=self.couple_signals_fn,
            name=self.name
        )


class GGTPCACorrection(CorrectionPipeline):    
    name="GGTPCA"

    def setup_params(self, params):
        self.gvtd_filter = False

        # gvtd
        self.C = params['c']
        self.bin_padding = params['bin_padding']

        self.gvtd_lowpass = params['gvtd_lowpass']
        self.gvtd_highpass = params['gvtd_highpass']
        self.trim_samples = params['gvtd_trim_samples']
        self.gvtd_filter_window = params['gvtd_filter_window']

        # tpca
        self.var_target = params['var_target']
        self.max_pcs = params['max_pcs']
        self.align_win = params['align_win']
        self.taper = params['taper']

        # clustering
        self.cluster_method = params["cluster_method"]
        self.channels_per_cluster = params["channels_per_cluster"]

        # total
        self.num_iter = params['ggtpca_iter']

        # Tracking/logging
        self.cluster_details = {}
        

    def pipeline(self) -> CorrectionPipelineOut:
        n_clusters = len(self.df_m) // self.channels_per_cluster if self.cluster_method == "MidpointKMeans" else None
        clusters = cluster(df=self.df_m, s=self.s_od.copy(), cluster_type=self.cluster_method, n_clusters=n_clusters)

        if not clusters:
            raise RuntimeError(f"{self.name} produced no non-empty clusters.")

        clusters_corr = []
        for i,(_df,_s) in enumerate(clusters):
            if _s.shape[0] == 0:
                continue
            
            clusters_corr.append( self.correct_cluster(s=_s, df=_df, clust_idx=i) )

        dfs, signals, bad = zip(*clusters_corr)

        s_out = np.concatenate(signals, axis=0)
        df_out = pd.concat(dfs, axis=0, ignore_index=True)
        bad_channels_out = np.concatenate(bad, axis=0)

        couple_clean = couple_signals_fn_builder(df_out)

        return CorrectionPipelineOut(
            s_od_before=self.s_od.copy(),
            s_od=s_out,
            df_m=df_out,
            bad_channels=bad_channels_out,
            couple_signals_fn=couple_clean,
            name=self.name
        )
    
    def correct_cluster(self, s: np.ndarray, df: pd.DataFrame, clust_idx:int):
        s_cluster_out = s.copy()

        active_channel_mask = np.ones(s.shape[0], dtype=bool)
        cluster_details = []

        for i in range(self.num_iter):
            s_active = s_cluster_out[active_channel_mask]

            gvtd_out = gvtd_pipeline(
                s=s_active,
                lowpass=self.gvtd_lowpass, 
                highpass=self.gvtd_highpass, 
                trim_samples=self.trim_samples, 
                fs=self.fs, 
                filter=self.gvtd_filter if i == 0 else False, 
                filter_window=self.gvtd_filter_window,
                c=self.C, 
                bin_padding=self.bin_padding, 
                bin_ref="original"
            )

            channel_mask = gvtd_out["channel_mask"]
            threshold_bins = gvtd_out["threshold_bins"]

            cluster_details.append({
                "gvtd": gvtd_out["gvtd"],
                "g_thresh": gvtd_out["g_thresh"],
                "threshold_bins": threshold_bins
            })


            if i==0 and self.gvtd_filter:
                active_channel_mask = channel_mask
                s_active = s_cluster_out[active_channel_mask]

                if not np.any(active_channel_mask):
                    if self.verbose:
                        print(f"No valid channels in cluster: {clust_idx}")
                    break

            if len(threshold_bins) == 0:
                if self.verbose:
                    print(f"No GVTD bins in iter {i} cluster {clust_idx}")
                break

            s_tpca = tpca(
                s=s_active, 
                bins=threshold_bins,  
                align_win=self.align_win, 
                taper=self.taper, 
                var_target=self.var_target, 
                max_pcs=self.max_pcs,
                verbose=self.verbose
            )   

            s_cluster_out[active_channel_mask] = s_tpca
            
        self.cluster_details[clust_idx] = cluster_details

        bad_channels = ~active_channel_mask
        return (df, s_cluster_out, bad_channels)

class FGGTPCACorrection(GGTPCACorrection):
    name = "FGGTPCA"

    def setup_params(self, params):
        super().setup_params(params)

        self.gvtd_filter = True














