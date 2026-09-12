import itertools

import pandas as pd 
import numpy as np

from snirf import Snirf

# region    ===>    Setup Functions
def generate_df_m(snirf: Snirf):
    measurement_list = list(snirf.nirs[0].data[0].measurementList)

    sample = measurement_list[0]
    fields = [attr for attr in dir(sample) if not attr.startswith("_") and not callable(getattr(sample, attr))]

    measurement_data = []
    for m in measurement_list:
        measurement_data.append({field: getattr(m, field) for field in fields})

    df_m = pd.DataFrame(measurement_data)
    df_m['channelIndex'] = df_m.index + 1
    df_m = df_m.loc[:, df_m.nunique(dropna=False) > 1]
    df_m = df_m.drop(['location'], axis=1)
    df_m = df_m.astype(int)

    source_pos = snirf.nirs[0].probe.sourcePos3D
    detector_pos = snirf.nirs[0].probe.detectorPos3D

    df_m['sourcePos3D'] = df_m['sourceIndex'].apply(lambda i: np.array(source_pos[i - 1]))
    df_m['detectorPos3D'] = df_m['detectorIndex'].apply(lambda i: np.array(detector_pos[i - 1]))
    df_m['midpoint'] = df_m.apply(lambda r: (r['sourcePos3D'] + r['detectorPos3D']) / 2.0, axis=1)

    source_labels = snirf.nirs[0].probe.sourceLabels
    detector_labels = snirf.nirs[0].probe.detectorLabels

    df_m['sourceLabel'] = df_m.apply(lambda r: source_labels[int(r['sourceIndex'] - 1)][0 if r['wavelengthIndex'] == 1 else 1], axis=1)
    df_m['detectorLabel'] = df_m.apply(lambda r: detector_labels[int(r['detectorIndex'] - 1)], axis=1)
    df_m['label'] = df_m['sourceLabel'] + "_" + df_m['detectorLabel']

    df_m['sourceNode'] = df_m['sourceLabel'].str.extract(r'N(\d+)')[0].astype(int)
    df_m['detectorNode'] = df_m['detectorLabel'].str.extract(r'N(\d+)')[0].astype(int)

    df_m['distance'] = df_m.apply(lambda r: np.linalg.norm(r['sourcePos3D'] - r['detectorPos3D']), axis=1)

    return df_m

def generate_signal_raw(snirf: Snirf):
    dataTimeSeries = snirf.nirs[0].data[0].dataTimeSeries
    s_raw = np.array(dataTimeSeries).T
    return s_raw

def generate_detector_distances(df_m: pd.DateOffset):
    num_detectors = df_m['detectorIndex'].max()
    detector_distances = np.zeros((num_detectors,num_detectors))

    for di in range(num_detectors):
        for dj in range(0,di):
            detector_distances[di,dj] = detector_distances[dj,di]
        for dj in range(di,num_detectors):
            di_m, dj_m = df_m.loc[di]['detectorPos3D'], df_m.loc[dj]['detectorPos3D']
            detector_distances[di,dj] = np.linalg.norm(di_m - dj_m)
    
    return detector_distances

def generate_stim_data(snirf: Snirf):
    stims = snirf.nirs[0].stim
    freq = 1 / np.mean(np.diff(snirf.nirs[0].data[0].time))

    stim_data = {}

    for stim in stims:
        stim_n = stim.name
        stim_d = stim.data

        times = stim_d[:,0]
        sample_idxs = (times * freq).astype(int)

        stim_data[stim_n] = sample_idxs.tolist()

    return stim_data

def generate_aux_data(snirf: Snirf, aux_wanted: dict={"accel_x", "accel_y", "accel_z"}):
    aux_data = list(snirf.nirs[0].aux).copy()

    aux_by_name = {}
    for aux in aux_data:
        name = getattr(aux, "name", None)
        if name in aux_wanted:
            aux_by_name[name] = aux

    missing = aux_wanted - set(aux_by_name)
    if missing:
        print(f"Warning: missing aux channels: {sorted(missing)}")

    return aux_by_name

def get_frequency(snirf: Snirf):
    return 1 / np.mean(np.diff(snirf.nirs[0].data[0].time))

def get_wavelengths(snirf: Snirf):
    return np.rint(snirf.nirs[0].probe.wavelengths).astype(int)
# endregion 

# region    ===>    BATCH GENERATE VARIABLES
def generate_variables(file_path: str=None, snirf: Snirf=None):
    if snirf is None:
        snirf = Snirf(file_path)

    s_raw = generate_signal_raw(snirf)
    df_m = generate_df_m(snirf)
    stim_data = generate_stim_data(snirf)
    aux_by_name = generate_aux_data(snirf, aux_wanted={"accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",})
    d_dist = generate_detector_distances(df_m)
    fs = get_frequency(snirf)
    wavelengths = get_wavelengths(snirf)

    snirf.close()

    return {
        "s_raw": s_raw,
        "df_m": df_m,
        "stim_data": stim_data,
        "aux_by_name": aux_by_name,
        "d_dist": d_dist,
        "fs": fs,
        "wavelengths": wavelengths
    }
# endregion


# region    ===>    Stim Region Helpers
def motion_intervals(stim_data): # Return time bins of motion and their respective labels
    motion_intervals = []
    motion_interval_labels = []
    
    for key, val in stim_data.items():
        if key.startswith("Task_Motion_Start"):
            suffix = key.replace("Task_Motion_Start_", "")
            end_key = f"Task_Motion_End_{suffix}"
            
            if end_key in stim_data:
                start = val[0]
                end = stim_data[end_key][0]
                motion_intervals.append((start, end))
                motion_interval_labels.append(suffix)
    
    return motion_intervals, motion_interval_labels
# endregion


# region    ===>    Couple signals of the same wavelength
def couple_signals_fn_builder(df_m: pd.DataFrame): # returns a function that can be used for coupling channels that only differ by wavelength. (2*N, T) -> (2, N, T)
    groups = df_m.groupby(["sourceIndex", "detectorIndex"], sort=False).indices
    pair_idx = np.array([sorted(idxs, key=lambda i: df_m.at[i, "wavelengthIndex"]) for idxs in groups.values() if len(idxs) == 2], dtype=int)

    df_len = len(df_m)
    w1_mask = np.zeros(df_len, dtype=bool)

    if len(pair_idx) == 0:
        def couple_signals(s=None, pair_mask=None):
            if s is not None and pair_mask is not None:
                raise ValueError("Expected at least 1 of \'s\' and \'pair_mask\' to be None.")
            if pair_mask is not None:
                return np.zeros(df_len, dtype=bool)
            if s is not None:
                return np.empty((2, 0, s.shape[-1]), dtype=s.dtype), w1_mask
            return w1_mask

    else:
        if pair_idx.shape[1] != 2:  raise ValueError("Expected exactly 2 channels per (sourceIndex, detectorIndex) pair.")
        w1_mask[pair_idx[:, 0]] = True

        def couple_signals(s=None, pair_mask=None):
            if s is not None and pair_mask is not None:
                raise ValueError("Expected at least 1 of \'s\' and \'pair_mask\' to be None.")

            if pair_mask is not None:
                raw_mask = np.zeros(len(df_m), dtype=bool)
                raw_mask[pair_idx[:, 0]] = pair_mask
                raw_mask[pair_idx[:, 1]] = pair_mask
                return raw_mask
            
            if s is not None:
                sort_order = np.argsort(pair_idx[:, 0])
                sout = s[pair_idx].transpose(1, 0, 2)
                sout = sout[:, sort_order, :]  # reorder pairs to match w1_mask order
                return sout, w1_mask # (2,channels,samples), [channels_mask]
            
            else:
                return w1_mask
        
    return couple_signals
# endregion