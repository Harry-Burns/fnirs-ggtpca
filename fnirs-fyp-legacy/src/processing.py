import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter,filtfilt


try:
    ROOT = Path(__file__).resolve().parents[1]
    COEF_TABLE = pd.read_csv(ROOT / "files" / "prahl_hemoglobin.csv")
except Exception as e:
    raise Exception(f"Couldnt find coef table for mbll... : {e}")


# region = Signal Processing
def raw_to_od(s_raw: np.ndarray, baseline_period: tuple[int,int]=None) -> np.ndarray:
    base_signal = s_raw[:, baseline_period[0]:baseline_period[1]] if baseline_period is not None else s_raw
    I_0 = np.mean(base_signal, axis=1, keepdims=True)
    s_od =  - np.log(s_raw / I_0)
    return s_od


# raw_to_od uses np.log which is base e
def od_to_mbll(s_od: np.ndarray, wavelengths: tuple[int,int], df_m: pd.DataFrame) -> np.ndarray:
    w1,w2 = wavelengths

    coef1 = COEF_TABLE.loc[COEF_TABLE['lambda'] == w1, ['hbo', 'hbr']]
    coef2 = COEF_TABLE.loc[COEF_TABLE['lambda'] == w2, ['hbo', 'hbr']]

    if len(coef1) != 1 or len(coef2) != 1:
        raise ValueError("Missing wavelength in COEF_TABLE.")

    hbo1, hbr1 = coef1.iloc[0].to_numpy(dtype=float)
    hbo2, hbr2 = coef2.iloc[0].to_numpy(dtype=float)

    eps = np.array([
        [hbo1, hbr1],
        [hbo2, hbr2]
    ]) * (np.log(10) / 10.0) # log10 -> ln and cm -> mm conversion

    dpf = np.array([6.0, 6.0], dtype=float)
    distances = df_m['distance'].to_numpy(dtype=float)

    delta_od = np.transpose(s_od, (1,2,0)) # shape (CH,T,2)
    
    def mbll(signal_pair, distance): # signal_pair: shape (T, 2)
        L = distance * dpf 
        M = eps * L[:,None]
        Minv = np.linalg.inv(M)
        out = signal_pair @ Minv.T
        return out  
    
    out_mbll = np.zeros_like(delta_od, dtype=float)
    for ch in range(delta_od.shape[0]):
        out_mbll[ch,:,:] = mbll(delta_od[ch,:,:], distances[ch])
    
    s_mbll = np.transpose(out_mbll, (2,0,1)) # shpe (2,CH,T)
    return s_mbll

def temporal_derivative(od: np.ndarray, addzero=True) -> np.ndarray:
    if od.ndim == 1:    return np.concatenate([[0.0], np.diff(od)]) if addzero else np.diff(od)
    if od.ndim == 2:    return np.concatenate([np.zeros((od.shape[0], 1)), np.diff(od, axis=1)],axis=1) if addzero else np.diff(od, axis=1)
    raise ValueError("od must be 1D (time,) or 2D (channels, time)")
# endregion


# region = Frequency-based Processing
def split_bands(x: np.ndarray, fs: float, lowpass: float = None, highpass: float = None, order: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nyq = 0.5 * fs
    low = band = high = None

    if highpass is not None and lowpass is not None:
        if not (0 < highpass < lowpass < nyq):
            raise ValueError("Require 0 < highpass < lowpass < nyquist.")
    elif highpass is not None:
        if not (0 < highpass < nyq):
            raise ValueError("Require 0 < highpass < nyquist.")
    elif lowpass is not None:
        if not (0 < lowpass < nyq):
            raise ValueError("Require 0 < lowpass < nyquist.")

    if lowpass is None and highpass is None:
        low = x.copy()
        high = np.zeros_like(x)
        return low, None, high

    if lowpass is not None and highpass is None:
        b, a = butter(order, lowpass / nyq, btype="low")
        low = filtfilt(b, a, x, axis=-1)
        high = x - low
        return low, None, high

    if highpass is not None and lowpass is None:
        b, a = butter(order, highpass / nyq, btype="high")
        high = filtfilt(b, a, x, axis=-1)
        low = x - high
        return low, None, high

    b, a = butter(order, [highpass / nyq, lowpass / nyq], btype="band")
    band = filtfilt(b, a, x, axis=-1)

    bL, aL = butter(order, highpass / nyq, btype="low")
    low = filtfilt(bL, aL, x, axis=-1)

    bH, aH = butter(order, lowpass / nyq, btype="high")
    high = filtfilt(bH, aH, x, axis=-1)

    return low, band, high
# endregion
