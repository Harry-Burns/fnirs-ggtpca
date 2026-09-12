import numpy as np


def global_regression_mbll(s_mbll: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if s_mbll.ndim != 3:
        raise ValueError(f"Expected 3D array (2, channels, samples), got {s_mbll.shape}")

    out = s_mbll.copy()
    for i in range(out.shape[0]):
        out[i] = global_regression(out[i], eps=eps)
    return out


def global_regression(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if s.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, samples), got shape {s.shape}")

    a = np.mean(s, axis=0, keepdims=True) # (1, sum_samples)

    denom = float(np.sum(a * a)) # (1)
    
    if denom <= eps: # No global signal
        return s.copy()

    b = (s @ a.T) / denom # (num_channels, 1)
    return s - (b @ a) # (num_channels, num_samples)


def ss_regression_mbll(s_mbll: np.ndarray, ss_indices: list[int], eps: float = 1e-12) -> np.ndarray:
    if s_mbll.ndim != 3:
        raise ValueError(f"Expected 3D array (2, channels, samples), got {s_mbll.shape}")
    out = s_mbll.copy()
    for i in range(out.shape[0]):
        out[i] = ss_regression(out[i], ss_indices, eps=eps)
    return out

def ss_regression(s: np.ndarray, ss_indices: list[int], eps: float = 1e-12) -> np.ndarray:
    if s.ndim != 2:
        raise ValueError(f"Expected 2D array (channels, samples), got {s.shape}")
    
    # Regressor: mean of SS channels instead of mean of all channels
    a = np.mean(s[ss_indices, :], axis=0, keepdims=True) # (1, n_samples)
    
    denom = float(np.sum(a * a))
    if denom <= eps:
        return s.copy()
    
    b = (s @ a.T) / denom # (n_channels, 1)
    return s - (b @ a) # (n_channels, n_samples)

