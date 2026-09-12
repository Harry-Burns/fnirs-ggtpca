import numpy as np
import pandas as pd

from scipy.cluster.vq import kmeans2

SHORT_LONG_THRESHOLD = 18.0

def cluster(df: pd.DataFrame,s: np.ndarray,cluster_type='NodePair', **kwargs) -> list[tuple[pd.DataFrame,np.ndarray]]:
    if len(df) != len(s): raise ValueError("cluster: df and s must be equal size")
    if len(df) == 0: return []

    if cluster_type == 'NodePair':
        return node_pair_cluster(df,s)
    elif cluster_type == "MidpointKMeans":
        return midpoint_k_means_sl_split(df,s, **kwargs)
    else:
        return [(df,s)]

def node_pair_cluster(df: pd.DataFrame, s: np.ndarray):
    source_nodes = df['sourceNode'].unique()
    detector_nodes = df['detectorNode'].unique()

    clusters_complete = []
    clusters = []

    for i in source_nodes:
        for j in detector_nodes:
            cluster_name = f"{i}-{j}"
            reverse_name = f"{j}-{i}"

            if reverse_name in clusters_complete or cluster_name in clusters_complete:
                continue

            cluster_mask = ((df['sourceNode'] == i) & (df['detectorNode'] == j)) | ((df['sourceNode'] == j) & (df['detectorNode'] == i))
            clusters.append((df[cluster_mask],s[cluster_mask.to_numpy()]))

            clusters_complete.append(cluster_name); clusters_complete.append(reverse_name)
    
    return clusters

# sl_split -> short_long_split : Means we split the channels into short-distance and long-distance using some threshold 
def midpoint_k_means_sl_split(df: pd.DataFrame, s: np.ndarray, n_clusters: int, split_thresh: float=18, max_iter: int=10, rng: int=21):
    distances = df['distance'].to_numpy()

    short_mask = distances <= split_thresh
    long_mask = distances > split_thresh

    clusters = []
    n_clusters_side = max(1, n_clusters // 2)

    for group_mask in [short_mask, long_mask]:
        if not np.any(group_mask):
            continue

        df_group = df[group_mask].reset_index(drop=True)
        s_group = s[group_mask]

        midpoints = np.vstack(df_group['midpoint'].to_numpy())
        k = min(max(1, n_clusters_side), len(df_group))

        _, labels = kmeans2(midpoints, k, iter=max_iter, rng=rng, minit='++')

        for c in np.unique(labels):
            mask = labels == c
            clusters.append((df_group[mask].reset_index(drop=True), s_group[mask]))

    return clusters



