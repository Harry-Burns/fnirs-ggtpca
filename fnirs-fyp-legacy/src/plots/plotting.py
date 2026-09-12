import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.snirf_processing import motion_intervals

# Plotting details SPECIFIC to my problem
def plot_motion_spans(ax, stims: dict, trim_samples: int=0):

    motion_bins,_ = motion_intervals(stims)

    for ms, me in motion_bins:
        ax.axvspan(ms-trim_samples,me-trim_samples , color="blue", alpha=0.8)

    
def plot_stims(ax, stims: dict, y_top: float = 1, stim_offset: int=0, excluded_strings=[], labels=False):
    for name, idx_list in stims.items():
        if any([s in name for s in excluded_strings]):
            continue

        for idx in idx_list:
            ax.axvline(x=idx-stim_offset, color="blue", linewidth=1, alpha=0.4)
            if labels:
                ax.text(idx-stim_offset,y_top,name,rotation=90,verticalalignment="bottom",horizontalalignment="center",fontsize=8,color="black",alpha=0.8)

def plot_stims_custom(ax, stims: dict, y_top: float=1, stim_offset: int = 0, labels=False, colours=["blue","green","yellow","red","lightblue"], motion_only=False):
    vline_handles = {}

    for name, idx_list in stims.items():
        assert len(idx_list) == 1
        idx = idx_list[0]

        if name.startswith("Task_Motion_Start_"):
            endname = name.replace("Start", "End")
            end = stims[endname][0]
            idx = (idx + end) / 2
            col = colours[0]; lw = 2; alpha = 0.7; label = "Motion Interval"
        elif re.fullmatch(r"Task_Start_\d+_NHM", name) and not motion_only:
            col = colours[1]; lw = 1; alpha = 0.4; label = "NHM Task Start"
        elif re.fullmatch(r"Task_Start_\d+_SHM", name) and not motion_only:
            col = colours[2]; lw = 1; alpha = 0.4; label = "SHM Task Start"
        elif re.fullmatch(r"Task_Start_\d+_LHM", name) and not motion_only:
            col = colours[3]; lw = 1; alpha = 0.4; label = "LHM Task Start"
        elif re.fullmatch(r"Baseline_Start", name) and not motion_only:
            col = colours[4]; lw = 1; alpha = 0.4; label = "Baseline Start"
        else:
            continue

        h = ax.axvline(x=idx - stim_offset, color=col, linewidth=lw, alpha=alpha, label=label if labels else None)
        vline_handles[label] = h

        if labels:
            ax.text(idx - stim_offset, y_top, name, rotation=90, verticalalignment="bottom", horizontalalignment="center", fontsize=8, color="black", alpha=0.8)
        else:
            legend = ax.legend(vline_handles.values(), vline_handles.keys(), loc="upper right", title="Events")
            ax.add_artist(legend)


# region Plotting in 3D
def plot_nodes_3d(ax, df_m, colfunc: callable=None, node_labels=True):
    detectors = df_m['detectorLabel'].unique()
    sources = df_m['sourceLabel'].unique()

    detector_pos = [df_m[df_m['detectorLabel'] == d].iloc[0]['detectorPos3D'] for d in detectors]
    source_pos = [df_m[df_m['sourceLabel'] == s].iloc[0]['sourcePos3D'] for s in sources]

    points = np.array(detector_pos + source_pos)
    colours = ["red"] * len(detector_pos) + ["orange"] * len(source_pos)

    labels = np.array(list(detectors) + list(sources)) if node_labels else None
    _plot_points_3d(ax, points, colours, labels=labels, s=20)

    for _, row in df_m.iterrows():
        if row['wavelengthIndex'] == 2:
            continue

        s = row['sourcePos3D']
        d = row['detectorPos3D']

        if colfunc is None: # Default colouring
            if row['distance'] < 15:
                col, alpha = "black", 0.8
            else:
                col, alpha = "black", 0.3
        else: # Custom colouring
            col, alpha = colfunc(row)

        _plot_lines_3d(ax, s, d, alpha, col, lw=1.5)

def _plot_lines_3d(ax, p0, p1, alpha=0.3, colour="black", lw=1.0):
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=(*plt.matplotlib.colors.to_rgb(colour), alpha),
        linewidth=lw
    )

def _plot_points_3d(ax, points, colours=None, labels=None, s=20, label_fs=7):
    if points.ndim == 1:
        points = points[None, :]
    if points.shape[1] != 3:
        raise ValueError(f"Expected (N,3), got {points.shape}")

    n = points.shape[0]

    if colours is None:
        colours = ['blue'] * n
    if len(colours) != n:
        raise ValueError("colours length must match points")

    if labels is not None and len(labels) != n:
        raise ValueError("labels length must match points")

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colours, s=s, edgecolors="none"
    )

    # --- draw labels ---
    if labels is not None:
        for (x, y, z), lab in zip(points, labels):
            if lab:
                ax.text(x, y, z, lab, fontsize=label_fs)
    # -------------------

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
# endregion


# region Plotting time series
def plot_channels(ax, s, df, norm=True, bin_mask=None, vertical_step: float=2.5, title: str="Time Series", s_label: str=None, bin_label: str="Bin Mask", show_legend=True):
    for i in range(s.shape[0]):
        if norm:
            norm_s = s[i]  / (max(max(s[i]), abs(min(s[i]))))
            plot_s = norm_s
        else:
            plot_s = s[i]

        line = ax.plot(plot_s + vertical_step * i, label=s_label)

        if bin_mask is not None:
            s_binned = np.where(bin_mask, plot_s, np.nan)
            bin = ax.plot(s_binned + vertical_step * (i), "red", label=bin_label)

        row = df.iloc[i]
        ax.text(-0.5, vertical_step*i + vertical_step * 0.4,  f"Channel {row['channelIndex']} | {row['label']} | {row['distance']:.2f} mm",fontsize=8, ha="left", va="center", color="black")

    ax.set_title(title)
    ax.set_yticks([]); ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude" if not norm else "Normalised amplitude")
    ax.grid(True, alpha=0.25)
    ax.figure.tight_layout()

    if show_legend:
        v = [line[0]] + ([bin[0]] if bin_mask is not None else [])
        l = [s_label] + ([bin_label] if bin_mask is not None  else [])
        legend = ax.legend(v, l, loc="upper left", title="Signals")
        ax.add_artist(legend)


def plot_paired_channels(ax, s, df, norm=True, bin_mask=None, vertical_step=2.5, title="Time Series", s_label: tuple[str,str]=("HbO", "HbR"), bin_label: str="Bin Mask"):
    for i in range(s.shape[1]):
        wv1 = s[0,i]
        wv2 = s[1,i]

        w1_norm_val = (max(max(wv1), abs(min(wv1)))) if norm else 1
        w2_norm_val = (max(max(wv1), abs(min(wv1)))) if norm else 1

        line1 = ax.plot(wv1 / w1_norm_val + vertical_step * i, color="blue", label=s_label[0])
        line2 = ax.plot(wv2 / w2_norm_val + vertical_step * i, color="orange", label=s_label[1])

        if bin_mask is not None :
            wv1_binned = np.where(bin_mask, wv1 / w1_norm_val, np.nan)
            wv2_binned = np.where(bin_mask, wv2 / w2_norm_val, np.nan)
            bin1 = ax.plot(wv1_binned + vertical_step * (i), "red", label=bin_label)
            ax.plot(wv2_binned + vertical_step * (i), "red")

        row = df.iloc[i]
        ax.text(-0.5, vertical_step*(i+0.4),  f"Channel {row['channelIndex']} | {row['label']} | {row['distance']:.2f} mm",fontsize=8, ha="left", va="center", color="black")

    ax.set_title(title)
    ax.set_yticks([]); ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude" if not norm else "Normalised amplitude")
    ax.grid(True, alpha=0.25)
    ax.figure.tight_layout()

    v = [line1[0], line2[0]] + ([bin1[0]] if bin_mask is not None  else [])
    l = [s_label[0], s_label[1]] + ([bin_label] if bin_mask is not None  else [])
    legend = ax.legend(v, l, loc="upper left", title="Signals")
    ax.add_artist(legend)

# endregion

# region Plotting Individual Channels
def plot_channel(ax, y, row, norm=False, bin_mask=None, title= "Channel Time Series", s_label: str = "Signal", bin_label: str = "Bin Mask"):
    y = np.asarray(y, dtype=float)

    if norm:
        scale = np.max(np.abs(y))
        if scale > 0:
            y = y / scale

    x = np.arange(len(y))
    ax.plot(x, y, linewidth=2)

    handles = [Line2D([0], [0], linewidth=2, label=s_label)]
    labels = [s_label]

    if bin_mask is not None:
        y_bin = np.where(bin_mask, y, np.nan)
        ax.plot(x, y_bin, color="red", linewidth=3)

        handles.append(Line2D([0], [0], color="red", linewidth=3, label=bin_label))
        labels.append(bin_label)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude" if not norm else "Normalised amplitude")
    ax.grid(True, alpha=0.25)

    legend = ax.legend(handles, labels, loc="upper left", title="Signals")
    ax.add_artist(legend)

    if row is not None:
        ax.text(0.01, 0.03, f"Channel {row['channelIndex']} | {row['label']} | {row['distance']:.2f} mm", transform=ax.transAxes, ha="left", va="bottom", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"))

    ax.figure.tight_layout()