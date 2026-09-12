import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_individual_block_avg(block_avg_eval: dict, df_w1: pd.DataFrame, fs: float, break_delay: float, select_channel_indexes: list[int] = None, plot_min_max: bool = False):
    rest_blocks = block_avg_eval['rest']
    task_blocks = block_avg_eval['task']

    if select_channel_indexes is not None:
        mask = df_w1['channelIndex'].isin(select_channel_indexes)
        channels = np.where(mask)[0]
    else:
        channels = np.arange(len(df_w1))
    
    rest_t = np.arange(rest_blocks['avg'].shape[-1]) / fs
    rest_t = rest_t - rest_t[-1]          # ends at 0
    task_t = np.arange(task_blocks['avg'].shape[-1]) / fs
    task_t = task_t + break_delay / fs        # starts at break_delay


    fig, axes = plt.subplots(
        len(channels), 3,
        figsize=(14, 2.2 * len(channels)),
        sharex='col', sharey='row'
    )

    for r, ch in enumerate(channels):
        df_w1_row = df_w1.iloc[ch]
        #axes[r, 1].sharey(axes[r, 0])
        # ------------------------
        # Column 1: rest traces
        # ------------------------

        axes[r, 0].plot(rest_t, rest_blocks['avg'][0, ch], color="blue", label="HbO (avg)")
        if plot_min_max:
            axes[r, 0].plot(rest_t, rest_blocks['max'][0, ch], color="lightblue", linestyle="--", linewidth=0.5, alpha=0.7, label="HbO (max)")
            axes[r, 0].plot(rest_t, rest_blocks['min'][0, ch], color="lightblue", linestyle="--", linewidth=0.5, alpha=0.7, label="HbO (min)")

        axes[r, 0].plot(rest_t, rest_blocks['avg'][1, ch], color="orange", label="HbR (avg)")
        if plot_min_max:
            axes[r, 0].plot(rest_t, rest_blocks['max'][1, ch], color="yellow", linestyle="--", linewidth=0.5, alpha=0.7, label="HbR (max)")
            axes[r, 0].plot(rest_t, rest_blocks['min'][1, ch], color="yellow", linestyle="--", linewidth=0.5, alpha=0.7, label="HbR (min)")

        axes[r, 0].set_ylabel(f"Ch {df_w1_row['channelIndex']} ({df_w1_row['distance']:.1f}mm) \nConcentration")
        if r == 0:
            axes[r, 0].set_title("Rest Interval")
            axes[r, 0].legend()

        # ------------------------
        # Column 2: task traces
        # ------------------------
        axes[r, 1].plot(task_t, task_blocks['avg'][0, ch], color="blue", label="HbO (avg)")
        if plot_min_max:
            axes[r, 1].plot(task_t, task_blocks['max'][0, ch], color="lightblue", linestyle="--", linewidth=0.5, alpha=0.7, label="HbO (max)")
            axes[r, 1].plot(task_t, task_blocks['min'][0, ch], color="lightblue", linestyle="--", linewidth=0.5, alpha=0.7, label="HbO (min)")

        axes[r, 1].plot(task_t, task_blocks['avg'][1, ch], color="orange", label="HbR (avg)")
        if plot_min_max:
            axes[r, 1].plot(task_t, task_blocks['max'][1, ch], color="yellow", linestyle="--", linewidth=0.5, alpha=0.7, label="HbR (max)")
            axes[r, 1].plot(task_t, task_blocks['min'][1, ch], color="yellow", linestyle="--", linewidth=0.5, alpha=0.7, label="HbR (min)")

        if r == 0:
            axes[r, 1].set_title("Task Interval")
            axes[r, 1].legend()

        # ------------------------
        # Column 3: full-span boxplots
        # ------------------------
        rest_hbo = rest_blocks['blocks_corrected'][0, ch].reshape(-1)
        rest_hbr = rest_blocks['blocks_corrected'][1, ch].reshape(-1)
        task_hbo = task_blocks['blocks_corrected'][0, ch].reshape(-1)
        task_hbr = task_blocks['blocks_corrected'][1, ch].reshape(-1)

        axes[r, 2].boxplot(
            [rest_hbo, rest_hbr, task_hbo, task_hbr],
            tick_labels=["Rest HbO", "Rest HbR", "Task HbO", "Taks HbR"],
            widths=0.6
        )

        axes[r, 0].xaxis.set_ticks_position("bottom")
        axes[r, 1].xaxis.set_ticks_position("bottom")
        axes[r, 2].xaxis.set_ticks_position("bottom")


    axes[0, 0].set_xlabel("Time (s) relative to starting task.")
    axes[0, 1].set_xlabel("Time (s) relative to starting task.")

    axes[0, 2].set_title("Value ranges")

    axes[-1, 0].set_xlabel("Time (s) relative to starting task.")
    axes[-1, 1].set_xlabel("Time (s) relative to starting task.")

    fig.suptitle("Rest vs Task HbO and HbR", fontsize=16, y=0.999)

    plt.tight_layout()
    plt.show()

def plot_combined_block_avg(combined: dict, fs: float, rest_window: float, task_window: float, break_delay: float, task_duration: int, title: str = ""):
    colours  = {"HbO": "blue", "HbR": "orange"}
    n_blocks = combined["n_blocks"]

    # Stitch pre-computed arrays — NaN gap for blocks, zeros for summary stats
    nan_gap = lambda *shape: np.full((2, *shape), np.nan)

    stitched_blocks = np.concatenate([combined["rest"]["blocks"], nan_gap(n_blocks, int(break_delay * fs)), combined["task"]["blocks"]], axis=2)  # (2, n_blocks, total_len)

    avg = np.concatenate([combined["rest"]["avg"], nan_gap(int(break_delay * fs)), combined["task"]["avg"]], axis=1)
    sem = np.concatenate([combined["rest"]["sem"], nan_gap(int(break_delay * fs)), combined["task"]["sem"]], axis=1)

    total_len = stitched_blocks.shape[2]
    t = (np.arange(total_len)) / fs - (rest_window)

    task_start_s = 0
    task_end_s = task_duration

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    for ax, (ci, chrom) in zip(axes, enumerate(["HbO", "HbR"])):
        col = colours[chrom]

        for b in range(n_blocks):
            plot_label = "Individual block" if b == 0 else None
            ax.plot(t, stitched_blocks[ci, b], color=col, alpha=0.18, linewidth=0.8, label=plot_label)

        ax.fill_between(t, avg[ci] - sem[ci], avg[ci] + sem[ci], color=col, alpha=0.25)
        ax.plot(t, avg[ci], color=col, linewidth=2, label=f"mean ± SEM ({n_blocks} blocks)")

        ax.axvline(0, color="black", linewidth=0.9, linestyle="--", label="Task onset")
        ax.axvspan(task_start_s, task_end_s, alpha=0.07, color="grey", label="Task period")
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)

        ax.set_xlabel("time relative to onset  (s)", fontsize=9)
        ax.set_ylabel("ΔConc  (a.u., baseline-corrected)", fontsize=9)
        ax.set_title(chrom, fontsize=11, fontweight="bold", color=col)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(title if title else "Block average",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()