# --- project bootstrap ---
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
# -------------------------

import os

import matplotlib.pyplot as plt
import numpy as np


STATS_PATH = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_sample_fit_stats_memopt.npz"
)
SAVE_PATH = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_sample_fit_stats_memopt_mean_slope_only.pdf"
)


def main():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(STATS_PATH)

    with np.load(STATS_PATH, allow_pickle=True) as data:
        required = {"time_steps", "cfg_names", "slope_mean", "slope_stderr"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Missing keys in stats file: {sorted(missing)}")

        time_steps = np.asarray(data["time_steps"], dtype=int)
        cfg_names = [str(x) for x in data["cfg_names"].tolist()]
        slope_mean = np.asarray(data["slope_mean"], dtype=float)
        slope_stderr = np.asarray(data["slope_stderr"], dtype=float)

    if slope_mean.shape != slope_stderr.shape:
        raise ValueError(
            f"slope_mean shape {slope_mean.shape} != slope_stderr shape {slope_stderr.shape}"
        )
    if slope_mean.shape[0] != len(cfg_names):
        raise ValueError(
            f"cfg_names length {len(cfg_names)} != slope rows {slope_mean.shape[0]}"
        )
    if slope_mean.shape[1] != len(time_steps):
        raise ValueError(
            f"time_steps length {len(time_steps)} != slope cols {slope_mean.shape[1]}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, name in enumerate(cfg_names):
        ax.errorbar(
            time_steps,
            slope_mean[i],
            yerr=slope_stderr[i],
            marker="o",
            lw=1.5,
            capsize=3,
            label=name,
        )

    ax.set_xlabel("t")
    ax.set_ylabel("Mean fit slope +/- stderr")
    ax.set_title("Mean fit slopes vs time-step")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    fig.savefig(SAVE_PATH)
    print(f"Saved plot: {SAVE_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
