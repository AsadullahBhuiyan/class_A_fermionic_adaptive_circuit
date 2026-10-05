"""
Local merge script: combine all block checkpoints into final products per Ny.

Run after all OSG jobs finish and results are synced back.

Usage:
    python merge_slope_vs_system_size.py --results-dir ./results --output-dir ./merged
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from strip_entropy_streaming_gpu import (
    HELPER_VERSION,
    StripContourAccumulator,
    entropy_curve_rows_from_accumulator,
    fit_full_x_log_chord,
    write_contour_batch_files,
    write_json_atomic,
)

NX              = 20
NSHELL          = 1
TOTAL_SAMPLES   = 100
FIT_AY_MIN      = 8
TARGET_SLOPE    = 1.0 / 3.0
AY_BATCH_SIZE   = 4
CYCLE_BATCH_SIZE = 1
CAMPAIGN_NAME   = "pure_state_entanglement_slope_vs_system_size"


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def merge_ny(Ny: int, block_paths: list[Path], output_dir: Path) -> dict:
    cycles    = Ny // 2
    fit_cycle = Ny // 2
    run_id    = f"N{NX}x{Ny}_nsh{NSHELL}_dwtrunc1_C{cycles}_S{TOTAL_SAMPLES}"
    run_root  = output_dir / "runs" / run_id
    fig_root  = run_root / "figures"
    run_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    summary_path  = run_root / "run_summary.json"
    entropy_path  = run_root / "entropy_curves.csv"
    fit_path      = run_root / "fit_rows.csv"
    full_ckpt     = run_root / "accumulator_checkpoint.npz"

    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        if s.get("status") == "complete":
            print(f"[skip] Ny={Ny} already merged")
            return s

    config = {
        "Nx": NX, "Ny": Ny, "nshell": NSHELL, "cycles": cycles,
        "fit_cycle": fit_cycle, "fit_Ay_min": FIT_AY_MIN,
        "total_samples": TOTAL_SAMPLES, "run_id": run_id,
        "Ay_values": list(range(Ny // 2 + 1)),
        "merged_from_blocks": True,
    }

    print(f"[merge] Ny={Ny}: {len(block_paths)} block(s) ...", flush=True)
    accumulator = StripContourAccumulator(
        nx=NX, ny=Ny,
        ay_values=range(Ny // 2 + 1),
        cycles=[fit_cycle],
        config=config,
    )
    for path in sorted(block_paths):
        accumulator.merge_checkpoint(path, require_contours=True)

    # Validate total count
    expected = TOTAL_SAMPLES * Ny
    for ay in accumulator.ay_values:
        count = accumulator.count[ay]
        if not np.all(count == expected):
            bad = [int(accumulator.cycles[i]) for i, v in enumerate(count) if int(v) != expected]
            raise RuntimeError(f"Ny={Ny}: incomplete count for Ay={ay}. Bad cycles: {bad[:5]}")

    accumulator.save_checkpoint(full_ckpt, include_contours=True)

    batch_records = write_contour_batch_files(
        accumulator=accumulator,
        output_dir=run_root / "contour_batches",
        ay_batch_size=AY_BATCH_SIZE,
        cycle_batch_size=CYCLE_BATCH_SIZE,
        config=config,
    )

    entropy_curves = (
        pd.DataFrame(entropy_curve_rows_from_accumulator(accumulator))
        .sort_values(["cycle", "Ay"]).reset_index(drop=True)
    )
    save_csv(entropy_curves, entropy_path)

    fit_row = fit_full_x_log_chord(
        entropy_curves, nx=NX, ny=Ny, nshell=NSHELL,
        cycle=fit_cycle, fit_ay_min=FIT_AY_MIN, target_slope=TARGET_SLOPE,
    )
    save_csv(pd.DataFrame([fit_row]), fit_path)

    # Figure
    fig_path = fig_root / f"entropy_vs_log_chord_cycle{fit_cycle}.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    sub = entropy_curves[(entropy_curves["cycle"] == fit_cycle) & (entropy_curves["Ay"] > 0)].sort_values("Ay")
    x = sub["log_sin_pi_Ay_over_Ny"].to_numpy(dtype=np.float64)
    y = sub["entropy_mean"].to_numpy(dtype=np.float64)
    ye = sub["entropy_sem"].to_numpy(dtype=np.float64)
    xl = np.linspace(x.min(), x.max(), 300)
    yl = fit_row["slope"] * xl + fit_row["intercept"]
    ax.axvspan(fit_row["fit_x_min"], fit_row["fit_x_max"], color="0.5", alpha=0.18,
               label=f"fit Ay={int(fit_row['Ay_fit_min'])}..{int(fit_row['Ay_fit_max'])}")
    ax.errorbar(x, y, yerr=ye, marker="o", ms=3.5, lw=1.3, capsize=2, color="tab:blue")
    ax.plot(xl, yl, color="black", ls="--", lw=1.5,
            label=f"m={fit_row['slope']:.4f}  r²={fit_row['r2']:.4f}")
    ax.set_title(f"cycle {fit_cycle}, N{NX}x{Ny}, nshell={NSHELL}")
    ax.set_xlabel(r"$\log[\sin(\pi A_y/N_y)]$")
    ax.set_ylabel(r"full_x $S(A_y)$")
    ax.legend(frameon=True, fontsize=8)
    fig.savefig(fig_path)
    plt.close(fig)

    summary = {
        "status": "complete", "Ny": Ny, "run_id": run_id,
        "fit": fit_row, "entropy_curves": str(entropy_path),
        "fit_rows": str(fit_path), "figure": str(fig_path),
        "checkpoint": str(full_ckpt),
        "block_checkpoints": [str(p) for p in sorted(block_paths)],
        "created_unix": time.time(),
    }
    write_json_atomic(summary_path, summary)
    print(f"[done] Ny={Ny}  slope={fit_row['slope']:.4f}  r²={fit_row['r2']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory containing job_N/ subdirs from HTCondor")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write merged outputs")
    args = parser.parse_args()

    # Collect all block checkpoints grouped by Ny
    blocks_by_ny: dict[int, list[Path]] = {}
    for block_path in sorted(args.results_dir.rglob("block_*.npz")):
        if "_status" in block_path.name:
            continue
        # parent dir is N{NX}x{Ny}
        ny_str = block_path.parent.name  # e.g. "N20x80"
        Ny = int(ny_str.split("x")[1])
        blocks_by_ny.setdefault(Ny, []).append(block_path)

    if not blocks_by_ny:
        raise FileNotFoundError(
            f"No block_*.npz files found under {args.results_dir}.\n"
            "Did you sync results back from the access point?"
        )

    print(f"Found blocks for Ny: {sorted(blocks_by_ny.keys())}")
    for Ny, paths in sorted(blocks_by_ny.items()):
        print(f"  Ny={Ny}: {len(paths)} block(s)")

    all_fit_rows = []
    for Ny in sorted(blocks_by_ny.keys()):
        summary = merge_ny(Ny, blocks_by_ny[Ny], args.output_dir)
        all_fit_rows.append(summary["fit"])

    # Campaign summary
    fit_summary = pd.DataFrame(all_fit_rows).sort_values("Ny").reset_index(drop=True)
    summary_path = args.output_dir / "slope_vs_system_size_summary.csv"
    save_csv(fit_summary, summary_path)

    # Campaign figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].errorbar(fit_summary["Ny"], fit_summary["slope"],
                     yerr=fit_summary["slope_err"], marker="o", capsize=3, lw=1.4)
    axes[0].axhline(TARGET_SLOPE, color="black", ls="--", lw=1.0, label="target 1/3")
    axes[0].set_xlabel("Ny")
    axes[0].set_ylabel("full_x log-chord slope")
    axes[0].set_title(f"N{NX} slope vs system size")
    axes[0].legend(frameon=True)

    axes[1].errorbar(1.0 / fit_summary["Ny"], fit_summary["slope"],
                     yerr=fit_summary["slope_err"], marker="o", capsize=3, lw=1.4)
    axes[1].axhline(TARGET_SLOPE, color="black", ls="--", lw=1.0, label="target 1/3")
    axes[1].set_xlabel("1 / Ny")
    axes[1].set_ylabel("full_x log-chord slope")
    axes[1].set_title(f"N{NX} slope vs 1/Ny")
    axes[1].legend(frameon=True)

    fig.savefig(args.output_dir / "slope_vs_system_size.png", dpi=180)
    plt.close(fig)

    print(f"\n[campaign complete]")
    print(fit_summary[["Ny", "slope", "slope_err", "r2"]].to_string(index=False))
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
