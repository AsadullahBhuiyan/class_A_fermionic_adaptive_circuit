"""
Local merge script: combine all 10 block checkpoints into the final
slope-vs-cycle products (entropy curves, slope table, figures, manifest).

Run this on your local machine after all 10 OSG jobs finish and you've
synced the output directory back.

Usage:
    python merge_slope_vs_cycle.py --output-dir /path/to/output
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

NX = 20
NY = 40
NSHELL = 1
SAMPLES = 100
CYCLES = 100
FIT_AY_MIN = 8
ALPHA_1 = 1.0
ALPHA_2 = 30.0
DW_TRUNCATION = True
SAMPLE_BLOCK_SIZE = 10
AY_BATCH_SIZE = 4
CYCLE_BATCH_SIZE = 5
TARGET_SLOPE_FULL_X = 1.0 / 3.0
SELECTED_PLOT_CYCLES = [0, 5, 10, 20, 50, 100]
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"

RUN_ID = f"N{NX}x{NY}_nsh{NSHELL}_dwtrunc{int(DW_TRUNCATION)}_C{CYCLES}_S{SAMPLES}"
CAMPAIGN_NAME = "pure_state_entanglement_slope_vs_cycle"

CAMPAIGN_CONFIG = {
    "Nx": NX, "Ny": NY, "nshell": NSHELL, "samples": SAMPLES, "cycles": CYCLES,
    "fit_Ay_min": FIT_AY_MIN, "alpha_1": ALPHA_1, "alpha_2": ALPHA_2,
    "dw_truncation": DW_TRUNCATION, "init_mode": "default", "dtype": "complex128",
    "sequence": "raster_y", "perfect_correction": True, "postselect": False,
    "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT, "gpu_target": "A100 40GB",
    "sample_block_size": SAMPLE_BLOCK_SIZE, "checkpoint_format": "full_contour",
}


def save_dataframe_atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    tmp.replace(path)


def validate_block_checkpoint(path: Path, *, block_count: int) -> None:
    acc = StripContourAccumulator.load_checkpoint(path, require_contours=True)
    expected = int(block_count) * NY
    for ay in acc.ay_values:
        count = acc.count[ay]
        if not np.all(count == expected):
            bad = [int(acc.cycles[i]) for i, v in enumerate(count) if int(v) != expected]
            raise RuntimeError(
                f"Block checkpoint {path} incomplete for Ay={ay}. "
                f"Expected count {expected}. Bad cycles: {bad[:10]}"
            )


def validate_final_counts(acc: StripContourAccumulator) -> None:
    expected = SAMPLES * NY
    for ay in acc.ay_values:
        count = acc.count[ay]
        if not np.all(count == expected):
            bad = [int(acc.cycles[i]) for i, v in enumerate(count) if int(v) != expected]
            raise RuntimeError(
                f"Merged accumulator incomplete for Ay={ay}. Expected {expected}. "
                f"Bad cycles: {bad[:10]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Same output-dir used in run_slope_vs_cycle_block.py")
    args = parser.parse_args()

    run_root = args.output_dir / "runs" / RUN_ID
    block_root = run_root / "sample_block_checkpoints"
    contour_root = run_root / "contour_batches"
    fig_root = run_root / "figures"
    run_root.mkdir(parents=True, exist_ok=True)
    contour_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    summary_path = run_root / "run_summary.json"
    fit_path = run_root / "slope_vs_cycle.csv"
    entropy_path = run_root / "entropy_curves.csv"
    full_checkpoint_path = run_root / "accumulator_checkpoint_full.npz"

    # ── Check for already-complete run ───────────────────────────────────────
    if summary_path.exists() and fit_path.exists() and entropy_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete":
            print(f"[skip] {RUN_ID} already complete at {summary_path}")
            return

    # ── Validate and collect all block checkpoints ───────────────────────────
    n_blocks = SAMPLES // SAMPLE_BLOCK_SIZE
    block_paths = []
    for i in range(n_blocks):
        block_start = i * SAMPLE_BLOCK_SIZE
        block_stop = block_start + SAMPLE_BLOCK_SIZE
        path = block_root / f"block_{block_start:03d}_{block_stop:03d}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing block checkpoint: {path}\n"
                f"Make sure all {n_blocks} OSG jobs have completed and you've synced the output dir."
            )
        print(f"[validate] {path.name} ...", end=" ", flush=True)
        validate_block_checkpoint(path, block_count=SAMPLE_BLOCK_SIZE)
        print("ok")
        block_paths.append(path)

    # ── Merge ─────────────────────────────────────────────────────────────────
    print(f"\n[merge] combining {n_blocks} blocks ...", flush=True)
    accumulator = StripContourAccumulator(
        nx=NX,
        ny=NY,
        ay_values=range(NY // 2 + 1),
        cycles=range(CYCLES + 1),
        config={**CAMPAIGN_CONFIG, "run_id": RUN_ID, "merged_from_blocks": True},
    )
    for path in block_paths:
        accumulator.merge_checkpoint(path, require_contours=True)
    validate_final_counts(accumulator)
    accumulator.save_checkpoint(full_checkpoint_path, include_contours=True)
    print(f"[merged] saved full checkpoint: {full_checkpoint_path}")

    # ── Write contour batches ─────────────────────────────────────────────────
    batch_records = write_contour_batch_files(
        accumulator=accumulator,
        output_dir=contour_root,
        ay_batch_size=AY_BATCH_SIZE,
        cycle_batch_size=CYCLE_BATCH_SIZE,
        config={**CAMPAIGN_CONFIG, "run_id": RUN_ID},
    )

    # ── Entropy curves ────────────────────────────────────────────────────────
    entropy_curves = (
        pd.DataFrame(entropy_curve_rows_from_accumulator(accumulator))
        .sort_values(["cycle", "Ay"])
        .reset_index(drop=True)
    )
    save_dataframe_atomic_csv(entropy_curves, entropy_path)
    print(f"[saved] entropy curves: {entropy_path}")

    # ── Slope vs cycle ────────────────────────────────────────────────────────
    slope_rows = []
    for cycle in range(CYCLES + 1):
        slope_rows.append(fit_full_x_log_chord(
            entropy_curves, nx=NX, ny=NY, nshell=NSHELL, cycle=cycle,
            fit_ay_min=FIT_AY_MIN, target_slope=TARGET_SLOPE_FULL_X,
        ))
    slope_vs_cycle = (
        pd.DataFrame(slope_rows).sort_values("cycle").reset_index(drop=True)
    )
    save_dataframe_atomic_csv(slope_vs_cycle, fit_path)
    save_npz_atomic(
        run_root / "slope_vs_cycle.npz",
        cycle=slope_vs_cycle["cycle"].to_numpy(dtype=np.int64),
        slope=slope_vs_cycle["slope"].to_numpy(dtype=np.float64),
        slope_err=slope_vs_cycle["slope_err"].to_numpy(dtype=np.float64),
        r2=slope_vs_cycle["r2"].to_numpy(dtype=np.float64),
        config_json=np.asarray(json.dumps(CAMPAIGN_CONFIG, sort_keys=True)),
    )
    print(f"[saved] slope vs cycle: {fit_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_slope = fig_root / "slope_vs_cycle.png"
    fig_r2 = fig_root / "r2_vs_cycle.png"
    fig_selected = fig_root / "entropy_vs_log_chord_selected_cycles.png"
    fig_heat = fig_root / "entropy_heatmap_cycle_Ay.png"

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.errorbar(slope_vs_cycle["cycle"], slope_vs_cycle["slope"],
                yerr=slope_vs_cycle["slope_err"], marker="o", ms=2.5, lw=1.2, capsize=2)
    ax.axhline(TARGET_SLOPE_FULL_X, color="black", ls="--", lw=1.0, label="target 1/3")
    ax.set_xlabel("cycle")
    ax.set_ylabel("full_x log-chord slope")
    ax.set_title(f"N{NX}x{NY} slope vs cycle")
    ax.legend(frameon=True)
    fig.savefig(fig_slope)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.plot(slope_vs_cycle["cycle"], slope_vs_cycle["r2"], marker="o", ms=2.5, lw=1.2)
    ax.set_xlabel("cycle")
    ax.set_ylabel("R²")
    ax.set_title("Log-chord fit R² vs cycle")
    fig.savefig(fig_r2)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.6), constrained_layout=True, sharey=True)
    for ax, cycle in zip(axes.ravel(), SELECTED_PLOT_CYCLES):
        fit_row = slope_vs_cycle[slope_vs_cycle["cycle"] == cycle].iloc[0].to_dict()
        sub = entropy_curves[(entropy_curves["cycle"] == cycle) & (entropy_curves["Ay"] > 0)].sort_values("Ay")
        x_all = sub["log_sin_pi_Ay_over_Ny"].to_numpy(dtype=np.float64)
        y_all = sub["entropy_mean"].to_numpy(dtype=np.float64)
        yerr = sub["entropy_sem"].to_numpy(dtype=np.float64)
        x_line = np.linspace(float(np.min(x_all)), float(np.max(x_all)), 300)
        y_line = float(fit_row["slope"]) * x_line + float(fit_row["intercept"])
        ax.axvspan(float(fit_row["fit_x_min"]), float(fit_row["fit_x_max"]), color="0.5", alpha=0.18)
        ax.errorbar(x_all, y_all, yerr=yerr, marker="o", ms=3, lw=1.1, capsize=2)
        ax.plot(x_line, y_line, color="black", ls="--", lw=1.3,
                label=f"m={float(fit_row['slope']):.3f}")
        ax.set_title(f"cycle {cycle}")
        ax.set_xlabel(r"$\log[\sin(\pi A_y/N_y)]$")
        ax.set_ylabel(r"full_x $S(A_y)$")
        ax.legend(frameon=True, fontsize=8)
    fig.savefig(fig_selected)
    plt.close(fig)

    pivot = (
        entropy_curves.pivot_table(index="cycle", columns="Ay", values="entropy_mean", aggfunc="first")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    im = ax.imshow(
        pivot.to_numpy(dtype=np.float64), origin="lower", aspect="auto", cmap="Blues",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    ax.set_xlabel("Ay")
    ax.set_ylabel("cycle")
    ax.set_title("Full-x strip entropy S(Ay, cycle)")
    fig.colorbar(im, ax=ax, label="S(Ay)")
    fig.savefig(fig_heat)
    plt.close(fig)
    print(f"[saved] figures in {fig_root}")

    # ── Summary and manifest ──────────────────────────────────────────────────
    summary = {
        "status": "complete",
        "run_id": RUN_ID,
        "config": CAMPAIGN_CONFIG,
        "run_root": str(run_root),
        "entropy_curves": str(entropy_path),
        "slope_vs_cycle": str(fit_path),
        "full_checkpoint": str(full_checkpoint_path),
        "block_checkpoints": [str(p) for p in block_paths],
        "figures": [str(p) for p in [fig_slope, fig_r2, fig_selected, fig_heat]],
        "created_unix": time.time(),
    }
    write_json_atomic(summary_path, summary)

    manifest = {
        "campaign_name": CAMPAIGN_NAME,
        "campaign_config": CAMPAIGN_CONFIG,
        "helper_version": HELPER_VERSION,
        "output_root": str(args.output_dir),
        "run_summary": str(summary_path),
        "sample_block_size": SAMPLE_BLOCK_SIZE,
    }
    manifest_path = args.output_dir / "campaign_manifest.json"
    write_json_atomic(manifest_path, manifest)

    print(f"\n[complete] {RUN_ID}")
    print(f"  slope_vs_cycle: {fit_path}")
    print(f"  figures:        {fig_root}")
    print(f"  manifest:       {manifest_path}")


if __name__ == "__main__":
    main()
