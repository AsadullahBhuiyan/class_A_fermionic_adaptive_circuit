#!/usr/bin/env python3
"""Sample-resolved endpoint CFT analysis for the independent hard-wall v2 run."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BUNDLE_ROOT = (
    ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "05_hard_wall_entropy_charge_batched_v2"
)
OUTPUT_ROOT = (
    BUNDLE_ROOT
    / "gpu_data"
    / "hard_wall_entropy_charge_nx20_ny30-60_s100_2ny_raster_v2_batched_endpoint"
)
DATA_DIR = HERE / "data"
FIGURE_DIR = HERE / "figures"
SAMPLE_CSV = DATA_DIR / "endpoint_sample_resolved_prefactors.csv"
SUMMARY_CSV = DATA_DIR / "endpoint_sample_resolved_summary.csv"
PDF_PATH = FIGURE_DIR / "endpoint_sample_resolved_prefactors.pdf"
PNG_PATH = FIGURE_DIR / "endpoint_sample_resolved_prefactors.png"
MANIFEST_PATH = HERE / "analysis_manifest.json"

NY_VALUES = (30, 35, 40, 45, 50, 55, 60)
LABELS = ("c1", "c2", "c3", "k")
DISPLAY = {"c1": r"$c_1$", "c2": r"$c_2$", "c3": r"$c_3$", "k": r"$k$"}
COLORS = {"c1": "#D92725", "c2": "#2CA02C", "c3": "#1F77B4", "k": "#6F4C9B"}
MARKERS = {"c1": "^", "c2": "s", "c3": "o", "k": "D"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle_analysis():
    name = "_endpoint_sample_resolved_verified_loader"
    spec = importlib.util.spec_from_file_location(name, BUNDLE_ROOT / "analyze_campaign.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load verified endpoint analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(BUNDLE_ROOT))
    try:
        spec.loader.exec_module(module)
        module._load_runner()
    finally:
        sys.path.remove(str(BUNDLE_ROOT))
    return module


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.8,
            "lines.markersize": 4.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def violin_panel(ax, values: dict[int, np.ndarray], label: str, letter: str) -> None:
    arrays = [values[ny][:, LABELS.index(label)] for ny in NY_VALUES]
    positions = np.arange(len(NY_VALUES))
    violin = ax.violinplot(
        arrays,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        bw_method=0.35,
    )
    for body in violin["bodies"]:
        body.set_facecolor(COLORS[label])
        body.set_edgecolor(COLORS[label])
        body.set_alpha(0.23)
    violin["cmedians"].set_color("0.25")
    violin["cmedians"].set_linewidth(0.8)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(positions, [str(ny) for ny in NY_VALUES])
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel(rf"sample-resolved {DISPLAY[label]}")
    ax.text(-0.13, 1.035, letter, transform=ax.transAxes, ha="left", va="bottom")


def make_figure(
    values: dict[int, np.ndarray],
    summaries: dict[int, dict[str, dict[str, float]]],
) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.05))

    ax = axes[0, 0]
    offsets = np.linspace(-0.75, 0.75, len(LABELS))
    for offset, label in zip(offsets, LABELS):
        means = np.array([summaries[ny][label]["mean"] for ny in NY_VALUES])
        sems = np.array([summaries[ny][label]["sem"] for ny in NY_VALUES])
        ax.errorbar(
            np.asarray(NY_VALUES) + offset,
            means,
            yerr=sems,
            color=COLORS[label],
            marker=MARKERS[label],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
            capsize=1.5,
            label=DISPLAY[label],
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel("trajectory-averaged prefactor")
    ax.set_ylim(1.00, 1.095)
    ax.legend(ncol=2, loc="upper right", columnspacing=0.8, handletextpad=0.3)
    ax.text(-0.13, 1.035, "(a)", transform=ax.transAxes, ha="left", va="bottom")

    violin_panel(axes[0, 1], values, "c1", "(b)")
    violin_panel(axes[1, 0], values, "k", "(c)")

    ax = axes[1, 1]
    for label, offset in zip(("c1", "c2", "c3"), (-0.55, 0.0, 0.55)):
        column = LABELS.index(label)
        differences = np.array(
            [values[ny][:, column].mean() - values[ny][:, LABELS.index("k")].mean() for ny in NY_VALUES]
        )
        sems = np.array(
            [
                (values[ny][:, column] - values[ny][:, LABELS.index("k")]).std(ddof=1)
                / np.sqrt(values[ny].shape[0])
                for ny in NY_VALUES
            ]
        )
        ax.errorbar(
            np.asarray(NY_VALUES) + offset,
            differences,
            yerr=sems,
            color=COLORS[label],
            marker=MARKERS[label],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=0.9,
            capsize=1.5,
            label=rf"{DISPLAY[label]}$-k$",
        )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel(r"$N_y$")
    ax.set_ylabel(r"paired $c_q-k$")
    ax.legend(loc="upper right", handletextpad=0.3)
    ax.text(-0.13, 1.035, "(d)", transform=ax.transAxes, ha="left", va="bottom")

    fig.suptitle(
        r"Independent hard-wall endpoint campaign, $N_x=20$, $S=100$, $t=2N_y$; error bars: SEM",
        fontsize=8.0,
        y=0.99,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.92, wspace=0.29, hspace=0.34)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_PATH)
    fig.savefig(PNG_PATH, dpi=300)
    plt.close(fig)


def main() -> None:
    analyzer = load_bundle_analysis()
    cases = analyzer.load_cases(analyzer.discover(OUTPUT_ROOT))
    values: dict[int, np.ndarray] = {}
    summaries: dict[int, dict[str, dict[str, float]]] = {}
    sample_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for ny in NY_VALUES:
        case = cases[ny]
        columns = []
        for label in LABELS:
            slopes, _ = analyzer.trajectory_slopes(
                case["ay_values"], case[analyzer.CURVE_KEYS[label]], ny
            )
            columns.append(analyzer.PREFACTOR[label] * slopes)
        resolved = np.column_stack(columns)
        values[ny] = resolved
        summaries[ny] = {}
        row: dict[str, float | int | str] = {
            "Nx": 20,
            "Ny": ny,
            "samples": 100,
            "endpoint_cycle": 2 * ny,
            "Ay_fit_min": 8,
            "Ay_fit_max": ny // 2,
        }
        for column, label in enumerate(LABELS):
            data = resolved[:, column]
            sem = float(data.std(ddof=1) / np.sqrt(data.size))
            summaries[ny][label] = {
                "mean": float(data.mean()),
                "sem": sem,
            }
            row.update(
                {
                    label: float(data.mean()),
                    f"{label}_median": float(np.median(data)),
                    f"{label}_sample_sd": float(data.std(ddof=1)),
                    f"{label}_sem": sem,
                }
            )
        for label in ("c1", "c2", "c3"):
            column = LABELS.index(label)
            delta = resolved[:, column] - resolved[:, LABELS.index("k")]
            row.update(
                {
                    f"{label}_minus_k": float(delta.mean()),
                    f"{label}_minus_k_sample_sd": float(delta.std(ddof=1)),
                    f"{label}_minus_k_sem": float(delta.std(ddof=1) / np.sqrt(delta.size)),
                }
            )
        summary_rows.append(row)
        for sample, sample_id in zip(resolved, case["sample_ids"]):
            sample_rows.append(
                {
                    "Nx": 20,
                    "Ny": ny,
                    "sample_id": int(sample_id),
                    "endpoint_cycle": 2 * ny,
                    "c1": float(sample[0]),
                    "c2": float(sample[1]),
                    "c3": float(sample[2]),
                    "k": float(sample[3]),
                    "c1_minus_k": float(sample[0] - sample[3]),
                    "c2_minus_k": float(sample[1] - sample[3]),
                    "c3_minus_k": float(sample[2] - sample[3]),
                }
            )

    write_csv(SAMPLE_CSV, sample_rows)
    write_csv(SUMMARY_CSV, summary_rows)
    make_figure(values, summaries)

    outputs = {}
    for label, path in {
        "sample_csv": SAMPLE_CSV,
        "summary_csv": SUMMARY_CSV,
        "figure_pdf": PDF_PATH,
        "figure_png": PNG_PATH,
    }.items():
        outputs[label] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    source_manifest = OUTPUT_ROOT / "DOWNLOAD_MANIFEST.json"
    manifest = {
        "schema": "endpoint_sample_resolved_entropy_charge_analysis_v1",
        "sampling_revision": analyzer.SAMPLING_REVISION,
        "separate_from_legacy_time_dependent_campaign": True,
        "verified_shards": 140,
        "samples_per_Ny": 100,
        "Ny_values": list(NY_VALUES),
        "endpoint": "t=2Ny",
        "fit_window": "Ay=8..Ny/2 inclusive",
        "resolved_estimators": {
            "c1": "3 times each trajectory's von Neumann log-chord slope",
            "c2": "4 times each trajectory's Renyi-2 log-chord slope",
            "c3": "9/2 times each trajectory's Renyi-3 log-chord slope",
            "k": "pi^2 times each trajectory's intrinsic charge-variance log-chord slope",
        },
        "uncertainty": {
            "unit": "one complete trajectory",
            "method": "sample-wise standard error SD/sqrt(100)",
            "paired_observables": True,
        },
        "input_download_manifest": {
            "path": str(source_manifest.relative_to(ROOT)),
            "bytes": source_manifest.stat().st_size,
            "sha256": sha256(source_manifest),
        },
        "outputs": outputs,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[saved] {PDF_PATH}")
    print(f"[saved] {PNG_PATH}")
    print(f"[saved] {SAMPLE_CSV}")
    print(f"[saved] {SUMMARY_CSV}")
    for row in summary_rows:
        print(
            f"Ny={row['Ny']}: c1={row['c1']:.6f} "
            f"+/- {row['c1_sem']:.6f} SEM, "
            f"k={row['k']:.6f} "
            f"+/- {row['k_sem']:.6f} SEM"
        )


if __name__ == "__main__":
    main()
