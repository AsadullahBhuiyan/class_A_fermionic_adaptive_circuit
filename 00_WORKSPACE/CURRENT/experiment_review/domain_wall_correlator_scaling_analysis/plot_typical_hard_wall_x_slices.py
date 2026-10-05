#!/usr/bin/env python3
"""Plot representative x-resolved correlators from the completed hard-wall S100 run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs"
    / "08_domain_wall_correlator_scaling/gpu_data"
    / "domain_wall_correlator_nx20_ny24-32_a1-1-3_nsh1-2-dense_s100_2ny_raster_v1"
    / "results/hard/Ny032/alpha1_1/nshell_1"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
STEM = "typical_hard_wall_x_resolved_correlator"
FIT_MIN = 2
FIT_MAX = 8
X_SLICES = (2, 4, 5, 6, 10)
X_LABELS = {
    2: r"$x=2$ (trivial bulk)",
    4: r"$x=4$ (trivial side)",
    5: r"$x=x_L=5$ (interface)",
    6: r"$x=6$ (topological side)",
    10: r"$x=10$ (topological bulk)",
}
COLORS = {
    2: "#7A7A7A",
    4: "#000000",
    5: "#0072B2",
    6: "#D55E00",
    10: "#009E73",
}
MARKERS = {2: "o", 4: "^", 5: "o", 6: "s", 10: "D"}
LINESTYLES = {2: ":", 4: "--", 5: "-", 6: "--", 10: "-."}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trajectories() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    files = sorted(DATA_DIR.glob("batch_*.npz"))
    if len(files) != 4:
        raise RuntimeError(f"Expected four hard-wall batches, found {len(files)}")

    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "global_sample_indices",
                "cycles",
                "ry_values",
                "x_values",
                "x_resolved_square_correlator",
                "xavg_square_correlator_vs_ry",
                "global_charge",
                "dw_location",
            }
            missing = required.difference(archive.files)
            if missing:
                raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
            cycles = np.asarray(archive["cycles"], dtype=np.int64)
            ry = np.asarray(archive["ry_values"], dtype=np.int64)
            xs = np.asarray(archive["x_values"], dtype=np.int64)
            x_resolved = np.asarray(
                archive["x_resolved_square_correlator"], dtype=np.float64
            )
            xavg = np.asarray(
                archive["xavg_square_correlator_vs_ry"], dtype=np.float64
            )
            charges = np.asarray(archive["global_charge"], dtype=np.int64)
            sample_ids = np.asarray(archive["global_sample_indices"], dtype=np.int64)
            dw_location = np.asarray(archive["dw_location"], dtype=np.int64)
            if not np.array_equal(dw_location, np.asarray([5, 15])):
                raise RuntimeError(f"Unexpected walls in {path}: {dw_location.tolist()}")
            if cycles[-1] != 64 or x_resolved.shape[1:] != (65, 20, 17):
                raise RuntimeError(f"Unexpected production shape in {path}")
            if not np.allclose(xavg, x_resolved.mean(axis=2), rtol=2e-14, atol=1e-15):
                raise RuntimeError(f"x-average identity failed for {path}")
            for local, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "sample_index": int(sample_id),
                        "cycle": int(cycles[-1]),
                        "ry": ry.copy(),
                        "x_values": xs.copy(),
                        "x_resolved": x_resolved[local, -1].copy(),
                        "xavg": xavg[local, -1].copy(),
                        "global_charge": int(charges[local, -1]),
                    }
                )
        provenance.append(
            {"path": str(path.relative_to(REPO_ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        )

    rows.sort(key=lambda row: int(row["sample_index"]))
    if [int(row["sample_index"]) for row in rows] != list(range(100)):
        raise RuntimeError("Expected exactly the 100 global sample indices 0,...,99")
    return rows, provenance


def fit_decay_exponent(row: dict[str, object]) -> float:
    ry = np.asarray(row["ry"], dtype=np.float64)
    curve = np.asarray(row["xavg"], dtype=np.float64)
    mask = (ry >= FIT_MIN) & (ry <= FIT_MAX)
    if np.any(curve[mask] <= 0.0):
        raise RuntimeError("Cannot fit a nonpositive correlator")
    slope, _ = np.polyfit(np.log(ry[mask]), np.log(curve[mask]), deg=1)
    return float(-slope)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "savefig.bbox": "tight",
        }
    )


def make_figure(typical: dict[str, object], median_beta: float, typical_beta: float) -> None:
    configure_matplotlib()
    figure, (axis, floor_axis) = plt.subplots(
        2,
        1,
        figsize=(3.375, 3.25),
        sharex=True,
        gridspec_kw={"height_ratios": [4.1, 1.0], "hspace": 0.08},
    )
    ry = np.asarray(typical["ry"], dtype=np.int64)[1:]
    x_resolved = np.asarray(typical["x_resolved"], dtype=np.float64)

    handles = []
    for x in X_SLICES:
        (line,) = axis.plot(
            ry,
            x_resolved[x, 1:],
            color=COLORS[x],
            marker=MARKERS[x],
            linestyle=LINESTYLES[x],
            linewidth=1.05,
            markersize=3.2,
            markerfacecolor="white" if x in (2, 4, 5) else COLORS[x],
            markeredgewidth=0.8,
            label=X_LABELS[x],
            clip_on=True,
        )
        handles.append(line)
        floor_axis.plot(
            ry,
            x_resolved[x, 1:],
            color=COLORS[x],
            marker=MARKERS[x],
            linestyle=LINESTYLES[x],
            linewidth=1.0,
            markersize=2.8,
            markerfacecolor="white" if x in (2, 4, 5) else COLORS[x],
            markeredgewidth=0.7,
            clip_on=True,
        )

    fit_mask = (ry >= FIT_MIN) & (ry <= FIT_MAX)
    wall_curve = x_resolved[5, 1:]
    guide_amplitude = float(np.exp(np.mean(np.log(wall_curve[fit_mask] * ry[fit_mask] ** 2))))
    guide_x = np.linspace(FIT_MIN, FIT_MAX, 100)
    axis.plot(
        guide_x,
        guide_amplitude * guide_x ** -2,
        color="#666666",
        linewidth=0.9,
        linestyle=(0, (2.5, 1.8)),
        zorder=0,
    )
    axis.text(2.25, guide_amplitude * 2.25 ** -2 * 0.58, r"$r_y^{-2}$", color="#555555")

    axis.set_yscale("log")
    floor_axis.set_yscale("log")
    axis.set_xlim(0.5, 16.5)
    axis.set_ylim(1e-12, 1.2e-1)
    floor_axis.set_ylim(3e-33, 8e-31)
    floor_axis.set_xticks([1, 4, 8, 12, 16])
    floor_axis.set_xlabel(r"separation $r_y$")
    figure.supylabel(r"squared correlator $G_x(r_y)$", x=0.005, fontsize=8.5)
    axis.set_title(
        rf"hard wall, $20\times32$, $t=2N_y$; typical trajectory {int(typical['sample_index'])}"
    )
    axis.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        ncol=1,
        handlelength=2.3,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.25,
    )
    floor_axis.text(
        0.02,
        0.14,
        "trivial-region numerical floor",
        transform=floor_axis.transAxes,
        fontsize=6.5,
        color="#444444",
    )

    axis.spines["bottom"].set_visible(False)
    floor_axis.spines["top"].set_visible(False)
    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    floor_axis.tick_params(axis="x", top=False)
    diagonal = 0.012
    kwargs = dict(color="black", clip_on=False, linewidth=0.8)
    axis.plot((-diagonal, +diagonal), (-diagonal, +diagonal), transform=axis.transAxes, **kwargs)
    axis.plot((1 - diagonal, 1 + diagonal), (-diagonal, +diagonal), transform=axis.transAxes, **kwargs)
    floor_axis.plot((-diagonal, +diagonal), (1 - diagonal, 1 + diagonal), transform=floor_axis.transAxes, **kwargs)
    floor_axis.plot((1 - diagonal, 1 + diagonal), (1 - diagonal, 1 + diagonal), transform=floor_axis.transAxes, **kwargs)

    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.13, top=0.92)
    metadata = {
        "Title": "Typical trajectory-resolved hard-wall correlator across x",
        "Author": "class_A_fermionic_adaptive_circuit analysis",
        "Subject": (
            f"Sample {int(typical['sample_index'])}; x-averaged beta={typical_beta:.8f}; "
            f"ensemble median beta={median_beta:.8f}"
        ),
    }
    figure.savefig(OUTPUT_DIR / f"{STEM}.pdf", metadata=metadata)
    figure.savefig(OUTPUT_DIR / f"{STEM}.png", dpi=300, metadata=metadata)
    plt.close(figure)


def write_source_data(typical: dict[str, object]) -> None:
    ry = np.asarray(typical["ry"], dtype=np.int64)
    x_resolved = np.asarray(typical["x_resolved"], dtype=np.float64)
    with (OUTPUT_DIR / f"{STEM}_source_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_index", "cycle", "x", "region", "r_y", "squared_correlator"],
        )
        writer.writeheader()
        for x in X_SLICES:
            for index in range(1, ry.size):
                writer.writerow(
                    {
                        "sample_index": int(typical["sample_index"]),
                        "cycle": int(typical["cycle"]),
                        "x": x,
                        "region": X_LABELS[x].replace("$", ""),
                        "r_y": int(ry[index]),
                        "squared_correlator": f"{x_resolved[x, index]:.17g}",
                    }
                )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, provenance = load_trajectories()
    betas = np.asarray([fit_decay_exponent(row) for row in rows], dtype=np.float64)
    median_beta = float(np.median(betas))
    typical_position = int(np.argmin(np.abs(betas - median_beta)))
    typical = rows[typical_position]
    typical_beta = float(betas[typical_position])

    make_figure(typical, median_beta, typical_beta)
    write_source_data(typical)
    summary = {
        "schema": "typical_hard_wall_x_resolved_correlator_v1",
        "selection": {
            "rule": "minimum absolute distance from the ensemble-median trajectory-level x-averaged decay exponent",
            "fit_model": "log G_xavg = intercept - beta log(r_y)",
            "fit_window": [FIT_MIN, FIT_MAX],
            "ensemble_size": len(rows),
            "ensemble_median_beta": median_beta,
            "selected_sample_index": int(typical["sample_index"]),
            "selected_beta": typical_beta,
        },
        "contract": {
            "construction": "hard/support-terminated",
            "Nx": 20,
            "Ny": 32,
            "alpha_1": 1,
            "alpha_2": 30,
            "nshell": 1,
            "cycle": int(typical["cycle"]),
            "initialization": "pure/default",
            "sequence": "raster_y",
            "perfect_correction": True,
            "dtype": "complex128",
            "dw_location": [5, 15],
            "selected_global_charge": int(typical["global_charge"]),
        },
        "x_slices": [
            {"x": x, "label": X_LABELS[x].replace("$", "")} for x in X_SLICES
        ],
        "input_files": provenance,
        "outputs": [f"{STEM}.pdf", f"{STEM}.png", f"{STEM}_source_data.csv"],
    }
    (OUTPUT_DIR / f"{STEM}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["selection"], indent=2))


if __name__ == "__main__":
    main()
