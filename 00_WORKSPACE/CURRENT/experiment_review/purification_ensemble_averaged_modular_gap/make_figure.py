#!/usr/bin/env python3
"""Single-column ensemble-averaged modular-gap figure.

The primary data are the completed hard- and soft-wall S=100 purification
campaigns at T=4 Ny.  The independent hard-wall T=2 Ny campaign is shown as a
depth-limited comparison because it supplies a denser circumference grid.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_ROOT = Path(__file__).resolve().parent
FIGURE_ROOT = ANALYSIS_ROOT / "figures"
TABLE_ROOT = ANALYSIS_ROOT / "tables"

PRIMARY_BUNDLE = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs/07_maxmix_hard_soft_purification"
)
COMPARISON_BUNDLE = (
    REPO_ROOT
    / "00_WORKSPACE/CURRENT/final_production_new_designs/04_maxmix_manybody_lyapunov_pilot"
)

SAMPLES = 100
OCCUPATION_TOLERANCE = 1.0e-9
MODULAR_CAP = math.log((1.0 - OCCUPATION_TOLERANCE) / OCCUPATION_TOLERANCE)


@dataclass(frozen=True)
class SeriesSpec:
    dataset: str
    construction: str
    depth_multiple: int
    ny_values: tuple[int, ...]
    root: Path
    revision: str
    completion_schema: str
    result_schema: str
    occupation_key: str
    sample_key: str
    cycle_key: str


SERIES = (
    SeriesSpec(
        dataset="primary_4Ny",
        construction="hard",
        depth_multiple=4,
        ny_values=(20, 30, 40),
        root=(
            PRIMARY_BUNDLE
            / "gpu_data/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2/hard"
        ),
        revision="maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v2",
        completion_schema="maxmix_purification_completion_v2",
        result_schema="maxmix_purification_result_v2",
        occupation_key="occupation_spectrum",
        sample_key="sample_indices",
        cycle_key="cycles",
    ),
    SeriesSpec(
        dataset="primary_4Ny",
        construction="soft",
        depth_multiple=4,
        ny_values=(20, 30, 40),
        root=(
            PRIMARY_BUNDLE
            / "gpu_data/maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3/soft"
        ),
        revision="maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v3",
        completion_schema="maxmix_purification_completion_v3",
        result_schema="maxmix_purification_result_v3",
        occupation_key="occupation_spectrum",
        sample_key="sample_indices",
        cycle_key="cycles",
    ),
    SeriesSpec(
        dataset="independent_hard_2Ny",
        construction="hard",
        depth_multiple=2,
        ny_values=(20, 22, 24, 26, 28, 30, 36, 40),
        root=(
            COMPARISON_BUNDLE
            / "gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/results"
        ),
        revision="maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2",
        completion_schema="maxmix_manybody_lyapunov_gpu_completion_v2",
        result_schema="maxmix_manybody_lyapunov_gpu_task_v2",
        occupation_key="occupations",
        sample_key="global_sample_indices",
        cycle_key="spectrum_cycles",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(data: Any, key: str) -> Any:
    value = np.asarray(data[key])
    return value.item() if value.shape == () else value


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1) / math.sqrt(values.size))


def modular_excitation_spectrum(occupations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ordered |epsilon| and whether each trajectory hits the cap.

    Exact numerical caps are retained at ``MODULAR_CAP``.  Consequently the
    resulting soft-wall mean is a lower bound when all modes of a trajectory
    are capped; no trajectories are silently discarded.
    """

    nu = np.asarray(occupations, dtype=np.float64)
    if nu.ndim != 2 or not np.all(np.isfinite(nu)):
        raise FloatingPointError("endpoint occupations must be a finite matrix")
    residual = max(0.0, float(-nu.min()), float(nu.max() - 1.0))
    if residual > OCCUPATION_TOLERANCE:
        raise FloatingPointError(
            f"occupation bound residual {residual:.3e} exceeds tolerance"
        )
    capped = (nu <= OCCUPATION_TOLERANCE) | (nu >= 1.0 - OCCUPATION_TOLERANCE)
    safe = np.clip(nu, OCCUPATION_TOLERANCE, 1.0 - OCCUPATION_TOLERANCE)
    epsilon = np.abs(np.log1p(-safe) - np.log(safe))
    ordered = np.sort(epsilon, axis=1)
    return ordered, np.all(capped, axis=1)


def _validate_completion(
    spec: SeriesSpec, ny: int, result_path: Path, completion_path: Path
) -> dict[str, Any]:
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "schema": spec.completion_schema,
        "sampling_revision": spec.revision,
        "Nx": 20,
        "Ny": ny,
        "cycles": spec.depth_multiple * ny,
    }
    if spec.dataset == "primary_4Ny":
        expected["construction"] = spec.construction
        expected["dtype"] = "complex128"
        expected["canonical_dynamics_entry_point"] = (
            "classA_U1FGTN_gpu.run_markov_circuit"
        )
    else:
        expected["canonical_entry_point"] = "classA_U1FGTN_gpu.run_markov_circuit"
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(f"{completion_path}: {key} mismatch")
    if completion.get("result_filename") != result_path.name:
        raise RuntimeError(f"{completion_path}: result filename mismatch")
    if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
        raise RuntimeError(f"{result_path}: byte-count mismatch")
    if completion.get("result_sha256") != sha256_file(result_path):
        raise RuntimeError(f"{result_path}: SHA-256 mismatch")
    return completion


def load_series(spec: SeriesSpec) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    verified_files = 0
    verified_bytes = 0
    common_config: str | None = None
    common_sources: dict[str, str] | None = None

    for ny in spec.ny_values:
        ny_root = spec.root / (f"Ny{ny:03d}" if spec.dataset == "primary_4Ny" else f"Ny{ny}")
        result_paths = sorted(ny_root.glob("*.npz"))
        if len(result_paths) != 20:
            raise RuntimeError(f"{spec.dataset}, {spec.construction}, Ny={ny}: expected 20 shards")
        sample_ids: list[int] = []
        endpoint_spectra: list[np.ndarray] = []
        all_capped: list[bool] = []

        for result_path in result_paths:
            completion_path = result_path.with_suffix(".complete.json")
            if not completion_path.is_file():
                raise RuntimeError(f"missing completion file: {completion_path}")
            completion = _validate_completion(spec, ny, result_path, completion_path)
            identity_key = "configuration_hash" if spec.dataset == "primary_4Ny" else "config_sha256"
            identity = str(completion[identity_key])
            source_hashes = dict(completion["source_hashes"])
            if common_config is None:
                common_config, common_sources = identity, source_hashes
            if identity != common_config or source_hashes != common_sources:
                raise RuntimeError(f"{completion_path}: campaign identity drift")

            with np.load(result_path, allow_pickle=False) as data:
                schema_key = "result_schema" if spec.dataset == "primary_4Ny" else "schema"
                if scalar(data, schema_key) != spec.result_schema:
                    raise RuntimeError(f"{result_path}: result schema mismatch")
                ids = np.asarray(data[spec.sample_key], dtype=np.int64)
                cycles = np.asarray(data[spec.cycle_key], dtype=np.int64)
                occupations = np.asarray(data[spec.occupation_key], dtype=np.float64)
            if ids.shape != (5,):
                raise RuntimeError(f"{result_path}: expected five sample IDs")
            completion_ids = completion.get(
                "sample_indices" if spec.dataset == "primary_4Ny" else "global_sample_indices"
            )
            if ids.tolist() != completion_ids:
                raise RuntimeError(f"{result_path}: sample identity mismatch")
            if cycles[-1] != spec.depth_multiple * ny:
                raise RuntimeError(f"{result_path}: endpoint cycle mismatch")
            if occupations.shape[0] != 5 or occupations.shape[1] != cycles.size:
                raise RuntimeError(f"{result_path}: occupation array shape mismatch")
            ordered, censored = modular_excitation_spectrum(occupations[:, -1, :])
            sample_ids.extend(ids.tolist())
            endpoint_spectra.append(ordered)
            all_capped.extend(censored.tolist())
            verified_files += 2
            verified_bytes += result_path.stat().st_size + completion_path.stat().st_size

        order = np.argsort(sample_ids)
        ids = np.asarray(sample_ids, dtype=np.int64)[order]
        if not np.array_equal(ids, np.arange(SAMPLES)):
            raise RuntimeError(
                f"{spec.dataset}, {spec.construction}, Ny={ny}: sample IDs are not 0..99"
            )
        spectrum = np.concatenate(endpoint_spectra, axis=0)[order]
        censored = np.asarray(all_capped, dtype=bool)[order]
        normalized = spectrum / float(spec.depth_multiple * ny)
        average_spectrum = normalized.mean(axis=0)
        sample_gaps = normalized[:, 0]
        rows.append(
            {
                "dataset": spec.dataset,
                "construction": spec.construction,
                "Nx": 20,
                "Ny": ny,
                "samples": SAMPLES,
                "endpoint_cycle": spec.depth_multiple * ny,
                "depth_multiple": spec.depth_multiple,
                "ensemble_spectrum_gap": float(average_spectrum[0]),
                "trajectory_sem": sem(sample_gaps),
                "all_modes_capped_count": int(censored.sum()),
                "all_modes_capped_fraction": float(censored.mean()),
                "resolution_limited": bool(np.any(censored)),
                "Ny_times_gap": float(ny * average_spectrum[0]),
            }
        )

    return rows, {
        "dataset": spec.dataset,
        "construction": spec.construction,
        "sampling_revision": spec.revision,
        "root": str(spec.root.relative_to(REPO_ROOT)),
        "verified_files": verified_files,
        "verified_bytes": verified_bytes,
        "configuration_hash": common_config,
        "source_hashes": common_sources,
    }


def fit_inverse_size(rows: list[dict[str, Any]]) -> tuple[float, float]:
    ny = np.asarray([row["Ny"] for row in rows], dtype=np.float64)
    gap = np.asarray([row["ensemble_spectrum_gap"] for row in rows], dtype=np.float64)
    inverse = 1.0 / ny
    coefficient = float(np.dot(inverse, gap) / np.dot(inverse, inverse))
    predicted = coefficient * inverse
    residual = float(np.sum((gap - predicted) ** 2))
    total = float(np.sum((gap - gap.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0.0 else math.nan
    return coefficient, r_squared


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Nimbus Roman", "Times New Roman", "Liberation Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": None,
        }
    )


def make_figure(rows: list[dict[str, Any]]) -> tuple[list[Path], list[dict[str, Any]]]:
    configure_plotting()
    styles = {
        ("primary_4Ny", "hard"): {
            "color": "#1F77B4",
            "marker": "o",
            "linestyle": "-",
            "label": r"hard, $T=4N_y$",
        },
        ("primary_4Ny", "soft"): {
            "color": "#D92725",
            "marker": "^",
            "linestyle": ":",
            "label": r"soft, $T=4N_y$ (lower bound)",
        },
        ("independent_hard_2Ny", "hard"): {
            "color": "0.45",
            "marker": "D",
            "linestyle": "--",
            "label": r"hard, $T=2N_y$ (independent)",
        },
    }
    fig, axis = plt.subplots(figsize=(3.375, 2.55))
    fit_rows: list[dict[str, Any]] = []
    for key in (
        ("primary_4Ny", "hard"),
        ("primary_4Ny", "soft"),
        ("independent_hard_2Ny", "hard"),
    ):
        selected = sorted(
            [row for row in rows if (row["dataset"], row["construction"]) == key],
            key=lambda row: row["Ny"],
        )
        style = styles[key]
        ny = np.asarray([row["Ny"] for row in selected], dtype=np.float64)
        gap = np.asarray([row["ensemble_spectrum_gap"] for row in selected], dtype=np.float64)
        error = np.asarray([row["trajectory_sem"] for row in selected], dtype=np.float64)
        coefficient, r_squared = fit_inverse_size(selected)
        label = style["label"] + rf", $A={coefficient:.2f}$"
        axis.errorbar(
            ny,
            gap,
            yerr=error,
            color=style["color"],
            marker=style["marker"],
            markerfacecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            capsize=2,
            label=label,
            zorder=3,
        )
        dense = np.linspace(float(ny.min()), float(ny.max()), 240)
        axis.plot(
            dense,
            coefficient / dense,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=0.9,
            zorder=2,
        )
        fit_rows.append(
            {
                "dataset": key[0],
                "construction": key[1],
                "fit_model": "gap=A/Ny through origin",
                "A": coefficient,
                "r_squared": r_squared,
                "points": len(selected),
            }
        )

    axis.set(
        xlabel=r"circumference $N_y$",
        ylabel=r"modular gap $\overline{\Delta}_{\rm sp}(T)$",
        xlim=(18.0, 42.0),
        ylim=(0.0, 0.255),
        xticks=(20, 25, 30, 35, 40),
    )
    axis.set_title(r"Purification modular gap, $N_x=20$, $S=100$", pad=5)
    axis.legend(loc="upper right", handlelength=2.2, borderaxespad=0.4)
    fig.subplots_adjust(left=0.18, right=0.975, bottom=0.18, top=0.91)

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    pdf = FIGURE_ROOT / "ensemble_averaged_modular_gap_vs_size.pdf"
    png = FIGURE_ROOT / "ensemble_averaged_modular_gap_vs_size.png"
    fig.savefig(pdf, bbox_inches=None)
    fig.savefig(png, dpi=300, bbox_inches=None)
    plt.close(fig)
    return [pdf, png], fit_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for spec in SERIES:
        series_rows, series_provenance = load_series(spec)
        rows.extend(series_rows)
        provenance.append(series_provenance)

    figure_paths, fit_rows = make_figure(rows)
    gap_table = TABLE_ROOT / "ensemble_averaged_modular_gap.csv"
    fit_table = TABLE_ROOT / "inverse_size_fits.csv"
    write_csv(gap_table, rows)
    write_csv(fit_table, fit_rows)
    outputs = [*figure_paths, gap_table, fit_table]

    manifest = {
        "analysis_schema": "purification_ensemble_averaged_modular_gap_v1",
        "estimator": {
            "modular_energy": "epsilon_a=log[(1-nu_a)/nu_a]",
            "trajectory_spectrum": "d_j^xi=sort_a |epsilon_a^xi|",
            "ensemble_spectrum": "dbar_j=(1/S) sum_xi d_j^xi",
            "plotted_gap": "dbar_1(T)/T",
            "uncertainty": "ordinary SEM of d_1^xi(T)/T over 100 independent trajectories; no bootstrap",
            "occupation_tolerance": OCCUPATION_TOLERANCE,
            "finite_cap": MODULAR_CAP,
            "alignment_reason": "absolute excitation order aligns spectra across trajectories with fluctuating particle number",
        },
        "interpretation": {
            "hard_4Ny": "resolved for every trajectory",
            "soft_4Ny": "lower bound because some endpoint trajectories have all modes capped at numerical precision",
            "hard_2Ny": "independent depth-limited comparison; never pooled with the 4Ny ensembles",
        },
        "input_provenance": provenance,
        "fit_rows": fit_rows,
        "outputs": [
            {
                "path": str(path.relative_to(ANALYSIS_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in outputs
        ],
    }
    manifest_path = ANALYSIS_ROOT / "analysis_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"rows": rows, "fits": fit_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
