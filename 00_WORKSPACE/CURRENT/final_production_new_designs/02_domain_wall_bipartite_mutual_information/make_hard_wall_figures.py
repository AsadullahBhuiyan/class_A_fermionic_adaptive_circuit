#!/usr/bin/env python3
"""Verify the completed hard-wall BMI lane and render analysis figures.

The production ``gpu_data`` tree is read-only.  Every downloaded NPZ and its
completion JSON are verified against ``DOWNLOAD_MANIFEST.json`` before any
trajectory enters the sample average.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BUNDLE_ROOT = Path(__file__).resolve().parent
REVISION = "domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_" "v2_batched_50-25-25"
DEFAULT_INPUT_ROOT = BUNDLE_ROOT / "gpu_data" / REVISION
DEFAULT_OUTPUT_DIR = BUNDLE_ROOT / "analysis_outputs" / "hard_wall_nshell1"

NX = 20
HARD_WALL_X_LEFT = 5
HARD_WALL_X_RIGHT = 15
NY_VALUES = (20, 24, 28)
WIDTHS = (5, 6, 7)
BATCH_SIZES = (50, 25, 25)
ALPHA_VALUES = np.asarray(
    (
        3.0,
        2.75,
        2.5,
        2.3,
        2.2,
        2.15,
        2.1,
        2.075,
        2.05,
        2.025,
        2.0,
        1.975,
        1.95,
        1.925,
        1.9,
        1.85,
        1.8,
        1.7,
        1.5,
        1.25,
        1.0,
    ),
    dtype=np.float64,
)
SAMPLES = 100
CFT_C1_QUARTER_STRIP_MI = float(np.log(2.0) / 3.0)
EXPECTED_CASES = 63
EXPECTED_TASKS = 210
EXPECTED_FILES = 420
CONFIG_SHA256 = "eafa0e1e437c01f03d1cf39a4fbc77da30d4aa3df4ea9e24e26bd3184209be90"
RESULT_SCHEMA = "domain_wall_bipartite_mutual_information_task_v2"
COMPLETION_SCHEMA = "domain_wall_bipartite_mutual_information_completion_v2"

TRAJECTORY_KEYS = (
    "mutual_information_y0avg",
    "entropy_a_y0avg",
    "entropy_b_y0avg",
    "entropy_union_y0avg",
)

INK = "#20252B"
MID_GRAY = "#707780"
LIGHT_GRAY = "#ECECF1"
TOPOLOGICAL = "#9BD0EA"
LEFT_WALL = "#8B1E2D"
RIGHT_WALL = "#164A7B"
STRIP_A = "#d62728"
STRIP_B = "#1f77b4"


def alpha_tag(alpha: float) -> str:
    text = f"{float(alpha):.3f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_relative_paths() -> set[str]:
    paths: set[str] = set()
    for ny, batch_size in zip(NY_VALUES, BATCH_SIZES):
        for alpha in ALPHA_VALUES:
            directory = (
                Path("results")
                / f"Ny{ny:02d}"
                / "wall-hard"
                / f"alpha1-{alpha_tag(float(alpha))}"
            )
            for batch_index, sample_start in enumerate(range(0, SAMPLES, batch_size)):
                sample_stop = sample_start + batch_size
                stem = (
                    f"macro_{batch_index:03d}_"
                    f"samples_{sample_start:03d}-{sample_stop - 1:03d}"
                )
                for suffix in (".npz", ".complete.json"):
                    paths.add((directory / f"{stem}{suffix}").as_posix())
    if len(paths) != EXPECTED_FILES:
        raise RuntimeError("internal expected-path table has the wrong size")
    return paths


def verify_download_manifest(input_root: Path) -> dict[str, Any]:
    input_root = input_root.resolve()
    manifest_path = input_root / "DOWNLOAD_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing download manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_scalars = {
        "campaign": REVISION,
        "scope": "completed hard-wall lane only",
        "cases": EXPECTED_CASES,
        "trajectories": EXPECTED_CASES * SAMPLES,
        "result_completion_pairs": EXPECTED_TASKS,
        "verified_result_completion_pairs": EXPECTED_TASKS,
        "verified_trajectories": EXPECTED_CASES * SAMPLES,
        "downloaded_files": EXPECTED_FILES,
        "configuration_sha256": CONFIG_SHA256,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"download manifest mismatch for {key}: "
                f"expected {expected!r}, found {manifest.get(key)!r}"
            )
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != EXPECTED_FILES:
        raise RuntimeError("download manifest has the wrong file table")
    expected_paths = expected_relative_paths()
    recorded_paths = {str(record.get("relative_path")) for record in records}
    if recorded_paths != expected_paths:
        raise RuntimeError("download manifest path coverage does not match hard lane")
    total_bytes = 0
    for record in records:
        relative = Path(str(record["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe manifest path: {relative}")
        path = input_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest file is missing: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != int(record.get("bytes", -1)):
            raise RuntimeError(f"byte-count mismatch: {path}")
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch: {path}")
        total_bytes += actual_bytes
    return {
        "manifest": str(manifest_path),
        "files": len(records),
        "bytes": total_bytes,
        "pairs": EXPECTED_TASKS,
        "cases": EXPECTED_CASES,
        "trajectories": EXPECTED_CASES * SAMPLES,
    }


def _scalar(archive: Any, key: str) -> Any:
    return np.asarray(archive[key]).item()


def load_verified_hard_wall(
    input_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    input_root = input_root.resolve()
    shape = (len(NY_VALUES), len(ALPHA_VALUES), SAMPLES)
    arrays = {key: np.full(shape, np.nan, dtype=np.float64) for key in TRAJECTORY_KEYS}
    fill_count = np.zeros(shape, dtype=np.uint8)
    origin = np.full(shape, "", dtype="U11")
    max_entropy_residual = 0.0
    max_hermiticity_error = 0.0
    minimum_occupation = np.inf
    maximum_occupation = -np.inf
    materially_negative = 0

    for ny_index, (ny, width, batch_size) in enumerate(
        zip(NY_VALUES, WIDTHS, BATCH_SIZES)
    ):
        for alpha_index, alpha in enumerate(ALPHA_VALUES):
            directory = (
                input_root
                / "results"
                / f"Ny{ny:02d}"
                / "wall-hard"
                / f"alpha1-{alpha_tag(float(alpha))}"
            )
            for batch_index, sample_start in enumerate(range(0, SAMPLES, batch_size)):
                sample_stop = sample_start + batch_size
                stem = (
                    f"macro_{batch_index:03d}_"
                    f"samples_{sample_start:03d}-{sample_stop - 1:03d}"
                )
                result_path = directory / f"{stem}.npz"
                completion_path = directory / f"{stem}.complete.json"
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                expected_completion = {
                    "schema": COMPLETION_SCHEMA,
                    "status": "complete",
                    "sampling_revision": REVISION,
                    "Nx": NX,
                    "Ny": ny,
                    "width": width,
                    "alpha_1": float(alpha),
                    "alpha_2": 30.0,
                    "wall": "hard",
                    "dw_truncation": True,
                    "nshell": 1,
                    "cycles": 2 * ny,
                    "batch_index": batch_index,
                    "sample_start": sample_start,
                    "sample_stop": sample_stop,
                    "sample_count": batch_size,
                    "config_sha256": CONFIG_SHA256,
                    "result_filename": result_path.name,
                    "result_bytes": result_path.stat().st_size,
                    "result_sha256": sha256_file(result_path),
                }
                for key, expected in expected_completion.items():
                    if completion.get(key) != expected:
                        raise RuntimeError(
                            f"completion mismatch for {key}: {completion_path}"
                        )
                expected_indices = np.arange(sample_start, sample_stop, dtype=np.int64)
                if completion.get("global_sample_indices") != expected_indices.tolist():
                    raise RuntimeError(
                        f"completion sample coverage mismatch: {completion_path}"
                    )

                with np.load(result_path, allow_pickle=False) as archive:
                    expected_result = {
                        "schema": RESULT_SCHEMA,
                        "sampling_revision": REVISION,
                        "Nx": NX,
                        "Ny": ny,
                        "width": width,
                        "alpha_1": float(alpha),
                        "alpha_2": 30.0,
                        "wall": "hard",
                        "dw_truncation": True,
                        "nshell": 1,
                        "endpoint_cycle": 2 * ny,
                        "batch_index": batch_index,
                        "sample_start": sample_start,
                        "sample_stop": sample_stop,
                    }
                    for key, expected in expected_result.items():
                        if _scalar(archive, key) != expected:
                            raise RuntimeError(
                                f"result mismatch for {key}: {result_path}"
                            )
                    indices = np.asarray(
                        archive["global_sample_indices"], dtype=np.int64
                    )
                    if not np.array_equal(indices, expected_indices):
                        raise RuntimeError(
                            f"result sample coverage mismatch: {result_path}"
                        )
                    target = (ny_index, alpha_index, indices)
                    for key in TRAJECTORY_KEYS:
                        values = np.asarray(archive[key], dtype=np.float64)
                        if (
                            values.shape != (batch_size,)
                            or not np.isfinite(values).all()
                        ):
                            raise RuntimeError(f"invalid {key}: {result_path}")
                        arrays[key][target] = values
                    origins = np.asarray(archive["trajectory_origin"])
                    if origins.shape != (batch_size,) or not np.all(
                        np.isin(origins, ("legacy_v1", "computed_v2"))
                    ):
                        raise RuntimeError(
                            f"invalid trajectory provenance: {result_path}"
                        )
                    origin[target] = origins
                    fill_count[target] += 1
                    max_entropy_residual = max(
                        max_entropy_residual,
                        float(_scalar(archive, "entropy_identity_max_abs_residual")),
                    )
                    max_hermiticity_error = max(
                        max_hermiticity_error,
                        float(
                            _scalar(archive, "full_covariance_max_hermiticity_error")
                        ),
                        float(_scalar(archive, "restricted_max_hermiticity_error")),
                    )
                    minimum_occupation = min(
                        minimum_occupation,
                        float(_scalar(archive, "restricted_occupation_eigenvalue_min")),
                    )
                    maximum_occupation = max(
                        maximum_occupation,
                        float(_scalar(archive, "restricted_occupation_eigenvalue_max")),
                    )
                    materially_negative += int(
                        _scalar(archive, "materially_negative_mi_count")
                    )

    if not np.all(fill_count == 1):
        raise RuntimeError("hard-wall sample coverage is not exactly one")
    for key, values in arrays.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"assembled {key} contains nonfinite values")
    entropy_residual = arrays["mutual_information_y0avg"] - (
        arrays["entropy_a_y0avg"]
        + arrays["entropy_b_y0avg"]
        - arrays["entropy_union_y0avg"]
    )
    if float(np.max(np.abs(entropy_residual))) > 1.0e-8:
        raise FloatingPointError("assembled entropy cancellation exceeds tolerance")
    if materially_negative:
        raise FloatingPointError(
            f"found {materially_negative} materially negative mutual informations"
        )
    origin_values, origin_counts = np.unique(origin, return_counts=True)
    validation = {
        "shape": list(shape),
        "minimum_mutual_information": float(np.min(arrays["mutual_information_y0avg"])),
        "maximum_mutual_information": float(np.max(arrays["mutual_information_y0avg"])),
        "maximum_entropy_identity_residual": max_entropy_residual,
        "maximum_hermiticity_error": max_hermiticity_error,
        "minimum_restricted_occupation": minimum_occupation,
        "maximum_restricted_occupation": maximum_occupation,
        "materially_negative_mi_count": materially_negative,
        "trajectory_origins": {
            str(value): int(count) for value, count in zip(origin_values, origin_counts)
        },
    }
    return {**arrays, "trajectory_origin": origin}, validation


def summarize(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mutual_information = np.asarray(data["mutual_information_y0avg"])
    return {
        "mean": mutual_information.mean(axis=-1),
        "sample_std": mutual_information.std(axis=-1, ddof=1),
        "sem": mutual_information.std(axis=-1, ddof=1) / np.sqrt(SAMPLES),
        "minimum": mutual_information.min(axis=-1),
        "maximum": mutual_information.max(axis=-1),
    }


def set_plot_defaults() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.85,
            "savefig.facecolor": "white",
        }
    )


def plot_styles() -> tuple[dict[str, Any], ...]:
    return (
        {"color": "#d62728", "marker": "^", "linestyle": ":"},
        {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
        {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    )


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    return [str(pdf.resolve()), str(png.resolve())]


def draw_sample_average_axis(
    axis: Any,
    summary: dict[str, np.ndarray],
    *,
    show_alpha_critical_reference: bool = True,
    show_run_annotation: bool = True,
) -> None:
    """Draw the verified hard-wall mean and SEM on an existing axis."""

    order = np.argsort(ALPHA_VALUES)
    alpha = ALPHA_VALUES[order]
    mean = np.asarray(summary["mean"])[:, order]
    sem = np.asarray(summary["sem"])[:, order]
    for ny_index, (ny, width, style) in enumerate(
        zip(NY_VALUES, WIDTHS, plot_styles())
    ):
        axis.errorbar(
            alpha,
            mean[ny_index],
            yerr=sem[ny_index],
            label=rf"$N_y={ny}$, $w={width}$",
            linewidth=1.35,
            markersize=3.8,
            markeredgewidth=0.55,
            elinewidth=0.65,
            capsize=1.4,
            capthick=0.65,
            **style,
        )
    axis.axhline(
        CFT_C1_QUARTER_STRIP_MI,
        color="#7F6F91",
        linestyle=(0, (4, 2.5)),
        linewidth=1.0,
        label=r"$\frac{\log 2}{3}$",
        zorder=0,
    )
    if show_alpha_critical_reference:
        axis.axvline(2.0, color="0.25", linestyle="--", linewidth=0.95)
    axis.set_xlim(0.98, 3.02)
    axis.set_ylim(bottom=-0.015)
    axis.set_xticks((1.0, 1.5, 2.0, 2.5, 3.0))
    axis.set_xlabel(r"$\alpha_1$")
    axis.set_ylabel(r"$\overline{I}_{a,b}$")
    axis.set_title(r"hard / support-truncated, $n_{\rm shell}=1$")
    axis.tick_params(direction="in", top=True, right=True, length=4)
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.legend(frameon=False, loc="upper right", handlelength=2.5)
    if show_run_annotation:
        axis.text(
            0.025,
            0.97,
            r"$S=100$; endpoint $2N_y$ cycles" "\n" r"error bars: $\pm1$ s.e.m.",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
        )


def make_sample_average_figure(
    summary: dict[str, np.ndarray], output_dir: Path
) -> list[str]:
    import matplotlib.pyplot as plt

    set_plot_defaults()
    fig, axis = plt.subplots(figsize=(3.375, 3.05))
    draw_sample_average_axis(axis, summary)
    fig.tight_layout(pad=0.6)
    paths = save_figure(fig, output_dir, "hard_wall_nshell1_sample_averaged_bmi")
    plt.close(fig)
    return paths


def draw_compact_geometry_axis(axis: Any) -> None:
    """Draw the BPJ-style opposite-strip geometry without auxiliary formulas."""

    from matplotlib.patches import Rectangle

    nx = 20.0
    x_left = float(HARD_WALL_X_LEFT)
    x_right = float(HARD_WALL_X_RIGHT)
    y0 = 0.08
    width = 0.25
    opposite_start = y0 + 0.5
    axis.add_patch(
        Rectangle((0, 0), nx, 1, facecolor=LIGHT_GRAY, edgecolor=MID_GRAY, lw=0.65)
    )
    axis.add_patch(
        Rectangle(
            (x_left, 0),
            x_right - x_left,
            1,
            facecolor=TOPOLOGICAL,
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (0, y0),
            nx,
            width,
            facecolor=STRIP_A,
            edgecolor=STRIP_A,
            alpha=0.36,
            lw=0.8,
        )
    )
    axis.add_patch(
        Rectangle(
            (0, opposite_start),
            nx,
            width,
            facecolor=STRIP_B,
            edgecolor=STRIP_B,
            alpha=0.36,
            lw=0.8,
        )
    )
    for wall_x, wall_label in ((x_left, r"$x_L$"), (x_right, r"$x_R$")):
        axis.plot(
            (wall_x, wall_x),
            (0, 1),
            color="#173f5f",
            linewidth=1.0,
            solid_capstyle="butt",
            zorder=5,
        )
        axis.text(
            wall_x,
            1.035,
            wall_label,
            ha="center",
            va="bottom",
            color="#173f5f",
            fontsize=5.8,
        )
    axis.text(0.65, y0 + width / 2, r"$a$", color="#8c1515", fontsize=7.5, va="center")
    axis.text(
        0.65,
        opposite_start + width / 2,
        r"$b$",
        color="#174f83",
        fontsize=7.5,
        va="center",
    )
    axis.text(
        x_left / 2,
        0.455,
        "triv.",
        color=INK,
        fontsize=4.3,
        ha="center",
        va="center",
    )
    axis.text(
        (x_left + x_right) / 2,
        0.455,
        "top.",
        color=INK,
        fontsize=4.8,
        ha="center",
        va="center",
    )
    axis.text(
        (x_right + nx) / 2,
        0.455,
        "triv.",
        color=INK,
        fontsize=4.3,
        ha="center",
        va="center",
    )
    axis.set_xlim(-0.15, nx + 0.15)
    axis.set_ylim(-0.04, 1.10)
    axis.axis("off")


def make_report_inset_figure(
    summary: dict[str, np.ndarray], output_dir: Path
) -> list[str]:
    """Render the report MI panel with a compact lower-left geometry inset."""

    import matplotlib.pyplot as plt

    set_plot_defaults()
    fig, data_axis = plt.subplots(figsize=(3.60, 3.05))
    draw_sample_average_axis(
        data_axis,
        summary,
        show_alpha_critical_reference=False,
        show_run_annotation=False,
    )
    geometry_axis = data_axis.inset_axes((0.055, 0.030, 0.21, 0.29), zorder=6)
    geometry_axis.set_facecolor("white")
    geometry_axis.patch.set_alpha(1.0)
    draw_compact_geometry_axis(geometry_axis)
    fig.tight_layout(pad=0.6)
    paths = save_figure(fig, output_dir, "hard_wall_nshell1_bmi_with_geometry")
    plt.close(fig)
    return paths


def make_geometry_figure(output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    set_plot_defaults()
    fig, axis = plt.subplots(figsize=(4.65, 2.85))
    nx = 20.0
    y0 = 0.08
    width = 0.25
    opposite_start = y0 + 0.5
    axis.add_patch(
        Rectangle((0, 0), nx, 1, facecolor=LIGHT_GRAY, edgecolor=MID_GRAY, lw=1.0)
    )
    axis.add_patch(
        Rectangle(
            (HARD_WALL_X_LEFT, 0),
            HARD_WALL_X_RIGHT - HARD_WALL_X_LEFT,
            1,
            facecolor=TOPOLOGICAL,
            edgecolor="none",
        )
    )
    axis.add_patch(
        Rectangle(
            (0, y0), nx, width, facecolor=STRIP_A, edgecolor=STRIP_A, alpha=0.38, lw=1.0
        )
    )
    axis.add_patch(
        Rectangle(
            (0, opposite_start),
            nx,
            width,
            facecolor=STRIP_B,
            edgecolor=STRIP_B,
            alpha=0.38,
            lw=1.0,
        )
    )
    for wall_x, wall_label in (
        (HARD_WALL_X_LEFT, r"$x_L=5$"),
        (HARD_WALL_X_RIGHT, r"$x_R=15$"),
    ):
        axis.axvline(wall_x, color="#173f5f", linewidth=1.8, zorder=5)
        axis.text(
            wall_x,
            1.025,
            wall_label,
            ha="center",
            va="bottom",
            color="#173f5f",
            fontsize=7,
        )
    axis.text(10.0, 0.94, r"$\alpha_1$", ha="center", va="top", fontsize=9)
    axis.text(2.5, 0.94, r"$\alpha_2=30$", ha="center", va="top", fontsize=7.2)
    axis.text(17.5, 0.94, r"$\alpha_2=30$", ha="center", va="top", fontsize=7.2)
    axis.text(0.65, y0 + width / 2, r"$a$", color="#8c1515", fontsize=14, va="center")
    axis.text(
        0.65,
        opposite_start + width / 2,
        r"$b$",
        color="#174f83",
        fontsize=14,
        va="center",
    )
    axis.add_patch(
        FancyArrowPatch(
            (20.45, y0 + width / 2),
            (20.45, opposite_start + width / 2),
            arrowstyle="<->",
            mutation_scale=8,
            linewidth=0.9,
            color=INK,
            clip_on=False,
        )
    )
    axis.text(20.72, y0 + width / 2 + 0.25, r"$N_y/2$", rotation=90, va="center")
    axis.set_xlim(0, nx)
    axis.set_ylim(0, 1)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"periodic $y$")
    axis.set_xticks((0, HARD_WALL_X_LEFT, HARD_WALL_X_RIGHT, 20))
    axis.set_yticks((0, 1), (r"$0$", r"$N_y$"))
    axis.tick_params(direction="in", top=True, right=True, length=4)
    axis.set_box_aspect(0.57)
    fig.subplots_adjust(left=0.12, right=0.93, bottom=0.19, top=0.84)
    paths = save_figure(fig, output_dir, "domain_wall_bipartite_mi_geometry")
    plt.close(fig)
    return paths


def write_summary_csv(summary: dict[str, np.ndarray], output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hard_wall_nshell1_sample_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "wall",
                "nshell",
                "Nx",
                "Ny",
                "width",
                "alpha_1",
                "cycles",
                "samples",
                "mutual_information_mean",
                "mutual_information_sample_std",
                "mutual_information_sem",
                "mutual_information_min",
                "mutual_information_max",
            )
        )
        for ny_index, (ny, width) in enumerate(zip(NY_VALUES, WIDTHS)):
            for alpha_index, alpha in enumerate(ALPHA_VALUES):
                writer.writerow(
                    (
                        "hard",
                        1,
                        NX,
                        ny,
                        width,
                        f"{float(alpha):.12g}",
                        2 * ny,
                        SAMPLES,
                        f"{summary['mean'][ny_index, alpha_index]:.17g}",
                        f"{summary['sample_std'][ny_index, alpha_index]:.17g}",
                        f"{summary['sem'][ny_index, alpha_index]:.17g}",
                        f"{summary['minimum'][ny_index, alpha_index]:.17g}",
                        f"{summary['maximum'][ny_index, alpha_index]:.17g}",
                    )
                )
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = verify_download_manifest(args.input_root)
    data, validation = load_verified_hard_wall(args.input_root)
    summary = summarize(data)
    maximum_mean_index = tuple(
        int(value)
        for value in np.unravel_index(np.argmax(summary["mean"]), summary["mean"].shape)
    )
    report: dict[str, Any] = {
        "status": "verified",
        "input_root": str(args.input_root.resolve()),
        "manifest": manifest,
        "validation": validation,
        "sample_mean_shape": list(summary["mean"].shape),
        "maximum_sample_mean": float(summary["mean"][maximum_mean_index]),
        "maximum_sample_mean_Ny": NY_VALUES[maximum_mean_index[0]],
        "maximum_sample_mean_alpha_1": float(ALPHA_VALUES[maximum_mean_index[1]]),
    }
    if not args.check_only:
        figures = []
        figures.extend(make_sample_average_figure(summary, args.output_dir))
        figures.extend(make_geometry_figure(args.output_dir))
        report["figures"] = figures
        report["summary_csv"] = write_summary_csv(summary, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
