#!/usr/bin/env python3
"""Validate flattened-ground-state aggregates and render manuscript figures.

This is a read-only analysis layer: it verifies the checksum-bound aggregate
NPZ files before loading them and never modifies production campaign products.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DOCS_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = DOCS_ROOT.parent
DEFAULT_SMALL_NPZ = (
    BUNDLE_ROOT
    / "results"
    / "flattened_ground_state_reference"
    / "flattened_ground_state_reference.npz"
)
DEFAULT_LARGE_NPZ = (
    BUNDLE_ROOT
    / "results"
    / "flattened_ground_state_large_ny"
    / "flattened_ground_state_large_ny.npz"
)
DEFAULT_OUTPUT_DIR = DOCS_ROOT / "figures"

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
WALL_LABELS = ("hard", "soft")
SMALL_NY = (20, 24, 28)
SMALL_WIDTHS = (5, 6, 7)
LARGE_NY = (40, 50, 60)
LARGE_WIDTHS = (10, 12, 15)
SHELL_LABELS = ("1", "2", "inf")
MI_NEGATIVE_TOLERANCE = 1.0e-8
DIAGNOSTIC_TOLERANCE = 1.0e-10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_aggregate(npz_path: Path) -> dict[str, Any]:
    npz_path = Path(npz_path).resolve()
    completion_path = npz_path.with_suffix(".complete.json")
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing aggregate NPZ: {npz_path}")
    if not completion_path.is_file():
        raise FileNotFoundError(f"missing completion JSON: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    product = completion.get("products", {}).get(npz_path.name)
    if not isinstance(product, dict):
        raise RuntimeError(
            f"completion JSON has no checksum record for {npz_path.name}"
        )
    actual_bytes = npz_path.stat().st_size
    if int(product.get("bytes", -1)) != actual_bytes:
        raise RuntimeError(
            f"byte-count mismatch for {npz_path.name}: "
            f"declared={product.get('bytes')}, actual={actual_bytes}"
        )
    actual_sha256 = sha256_file(npz_path)
    if product.get("sha256") != actual_sha256:
        raise RuntimeError(f"SHA-256 mismatch for {npz_path.name}")
    return {
        "path": str(npz_path),
        "completion_path": str(completion_path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "schema": completion.get("schema"),
    }


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _as_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(str(value) for value in np.asarray(values).tolist())


def _assert_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, found {actual!r}")


def _assert_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{name} contains nonfinite values")


def validate_small(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    _assert_equal("small wall labels", _as_strings(payload["wall_labels"]), WALL_LABELS)
    _assert_equal(
        "small Ny values",
        tuple(int(value) for value in payload["Ny_values"]),
        SMALL_NY,
    )
    _assert_equal(
        "small widths",
        tuple(int(value) for value in payload["width_values"]),
        SMALL_WIDTHS,
    )
    _assert_equal("small nshell", int(payload["nshell"]), 1)
    np.testing.assert_array_equal(payload["alpha_1_values"], ALPHA_VALUES)
    expected_shape = (2, 3, 21)
    for name in (
        "mutual_information_y0avg",
        "entropy_a_y0avg",
        "entropy_b_y0avg",
        "entropy_union_y0avg",
        "c_fit",
        "c_fit_se",
    ):
        _assert_equal(f"small {name} shape", payload[name].shape, expected_shape)
        _assert_finite(f"small {name}", payload[name])
    if float(np.min(payload["mutual_information_y0avg"])) < -MI_NEGATIVE_TOLERANCE:
        raise FloatingPointError("small mutual information is materially negative")
    for name in (
        "hamiltonian_hermiticity_max_abs",
        "projector_hermiticity_max_abs",
        "projector_idempotency_max_abs",
        "projector_y_translation_max_abs",
        "entropy_profile_y0_shift_max_abs",
        "mutual_information_y0_spread",
    ):
        if float(np.max(payload[name])) > DIAGNOSTIC_TOLERANCE:
            raise AssertionError(f"small diagnostic exceeds tolerance: {name}")
    if float(np.min(payload["restricted_occupation_min"])) < -MI_NEGATIVE_TOLERANCE:
        raise AssertionError("small restricted occupation spectrum is below zero")
    if float(np.max(payload["restricted_occupation_max"])) > 1 + MI_NEGATIVE_TOLERANCE:
        raise AssertionError("small restricted occupation spectrum exceeds one")
    return {
        "shape": list(expected_shape),
        "minimum_mutual_information": float(
            np.min(payload["mutual_information_y0avg"])
        ),
        "maximum_mutual_information": float(
            np.max(payload["mutual_information_y0avg"])
        ),
        "maximum_translation_spread": float(
            np.max(payload["mutual_information_y0_spread"])
        ),
    }


def validate_large(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    _assert_equal("large wall labels", _as_strings(payload["wall_labels"]), WALL_LABELS)
    _assert_equal(
        "large shell labels", _as_strings(payload["nshell_labels"]), SHELL_LABELS
    )
    _assert_equal(
        "large Ny values",
        tuple(int(value) for value in payload["Ny_values"]),
        LARGE_NY,
    )
    _assert_equal(
        "large widths",
        tuple(int(value) for value in payload["width_values"]),
        LARGE_WIDTHS,
    )
    np.testing.assert_array_equal(payload["alpha_1_values"], ALPHA_VALUES)
    expected_shape = (2, 3, 3, 21)
    for name in (
        "mutual_information_y0avg",
        "entropy_a_y0avg",
        "entropy_b_y0avg",
        "entropy_union_y0avg",
        "c_fit",
        "c_fit_se",
    ):
        _assert_equal(f"large {name} shape", payload[name].shape, expected_shape)
        _assert_finite(f"large {name}", payload[name])
    if float(np.min(payload["mutual_information_y0avg"])) < -MI_NEGATIVE_TOLERANCE:
        raise FloatingPointError("large mutual information is materially negative")
    for name in (
        "ow_y_translation_covariance_max_abs",
        "hamiltonian_block_hermiticity_max_abs",
        "projector_block_hermiticity_max_abs",
        "projector_idempotency_max_abs",
        "entropy_translation_max_abs",
        "mutual_information_y0_shift_abs",
    ):
        if float(np.max(payload[name])) > DIAGNOSTIC_TOLERANCE:
            raise AssertionError(f"large diagnostic exceeds tolerance: {name}")
    if float(np.min(payload["restricted_occupation_min"])) < -MI_NEGATIVE_TOLERANCE:
        raise AssertionError("large restricted occupation spectrum is below zero")
    if float(np.max(payload["restricted_occupation_max"])) > 1 + MI_NEGATIVE_TOLERANCE:
        raise AssertionError("large restricted occupation spectrum exceeds one")
    cft = np.asarray(payload["mutual_information_cft_c1_by_Ny"], dtype=float)
    expected_cft = np.asarray(
        [cft_mutual_information_reference(ny, width) for ny, width in zip(LARGE_NY, LARGE_WIDTHS)]
    )
    np.testing.assert_allclose(cft, expected_cft, rtol=0.0, atol=1.0e-14)
    mutual_information = np.asarray(payload["mutual_information_y0avg"], dtype=float)
    c_fit = np.asarray(payload["c_fit"], dtype=float)
    half_filling_gap = np.asarray(payload["half_filling_gap"], dtype=float)
    maximum_index = tuple(
        int(value) for value in np.unravel_index(np.argmax(mutual_information), expected_shape)
    )
    c_maximum_index = tuple(
        int(value) for value in np.unravel_index(np.argmax(c_fit), expected_shape)
    )
    minimum_gap_index = tuple(
        int(value) for value in np.unravel_index(np.argmin(half_filling_gap), expected_shape)
    )
    expected_maximum_index = (0, 2, 0, 11)
    expected_minimum_gap_index = (0, 2, 0, 20)
    _assert_equal("large MI maximum index", maximum_index, expected_maximum_index)
    _assert_equal("large c_fit maximum index", c_maximum_index, expected_maximum_index)
    _assert_equal(
        "large minimum half-filling-gap index",
        minimum_gap_index,
        expected_minimum_gap_index,
    )
    rank_mismatch = np.asarray(payload["negative_energy_count"]) != np.asarray(
        payload["half_filling_rank"]
    )
    mismatch_indices = tuple(
        tuple(int(value) for value in index)
        for index in np.argwhere(rank_mismatch)
    )
    _assert_equal(
        "large negative-level/rank mismatch indices",
        mismatch_indices,
        (expected_minimum_gap_index,),
    )
    return {
        "shape": list(expected_shape),
        "minimum_mutual_information": float(
            np.min(payload["mutual_information_y0avg"])
        ),
        "maximum_mutual_information": float(
            np.max(payload["mutual_information_y0avg"])
        ),
        "minimum_half_filling_gap": float(np.min(payload["half_filling_gap"])),
        "maximum_index": list(maximum_index),
        "maximum_alpha_1": float(ALPHA_VALUES[maximum_index[-1]]),
        "gap_at_maximum": float(half_filling_gap[maximum_index]),
        "minimum_gap_index": list(minimum_gap_index),
        "minimum_gap_alpha_1": float(ALPHA_VALUES[minimum_gap_index[-1]]),
        "rank_mismatch_indices": [list(index) for index in mismatch_indices],
        "cft_references": cft.tolist(),
    }


def cft_mutual_information_reference(ny: int, width: int, c_eff: float = 1.0) -> float:
    x = np.sin(np.pi * float(width) / float(ny)) ** 2
    return float(-(c_eff / 3.0) * np.log1p(-x))


def set_plot_defaults() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.9,
            "savefig.facecolor": "white",
        }
    )


def plot_styles() -> tuple[dict[str, Any], ...]:
    return (
        {"color": "#d62728", "marker": "^", "linestyle": ":"},
        {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
        {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    )


def save_figure(fig: Any, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")


def make_geometry_figure(output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

    set_plot_defaults()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.55))

    axis = axes[0]
    nx, ny, y0, width = 20, 40, 3, 10
    axis.add_patch(Rectangle((0, 0), nx, ny, color="0.91", zorder=0))
    axis.add_patch(Rectangle((5, 0), 11, ny, color="#b7e1b0", alpha=0.78, zorder=1))
    axis.add_patch(Rectangle((0, y0), nx, width, color="#d62728", alpha=0.38, zorder=2))
    axis.add_patch(
        Rectangle((0, y0 + ny // 2), nx, width, color="#1f77b4", alpha=0.38, zorder=2)
    )
    for x_position in (5, 16):
        axis.axvline(x_position, color="black", linestyle="--", linewidth=1.15, zorder=3)
    axis.text(10.5, 38.2, r"topological slab: $\alpha_1$", ha="center", va="top", fontsize=9)
    axis.text(2.5, 38.2, r"$\alpha_2=30$", ha="center", va="top", fontsize=8)
    axis.text(18.0, 38.2, r"$\alpha_2=30$", ha="center", va="top", fontsize=8)
    axis.text(1.0, y0 + width / 2, r"$a$", color="#8c1515", va="center", fontsize=15)
    axis.text(1.0, y0 + ny // 2 + width / 2, r"$b$", color="#174f83", va="center", fontsize=15)
    axis.text(5, -3.0, r"DW: $x=5$", ha="center", va="top", fontsize=8)
    axis.text(16, -3.0, r"DW: $x=16$", ha="center", va="top", fontsize=8)
    axis.add_patch(
        FancyArrowPatch(
            (21.1, y0 + width / 2),
            (21.1, y0 + ny // 2 + width / 2),
            arrowstyle="<->",
            mutation_scale=10,
            linewidth=1.0,
            clip_on=False,
        )
    )
    axis.text(21.6, y0 + ny // 4 + width / 2, r"$N_y/2$", rotation=90, va="center", fontsize=9)
    axis.annotate(
        "periodic $y$",
        xy=(-0.7, 39.2),
        xytext=(-0.7, 0.8),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9},
        rotation=90,
        va="center",
        ha="right",
        annotation_clip=False,
        fontsize=8,
    )
    axis.set_xlim(0, nx)
    axis.set_ylim(0, ny)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_xticks((0, 5, 10, 16, 20))
    axis.set_yticks((0, y0, y0 + width, y0 + ny // 2, y0 + ny // 2 + width, ny))
    axis.tick_params(direction="in", top=True, right=True)
    axis.set_title(r"full-$x$ strips on a spatial torus")

    axis = axes[1]
    radius = 1.0
    theta0 = 20.0
    interval_angle = 90.0
    axis.add_patch(Circle((0, 0), radius, fill=False, color="0.55", linewidth=2.0))
    axis.add_patch(
        Arc((0, 0), 2 * radius, 2 * radius, theta1=theta0, theta2=theta0 + interval_angle,
            color="#d62728", linewidth=8.0, capstyle="round")
    )
    axis.add_patch(
        Arc((0, 0), 2 * radius, 2 * radius, theta1=theta0 + 180.0,
            theta2=theta0 + 180.0 + interval_angle, color="#1f77b4",
            linewidth=8.0, capstyle="round")
    )

    def point(angle_degrees: float) -> np.ndarray:
        angle = np.deg2rad(angle_degrees)
        return np.asarray((np.cos(angle), np.sin(angle)))

    u1 = point(theta0)
    v1 = point(theta0 + interval_angle)
    u2 = point(theta0 + 180.0)
    v2 = point(theta0 + 180.0 + interval_angle)
    for label, point_xy in zip((r"$u_1$", r"$v_1$", r"$u_2$", r"$v_2$"), (u1, v1, u2, v2)):
        axis.plot(*point_xy, "o", color="black", markersize=4.0, zorder=5)
        axis.text(*(1.13 * point_xy), label, ha="center", va="center", fontsize=9)
    axis.plot((u1[0], v1[0]), (u1[1], v1[1]), color="#d62728", linestyle="--", linewidth=1.2)
    axis.plot((u1[0], u2[0]), (u1[1], u2[1]), color="0.25", linestyle=":", linewidth=1.2)
    axis.text(0.53, 0.55, r"$D(w)$", color="#8c1515", fontsize=9)
    axis.text(
        -0.18,
        -0.06,
        r"$D(w+d)$",
        rotation=20,
        color="0.2",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
    )
    axis.text(0.72, 0.84, r"$a$", color="#8c1515", fontsize=13)
    axis.text(-0.72, -0.84, r"$b$", color="#174f83", fontsize=13)
    axis.text(
        0.0,
        -1.30,
        r"$D(s)=\dfrac{N_y}{\pi}\sin\!\dfrac{\pi s}{N_y}$"
        "\n"
        r"$x=\dfrac{D(w)^2}{D(w+d)^2}=\sin^2\!\dfrac{\pi w}{N_y}$",
        ha="center",
        va="top",
        fontsize=9,
    )
    axis.set_xlim(-1.45, 1.45)
    axis.set_ylim(-1.56, 1.35)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title("circle cross-ratio", pad=5)

    for label, axis in zip(("(a)", "(b)"), axes):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontsize=11)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, output_dir, "geometry_and_cross_ratio")
    plt.close(fig)


def make_result_figure(
    *,
    alpha: np.ndarray,
    mutual_information: np.ndarray,
    c_fit: np.ndarray,
    ny_values: tuple[int, ...],
    widths: tuple[int, ...],
    title: str,
    output_dir: Path,
    stem: str,
) -> None:
    import matplotlib.pyplot as plt

    set_plot_defaults()
    order = np.argsort(alpha)
    alpha_display = np.asarray(alpha)[order]
    critical_indices = np.flatnonzero(np.isclose(alpha_display, 2.0, rtol=0.0, atol=1.0e-14))
    if critical_indices.size != 1:
        raise AssertionError("display grid must contain exactly one alpha_1=2 point")
    critical_index = int(critical_indices[0])
    mi = np.asarray(mutual_information)[..., order]
    c_values = np.asarray(c_fit)[..., order]
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 7.9), sharex=True)
    handles: list[Any] = []
    labels: list[str] = []
    for wall_index, wall in enumerate(WALL_LABELS):
        for ny_index, (ny, width, style) in enumerate(
            zip(ny_values, widths, plot_styles())
        ):
            line = axes[0, wall_index].plot(
                alpha_display,
                mi[wall_index, ny_index],
                linewidth=1.55,
                markersize=5.0,
                markeredgewidth=0.7,
                **style,
            )[0]
            axes[1, wall_index].plot(
                alpha_display,
                c_values[wall_index, ny_index],
                linewidth=1.55,
                markersize=5.0,
                markeredgewidth=0.7,
                **style,
            )
            for row, values in ((0, mi), (1, c_values)):
                axes[row, wall_index].plot(
                    [2.0],
                    [values[wall_index, ny_index, critical_index]],
                    linestyle="none",
                    marker=style["marker"],
                    markerfacecolor="white",
                    markeredgecolor=style["color"],
                    markeredgewidth=1.15,
                    markersize=6.3,
                    zorder=5,
                )
            axes[0, wall_index].axhline(
                cft_mutual_information_reference(ny, width),
                color=style["color"],
                linewidth=0.9,
                alpha=0.38,
            )
            if wall_index == 0:
                handles.append(line)
                labels.append(rf"$N_y={ny}$, $w={width}$")
        axes[0, wall_index].set_title(
            "hard / support-truncated" if wall == "hard" else "soft / untruncated"
        )
        axes[1, wall_index].set_xlabel(r"$\alpha_1$")
        axes[1, wall_index].axhline(1.0, color="0.50", linestyle=":", linewidth=1.0)
        for row in range(2):
            axis = axes[row, wall_index]
            axis.axvline(2.0, color="0.22", linestyle="--", linewidth=1.05)
            axis.tick_params(direction="in", top=True, right=True, length=5)
            axis.set_xlim(0.98, 3.02)
            for spine in axis.spines.values():
                spine.set_visible(True)
    axes[0, 0].set_ylabel(r"$I_{a,b}^{\rm flat}$")
    axes[1, 0].set_ylabel(r"$c_{\rm fit}=3m$")
    axes[0, 1].tick_params(labelleft=True)
    axes[1, 1].tick_params(labelleft=True)
    mi_top = max(0.25, 1.08 * float(np.max(mi)))
    c_top = max(1.10, 1.08 * float(np.max(c_values)))
    for axis in axes[0]:
        axis.set_ylim(-0.008, mi_top)
    for axis in axes[1]:
        axis.set_ylim(-0.035, c_top)
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        handlelength=2.8,
    )
    fig.suptitle(title, y=0.992, fontsize=12)
    for label, axis in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        axis.text(-0.13, 1.025, label, transform=axis.transAxes, fontsize=11)
    fig.text(
        0.5,
        0.012,
        r"Horizontal lines: $I_{a,b}^{\rm CFT}=-(1/3)\log[1-\sin^2(\pi w/N_y)]$; "
        r"vertical line: $\alpha_1=2$."
        "\n"
        r"Open markers at $\alpha_1=2$ denote the symmetric critical-grid prescription.",
        ha="center",
        fontsize=8.2,
    )
    fig.tight_layout(rect=(0.0, 0.050, 1.0, 0.94), h_pad=2.2, w_pad=1.6)
    save_figure(fig, output_dir, stem)
    plt.close(fig)


def render_figures(
    small: dict[str, np.ndarray], large: dict[str, np.ndarray], output_dir: Path
) -> list[str]:
    make_geometry_figure(output_dir)
    make_result_figure(
        alpha=small["alpha_1_values"],
        mutual_information=small["mutual_information_y0avg"],
        c_fit=small["c_fit"],
        ny_values=SMALL_NY,
        widths=SMALL_WIDTHS,
        title=r"Flattened ground state: $n_{\rm shell}=1$, small circumferences",
        output_dir=output_dir,
        stem="small_ny_nshell1",
    )
    stems = ["geometry_and_cross_ratio", "small_ny_nshell1"]
    for shell_index, shell_label in enumerate(SHELL_LABELS):
        stem_label = "inf" if shell_label == "inf" else shell_label
        stem = f"large_ny_nshell{stem_label}"
        display_label = r"\infty" if shell_label == "inf" else shell_label
        make_result_figure(
            alpha=large["alpha_1_values"],
            mutual_information=large["mutual_information_y0avg"][:, shell_index],
            c_fit=large["c_fit"][:, shell_index],
            ny_values=LARGE_NY,
            widths=LARGE_WIDTHS,
            title=rf"Flattened ground state: $n_{{\rm shell}}={display_label}$, large circumferences",
            output_dir=output_dir,
            stem=stem,
        )
        stems.append(stem)
    return stems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small-npz", type=Path, default=DEFAULT_SMALL_NPZ)
    parser.add_argument("--large-npz", type=Path, default=DEFAULT_LARGE_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verification = {
        "small": verify_aggregate(args.small_npz),
        "large": verify_aggregate(args.large_npz),
    }
    small = load_npz(args.small_npz)
    large = load_npz(args.large_npz)
    validation = {
        "small": validate_small(small),
        "large": validate_large(large),
    }
    exact_quarter = cft_mutual_information_reference(40, 10)
    ny50 = cft_mutual_information_reference(50, 12)
    if not np.isclose(exact_quarter, 0.23104906018664842, atol=1.0e-14):
        raise AssertionError("exact-quarter CFT reference changed")
    if not np.isclose(ny50, 0.21074972199116115, atol=1.0e-14):
        raise AssertionError("Ny=50 CFT reference changed")
    summary: dict[str, Any] = {
        "status": "verified",
        "verification": verification,
        "validation": validation,
        "cft_reference_exact_quarter": exact_quarter,
        "cft_reference_Ny50_w12": ny50,
    }
    if not args.check_only:
        stems = render_figures(small, large, args.output_dir)
        summary["figures"] = [
            str(Path(args.output_dir).resolve() / f"{stem}.{suffix}")
            for stem in stems
            for suffix in ("pdf", "png")
        ]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
