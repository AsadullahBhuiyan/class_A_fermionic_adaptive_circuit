#!/usr/bin/env python3
"""Reconstruct finite-envelope OW form factors in the technical-report convention.

The calculation follows the projector-Fourier reconstruction used by the legacy
``windowed_chern`` note and the audited legacy-evidence atlas, using the same
Hamiltonian convention as those sources and ``technical_report.tex``:

    n(k) = (sin(k_x), sin(k_y), alpha - cos(k_x) - cos(k_y)).

For positive alpha the lower-band transition is at the Gamma point.  Integer
``w`` denotes the square envelope
``|r_x|, |r_y| <= w``; the legacy half-shell cross is deliberately excluded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit, minimize_scalar


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 1.0
FOURIER_GRID = 1024
MAP_GRID = 241
WINDOW_WIDTHS = np.arange(1, 13, dtype=int)
DISPLAY_WIDTHS = (1, 2, 4, 6, 8)

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY = np.eye(2, dtype=np.complex128)
TAU_A = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
TAU_B = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
TRIAL_SPINORS = (TAU_A, TAU_B)

BPJ_RED = "#D55E4A"
BPJ_GREEN = "#3A9D5D"
BPJ_BLUE = "#2878B5"
BPJ_GOLD = "#D19A2A"
BPJ_BLACK = "#202020"
BPJ_PURPLE = "#8E5AA9"
BPJ_GRAY = "#777777"


def configure_plotting() -> None:
    """Apply the repository's compact single-/double-column plotting grammar."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.14,
        1.06,
        rf"$\mathbf{{({label})}}$",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        clip_on=False,
    )


def band_projector(
    kx: np.ndarray | float,
    ky: np.ndarray | float,
    alpha: float,
    band: int = -1,
) -> np.ndarray:
    """Return P_band for the Hamiltonian convention in technical_report.tex."""

    kx_array, ky_array = np.broadcast_arrays(
        np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)
    )
    nx = np.sin(kx_array)
    ny = np.sin(ky_array)
    nz = alpha - np.cos(kx_array) - np.cos(ky_array)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    if np.any(norm == 0.0):
        raise ValueError("The band projector is undefined at a gap closing.")
    h_hat = (
        nx[..., None, None] * SIGMA_X
        + ny[..., None, None] * SIGMA_Y
        + nz[..., None, None] * SIGMA_Z
    ) / norm[..., None, None]
    return 0.5 * (IDENTITY + float(band) * h_hat)


FOURIER_K = 2.0 * np.pi * np.arange(FOURIER_GRID) / FOURIER_GRID
FOURIER_KX, FOURIER_KY = np.meshgrid(FOURIER_K, FOURIER_K, indexing="ij")


def projector_fourier_coefficients(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return coefficient arrays C_{-,r} and C_{+,r} on the FFT torus."""

    projector_minus = band_projector(FOURIER_KX, FOURIER_KY, alpha, band=-1)
    coefficient_minus = np.fft.ifft2(projector_minus, axes=(0, 1))
    coefficient_plus = -coefficient_minus
    coefficient_plus[0, 0] += IDENTITY
    return coefficient_minus, coefficient_plus


def support_positions(width: int) -> list[tuple[int, int]]:
    return [
        (rx, ry)
        for rx in range(-int(width), int(width) + 1)
        for ry in range(-int(width), int(width) + 1)
    ]


def coefficient_vector(
    coefficients: np.ndarray, trial: np.ndarray, rx: int, ry: int
) -> np.ndarray:
    return coefficients[rx % FOURIER_GRID, ry % FOURIER_GRID] @ trial


def support_weight(
    coefficients: np.ndarray, trial: np.ndarray, width: int
) -> float:
    return float(
        sum(
            np.vdot(vector, vector).real
            for rx, ry in support_positions(width)
            for vector in [coefficient_vector(coefficients, trial, rx, ry)]
        )
    )


def total_weight(coefficients: np.ndarray, trial: np.ndarray) -> float:
    vectors = np.einsum("...ab,b->...a", coefficients, trial, optimize=True)
    return float(np.square(np.abs(vectors)).sum())


def truncated_spinor(
    kx: np.ndarray | float,
    ky: np.ndarray | float,
    coefficients: np.ndarray,
    trial: np.ndarray,
    width: int,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Evaluate the finite Fourier polynomial associated with a square envelope."""

    kx_array, ky_array = np.broadcast_arrays(
        np.asarray(kx, dtype=float), np.asarray(ky, dtype=float)
    )
    value = np.zeros(kx_array.shape + (2,), dtype=np.complex128)
    for rx, ry in support_positions(width):
        vector = coefficient_vector(coefficients, trial, rx, ry)
        value += np.exp(-1.0j * (kx_array * rx + ky_array * ry))[..., None] * vector
    if normalize:
        value /= np.sqrt(support_weight(coefficients, trial, width))
    return value


def effective_form_factor_abs(
    kx: np.ndarray | float,
    ky: np.ndarray | float,
    alpha: float,
    coefficients: np.ndarray | None = None,
    width: int | None = None,
) -> np.ndarray:
    """Gauge-invariant magnitude of the A-family lower-band overlap."""

    projector = band_projector(kx, ky, alpha, band=-1)
    if coefficients is None or width is None:
        spinor = np.broadcast_to(TAU_A, projector.shape[:-1])
    else:
        spinor = truncated_spinor(kx, ky, coefficients, TAU_A, width)
    squared = np.real(
        np.einsum(
            "...a,...ab,...b->...", spinor.conj(), projector, spinor, optimize=True
        )
    )
    return np.sqrt(np.maximum(squared, 0.0))


def local_frame(
    kx: np.ndarray | float, ky: np.ndarray | float, alpha: float, reference: np.ndarray
) -> np.ndarray:
    projector = band_projector(kx, ky, alpha, band=-1)
    frame = np.einsum("...ab,b->...a", projector, reference, optimize=True)
    frame /= np.linalg.norm(frame, axis=-1)[..., None]
    return frame


def local_overlap_winding(
    zero_kx: float,
    zero_ky: float,
    alpha: float,
    coefficients: np.ndarray,
    width: int,
    loop_radius: float = 0.02,
) -> int:
    references = (IDENTITY[:, 0], IDENTITY[:, 1], TAU_A, TAU_B)
    projector_at_zero = band_projector(zero_kx, zero_ky, alpha, band=-1)
    reference = max(references, key=lambda vector: np.linalg.norm(projector_at_zero @ vector))
    theta = np.linspace(0.0, 2.0 * np.pi, 721)
    kx = zero_kx + loop_radius * np.cos(theta)
    ky = zero_ky + loop_radius * np.sin(theta)
    frame = local_frame(kx, ky, alpha, reference)
    spinor = truncated_spinor(kx, ky, coefficients, TAU_A, width)
    overlap = np.einsum("...a,...a->...", frame.conj(), spinor, optimize=True)
    phase = np.unwrap(np.angle(overlap))
    return int(np.rint((phase[-1] - phase[0]) / (2.0 * np.pi)))


def locate_overlap_zero(alpha: float, coefficients: np.ndarray, width: int) -> tuple[float, float]:
    """Refine the symmetry-line zero at k_y=0 for 0 < alpha < 2."""

    fit = minimize_scalar(
        lambda kx: float(
            effective_form_factor_abs(kx, 0.0, alpha, coefficients, width)
        ),
        bounds=(0.05 * np.pi, 0.999999 * np.pi),
        method="bounded",
        options={"xatol": 1.0e-13},
    )
    return float(fit.x), float(fit.fun)


def normalized_mode_at_point(
    kx: float,
    ky: float,
    coefficients: np.ndarray,
    trial: np.ndarray,
    width: int,
) -> np.ndarray:
    return truncated_spinor(
        kx, ky, coefficients, trial, width, normalize=True
    ).reshape(2)


def flattened_frame_hamiltonian_at_gamma(alpha: float, width: int) -> np.ndarray:
    """Return the normalized OW-frame Hamiltonian h_w at Gamma=(0,0)."""

    coefficient_minus, coefficient_plus = projector_fourier_coefficients(alpha)
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    for sign, coefficients in ((+1.0, coefficient_plus), (-1.0, coefficient_minus)):
        for trial in TRIAL_SPINORS:
            spinor = normalized_mode_at_point(
                0.0, 0.0, coefficients, trial, width
            )
            hamiltonian += sign * np.outer(spinor, spinor.conj())
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def flattened_frame_mass(alpha: float, width: int) -> float:
    hamiltonian = flattened_frame_hamiltonian_at_gamma(alpha, width)
    return float(0.5 * np.trace(hamiltonian @ SIGMA_Z).real)


def locate_flattened_critical_alpha(width: int) -> float:
    """Find the Gamma-point gap closure of the normalized finite-w OW frame."""

    return float(
        brentq(
            lambda alpha: flattened_frame_mass(alpha, width),
            1.50,
            1.9995,
            xtol=2.0e-11,
            rtol=2.0e-11,
            maxiter=100,
        )
    )


def critical_fit(width: np.ndarray, alpha_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def model(w: np.ndarray, alpha_infinity: float, amplitude: float, offset: float, power: float) -> np.ndarray:
        return alpha_infinity - amplitude / np.power(w + offset, power)

    parameters, covariance = curve_fit(
        model,
        width,
        alpha_c,
        p0=(2.0, 0.4, 0.4, 2.0),
        bounds=([1.95, 0.0, -0.9, 0.5], [2.05, 5.0, 5.0, 4.0]),
        maxfev=100000,
    )
    return parameters, covariance


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(FIGURE_DIR / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_summary_figure(
    coefficient_minus: np.ndarray,
    rows: pd.DataFrame,
    fit_parameters: np.ndarray,
    gap_rows: pd.DataFrame,
    gap_diagnostics: dict[str, object],
) -> None:
    """Combine the form-factor cut and finite-window diagnostics in one column."""

    figure, axes_grid = plt.subplots(2, 2, figsize=(3.38, 3.85))
    axes = axes_grid.ravel()
    styles = {
        1: dict(color=BPJ_RED, marker="^", linestyle=":"),
        2: dict(color=BPJ_GREEN, marker="s", linestyle="--"),
        4: dict(color=BPJ_GOLD, marker="D", linestyle="-."),
        6: dict(color=BPJ_BLUE, marker="o", linestyle="-"),
        8: dict(color=BPJ_PURPLE, marker="v", linestyle=(0, (3, 1, 1, 1))),
    }

    norm_k = np.linspace(-np.pi, np.pi, MAP_GRID, endpoint=False)
    norm_kx, norm_ky = np.meshgrid(norm_k, norm_k, indexing="ij")
    normalization_factors: dict[int | None, float] = {}
    for width in (*DISPLAY_WIDTHS, None):
        if width is None:
            field = effective_form_factor_abs(norm_kx, norm_ky, ALPHA)
        else:
            field = effective_form_factor_abs(
                norm_kx, norm_ky, ALPHA, coefficient_minus, width
            )
        normalization_factors[width] = float(np.sqrt(np.mean(np.square(field))))

    axis = axes[0]
    cut_kx_over_pi = np.linspace(0.40, 0.60, 601)
    cut_kx = np.pi * cut_kx_over_pi
    axis.plot(
        cut_kx_over_pi,
        effective_form_factor_abs(cut_kx, 0.0, ALPHA)
        / normalization_factors[None],
        color=BPJ_BLACK,
        linestyle="--",
        linewidth=0.9,
        label=r"$w=\infty$",
    )
    for width in DISPLAY_WIDTHS:
        style = styles[width]
        axis.plot(
            cut_kx_over_pi,
            effective_form_factor_abs(
                cut_kx, 0.0, ALPHA, coefficient_minus, width
            )
            / normalization_factors[width],
            color=style["color"],
            marker=style["marker"],
            markevery=150,
            markersize=2.2,
            markerfacecolor="white",
            linestyle=style["linestyle"],
            linewidth=0.9,
            label=rf"$w={width}$",
        )
        zero = float(rows.loc[rows.w == width, "zero_kx_over_pi"].iloc[0])
        axis.plot(
            zero,
            0.0,
            color=style["color"],
            marker=style["marker"],
            markersize=3.0,
            linestyle="none",
            clip_on=False,
        )
    axis.axvline(0.5, color=BPJ_GRAY, linestyle=":", linewidth=0.6)
    axis.set(
        xlabel=r"$k_x/\pi\ (k_y=0)$",
        ylabel=r"$|f_{A,-;w}^{\mathrm{eff}}|$",
        xlim=(0.40, 0.60),
        ylim=(-0.017, 0.345),
        xticks=[0.4, 0.5, 0.6],
        yticks=[0.0, 0.15, 0.30],
    )
    handles, labels = axis.get_legend_handles_labels()
    legend_order = list(range(1, len(handles))) + [0]
    axis.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=4.8,
        columnspacing=0.7,
        handlelength=1.5,
        handletextpad=0.3,
    )
    panel_label(axis, "a")

    zero_axis = axes[1]
    zero_axis.plot(
        rows.w,
        rows.zero_kx_over_pi,
        color=BPJ_RED,
        marker="^",
        markerfacecolor="white",
        linestyle=":",
        linewidth=0.9,
        markersize=3.0,
    )
    zero_axis.axhline(0.5, color=BPJ_RED, linestyle="--", linewidth=0.7, alpha=0.75)
    zero_axis.set(
        xlabel=r"$w$",
        ylabel=r"$k_x^{(0)}/\pi$",
        xlim=(0.5, 12.5),
        ylim=(0.4918, 0.5006),
        xticks=[1, 2, 4, 6, 8, 10, 12],
        yticks=[0.492, 0.494, 0.496, 0.498, 0.500],
    )
    zero_axis.tick_params(axis="y", colors=BPJ_RED)
    zero_axis.yaxis.label.set_color(BPJ_RED)
    zero_axis.spines["left"].set_color(BPJ_RED)

    alpha_axis = zero_axis.twinx()
    alpha_axis.plot(
        rows.w,
        rows.flat_parent_alpha_c,
        color=BPJ_BLUE,
        marker="o",
        markerfacecolor="white",
        linestyle="none",
        markersize=3.0,
    )
    dense_width = np.linspace(1.0, 13.0, 500)
    alpha_infinity, amplitude, offset, power = fit_parameters
    fitted = alpha_infinity - amplitude / np.power(dense_width + offset, power)
    alpha_axis.plot(dense_width, fitted, color=BPJ_BLUE, linestyle="-", linewidth=0.9)
    alpha_axis.axhline(2.0, color=BPJ_BLUE, linestyle="--", linewidth=0.7, alpha=0.75)
    alpha_axis.text(
        0.96,
        0.08,
        rf"$p={power:.3f}$",
        transform=alpha_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.9,
        color=BPJ_BLUE,
    )
    alpha_axis.set(
        ylabel=r"$\alpha_c(w)$",
        ylim=(1.78, 2.005),
        yticks=[1.80, 1.85, 1.90, 1.95, 2.00],
    )
    alpha_axis.tick_params(axis="y", colors=BPJ_BLUE)
    alpha_axis.tick_params(axis="x", bottom=False, top=False, labelbottom=False)
    alpha_axis.yaxis.label.set_color(BPJ_BLUE)
    alpha_axis.yaxis.labelpad = 2.0
    alpha_axis.spines["right"].set_color(BPJ_BLUE)
    zero_axis.grid(alpha=0.16, linewidth=0.5)
    panel_label(zero_axis, "b")

    axis = axes[2]
    sigma_x_rows = gap_rows.loc[
        (gap_rows.tau_axis == "X") & (gap_rows.w >= 1)
    ].sort_values("w")
    gap_infinity = float(
        gap_diagnostics["infinite_window"]["X"]["half_filling_gap"]
    )
    gap_error = np.abs(
        sigma_x_rows.half_filling_gap.to_numpy(dtype=float) - gap_infinity
    )
    axis.semilogy(
        sigma_x_rows.w,
        gap_error,
        color=BPJ_RED,
        marker="^",
        markerfacecolor="white",
        linestyle=":",
        linewidth=0.9,
        markersize=3.0,
        label=r"full-BZ minimum",
    )
    tail_fit = gap_diagnostics["tail_fits"]["X"]
    tail_width = np.linspace(float(tail_fit["minimum_width"]), 12.0, 300)
    tail_curve = float(tail_fit["amplitude"]) * np.exp(
        -tail_width / float(tail_fit["decay_length"])
    )
    axis.plot(
        tail_width,
        tail_curve,
        color=BPJ_BLACK,
        linestyle="--",
        linewidth=0.75,
        label=rf"$\xi={float(tail_fit['decay_length']):.2f}$",
    )
    axis.set(
        xlabel=r"$w$",
        ylabel=r"$|\Delta_0(w)-2|$",
        xlim=(0.5, 12.5),
        ylim=(3.0e-5, 3.0e-1),
        xticks=[1, 2, 4, 6, 8, 10, 12],
    )
    axis.legend(
        loc="lower left",
        frameon=False,
        fontsize=4.7,
        handlelength=1.6,
        handletextpad=0.35,
    )
    axis.grid(alpha=0.16, linewidth=0.5)
    panel_label(axis, "c")

    axis = axes[3]
    axis.plot(
        rows.w,
        100.0 * rows.retained_weight,
        color=BPJ_GREEN,
        marker="s",
        markerfacecolor="white",
        linestyle="--",
        linewidth=0.9,
        markersize=2.8,
    )
    axis.axhline(99.0, color=BPJ_BLACK, linestyle="--", linewidth=0.7)
    row_one = rows.loc[rows.w == 1].iloc[0]
    axis.annotate(
        rf"$w=1:\ {100.0 * row_one.retained_weight:.4f}\%$",
        xy=(1.0, 100.0 * row_one.retained_weight),
        xytext=(7, 8),
        textcoords="offset points",
        fontsize=4.9,
        arrowprops=dict(arrowstyle="-", color=BPJ_BLACK, linewidth=0.5),
    )
    axis.set(
        xlabel=r"$w$",
        ylabel="retained weight (\\%)",
        xlim=(0.5, 12.5),
        ylim=(98.95, 100.02),
        xticks=[1, 4, 8, 12],
    )
    panel_label(axis, "d")

    for axis in axes:
        axis.tick_params(pad=1.5)
    alpha_axis.tick_params(pad=1.5)
    figure.subplots_adjust(
        left=0.15, right=0.98, bottom=0.115, top=0.975, wspace=0.46, hspace=0.50
    )
    save_figure(figure, "ow_truncation_summary")


def main() -> None:
    configure_plotting()

    coefficient_minus, _ = projector_fourier_coefficients(ALPHA)
    exact_total_weight = total_weight(coefficient_minus, TAU_A)
    rows: list[dict[str, float | int]] = []
    for width in WINDOW_WIDTHS:
        zero_kx, zero_residual = locate_overlap_zero(
            ALPHA, coefficient_minus, int(width)
        )
        winding = local_overlap_winding(
            zero_kx, 0.0, ALPHA, coefficient_minus, int(width)
        )
        retained = support_weight(coefficient_minus, TAU_A, int(width)) / exact_total_weight
        alpha_c = locate_flattened_critical_alpha(int(width))
        rows.append(
            {
                "w": int(width),
                "support_cells": int((2 * width + 1) ** 2),
                "retained_weight": retained,
                "zero_kx_over_pi": zero_kx / np.pi,
                "zero_ky_over_pi": 0.0,
                "zero_residual": zero_residual,
                "overlap_winding": winding,
                "flat_parent_alpha_c": alpha_c,
            }
        )
        print(
            f"w={width:2d}: retained={retained:.10f}, "
            f"zero/pi=({zero_kx / np.pi:.9f}, 0), winding={winding:+d}, "
            f"alpha_c={alpha_c:.9f}"
        )

    summary = pd.DataFrame(rows)
    if not np.all(summary.zero_residual < 2.0e-6):
        raise RuntimeError("At least one alpha=1 overlap zero was not resolved.")
    if set(summary.overlap_winding) != {-1}:
        raise RuntimeError("The finite-w overlap winding is inconsistent with C_-=+1.")
    if abs(float(summary.loc[summary.w == 1, "retained_weight"].iloc[0]) - 0.992155) > 3.0e-6:
        raise RuntimeError("The w=1 retained weight does not reproduce the audited legacy value.")
    if not np.all(np.diff(summary.flat_parent_alpha_c) > 0.0):
        raise RuntimeError("The flattened-frame critical masses should approach two monotonically.")

    fit_rows = summary.query("w >= 3")
    parameters, covariance = critical_fit(
        fit_rows.w.to_numpy(dtype=float),
        fit_rows.flat_parent_alpha_c.to_numpy(dtype=float),
    )
    standard_errors = np.sqrt(np.diag(covariance))
    alpha_infinity, amplitude, offset, power = parameters
    print(
        "critical fit: "
        f"alpha_infinity={alpha_infinity:.9f}, A={amplitude:.9f}, "
        f"b={offset:.9f}, p={power:.9f}"
    )

    summary.to_csv(DATA_DIR / "ow_truncation_diagnostics.csv", index=False)
    fit_payload = {
        "hamiltonian_convention": ["sin(kx)", "sin(ky)", "alpha-cos(kx)-cos(ky)"],
        "alpha_form_factor": ALPHA,
        "fourier_grid": FOURIER_GRID,
        "integer_square_envelope": "|r_x| <= w and |r_y| <= w",
        "display_widths": list(DISPLAY_WIDTHS),
        "form_factor_plot_normalization": "unit scalar L2 norm over the BZ using the uniform MAP_GRID quadrature",
        "critical_fit_widths": fit_rows.w.astype(int).tolist(),
        "critical_fit_model": "alpha_c(w) = alpha_infinity - A / (w + b)^p",
        "critical_fit": {
            "alpha_infinity": float(alpha_infinity),
            "A": float(amplitude),
            "b": float(offset),
            "p": float(power),
        },
        "critical_fit_standard_errors": {
            "alpha_infinity": float(standard_errors[0]),
            "A": float(standard_errors[1]),
            "b": float(standard_errors[2]),
            "p": float(standard_errors[3]),
        },
        "interpretation": (
            "alpha_c(w) is the Gamma-point gap closing of the normalized finite-w OW-frame "
            "Hamiltonian, not a shifted transition of the target band or scalar overlap."
        ),
    }
    (DATA_DIR / "ow_truncation_fit.json").write_text(
        json.dumps(fit_payload, indent=2) + "\n", encoding="utf-8"
    )

    gap_table_path = DATA_DIR / "flattened_pauli_gap_vs_w.csv"
    gap_diagnostics_path = DATA_DIR / "flattened_pauli_gap_analysis.json"
    if not gap_table_path.exists() or not gap_diagnostics_path.exists():
        raise FileNotFoundError(
            "Run analyze_flattened_pauli_gaps.py before assembling Fig. 1."
        )
    gap_summary = pd.read_csv(gap_table_path)
    gap_diagnostics = json.loads(gap_diagnostics_path.read_text(encoding="utf-8"))
    sigma_x_widths = gap_summary.loc[gap_summary.tau_axis == "X", "w"].to_numpy()
    if not np.array_equal(sigma_x_widths, np.arange(0, 13, dtype=int)):
        raise RuntimeError("The sigma_x gap table does not contain w=0,...,12.")

    make_summary_figure(
        coefficient_minus,
        summary,
        parameters,
        gap_summary,
        gap_diagnostics,
    )


if __name__ == "__main__":
    main()
