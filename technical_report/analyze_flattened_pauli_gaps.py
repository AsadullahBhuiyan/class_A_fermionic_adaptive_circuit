#!/usr/bin/env python3
"""Analyze the bulk finite-window flattened-parent gap for Pauli trial bases.

For each choice ``tau in {X, Y, Z}``, the two trial spinors are the +1 and -1
eigenstates of the corresponding Pauli matrix.  The four projected modes in
Eq. (15) of ``technical_report.tex`` are normalized separately before their
rank-one projectors are summed.  We report the half-filling (zero-energy) gap

    Delta_0(w) = min_{k,n} |E_n[h_w(k)]|.

The Hamiltonian is numerically traceless for these uniform square windows, so
the direct occupied-to-empty band gap is ``2 * Delta_0``.  The Brillouin-zone
minimum is first located on a regular grid and then refined continuously.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from analyze_ow_truncation import (
    ALPHA,
    BPJ_BLUE,
    BPJ_GREEN,
    BPJ_RED,
    FOURIER_GRID,
    SIGMA_Z,
    band_projector,
    configure_plotting,
    panel_label,
    projector_fourier_coefficients,
    support_positions,
    total_weight,
)


HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
DATA_DIR = HERE / "data"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

WIDTHS = np.arange(0, 13, dtype=int)
GAP_GRID = 512
TAIL_FIT_MIN_WIDTH = 4

PAULI_TRIALS = {
    "X": (
        np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0),
        np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0),
    ),
    "Y": (
        np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0),
        np.array([1.0, -1.0j], dtype=np.complex128) / np.sqrt(2.0),
    ),
    "Z": (
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
    ),
}


def canonical_momentum(momentum: float) -> float:
    """Map a momentum to [-pi, pi), preserving periodic objective values."""

    return float((momentum + np.pi) % (2.0 * np.pi) - np.pi)


def finite_mode_data(
    coefficients: np.ndarray,
    trial: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return positions, coefficient vectors, and norm for one finite mode."""

    positions = np.asarray(support_positions(width), dtype=int)
    rx = positions[:, 0]
    ry = positions[:, 1]
    vectors = coefficients[rx % FOURIER_GRID, ry % FOURIER_GRID] @ trial
    weight = float(np.sum(np.abs(vectors) ** 2))
    if not np.isfinite(weight) or weight <= 0.0:
        raise RuntimeError(f"Invalid finite-mode weight at w={width}: {weight}")
    return rx, ry, vectors, weight


def finite_mode_on_grid(
    rx: np.ndarray,
    ry: np.ndarray,
    vectors: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Evaluate a finite Fourier polynomial on the regular gap grid."""

    coefficient_grid = np.zeros((GAP_GRID, GAP_GRID, 2), dtype=np.complex128)
    coefficient_grid[rx % GAP_GRID, ry % GAP_GRID] = vectors
    return np.fft.fft2(coefficient_grid, axes=(0, 1)) / np.sqrt(weight)


def accumulate_mode(
    h00: np.ndarray,
    h01: np.ndarray,
    h11: np.ndarray,
    mode: np.ndarray,
    sign: float,
) -> None:
    """Accumulate sign * |mode><mode| into packed Hermitian components."""

    h00 += sign * np.abs(mode[..., 0]) ** 2
    h01 += sign * mode[..., 0] * mode[..., 1].conj()
    h11 += sign * np.abs(mode[..., 1]) ** 2


def gaps_from_components(
    h00: np.ndarray,
    h01: np.ndarray,
    h11: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the zero-energy gap, direct gap, and trace on a grid."""

    trace = h00 + h11
    radius = np.sqrt(np.square(0.5 * (h00 - h11)) + np.abs(h01) ** 2)
    lower = 0.5 * trace - radius
    upper = 0.5 * trace + radius
    zero_gap = np.minimum(np.abs(lower), np.abs(upper))
    direct_gap = 2.0 * radius
    return zero_gap, direct_gap, trace


def finite_grid_scan(
    coefficient_minus: np.ndarray,
    coefficient_plus: np.ndarray,
    trials: tuple[np.ndarray, np.ndarray],
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]]]:
    """Evaluate h_w on the gap grid and retain scalar mode data for refinement."""

    h00 = np.zeros((GAP_GRID, GAP_GRID), dtype=float)
    h01 = np.zeros((GAP_GRID, GAP_GRID), dtype=np.complex128)
    h11 = np.zeros((GAP_GRID, GAP_GRID), dtype=float)
    scalar_modes = []
    for sign, coefficients in ((+1.0, coefficient_plus), (-1.0, coefficient_minus)):
        for trial in trials:
            rx, ry, vectors, weight = finite_mode_data(coefficients, trial, width)
            mode = finite_mode_on_grid(rx, ry, vectors, weight)
            accumulate_mode(h00, h01, h11, mode, sign)
            scalar_modes.append((sign, rx, ry, vectors, weight))
    zero_gap, direct_gap, trace = gaps_from_components(h00, h01, h11)
    return zero_gap, direct_gap, trace, scalar_modes


def infinite_grid_scan(
    coefficient_minus: np.ndarray,
    coefficient_plus: np.ndarray,
    trials: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, int, np.ndarray, float]]]:
    """Evaluate the unwindowed parent directly from its band projectors."""

    momenta = 2.0 * np.pi * np.arange(GAP_GRID) / GAP_GRID
    kx, ky = np.meshgrid(momenta, momenta, indexing="ij")
    h00 = np.zeros((GAP_GRID, GAP_GRID), dtype=float)
    h01 = np.zeros((GAP_GRID, GAP_GRID), dtype=np.complex128)
    h11 = np.zeros((GAP_GRID, GAP_GRID), dtype=float)
    scalar_modes = []
    for sign, band, coefficients in (
        (+1.0, +1, coefficient_plus),
        (-1.0, -1, coefficient_minus),
    ):
        projector = band_projector(kx, ky, ALPHA, band=band)
        for trial in trials:
            weight = total_weight(coefficients, trial)
            mode = np.einsum("...ab,b->...a", projector, trial, optimize=True)
            mode /= np.sqrt(weight)
            accumulate_mode(h00, h01, h11, mode, sign)
            scalar_modes.append((sign, band, trial, weight))
    zero_gap, direct_gap, trace = gaps_from_components(h00, h01, h11)
    return zero_gap, direct_gap, trace, scalar_modes


def finite_hamiltonian_at(
    momentum: np.ndarray,
    scalar_modes: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]],
) -> np.ndarray:
    """Evaluate one finite-w parent at an arbitrary periodic momentum."""

    kx, ky = [canonical_momentum(value) for value in momentum]
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    for sign, rx, ry, vectors, weight in scalar_modes:
        phase = np.exp(-1.0j * (kx * rx + ky * ry))
        mode = phase @ vectors / np.sqrt(weight)
        hamiltonian += sign * np.outer(mode, mode.conj())
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def infinite_hamiltonian_at(
    momentum: np.ndarray,
    scalar_modes: list[tuple[float, int, np.ndarray, float]],
) -> np.ndarray:
    """Evaluate the unwindowed parent at an arbitrary periodic momentum."""

    kx, ky = [canonical_momentum(value) for value in momentum]
    hamiltonian = np.zeros((2, 2), dtype=np.complex128)
    for sign, band, trial, weight in scalar_modes:
        projector = band_projector(kx, ky, ALPHA, band=band).reshape(2, 2)
        mode = projector @ trial / np.sqrt(weight)
        hamiltonian += sign * np.outer(mode, mode.conj())
    return 0.5 * (hamiltonian + hamiltonian.conj().T)


def refine_gap(
    grid_gap: np.ndarray,
    hamiltonian_at,
) -> tuple[float, float, float, np.ndarray, bool]:
    """Refine the lowest grid minima with a periodic continuous optimizer."""

    candidate_count = 24
    flat_indices = np.argpartition(grid_gap.ravel(), candidate_count)[:candidate_count]
    starts = []
    for flat_index in flat_indices:
        ix, iy = np.unravel_index(flat_index, grid_gap.shape)
        starts.append(2.0 * np.pi * np.asarray([ix, iy], dtype=float) / GAP_GRID)
    starts.extend(
        np.asarray(point, dtype=float)
        for point in (
            (0.0, 0.0),
            (np.pi, 0.0),
            (0.0, np.pi),
            (np.pi, np.pi),
        )
    )

    def objective(momentum: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(hamiltonian_at(momentum))
        return float(np.min(np.abs(eigenvalues)))

    best = None
    all_successful = True
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"xatol": 2.0e-12, "fatol": 2.0e-13, "maxiter": 1000},
        )
        all_successful = all_successful and bool(result.success)
        value = objective(result.x)
        if best is None or value < best[0]:
            best = (value, result.x)
    if best is None:
        raise RuntimeError("No gap-refinement candidates were evaluated.")

    gap, momentum = best
    kx, ky = [canonical_momentum(value) for value in momentum]
    eigenvalues = np.linalg.eigvalsh(hamiltonian_at(np.asarray([kx, ky])))
    return float(gap), kx, ky, eigenvalues, all_successful


def analyze_axis(
    axis: str,
    coefficient_minus: np.ndarray,
    coefficient_plus: np.ndarray,
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, float | str | bool]]:
    """Compute finite-w and unwindowed gaps for one Pauli trial basis."""

    trials = PAULI_TRIALS[axis]
    rows: list[dict[str, float | int | str | bool]] = []
    for width in WIDTHS:
        grid_gap, grid_direct, trace, scalar_modes = finite_grid_scan(
            coefficient_minus,
            coefficient_plus,
            trials,
            int(width),
        )
        hamiltonian_at = lambda momentum, modes=scalar_modes: finite_hamiltonian_at(
            momentum, modes
        )
        gap, kx, ky, eigenvalues, optimizer_success = refine_gap(
            grid_gap,
            hamiltonian_at,
        )
        rows.append(
            {
                "tau_axis": axis,
                "w": int(width),
                "half_filling_gap": gap,
                "direct_gap": float(eigenvalues[1] - eigenvalues[0]),
                "lower_energy_at_min": float(eigenvalues[0]),
                "upper_energy_at_min": float(eigenvalues[1]),
                "kx_min_over_pi": kx / np.pi,
                "ky_min_over_pi": ky / np.pi,
                "grid_half_filling_gap": float(np.min(grid_gap)),
                "grid_direct_gap": float(np.min(grid_direct)),
                "trace_abs_max": float(np.max(np.abs(trace))),
                "optimizer_success": optimizer_success,
            }
        )
        print(
            f"tau={axis}, w={width:2d}: Delta_0={gap:.12f}, "
            f"Delta_dir={eigenvalues[1] - eigenvalues[0]:.12f}, "
            f"k*/pi=({kx / np.pi:+.8f}, {ky / np.pi:+.8f})"
        )

    grid_gap, grid_direct, trace, scalar_modes = infinite_grid_scan(
        coefficient_minus,
        coefficient_plus,
        trials,
    )
    hamiltonian_at = lambda momentum, modes=scalar_modes: infinite_hamiltonian_at(
        momentum, modes
    )
    gap, kx, ky, eigenvalues, optimizer_success = refine_gap(grid_gap, hamiltonian_at)
    infinity = {
        "tau_axis": axis,
        "half_filling_gap": gap,
        "direct_gap": float(eigenvalues[1] - eigenvalues[0]),
        "lower_energy_at_min": float(eigenvalues[0]),
        "upper_energy_at_min": float(eigenvalues[1]),
        "kx_min_over_pi": kx / np.pi,
        "ky_min_over_pi": ky / np.pi,
        "grid_half_filling_gap": float(np.min(grid_gap)),
        "grid_direct_gap": float(np.min(grid_direct)),
        "trace_abs_max": float(np.max(np.abs(trace))),
        "optimizer_success": optimizer_success,
    }
    print(
        f"tau={axis}, w=inf: Delta_0={gap:.12f}, "
        f"Delta_dir={eigenvalues[1] - eigenvalues[0]:.12f}, "
        f"k*/pi=({kx / np.pi:+.8f}, {ky / np.pi:+.8f})"
    )
    return rows, infinity


def exponential_tail_fit(
    rows: pd.DataFrame,
    infinity_gap: float,
) -> dict[str, float | int]:
    """Fit |Delta_0(w)-Delta_0(infinity)| = A exp(-w/xi) on the tail."""

    tail = rows.loc[rows.w >= TAIL_FIT_MIN_WIDTH].copy()
    deviation = np.abs(tail.half_filling_gap.to_numpy(dtype=float) - infinity_gap)
    if np.any(deviation <= 0.0):
        raise RuntimeError("Cannot log-fit a zero asymptotic gap deviation.")
    widths = tail.w.to_numpy(dtype=float)
    slope, intercept = np.polyfit(widths, np.log(deviation), 1)
    prediction = intercept + slope * widths
    residual = np.log(deviation) - prediction
    total = np.log(deviation) - np.mean(np.log(deviation))
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2))
    return {
        "minimum_width": TAIL_FIT_MIN_WIDTH,
        "amplitude": float(np.exp(intercept)),
        "decay_length": float(-1.0 / slope),
        "log_space_r_squared": r_squared,
    }


def make_gap_figure(
    table: pd.DataFrame,
    infinity: dict[str, dict[str, float | str | bool]],
    fits: dict[str, dict[str, float | int]],
) -> None:
    """Plot the gap and its approach to the unwindowed value."""

    configure_plotting()
    figure, axes = plt.subplots(2, 1, figsize=(3.375, 4.35), sharex=True)
    styles = {
        "X": dict(color=BPJ_RED, marker="^", linestyle=":"),
        "Y": dict(color=BPJ_GREEN, marker="s", linestyle="--"),
        "Z": dict(color=BPJ_BLUE, marker="o", linestyle="-"),
    }
    labels = {
        "X": r"$\tau\in\sigma_X$",
        "Y": r"$\tau\in\sigma_Y$",
        "Z": r"$\tau\in\sigma_Z$",
    }

    for axis_name in ("X", "Y", "Z"):
        subset = table.loc[table.tau_axis == axis_name]
        style = styles[axis_name]
        marker_size = 5.0 if axis_name == "X" else 3.8
        marker_face = "white" if axis_name == "Y" else style["color"]
        axes[0].plot(
            subset.w,
            subset.half_filling_gap,
            label=labels[axis_name],
            linewidth=1.25,
            markersize=marker_size,
            markerfacecolor=marker_face,
            markeredgewidth=0.85,
            zorder=2 if axis_name == "X" else 3,
            **style,
        )
        infinity_gap = float(infinity[axis_name]["half_filling_gap"])
        deviation = np.abs(subset.half_filling_gap.to_numpy(dtype=float) - infinity_gap)
        axes[1].semilogy(
            subset.w,
            deviation,
            linewidth=1.25,
            markersize=marker_size,
            markerfacecolor=marker_face,
            markeredgewidth=0.85,
            zorder=2 if axis_name == "X" else 3,
            **style,
        )

    xy_infinity = float(infinity["X"]["half_filling_gap"])
    z_infinity = float(infinity["Z"]["half_filling_gap"])
    axes[0].axhline(xy_infinity, color="#555555", linestyle=(0, (4, 2)), linewidth=0.8)
    axes[0].axhline(z_infinity, color="#888888", linestyle=(0, (2, 2)), linewidth=0.8)
    axes[0].text(
        11.75,
        xy_infinity - 0.025,
        rf"$\Delta_0^{{X,Y}}(\infty)={xy_infinity:.3f}$",
        ha="right",
        va="top",
        fontsize=6.2,
        color="#444444",
    )
    axes[0].text(
        11.75,
        z_infinity + 0.025,
        rf"$\Delta_0^Z(\infty)={z_infinity:.3f}$",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#666666",
    )
    axes[0].text(
        0.04,
        0.08,
        r"$\alpha=1$",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
    )

    axes[0].set_ylabel(r"$\Delta_0(w)=\min_{\boldsymbol{k},n}|E_n|$")
    axes[0].set_ylim(1.20, 2.06)
    axes[0].legend(loc="center right", frameon=False, handlelength=2.4)
    axes[1].set_xlabel(r"envelope width $w$")
    axes[1].set_ylabel(r"$|\Delta_0(w)-\Delta_0(\infty)|$")
    axes[1].set_xticks([0, 2, 4, 6, 8, 10, 12])
    axes[1].set_xlim(-0.3, 12.3)
    axes[1].text(
        0.97,
        0.93,
        r"tail fits ($w\geq4$)" + "\n"
        + rf"$\xi_X={float(fits['X']['decay_length']):.2f}$, "
        + rf"$\xi_Y={float(fits['Y']['decay_length']):.2f}$" + "\n"
        + rf"$\xi_Z={float(fits['Z']['decay_length']):.2f}$",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
    )

    for label, axis in zip(("a", "b"), axes):
        panel_label(axis, label)
        axis.grid(alpha=0.16, linewidth=0.5)
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.11, top=0.98, hspace=0.10)
    figure.savefig(FIGURE_DIR / "flattened_pauli_gap_vs_w.pdf", bbox_inches="tight")
    figure.savefig(
        FIGURE_DIR / "flattened_pauli_gap_vs_w.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    coefficient_minus, coefficient_plus = projector_fourier_coefficients(ALPHA)
    all_rows = []
    infinity = {}
    for axis in ("X", "Y", "Z"):
        rows, infinity_row = analyze_axis(axis, coefficient_minus, coefficient_plus)
        all_rows.extend(rows)
        infinity[axis] = infinity_row

    table = pd.DataFrame(all_rows)
    table.to_csv(DATA_DIR / "flattened_pauli_gap_vs_w.csv", index=False)

    fits = {}
    for axis in ("X", "Y", "Z"):
        fits[axis] = exponential_tail_fit(
            table.loc[table.tau_axis == axis],
            float(infinity[axis]["half_filling_gap"]),
        )

    x_gap = table.loc[table.tau_axis == "X", "half_filling_gap"].to_numpy()
    y_gap = table.loc[table.tau_axis == "Y", "half_filling_gap"].to_numpy()
    if not np.allclose(x_gap, y_gap, rtol=0.0, atol=2.0e-11):
        raise AssertionError("The X/Y gaps violate the expected C4 equality.")
    if max(float(row["trace_abs_max"]) for row in infinity.values()) > 1.0e-11:
        raise AssertionError("The unwindowed parent is not numerically traceless.")
    if float(table.trace_abs_max.max()) > 1.0e-11:
        raise AssertionError("A finite-window parent is not numerically traceless.")
    if not bool(table.optimizer_success.all()):
        raise AssertionError("At least one continuous gap refinement did not converge.")

    diagnostics = {
        "alpha": ALPHA,
        "hamiltonian_convention": "n=(sin kx, sin ky, alpha-cos kx-cos ky)",
        "window": "square support |rx|,|ry| <= w",
        "pauli_trial_definition": (
            "tau_A and tau_B are the +1 and -1 eigenstates of sigma_X, sigma_Y, "
            "or sigma_Z; all four band-projected modes are normalized separately"
        ),
        "half_filling_gap_definition": "min_{k,n} |E_n[h_w(k)]|",
        "direct_gap_definition": "min_k (E_+(k)-E_-(k)) = 2*half_filling_gap",
        "fourier_coefficient_grid": FOURIER_GRID,
        "gap_search_grid": GAP_GRID,
        "finite_widths": WIDTHS.tolist(),
        "infinite_window": infinity,
        "tail_fit_model": "|Delta_0(w)-Delta_0(infinity)| = A exp(-w/xi)",
        "tail_fits": fits,
        "unwindowed_normalization_identity": {
            "bz_average_nz": float(
                np.trace(coefficient_plus[0, 0] @ SIGMA_Z).real
            ),
            "predicted_XY_half_filling_gap": 2.0,
            "predicted_Z_half_filling_gap": float(
                2.0
                / (
                    1.0
                    + np.trace(coefficient_plus[0, 0] @ SIGMA_Z).real
                )
            ),
            "explanation": (
                "For X and Y, every unwindowed projected trial mode has BZ norm "
                "1/2, so the normalized frame is tight and h_infinity=2(P_+-P_-). "
                "For Z, the two norms are (1+/-<n_z>_BZ)/2, and separate mode "
                "normalization lowers the minimum zero-energy gap to "
                "2/(1+<n_z>_BZ)."
            ),
        },
        "validation": {
            "max_abs_X_minus_Y_gap": float(np.max(np.abs(x_gap - y_gap))),
            "max_finite_trace_abs": float(table.trace_abs_max.max()),
            "all_refinements_converged": bool(table.optimizer_success.all()),
        },
    }
    (DATA_DIR / "flattened_pauli_gap_analysis.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    make_gap_figure(table, infinity, fits)

    print("\nTail fits")
    for axis in ("X", "Y", "Z"):
        print(f"tau={axis}: {fits[axis]}")
    print(f"\nSaved {DATA_DIR / 'flattened_pauli_gap_vs_w.csv'}")
    print(f"Saved {DATA_DIR / 'flattened_pauli_gap_analysis.json'}")
    print(f"Saved {FIGURE_DIR / 'flattened_pauli_gap_vs_w.pdf'}")
    print(f"Saved {FIGURE_DIR / 'flattened_pauli_gap_vs_w.png'}")


if __name__ == "__main__":
    main()
