#!/usr/bin/env python3
"""Resolve the equilibrium hard-wall ground state into its two edge sectors.

This deterministic analysis uses the exact flattened OW parent at Nx=20,
Ny=60, alpha_1=1, alpha_2=30, and nshell=1.  It keeps two complementary
notions of boundary isolation separate:

1. an entropy-contour decomposition of the full pure-state interval entropy,
   which assigns the universal logarithm to physical x cells without tracing
   out the rest of the cylinder; and
2. a two-state spectral diagnostic at each ky, trusted only where the selected
   low-energy subspace is demonstrably localized at the two walls.

A naive reduced state of a narrow x window has an extensive entropy from its
artificial x cuts.  It is deliberately not used to infer a central charge.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


BUNDLE_ROOT = Path(__file__).resolve().parent
REFERENCE_RUNNER = BUNDLE_ROOT / "run_flattened_ground_state_large_ny.py"
DEFAULT_OUTPUT = (
    BUNDLE_ROOT
    / "analysis_outputs"
    / "wall_projected_ground_state_nx20_ny60_v1"
)

NX = 20
NY = 60
ALPHA_1 = 1.0
ALPHA_2 = 30.0
NSHELL = 1
LEFT_WALL = 5
RIGHT_WALL = 15
ENTROPY_EPS = 1.0e-12
EDGE_WINDOW_RADIUS = 2
EDGE_WEIGHT_THRESHOLD = 0.5
FIT_AY_MIN = 2
CORRELATION_FIT_MIN = 5
CORRELATION_FIT_MAX = 25
SCHEMA = "wall_projected_flattened_ground_state_nx20_ny60_v1"


def _load_reference_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "flattened_ground_state_reference_large_ny", REFERENCE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {REFERENCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REFERENCE = _load_reference_module()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_flattened_blocks(model: Any) -> np.ndarray:
    """Construct the exact y-momentum blocks of the flattened OW parent."""

    q = 2 * NX
    h_k = np.zeros((NY, q, q), dtype=np.complex128)
    phase = np.exp(
        -2j
        * np.pi
        * np.arange(NY, dtype=np.float64)[:, None]
        * np.arange(NY, dtype=np.float64)[None, :]
        / float(NY)
    )
    translation_residual = 0.0
    for name, sign in (
        ("WF_Ap", 1.0),
        ("WF_Bp", 1.0),
        ("WF_Am", -1.0),
        ("WF_Bm", -1.0),
    ):
        frame = np.asarray(getattr(model, name), dtype=np.complex128)
        frame = frame.reshape(NY, q, NX, NY)
        frame_k = np.fft.fft(frame, axis=0) / np.sqrt(float(NY))
        expected = frame_k[:, :, :, :1] * phase[:, None, None, :]
        translation_residual = max(
            translation_residual, float(np.max(np.abs(frame_k - expected)))
        )
        representative = frame_k[:, :, :, 0]
        h_k += sign * float(NY) * np.einsum(
            "kar,kbr->kab", representative, representative.conj(), optimize=True
        )
    if translation_residual > 2.0e-10:
        raise FloatingPointError(
            f"OW y-translation residual is {translation_residual:.6e}"
        )
    hermiticity = float(
        np.max(np.abs(h_k - h_k.conj().transpose(0, 2, 1)))
    )
    if hermiticity > 2.0e-10:
        raise FloatingPointError(
            f"flattened parent is not Hermitian: {hermiticity:.6e}"
        )
    return 0.5 * (h_k + h_k.conj().transpose(0, 2, 1))


def ground_state_projector(
    h_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Half fill the globally lowest flattened-parent single-particle modes."""

    q = h_k.shape[1]
    energies = np.empty((NY, q), dtype=np.float64)
    eigenvectors = np.empty((NY, q, q), dtype=np.complex128)
    for momentum in range(NY):
        energies[momentum], eigenvectors[momentum] = eigh(
            h_k[momentum], check_finite=False, driver="evd"
        )
    total_rank = NY * q // 2
    flat_order = np.argsort(energies.ravel(), kind="stable")
    occupied = np.zeros(NY * q, dtype=bool)
    occupied[flat_order[:total_rank]] = True
    occupied = occupied.reshape(NY, q)
    projector_k = np.empty_like(h_k)
    for momentum in range(NY):
        vectors = eigenvectors[momentum][:, occupied[momentum]]
        projector_k[momentum] = vectors @ vectors.conj().T
    projector_delta = np.fft.ifft(projector_k, axis=0)
    diagnostics = {
        "half_filling_rank": float(total_rank),
        "negative_energy_count": float(np.count_nonzero(energies < 0.0)),
        "half_filling_gap": float(
            energies.ravel()[flat_order[total_rank]]
            - energies.ravel()[flat_order[total_rank - 1]]
        ),
        "minimum_absolute_energy": float(np.min(np.abs(energies))),
        "projector_idempotency_max_abs": float(
            max(np.max(np.abs(block @ block - block)) for block in projector_k)
        ),
        "projector_rank_residual": float(
            abs(np.trace(projector_k, axis1=1, axis2=2).real.sum() - total_rank)
        ),
    }
    if not np.all(occupied.sum(axis=1) == NX):
        raise RuntimeError("half filling did not occupy exactly Nx modes at every ky")
    if diagnostics["projector_idempotency_max_abs"] > 1.0e-10:
        raise FloatingPointError("ground-state correlation matrix is not a projector")
    return energies, eigenvectors, occupied, projector_delta, diagnostics


def restricted_interval(projector_delta: np.ndarray, ay: int) -> np.ndarray:
    rows = np.arange(int(ay), dtype=np.int64)
    delta = (rows[:, None] - rows[None, :]) % NY
    q = projector_delta.shape[1]
    return (
        projector_delta[delta]
        .transpose(0, 2, 1, 3)
        .reshape(rows.size * q, rows.size * q)
        .copy()
    )


def interval_entropy_contour(
    projector_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return S(Ay) and its additive x-resolved Gaussian entropy contour."""

    ay_values = np.arange(1, NY // 2 + 1, dtype=np.int64)
    contour_x = np.empty((ay_values.size, NX), dtype=np.float64)
    occupation_min = np.inf
    occupation_max = -np.inf
    for index, ay in enumerate(ay_values):
        restricted = restricted_interval(projector_delta, int(ay))
        restricted = 0.5 * (restricted + restricted.conj().T)
        occupations, modes = eigh(
            restricted, check_finite=False, overwrite_a=True, driver="evd"
        )
        occupation_min = min(occupation_min, float(occupations.min()))
        occupation_max = max(occupation_max, float(occupations.max()))
        if occupations.min() < -1.0e-8 or occupations.max() > 1.0 + 1.0e-8:
            raise FloatingPointError("restricted occupations left [0,1]")
        p = np.clip(occupations, ENTROPY_EPS, 1.0 - ENTROPY_EPS)
        mode_entropy = -p * np.log(p) - (1.0 - p) * np.log(1.0 - p)
        site_contour = (np.abs(modes) ** 2) @ mode_entropy
        contour_x[index] = site_contour.reshape(int(ay), NX, 2).sum(axis=(0, 2))
    return ay_values, contour_x, occupation_min, occupation_max


def fit_log_chord(ay_values: np.ndarray, values: np.ndarray) -> dict[str, float]:
    log_chord = np.log(
        (float(NY) / np.pi) * np.sin(np.pi * ay_values / float(NY))
    )
    mask = ay_values >= FIT_AY_MIN
    x = log_chord[mask]
    y = np.asarray(values, dtype=np.float64)[mask]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    variance = residual_sum / float(x.size - 2)
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    slope_se = float(np.sqrt(max(variance, 0.0) / sxx))
    return {
        "slope": float(slope),
        "slope_fit_se": slope_se,
        "intercept": float(intercept),
        "c_fit": float(3.0 * slope),
        "c_fit_se": float(3.0 * slope_se),
        "r2": float(1.0 - residual_sum / total_sum),
        "rms_residual": float(np.sqrt(residual_sum / x.size)),
    }


def wall_cells(center: int, radius: int) -> list[int]:
    return [int((center + offset) % NX) for offset in range(-radius, radius + 1)]


def contour_fits(
    ay_values: np.ndarray, contour_x: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    profiles: dict[str, np.ndarray] = {"full": contour_x.sum(axis=1)}
    for radius in range(5):
        left = wall_cells(LEFT_WALL, radius)
        right = wall_cells(RIGHT_WALL, radius)
        both = sorted(set(left + right))
        for region, cells in (("left", left), ("right", right), ("both", both)):
            profile = contour_x[:, cells].sum(axis=1)
            profiles[f"{region}_r{radius}"] = profile
            rows.append(
                {
                    "region": region,
                    "radius": radius,
                    "x_cells": ";".join(str(cell) for cell in cells),
                    "endpoint_entropy_contour": float(profile[-1]),
                    **fit_log_chord(ay_values, profile),
                }
            )
    full_fit = fit_log_chord(ay_values, profiles["full"])
    rows.append(
        {
            "region": "full",
            "radius": -1,
            "x_cells": ";".join(str(cell) for cell in range(NX)),
            "endpoint_entropy_contour": float(profiles["full"][-1]),
            **full_fit,
        }
    )
    primary_cells = sorted(
        set(wall_cells(LEFT_WALL, EDGE_WINDOW_RADIUS))
        | set(wall_cells(RIGHT_WALL, EDGE_WINDOW_RADIUS))
    )
    background = contour_x[:, [x for x in range(NX) if x not in primary_cells]].sum(
        axis=1
    )
    profiles["background"] = background
    rows.append(
        {
            "region": "background",
            "radius": EDGE_WINDOW_RADIUS,
            "x_cells": ";".join(
                str(cell) for cell in range(NX) if cell not in primary_cells
            ),
            "endpoint_entropy_contour": float(background[-1]),
            **fit_log_chord(ay_values, background),
        }
    )
    return rows, profiles


def edge_spectral_diagnostic(
    h_k: np.ndarray,
    energies: np.ndarray,
    eigenvectors: np.ndarray,
    occupied: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Localize the two lowest-|E| modes onto opposite walls at each ky."""

    q = 2 * NX
    left_cells = wall_cells(LEFT_WALL, EDGE_WINDOW_RADIUS)
    right_cells = wall_cells(RIGHT_WALL, EDGE_WINDOW_RADIUS)
    left_modes = [2 * x + orbital for x in left_cells for orbital in (0, 1)]
    right_modes = [2 * x + orbital for x in right_cells for orbital in (0, 1)]
    contrast = np.zeros(q, dtype=np.float64)
    contrast[left_modes] = 1.0
    contrast[right_modes] = -1.0
    left_projector = np.diag((contrast > 0.0).astype(np.float64))
    right_projector = np.diag((contrast < 0.0).astype(np.float64))

    ky = 2.0 * np.pi * np.fft.fftfreq(NY)
    left_energy = np.empty(NY)
    right_energy = np.empty(NY)
    left_occupation = np.empty(NY)
    right_occupation = np.empty(NY)
    left_weight = np.empty(NY)
    right_weight = np.empty(NY)
    h_offdiagonal = np.empty(NY)
    selected_energies = np.empty((NY, 2))
    left_profiles = np.empty((NY, NX))
    right_profiles = np.empty((NY, NX))

    for momentum in range(NY):
        chosen = np.argsort(np.abs(energies[momentum]), kind="stable")[:2]
        selected_energies[momentum] = np.sort(energies[momentum, chosen])
        subspace = eigenvectors[momentum][:, chosen]
        localized_operator = subspace.conj().T @ (contrast[:, None] * subspace)
        _, rotation = eigh(localized_operator, check_finite=False, driver="evd")
        localized = subspace @ rotation[:, ::-1]
        left_state = localized[:, 0]
        right_state = localized[:, 1]
        left_profiles[momentum] = (
            np.abs(left_state.reshape(NX, 2)) ** 2
        ).sum(axis=1)
        right_profiles[momentum] = (
            np.abs(right_state.reshape(NX, 2)) ** 2
        ).sum(axis=1)
        left_weight[momentum] = float(
            np.vdot(left_state, left_projector @ left_state).real
        )
        right_weight[momentum] = float(
            np.vdot(right_state, right_projector @ right_state).real
        )
        h_edge = localized.conj().T @ h_k[momentum] @ localized
        projector = (
            eigenvectors[momentum][:, occupied[momentum]]
            @ eigenvectors[momentum][:, occupied[momentum]].conj().T
        )
        c_edge = localized.conj().T @ projector @ localized
        left_energy[momentum] = float(h_edge[0, 0].real)
        right_energy[momentum] = float(h_edge[1, 1].real)
        left_occupation[momentum] = float(c_edge[0, 0].real)
        right_occupation[momentum] = float(c_edge[1, 1].real)
        h_offdiagonal[momentum] = float(abs(h_edge[0, 1]))

    trusted = (left_weight >= EDGE_WEIGHT_THRESHOLD) & (
        right_weight >= EDGE_WEIGHT_THRESHOLD
    )
    velocity_mask = trusted & (np.abs(ky) <= 0.55)
    left_velocity, left_intercept = np.polyfit(
        ky[velocity_mask], left_energy[velocity_mask], 1
    )
    right_velocity, right_intercept = np.polyfit(
        ky[velocity_mask], right_energy[velocity_mask], 1
    )
    return {
        "ky": ky,
        "selected_energies": selected_energies,
        "left_energy": left_energy,
        "right_energy": right_energy,
        "left_occupation": left_occupation,
        "right_occupation": right_occupation,
        "left_weight": left_weight,
        "right_weight": right_weight,
        "h_offdiagonal": h_offdiagonal,
        "left_profiles": left_profiles,
        "right_profiles": right_profiles,
        "trusted": trusted,
        "left_velocity": float(left_velocity),
        "right_velocity": float(right_velocity),
        "left_velocity_intercept": float(left_intercept),
        "right_velocity_intercept": float(right_intercept),
    }


def correlation_diagnostic(projector_delta: np.ndarray) -> dict[str, Any]:
    distance = np.arange(1, NY // 2, dtype=np.int64)
    values = {}
    for label, x in (("left", LEFT_WALL), ("right", RIGHT_WALL), ("bulk", 0)):
        modes = [2 * x, 2 * x + 1]
        values[label] = np.asarray(
            [
                np.linalg.norm(projector_delta[d][np.ix_(modes, modes)])
                for d in distance
            ],
            dtype=np.float64,
        )
    finite_ring_coordinate = np.abs(1.0 / np.tan(np.pi * distance / float(NY)))
    mask = (distance >= CORRELATION_FIT_MIN) & (
        distance <= CORRELATION_FIT_MAX
    )
    exponent, log_amplitude = np.polyfit(
        np.log(finite_ring_coordinate[mask]), np.log(values["left"][mask]), 1
    )
    fitted = exponent * np.log(finite_ring_coordinate[mask]) + log_amplitude
    residual = np.log(values["left"][mask]) - fitted
    total = np.log(values["left"][mask]) - np.log(values["left"][mask]).mean()
    return {
        "distance": distance,
        "finite_ring_coordinate": finite_ring_coordinate,
        **values,
        "wall_correlation_exponent": float(exponent),
        "wall_correlation_amplitude": float(np.exp(log_amplitude)),
        "wall_correlation_r2": float(
            1.0 - np.sum(residual**2) / np.sum(total**2)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.fonttype": 42,
        }
    )


def make_figure(
    output: Path,
    energies: np.ndarray,
    contour_x: np.ndarray,
    ay_values: np.ndarray,
    profiles: dict[str, np.ndarray],
    fits: list[dict[str, Any]],
    edge: dict[str, Any],
    correlation: dict[str, Any],
) -> None:
    configure_plotting()
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.25))
    ax = axes[0, 0]
    order = np.argsort(edge["ky"])
    ky = edge["ky"][order]
    for band in range(energies.shape[1]):
        ax.plot(ky, energies[order, band], color="0.85", lw=0.35, zorder=1)
    trusted = edge["trusted"][order]
    for values, color, marker, label in (
        (edge["left_energy"][order], "#d62728", "^", "left wall"),
        (edge["right_energy"][order], "#1f77b4", "o", "right wall"),
    ):
        ax.plot(ky[~trusted], values[~trusted], ls="none", marker=marker,
                ms=2.2, mfc="none", mec="0.65", mew=0.6, zorder=2)
        ax.plot(ky[trusted], values[trusted], ls="none", marker=marker,
                ms=3.0, color=color, label=label, zorder=3)
    ax.axhline(0.0, color="0.25", ls="--", lw=0.7)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-2.15, 2.15)
    ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"$k_y$")
    ax.set_ylabel(r"flattened-parent energy")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    ax = axes[0, 1]
    representative = [1, 5, 10, 15]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
    for index, color in zip(representative, colors):
        ax.plot(
            np.arange(NX),
            edge["left_profiles"][index],
            marker="o",
            ms=2.5,
            lw=1.0,
            color=color,
            label=rf"$k_y={edge['ky'][index] / np.pi:.2f}\pi$",
        )
    ax.axvline(LEFT_WALL, color="0.2", ls="--", lw=0.7)
    ax.axvline(RIGHT_WALL, color="0.2", ls="--", lw=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("left-channel weight")
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.legend(frameon=False, loc="upper right")

    fit_lookup = {(row["region"], row["radius"]): row for row in fits}
    ax = axes[1, 0]
    log_chord = np.log(
        (float(NY) / np.pi) * np.sin(np.pi * ay_values / float(NY))
    )
    series = (
        ("left_r0", "left wall", "#d62728", "^", ":"),
        ("right_r0", "right wall", "#1f77b4", "o", "-"),
        ("both_r0", "both walls", "#2ca02c", "s", "--"),
        ("full", "full strip", "0.2", "D", "-."),
    )
    for key, label, color, marker, linestyle in series:
        region = "full" if key == "full" else key.split("_r")[0]
        radius = -1 if key == "full" else 0
        row = fit_lookup[(region, radius)]
        y = profiles[key]
        ax.plot(log_chord, y - row["intercept"], ls="none", marker=marker,
                ms=2.8, mfc="none", mec=color, mew=0.7)
        ax.plot(log_chord, row["slope"] * log_chord, color=color,
                ls=linestyle, lw=1.0,
                label=rf"{label}: $c_{{\rm fit}}={row['c_fit']:.3f}$")
    ax.set_xlabel(r"$\ln[(N_y/\pi)\sin(\pi A_y/N_y)]$")
    ax.set_ylabel(r"contour entropy $-s_0$")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1, 1]
    distance = correlation["distance"]
    ax.semilogy(distance, correlation["left"], "^", ms=3.0,
                color="#d62728", label="left wall")
    ax.semilogy(distance, correlation["right"], "o", ms=2.8,
                mfc="none", color="#1f77b4", label="right wall")
    ax.semilogy(distance, correlation["bulk"], "s", ms=2.6,
                mfc="none", color="#2ca02c", label="bulk $x=0$")
    reference = (
        correlation["wall_correlation_amplitude"]
        * correlation["finite_ring_coordinate"]
        ** correlation["wall_correlation_exponent"]
    )
    ax.semilogy(distance, reference, color="0.2", ls="--", lw=0.9,
                label=rf"$|\cot(\pi d/N_y)|^{{{correlation['wall_correlation_exponent']:.3f}}}$")
    ax.set_xlabel(r"separation $d_y$")
    ax.set_ylabel(r"$\|C_x(d_y)\|_{\rm F}$")
    ax.set_ylim(1.0e-16, 1.0)
    ax.legend(frameon=False, loc="lower left")

    for letter, axis in zip("abcd", axes.ravel()):
        axis.text(-0.15, 1.04, f"({letter})", transform=axis.transAxes,
                  fontsize=9, va="bottom", ha="left")
    fig.suptitle(
        r"Hard-wall flattened ground state: $N_x=20$, $N_y=60$, $\alpha_1=1$",
        y=0.995,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98), pad=0.7, w_pad=0.9, h_pad=1.0)
    fig.savefig(output / "wall_projected_ground_state.pdf", bbox_inches="tight")
    fig.savefig(
        output / "wall_projected_ground_state.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    case = REFERENCE.Case(0, 0, 2, 20, "hard", "1", NSHELL, NY, ALPHA_1)
    model = REFERENCE._build_model(case)
    if tuple(int(value) for value in model.DW_loc) != (LEFT_WALL, RIGHT_WALL):
        raise RuntimeError(f"unexpected wall positions: {model.DW_loc}")
    h_k = build_flattened_blocks(model)
    energies, eigenvectors, occupied, projector_delta, diagnostics = (
        ground_state_projector(h_k)
    )
    ay_values, contour_x, occupation_min, occupation_max = (
        interval_entropy_contour(projector_delta)
    )
    fit_rows, contour_profiles = contour_fits(ay_values, contour_x)
    edge = edge_spectral_diagnostic(h_k, energies, eigenvectors, occupied)
    correlation = correlation_diagnostic(projector_delta)

    primary = {
        row["region"]: row
        for row in fit_rows
        if (row["radius"] == 0 and row["region"] in {"left", "right", "both"})
        or row["region"] == "full"
    }
    summary = {
        "schema": SCHEMA,
        "scientific_contract": {
            "Nx": NX,
            "Ny": NY,
            "wall": "hard/support-truncated",
            "DW_loc": [LEFT_WALL, RIGHT_WALL],
            "nshell": NSHELL,
            "alpha_1": ALPHA_1,
            "alpha_2": ALPHA_2,
            "filling": 0.5,
            "trial_orbital": "X",
            "parent": "exact flattened OW parent",
        },
        "interpretation": {
            "entropy_method": "additive x-resolved contour of the full pure-state interval entropy",
            "single_wall_expected_fit": 0.5,
            "two_wall_expected_fit": 1.0,
            "spectral_projection_rule": "two lowest-|E| states, localized by the left-minus-right wall-window operator",
            "spectral_trust_rule": f"both own-wall weights >= {EDGE_WEIGHT_THRESHOLD}",
            "warning": "the edge branches merge into the bulk outside the trusted momentum window; no all-k two-band boundary Hamiltonian is claimed",
        },
        "primary_entropy_fits": primary,
        "edge_diagnostic": {
            "trusted_momentum_count": int(np.count_nonzero(edge["trusted"])),
            "trusted_abs_ky_max": float(np.max(np.abs(edge["ky"][edge["trusted"]]))),
            "left_velocity": edge["left_velocity"],
            "right_velocity": edge["right_velocity"],
            "left_velocity_intercept": edge["left_velocity_intercept"],
            "right_velocity_intercept": edge["right_velocity_intercept"],
            "maximum_trusted_edge_hybridization": float(
                np.max(edge["h_offdiagonal"][edge["trusted"]])
            ),
        },
        "correlation_diagnostic": {
            "fit_distance_min": CORRELATION_FIT_MIN,
            "fit_distance_max": CORRELATION_FIT_MAX,
            "finite_ring_reference": "abs(cot(pi*d/Ny))",
            "exponent": correlation["wall_correlation_exponent"],
            "r2": correlation["wall_correlation_r2"],
        },
        "numerical_diagnostics": {
            **diagnostics,
            "restricted_occupation_min": occupation_min,
            "restricted_occupation_max": occupation_max,
        },
        "source_hashes": {
            "analyze_wall_projected_ground_state.py": sha256_file(Path(__file__)),
            "run_flattened_ground_state_large_ny.py": sha256_file(REFERENCE_RUNNER),
            "src/classA_U1FGTN.py": sha256_file(
                BUNDLE_ROOT / "src" / "classA_U1FGTN.py"
            ),
            "src/occupied_frame.py": sha256_file(
                BUNDLE_ROOT / "src" / "occupied_frame.py"
            ),
        },
    }

    edge_rows = []
    for index in np.argsort(edge["ky"]):
        edge_rows.append(
            {
                "ky_index": int(index),
                "ky": float(edge["ky"][index]),
                "left_energy": float(edge["left_energy"][index]),
                "right_energy": float(edge["right_energy"][index]),
                "left_occupation": float(edge["left_occupation"][index]),
                "right_occupation": float(edge["right_occupation"][index]),
                "left_wall_weight": float(edge["left_weight"][index]),
                "right_wall_weight": float(edge["right_weight"][index]),
                "edge_hybridization_abs": float(edge["h_offdiagonal"][index]),
                "trusted": bool(edge["trusted"][index]),
            }
        )
    write_csv(output / "entropy_contour_fits.csv", fit_rows)
    write_csv(output / "edge_spectrum.csv", edge_rows)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    np.savez_compressed(
        output / "wall_projected_ground_state_data.npz",
        schema=np.asarray(SCHEMA),
        energies=energies,
        occupied=occupied,
        projector_delta=projector_delta,
        ay_values=ay_values,
        entropy_contour_x=contour_x,
        entropy_full=contour_profiles["full"],
        entropy_left_wall_cell=contour_profiles["left_r0"],
        entropy_right_wall_cell=contour_profiles["right_r0"],
        entropy_both_wall_cells=contour_profiles["both_r0"],
        ky=edge["ky"],
        edge_left_energy=edge["left_energy"],
        edge_right_energy=edge["right_energy"],
        edge_left_occupation=edge["left_occupation"],
        edge_right_occupation=edge["right_occupation"],
        edge_left_weight=edge["left_weight"],
        edge_right_weight=edge["right_weight"],
        edge_trusted=edge["trusted"],
        edge_left_profiles=edge["left_profiles"],
        edge_right_profiles=edge["right_profiles"],
        correlation_distance=correlation["distance"],
        correlation_left=correlation["left"],
        correlation_right=correlation["right"],
        correlation_bulk=correlation["bulk"],
    )
    make_figure(
        output,
        energies,
        contour_x,
        ay_values,
        contour_profiles,
        fit_rows,
        edge,
        correlation,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
