#!/usr/bin/env python3
"""Finite-size and genuine-reversal audit for the exact flattened-H wall pump."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ny", nargs="+", type=int, default=[16, 24])
    parser.add_argument("--grid-points", type=int, default=65)
    parser.add_argument("--offsets", nargs="+", type=float, default=[-1e-7, 1e-7])
    parser.add_argument("--truncations", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--cpu-start", type=int, default=0)
    parser.add_argument("--cpu-end", type=int, default=7)
    parser.add_argument("--output-root", type=str, default="")
    return parser.parse_args()


ARGS = parse_args()
CPU_SET = set(range(ARGS.cpu_start, ARGS.cpu_end + 1))
try:
    AVAILABLE = os.sched_getaffinity(0)
    RESOLVED_CPUS = sorted(CPU_SET & AVAILABLE)
    if RESOLVED_CPUS:
        os.sched_setaffinity(0, RESOLVED_CPUS)
    else:
        RESOLVED_CPUS = sorted(AVAILABLE)
except (AttributeError, OSError):
    RESOLVED_CPUS = []
THREADS = max(1, len(RESOLVED_CPUS))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = str(THREADS)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve()
REPO_ROOT = next(p for p in [HERE.parent, *HERE.parents]
                 if (p / "src" / "fgtn" / "classA_U1FGTN.py").exists())
sys.path.insert(0, str(REPO_ROOT / "src"))
from fgtn.classA_U1FGTN import classA_U1FGTN


NX = 20
DW_INTERVAL = (5, 15)
NSHELL = 1
ALPHA_TOP = 1.0
ALPHA_TRIV = 30.0
TRIAL_ORBITAL = "X"
DEGENERACY_TOL = 1e-9
RUN_ID = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
DEFAULT_ROOT = (REPO_ROOT / "notebooks" / "flattened_hamiltonian_analysis" / "outputs"
                / "exact_wall_charge_pump_reversibility_v1")
OUTPUT_ROOT = Path(ARGS.output_root).resolve() if ARGS.output_root else DEFAULT_ROOT
OUTPUT_DIR = OUTPUT_ROOT / RUN_ID
SOURCE = REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py"


def frame_projector(frame: np.ndarray, nx: int, ny: int) -> np.ndarray:
    v = np.asarray(frame, dtype=np.complex128).reshape(2 * nx * ny, nx * ny, order="F")
    return v @ v.conj().T


def build_hflat(phi: float, ny: int, truncation: bool) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):
        model = classA_U1FGTN(
            Nx=NX, Ny=ny, DW=True, nshell=NSHELL, filling_frac=0.5,
            alpha_1=ALPHA_TOP, alpha_2=ALPHA_TRIV,
            trial_orbitals=TRIAL_ORBITAL, dw_truncation=truncation,
            twist_y=float(phi), dw_interval=DW_INTERVAL,
        )
        model.construct_OW_projectors(
            nshell=NSHELL, DW=True, trial_orbitals=TRIAL_ORBITAL,
            dw_truncation=truncation, twist_y=float(phi),
        )
    h = (frame_projector(model.WF_Ap, NX, ny) + frame_projector(model.WF_Bp, NX, ny)
         - frame_projector(model.WF_Am, NX, ny) - frame_projector(model.WF_Bm, NX, ny))
    return np.asarray(0.5 * (h + h.conj().T), dtype=np.complex128)


def y_blocks(matrix: np.ndarray, ny: int) -> tuple[np.ndarray, float]:
    orbitals = 2 * NX
    shaped = matrix.reshape(ny, orbitals, ny, orbitals)
    transformed = np.fft.ifft(np.fft.fft(shaped, axis=0), axis=2)
    blocks = np.stack([transformed[k, :, k, :] for k in range(ny)])
    off = transformed.copy()
    for k in range(ny):
        off[k, :, k, :] = 0
    return blocks, float(np.max(np.abs(off)))


def reconstruct_from_blocks(blocks: np.ndarray, ny: int) -> np.ndarray:
    orbitals = 2 * NX
    transformed = np.zeros((ny, orbitals, ny, orbitals), dtype=np.complex128)
    for k in range(ny):
        transformed[k, :, k, :] = blocks[k]
    shaped = np.fft.ifft(np.fft.fft(transformed, axis=2), axis=0)
    return shaped.reshape(2 * NX * ny, 2 * NX * ny)


def diagonalize(phi: float, ny: int, truncation: bool):
    h = build_hflat(phi, ny, truncation)
    blocks, off = y_blocks(h, ny)
    evals, vecs = [], []
    for block in blocks:
        e, v = np.linalg.eigh(block)
        evals.append(e)
        vecs.append(v)
    return h, np.asarray(evals), np.asarray(vecs), off


def lowest_occupations(evals: np.ndarray, ny: int) -> np.ndarray:
    occupied = np.zeros_like(evals, dtype=bool)
    order = np.argsort(evals, axis=None)[:NX * ny]
    occupied[np.unravel_index(order, evals.shape)] = True
    return occupied


def projector(vecs: np.ndarray, occupied: np.ndarray, ny: int) -> np.ndarray:
    blocks = np.empty((ny, 2 * NX, 2 * NX), dtype=np.complex128)
    for k in range(ny):
        v = vecs[k][:, occupied[k]]
        blocks[k] = v @ v.conj().T
    return reconstruct_from_blocks(blocks, ny)


def continue_basis(previous: np.ndarray, evals: np.ndarray, vecs: np.ndarray):
    tracked = np.empty_like(vecs)
    assigned_min = 1.0
    principal_min = 1.0
    for k in range(len(evals)):
        overlap = np.abs(previous[k].conj().T @ vecs[k]) ** 2
        rows, cols = linear_sum_assignment(-overlap)
        permutation = cols[np.argsort(rows)]
        e = evals[k, permutation]
        v = vecs[k][:, permutation]
        assigned_min = min(assigned_min, float(np.sqrt(overlap[np.arange(len(e)), permutation]).min()))
        unused = set(range(2 * NX))
        while unused:
            seed = min(unused)
            cluster = sorted(j for j in unused if abs(e[j] - e[seed]) < DEGENERACY_TOL)
            unused -= set(cluster)
            idx = np.asarray(cluster)
            ov = previous[k][:, idx].conj().T @ v[:, idx]
            u, singular, vh = np.linalg.svd(ov, full_matrices=False)
            v[:, idx] = v[:, idx] @ (vh.conj().T @ u.conj().T)
            principal_min = min(principal_min, float(singular.min()))
        tracked[k] = v
    return tracked, assigned_min, principal_min


def basin_charges(p: np.ndarray, ny: int) -> np.ndarray:
    x = np.tile(np.repeat(np.arange(NX), 2), ny)
    dl = np.minimum((x - DW_INTERVAL[0]) % NX, (DW_INTERVAL[0] - x) % NX)
    dr = np.minimum((x - DW_INTERVAL[1]) % NX, (DW_INTERVAL[1] - x) % NX)
    left = (dl < dr).astype(float) + 0.5 * (dl == dr)
    density = np.real(np.diag(p))
    return np.array([left @ density, (1.0 - left) @ density])


def pump_between(p0: np.ndarray, p1: np.ndarray, ny: int) -> tuple[float, float, float]:
    dq = basin_charges(p1, ny) - basin_charges(p0, ny)
    return float(0.5 * (dq[1] - dq[0])), float(dq[0]), float(dq[1])


def follow_cached(cache, initial_vecs, occupied, ny):
    previous = initial_vecs.copy()
    assigned = principal = 1.0
    for _, evals, raw_vecs in cache[1:]:
        previous, a, p = continue_basis(previous, evals, raw_vecs)
        assigned, principal = min(assigned, a), min(principal, p)
    return previous, projector(previous, occupied, ny), assigned, principal


def run_case(ny: int, truncation: bool, phi0: float) -> tuple[dict, list[dict]]:
    forward_phis = phi0 + np.linspace(0.0, 2 * np.pi, ARGS.grid_points)
    negative_phis = phi0 - np.linspace(0.0, 2 * np.pi, ARGS.grid_points)
    cache = []
    hermiticity = offdiag = 0.0
    h0 = hend = None
    for index, phi in enumerate(forward_phis):
        h, evals, vecs, off = diagonalize(float(phi), ny, truncation)
        h0 = h if index == 0 else h0
        hend = h
        hermiticity = max(hermiticity, float(np.max(np.abs(h - h.conj().T))))
        offdiag = max(offdiag, off)
        cache.append((float(phi), evals, vecs))

    initial_vecs = cache[0][2]
    occupied = lowest_occupations(cache[0][1], ny)
    p0 = projector(initial_vecs, occupied, ny)
    forward_vecs, p_forward, af, pf = follow_cached(cache, initial_vecs, occupied, ny)

    reverse_cache = list(reversed(cache))
    returned_vecs, p_return, ar, pr = follow_cached(reverse_cache, forward_vecs, occupied, ny)

    negative_cache = [cache[0]]
    for phi in negative_phis[1:]:
        _, evals, vecs, _ = diagonalize(float(phi), ny, truncation)
        negative_cache.append((float(phi), evals, vecs))
    negative_vecs, p_negative, an, pn = follow_cached(negative_cache, initial_vecs, occupied, ny)

    qf, dlf, drf = pump_between(p0, p_forward, ny)
    qn, dln, drn = pump_between(p0, p_negative, ny)
    qundo, dlundo, drundo = pump_between(p_forward, p_return, ny)
    dim = p0.shape[0]
    y = np.repeat(np.arange(ny), 2 * NX)
    phase = np.exp(1j * 2 * np.pi * y / ny)
    gauged_h0 = phase[:, None] * h0 * phase.conj()[None, :]
    large_gauge_h_error = float(np.max(np.abs(hend - gauged_h0)))

    summary = {
        "Ny": ny, "dw_truncation": truncation, "phi0": phi0,
        "grid_points": ARGS.grid_points,
        "q_forward": qf, "q_opposite_from_same_initial": qn,
        "same_initial_reversal_error": abs(qf + qn),
        "q_true_undo": qundo, "true_undo_error": abs(qf + qundo),
        "roundtrip_projector_frobenius_per_dim": float(np.linalg.norm(p_return - p0) / dim),
        "forward_left": dlf, "forward_right": drf,
        "opposite_left": dln, "opposite_right": drn,
        "undo_left": dlundo, "undo_right": drundo,
        "forward_assigned_min": af, "forward_principal_min": pf,
        "undo_assigned_min": ar, "undo_principal_min": pr,
        "opposite_assigned_min": an, "opposite_principal_min": pn,
        "hermiticity_max": hermiticity, "block_offdiag_max": offdiag,
        "large_gauge_h_error": large_gauge_h_error,
    }
    density = []
    for label, p_start, p_end in (
        ("forward", p0, p_forward), ("opposite_same_initial", p0, p_negative),
        ("true_undo", p_forward, p_return), ("roundtrip", p0, p_return),
    ):
        dn = np.real(np.diag(p_end - p_start)).reshape(ny, NX, 2).sum(axis=(0, 2))
        density.extend({"Ny": ny, "dw_truncation": truncation, "phi0": phi0,
                        "path": label, "x": x, "delta_n": float(value)}
                       for x, value in enumerate(dn))
    return summary, density


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    print(json.dumps({
        "Nx": NX, "Ny": ARGS.ny, "truncations": ARGS.truncations,
        "offsets": ARGS.offsets, "grid_points": ARGS.grid_points,
        "resolved_cpus": RESOLVED_CPUS, "output": str(OUTPUT_DIR),
    }, indent=2))
    started = time.perf_counter()
    summaries, densities = [], []
    total = len(ARGS.ny) * len(ARGS.truncations) * len(ARGS.offsets)
    index = 0
    for ny in ARGS.ny:
        for truncation_int in ARGS.truncations:
            for phi0 in ARGS.offsets:
                index += 1
                print(f"[{index}/{total}] Ny={ny}, trunc={bool(truncation_int)}, phi0={phi0:+.3e}", flush=True)
                summary, density = run_case(ny, bool(truncation_int), float(phi0))
                summaries.append(summary)
                densities.extend(density)
                print("  q+={q_forward:+.9f} q-={q_opposite_from_same_initial:+.9f} "
                      "undo={q_true_undo:+.9f} closure={roundtrip_projector_frobenius_per_dim:.3e}".format(**summary), flush=True)

    runtime = time.perf_counter() - started
    frame = pd.DataFrame(summaries)
    frame.to_csv(OUTPUT_DIR / "reversibility_summary.csv", index=False)
    pd.DataFrame(densities).to_csv(OUTPUT_DIR / "density_profiles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.2), constrained_layout=True)
    markers = {False: "o", True: "s"}
    colors = {-1: "tab:blue", 1: "tab:red"}
    for (truncation, phi0), group in frame.groupby(["dw_truncation", "phi0"]):
        group = group.sort_values("Ny")
        sign = -1 if phi0 < 0 else 1
        label = f"trunc={truncation}, phi0={phi0:+.0e}"
        axes[0].plot(group.Ny, group.q_forward, marker=markers[truncation], color=colors[sign],
                     ls="-" if not truncation else "--", label=label)
        axes[0].plot(group.Ny, -group.q_opposite_from_same_initial,
                     marker=markers[truncation], color=colors[sign], ls=":")
        axes[1].semilogy(group.Ny, group.true_undo_error.clip(lower=1e-18),
                        marker=markers[truncation], color=colors[sign],
                        ls="-" if not truncation else "--", label=label)
    axes[0].axhline(1, color="0.4", ls="--", lw=0.8)
    axes[0].set(xlabel=r"$N_y$", ylabel=r"sign-aligned transferred charge")
    axes[1].set(xlabel=r"$N_y$", ylabel=r"true-undo charge error")
    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].legend(fontsize=6, frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"reversibility_and_offset_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "run_id": RUN_ID, "runtime_seconds": runtime,
        "configuration": vars(ARGS), "resolved_cpus": RESOLVED_CPUS,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "interpretation": {
            "opposite_same_initial": "Independent opposite winding from the same instantaneous initial projector.",
            "true_undo": "Backward traversal initialized from the forward continued endpoint frame.",
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(frame.to_string(index=False))
    print(f"[complete] runtime={runtime / 60:.2f} min output={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
