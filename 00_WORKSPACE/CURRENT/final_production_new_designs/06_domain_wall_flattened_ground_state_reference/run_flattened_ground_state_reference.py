#!/usr/bin/env python3
"""Compute the deterministic half-filled OW flattened-parent reference curves."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = BUNDLE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from mutual_information_observer import (  # noqa: E402
    mutual_information_components_for_translation,
    strip_mode_indices,
)


SCHEMA = "domain_wall_ow_flattened_ground_state_reference_v1"
COMPLETION_SCHEMA = (
    "domain_wall_ow_flattened_ground_state_reference_completion_v1"
)
NX = 20
NY_VALUES = (20, 24, 28)
ALPHA_VALUES = (
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
)
WALLS = ("hard", "soft")
NSHELL = 1
ALPHA_2 = 30.0
TRIAL_ORBITAL = "X"
FIT_AY_MIN = 2
ENTROPY_EPS = 1.0e-12
TRANSLATION_TOLERANCE = 2.0e-10
REFERENCE_DIRNAME = "flattened_ground_state_reference"
PRODUCT_NAMES = (
    "flattened_ground_state_reference.npz",
    "flattened_ground_state_reference.csv",
    "flattened_ground_state_reference.pdf",
    "flattened_ground_state_reference.png",
)
SOURCE_FILES = (
    "run_flattened_ground_state_reference.py",
    "mutual_information_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(bundle_root: Path = BUNDLE_ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = bundle_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing flattened-reference source: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def reference_paths(output_root: Path) -> tuple[Path, ...]:
    root = Path(output_root) / REFERENCE_DIRNAME
    return tuple(root / name for name in PRODUCT_NAMES) + (
        root / "flattened_ground_state_reference.complete.json",
    )


def _identity(hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "result_schema": SCHEMA,
        "source_hashes": hashes,
        "Nx": NX,
        "Ny_values": list(NY_VALUES),
        "alpha_1_values": list(ALPHA_VALUES),
        "alpha_2": ALPHA_2,
        "walls": list(WALLS),
        "nshell": NSHELL,
        "trial_orbital": TRIAL_ORBITAL,
        "filling": 0.5,
        "fit_Ay_min": FIT_AY_MIN,
    }


def verified_reference(
    output_root: Path, hashes: dict[str, str]
) -> tuple[bool, str]:
    *products, completion_path = reference_paths(output_root)
    if not completion_path.is_file() and not any(path.is_file() for path in products):
        return False, "missing reference products"
    if not completion_path.is_file() or not all(path.is_file() for path in products):
        return False, "incomplete reference products"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion JSON: {exc}"
    for key, value in _identity(hashes).items():
        if completion.get(key) != value:
            return False, f"reference identity mismatch: {key}"
    declared = completion.get("products", {})
    for path in products:
        record = declared.get(path.name)
        if not isinstance(record, dict):
            return False, f"missing product record: {path.name}"
        if int(record.get("bytes", -1)) != path.stat().st_size:
            return False, f"byte count mismatch: {path.name}"
        if record.get("sha256") != sha256_file(path):
            return False, f"checksum mismatch: {path.name}"
    return True, "verified"


def validate_device(device_text: str, require_a100: bool) -> torch.device:
    if device_text == "auto":
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_text)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        name = torch.cuda.get_device_name(device)
        if require_a100 and "A100" not in name.upper():
            raise RuntimeError(f"an NVIDIA A100 is required; detected {name!r}")
        total_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(
            f"[device] {name}; memory={total_gib:.2f} GiB; dtype=complex128",
            flush=True,
        )
    else:
        if require_a100:
            raise RuntimeError("--require-a100 cannot be used with the CPU device")
        print(
            f"[device] CPU; torch_threads={torch.get_num_threads()}; dtype=complex128",
            flush=True,
        )
    return device


def _build_model(*, ny: int, alpha_1: float, wall: str, device: torch.device):
    if wall not in WALLS:
        raise ValueError(f"unknown wall {wall!r}")
    # The canonical constructor prints one line per model. Suppress that repeated
    # message because the campaign tqdm bar already identifies every case.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        model = classA_U1FGTN_gpu(
            Nx=NX,
            Ny=int(ny),
            DW=True,
            nshell=NSHELL,
            filling_frac=0.5,
            alpha_1=float(alpha_1),
            alpha_2=ALPHA_2,
            trial_orbitals=TRIAL_ORBITAL,
            dw_truncation=(wall == "hard"),
            triv_region_local_mode=False,
            device=device,
            dtype="complex128",
            backend="local",
        )
    return model


def flattened_parent_and_projector(model: Any) -> tuple[torch.Tensor, ...]:
    """Return H_flat, its spectrum, and the lowest-half occupied projector."""

    modes = 2 * int(model.Nx) * int(model.Ny)
    frames = [
        getattr(model, name).reshape(modes, -1)
        for name in ("WF_Ap", "WF_Bp", "WF_Am", "WF_Bm")
    ]
    hamiltonian = (
        frames[0] @ frames[0].mH
        + frames[1] @ frames[1].mH
        - frames[2] @ frames[2].mH
        - frames[3] @ frames[3].mH
    )
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.mH)
    eigenvalues, eigenvectors = torch.linalg.eigh(hamiltonian)
    rank = modes // 2
    occupied = eigenvectors[:, :rank]
    projector = occupied @ occupied.mH
    projector = 0.5 * (projector + projector.mH)
    return hamiltonian, eigenvalues.real, projector


def entropy_from_projector(
    projector: torch.Tensor, indices: torch.Tensor
) -> tuple[float, float, float]:
    indices = indices.to(projector.device)
    restricted = projector.index_select(0, indices).index_select(1, indices)
    restricted = 0.5 * (restricted + restricted.mH)
    eigenvalues = torch.linalg.eigvalsh(restricted).real
    eigen_min = float(eigenvalues.min().detach().cpu())
    eigen_max = float(eigenvalues.max().detach().cpu())
    if eigen_min < -1.0e-8 or eigen_max > 1.0 + 1.0e-8:
        raise FloatingPointError(
            "restricted occupation spectrum left [0,1]: "
            f"min={eigen_min:.6e}, max={eigen_max:.6e}"
        )
    p = eigenvalues.clamp(ENTROPY_EPS, 1.0 - ENTROPY_EPS)
    entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).sum()
    return float(entropy.detach().cpu()), eigen_min, eigen_max


def fit_central_charge(
    ay_values: np.ndarray, entropies: np.ndarray, ny: int
) -> dict[str, float]:
    ay_values = np.asarray(ay_values, dtype=np.int64)
    entropies = np.asarray(entropies, dtype=np.float64)
    log_chord = np.log(
        (float(ny) / np.pi) * np.sin(np.pi * ay_values.astype(float) / float(ny))
    )
    mask = (ay_values >= FIT_AY_MIN) & (ay_values <= int(ny) // 2)
    x = log_chord[mask]
    y = entropies[mask]
    if x.size < 3:
        raise ValueError("central-charge fit needs at least three points")
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    variance = residual_sum / float(x.size - 2)
    slope_se = float(np.sqrt(max(variance, 0.0) / sxx))
    intercept_se = float(
        np.sqrt(max(variance, 0.0) * (1.0 / x.size + np.mean(x) ** 2 / sxx))
    )
    r2 = (
        1.0 if np.isclose(total_sum, 0.0) and np.isclose(residual_sum, 0.0)
        else 0.0 if np.isclose(total_sum, 0.0)
        else 1.0 - residual_sum / total_sum
    )
    return {
        "slope": float(slope),
        "slope_se": slope_se,
        "intercept": float(intercept),
        "intercept_se": intercept_se,
        "c_fit": float(3.0 * slope),
        "c_fit_se": float(3.0 * slope_se),
        "r2": float(r2),
    }


def compute_case(
    *, ny: int, alpha_1: float, wall: str, device: torch.device
) -> dict[str, Any]:
    model = _build_model(ny=ny, alpha_1=alpha_1, wall=wall, device=device)
    hamiltonian, energies, projector = flattened_parent_and_projector(model)
    modes = 2 * NX * ny
    rank = modes // 2
    identity = torch.eye(modes, dtype=torch.complex128, device=device)
    centered = (2.0 * projector - identity).unsqueeze(0)

    h_hermiticity = float(torch.max(torch.abs(hamiltonian - hamiltonian.mH)).cpu())
    p_hermiticity = float(torch.max(torch.abs(projector - projector.mH)).cpu())
    p_idempotency = float(torch.max(torch.abs(projector @ projector - projector)).cpu())
    rank_residual = abs(float(torch.trace(projector).real.cpu()) - float(rank))
    p4 = projector.reshape(ny, 2 * NX, ny, 2 * NX)
    translated = torch.roll(p4, shifts=(1, 1), dims=(0, 2))
    translation_residual = float(torch.max(torch.abs(p4 - translated)).cpu())
    if translation_residual > TRANSLATION_TOLERANCE:
        raise FloatingPointError(
            "flattened ground state is not translation invariant in y: "
            f"{translation_residual:.6e}"
        )

    entropy_a_values: list[float] = []
    entropy_b_values: list[float] = []
    entropy_union_values: list[float] = []
    mutual_information_values: list[float] = []
    restricted_min = np.inf
    restricted_max = -np.inf
    for y0 in range(ny // 2):
        entropy_a, entropy_b, entropy_union, mutual_information, diagnostics = (
            mutual_information_components_for_translation(
                centered,
                nx=NX,
                ny=ny,
                width=ny // 4,
                y0=y0,
                eps=ENTROPY_EPS,
            )
        )
        entropy_a_values.append(float(entropy_a.item()))
        entropy_b_values.append(float(entropy_b.item()))
        entropy_union_values.append(float(entropy_union.item()))
        mutual_information_values.append(float(mutual_information.item()))
        restricted_min = min(restricted_min, diagnostics.min_occupation_eigenvalue)
        restricted_max = max(restricted_max, diagnostics.max_occupation_eigenvalue)

    # The omitted half orbit exchanges A and B exactly, matching the trajectory
    # observer's estimator order and averaging convention.
    sa = np.asarray(entropy_a_values, dtype=np.float64)
    sb = np.asarray(entropy_b_values, dtype=np.float64)
    su = np.asarray(entropy_union_values, dtype=np.float64)
    mi = np.asarray(mutual_information_values, dtype=np.float64)
    entropy_single = float(0.5 * np.mean(sa + sb))
    entropy_union = float(np.mean(su))
    mutual_information = float(np.mean(mi))

    ay_values = np.arange(1, ny // 2 + 1, dtype=np.int64)
    entropy_profile = []
    for ay in ay_values:
        indices = strip_mode_indices(
            nx=NX, ny=ny, width=int(ay), y0=0, device=device
        )
        entropy, eigen_min, eigen_max = entropy_from_projector(projector, indices)
        entropy_profile.append(entropy)
        restricted_min = min(restricted_min, eigen_min)
        restricted_max = max(restricted_max, eigen_max)
    entropy_profile_array = np.asarray(entropy_profile, dtype=np.float64)

    # A full y0 average equals the representative profile because P is block
    # circulant in y. Verify the equality numerically at every fit endpoint by
    # repeating the profile at y0=1.
    shifted_profile = []
    for ay in ay_values:
        indices = strip_mode_indices(
            nx=NX, ny=ny, width=int(ay), y0=1, device=device
        )
        entropy, _, _ = entropy_from_projector(projector, indices)
        shifted_profile.append(entropy)
    profile_y0_residual = float(
        np.max(np.abs(entropy_profile_array - np.asarray(shifted_profile)))
    )
    if profile_y0_residual > TRANSLATION_TOLERANCE:
        raise FloatingPointError(
            "representative and shifted entropy profiles differ: "
            f"{profile_y0_residual:.6e}"
        )
    fit = fit_central_charge(ay_values, entropy_profile_array, ny)

    record: dict[str, Any] = {
        "wall": wall,
        "dw_truncation": wall == "hard",
        "Nx": NX,
        "Ny": ny,
        "width": ny // 4,
        "alpha_1": float(alpha_1),
        "alpha_2": ALPHA_2,
        "nshell": NSHELL,
        "trial_orbital": TRIAL_ORBITAL,
        "half_filling_rank": rank,
        "negative_energy_count": int(torch.count_nonzero(energies < 0.0).cpu()),
        "half_filling_gap": float((energies[rank] - energies[rank - 1]).cpu()),
        "minimum_absolute_energy": float(torch.min(torch.abs(energies)).cpu()),
        "mutual_information_y0avg": mutual_information,
        "entropy_a_y0avg": entropy_single,
        "entropy_b_y0avg": entropy_single,
        "entropy_union_y0avg": entropy_union,
        "mutual_information_y0_spread": float(np.ptp(mi)),
        "hamiltonian_hermiticity_max_abs": h_hermiticity,
        "projector_hermiticity_max_abs": p_hermiticity,
        "projector_idempotency_max_abs": p_idempotency,
        "projector_rank_residual": rank_residual,
        "projector_y_translation_max_abs": translation_residual,
        "entropy_profile_y0_shift_max_abs": profile_y0_residual,
        "restricted_occupation_min": float(restricted_min),
        "restricted_occupation_max": float(restricted_max),
        "Ay_values": ay_values,
        "entropy_profile_y0avg": entropy_profile_array,
        **fit,
    }
    del centered, identity, projector, energies, hamiltonian, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def compute_sweep(device: torch.device) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    shape = (len(WALLS), len(NY_VALUES), len(ALPHA_VALUES))
    scalar_keys = (
        "mutual_information_y0avg",
        "entropy_a_y0avg",
        "entropy_b_y0avg",
        "entropy_union_y0avg",
        "mutual_information_y0_spread",
        "c_fit",
        "c_fit_se",
        "slope",
        "slope_se",
        "intercept",
        "intercept_se",
        "r2",
        "half_filling_gap",
        "minimum_absolute_energy",
        "hamiltonian_hermiticity_max_abs",
        "projector_hermiticity_max_abs",
        "projector_idempotency_max_abs",
        "projector_rank_residual",
        "projector_y_translation_max_abs",
        "entropy_profile_y0_shift_max_abs",
        "restricted_occupation_min",
        "restricted_occupation_max",
    )
    integer_keys = ("half_filling_rank", "negative_energy_count")
    arrays = {key: np.full(shape, np.nan, dtype=np.float64) for key in scalar_keys}
    arrays.update({key: np.full(shape, -1, dtype=np.int64) for key in integer_keys})
    max_ay = max(NY_VALUES) // 2
    entropy_profiles = np.full(shape + (max_ay,), np.nan, dtype=np.float64)
    ay_grid = np.full((len(NY_VALUES), max_ay), -1, dtype=np.int64)
    records: list[dict[str, Any]] = []

    cases = [
        (wall_index, ny_index, alpha_index, wall, ny, alpha)
        for wall_index, wall in enumerate(WALLS)
        for ny_index, ny in enumerate(NY_VALUES)
        for alpha_index, alpha in enumerate(ALPHA_VALUES)
    ]
    bar = tqdm(
        cases,
        total=len(cases),
        desc="flattened ground-state sweep",
        unit="case",
        dynamic_ncols=True,
    )
    for wall_index, ny_index, alpha_index, wall, ny, alpha in bar:
        bar.set_postfix(wall=wall, Ny=ny, alpha=f"{alpha:g}")
        record = compute_case(ny=ny, alpha_1=alpha, wall=wall, device=device)
        target = (wall_index, ny_index, alpha_index)
        for key in scalar_keys + integer_keys:
            arrays[key][target] = record[key]
        count = len(record["Ay_values"])
        entropy_profiles[target + (slice(0, count),)] = record[
            "entropy_profile_y0avg"
        ]
        ay_grid[ny_index, :count] = record["Ay_values"]
        records.append(record)

    arrays.update(
        {
            "schema": np.asarray(SCHEMA),
            "wall_labels": np.asarray(WALLS),
            "Ny_values": np.asarray(NY_VALUES, dtype=np.int64),
            "width_values": np.asarray([ny // 4 for ny in NY_VALUES], dtype=np.int64),
            "alpha_1_values": np.asarray(ALPHA_VALUES, dtype=np.float64),
            "alpha_2": np.asarray(ALPHA_2, dtype=np.float64),
            "nshell": np.asarray(NSHELL, dtype=np.int64),
            "trial_orbital": np.asarray(TRIAL_ORBITAL),
            "filling": np.asarray(0.5, dtype=np.float64),
            "fit_Ay_min": np.asarray(FIT_AY_MIN, dtype=np.int64),
            "fit_Ay_max_by_Ny": np.asarray(
                [ny // 2 for ny in NY_VALUES], dtype=np.int64
            ),
            "Ay_values_by_Ny": ay_grid,
            "entropy_profile_y0avg": entropy_profiles,
            "entropy_profile_translation_reduction": np.asarray(
                "representative y0=0 after projector and y0=1 entropy checks"
            ),
            "mutual_information_translation_reduction": np.asarray(
                "explicit Ny/2 placements; omitted half exchanges A and B"
            ),
            "flattened_parent_definition": np.asarray(
                "sum_R(P_A+ + P_B+ - P_A- - P_B-)"
            ),
            "ground_state_rule": np.asarray(
                "lowest Nlayer/2 eigenvectors (exact half filling)"
            ),
        }
    )
    return arrays, records


def _plot_style() -> tuple[dict[str, Any], ...]:
    return (
        {"color": "#d62728", "marker": "^", "linestyle": ":"},
        {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
        {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    )


def make_figure(payload: dict[str, np.ndarray], pdf: Path, png: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
        }
    )
    alpha = np.asarray(payload["alpha_1_values"], dtype=np.float64)
    mi = np.asarray(payload["mutual_information_y0avg"], dtype=np.float64)
    c_fit = np.asarray(payload["c_fit"], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.4), sharex=True)
    for wall_index, wall in enumerate(WALLS):
        for ny_index, (ny, style) in enumerate(zip(NY_VALUES, _plot_style())):
            axes[0, wall_index].plot(
                alpha,
                mi[wall_index, ny_index],
                label=rf"$N_y={ny}$",
                linewidth=1.2,
                markersize=3.5,
                markeredgewidth=0.6,
                **style,
            )
            axes[1, wall_index].plot(
                alpha,
                c_fit[wall_index, ny_index],
                label=rf"$N_y={ny}$",
                linewidth=1.2,
                markersize=3.5,
                markeredgewidth=0.6,
                **style,
            )
        for row in range(2):
            axes[row, wall_index].axvline(
                2.0, color="0.25", linestyle="--", linewidth=0.9
            )
            axes[row, wall_index].tick_params(direction="in", top=True, right=True)
            for spine in axes[row, wall_index].spines.values():
                spine.set_visible(True)
        axes[0, wall_index].set_title(
            "hard / support-truncated" if wall == "hard"
            else "soft / untruncated"
        )
        axes[1, wall_index].set_xlabel(r"$\alpha_1$")
    for axis in axes[1]:
        axis.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
    axes[0, 0].set_ylabel(r"$I_{a,b}^{\rm flat}$")
    axes[1, 0].set_ylabel(r"$c_{\rm fit}=3m$")
    axes[0, 0].legend(frameon=False, loc="best")
    for axis, label in zip(axes.ravel(), ("(a)", "(b)", "(c)", "(d)")):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes)
    fig.tight_layout()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "wall",
        "dw_truncation",
        "Nx",
        "Ny",
        "width",
        "alpha_1",
        "alpha_2",
        "nshell",
        "trial_orbital",
        "half_filling_rank",
        "negative_energy_count",
        "half_filling_gap",
        "minimum_absolute_energy",
        "mutual_information_y0avg",
        "entropy_a_y0avg",
        "entropy_b_y0avg",
        "entropy_union_y0avg",
        "c_fit",
        "c_fit_se",
        "slope",
        "slope_se",
        "intercept",
        "intercept_se",
        "r2",
        "mutual_information_y0_spread",
        "projector_y_translation_max_abs",
        "entropy_profile_y0_shift_max_abs",
        "hamiltonian_hermiticity_max_abs",
        "projector_hermiticity_max_abs",
        "projector_idempotency_max_abs",
        "projector_rank_residual",
        "restricted_occupation_min",
        "restricted_occupation_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copy2(local_path, temporary)
        if temporary.stat().st_size != local_path.stat().st_size:
            raise OSError(f"published byte count mismatch for {final_path.name}")
        if sha256_file(temporary) != sha256_file(local_path):
            raise OSError(f"published checksum mismatch for {final_path.name}")
        os.replace(temporary, final_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bytes": final_path.stat().st_size, "sha256": sha256_file(final_path)}


def run(
    *,
    output_root: Path,
    scratch_root: Path,
    device_text: str,
    require_a100: bool,
    report_only: bool,
    force: bool,
) -> dict[str, Any]:
    hashes = source_hashes()
    verified, reason = verified_reference(output_root, hashes)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_directory": str(Path(output_root) / REFERENCE_DIRNAME),
                "scratch_directory": str(scratch_root),
                "workload": {
                    "cases": len(WALLS) * len(NY_VALUES) * len(ALPHA_VALUES),
                    "walls": list(WALLS),
                    "Ny_values": list(NY_VALUES),
                    "alpha_1_values": list(ALPHA_VALUES),
                    "fit_Ay": f"{FIT_AY_MIN}..Ny//2",
                },
                "existing_status": {"verified": verified, "reason": reason},
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if report_only:
        return {"status": "verified" if verified else "missing_or_invalid", "reason": reason}
    if verified and not force:
        print("[reference] verified existing products; skipping", flush=True)
        return {"status": "existing", "reason": reason}

    device = validate_device(device_text, require_a100)
    started = time.monotonic()
    payload, records = compute_sweep(device)
    elapsed = time.monotonic() - started
    if len(records) != len(WALLS) * len(NY_VALUES) * len(ALPHA_VALUES):
        raise RuntimeError("flattened sweep returned the wrong number of cases")
    if not np.isfinite(payload["mutual_information_y0avg"]).all():
        raise FloatingPointError("nonfinite mutual information in flattened sweep")
    if not np.isfinite(payload["c_fit"]).all():
        raise FloatingPointError("nonfinite central-charge fit in flattened sweep")

    local_root = Path(scratch_root) / REFERENCE_DIRNAME
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True)
    local_npz = local_root / PRODUCT_NAMES[0]
    local_csv = local_root / PRODUCT_NAMES[1]
    local_pdf = local_root / PRODUCT_NAMES[2]
    local_png = local_root / PRODUCT_NAMES[3]
    write_npz(local_npz, payload)
    write_csv(local_csv, records)
    make_figure(payload, local_pdf, local_png)

    *final_products, completion_path = reference_paths(output_root)
    products: dict[str, dict[str, Any]] = {}
    for local_path, final_path in zip(
        (local_npz, local_csv, local_pdf, local_png), final_products
    ):
        products[final_path.name] = publish_file(local_path, final_path)
    completion = {
        **_identity(hashes),
        "products": products,
        "elapsed_seconds": elapsed,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "diagnostic_maxima": {
            "mutual_information_y0_spread": float(
                np.max(payload["mutual_information_y0_spread"])
            ),
            "projector_y_translation_max_abs": float(
                np.max(payload["projector_y_translation_max_abs"])
            ),
            "entropy_profile_y0_shift_max_abs": float(
                np.max(payload["entropy_profile_y0_shift_max_abs"])
            ),
            "projector_idempotency_max_abs": float(
                np.max(payload["projector_idempotency_max_abs"])
            ),
        },
    }
    local_completion = local_root / completion_path.name
    with local_completion.open("w", encoding="utf-8") as handle:
        json.dump(completion, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    publish_file(local_completion, completion_path)
    verified, reason = verified_reference(output_root, hashes)
    if not verified:
        raise OSError(f"published flattened reference failed verification: {reason}")
    shutil.rmtree(local_root)
    print(
        "[reference summary] "
        + json.dumps(
            {
                "status": "complete",
                "cases": len(records),
                "elapsed_seconds": elapsed,
                "output_directory": str(completion_path.parent),
                "verification": reason,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {"status": "complete", "cases": len(records), "reason": reason}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/tmp/domain_wall_flattened_reference_scratch"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--require-a100", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        device_text=args.device,
        require_a100=bool(args.require_a100),
        report_only=bool(args.report_only),
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
