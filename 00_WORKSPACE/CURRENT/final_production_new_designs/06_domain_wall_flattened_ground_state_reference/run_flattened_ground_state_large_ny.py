#!/usr/bin/env python3
"""Local CPU sweep of large-Ny flattened domain-wall ground states."""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = BUNDLE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classA_U1FGTN import classA_U1FGTN  # noqa: E402


SCHEMA = "domain_wall_ow_flattened_ground_state_large_ny_v1"
TASK_SCHEMA = f"{SCHEMA}_task_v1"
TASK_COMPLETION_SCHEMA = f"{SCHEMA}_task_completion_v1"
AGGREGATE_COMPLETION_SCHEMA = f"{SCHEMA}_aggregate_completion_v1"
NX = 20
NY_VALUES = (40, 50, 60)
NSHELL_LABELS = ("1", "2", "inf")
NSHELL_VALUES = (1, 2, None)
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
ALPHA_2 = 30.0
TRIAL_ORBITAL = "X"
FIT_AY_MIN = 2
ENTROPY_EPS = 1.0e-12
TRANSLATION_TOLERANCE = 2.0e-10
RESULT_DIRNAME = "flattened_ground_state_large_ny"
SOURCE_FILES = (
    "run_flattened_ground_state_large_ny.py",
    "src/classA_U1FGTN.py",
    "src/occupied_frame.py",
)
PRODUCT_NAMES = (
    "flattened_ground_state_large_ny.npz",
    "flattened_ground_state_large_ny.csv",
    "flattened_ground_state_large_ny.pdf",
    "flattened_ground_state_large_ny.png",
    "mutual_information_geometry.pdf",
    "mutual_information_geometry.png",
)


@dataclass(frozen=True)
class Case:
    wall_index: int
    nshell_index: int
    ny_index: int
    alpha_index: int
    wall: str
    nshell_label: str
    nshell: int | None
    ny: int
    alpha_1: float

    @property
    def width(self) -> int:
        return self.ny // 4

    @property
    def case_id(self) -> str:
        return (
            f"w{self.wall_index:02d}_n{self.nshell_index:02d}_"
            f"y{self.ny_index:02d}_a{self.alpha_index:02d}"
        )


def cases() -> list[Case]:
    return [
        Case(wi, si, yi, ai, wall, label, nshell, ny, alpha)
        for wi, wall in enumerate(WALLS)
        for si, (label, nshell) in enumerate(zip(NSHELL_LABELS, NSHELL_VALUES))
        for yi, ny in enumerate(NY_VALUES)
        for ai, alpha in enumerate(ALPHA_VALUES)
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(bundle_root: Path = BUNDLE_ROOT) -> dict[str, str]:
    return {
        relative: sha256_file(bundle_root / relative) for relative in SOURCE_FILES
    }


def campaign_identity(hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source_hashes": hashes,
        "Nx": NX,
        "Ny_values": list(NY_VALUES),
        "width_rule": "Ny//4",
        "opposite_displacement_rule": "Ny//2",
        "nshell_labels": list(NSHELL_LABELS),
        "nshell_values": [1, 2, None],
        "alpha_1_values": list(ALPHA_VALUES),
        "alpha_2": ALPHA_2,
        "walls": list(WALLS),
        "trial_orbital": TRIAL_ORBITAL,
        "filling": 0.5,
        "fit_Ay_min": FIT_AY_MIN,
        "canonical_cpu_source": "classA_U1FGTN",
        "flattened_parent": "sum_R(P_A+ + P_B+ - P_A- - P_B-)",
    }


def task_paths(output_root: Path, case: Case) -> tuple[Path, Path]:
    task_root = Path(output_root) / RESULT_DIRNAME / "tasks"
    return (
        task_root / f"{case.case_id}.npz",
        task_root / f"{case.case_id}.complete.json",
    )


def case_identity(case: Case, hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": TASK_COMPLETION_SCHEMA,
        "result_schema": TASK_SCHEMA,
        "source_hashes": hashes,
        "case": asdict(case),
        "width": case.width,
        "Nx": NX,
        "alpha_2": ALPHA_2,
        "trial_orbital": TRIAL_ORBITAL,
        "filling": 0.5,
        "fit_Ay_min": FIT_AY_MIN,
    }


def verified_task(
    output_root: Path, case: Case, hashes: dict[str, str]
) -> tuple[bool, str]:
    result_path, completion_path = task_paths(output_root, case)
    if not result_path.is_file() and not completion_path.is_file():
        return False, "missing"
    if not result_path.is_file() or not completion_path.is_file():
        return False, "incomplete pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion: {exc}"
    for key, value in case_identity(case, hashes).items():
        if completion.get(key) != value:
            return False, f"identity mismatch: {key}"
    if completion.get("result_file") != result_path.name:
        return False, "result filename mismatch"
    if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
        return False, "result byte count mismatch"
    if completion.get("result_sha256") != sha256_file(result_path):
        return False, "result checksum mismatch"
    return True, "verified"


def _atomic_write_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_task(
    output_root: Path,
    case: Case,
    record: dict[str, Any],
    hashes: dict[str, str],
) -> None:
    result_path, completion_path = task_paths(output_root, case)
    payload: dict[str, Any] = {
        "schema": np.asarray(TASK_SCHEMA),
        "case_id": np.asarray(case.case_id),
        "wall": np.asarray(case.wall),
        "nshell_label": np.asarray(case.nshell_label),
        "nshell": np.asarray(-1 if case.nshell is None else case.nshell),
        "Nx": np.asarray(NX),
        "Ny": np.asarray(case.ny),
        "width": np.asarray(case.width),
        "alpha_1": np.asarray(case.alpha_1),
        "alpha_2": np.asarray(ALPHA_2),
    }
    payload.update({key: np.asarray(value) for key, value in record.items()})
    _atomic_write_npz(result_path, payload)
    completion = {
        **case_identity(case, hashes),
        "result_file": result_path.name,
        "result_bytes": result_path.stat().st_size,
        "result_sha256": sha256_file(result_path),
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write_json(completion_path, completion)
    verified, reason = verified_task(output_root, case, hashes)
    if not verified:
        raise OSError(f"published case {case.case_id} failed verification: {reason}")


def _build_model(case: Case) -> classA_U1FGTN:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        model = classA_U1FGTN(
            Nx=NX,
            Ny=case.ny,
            DW=True,
            nshell=case.nshell,
            filling_frac=0.5,
            alpha_1=case.alpha_1,
            alpha_2=ALPHA_2,
            trial_orbitals=TRIAL_ORBITAL,
            dw_truncation=(case.wall == "hard"),
        )
        model.construct_OW_projectors(
            nshell=case.nshell,
            DW=True,
            trial_orbitals=TRIAL_ORBITAL,
            dw_truncation=(case.wall == "hard"),
        )
    return model


def flattened_momentum_projector(
    model: classA_U1FGTN,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagonalize the exact flattened parent in conserved y momentum."""

    ny = int(model.Ny)
    q = 2 * int(model.Nx)
    h_k = np.zeros((ny, q, q), dtype=np.complex128)
    translation_covariance = 0.0
    phase = np.exp(
        -2j
        * np.pi
        * np.arange(ny, dtype=np.float64)[:, None]
        * np.arange(ny, dtype=np.float64)[None, :]
        / float(ny)
    )
    channels = (
        ("WF_Ap", 1.0),
        ("WF_Bp", 1.0),
        ("WF_Am", -1.0),
        ("WF_Bm", -1.0),
    )
    for name, sign in channels:
        frame = np.asarray(getattr(model, name), dtype=np.complex128)
        frame = frame.reshape(ny, q, NX, ny)
        frame_k = np.fft.fft(frame, axis=0) / np.sqrt(float(ny))
        expected = frame_k[:, :, :, :1] * phase[:, None, None, :]
        translation_covariance = max(
            translation_covariance, float(np.max(np.abs(frame_k - expected)))
        )
        representative = frame_k[:, :, :, 0]
        h_k += sign * float(ny) * np.einsum(
            "kar,kbr->kab",
            representative,
            representative.conj(),
            optimize=True,
        )
    h_hermiticity = float(
        np.max(np.abs(h_k - h_k.conj().transpose(0, 2, 1)))
    )
    h_k = 0.5 * (h_k + h_k.conj().transpose(0, 2, 1))

    energies = np.empty((ny, q), dtype=np.float64)
    eigenvectors = np.empty((ny, q, q), dtype=np.complex128)
    for momentum in range(ny):
        energies[momentum], eigenvectors[momentum] = eigh(
            h_k[momentum], check_finite=False, driver="evd"
        )
    total_rank = ny * q // 2
    order = np.argsort(energies.ravel(), kind="stable")
    occupied = np.zeros(ny * q, dtype=bool)
    occupied[order[:total_rank]] = True
    occupied = occupied.reshape(ny, q)
    projector_k = np.empty_like(h_k)
    for momentum in range(ny):
        vectors = eigenvectors[momentum][:, occupied[momentum]]
        projector_k[momentum] = vectors @ vectors.conj().T
    projector_delta = np.fft.ifft(projector_k, axis=0)

    diagnostics = {
        "half_filling_rank": total_rank,
        "negative_energy_count": int(np.count_nonzero(energies < 0.0)),
        "half_filling_gap": float(
            energies.ravel()[order[total_rank]]
            - energies.ravel()[order[total_rank - 1]]
        ),
        "minimum_absolute_energy": float(np.min(np.abs(energies))),
        "occupied_rank_by_ky": occupied.sum(axis=1).astype(np.int64),
        "ow_y_translation_covariance_max_abs": translation_covariance,
        "hamiltonian_block_hermiticity_max_abs": h_hermiticity,
        "projector_block_hermiticity_max_abs": float(
            np.max(
                [
                    np.max(
                        np.abs(
                            projector_delta[delta]
                            - projector_delta[-delta % ny].conj().T
                        )
                    )
                    for delta in range(ny)
                ]
            )
        ),
        "projector_idempotency_max_abs": float(
            np.max(
                [
                    np.max(np.abs(block @ block - block))
                    for block in projector_k
                ]
            )
        ),
        "projector_rank_residual": abs(
            float(np.trace(projector_k, axis1=1, axis2=2).real.sum())
            - float(total_rank)
        ),
    }
    if translation_covariance > TRANSLATION_TOLERANCE:
        raise FloatingPointError(
            "OW modes violate y-translation covariance: "
            f"{translation_covariance:.6e}"
        )
    return projector_delta, diagnostics


def restricted_projector(projector_delta: np.ndarray, rows: Any) -> np.ndarray:
    rows = np.asarray(tuple(rows), dtype=np.int64)
    ny, q = projector_delta.shape[:2]
    if rows.size == 0 or np.any(rows < 0) or np.any(rows >= ny):
        raise ValueError("restricted rows must be nonempty representatives in [0,Ny)")
    if np.unique(rows).size != rows.size:
        raise ValueError("restricted rows must be distinct")
    delta = (rows[:, None] - rows[None, :]) % ny
    return (
        projector_delta[delta]
        .transpose(0, 2, 1, 3)
        .reshape(rows.size * q, rows.size * q)
        .copy()
    )


def entropy_from_projector(projector: np.ndarray) -> tuple[float, float, float]:
    projector = 0.5 * (projector + projector.conj().T)
    eigenvalues = eigh(
        projector,
        eigvals_only=True,
        check_finite=False,
        overwrite_a=True,
        driver="evd",
    )
    eigen_min = float(eigenvalues.min())
    eigen_max = float(eigenvalues.max())
    if eigen_min < -1.0e-8 or eigen_max > 1.0 + 1.0e-8:
        raise FloatingPointError(
            "restricted occupation spectrum left [0,1]: "
            f"min={eigen_min:.6e}, max={eigen_max:.6e}"
        )
    p = np.clip(eigenvalues, ENTROPY_EPS, 1.0 - ENTROPY_EPS)
    entropy = -np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(entropy), eigen_min, eigen_max


def fit_central_charge(
    ay_values: np.ndarray, entropies: np.ndarray, ny: int
) -> dict[str, float]:
    ay_values = np.asarray(ay_values, dtype=np.int64)
    entropies = np.asarray(entropies, dtype=np.float64)
    log_chord = np.log(
        (float(ny) / np.pi) * np.sin(np.pi * ay_values / float(ny))
    )
    mask = (ay_values >= FIT_AY_MIN) & (ay_values <= ny // 2)
    x = log_chord[mask]
    y = entropies[mask]
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    variance = residual_sum / float(x.size - 2)
    slope_se = float(np.sqrt(max(variance, 0.0) / sxx))
    r2 = (
        1.0
        if np.isclose(total_sum, 0.0) and np.isclose(residual_sum, 0.0)
        else 0.0
        if np.isclose(total_sum, 0.0)
        else 1.0 - residual_sum / total_sum
    )
    return {
        "slope": float(slope),
        "slope_se": slope_se,
        "intercept": float(intercept),
        "c_fit": float(3.0 * slope),
        "c_fit_se": float(3.0 * slope_se),
        "r2": float(r2),
    }


def cft_mutual_information_reference(ny: int, width: int, c_eff: float = 1.0) -> float:
    cross_ratio = np.sin(np.pi * float(width) / float(ny)) ** 2
    return float(-(float(c_eff) / 3.0) * np.log1p(-cross_ratio))


def compute_case(case: Case, threads_per_worker: int = 1) -> dict[str, Any]:
    started = time.monotonic()
    with threadpool_limits(limits=max(1, int(threads_per_worker))):
        model = _build_model(case)
        if tuple(int(value) for value in model.DW_loc) != (5, 15):
            raise RuntimeError(f"unexpected domain-wall slab: {model.DW_loc}")
        projector_delta, diagnostics = flattened_momentum_projector(model)
        del model

        ay_values = np.arange(1, case.ny // 2 + 1, dtype=np.int64)
        entropy_profile = np.empty(ay_values.size, dtype=np.float64)
        restricted_min = np.inf
        restricted_max = -np.inf
        for index, ay in enumerate(ay_values):
            entropy, eigen_min, eigen_max = entropy_from_projector(
                restricted_projector(projector_delta, range(int(ay)))
            )
            entropy_profile[index] = entropy
            restricted_min = min(restricted_min, eigen_min)
            restricted_max = max(restricted_max, eigen_max)

        def mutual_information(y0: int) -> tuple[float, float, float, float]:
            a_rows = [int((y0 + offset) % case.ny) for offset in range(case.width)]
            b_rows = [
                int((y0 + case.ny // 2 + offset) % case.ny)
                for offset in range(case.width)
            ]
            if set(a_rows) & set(b_rows):
                raise RuntimeError("mutual-information strips overlap")
            sa, _, _ = entropy_from_projector(
                restricted_projector(projector_delta, a_rows)
            )
            sb, _, _ = entropy_from_projector(
                restricted_projector(projector_delta, b_rows)
            )
            su, _, _ = entropy_from_projector(
                restricted_projector(projector_delta, a_rows + b_rows)
            )
            return sa, sb, su, sa + sb - su

        sa, sb, su, mi = mutual_information(0)
        sa_shift, sb_shift, su_shift, mi_shift = mutual_information(1)
        translation_entropy_residual = float(
            max(
                abs(sa - sa_shift),
                abs(sb - sb_shift),
                abs(su - su_shift),
                abs(mi - mi_shift),
            )
        )
        if translation_entropy_residual > TRANSLATION_TOLERANCE:
            raise FloatingPointError(
                "translated entropy observable differs: "
                f"{translation_entropy_residual:.6e}"
            )
        fit = fit_central_charge(ay_values, entropy_profile, case.ny)

    record: dict[str, Any] = {
        **diagnostics,
        "Ay_values": ay_values,
        "entropy_profile_y0avg": entropy_profile,
        "entropy_a_y0avg": 0.5 * (sa + sb),
        "entropy_b_y0avg": 0.5 * (sa + sb),
        "entropy_union_y0avg": su,
        "mutual_information_y0avg": mi,
        "mutual_information_y0_shift_abs": abs(mi - mi_shift),
        "entropy_translation_max_abs": translation_entropy_residual,
        "restricted_occupation_min": float(restricted_min),
        "restricted_occupation_max": float(restricted_max),
        "elapsed_seconds": time.monotonic() - started,
        **fit,
    }
    if not np.isfinite(record["mutual_information_y0avg"]):
        raise FloatingPointError("mutual information is nonfinite")
    if record["mutual_information_y0avg"] < -1.0e-8:
        raise FloatingPointError("mutual information is materially negative")
    return record


def _worker(payload: tuple[Case, int]) -> tuple[Case, dict[str, Any]]:
    case, threads_per_worker = payload
    return case, compute_case(case, threads_per_worker)


def _task_record(output_root: Path, case: Case) -> dict[str, Any]:
    result_path, _ = task_paths(output_root, case)
    with np.load(result_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def aggregate_payload(output_root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    shape = (len(WALLS), len(NSHELL_VALUES), len(NY_VALUES), len(ALPHA_VALUES))
    scalar_keys = (
        "mutual_information_y0avg",
        "entropy_a_y0avg",
        "entropy_b_y0avg",
        "entropy_union_y0avg",
        "mutual_information_y0_shift_abs",
        "entropy_translation_max_abs",
        "c_fit",
        "c_fit_se",
        "slope",
        "slope_se",
        "intercept",
        "r2",
        "half_filling_gap",
        "minimum_absolute_energy",
        "ow_y_translation_covariance_max_abs",
        "hamiltonian_block_hermiticity_max_abs",
        "projector_block_hermiticity_max_abs",
        "projector_idempotency_max_abs",
        "projector_rank_residual",
        "restricted_occupation_min",
        "restricted_occupation_max",
        "elapsed_seconds",
    )
    integer_keys = ("half_filling_rank", "negative_energy_count")
    arrays = {key: np.full(shape, np.nan) for key in scalar_keys}
    arrays.update({key: np.full(shape, -1, dtype=np.int64) for key in integer_keys})
    max_ay = max(NY_VALUES) // 2
    profiles = np.full(shape + (max_ay,), np.nan)
    ranks = np.full(shape + (max(NY_VALUES),), -1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for case in cases():
        record = _task_record(output_root, case)
        target = (
            case.wall_index,
            case.nshell_index,
            case.ny_index,
            case.alpha_index,
        )
        for key in scalar_keys + integer_keys:
            arrays[key][target] = record[key].item()
        ay = record["Ay_values"].astype(np.int64)
        profile = record["entropy_profile_y0avg"].astype(np.float64)
        profiles[target + (slice(0, ay.size),)] = profile
        rank_by_ky = record["occupied_rank_by_ky"].astype(np.int64)
        ranks[target + (slice(0, rank_by_ky.size),)] = rank_by_ky
        row = {
            "wall": case.wall,
            "nshell": case.nshell_label,
            "Nx": NX,
            "Ny": case.ny,
            "width": case.width,
            "alpha_1": case.alpha_1,
            "alpha_2": ALPHA_2,
        }
        row.update({key: arrays[key][target].item() for key in scalar_keys + integer_keys})
        rows.append(row)
    arrays.update(
        {
            "schema": np.asarray(SCHEMA),
            "wall_labels": np.asarray(WALLS),
            "nshell_labels": np.asarray(NSHELL_LABELS),
            "nshell_values": np.asarray([1, 2, -1], dtype=np.int64),
            "Ny_values": np.asarray(NY_VALUES, dtype=np.int64),
            "width_values": np.asarray([ny // 4 for ny in NY_VALUES], dtype=np.int64),
            "alpha_1_values": np.asarray(ALPHA_VALUES),
            "alpha_2": np.asarray(ALPHA_2),
            "fit_Ay_min": np.asarray(FIT_AY_MIN),
            "entropy_profile_y0avg": profiles,
            "occupied_rank_by_ky": ranks,
            "DW_slab_inclusive_sites": np.asarray([5, 15], dtype=np.int64),
            "DW_interface_coordinates": np.asarray([5.0, 16.0]),
            "mutual_information_translation_reduction": np.asarray(
                "translation-invariant projector; y0=0 and y0=1 explicitly checked"
            ),
            "mutual_information_cft_c1_by_Ny": np.asarray(
                [cft_mutual_information_reference(ny, ny // 4) for ny in NY_VALUES]
            ),
        }
    )
    return arrays, rows


def _plot_styles() -> tuple[dict[str, Any], ...]:
    return (
        {"color": "#d62728", "marker": "^", "linestyle": ":"},
        {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
        {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    )


def _set_plot_defaults() -> None:
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


def make_results_figure(payload: dict[str, np.ndarray], pdf: Path, png: Path) -> None:
    import matplotlib.pyplot as plt

    _set_plot_defaults()
    alpha = payload["alpha_1_values"]
    mi = payload["mutual_information_y0avg"]
    c_fit = payload["c_fit"]
    fig, axes = plt.subplots(3, 4, figsize=(7.05, 6.35), sharex=True)
    column_titles = (
        r"hard: $I_{a,b}^{\rm flat}$",
        r"soft: $I_{a,b}^{\rm flat}$",
        r"hard: $c_{\rm fit}$",
        r"soft: $c_{\rm fit}$",
    )
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)
    handles = []
    labels = []
    for shell_index, shell_label in enumerate(NSHELL_LABELS):
        for wall_index in range(2):
            mi_axis = axes[shell_index, wall_index]
            c_axis = axes[shell_index, 2 + wall_index]
            for ny_index, (ny, style) in enumerate(zip(NY_VALUES, _plot_styles())):
                line = mi_axis.plot(
                    alpha,
                    mi[wall_index, shell_index, ny_index],
                    linewidth=1.05,
                    markersize=2.9,
                    markeredgewidth=0.55,
                    **style,
                )[0]
                c_axis.plot(
                    alpha,
                    c_fit[wall_index, shell_index, ny_index],
                    linewidth=1.05,
                    markersize=2.9,
                    markeredgewidth=0.55,
                    **style,
                )
                if shell_index == 0 and wall_index == 0:
                    handles.append(line)
                    labels.append(rf"$N_y={ny}$")
                mi_axis.axhline(
                    cft_mutual_information_reference(ny, ny // 4),
                    color=style["color"],
                    linewidth=0.55,
                    alpha=0.30,
                )
            c_axis.axhline(1.0, color="0.55", linestyle=":", linewidth=0.8)
        axes[shell_index, 0].set_ylabel(
            rf"$n_{{\rm shell}}={shell_label}$" + "\n" + r"$I_{a,b}^{\rm flat}$"
        )
        axes[shell_index, 2].set_ylabel(r"$c_{\rm fit}=3m$")
        for axis in axes[shell_index]:
            axis.axvline(2.0, color="0.25", linestyle="--", linewidth=0.85)
            axis.tick_params(direction="in", top=True, right=True)
            for spine in axis.spines.values():
                spine.set_visible(True)
    for axis in axes[-1]:
        axis.set_xlabel(r"$\alpha_1$")
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
    )
    fig.text(
        0.5,
        0.012,
        r"Faint horizontal lines: $I_{a,b}^{\rm CFT}=-(1/3)\log[1-\sin^2(\pi w/N_y)]$.",
        ha="center",
        fontsize=7,
    )
    for index, axis in enumerate(axes.ravel()):
        axis.text(
            -0.18,
            1.04,
            f"({chr(ord('a') + index)})",
            transform=axis.transAxes,
        )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.965))
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_geometry_figure(pdf: Path, png: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    _set_plot_defaults()
    nx = NX
    ny = 40
    y0 = 3
    width = ny // 4
    fig, axis = plt.subplots(figsize=(3.375, 3.15))
    axis.add_patch(Rectangle((0, 0), nx, ny, color="0.90", zorder=0))
    axis.add_patch(
        Rectangle((5, 0), 11, ny, color="#b7e1b0", alpha=0.75, zorder=1)
    )
    axis.add_patch(
        Rectangle((0, y0), nx, width, color="#d62728", alpha=0.35, zorder=2)
    )
    axis.add_patch(
        Rectangle(
            (0, y0 + ny // 2), nx, width, color="#1f77b4", alpha=0.35, zorder=2
        )
    )
    for x_position in (5, 16):
        axis.axvline(x_position, color="black", linestyle="--", linewidth=1.0, zorder=3)
    axis.text(10.5, 38.1, r"topological slab: $\alpha_1$", ha="center", va="top")
    axis.text(2.5, 38.1, r"$\alpha_2=30$", ha="center", va="top", fontsize=7)
    axis.text(18.0, 38.1, r"$\alpha_2=30$", ha="center", va="top", fontsize=7)
    axis.text(0.8, y0 + width / 2, r"$a$", color="#8c1515", va="center", fontsize=10)
    axis.text(
        0.8,
        y0 + ny // 2 + width / 2,
        r"$b$",
        color="#174f83",
        va="center",
        fontsize=10,
    )
    axis.text(5, -2.3, r"DW: $x=5$", ha="center", va="top", fontsize=7)
    axis.text(16, -2.3, r"DW: $x=16$", ha="center", va="top", fontsize=7)
    axis.add_patch(
        FancyArrowPatch(
            (20.7, y0 + width / 2),
            (20.7, y0 + ny // 2 + width / 2),
            arrowstyle="<->",
            mutation_scale=9,
            linewidth=0.9,
            clip_on=False,
        )
    )
    axis.text(
        21.2,
        y0 + ny // 4 + width / 2,
        r"$N_y/2$",
        rotation=90,
        va="center",
        fontsize=7,
    )
    axis.text(19.2, y0 + width / 2, r"$w=\lfloor N_y/4\rfloor$", ha="right", va="center", fontsize=7)
    axis.annotate(
        "periodic $y$",
        xy=(-0.5, 39.2),
        xytext=(-0.5, 0.8),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.8},
        rotation=90,
        va="center",
        ha="right",
        annotation_clip=False,
        fontsize=7,
    )
    axis.set_xlim(0, nx)
    axis.set_ylim(0, ny)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$y$")
    axis.set_xticks((0, 5, 10, 16, 20))
    axis.set_yticks((0, y0, y0 + width, y0 + ny // 2, y0 + ny // 2 + width, ny))
    axis.tick_params(direction="in", top=True, right=True)
    axis.set_title(r"Opposite full-$x$ mutual-information strips")
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copy2(local_path, temporary)
        if sha256_file(temporary) != sha256_file(local_path):
            raise OSError(f"published checksum mismatch for {final_path.name}")
        os.replace(temporary, final_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"bytes": final_path.stat().st_size, "sha256": sha256_file(final_path)}


def aggregate_paths(output_root: Path) -> tuple[Path, ...]:
    root = Path(output_root) / RESULT_DIRNAME
    return tuple(root / name for name in PRODUCT_NAMES) + (
        root / "flattened_ground_state_large_ny.complete.json",
    )


def verified_aggregate(
    output_root: Path, hashes: dict[str, str]
) -> tuple[bool, str]:
    *products, completion_path = aggregate_paths(output_root)
    if not completion_path.is_file() and not any(path.is_file() for path in products):
        return False, "missing aggregate"
    if not completion_path.is_file() or not all(path.is_file() for path in products):
        return False, "incomplete aggregate"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return False, f"unreadable aggregate completion: {exc}"
    expected = {
        **campaign_identity(hashes),
        "schema": AGGREGATE_COMPLETION_SCHEMA,
        "case_count": len(cases()),
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            return False, f"aggregate identity mismatch: {key}"
    declared = completion.get("products", {})
    for path in products:
        record = declared.get(path.name)
        if not isinstance(record, dict):
            return False, f"missing aggregate product record: {path.name}"
        if int(record.get("bytes", -1)) != path.stat().st_size:
            return False, f"aggregate byte count mismatch: {path.name}"
        if record.get("sha256") != sha256_file(path):
            return False, f"aggregate checksum mismatch: {path.name}"
    return True, "verified"


def write_aggregate(
    output_root: Path,
    scratch_root: Path,
    hashes: dict[str, str],
    elapsed_seconds: float,
) -> None:
    payload, rows = aggregate_payload(output_root)
    scratch = Path(scratch_root) / RESULT_DIRNAME / f"aggregate-{uuid.uuid4().hex}"
    scratch.mkdir(parents=True)
    try:
        local_paths = [scratch / name for name in PRODUCT_NAMES]
        _atomic_write_npz(local_paths[0], payload)
        write_csv(local_paths[1], rows)
        make_results_figure(payload, local_paths[2], local_paths[3])
        make_geometry_figure(local_paths[4], local_paths[5])
        *final_paths, completion_path = aggregate_paths(output_root)
        products = {
            final.name: publish_file(local, final)
            for local, final in zip(local_paths, final_paths)
        }
        diagnostic_keys = (
            "ow_y_translation_covariance_max_abs",
            "hamiltonian_block_hermiticity_max_abs",
            "projector_block_hermiticity_max_abs",
            "projector_idempotency_max_abs",
            "projector_rank_residual",
            "entropy_translation_max_abs",
        )
        completion = {
            **campaign_identity(hashes),
            "schema": AGGREGATE_COMPLETION_SCHEMA,
            "case_count": len(cases()),
            "products": products,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "diagnostic_maxima": {
                key: float(np.max(payload[key])) for key in diagnostic_keys
            },
        }
        _atomic_write_json(completion_path, completion)
        verified, reason = verified_aggregate(output_root, hashes)
        if not verified:
            raise OSError(f"published aggregate failed verification: {reason}")
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def run(
    *,
    output_root: Path,
    scratch_root: Path,
    workers: int,
    threads_per_worker: int,
    report_only: bool,
    force: bool,
    max_new_cases: int | None,
) -> dict[str, Any]:
    hashes = source_hashes()
    all_cases = cases()
    status = {
        case.case_id: verified_task(output_root, case, hashes) for case in all_cases
    }
    verified_cases = [case for case in all_cases if status[case.case_id][0]]
    pending = [case for case in all_cases if force or not status[case.case_id][0]]
    aggregate_ok, aggregate_reason = verified_aggregate(output_root, hashes)
    print(
        json.dumps(
            {
                **campaign_identity(hashes),
                "case_count": len(all_cases),
                "verified_cases": len(verified_cases),
                "pending_cases": len(pending),
                "aggregate": {"verified": aggregate_ok, "reason": aggregate_reason},
                "workers": workers,
                "threads_per_worker": threads_per_worker,
                "output_directory": str(Path(output_root) / RESULT_DIRNAME),
                "scratch_directory": str(scratch_root),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if report_only:
        return {
            "status": "verified" if aggregate_ok else "missing_or_invalid",
            "verified_cases": len(verified_cases),
            "aggregate_reason": aggregate_reason,
        }
    if aggregate_ok and not force:
        print("[large-Ny reference] verified existing aggregate; skipping", flush=True)
        return {"status": "existing", "case_count": len(all_cases)}
    if max_new_cases is not None:
        pending = pending[: max(0, int(max_new_cases))]
    started = time.monotonic()
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = {
                executor.submit(_worker, (case, threads_per_worker)): case
                for case in pending
            }
            progress = tqdm(
                total=len(all_cases),
                initial=len(verified_cases),
                desc="large-Ny flattened sweep",
                unit="case",
                dynamic_ncols=True,
            )
            try:
                for future in as_completed(futures):
                    case = futures[future]
                    returned_case, record = future.result()
                    if returned_case != case:
                        raise RuntimeError("worker returned a mismatched case")
                    save_task(output_root, case, record, hashes)
                    progress.set_postfix(
                        wall=case.wall,
                        nshell=case.nshell_label,
                        Ny=case.ny,
                        alpha=f"{case.alpha_1:g}",
                    )
                    progress.update(1)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
            finally:
                progress.close()
    elapsed = time.monotonic() - started
    verified_after = [
        case for case in all_cases if verified_task(output_root, case, hashes)[0]
    ]
    if len(verified_after) != len(all_cases):
        print(
            f"[large-Ny reference] partial: {len(verified_after)}/{len(all_cases)} cases verified",
            flush=True,
        )
        return {"status": "partial", "verified_cases": len(verified_after)}
    write_aggregate(output_root, scratch_root, hashes, elapsed)
    print(
        "[large-Ny reference summary] "
        + json.dumps(
            {
                "status": "complete",
                "cases": len(all_cases),
                "elapsed_seconds": elapsed,
                "verification": "verified",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return {"status": "complete", "case_count": len(all_cases)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/tmp/domain_wall_flattened_large_ny_scratch"),
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-cases", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        output_root=args.output_root,
        scratch_root=args.scratch_root,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        report_only=bool(args.report_only),
        force=bool(args.force),
        max_new_cases=args.max_new_cases,
    )


if __name__ == "__main__":
    main()
