#!/usr/bin/env python3
"""Adaptive Kato continuation of saved monitored-circuit endpoint projectors."""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_state_projector_pump_s100 as endpoint20  # noqa: E402
import run_state_projector_pump_variants as endpoint_variants  # noqa: E402


CAMPAIGN_SCHEMA = "kato_overlap_continuation_campaign_v1"
RESULT_SCHEMA = "kato_overlap_continuation_path_v1"
COMPLETION_SCHEMA = "kato_overlap_continuation_completion_v1"
CHECKPOINT_SCHEMA = "kato_overlap_continuation_checkpoint_v1"
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "campaign_config.kato_overlap_continuation_n20x24_n24x24_s25_v1.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "N20x24_N24x24_kato_overlap_continuation_s25_v1"
)
SOURCE_PATHS = {
    "campaign_runner": Path(__file__).resolve(),
    "n20_endpoint_runner": Path(endpoint20.__file__).resolve(),
    "variant_endpoint_runner": Path(endpoint_variants.__file__).resolve(),
    "n20_endpoint_config": PROJECT_ROOT / "campaign_config.state_projector_pump_n20x24_s100_v1.json",
    "n24_endpoint_config": PROJECT_ROOT / "campaign_config.state_projector_pump_n24x24_s100_v1.json",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "sources", "endpoint_selection", "continuation",
        "ensemble", "analysis", "acceptance",
    )
    return hashlib.sha256(
        canonical_json({key: config[key] for key in keys}).encode("utf-8")
    ).hexdigest()


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N20x24_N24x24_kato_overlap_continuation_s25_v1":
        raise ValueError("unexpected campaign identity")
    if config["endpoint_selection"]["sample_ids"] != list(range(0, 100, 4)):
        raise ValueError("endpoint selection must be sample IDs 0,4,...,96")
    expected_sources = {
        "N20x24": (20, 24, [5, 15], 10, "N20x24_state_projector_pump_s100_v1"),
        "N24x24": (24, 24, [6, 18], 12, "N24x24_state_projector_pump_s100_v1"),
    }
    for label, (nx, ny, walls, split, campaign_id) in expected_sources.items():
        source = config["sources"].get(label, {})
        if (
            int(source.get("Nx", -1)) != nx
            or int(source.get("Ny", -1)) != ny
            or source.get("wall_x") != walls
            or int(source.get("left_x_stop_exclusive", -1)) != split
            or source.get("campaign_id") != campaign_id
        ):
            raise ValueError(f"source geometry or identity changed for {label}")
    continuation = config["continuation"]
    if (
        int(continuation["grid_intervals"]) != 128
        or float(continuation["regulator"]) != 1e-7
        or continuation["directions"] != {"ccw": 1, "cw": -1}
        or continuation["branch_selection"]
        != "largest previous-projector overlap with legacy energy tie-break"
        or continuation["transport_equation"]
        != "dF/dphi = [dP_spec/dphi, P_spec] F"
        or float(continuation["adaptive_tolerance"]) != 1e-8
        or float(continuation["minimum_step_fraction_of_interval"]) != 1.0 / 4096.0
        or continuation["frame_stabilization"]
        != "symmetric polar orthonormalization after accepted steps"
        or bool(continuation["reproject_after_step"])
        or continuation["dtype"] != "complex128"
    ):
        raise ValueError("Kato continuation contract changed")
    ensemble = config["ensemble"]
    if (
        ensemble["sizes"] != ["N20x24", "N24x24"]
        or ensemble["walls"] != ["soft", "hard"]
        or int(ensemble["samples_per_wall_and_size"]) != 25
        or int(ensemble["directions_per_endpoint"]) != 2
        or int(ensemble["paths_total"]) != 200
    ):
        raise ValueError("expected two sizes, two walls, S25, and two directions")
    acceptance = config["acceptance"]
    if bool(acceptance["quantization_is_acceptance_gate"]):
        raise ValueError("quantization must remain a measured outcome")
    if int(config["execution"]["checkpoint_every_intervals"]) != 8:
        raise ValueError("rolling checkpoint interval must remain eight observations")


def tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "task_id": f"kato_{size}_{wall}_{direction}_sample_{sample_id:03d}",
            "size": size,
            "Nx": int(config["sources"][size]["Nx"]),
            "Ny": int(config["sources"][size]["Ny"]),
            "wall": wall,
            "direction": direction,
            "sigma": int(config["continuation"]["directions"][direction]),
            "sample_id": int(sample_id),
            "source_task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        }
        for size in config["ensemble"]["sizes"]
        for wall in config["ensemble"]["walls"]
        for sample_id in config["endpoint_selection"]["sample_ids"]
        for direction in ("ccw", "cw")
    ]
    if len(rows) != 200 or len({row["task_id"] for row in rows}) != 200:
        raise RuntimeError("expected exactly 200 unique Kato paths")
    return rows


def flux_grid(config: dict[str, Any], sigma: int) -> np.ndarray:
    intervals = int(config["continuation"]["grid_intervals"])
    epsilon = float(config["continuation"]["regulator"])
    return -int(sigma) * epsilon + int(sigma) * np.linspace(
        0.0, 2.0 * np.pi, intervals + 1
    )


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(output_root) / "paths" / task["size"] / task["wall"] / task["direction"]
    result = root / f"sample_{int(task['sample_id']):03d}.npz"
    return result, result.with_suffix(".completion.json")


def checkpoint_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    root = Path(output_root) / "checkpoints" / task["size"] / task["wall"] / task["direction"]
    checkpoint = root / f"sample_{int(task['sample_id']):03d}.checkpoint.npz"
    return checkpoint, checkpoint.with_suffix(".json")


def failure_path(output_root: Path, task: dict[str, Any]) -> Path:
    return Path(output_root) / "failures" / f"{task['task_id']}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, Any], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            writer = np.savez_compressed if compressed else np.savez
            writer(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_context_n20(source: dict[str, Any], sample_ids: list[int]) -> dict[str, Any]:
    config_path = (PROJECT_ROOT / source["config"]).resolve()
    output_root = (PROJECT_ROOT / source["output_root"]).resolve()
    source_config = endpoint20.load_config(config_path)
    endpoint20.validate_config(source_config)
    if source_config["campaign_id"] != source["campaign_id"]:
        raise RuntimeError("N20 source campaign identity mismatch")
    config_hash = endpoint20.scientific_config_hash(source_config)
    hashes = endpoint20.source_hashes()
    lookup = {row["task_id"]: row for row in endpoint20.burnin_tasks(source_config)}
    rows: dict[str, dict[str, Any]] = {}
    for wall in ("soft", "hard"):
        for sample_id in sample_ids:
            task_id = f"burnin_{wall}_sample_{sample_id:03d}"
            task = lookup[task_id]
            ok, reason, completion = endpoint20.verify_pair(
                output_root, task, config_hash, hashes, source_config
            )
            if not ok or completion is None:
                raise RuntimeError(f"unverified N20 endpoint {task_id}: {reason}")
            path, _ = endpoint20.result_paths(output_root, task)
            with np.load(path, allow_pickle=False) as saved:
                rank = int(np.asarray(saved["rank"]).item())
                frame = np.asarray(saved["frame"])
                if frame.dtype != np.complex128 or frame.shape != (2 * 20 * 24, rank):
                    raise RuntimeError(f"invalid N20 source frame: {task_id}")
            rows[task_id] = {
                "path": str(path), "name": path.name,
                "bytes": int(completion["result"]["bytes"]),
                "sha256": str(completion["result"]["sha256"]),
                "source_campaign_id": source_config["campaign_id"],
                "source_config_hash": config_hash,
                "source_hashes": hashes,
                "rank": rank,
            }
    return {"rows": rows, "config_hash": config_hash, "source_hashes": hashes}


def _source_context_n24(source: dict[str, Any], sample_ids: list[int]) -> dict[str, Any]:
    config_path = (PROJECT_ROOT / source["config"]).resolve()
    output_root = (PROJECT_ROOT / source["output_root"]).resolve()
    source_config = endpoint_variants.load_config(config_path)
    endpoint_variants.validate_config(source_config)
    if source_config["campaign_id"] != source["campaign_id"]:
        raise RuntimeError("N24 source campaign identity mismatch")
    config_hash = endpoint_variants.scientific_config_hash(source_config)
    hashes = endpoint_variants.source_hashes()
    lookup = {row["task_id"]: row for row in endpoint_variants.burnin_tasks(source_config)}
    rows: dict[str, dict[str, Any]] = {}
    for wall in ("soft", "hard"):
        for sample_id in sample_ids:
            task_id = f"burnin_{wall}_sample_{sample_id:03d}"
            task = lookup[task_id]
            ok, reason, completion = endpoint_variants._verify_own_pair(
                output_root, task, config_hash, hashes, source_config
            )
            if not ok or completion is None:
                raise RuntimeError(f"unverified N24 endpoint {task_id}: {reason}")
            path, _ = endpoint_variants.result_paths(output_root, task)
            with np.load(path, allow_pickle=False) as saved:
                rank = int(np.asarray(saved["rank"]).item())
                frame = np.asarray(saved["frame"])
                if frame.dtype != np.complex128 or frame.shape != (2 * 24 * 24, rank):
                    raise RuntimeError(f"invalid N24 source frame: {task_id}")
            rows[task_id] = {
                "path": str(path), "name": path.name,
                "bytes": int(completion["result"]["bytes"]),
                "sha256": str(completion["result"]["sha256"]),
                "source_campaign_id": source_config["campaign_id"],
                "source_config_hash": config_hash,
                "source_hashes": hashes,
                "rank": rank,
            }
    return {"rows": rows, "config_hash": config_hash, "source_hashes": hashes}


def source_context(config: dict[str, Any]) -> dict[str, Any]:
    sample_ids = [int(value) for value in config["endpoint_selection"]["sample_ids"]]
    contexts = {
        "N20x24": _source_context_n20(config["sources"]["N20x24"], sample_ids),
        "N24x24": _source_context_n24(config["sources"]["N24x24"], sample_ids),
    }
    if any(len(contexts[label]["rows"]) != 50 for label in contexts):
        raise RuntimeError("expected 50 verified endpoint states at each size")
    return contexts


def _load_frame(source_row: dict[str, Any], nx: int, ny: int) -> np.ndarray:
    path = Path(source_row["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(source_row["bytes"])
        or sha256_path(path) != source_row["sha256"]
    ):
        raise RuntimeError(f"source endpoint changed after verification: {path}")
    with np.load(path, allow_pickle=False) as saved:
        raw = np.asarray(saved["frame"])
        if raw.dtype != np.complex128:
            raise RuntimeError(f"source frame is {raw.dtype}, expected complex128")
        frame = np.array(raw, dtype=np.complex128, order="F", copy=True)
        rank = int(np.asarray(saved["rank"]).item())
    if frame.shape != (2 * nx * ny, rank):
        raise RuntimeError(f"source frame has invalid shape {frame.shape}")
    return frame


def _coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)
    y = np.repeat(np.arange(ny, dtype=np.int64), 2 * nx)
    dy = y[:, None] - y[None, :]
    dy = ((dy + ny // 2) % ny) - ny // 2
    return x, np.asarray(dy, dtype=np.float64)


def _twisted_parent_and_derivative(
    h0: np.ndarray, dy: np.ndarray, phi: float, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    phase = np.exp(1j * float(phi) * dy / int(ny))
    raw = h0 * phase
    derivative_raw = raw * (1j * dy / int(ny))
    h = np.asarray(0.5 * (raw + raw.conj().T), dtype=np.complex128, order="F")
    dh = np.asarray(
        0.5 * (derivative_raw + derivative_raw.conj().T),
        dtype=np.complex128,
        order="F",
    )
    return h, dh


def _orthonormal_basis(frame: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(frame, mode="reduced")
    return np.asarray(q, dtype=np.complex128, order="F")


def _symmetric_polar(frame: np.ndarray) -> tuple[np.ndarray, float]:
    gram_raw = frame.conj().T @ frame
    gram = np.asarray(0.5 * (gram_raw + gram_raw.conj().T))
    values, vectors = np.linalg.eigh(gram)
    threshold = 128.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(values)))
    if float(np.min(values)) <= threshold:
        raise FloatingPointError("frame lost rank during symmetric polar stabilization")
    inverse_sqrt = (vectors * (values ** -0.5)[None, :]) @ vectors.conj().T
    result = np.asarray(frame @ inverse_sqrt, dtype=np.complex128, order="F")
    residual = float(np.max(np.abs(result.conj().T @ result - np.eye(result.shape[1]))))
    return result, residual


def _projector_distance(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    rank = frame_a.shape[1]
    overlap = frame_a.conj().T @ frame_b
    squared = float(np.sum(np.abs(overlap) ** 2))
    return float(np.sqrt(max(0.0, 2.0 * rank - 2.0 * squared)) / np.sqrt(rank))


def select_by_previous_projector(
    previous_frame: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Select the fixed-rank spectral branch from projector overlaps.

    The scalar weights are diagonal elements of W^dagger P_previous W.
    Eigenvalue ordering is used only as the legacy deterministic tie-break.
    """
    previous_basis = _orthonormal_basis(previous_frame)
    weights = np.real(np.sum(np.abs(previous_basis.conj().T @ eigenvectors) ** 2, axis=0))
    order = np.lexsort((eigenvalues, -weights))
    selected_indices = np.asarray(order[:rank], dtype=np.int64)
    excluded_indices = np.asarray(order[rank:], dtype=np.int64)
    selected = np.asarray(eigenvectors[:, selected_indices], dtype=np.complex128, order="F")
    singular = np.linalg.svd(previous_basis.conj().T @ selected, compute_uv=False)
    excluded_max = float(np.max(weights[excluded_indices])) if excluded_indices.size else 0.0
    diagnostics = {
        "minimum_principal_overlap": float(np.min(singular)),
        "selected_weight_floor": float(np.min(weights[selected_indices])),
        "selected_weight_margin": float(np.min(weights[selected_indices]) - excluded_max),
    }
    return selected, selected_indices, excluded_indices, diagnostics


def spectral_projector_generator(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    selected_indices: np.ndarray,
    excluded_indices: np.ndarray,
    h_derivative: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the analytic spectral derivative and anti-Hermitian Kato generator."""
    selected = np.asarray(eigenvectors[:, selected_indices], dtype=np.complex128, order="F")
    excluded = np.asarray(eigenvectors[:, excluded_indices], dtype=np.complex128, order="F")
    denominator = eigenvalues[selected_indices][None, :] - eigenvalues[excluded_indices][:, None]
    delta_sel = float(np.min(np.abs(denominator)))
    energy_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    singular_floor = 256.0 * np.finfo(np.float64).eps * energy_scale
    if not np.isfinite(delta_sel) or delta_sel <= singular_floor:
        raise FloatingPointError(
            f"selected/excluded spectral derivative is unresolved: delta_sel={delta_sel:.3e}"
        )
    x_block = (excluded.conj().T @ (h_derivative @ selected)) / denominator
    derivative = excluded @ x_block @ selected.conj().T
    derivative = np.asarray(derivative + derivative.conj().T, dtype=np.complex128)
    generator = np.asarray(
        excluded @ x_block @ selected.conj().T
        - selected @ x_block.conj().T @ excluded.conj().T,
        dtype=np.complex128,
    )
    return derivative, generator, delta_sel


def _spectral_rhs(
    frame: np.ndarray,
    *,
    phi: float,
    sigma: int,
    h0: np.ndarray,
    dy: np.ndarray,
    ny: int,
) -> tuple[np.ndarray, dict[str, float]]:
    h, dh = _twisted_parent_and_derivative(h0, dy, phi, ny)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    rank = frame.shape[1]
    selected, selected_indices, excluded_indices, selection = select_by_previous_projector(
        frame, eigenvalues, eigenvectors, rank
    )
    projector_derivative, generator, delta_sel = spectral_projector_generator(
        eigenvalues, eigenvectors, selected_indices, excluded_indices, dh
    )
    derivative = generator @ frame
    derivative = np.asarray(int(sigma) * derivative, dtype=np.complex128, order="F")
    diagnostics = {
        **selection,
        "delta_sel": delta_sel,
        "generator_norm": float(np.linalg.norm(generator, ord="fro")),
        "projector_derivative_norm": float(np.linalg.norm(projector_derivative, ord="fro")),
    }
    return derivative, diagnostics


def _rk4_trial(
    frame: np.ndarray,
    s: float,
    step: float,
    *,
    sigma: int,
    epsilon: float,
    h0: np.ndarray,
    dy: np.ndarray,
    ny: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    def rhs(local_s: float, local_frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        phi = -int(sigma) * epsilon + int(sigma) * local_s
        return _spectral_rhs(
            local_frame, phi=phi, sigma=sigma, h0=h0, dy=dy, ny=ny
        )

    k1, d1 = rhs(s, frame)
    k2, d2 = rhs(s + 0.5 * step, frame + 0.5 * step * k1)
    k3, d3 = rhs(s + 0.5 * step, frame + 0.5 * step * k2)
    k4, d4 = rhs(s + step, frame + step * k3)
    candidate = frame + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.asarray(candidate, dtype=np.complex128, order="F"), [d1, d2, d3, d4]


def _step_doubling(
    frame: np.ndarray,
    s: float,
    step: float,
    *,
    sigma: int,
    epsilon: float,
    h0: np.ndarray,
    dy: np.ndarray,
    ny: int,
) -> tuple[np.ndarray, float, float, list[dict[str, float]]]:
    full_raw, full_diagnostics = _rk4_trial(
        frame, s, step, sigma=sigma, epsilon=epsilon, h0=h0, dy=dy, ny=ny
    )
    full, full_gram = _symmetric_polar(full_raw)
    half_raw, first_diagnostics = _rk4_trial(
        frame, s, 0.5 * step, sigma=sigma, epsilon=epsilon, h0=h0, dy=dy, ny=ny
    )
    half, half_gram = _symmetric_polar(half_raw)
    fine_raw, second_diagnostics = _rk4_trial(
        half, s + 0.5 * step, 0.5 * step,
        sigma=sigma, epsilon=epsilon, h0=h0, dy=dy, ny=ny,
    )
    fine, fine_gram = _symmetric_polar(fine_raw)
    error = _projector_distance(fine, full)
    return fine, error, max(full_gram, half_gram, fine_gram), (
        full_diagnostics + first_diagnostics + second_diagnostics
    )


def _density_summary(frame: np.ndarray, x: np.ndarray, nx: int) -> tuple[np.ndarray, float, float]:
    density = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    density_x = np.bincount(x, weights=density, minlength=nx).astype(np.float64)
    split = nx // 2
    return density_x, float(np.sum(density_x[:split])), float(np.sum(density_x[split:]))


def _observation_diagnostics(
    frame: np.ndarray,
    *,
    phi: float,
    h0: np.ndarray,
    dy: np.ndarray,
    ny: int,
) -> dict[str, float]:
    h, dh = _twisted_parent_and_derivative(h0, dy, phi, ny)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    selected, selected_indices, excluded_indices, selection = select_by_previous_projector(
        frame, eigenvalues, eigenvectors, frame.shape[1]
    )
    denominator = eigenvalues[selected_indices][None, :] - eigenvalues[excluded_indices][:, None]
    delta_sel = float(np.min(np.abs(denominator)))
    energy_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if delta_sel <= 256.0 * np.finfo(np.float64).eps * energy_scale:
        raise FloatingPointError(
            f"selected/excluded observation gap is unresolved: {delta_sel:.3e}"
        )
    excluded = eigenvectors[:, excluded_indices]
    x_block = (excluded.conj().T @ (dh @ selected)) / denominator
    return {
        **selection,
        "direct_projector_mismatch": _projector_distance(frame, selected),
        "delta_sel": delta_sel,
        "generator_norm": float(np.sqrt(2.0) * np.linalg.norm(x_block, ord="fro")),
    }


HISTORY_FIELDS = (
    "phi", "N_left", "N_right", "N_total", "delta_N_left", "delta_N_right",
    "delta_N_total", "q_x", "direct_projector_mismatch", "minimum_principal_overlap",
    "selected_weight_floor", "selected_weight_margin", "delta_sel", "generator_norm",
    "adaptive_error", "minimum_stage_delta_sel", "maximum_stage_generator_norm",
    "accepted_steps", "rejected_steps",
)


def _initial_arrays(count: int, nx: int) -> dict[str, np.ndarray]:
    arrays = {key: np.full(count, np.nan, dtype=np.float64) for key in HISTORY_FIELDS}
    arrays["density_x"] = np.full((count, nx), np.nan, dtype=np.float64)
    arrays["accepted_steps"] = np.zeros(count, dtype=np.int64)
    arrays["rejected_steps"] = np.zeros(count, dtype=np.int64)
    return arrays


def _metadata(
    task: dict[str, Any], config_hash: str, hashes: dict[str, str],
    source_row: dict[str, Any], adaptive_tolerance: float,
) -> dict[str, Any]:
    return {
        **task,
        "config_hash": config_hash,
        "source_hashes": hashes,
        "source_result": source_row,
        "adaptive_tolerance": float(adaptive_tolerance),
    }


def _checkpoint_payload(
    *, frame: np.ndarray, completed_interval: int, s: float, next_step: float,
    arrays: dict[str, np.ndarray], metadata: dict[str, Any],
    maximum_orthonormality_residual: float,
) -> dict[str, Any]:
    return {
        "schema": np.asarray(CHECKPOINT_SCHEMA),
        "metadata_json": np.asarray(canonical_json(metadata)),
        "frame": frame,
        "completed_interval": np.asarray(completed_interval, dtype=np.int64),
        "s": np.asarray(s),
        "next_step": np.asarray(next_step),
        "maximum_orthonormality_residual": np.asarray(maximum_orthonormality_residual),
        **arrays,
    }


def _write_checkpoint(
    output_root: Path, task: dict[str, Any], *, frame: np.ndarray,
    completed_interval: int, s: float, next_step: float,
    arrays: dict[str, np.ndarray], metadata: dict[str, Any],
    maximum_orthonormality_residual: float,
) -> None:
    checkpoint, receipt = checkpoint_paths(output_root, task)
    _atomic_npz(
        checkpoint,
        _checkpoint_payload(
            frame=frame, completed_interval=completed_interval, s=s, next_step=next_step,
            arrays=arrays, metadata=metadata,
            maximum_orthonormality_residual=maximum_orthonormality_residual,
        ),
        compressed=False,
    )
    _atomic_json(
        receipt,
        {
            "schema": CHECKPOINT_SCHEMA,
            "metadata": metadata,
            "completed_interval": int(completed_interval),
            "checkpoint": {
                "name": checkpoint.name,
                "bytes": checkpoint.stat().st_size,
                "sha256": sha256_path(checkpoint),
            },
        },
    )


def _load_checkpoint(
    output_root: Path, task: dict[str, Any], *, metadata: dict[str, Any],
    count: int, nx: int, ambient: int, rank: int,
) -> dict[str, Any] | None:
    checkpoint, receipt_path = checkpoint_paths(output_root, task)
    if not checkpoint.exists() and not receipt_path.exists():
        return None
    try:
        if not checkpoint.is_file() or not receipt_path.is_file():
            raise RuntimeError("incomplete checkpoint pair")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        record = receipt["checkpoint"]
        if receipt.get("schema") != CHECKPOINT_SCHEMA or receipt.get("metadata") != metadata:
            raise RuntimeError("checkpoint identity mismatch")
        if record.get("name") != checkpoint.name or int(record.get("bytes", -1)) != checkpoint.stat().st_size:
            raise RuntimeError("checkpoint name or byte count mismatch")
        if record.get("sha256") != sha256_path(checkpoint):
            raise RuntimeError("checkpoint checksum mismatch")
        with np.load(checkpoint, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != CHECKPOINT_SCHEMA:
                raise RuntimeError("checkpoint schema mismatch")
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != metadata:
                raise RuntimeError("checkpoint metadata mismatch")
            frame_raw = np.asarray(saved["frame"])
            if frame_raw.dtype != np.complex128:
                raise RuntimeError("checkpoint frame dtype mismatch")
            frame = np.array(frame_raw, dtype=np.complex128, order="F", copy=True)
            completed_interval = int(np.asarray(saved["completed_interval"]).item())
            s = float(np.asarray(saved["s"]).item())
            next_step = float(np.asarray(saved["next_step"]).item())
            maximum_orthonormality_residual = float(
                np.asarray(saved["maximum_orthonormality_residual"]).item()
            )
            arrays = {key: np.array(saved[key], copy=True) for key in HISTORY_FIELDS}
            arrays["density_x"] = np.array(saved["density_x"], copy=True)
        if frame.shape != (ambient, rank):
            raise RuntimeError("checkpoint frame shape mismatch")
        if completed_interval < 0 or completed_interval >= count:
            raise RuntimeError("checkpoint interval out of range")
        expected_s = completed_interval * 2.0 * np.pi / (count - 1)
        if abs(s - expected_s) > 64.0 * np.finfo(float).eps * max(1.0, abs(expected_s)):
            raise RuntimeError("checkpoint is not on an observation boundary")
        if not np.isfinite(next_step) or next_step <= 0.0:
            raise RuntimeError("checkpoint next step is invalid")
        expected_shapes = {key: (count,) for key in HISTORY_FIELDS}
        expected_shapes["density_x"] = (count, nx)
        for key, shape in expected_shapes.items():
            if arrays[key].shape != shape:
                raise RuntimeError(f"checkpoint {key} shape mismatch")
            if not np.all(np.isfinite(arrays[key][: completed_interval + 1])):
                raise RuntimeError(f"checkpoint {key} prefix is nonfinite")
        gram = float(np.max(np.abs(frame.conj().T @ frame - np.eye(rank))))
        if gram > 1e-10:
            raise RuntimeError(f"checkpoint Gram residual {gram:.3e}")
        return {
            "frame": frame, "completed_interval": completed_interval, "s": s,
            "next_step": next_step, "arrays": arrays,
            "maximum_orthonormality_residual": maximum_orthonormality_residual,
        }
    except Exception as exc:
        print(f"[checkpoint reset] {task['task_id']}: {type(exc).__name__}: {exc}", flush=True)
        checkpoint.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)
        return None


def _delete_checkpoint(output_root: Path, task: dict[str, Any]) -> None:
    checkpoint, receipt = checkpoint_paths(output_root, task)
    checkpoint.unlink(missing_ok=True)
    receipt.unlink(missing_ok=True)


def compute_path(
    frame0: np.ndarray,
    task: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    metadata: dict[str, Any],
    *,
    adaptive_tolerance: float,
    progress_queue: Any = None,
) -> dict[str, Any]:
    nx, ny = int(task["Nx"]), int(task["Ny"])
    sigma = int(task["sigma"])
    intervals = int(config["continuation"]["grid_intervals"])
    count = intervals + 1
    epsilon = float(config["continuation"]["regulator"])
    observation_step = 2.0 * np.pi / intervals
    min_step = observation_step * float(
        config["continuation"]["minimum_step_fraction_of_interval"]
    )
    rank = frame0.shape[1]
    input_gram = float(np.max(np.abs(frame0.conj().T @ frame0 - np.eye(rank))))
    if input_gram > float(config["acceptance"]["input_gram_tolerance"]):
        raise RuntimeError(f"source frame Gram residual {input_gram:.3e}")
    projector0 = frame0 @ frame0.conj().T
    h0 = np.asarray(np.eye(frame0.shape[0]) - 2.0 * projector0, dtype=np.complex128, order="F")
    x, dy = _coordinates(nx, ny)
    source_density_x, source_left, source_right = _density_summary(frame0, x, nx)
    checkpoint = _load_checkpoint(
        output_root, task, metadata=metadata, count=count, nx=nx,
        ambient=frame0.shape[0], rank=rank,
    )
    if checkpoint is None:
        phi0 = -sigma * epsilon
        h_initial, _ = _twisted_parent_and_derivative(h0, dy, phi0, ny)
        eigenvalues, eigenvectors = np.linalg.eigh(h_initial)
        frame, _, _, _ = select_by_previous_projector(
            frame0, eigenvalues, eigenvectors, rank
        )
        # Fix only the in-subspace gauge so the initial frame is maximally close
        # to the monitored endpoint; observables depend solely on its projector.
        overlap = frame.conj().T @ frame0
        left_svd, _, right_svd = np.linalg.svd(overlap, full_matrices=False)
        frame = np.asarray(frame @ (left_svd @ right_svd), dtype=np.complex128, order="F")
        frame, polar_residual = _symmetric_polar(frame)
        arrays = _initial_arrays(count, nx)
        density_x, left, right = _density_summary(frame, x, nx)
        diagnostic = _observation_diagnostics(
            frame, phi=phi0, h0=h0, dy=dy, ny=ny
        )
        arrays["phi"][0] = phi0
        arrays["N_left"][0], arrays["N_right"][0] = left, right
        arrays["N_total"][0] = left + right
        arrays["delta_N_left"][0] = arrays["delta_N_right"][0] = 0.0
        arrays["delta_N_total"][0] = arrays["q_x"][0] = 0.0
        arrays["density_x"][0] = density_x
        for key in (
            "direct_projector_mismatch", "minimum_principal_overlap",
            "selected_weight_floor", "selected_weight_margin", "delta_sel", "generator_norm",
        ):
            arrays[key][0] = diagnostic[key]
        arrays["adaptive_error"][0] = 0.0
        arrays["minimum_stage_delta_sel"][0] = diagnostic["delta_sel"]
        arrays["maximum_stage_generator_norm"][0] = diagnostic["generator_norm"]
        completed_interval, s = 0, 0.0
        next_step = observation_step * float(
            config["continuation"]["initial_step_fraction_of_interval"]
        )
        maximum_orthonormality_residual = max(input_gram, polar_residual)
    else:
        frame = checkpoint["frame"]
        arrays = checkpoint["arrays"]
        completed_interval = int(checkpoint["completed_interval"])
        s = float(checkpoint["s"])
        next_step = float(checkpoint["next_step"])
        maximum_orthonormality_residual = float(
            checkpoint["maximum_orthonormality_residual"]
        )
    reference_left = float(arrays["N_left"][0])
    reference_right = float(arrays["N_right"][0])
    reference_total = reference_left + reference_right
    checkpoint_every = int(config["execution"]["checkpoint_every_intervals"])
    safety = float(config["continuation"]["adaptive_safety"])
    min_scale = float(config["continuation"]["adaptive_min_scale"])
    max_scale = float(config["continuation"]["adaptive_max_scale"])
    for observation_index in range(completed_interval + 1, count):
        target_s = observation_index * observation_step
        interval_max_error = 0.0
        interval_min_gap = np.inf
        interval_max_generator = 0.0
        accepted = rejected = 0
        while s < target_s - 16.0 * np.finfo(float).eps:
            step = min(next_step, target_s - s)
            if step < min_step * (1.0 - 64.0 * np.finfo(float).eps):
                raise FloatingPointError(
                    f"adaptive step fell below interval/4096 at observation {observation_index}: "
                    f"h={step:.3e}, min={min_step:.3e}"
                )
            candidate, error, polar_residual, stage = _step_doubling(
                frame, s, step, sigma=sigma, epsilon=epsilon, h0=h0, dy=dy, ny=ny
            )
            if not np.isfinite(error):
                raise FloatingPointError("nonfinite adaptive projector error")
            stage_gap = min(item["delta_sel"] for item in stage)
            stage_generator = max(item["generator_norm"] for item in stage)
            interval_min_gap = min(interval_min_gap, stage_gap)
            interval_max_generator = max(interval_max_generator, stage_generator)
            if error <= adaptive_tolerance:
                frame = candidate
                s = min(target_s, s + step)
                accepted += 1
                interval_max_error = max(interval_max_error, error)
                maximum_orthonormality_residual = max(
                    maximum_orthonormality_residual, polar_residual
                )
            else:
                rejected += 1
            if error == 0.0:
                scale = max_scale
            else:
                scale = safety * (adaptive_tolerance / error) ** 0.2
                scale = min(max_scale, max(min_scale, scale))
            next_step = step * scale
            if error > adaptive_tolerance and next_step < min_step:
                raise FloatingPointError(
                    f"unresolved Kato step at observation {observation_index}: "
                    f"error={error:.3e}, proposed_h={next_step:.3e}"
                )
        s = target_s
        phi = -sigma * epsilon + sigma * s
        density_x, left, right = _density_summary(frame, x, nx)
        diagnostic = _observation_diagnostics(frame, phi=phi, h0=h0, dy=dy, ny=ny)
        arrays["phi"][observation_index] = phi
        arrays["N_left"][observation_index] = left
        arrays["N_right"][observation_index] = right
        arrays["N_total"][observation_index] = left + right
        arrays["delta_N_left"][observation_index] = left - reference_left
        arrays["delta_N_right"][observation_index] = right - reference_right
        arrays["delta_N_total"][observation_index] = left + right - reference_total
        arrays["q_x"][observation_index] = 0.5 * (
            arrays["delta_N_right"][observation_index]
            - arrays["delta_N_left"][observation_index]
        )
        arrays["density_x"][observation_index] = density_x
        for key in (
            "direct_projector_mismatch", "minimum_principal_overlap",
            "selected_weight_floor", "selected_weight_margin", "delta_sel", "generator_norm",
        ):
            arrays[key][observation_index] = diagnostic[key]
        arrays["adaptive_error"][observation_index] = interval_max_error
        arrays["minimum_stage_delta_sel"][observation_index] = interval_min_gap
        arrays["maximum_stage_generator_norm"][observation_index] = interval_max_generator
        arrays["accepted_steps"][observation_index] = accepted
        arrays["rejected_steps"][observation_index] = rejected
        if progress_queue is not None:
            progress_queue.put(1)
        if observation_index % checkpoint_every == 0 and observation_index < intervals:
            _write_checkpoint(
                output_root, task, frame=frame, completed_interval=observation_index,
                s=s, next_step=next_step, arrays=arrays, metadata=metadata,
                maximum_orthonormality_residual=maximum_orthonormality_residual,
            )
    result = dict(arrays)
    result.update(
        {
            "source_density_x": source_density_x,
            "source_N_left": np.asarray(source_left),
            "source_N_right": np.asarray(source_right),
            "source_rank": np.asarray(rank, dtype=np.int64),
            "input_gram_residual": np.asarray(input_gram),
            "maximum_orthonormality_residual": np.asarray(maximum_orthonormality_residual),
            "maximum_charge_residual": np.asarray(
                float(np.max(np.abs(arrays["delta_N_total"])))
            ),
            "maximum_direct_projector_mismatch": np.asarray(
                float(np.max(arrays["direct_projector_mismatch"]))
            ),
            "minimum_principal_overlap_over_path": np.asarray(
                float(np.min(arrays["minimum_principal_overlap"]))
            ),
            "minimum_selected_weight_margin_over_path": np.asarray(
                float(np.min(arrays["selected_weight_margin"]))
            ),
            "minimum_delta_sel_over_path": np.asarray(float(np.min(arrays["minimum_stage_delta_sel"]))),
            "accepted_steps_total": np.asarray(int(np.sum(arrays["accepted_steps"])), dtype=np.int64),
            "rejected_steps_total": np.asarray(int(np.sum(arrays["rejected_steps"])), dtype=np.int64),
        }
    )
    return result


def _validate_result_arrays(arrays: dict[str, Any], task: dict[str, Any], config: dict[str, Any]) -> None:
    count = int(config["continuation"]["grid_intervals"]) + 1
    nx = int(task["Nx"])
    required = {key: (count,) for key in HISTORY_FIELDS}
    required["density_x"] = (count, nx)
    required["source_density_x"] = (nx,)
    for key, shape in required.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid result field {key}")
    if np.asarray(arrays["phi"])[0] != -int(task["sigma"]) * float(config["continuation"]["regulator"]):
        raise RuntimeError("signed regulator origin changed")
    if float(np.asarray(arrays["maximum_charge_residual"]).item()) > float(
        config["acceptance"]["charge_conservation_tolerance"]
    ):
        raise FloatingPointError("Kato path violates charge conservation")
    if float(np.asarray(arrays["maximum_orthonormality_residual"]).item()) > float(
        config["acceptance"]["orthonormality_tolerance"]
    ):
        raise FloatingPointError("Kato frame is not orthonormal")
    if float(np.asarray(arrays["maximum_direct_projector_mismatch"]).item()) > float(
        config["acceptance"]["transported_selected_projector_mismatch_tolerance"]
    ):
        raise FloatingPointError("transported projector left the selected spectral branch")


def publish_result(
    output_root: Path, task: dict[str, Any], arrays: dict[str, Any], *,
    config_hash: str, hashes: dict[str, str], source_row: dict[str, Any],
    adaptive_tolerance: float, elapsed_seconds: float,
) -> None:
    result, completion = result_paths(output_root, task)
    metadata = _metadata(task, config_hash, hashes, source_row, adaptive_tolerance)
    payload = dict(arrays)
    payload["schema"] = np.asarray(RESULT_SCHEMA)
    payload["metadata_json"] = np.asarray(canonical_json(metadata))
    _atomic_npz(result, payload, compressed=True)
    _atomic_json(
        completion,
        {
            "schema": COMPLETION_SCHEMA,
            "metadata": metadata,
            "result": {
                "name": result.name,
                "bytes": result.stat().st_size,
                "sha256": sha256_path(result),
            },
            "elapsed_seconds": float(elapsed_seconds),
            "completed_unix": time.time(),
        },
    )


def verify_result(
    output_root: Path, task: dict[str, Any], *, config_hash: str,
    hashes: dict[str, str], source_row: dict[str, Any], config: dict[str, Any],
    adaptive_tolerance: float,
) -> tuple[bool, str, dict[str, Any] | None]:
    result, completion_path = result_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair", None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        metadata = _metadata(task, config_hash, hashes, source_row, adaptive_tolerance)
        if completion.get("schema") != COMPLETION_SCHEMA or completion.get("metadata") != metadata:
            return False, "completion identity mismatch", None
        record = completion["result"]
        if record.get("name") != result.name or int(record.get("bytes", -1)) != result.stat().st_size:
            return False, "result name or byte count mismatch", None
        if record.get("sha256") != sha256_path(result):
            return False, "result checksum mismatch", None
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != RESULT_SCHEMA:
                return False, "result schema mismatch", None
            if json.loads(str(np.asarray(saved["metadata_json"]).item())) != metadata:
                return False, "result metadata mismatch", None
            arrays = {
                key: np.array(saved[key], copy=True)
                for key in saved.files if key not in {"schema", "metadata_json"}
            }
        _validate_result_arrays(arrays, task, config)
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def inventory(
    config: dict[str, Any], output_root: Path, *, context: dict[str, Any] | None = None,
    adaptive_tolerance: float | None = None,
) -> dict[str, Any]:
    context = source_context(config) if context is None else context
    config_hash, hashes = scientific_config_hash(config), source_hashes()
    tolerance = float(
        config["continuation"]["adaptive_tolerance"]
        if adaptive_tolerance is None else adaptive_tolerance
    )
    rows = {}
    for task in tasks(config):
        source_row = context[task["size"]]["rows"][task["source_task_id"]]
        rows[task["task_id"]] = verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config, adaptive_tolerance=tolerance,
        )
    return {
        "paths": rows, "config_hash": config_hash, "source_hashes": hashes,
        "source": context, "adaptive_tolerance": tolerance,
    }


def _record_failure(output_root: Path, task: dict[str, Any], exc: BaseException) -> None:
    _atomic_json(
        failure_path(output_root, task),
        {
            "task": task,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "failed_unix": time.time(),
        },
    )


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, config, output_text, config_hash, hashes, source_row, tolerance, queue = payload
    output_root = Path(output_text)
    started = time.perf_counter()
    try:
        frame = _load_frame(source_row, int(task["Nx"]), int(task["Ny"]))
        metadata = _metadata(task, config_hash, hashes, source_row, tolerance)
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            arrays = compute_path(
                frame, task, config, output_root, metadata,
                adaptive_tolerance=float(tolerance), progress_queue=queue,
            )
        _validate_result_arrays(arrays, task, config)
        publish_result(
            output_root, task, arrays, config_hash=config_hash, hashes=hashes,
            source_row=source_row, adaptive_tolerance=float(tolerance),
            elapsed_seconds=time.perf_counter() - started,
        )
        ok, reason, _ = verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config, adaptive_tolerance=float(tolerance),
        )
        if not ok:
            raise RuntimeError(f"published result failed readback: {reason}")
        _delete_checkpoint(output_root, task)
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _selected_tasks(
    config: dict[str, Any], task_ids: Iterable[str] | None,
) -> list[dict[str, Any]]:
    rows = tasks(config)
    if not task_ids:
        return rows
    requested = set(task_ids)
    selected = [row for row in rows if row["task_id"] in requested]
    missing = sorted(requested - {row["task_id"] for row in selected})
    if missing:
        raise ValueError(f"unknown task IDs: {missing}")
    return selected


def write_identity(
    config: dict[str, Any], output_root: Path, status: dict[str, Any],
) -> None:
    _atomic_json(
        output_root / "campaign_identity.json",
        {
            "schema": CAMPAIGN_SCHEMA,
            "campaign_id": config["campaign_id"],
            "config_hash": status["config_hash"],
            "source_hashes": status["source_hashes"],
            "adaptive_tolerance": status["adaptive_tolerance"],
            "configuration": config,
            "verified_source_endpoints": {
                size: context["rows"] for size, context in status["source"].items()
            },
        },
    )


def print_inventory(config: dict[str, Any], output_root: Path, status: dict[str, Any]) -> None:
    done = sum(row[0] for row in status["paths"].values())
    print(f"[campaign] {config['campaign_id']}")
    print("[sources] 100 verified endpoints: 25 soft + 25 hard at each of N20x24 and N24x24")
    print("[selection] sample IDs 0,4,...,96; raw CW and CCW retained separately")
    print("[path] 129 regulated flux points; overlap branch; analytic Kato generator")
    print(
        f"[adaptive] RK4 step doubling tol={status['adaptive_tolerance']:.3e}; "
        "minimum h=one observation interval/4096"
    )
    print(f"[resume] verified={done}/200 pending={200-done}")
    print(f"[output] {output_root}")
    print(f"[identity] config_sha256={status['config_hash']}")


def run(
    config: dict[str, Any], output_root: Path, *, workers: int, resume: bool,
    task_ids: Iterable[str] | None = None, adaptive_tolerance: float | None = None,
) -> None:
    context = source_context(config)
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    tolerance = float(
        config["continuation"]["adaptive_tolerance"]
        if adaptive_tolerance is None else adaptive_tolerance
    )
    selected = _selected_tasks(config, task_ids)
    verified: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    initial_intervals = 0
    intervals = int(config["continuation"]["grid_intervals"])
    for task in selected:
        source_row = context[task["size"]]["rows"][task["source_task_id"]]
        ok, _, _ = verify_result(
            output_root, task, config_hash=config_hash, hashes=hashes,
            source_row=source_row, config=config, adaptive_tolerance=tolerance,
        )
        if resume and ok:
            verified.append(task)
            initial_intervals += intervals
            continue
        metadata = _metadata(task, config_hash, hashes, source_row, tolerance)
        checkpoint = _load_checkpoint(
            output_root, task, metadata=metadata, count=intervals + 1,
            nx=int(task["Nx"]), ambient=2 * int(task["Nx"]) * int(task["Ny"]),
            rank=int(source_row["rank"]),
        )
        if resume and checkpoint is not None:
            initial_intervals += int(checkpoint["completed_interval"])
        pending.append(task)
    print(
        f"[run] selected={len(selected)} verified={len(verified)} pending={len(pending)} "
        f"workers={workers} adaptive_tolerance={tolerance:.3e}", flush=True,
    )
    if not pending:
        return
    manager = mp.Manager()
    queue = manager.Queue()
    context_mp = mp.get_context("spawn")
    failures: list[str] = []
    with (
        tqdm(total=len(selected), initial=len(verified), desc="Kato paths", unit="path", position=0) as path_bar,
        tqdm(
            total=len(selected) * intervals, initial=initial_intervals,
            desc="Flux intervals", unit="interval", position=1,
        ) as interval_bar,
        ProcessPoolExecutor(max_workers=workers, mp_context=context_mp) as pool,
    ):
        future_map = {}
        for task in pending:
            source_row = context[task["size"]]["rows"][task["source_task_id"]]
            payload = (
                task, config, str(output_root), config_hash, hashes, source_row,
                tolerance, queue,
            )
            future_map[pool.submit(_worker, payload)] = task
        remaining = set(future_map)
        while remaining:
            drained = 0
            while True:
                try:
                    drained += int(queue.get_nowait())
                except Exception:
                    break
            if drained:
                interval_bar.update(drained)
            finished, remaining = wait(remaining, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                path_bar.update(1)
                if result["ok"]:
                    path_bar.set_postfix_str(result["task_id"], refresh=False)
                else:
                    failures.append(f"{result['task_id']}: {result['error']}")
                    path_bar.set_postfix_str(f"FAILED {result['task_id']}", refresh=True)
        while True:
            try:
                interval_bar.update(int(queue.get_nowait()))
            except Exception:
                break
    manager.shutdown()
    if failures:
        raise RuntimeError("Kato paths failed:\n" + "\n".join(failures))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--adaptive-tolerance", type=float, default=None)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_root = args.output_root.resolve()
    workers = int(config["execution"]["workers"] if args.workers is None else args.workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    tolerance = float(
        config["continuation"]["adaptive_tolerance"]
        if args.adaptive_tolerance is None else args.adaptive_tolerance
    )
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("adaptive tolerance must be finite and positive")
    context = source_context(config)
    status = inventory(
        config, output_root, context=context, adaptive_tolerance=tolerance
    )
    print_inventory(config, output_root, status)
    write_identity(config, output_root, status)
    if args.report_only:
        return 0
    run(
        config, output_root, workers=workers, resume=not args.no_resume,
        task_ids=args.task_id, adaptive_tolerance=tolerance,
    )
    final = inventory(
        config, output_root, context=context, adaptive_tolerance=tolerance
    )
    selected = _selected_tasks(config, args.task_id)
    done = sum(final["paths"][task["task_id"]][0] for task in selected)
    if done != len(selected):
        raise RuntimeError(f"final verification found {done}/{len(selected)} selected paths")
    print(f"[complete] verified={done}/{len(selected)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
