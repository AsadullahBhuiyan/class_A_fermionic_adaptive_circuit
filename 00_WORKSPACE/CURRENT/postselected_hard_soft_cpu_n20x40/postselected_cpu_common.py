#!/usr/bin/env python3
"""Shared implementation for the separate hard/soft postselected CPU runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NX = 20
NY = 40
DEFAULT_CYCLES = 4 * NY
NSHELL = 1
ALPHA_1 = 1.0
ALPHA_2 = 30.0
ROOT_SEED = 2026092401
SCHEMA = "postselected_maxmix_n20x40_cpu_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _verified_complete(result_path: Path, completion_path: Path) -> bool:
    if not result_path.is_file() or not completion_path.is_file():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        return (
            completion["schema"] == SCHEMA
            and completion["result_filename"] == result_path.name
            and int(completion["bytes"]) == result_path.stat().st_size
            and completion["sha256"] == _sha256(result_path)
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _allocate(cycles: int) -> dict[str, np.ndarray]:
    return {
        "total_entropy_nats": np.full(cycles + 1, np.nan, dtype=np.float64),
        "active_charge": np.full(cycles + 1, np.nan, dtype=np.float64),
        "centered_min_abs": np.full(cycles + 1, np.nan, dtype=np.float64),
        "raw_logit_gap": np.full(cycles + 1, np.nan, dtype=np.float64),
        "lyapunov_gap": np.full(cycles + 1, np.nan, dtype=np.float64),
        "hermiticity_residual": np.full(cycles + 1, np.nan, dtype=np.float64),
        "spectral_bound_violation": np.full(cycles + 1, np.nan, dtype=np.float64),
    }


def _spectrum_observables(centered_covariance: np.ndarray, cycle: int) -> tuple[dict[str, float], np.ndarray]:
    matrix = np.asarray(centered_covariance, dtype=np.complex128)
    hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
    if not np.isfinite(hermiticity) or hermiticity > 2e-9:
        raise FloatingPointError(f"Hermiticity residual {hermiticity:.3e} at cycle {cycle}")
    centered = np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))
    violation = float(max(0.0, -1.0 - centered.min(), centered.max() - 1.0))
    if violation > 2e-8:
        raise FloatingPointError(f"Centered spectrum left [-1,1] by {violation:.3e} at cycle {cycle}")
    centered = np.clip(centered, -1.0, 1.0)
    occupations = 0.5 * (1.0 + centered)
    interior = (occupations > 0.0) & (occupations < 1.0)
    entropy = float(
        -np.sum(
            occupations[interior] * np.log(occupations[interior])
            + (1.0 - occupations[interior]) * np.log1p(-occupations[interior])
        )
    )
    centered_min_abs = float(np.min(np.abs(centered)))
    finite = np.abs(centered) < 1.0
    raw_half_logit = (
        float(np.min(np.abs(np.arctanh(centered[finite]))))
        if np.any(finite)
        else float("inf")
    )
    return {
        "total_entropy_nats": entropy,
        "active_charge": float(np.sum(occupations)),
        "centered_min_abs": centered_min_abs,
        "raw_logit_gap": 2.0 * raw_half_logit,
        "lyapunov_gap": float("nan") if cycle == 0 else raw_half_logit / float(cycle),
        "hermiticity_residual": hermiticity,
        "spectral_bound_violation": violation,
    }, centered


def _load_checkpoint(path: Path, *, construction: str, cycles: int, arrays: dict[str, np.ndarray]):
    if not path.is_file():
        return None, 0.0
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"Checkpoint schema mismatch: {path}")
    if payload.get("construction") != construction or int(payload.get("target_cycles", -1)) != cycles:
        raise RuntimeError(f"Checkpoint contract mismatch: {path}")
    completed = int(payload["engine_state"]["completed_cycles"])
    for key, destination in arrays.items():
        source = np.asarray(payload["observables"][key], dtype=destination.dtype)
        destination[: source.size] = source
    return payload["engine_state"], float(payload.get("elapsed_seconds", 0.0))


def run(construction: str, argv: list[str] | None = None) -> int:
    if construction not in {"hard", "soft"}:
        raise ValueError("construction must be hard or soft")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--checkpoint-stride", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    args = parser.parse_args(argv)
    if args.cycles <= 0 or args.checkpoint_stride <= 0:
        parser.error("cycles and checkpoint-stride must be positive")

    from src.fgtn.classA_U1FGTN import classA_U1FGTN

    hard = construction == "hard"
    output_dir = args.output_root.resolve() / construction
    result_path = output_dir / "postselected_trajectory.npz"
    completion_path = output_dir / "completion.json"
    checkpoint_path = output_dir / "checkpoint.pkl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if _verified_complete(result_path, completion_path):
        print(f"[{construction}] verified result already complete: {result_path}", flush=True)
        return 0

    source_path = REPO_ROOT / "src" / "fgtn" / "classA_U1FGTN.py"
    config = {
        "schema": SCHEMA,
        "construction": construction,
        "Nx": NX,
        "Ny": NY,
        "cycles": int(args.cycles),
        "samples": 1,
        "postselect": True,
        "postselect_probability": 1.0,
        "init_mode": "maxmix",
        "DW": True,
        "dw_truncation": hard,
        "meas_slab_only": hard,
        "nshell": NSHELL,
        "alpha_1": ALPHA_1,
        "alpha_2": ALPHA_2,
        "sequence": "raster_y",
        "perfect_correction": False,
        "dtype": "complex128",
        "root_seed": ROOT_SEED,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
        "engine_source": str(source_path),
        "engine_sha256": _sha256(source_path),
        "gap_convention": "min(abs(-atanh(a_j)/t)), a_j=eig((G+G^dagger)/2)",
    }
    print(f"[{construction}] configuration\n{json.dumps(config, indent=2, sort_keys=True)}", flush=True)

    model = classA_U1FGTN(
        NX,
        NY,
        DW=True,
        nshell=NSHELL,
        alpha_1=ALPHA_1,
        alpha_2=ALPHA_2,
        trial_orbitals="X",
        dw_truncation=hard,
    )
    model.construct_OW_projectors(
        nshell=NSHELL,
        DW=True,
        trial_orbitals="X",
        dw_truncation=hard,
    )
    effective = model._meas_slab_only_effective(hard)
    if effective is not hard:
        raise RuntimeError("Unexpected slab-only resolution")
    active_indices = np.asarray(model.active_top_layer_indices(meas_slab_only=hard), dtype=np.int64)
    expected_modes = (22 * NY) if hard else (40 * NY)
    if active_indices.size != expected_modes:
        raise RuntimeError(f"Expected {expected_modes} analyzed modes, found {active_indices.size}")
    config["wall_locations"] = [int(value) for value in model.DW_loc]
    config["active_mode_count"] = int(active_indices.size)
    print(
        f"[{construction}] walls={config['wall_locations']} active_modes={active_indices.size} "
        f"meas_slab_only_effective={effective}",
        flush=True,
    )

    arrays = _allocate(args.cycles)
    checkpoint_state, elapsed_before = _load_checkpoint(
        checkpoint_path, construction=construction, cycles=args.cycles, arrays=arrays
    )
    completed_before = -1 if checkpoint_state is None else int(checkpoint_state["completed_cycles"])
    print(f"[{construction}] resume completed_cycle={completed_before}", flush=True)
    started = time.perf_counter()

    def observe(*, cycle: int, G: np.ndarray, **_: Any) -> None:
        full = np.asarray(G, dtype=np.complex128)
        active = full[np.ix_(active_indices, active_indices)]
        values, _ = _spectrum_observables(active, int(cycle))
        for key, value in values.items():
            arrays[key][int(cycle)] = value
        if cycle == 0 or cycle == args.cycles or cycle % 10 == 0:
            print(
                f"[{construction}] observable cycle={cycle}/{args.cycles} "
                f"S={values['total_entropy_nats']:.9g} "
                f"Delta={values['lyapunov_gap']:.9g}",
                flush=True,
            )

    def checkpoint_observer(*, cycle: int, state: dict[str, Any]) -> None:
        cycle = int(cycle)
        if cycle % args.checkpoint_stride and cycle != args.cycles:
            return
        partial = {key: np.array(value[: cycle + 1], copy=True) for key, value in arrays.items()}
        _atomic_pickle(
            checkpoint_path,
            {
                "schema": SCHEMA,
                "construction": construction,
                "target_cycles": int(args.cycles),
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "engine_state": state,
                "observables": partial,
            },
        )
        print(f"[{construction}] checkpoint cycle={cycle}: {checkpoint_path}", flush=True)

    if checkpoint_state is not None and completed_before == args.cycles:
        final_G = np.asarray(checkpoint_state["G"], dtype=np.complex128)
        print(f"[{construction}] final engine checkpoint found; finishing endpoint publication", flush=True)
    else:
        final_G = model.run_markov_circuit(
            G_history=False,
            progress=True,
            cycles=int(args.cycles),
            postselect=True,
            perfect_correction=False,
            samples=1,
            parallelize_samples=False,
            init_mode="maxmix",
            save=False,
            n_a=0.5,
            sequence="raster_y",
            meas_slab_only=hard,
            random_seed=ROOT_SEED,
            state_representation="covariance",
            cycle_observer=observe,
            checkpoint_state=checkpoint_state,
            checkpoint_observer=checkpoint_observer,
        )
        final_G = np.asarray(final_G, dtype=np.complex128)

    # The only intentionally non-finite entry is the undefined t=0 rate gap.
    for key, value in arrays.items():
        if key == "lyapunov_gap":
            if not np.isnan(value[0]) or not np.all(np.isfinite(value[1:])):
                raise RuntimeError(f"Incomplete observable {key}")
        elif not np.all(np.isfinite(value)):
            raise RuntimeError(f"Incomplete observable {key}")

    active_final = final_G[np.ix_(active_indices, active_indices)]
    endpoint_centered, endpoint_vectors = np.linalg.eigh(
        0.5 * (active_final + active_final.conj().T)
    )
    endpoint_centered = np.clip(endpoint_centered, -1.0, 1.0)
    endpoint_occupations = 0.5 * (1.0 + endpoint_centered)
    elapsed = elapsed_before + time.perf_counter() - started
    _atomic_npz(
        result_path,
        schema=np.asarray(SCHEMA),
        config_json=np.asarray(json.dumps(config, sort_keys=True)),
        cycles=np.arange(args.cycles + 1, dtype=np.int64),
        normalized_cycles=np.arange(args.cycles + 1, dtype=np.float64) / NY,
        analyzed_basis_indices=active_indices,
        endpoint_centered_covariance=active_final,
        endpoint_centered_spectrum=endpoint_centered,
        endpoint_occupations=endpoint_occupations,
        endpoint_eigenvectors=endpoint_vectors,
        elapsed_seconds=np.asarray(elapsed, dtype=np.float64),
        **arrays,
    )
    result_bytes = result_path.stat().st_size
    result_hash = _sha256(result_path)
    with np.load(result_path, allow_pickle=False) as check:
        if check["cycles"].shape != (args.cycles + 1,) or int(check["cycles"][-1]) != args.cycles:
            raise RuntimeError("Result readback failed")
    _atomic_json(
        completion_path,
        {
            "schema": SCHEMA,
            "construction": construction,
            "result_filename": result_path.name,
            "bytes": result_bytes,
            "sha256": result_hash,
            "elapsed_seconds": elapsed,
            "completed_cycles": int(args.cycles),
        },
    )
    if not _verified_complete(result_path, completion_path):
        raise RuntimeError("Completion verification failed")
    checkpoint_path.unlink(missing_ok=True)
    print(
        f"[{construction}] complete elapsed={elapsed / 3600:.3f} h "
        f"endpoint_gap={arrays['lyapunov_gap'][-1]:.9g} result={result_path}",
        flush=True,
    )
    return 0
