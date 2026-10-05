#!/usr/bin/env python3
"""Numerically robust recovery of the five failed v1 spectral-pump tasks.

This runner deliberately leaves the checksum-bound v1 runner and its 1,595
verified result/completion pairs unchanged.  It reuses the identical endpoint
states and scientific configuration, but publishes the recovered tasks in a
separate v2 output root with an explicit recovery provenance record.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import contextlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

import run_wall_diabatic_spectral_pump_s100 as base


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.wall_diabatic_numerical_recovery_v2.json"
DEFAULT_ENDPOINT_ROOT = PROJECT_ROOT / "imported_endpoints/wall_pump_width_endpoints_s100_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results/N20_24_28_32x24_wall_diabatic_spectral_pump_numerical_recovery_v2"
)


def _sha256(path: Path) -> str:
    return base.sha256_path(Path(path))


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(base.canonical_json(payload).encode("utf-8")).hexdigest()


def load_recovery_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    recovery = json.loads(Path(path).read_text(encoding="utf-8"))
    if recovery.get("schema") != "wall_diabatic_spectral_pump_numerical_recovery_campaign_v2":
        raise ValueError("unexpected numerical-recovery schema")
    if recovery.get("campaign_id") != (
        "N20_24_28_32x24_wall_diabatic_spectral_pump_numerical_recovery_v2"
    ):
        raise ValueError("unexpected numerical-recovery campaign identity")
    parent_config_path = PROJECT_ROOT / str(recovery["parent_config"])
    parent_runner_path = PROJECT_ROOT / str(recovery["parent_runner"])
    if _sha256(parent_config_path) != recovery.get("parent_config_sha256"):
        raise RuntimeError("the checksum-pinned v1 configuration changed")
    if _sha256(parent_runner_path) != recovery.get("parent_runner_sha256"):
        raise RuntimeError("the checksum-pinned v1 runner changed")
    parent = base.load_config(parent_config_path)
    base.validate_config(parent)
    expected = {
        "wall_diabatic_dense_N28x24_hard_sample_014",
        "wall_diabatic_dense_N32x24_hard_sample_012",
        "wall_diabatic_dense_N32x24_hard_sample_037",
        "wall_diabatic_nsh1_N24x24_hard_sample_092",
        "wall_diabatic_nsh1_N32x24_hard_sample_068",
    }
    actual = set(recovery.get("recovery_task_ids", []))
    if actual != expected or len(recovery.get("recovery_task_ids", [])) != 5:
        raise ValueError("recovery selection must contain the five locked v1 failures")
    numeric = recovery.get("numerical_recovery", {})
    if numeric.get("scientific_thresholds_changed") is not False:
        raise ValueError("the recovery must not change scientific thresholds")
    return recovery, parent


def recovery_source_hashes(config_path: Path, recovery: dict[str, Any]) -> dict[str, str]:
    return {
        "parent_campaign_runner": _sha256(PROJECT_ROOT / recovery["parent_runner"]),
        "parent_campaign_config": _sha256(PROJECT_ROOT / recovery["parent_config"]),
        "numerical_recovery_runner": _sha256(Path(__file__).resolve()),
        "numerical_recovery_config": _sha256(Path(config_path).resolve()),
    }


def _fallback_checks(
    matrix: np.ndarray, values: np.ndarray, vectors: np.ndarray, numeric: dict[str, Any]
) -> None:
    gram = vectors.conj().T @ vectors
    gram_error = float(np.max(np.abs(gram - np.eye(vectors.shape[1]))))
    scale = max(1.0, float(np.linalg.norm(matrix, ord=np.inf)))
    eigen_error = float(
        np.max(np.abs(matrix @ vectors - vectors * values[None, :])) / scale
    )
    if gram_error > float(numeric["fallback_orthonormality_tolerance"]):
        raise RuntimeError(f"fallback eigenvectors are not orthonormal: {gram_error:.3e}")
    if eigen_error > float(numeric["fallback_eigenpair_residual_tolerance"]):
        raise RuntimeError(f"fallback eigenpair residual is too large: {eigen_error:.3e}")


_ACTIVE_NUMERICAL_CONFIG: dict[str, Any] | None = None


def _recovery_parent_eigensystem(
    h0: np.ndarray, dy: np.ndarray, phi: float, ny: int, y: np.ndarray,
    twist_gauge: str, subset: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the v1 solve first, then validated independent LAPACK fallbacks."""
    if _ACTIVE_NUMERICAL_CONFIG is None:
        raise RuntimeError("numerical recovery is not configured")
    numeric = _ACTIVE_NUMERICAL_CONFIG
    matrix = base.twisted_parent(h0, dy, phi, ny, twist_gauge=twist_gauge, y=y)
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"nonfinite threaded parent at phi={phi:.17g}")
    hermiticity = float(np.max(np.abs(matrix - matrix.conj().T)))
    if hermiticity > 1e-13:
        raise RuntimeError(
            f"threaded parent is not Hermitian at phi={phi:.17g}: {hermiticity:.3e}"
        )

    attempts: list[tuple[str, str | None]]
    if subset is None:
        attempts = [("numpy_heevd", None)] + [
            (name, name.removeprefix("scipy_"))
            for name in numeric["full_fallback_drivers"]
        ]
    else:
        attempts = [
            (name, name.removeprefix("scipy_")) for name in numeric["subset_drivers"]
        ]
    errors: list[str] = []
    for attempt_index, (label, driver) in enumerate(attempts):
        try:
            if label == "numpy_heevd":
                values, vectors = np.linalg.eigh(matrix)
            else:
                values, vectors = scipy.linalg.eigh(
                    matrix.copy(), subset_by_index=subset, driver=driver,
                    check_finite=False, overwrite_a=True,
                )
            values = np.asarray(values, dtype=np.float64)
            vectors = np.asarray(vectors, dtype=np.complex128)
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(vectors)):
                raise RuntimeError("eigensystem contains NaN or Inf")
            if attempt_index > 0:
                _fallback_checks(matrix, values, vectors, numeric)
            if twist_gauge == "seam":
                basis = np.exp(1j * float(phi) * y / ny)
                vectors = basis[:, None] * vectors
            return values, vectors
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "all Hermitian eigensolvers failed at "
        f"phi={phi:.17g}, subset={subset}: " + " | ".join(errors)
    )


def _polar_stabilize(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Restore frame orthonormality without changing its occupied subspace."""
    if _ACTIVE_NUMERICAL_CONFIG is None:
        raise RuntimeError("numerical recovery is not configured")
    numeric = _ACTIVE_NUMERICAL_CONFIG
    gram = frame.conj().T @ frame
    gram = 0.5 * (gram + gram.conj().T)
    before = float(np.max(np.abs(gram - np.eye(frame.shape[1]))))
    if not np.isfinite(before):
        raise RuntimeError("continued frame Gram matrix contains NaN or Inf")
    if before <= float(numeric["frame_polar_trigger"]):
        return np.asarray(frame, dtype=np.complex128), before
    values, vectors = scipy.linalg.eigh(
        gram, driver="evr", check_finite=True, overwrite_a=True,
    )
    if not np.all(np.isfinite(values)) or float(np.min(values)) <= 0.0:
        raise RuntimeError("continued frame lost rank before polar stabilization")
    inverse_root = (vectors * values[None, :] ** -0.5) @ vectors.conj().T
    stabilized = np.asarray(frame @ inverse_root, dtype=np.complex128)
    after_gram = stabilized.conj().T @ stabilized
    after = float(np.max(np.abs(after_gram - np.eye(stabilized.shape[1]))))
    if after > float(numeric["frame_polar_acceptance"]):
        raise RuntimeError(f"polar frame stabilization failed: {after:.3e}")
    return stabilized, after


def _recovery_branch_path(
    frame: np.ndarray, h0: np.ndarray, dy: np.ndarray, phi: np.ndarray,
    rank: int, nx: int, ny: int, mode_x: np.ndarray, y: np.ndarray,
    b_diagonal: np.ndarray, qualification: dict[str, Any],
    edge_block_rank: int, twist_gauge: str, queue: Any = None,
) -> dict[str, Any]:
    """The v1 branch rule with conditional, subspace-preserving polar cleanup."""
    count = phi.size
    left, right = np.empty(count), np.empty(count)
    density_x = np.empty((count, nx))
    ordinary_left, ordinary_right = np.empty(count), np.empty(count)
    instant_left, instant_right = np.empty(count), np.empty(count)
    principal, weight_floor = np.empty(count), np.empty(count)
    ordinary_principal, ordinary_weight_floor = np.empty(count), np.empty(count)
    projector_residual, charge_residual = np.empty(count), np.empty(count)
    previous_main = np.asarray(frame, dtype=np.complex128)
    previous_ordinary = np.asarray(frame, dtype=np.complex128)
    previous_spectator = previous_cluster = previous_modes = None
    occupied_in_block = int(edge_block_rank) // 2
    entering_labels = None
    start_main = endpoint_main = endpoint_ordinary = endpoint_instant = None
    active = np.asarray(qualification["active"], dtype=bool)

    for point, value in enumerate(phi):
        eigenvalues, eigenvectors = _recovery_parent_eigensystem(
            h0, dy, float(value), ny, y, twist_gauge,
        )
        ordinary, ordinary_principal[point], ordinary_weight_floor[point] = (
            base._ordinary_select(previous_ordinary, eigenvalues, eigenvectors, rank)
        )
        instantaneous = np.asarray(eigenvectors[:, :rank], dtype=np.complex128)
        if active[point]:
            edge_indices = tuple(range(rank - occupied_in_block, rank + occupied_in_block))
            cluster = np.asarray(eigenvectors[:, list(edge_indices)], dtype=np.complex128)
            _, modes = base._wall_modes(cluster, b_diagonal, previous_cluster, previous_modes)
            reference = previous_main if previous_spectator is None else previous_spectator
            spectator = base._spectator_select(
                reference, eigenvalues, eigenvectors, rank - occupied_in_block, edge_indices
            )
            if entering_labels is None:
                occupations = np.real(
                    np.sum(np.abs(previous_main.conj().T @ modes) ** 2, axis=0)
                )
                entering_labels = np.sort(
                    np.argsort(-occupations, kind="stable")[:occupied_in_block]
                )
            raw_main = np.column_stack((spectator, modes[:, entering_labels]))
            previous_spectator, previous_cluster, previous_modes = spectator, cluster, modes
            singular = np.linalg.svd(previous_main.conj().T @ raw_main, compute_uv=False)
            weights = np.real(
                np.sum(np.abs(previous_main.conj().T @ raw_main) ** 2, axis=0)
            )
            principal[point], weight_floor[point] = (
                float(np.min(singular)), float(np.min(weights))
            )
        else:
            raw_main, principal[point], weight_floor[point] = base._ordinary_select(
                previous_main, eigenvalues, eigenvectors, rank
            )
            previous_spectator = previous_cluster = previous_modes = None

        main, projector_residual[point] = _polar_stabilize(raw_main)
        left[point], right[point], density_x[point] = base._frame_observables(main, mode_x, nx)
        ordinary_left[point], ordinary_right[point], _ = base._frame_observables(
            ordinary, mode_x, nx
        )
        instant_left[point], instant_right[point], _ = base._frame_observables(
            instantaneous, mode_x, nx
        )
        charge_residual[point] = abs(left[point] + right[point] - rank)
        previous_main, previous_ordinary = main, ordinary
        start_main = main if point == 0 else start_main
        endpoint_main, endpoint_ordinary, endpoint_instant = main, ordinary, instantaneous
        if queue is not None:
            queue.put(1)

    assert start_main is not None and endpoint_main is not None
    assert endpoint_ordinary is not None and endpoint_instant is not None
    return {
        "left": left, "right": right, "density_x": density_x,
        "ordinary_left": ordinary_left, "ordinary_right": ordinary_right,
        "instant_left": instant_left, "instant_right": instant_right,
        "principal": principal, "weight_floor": weight_floor,
        "ordinary_principal": ordinary_principal,
        "ordinary_weight_floor": ordinary_weight_floor,
        "projector_residual": projector_residual,
        "charge_residual": charge_residual,
        "entering_labels": (
            np.full(occupied_in_block, -1, dtype=np.int8)
            if entering_labels is None else entering_labels.astype(np.int8)
        ),
        "start_main": start_main,
        "endpoint_main": endpoint_main,
        "endpoint_ordinary": endpoint_ordinary,
        "endpoint_instant": endpoint_instant,
    }


def _install_recovery_numerics(numeric: dict[str, Any]) -> None:
    global _ACTIVE_NUMERICAL_CONFIG
    _ACTIVE_NUMERICAL_CONFIG = dict(numeric)
    base._parent_eigensystem = _recovery_parent_eigensystem
    base._branch_path = _recovery_branch_path


def _recovery_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, ref, parent, recovery, output_text, config_hash, hashes, queue = payload
    started = time.perf_counter()
    output_root = Path(output_text)
    _install_recovery_numerics(recovery["numerical_recovery"])
    log_path = output_root / "logs/tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = base.load_endpoint(ref, int(task["Nx"]), int(task["Ny"]))
        with threadpool_limits(limits=int(recovery["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, \
                    contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                arrays = base.compute_wall_diabatic_pump(frame, task, parent, queue=queue)
        base.publish_pair(
            output_root, task, arrays, config_hash, hashes, ref,
            time.perf_counter() - started,
        )
        base.failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        base._atomic_json(base.failure_path(output_root, task), {
            "task_id": task["task_id"], "failed_unix": time.time(),
            "error_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        })
        return {
            "ok": False, "task_id": task["task_id"],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _selected_tasks(parent: dict[str, Any], recovery: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        row["task_id"]: row for row in base.tasks(parent, include_bridge=False)
    }
    rows = []
    for task_id in recovery["recovery_task_ids"]:
        task = dict(by_id[task_id])
        task["recovery_revision"] = recovery["campaign_id"]
        task["parent_campaign_id"] = recovery["parent_campaign_id"]
        rows.append(task)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "run"), nargs="?", default="report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--new-endpoint-root", type=Path, default=DEFAULT_ENDPOINT_ROOT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    recovery, parent = load_recovery_config(config_path)
    _install_recovery_numerics(recovery["numerical_recovery"])
    output_root = args.output_root.resolve()
    endpoint_root = args.new_endpoint_root.resolve()
    rows = _selected_tasks(parent, recovery)
    source_map = base.source_rows(parent, include_bridge=False)
    refs, missing = base._discover(rows, parent, endpoint_root)
    hashes = recovery_source_hashes(config_path, recovery)
    config_hash = _canonical_hash(recovery)
    verified, pending, invalid = [], [], {}
    for task in rows:
        ref = refs.get(task["task_id"])
        if ref is None:
            continue
        ok, reason, _ = base.verify_pair(
            output_root, task, parent, config_hash, hashes, ref
        )
        if ok:
            verified.append(task)
        else:
            pending.append(task)
            if "missing result/completion" not in reason:
                invalid[task["task_id"]] = reason

    print(f"[campaign] {recovery['campaign_id']}")
    print(f"[parent] {recovery['parent_campaign_id']} (immutable)")
    print(f"[sources] verified={len(refs)}/5 unavailable={len(missing)}")
    print(f"[resume] verified={len(verified)}/5 pending={len(pending)} invalid={len(invalid)}")
    print("[numerics] NumPy Hermitian solve + SciPy evr/evd/evx fallbacks; conditional polar frame stabilization")
    print("[science] endpoint states, M=256 grid, regulator, branch rule, and acceptance gates unchanged")
    print(f"[output] {output_root}")
    if args.stage == "report":
        return 0
    if missing:
        raise RuntimeError(f"{len(missing)} checksum-pinned endpoint sources are unavailable")
    output_root.mkdir(parents=True, exist_ok=True)
    base._atomic_json(output_root / "campaign_identity.json", {
        "schema": recovery["schema"], "campaign_id": recovery["campaign_id"],
        "parent_campaign_id": recovery["parent_campaign_id"],
        "configuration": recovery, "configuration_sha256": config_hash,
        "source_hashes": hashes,
    })
    run_rows = pending if args.resume else rows
    if not run_rows:
        print("[complete] all five numerical-recovery tasks are verified")
        return 0
    workers = int(args.workers or recovery["execution"]["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    context = mp.get_context("spawn")
    points_per_task = 4 * (int(parent["flux"]["grid_intervals"]) + 1)
    failures: list[dict[str, Any]] = []
    with context.Manager() as manager:
        queue = manager.Queue()
        with tqdm(
            total=5, initial=(len(verified) if args.resume else 0),
            desc="recovery tasks", unit="task", position=0,
        ) as task_bar, tqdm(
            total=5 * points_per_task,
            initial=(len(verified) * points_per_task if args.resume else 0),
            desc="continued flux points", unit="point", position=1,
        ) as point_bar, ProcessPoolExecutor(
            max_workers=min(workers, len(run_rows)), mp_context=context,
        ) as pool:
            futures = {
                pool.submit(_recovery_worker, (
                    task, refs[task["task_id"]], parent, recovery,
                    str(output_root), config_hash, hashes, queue,
                ))
                for task in run_rows
            }
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        point_bar.update(int(queue.get_nowait()))
                    except Exception:
                        break
                for future in done:
                    row = future.result()
                    task_bar.update(1)
                    if not row["ok"]:
                        failures.append(row)
                        task_bar.write(f"[failure] {row['task_id']}: {row['error']}")
            while True:
                try:
                    point_bar.update(int(queue.get_nowait()))
                except Exception:
                    break
    if failures:
        raise RuntimeError(f"{len(failures)} numerical-recovery tasks failed")
    print("[complete] all five numerical-recovery tasks finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
