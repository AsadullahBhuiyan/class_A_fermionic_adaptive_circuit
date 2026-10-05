#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from campaign_schema import (
    case_requires_response,
    expand_cases,
    load_config,
    sha256_file,
    spectral_checkpoint_cycles,
)
from dynamics_adapters import (
    LINDBLAD_ADAPTER_READY,
    MARKOV_ADAPTER_READY,
    AdapterResult,
    run_lindblad_case,
    run_markov_channel_case,
)
from observables import enrich_terminal_arrays


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_CONFIG = HERE / "campaign_config.v1.json"
_WORKER_CPU: int | None = None
_WORKER_THREADS: int | None = None


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=REPO, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short", "--untracked-files=no"),
    }


def _source_hashes(config_path: Path) -> dict[str, str | None]:
    candidates = {
        "runner": Path(__file__),
        "config": config_path,
        "campaign_schema": HERE / "campaign_schema.py",
        "dynamics_adapters": HERE / "dynamics_adapters.py",
        "markov_adapter": HERE / "markov_adapter.py",
        "lindblad_adapter": HERE / "lindblad_adapter.py",
        "lindblad_response": HERE / "lindblad_response.py",
        "matched_model": HERE / "matched_model.py",
        "observables": HERE / "observables.py",
        "analysis": HERE / "analyze_campaign.py",
        "canonical_cpu_class": REPO / "src" / "fgtn" / "classA_U1FGTN.py",
        "canonical_mean_lindblad": REPO / "src" / "fgtn" / "diagnostics" / "mean_lindblad.py",
        "lindblad_reference": Path("/home/abhuiyan/Fermionic_Lindbladian/CI_Lindblad_DW.py"),
    }
    return {
        name: sha256_file(path) if path.exists() else None
        for name, path in candidates.items()
    }


def _validate_adapter_result(
    case: dict[str, Any], result: AdapterResult, config: dict[str, Any]
) -> None:
    arrays = result.arrays
    cycles = int(case["dynamics"]["cycles"])
    samples = len(case["dynamics"]["sample_ids"])
    expected_cycle = np.arange(cycles + 1, dtype=np.int64)
    if "cycle" not in arrays or not np.array_equal(np.asarray(arrays["cycle"]), expected_cycle):
        raise ValueError("adapter must return the exact cycle coordinate 0..2Ny")
    required_cycle = set(config["observation_contract"]["cycle_resolved_observables"])
    for name in required_cycle:
        if name not in arrays:
            raise ValueError(f"adapter omitted cycle observable {name!r}")
        values = np.asarray(arrays[name])
        if values.shape != (samples, cycles + 1):
            raise ValueError(f"{name} must have shape ({samples},{cycles + 1})")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite entries")

    expected_checkpoints = np.asarray(spectral_checkpoint_cycles(case), dtype=np.int64)
    if "spectral_checkpoint_cycle" not in arrays or not np.array_equal(
        np.asarray(arrays["spectral_checkpoint_cycle"]), expected_checkpoints
    ):
        raise ValueError("adapter returned the wrong spectral checkpoint coordinate")
    required_spectral = set(
        config["observation_contract"]["spectral_checkpoint_observables"]
    )
    for name in required_spectral:
        if name not in arrays:
            raise ValueError(f"adapter omitted spectral-checkpoint observable {name!r}")
        values = np.asarray(arrays[name])
        if values.shape != (samples, expected_checkpoints.size):
            raise ValueError(
                f"{name} must have shape ({samples},{expected_checkpoints.size})"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite entries")
    physicality_tolerance = float(config["analysis"]["physicality_tolerance"])
    if np.min(arrays["occupation_min"]) < -physicality_tolerance:
        raise ValueError("checkpoint occupation minimum is below zero")
    if np.max(arrays["occupation_max"]) > 1.0 + physicality_tolerance:
        raise ValueError("checkpoint occupation maximum exceeds one")

    nx, ny = int(case["model"]["Nx"]), int(case["model"]["Ny"])
    dimension = 2 * nx * ny
    for name in ("G_final", "G_late_cycle_average"):
        if name not in arrays:
            raise ValueError(f"adapter omitted dense checkpoint {name!r}")
        matrix = np.asarray(arrays[name])
        if matrix.shape != (samples, dimension, dimension):
            raise ValueError(
                f"{name} must have shape ({samples},{dimension},{dimension})"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} contains non-finite entries")
        scale = max(float(np.linalg.norm(matrix)), 1.0)
        if float(np.linalg.norm(matrix - matrix.conj().swapaxes(-1, -2))) > 1e-10 * scale:
            raise ValueError(f"{name} is not Hermitian within tolerance")

    if "leading_relaxation_spectrum" not in arrays:
        raise ValueError("adapter omitted leading_relaxation_spectrum")
    leading = np.asarray(arrays["leading_relaxation_spectrum"])
    if leading.ndim != 2 or leading.shape[0] != samples or not np.all(np.isfinite(leading)):
        raise ValueError("leading_relaxation_spectrum must be finite with a sample axis")

    if case["dynamics"]["family"] == "markov_channel":
        schedule = np.asarray(arrays.get("schedule_site_ids"))
        if schedule.shape != (samples, cycles, nx * ny):
            raise ValueError("channel adapter omitted the complete realized schedule words")
        if schedule.size and (
            int(np.min(schedule)) < 0 or int(np.max(schedule)) >= nx * ny
        ):
            raise ValueError("channel schedule contains an out-of-range site ID")
        if not np.all(
            np.sort(schedule, axis=-1)
            == np.arange(nx * ny, dtype=schedule.dtype)[None, None, :]
        ):
            raise ValueError("a channel schedule word is not a permutation of all sites")

    response_required = case_requires_response(case, config)
    response_names = {
        "response_times",
        "response_source_y",
        "response_density_source_wall_time_y",
        "response_density_ty_mean_source",
        "response_norm_time",
        "response_wall_retention_time",
        "response_positive_center_time",
        "response_velocity_source_wall",
        "response_velocity_r2_source_wall",
        "response_epsilon_relative_error",
    }
    missing_response = response_names.difference(arrays)
    if response_required and missing_response:
        raise ValueError(f"adapter omitted response arrays: {sorted(missing_response)}")
    if bool(result.metadata.get("response_enabled", False)) != response_required:
        raise ValueError("adapter response scope does not match the campaign contract")
    if response_required:
        times = np.asarray(arrays["response_times"])
        source_y = np.asarray(arrays["response_source_y"])
        raw = np.asarray(arrays["response_density_source_wall_time_y"])
        mean = np.asarray(arrays["response_density_ty_mean_source"])
        source_count = int(source_y.size)
        time_count = int(times.size)
        if (
            times.ndim != 1
            or time_count == 0
            or not np.all(np.isfinite(times))
            or not np.isclose(times[0], 0.0)
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("response_times must be a finite increasing coordinate from zero")
        if (
            source_y.ndim != 1
            or np.unique(source_y).size != source_count
            or np.any(source_y < 0)
            or np.any(source_y >= ny)
        ):
            raise ValueError("response_source_y contains invalid source coordinates")
        if raw.shape != (samples, source_count, 2, time_count, ny):
            raise ValueError("response_density_source_wall_time_y has the wrong shape")
        if mean.shape != (samples, 2, time_count, ny):
            raise ValueError("response_density_ty_mean_source has the wrong shape")
        for name in (
            "response_norm_time",
            "response_wall_retention_time",
            "response_positive_center_time",
        ):
            if np.asarray(arrays[name]).shape != (samples, source_count, 2, time_count):
                raise ValueError(f"{name} has the wrong shape")
        for name in ("response_velocity_source_wall", "response_velocity_r2_source_wall"):
            if np.asarray(arrays[name]).shape != (samples, source_count, 2):
                raise ValueError(f"{name} has the wrong shape")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(mean)):
            raise ValueError("saved response profiles contain non-finite entries")
        if not np.all(np.isfinite(arrays["response_norm_time"])):
            raise ValueError("response_norm_time contains non-finite entries")
        epsilon_error = np.asarray(arrays["response_epsilon_relative_error"])
        if epsilon_error.shape != (
            samples,
            len(config["response"]["epsilon_multipliers"]),
        ) or not np.all(np.isfinite(epsilon_error)):
            raise ValueError("response_epsilon_relative_error has the wrong contract")
    if result.metadata.get("canonical_dynamics_entry_point") is None:
        raise ValueError("adapter metadata must identify its canonical dynamics entry point")
    if list(result.metadata.get("sample_seeds", ())) != list(
        case["dynamics"]["sample_seeds"]
    ):
        raise ValueError("adapter metadata does not preserve the declared sample seeds")
    if list(result.metadata.get("wall_locations", ())) != list(
        case["model"]["wall_locations"]
    ):
        raise ValueError("adapter metadata does not preserve the explicit wall locations")


def _run_one(
    case: dict[str, Any],
    config: dict[str, Any],
    cases_dir: Path,
    *,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_dir = cases_dir / case["case_id"]
    if final_dir.exists():
        raise FileExistsError(f"refusing to overwrite case directory {final_dir}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{case['case_id']}.tmp-", dir=cases_dir))
    started = time.perf_counter()
    try:
        if case["dynamics"]["family"] == "markov_channel":
            result = run_markov_channel_case(case, config)
        elif case["dynamics"]["family"] == "lindblad":
            result = run_lindblad_case(case, config)
        else:
            raise ValueError(f"unknown dynamics family {case['dynamics']['family']!r}")
        _validate_adapter_result(case, result, config)
        arrays = enrich_terminal_arrays(case, result.arrays)
        observables = temporary / "observables.npz"
        _atomic_npz(observables, arrays)
        metadata = {
            "schema": config["storage"]["result_schema"],
            "case": case,
            "adapter": result.metadata,
            "execution": {} if execution is None else execution,
            "elapsed_seconds": time.perf_counter() - started,
            "array_names": sorted(arrays),
            "observables_sha256": sha256_file(observables),
        }
        _atomic_json(temporary / "metadata.json", metadata)
        (temporary / config["storage"]["success_marker"]).touch()
        os.replace(temporary, final_dir)
        return {
            "case_id": case["case_id"],
            "status": "complete",
            "metadata_sha256": sha256_file(final_dir / "metadata.json"),
            "observables_sha256": sha256_file(final_dir / "observables.npz"),
        }
    except Exception as exc:
        failure = {
            "case_id": case["case_id"],
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(temporary / "failure.json", failure)
        (temporary / config["storage"]["failure_marker"]).touch()
        os.replace(temporary, final_dir)
        raise


def _parse_cpu_range(specification: str) -> list[int]:
    cpus: list[int] = []
    for token in str(specification).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            fields = token.split("-")
            if len(fields) != 2:
                raise ValueError(f"invalid CPU range token {token!r}")
            start, end = (int(value) for value in fields)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range token {token!r}")
            cpus.extend(range(start, end + 1))
        else:
            value = int(token)
            if value < 0:
                raise ValueError("CPU IDs must be nonnegative")
            cpus.append(value)
    cpus = list(dict.fromkeys(cpus))
    if not cpus:
        raise ValueError("CPU range resolved to an empty set")
    return cpus


def _set_thread_policy(threads: int) -> None:
    value = str(int(threads))
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = value
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(threads))
    except ImportError:
        pass


def _run_one_pinned(
    case: dict[str, Any],
    config: dict[str, Any],
    cases_dir: Path,
    cpu: int,
    threads: int,
) -> dict[str, Any]:
    _set_thread_policy(threads)
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("this campaign requires Linux CPU-affinity support")
    os.sched_setaffinity(0, {int(cpu)})
    actual_affinity = sorted(int(value) for value in os.sched_getaffinity(0))
    if actual_affinity != [int(cpu)]:
        raise RuntimeError(f"failed to pin worker to CPU {cpu}: {actual_affinity}")
    execution = {
        "pid": os.getpid(),
        "cpu": int(cpu),
        "affinity": actual_affinity,
        "blas_threads": int(threads),
    }
    return _run_one(case, config, cases_dir, execution=execution)


def _initialize_worker(cpu_queue: Any, threads: int) -> None:
    global _WORKER_CPU, _WORKER_THREADS
    _WORKER_CPU = int(cpu_queue.get())
    _WORKER_THREADS = int(threads)
    _set_thread_policy(_WORKER_THREADS)
    os.sched_setaffinity(0, {_WORKER_CPU})


def _run_one_worker(
    case: dict[str, Any], config: dict[str, Any], cases_dir: Path
) -> dict[str, Any]:
    if _WORKER_CPU is None or _WORKER_THREADS is None:
        raise RuntimeError("campaign worker was not initialized with a CPU")
    actual_affinity = sorted(int(value) for value in os.sched_getaffinity(0))
    if actual_affinity != [_WORKER_CPU]:
        raise RuntimeError(
            f"worker affinity drifted from CPU {_WORKER_CPU}: {actual_affinity}"
        )
    return _run_one(
        case,
        config,
        cases_dir,
        execution={
            "pid": os.getpid(),
            "cpu": _WORKER_CPU,
            "affinity": actual_affinity,
            "blas_threads": _WORKER_THREADS,
        },
    )


def _smoke_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a tiny six-case matrix that exercises every production branch."""

    nx, ny = 4, 6
    seed = int(config["dynamics"]["markov_channel"]["root_seed"])
    cases: list[dict[str, Any]] = []
    for dw_truncation in (True, False):
        for family, dephasing_values in (
            ("markov_channel", (True,)),
            ("lindblad", (False, True)),
        ):
            for dephasing in dephasing_values:
                case_id = (
                    f"smoke_N{nx}x{ny}_nsh1_dwtrunc{int(dw_truncation)}"
                    f"_{family}_deph{int(dephasing)}_pc1"
                )
                cases.append(
                    {
                        "schema": "matched_markov_lindblad_case_spec_v1",
                        "case_id": case_id,
                        "campaign_roles": ["smoke", "matched_endpoint_size_scan"],
                        "model": {
                            "Nx": nx,
                            "Ny": ny,
                            "domain_wall": True,
                            "wall_locations": [1, 2],
                            "alpha_run_in": 1.0,
                            "alpha_run_out": 30.0,
                            "trial_orbitals": "X",
                            "nshell": 1,
                            "dw_truncation": dw_truncation,
                        },
                        "dynamics": {
                            "family": family,
                            "dephasing": dephasing,
                            "perfect_correction": True,
                            "init_mode": "maxmix",
                            "cycles": 2 * ny,
                            "physical_time": 2.0 * ny,
                            "site_schedule": "random",
                            "root_seed": seed,
                            "matched_seed": seed,
                            "schedule_match_key": f"smoke_N{nx}x{ny}:random-site-word",
                            "match_key": f"smoke_N{nx}x{ny}_nsh1",
                            "sample_ids": [0],
                            "sample_seeds": [seed],
                        },
                    }
                )
    return sorted(cases, key=lambda row: row["case_id"])


def _validate_adapter_readiness(cases: list[dict[str, Any]]) -> None:
    families = {case["dynamics"]["family"] for case in cases}
    missing = []
    if "markov_channel" in families and not MARKOV_ADAPTER_READY:
        missing.append("markov_channel")
    if "lindblad" in families and not LINDBLAD_ADAPTER_READY:
        missing.append("lindblad")
    if missing:
        raise RuntimeError(f"production adapters are not integrated: {missing}")


def _validate_complete_case(
    case: dict[str, Any],
    case_dir: Path,
    config: dict[str, Any],
    receipt: dict[str, Any] | None,
) -> dict[str, Any]:
    success = case_dir / config["storage"]["success_marker"]
    failure = case_dir / config["storage"]["failure_marker"]
    if not success.is_file() or failure.exists():
        raise RuntimeError(f"existing case is not a completed immutable result: {case_dir}")
    metadata_path = case_dir / "metadata.json"
    observables_path = case_dir / "observables.npz"
    if not metadata_path.is_file() or not observables_path.is_file():
        raise RuntimeError(f"completed case is missing required files: {case_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("case") != case:
        raise RuntimeError(f"existing case specification differs for {case['case_id']}")
    if metadata.get("schema") != config["storage"]["result_schema"]:
        raise RuntimeError(f"existing case schema differs for {case['case_id']}")
    metadata_hash = sha256_file(metadata_path)
    observables_hash = sha256_file(observables_path)
    if metadata.get("observables_sha256") != observables_hash:
        raise RuntimeError(f"observables hash mismatch for {case['case_id']}")
    reconstructed = {
        "case_id": case["case_id"],
        "status": "complete",
        "metadata_sha256": metadata_hash,
        "observables_sha256": observables_hash,
    }
    if receipt is not None:
        for key, value in reconstructed.items():
            if receipt.get(key) != value:
                raise RuntimeError(f"manifest receipt mismatch for {case['case_id']}: {key}")
    return reconstructed


def _new_run_root(config: dict[str, Any], config_hash: str, requested: Path | None) -> Path:
    if requested is not None:
        root = requested.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = HERE / "results" / f"{config['campaign']}_{stamp}_{config_hash[:12]}"
    if root.exists():
        raise FileExistsError(f"run root already exists: {root}")
    return root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cpu-range")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    config, config_hash = load_config(args.config)
    cases = _smoke_cases(config) if args.smoke else expand_cases(config)
    if args.smoke and args.case_id:
        raise SystemExit("--smoke and --case-id are mutually exclusive")
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
        missing = wanted.difference(case["case_id"] for case in cases)
        if missing:
            raise SystemExit(f"unknown case IDs: {sorted(missing)}")
    if args.list_cases:
        print("\n".join(case["case_id"] for case in cases))
        return 0
    if args.validate_only:
        counts: dict[str, int] = {}
        for case in cases:
            key = f"{case['dynamics']['family']}:deph{int(case['dynamics']['dephasing'])}"
            counts[key] = counts.get(key, 0) + 1
        print(json.dumps({"config_sha256": config_hash, "cases": len(cases), "arms": counts}, indent=2))
        return 0

    try:
        _validate_adapter_readiness(cases)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    workers = int(
        config["execution"]["default_workers"] if args.workers is None else args.workers
    )
    if workers <= 0:
        raise SystemExit("--workers must be positive")
    cpu_specification = (
        config["execution"]["default_cpu_range"]
        if args.cpu_range is None
        else args.cpu_range
    )
    try:
        cpus = _parse_cpu_range(cpu_specification)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid --cpu-range: {exc}") from exc
    if workers > len(cpus):
        raise SystemExit("--workers cannot exceed the number of CPUs in --cpu-range")
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("this campaign requires Linux CPU-affinity support")
    available = set(int(value) for value in os.sched_getaffinity(0))
    unavailable = sorted(set(cpus).difference(available))
    if unavailable:
        raise SystemExit(f"requested CPUs are outside the current process affinity: {unavailable}")
    threads = int(config["execution"]["blas_threads_per_worker"])
    if threads <= 0:
        raise SystemExit("blas_threads_per_worker must be positive")
    _set_thread_policy(threads)

    expected_ids = [case["case_id"] for case in cases]
    source_hashes = _source_hashes(args.config.resolve())
    if args.resume:
        if args.run_root is None:
            raise SystemExit("--resume requires --run-root")
        run_root = args.run_root.resolve()
        if not run_root.is_dir():
            raise SystemExit(f"resume root does not exist: {run_root}")
        if (run_root / "_FAILED").exists():
            raise SystemExit("refusing to resume a run with a terminal _FAILED marker")
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit("resume root has no manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != config_hash:
            raise SystemExit("resume config hash differs from the existing manifest")
        if manifest.get("expected_case_ids") != expected_ids:
            raise SystemExit("resume case set/order differs from the existing manifest")
        if manifest.get("source_hashes") != source_hashes:
            raise SystemExit("resume source hashes differ from the existing manifest")
        copied_config = run_root / "campaign_config.json"
        if not copied_config.is_file() or sha256_file(copied_config) != config_hash:
            raise SystemExit("resume root does not contain the immutable campaign config")
        cases_dir = run_root / "cases"
        if not cases_dir.is_dir():
            raise SystemExit("resume root has no cases directory")
        receipt_by_id = {
            row["case_id"]: row for row in manifest.get("cases", [])
        }
        completed: dict[str, dict[str, Any]] = {}
        pending_cases = []
        for case in cases:
            case_dir = cases_dir / case["case_id"]
            if case_dir.exists():
                completed[case["case_id"]] = _validate_complete_case(
                    case, case_dir, config, receipt_by_id.get(case["case_id"])
                )
            else:
                if case["case_id"] in receipt_by_id:
                    raise SystemExit(
                        f"manifest records a missing case directory: {case['case_id']}"
                    )
                pending_cases.append(case)
        if (run_root / "_SUCCESS").exists():
            if pending_cases:
                raise SystemExit("_SUCCESS run is missing expected case directories")
            print(str(run_root))
            return 0
        manifest["status"] = "running"
        manifest.setdefault("resumed_utc", []).append(datetime.now(timezone.utc).isoformat())
        manifest.setdefault("resume_execution", []).append(
            {
                "workers": workers,
                "active_workers": min(workers, len(pending_cases)),
                "cpu_range": cpu_specification,
                "resolved_cpus": cpus,
                "blas_threads_per_worker": threads,
            }
        )
        manifest["cases"] = [completed[key] for key in expected_ids if key in completed]
        _atomic_json(run_root / "manifest.json", manifest)
    else:
        run_root = _new_run_root(config, config_hash, args.run_root)
        cases_dir = run_root / "cases"
        cases_dir.mkdir(parents=True)
        shutil.copy2(args.config, run_root / "campaign_config.json")
        completed = {}
        pending_cases = list(cases)
        manifest = {
            "schema": "matched_markov_lindblad_manifest_v1",
            "status": "running",
            "campaign": config["campaign"],
            "smoke": bool(args.smoke),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": config_hash,
            "source_hashes": source_hashes,
            "git": _git_state(),
            "host": {"platform": platform.platform(), "python": sys.version},
            "execution": {
                "workers": workers,
                "active_workers": min(workers, len(pending_cases)),
                "cpu_range": cpu_specification,
                "resolved_cpus": cpus,
                "blas_threads_per_worker": threads,
            },
            "expected_case_ids": expected_ids,
            "cases": [],
        }
        _atomic_json(run_root / "manifest.json", manifest)

    def record(receipt: dict[str, Any]) -> None:
        completed[receipt["case_id"]] = receipt
        manifest["cases"] = [completed[key] for key in expected_ids if key in completed]
        _atomic_json(run_root / "manifest.json", manifest)
        print(f"complete {receipt['case_id']}", flush=True)

    try:
        failures: list[tuple[str, Exception]] = []
        if workers == 1:
            for case in pending_cases:
                try:
                    record(_run_one_pinned(case, config, cases_dir, cpus[0], threads))
                except Exception as exc:
                    failures.append((case["case_id"], exc))
                    break
        elif pending_cases:
            worker_count = min(workers, len(pending_cases))
            process_context = multiprocessing.get_context("spawn")
            cpu_queue = process_context.Queue()
            for cpu in cpus[:worker_count]:
                cpu_queue.put(cpu)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=process_context,
                initializer=_initialize_worker,
                initargs=(cpu_queue, threads),
            ) as executor:
                futures = {
                    executor.submit(
                        _run_one_worker,
                        case,
                        config,
                        cases_dir,
                    ): case["case_id"]
                    for case in pending_cases
                }
                for future in concurrent.futures.as_completed(futures):
                    case_id = futures[future]
                    try:
                        record(future.result())
                    except Exception as exc:
                        failures.append((case_id, exc))
            cpu_queue.close()
        if failures:
            case_id, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} case(s) failed; first failure {case_id}: {error}"
            ) from error
        if set(completed) != set(expected_ids):
            missing = sorted(set(expected_ids).difference(completed))
            raise RuntimeError(f"run ended without all expected case receipts: {missing}")
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["interrupted_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(run_root / "manifest.json", manifest)
        raise
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(run_root / "manifest.json", manifest)
        (run_root / "_FAILED").touch()
        raise
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(run_root / "manifest.json", manifest)
    (run_root / "_SUCCESS").touch()
    print(str(run_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
