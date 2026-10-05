from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from b1_controller_frame import (
    ControllerFrameObserver,
    b1_cases,
    construct_controller_frame,
    save_static_frame,
    validate_b1_config,
)
from production_runtime import (
    archive_run_to_drive,
    environment_manifest,
    existing_archive_receipt,
    find_compatible_legacy_archive,
    git_commit,
    initialize_run_directory,
    load_config,
    make_run_paths,
    output_collection_for_mode,
    require_a100,
    sha256_file,
    sha256_json,
    source_hashes,
    write_json_atomic,
)
from record_observables import OrderedBornRecordWriter
from drive_storage_guard import storage_status


def _load_config(bundle_root: Path) -> dict[str, Any]:
    config = load_config(bundle_root)
    validate_b1_config(config)
    return config


def _shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _save_rng(path: Path, torch: Any) -> None:
    payload = {"torch_cpu_rng_state": torch.get_rng_state().cpu().numpy()}
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            payload[f"torch_cuda_rng_state_{index}"] = state.cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)


def storage_preflight(case: dict[str, Any], *, shard_samples: int) -> dict[str, Any]:
    dimension = 2 * int(case["model"]["Nx"]) * int(case["model"]["Ny"])
    complex_bytes = np.dtype(np.complex128).itemsize
    covariance_batch = shard_samples * dimension * dimension * complex_bytes
    static_dense = 5 * dimension * dimension * complex_bytes
    sites = int(case["model"]["Nx"]) * int(case["model"]["Ny"])
    cycles = int(case["run"]["cycles"])
    record_upper = shard_samples * cycles * sites * 7
    record_device_buffer = shard_samples * cycles * sites * 113
    return {
        "Nx": int(case["model"]["Nx"]),
        "Ny": int(case["model"]["Ny"]),
        "samples_in_shard": int(shard_samples),
        "cycles": cycles,
        "active_dimension": dimension,
        "transient_covariance_batch_bytes": covariance_batch,
        "transient_static_dense_upper_bytes": static_dense,
        "compact_record_upper_bytes": record_upper,
        "record_device_buffer_bytes": record_device_buffer,
        "permanent_covariance_bytes": 0,
        "estimated_peak_bytes": covariance_batch + static_dense + record_device_buffer,
        "planning_ceiling_bytes": 12_000_000_000,
    }


def run_b1_shard(
    *,
    bundle_root: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    shard_index: int,
    drive_root: Path,
    mode: str,
    prepared_model: Any | None = None,
    prepared_frame: Any | None = None,
    gpu_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu

    mode = str(mode)
    if mode not in ("production", "pilot", "smoke"):
        raise ValueError(f"unsupported run mode {mode!r}")
    smoke = mode == "smoke"
    output_collection = output_collection_for_mode(mode)
    output_bundle = str(config.get("output_bundle", config["bundle"]))
    gpu = require_a100(smoke=smoke) if gpu_preflight is None else gpu_preflight
    total_samples = int(case["run"]["samples"])
    shard_size = 5
    shard_count = total_samples // shard_size
    if total_samples % shard_size or not (0 <= int(shard_index) < shard_count):
        raise IndexError(f"B1 shard index must be in 0..{shard_count - 1}")
    sample_start = int(shard_index) * shard_size
    sample_stop = sample_start + shard_size
    preflight = storage_preflight(case, shard_samples=shard_size)
    if preflight["estimated_peak_bytes"] > preflight["planning_ceiling_bytes"]:
        raise MemoryError("B1 transient allocation exceeds the 12 GB planning ceiling")
    seed = _shard_seed(int(config["root_seed"]), case["case_id"], int(shard_index))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runtime_source_hashes = source_hashes(bundle_root / "src")
    run_config = {
        "mode": mode,
        "case": case,
        "shard_index": int(shard_index),
        "sample_start": sample_start,
        "sample_stop": sample_stop,
        "shard_generator_seed": seed,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "canonical_engine_sha256": sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py"),
        "runtime_sources_sha256": sha256_json(runtime_source_hashes),
    }
    paths = make_run_paths(
        bundle_root=bundle_root,
        bundle_name=output_bundle,
        run_config=run_config,
        drive_root=drive_root,
        output_collection=output_collection,
    )
    existing = existing_archive_receipt(paths)
    if existing is not None:
        return {"status": "already_archived", "receipt": existing}
    if mode == "production":
        legacy = find_compatible_legacy_archive(
            drive_root=drive_root,
            bundle_name=config["bundle"],
            case=case,
            shard_index=int(shard_index),
            root_seed=int(config["root_seed"]),
            canonical_engine_sha256=run_config["canonical_engine_sha256"],
        )
        if legacy is not None:
            return {"status": "compatible_legacy_superset", "reuse": legacy}
    drive_status = storage_status(
        drive_root / output_collection,
    )
    if not drive_status["clear_to_run"]:
        raise RuntimeError(
            "Drive storage guard blocked B1 before a new shard; offload verified "
            f"archives and retry (active={drive_status['used_gb']:.3f} GB)"
        )
    manifest = {
        "schema_version": 1,
        "status": "running",
        "bundle": config["bundle"],
        "output_bundle": output_bundle,
        "sampling_revision": config["sampling_revision"],
        "case_id": case["case_id"],
        "shard_index": int(shard_index),
        "global_sample_indices": list(range(sample_start, sample_stop)),
        "run_config": run_config,
        "run_config_hash": sha256_json(run_config),
        "root_seed": int(config["root_seed"]),
        "shard_generator_seed": seed,
        "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        "audit_sha256": config["audit_sha256"],
        "rng_contract": "stateful_torch_stream_per_immutable_five_trajectory_shard",
        "source_hashes": runtime_source_hashes,
        "environment": environment_manifest(),
        "git_commit": git_commit(bundle_root),
        "gpu_preflight": gpu,
        "storage_preflight": preflight,
        "drive_storage_preflight": drive_status,
        "created_unix": time.time(),
    }
    initialize_run_directory(paths, manifest)
    # A retry after a rejected validation reuses the immutable run ID but must
    # explicitly replace the prior scratch manifest's terminal failure status.
    write_json_atomic(paths.manifest_path, manifest)
    shard_root = paths.run_root / "shards" / f"shard_{int(shard_index):03d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    _save_rng(shard_root / "rng_before.npz", torch)
    model = prepared_model if prepared_model is not None else classA_U1FGTN_gpu(**case["model"])
    frame = prepared_frame
    if frame is None:
        frame = construct_controller_frame(
            model, degeneracy_tolerance=float(config["B1"]["degeneracy_tolerance"])
        )
    static_product = save_static_frame(shard_root / "controller_frame_static.npz", frame)
    sequence = model._sequence_helper("random", meas_slab_only=False)
    expected_sites = [int(x + model.Nx * y) for x, y in sequence["coords_for_len"]]
    cycles = int(case["run"]["cycles"])
    record = OrderedBornRecordWriter(
        samples=shard_size,
        cycles=cycles,
        sites_per_cycle=len(expected_sites),
        expected_site_ids=expected_sites,
        buffer_device=model.device,
    )
    observer = ControllerFrameObserver(frame=frame, samples=shard_size, cycles=cycles)
    run_args = dict(case["run"])
    run_args.update(
        {
            "samples": shard_size,
            "batch_size": shard_size,
            "native_cycle_observer": observer,
            "state_representation": "auto",
            "require_no_covariance_materialization": True,
            "record_observer": record,
        }
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.time()
    result = model.run_markov_circuit(**run_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    elapsed = time.time() - started
    if (
        result.get("site_update_batching")
        != "padded_variable_rank_frame_v2"
    ):
        raise RuntimeError("B1 did not use the full-trajectory sitewise GPU batch")
    print(
        f"[GPU batching] case={case['case_id']}, shard={shard_index}, "
        f"samples={shard_size}, mode={result.get('site_update_batching')}"
    )
    gpu_peak_allocated = (
        int(torch.cuda.max_memory_allocated(model.device))
        if torch.cuda.is_available()
        else 0
    )
    gpu_peak_reserved = (
        int(torch.cuda.max_memory_reserved(model.device))
        if torch.cuda.is_available()
        else 0
    )
    _save_rng(shard_root / "rng_after.npz", torch)
    # Persist both compact trajectory products before scientific validation.
    # If an acceptance check fails, the rejected run is archived below instead
    # of losing a completed GPU trajectory when the child process exits.
    products = {
        "controller_frame_static": static_product,
        "ordered_record": record.save(shard_root / "ordered_record.npz"),
        "controller_observables": observer.save_raw(
            shard_root / "controller_observables.npz"
        ),
    }
    validation_settings = {
        "numerical_tolerance": float(config["B1"]["numerical_tolerance"]),
        "charge_integer_tolerance": float(
            config["B1"]["charge_integer_tolerance"]
        ),
    }
    try:
        observer_diagnostics = observer.validate(
            tolerance=validation_settings["numerical_tolerance"],
            charge_integer_tolerance=validation_settings[
                "charge_integer_tolerance"
            ],
        )
    except Exception as error:
        manifest.update(
            {
                "status": "validation_failed_local",
                "elapsed_seconds": elapsed,
                "products": products,
                "observer_validation": {
                    "accepted": False,
                    **validation_settings,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                "permanent_covariance_bytes": 0,
                "completed_unix": time.time(),
            }
        )
        write_json_atomic(paths.manifest_path, manifest)
        rejected_paths = replace(
            paths,
            drive_output_root=paths.drive_output_root / "_rejected_validation",
            run_id=f"{paths.run_id}_rejected_{time.time_ns()}",
        )
        rejected_receipt = archive_run_to_drive(rejected_paths)
        raise RuntimeError(
            "B1 observer validation failed after compact products were preserved at "
            f"{rejected_receipt['archive']}"
        ) from error
    products["controller_observables"].update(observer_diagnostics)
    manifest.update(
        {
            "status": "complete_local",
            "elapsed_seconds": elapsed,
            "trajectories_per_hour": 3600.0 * float(shard_size) / max(elapsed, 1e-12),
            "gpu_peak_allocated_bytes": gpu_peak_allocated,
            "gpu_peak_reserved_bytes": gpu_peak_reserved,
            "projected_20_shard_case_gpu_hours": 20.0 * elapsed / 3600.0,
            "products": products,
            "observer_validation": {
                "accepted": True,
                **observer_diagnostics,
            },
            "canonical_result_metadata": {
                key: value
                for key, value in result.items()
                if key not in ("G_final", "G_final_avg", "G_hist", "G_hist_avg")
            },
            "permanent_covariance_bytes": 0,
            "completed_unix": time.time(),
        }
    )
    write_json_atomic(paths.manifest_path, manifest)
    receipt = archive_run_to_drive(paths)
    return {"status": "archived_to_drive", "manifest": str(paths.manifest_path), "receipt": receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one B1 controller-frame GPU shard")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--mode", choices=("production", "pilot", "smoke"), default="production"
    )
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--shard-indices",
        type=int,
        nargs="+",
        help="Run several immutable shards in one process and reuse one frame diagonalization.",
    )
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-cases-json", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    src_dir = bundle_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    config = _load_config(bundle_root)
    cases = b1_cases(config, smoke=args.mode == "smoke")
    if args.list_cases_json:
        print(json.dumps([
            {
                "case_id": case["case_id"],
                "shard_count": int(case["run"]["samples"]) // 5,
                "case": case,
            }
            for case in cases
        ]))
        return 0
    if args.list_cases:
        for case in cases:
            print(case["case_id"])
        return 0
    indexed = {case["case_id"]: case for case in cases}
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in indexed:
        raise KeyError(f"unknown B1 case {case_id!r}; use --list-cases")
    case = indexed[case_id]
    if args.shard_indices is not None and args.shard_index is not None:
        raise ValueError("use either --shard-index or --shard-indices, not both")
    shard_indices = args.shard_indices or [0 if args.shard_index is None else args.shard_index]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError("--shard-indices contains a duplicate")
    shard_count = int(case["run"]["samples"]) // 5
    if any(index < 0 or index >= shard_count for index in shard_indices):
        raise IndexError(f"B1 shard indices must be in 0..{shard_count - 1}")
    print(json.dumps(storage_preflight(case, shard_samples=5), indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu

    gpu = require_a100(smoke=args.mode == "smoke")
    model = classA_U1FGTN_gpu(**case["model"])
    frame = construct_controller_frame(
        model, degeneracy_tolerance=float(config["B1"]["degeneracy_tolerance"])
    )
    results = []
    shard_wall_seconds: list[float] = []
    for shard_index in tqdm(
        shard_indices, desc=f"B1 shards ({case_id})", unit="shard"
    ):
        shard_started = time.perf_counter()
        print(f"[B1 start] case={case_id}, shard={shard_index}/{shard_count - 1}")
        results.append(
            run_b1_shard(
                bundle_root=bundle_root,
                config=config,
                case=case,
                shard_index=shard_index,
                drive_root=args.drive_root,
                mode=args.mode,
                prepared_model=model,
                prepared_frame=frame,
                gpu_preflight=gpu,
            )
        )
        elapsed = time.perf_counter() - shard_started
        shard_wall_seconds.append(elapsed)
        completed = len(results)
        mean_seconds = sum(shard_wall_seconds) / completed
        print(
            f"[B1 done] case={case_id}, shard={shard_index}, wall={elapsed:.2f}s, "
            f"processed={completed}/{len(shard_indices)}, rough remaining ETA="
            f"{mean_seconds * (len(shard_indices) - completed) / 3600.0:.3f}h"
        )
    print(
        json.dumps(
            {
                "case_id": case_id,
                "frame_diagonalizations": 1,
                "shard_indices": shard_indices,
                "results": results,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
