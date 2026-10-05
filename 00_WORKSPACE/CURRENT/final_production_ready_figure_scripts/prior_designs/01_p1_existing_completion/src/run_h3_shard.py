from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from campaign_cases import case_index, expand_cases
from h3_twist_observables import (
    H3_SCHEMA,
    BranchWeightObserver,
    FinalEntanglementObserver,
    crossing_summary,
    initial_order,
    track_step,
)
from production_runtime import (
    archive_run_to_drive,
    base_manifest,
    initialize_run_directory,
    load_config,
    make_run_paths,
    existing_archive_receipt,
    require_a100,
    save_npz_atomic,
    sha256_file,
    verify_archive_receipt,
    write_json_atomic,
)
from record_observables import load_ordered_record


def _accepted_width(path: str | None) -> int:
    if path is None:
        raise ValueError("03_chirality_replay requires --accepted-width-json")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") not in (None, "passed", "accepted"):
        raise RuntimeError("accepted-width gate has not passed")
    value = payload.get("accepted_Nx", payload.get("accepted_width"))
    if value is None:
        raise ValueError("accepted-width gate does not contain accepted_Nx")
    value = int(value)
    if payload.get("schema_version") != 1:
        raise ValueError("H3 requires the schema-v1 joint W1/B0 gate")
    if int(payload.get("W1_candidate_Nx", -1)) != value:
        raise ValueError("accepted width does not match the W1 candidate")
    exact = payload.get("exact_B0_transverse_gate")
    if not isinstance(exact, dict) or exact.get("status") != "accepted":
        raise ValueError("accepted width does not embed the pinned accepted B0 gate")
    if int(exact.get("accepted_Nx", -1)) != value:
        raise ValueError("W1 and exact B0 width decisions disagree")
    reference_hash = payload.get("exact_B0_reference_sha256")
    if not isinstance(reference_hash, str) or len(reference_hash) != 64:
        raise ValueError("accepted width is missing the pinned B0 reference hash")
    return value


def _root_manifest_from_tar(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        candidates = [member for member in archive.getmembers() if member.name.lstrip("./") == "manifest.json"]
        if len(candidates) != 1:
            raise ValueError(f"{path}: expected one root manifest, found {len(candidates)}")
        handle = archive.extractfile(candidates[0])
        if handle is None:
            raise ValueError(f"{path}: root manifest is unreadable")
        return json.loads(handle.read().decode("utf-8"))


def find_parent_archive(
    *,
    drive_root: Path,
    case: dict[str, Any],
    shard_index: int,
    output_collection: str = "classA_final_production_outputs",
) -> tuple[Path, dict[str, Any]]:
    root = drive_root / output_collection / "02_pure_wall_master"
    matches: list[tuple[Path, dict[str, Any]]] = []
    integrity_failures: list[str] = []
    expected_case_id = (
        f"MASTER_N{int(case['Nx'])}x{int(case['Ny'])}_{case['parent_protocol']}"
    )
    for path in sorted(root.glob("*.tar.gz")):
        try:
            manifest = _root_manifest_from_tar(path)
        except Exception:
            continue
        parent_case = manifest.get("run_config", {}).get("case", {})
        model = parent_case.get("model", {})
        if (
            int(manifest.get("shard_index", -1)) == int(shard_index)
            and int(model.get("Nx", -1)) == int(case["Nx"])
            and int(model.get("Ny", -1)) == int(case["Ny"])
            and str(parent_case.get("case_id", "")) == expected_case_id
        ):
            try:
                verify_archive_receipt(path)
            except Exception as exc:
                integrity_failures.append(f"{path}: {exc}")
                continue
            matches.append((path, manifest))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one checksum-verified compact parent archive for "
            f"{case['case_id']} shard {shard_index}; found "
            f"{[str(path) for path, _ in matches]}; integrity failures={integrity_failures}"
        )
    return matches[0]


def _safe_extract(path: Path, destination: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe parent archive member {member.name!r}")
        archive.extractall(destination, filter="data")


def _restore_rng(path: Path, torch: Any) -> None:
    with np.load(path, allow_pickle=False) as data:
        torch.set_rng_state(torch.as_tensor(np.asarray(data["torch_cpu_rng_state"]), dtype=torch.uint8))
        cuda_keys = sorted(
            (key for key in data.files if key.startswith("torch_cuda_rng_state_")),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        if torch.cuda.is_available() and cuda_keys:
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(np.asarray(data[key]), dtype=torch.uint8) for key in cuda_keys]
            )


def _take_modes(array: np.ndarray, order: np.ndarray, *, vector: bool = False) -> np.ndarray:
    output = np.empty_like(array)
    for sample in range(array.shape[0]):
        output[sample] = array[sample][:, order[sample]] if vector else array[sample][order[sample]]
    return output


def _checkpoint_payload(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": np.asarray(H3_SCHEMA),
        "completed_points": np.asarray(state["completed_points"], dtype=np.int64),
        "phi": state["phi"],
        "occupation_values": state["occupation_values"],
        "entanglement_energies": state["entanglement_energies"],
        "wall_weights": state["wall_weights"],
        "branch_log_probability": state["branch_log_probability"],
        "minimum_event_probability": state["minimum_event_probability"],
        "reference_gap": state["reference_gap"],
        "overlap_matrices": state["overlap_matrices"],
        "tracking_assignments": state["tracking_assignments"],
        "subspace_min_singular_value": state["subspace_min_singular_value"],
        "point_elapsed_seconds": state["point_elapsed_seconds"],
        "point_peak_allocated_bytes": state["point_peak_allocated_bytes"],
        "point_peak_reserved_bytes": state["point_peak_reserved_bytes"],
        "first_vectors": state["first_vectors"],
        "last_vectors": state["last_vectors"],
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != H3_SCHEMA:
            raise ValueError("incompatible H3 checkpoint schema")
        return {key: np.array(data[key], copy=True) for key in data.files if key != "schema"}


def run_h3(
    *,
    bundle_root: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    shard_index: int,
    drive_root: Path,
    mode: str,
) -> dict[str, Any]:
    import torch
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu

    mode = str(mode)
    if mode not in ("production", "pilot", "smoke"):
        raise ValueError(f"unsupported run mode {mode!r}")
    smoke = mode == "smoke"
    output_collection = (
        "classA_pilot_outputs"
        if mode == "pilot"
        else "classA_final_production_outputs"
    )
    gpu = require_a100(smoke=smoke)
    parent_archive, parent_manifest = find_parent_archive(
        drive_root=drive_root,
        case=case,
        shard_index=shard_index,
        output_collection=output_collection,
    )
    parent_case = parent_manifest["run_config"]["case"]
    parent_shard_samples = len(parent_manifest["global_sample_indices"])
    requested_record_indices = case.get("record_indices")
    record_indices = np.asarray(
        list(range(parent_shard_samples))
        if requested_record_indices is None
        else requested_record_indices,
        dtype=np.int64,
    )
    if record_indices.ndim != 1 or record_indices.size == 0:
        raise ValueError("H3 record_indices must select at least one parent record")
    if len(np.unique(record_indices)) != len(record_indices):
        raise ValueError("H3 record_indices contains a duplicate")
    if np.any(record_indices < 0) or np.any(record_indices >= parent_shard_samples):
        raise IndexError(
            f"H3 record_indices must lie in 0..{parent_shard_samples - 1}"
        )
    shard_samples = int(record_indices.size)
    selected_global_sample_indices = [
        int(parent_manifest["global_sample_indices"][index])
        for index in record_indices
    ]
    points = int(case["grid_points"])
    if smoke:
        points = min(points, 5)
    phi = np.linspace(0.0, 2.0 * math.pi, points, dtype=np.float64)
    run_config = {
        "mode": mode,
        "case": case,
        "shard_index": int(shard_index),
        "parent_archive": parent_archive.name,
        "parent_archive_sha256": sha256_file(parent_archive),
        "grid_points_effective": points,
        "parent_shard_samples": int(parent_shard_samples),
        "replayed_parent_record_indices": record_indices.tolist(),
        "replayed_global_sample_indices": selected_global_sample_indices,
        "canonical_engine_sha256": sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py"),
        "audit_sha256": config["audit_sha256"],
    }
    paths = make_run_paths(
        bundle_root=bundle_root,
        bundle_name=config["bundle"],
        run_config=run_config,
        drive_root=drive_root,
        output_collection=output_collection,
    )
    checkpoint_drive = paths.drive_output_root / f"{paths.run_id}.h3_checkpoint.npz"
    existing = existing_archive_receipt(paths)
    if existing is not None:
        checkpoint_drive.unlink(missing_ok=True)
        return {"status": "already_archived", "receipt": existing, "products": {}}
    manifest = base_manifest(
        bundle_root=bundle_root,
        bundle_name=config["bundle"],
        run_config=run_config,
        root_seed=int(config["root_seed"]),
    )
    manifest.update(
        {
            "status": "running",
            "case_id": case["case_id"],
            "shard_index": int(shard_index),
            "gpu_preflight": gpu,
            "permanent_covariance_bytes_planned": 0,
            "parent_manifest_hash": parent_manifest["run_config_hash"],
            "parent_shard_samples": int(parent_shard_samples),
            "replayed_parent_record_indices": record_indices.tolist(),
            "replayed_global_sample_indices": selected_global_sample_indices,
        }
    )
    initialize_run_directory(paths, manifest)
    shard_root = paths.run_root / "shards" / f"shard_{int(shard_index):03d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    checkpoint_drive.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="classA_h3_parent_") as temporary:
        extracted = Path(temporary)
        _safe_extract(parent_archive, extracted)
        parent_shard = extracted / "shards" / f"shard_{int(shard_index):03d}"
        record_path = parent_shard / "ordered_record.npz"
        record = load_ordered_record(record_path)
        replay_site_ids = record["site_ids"][record_indices]
        replay_outcomes = record["outcomes"][record_indices]
        replay_parent_log_probability = record["total_log_probability"][record_indices]
        record_sha256 = sha256_file(record_path)
        rng_path = parent_shard / "rng_before.npz"
        model_config = dict(parent_case["model"])
        init_mode = model_config.pop("init_mode")
        meas_slab_only = bool(model_config.pop("meas_slab_only"))
        cycles = int(parent_case["run"]["cycles"])
        tracked_modes = int(config["H3"]["tracked_entanglement_modes"])

        state: dict[str, Any]
        if checkpoint_drive.exists():
            state = _load_checkpoint(checkpoint_drive)
            if not np.array_equal(state["phi"], phi):
                raise RuntimeError("existing H3 checkpoint uses a different twist grid")
            completed = int(state["completed_points"])
            # Migrate checkpoints written before point-level A100 telemetry existed.
            state.setdefault("point_elapsed_seconds", np.full((points,), np.nan))
            state.setdefault(
                "point_peak_allocated_bytes", np.zeros((points,), dtype=np.int64)
            )
            state.setdefault(
                "point_peak_reserved_bytes", np.zeros((points,), dtype=np.int64)
            )
        else:
            dim = int(case["Nx"] * (case["Ny"] // 2) * 2)
            state = {
                "completed_points": 0,
                "phi": phi,
                "occupation_values": np.full((shard_samples, points, tracked_modes), np.nan),
                "entanglement_energies": np.full((shard_samples, points, tracked_modes), np.nan),
                "wall_weights": np.full((shard_samples, points, tracked_modes, 2), np.nan),
                "branch_log_probability": np.full((shard_samples, points), np.nan),
                "minimum_event_probability": np.full((shard_samples, points), np.nan),
                "reference_gap": np.full((shard_samples, points), np.nan),
                "overlap_matrices": np.full((shard_samples, points - 1, tracked_modes, tracked_modes), np.nan + 0j),
                "tracking_assignments": np.full((shard_samples, points - 1, tracked_modes), -1, dtype=np.int16),
                "subspace_min_singular_value": np.full((shard_samples, points - 1), np.nan),
                "point_elapsed_seconds": np.full((points,), np.nan),
                "point_peak_allocated_bytes": np.zeros((points,), dtype=np.int64),
                "point_peak_reserved_bytes": np.zeros((points,), dtype=np.int64),
                "first_vectors": np.empty((shard_samples, dim, tracked_modes), dtype=np.complex128),
                "last_vectors": np.empty((shard_samples, dim, tracked_modes), dtype=np.complex128),
            }
            completed = 0

        point_bar = tqdm(
            range(completed, points),
            desc=f"H3 flux points ({case['case_id']}, shard {shard_index})",
            unit="point",
            initial=completed,
            total=points,
        )
        for point_index in point_bar:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            point_started = time.perf_counter()
            _restore_rng(rng_path, torch)
            model = classA_U1FGTN_gpu(**model_config)
            seam_y = (int(case["seam_shift"]) - 1) % int(case["Ny"])
            model.set_controller_twist(
                float(phi[point_index]), gauge=str(case["gauge"]), seam_y=seam_y
            )
            # Recreate the exact parent initialization once, then transform it when a
            # uniform-gauge representative is requested. This is required for seam and
            # uniform descriptions to be genuinely gauge equivalent recordwise.
            G_parent_init = model._prepare_initial_batch(
                batch_size=parent_shard_samples, init_mode=init_mode
            )
            G_init = G_parent_init.index_select(
                0,
                torch.as_tensor(record_indices, dtype=torch.long, device=model.device),
            )
            del G_parent_init
            if str(case["gauge"]) == "uniform":
                mode_indices = torch.arange(model.Nlayer, device=model.device)
                ycoord = mode_indices // (2 * model.Nx)
                phase = torch.exp(
                    1j * float(phi[point_index]) * ycoord.to(model.real_dtype) / float(model.Ny)
                ).to(model.dtype)
                G_init = phase[None, :, None] * G_init * phase.conj()[None, None, :]
            branch = BranchWeightObserver(samples=shard_samples, cycles=cycles)
            entanglement = FinalEntanglementObserver(
                samples=shard_samples,
                nx=int(case["Nx"]),
                ny=int(case["Ny"]),
                final_cycle=cycles,
                tracked_modes=tracked_modes,
                wall_half_width=int(config["H3"]["wall_assignment_half_width"]),
            )
            run_args = dict(parent_case["run"])
            for key in list(run_args):
                if key.startswith("lyapunov_"):
                    run_args.pop(key)
            run_args.update(
                {
                    "samples": shard_samples,
                    "batch_size": shard_samples,
                    "init_mode": init_mode,
                    "G_init": G_init,
                    "meas_slab_only": meas_slab_only,
                    "G_history": False,
                    "save": False,
                    "return_data": False,
                    "frozen_schedule": replay_site_ids,
                    "frozen_outcomes": replay_outcomes,
                    "record_observer": branch,
                    "cycle_observer": entanglement,
                }
            )
            replay_result = model.run_markov_circuit(**run_args)
            if (
                replay_result.get("site_update_batching")
                != "sitewise_rank1_full_trajectory_batch_v1"
            ):
                raise RuntimeError(
                    "H3 replay did not use the full-trajectory sitewise GPU batch"
                )
            print(
                f"[GPU batching] H3 point={point_index + 1}/{points}, "
                f"samples={shard_samples}, mode={replay_result.get('site_update_batching')}"
            )
            values, vectors, weights, reference_gap = entanglement.payload()
            if point_index == 0:
                order = initial_order(values)
                values = _take_modes(values, order)
                vectors = _take_modes(vectors, order, vector=True)
                weights = _take_modes(weights, order)
                state["first_vectors"] = vectors
            else:
                values, vectors, weights, overlap, assignment = track_step(
                    state["last_vectors"], values, vectors, weights
                )
                state["overlap_matrices"][:, point_index - 1] = overlap
                state["tracking_assignments"][:, point_index - 1] = assignment
                state["subspace_min_singular_value"][:, point_index - 1] = np.asarray(
                    [np.min(np.linalg.svd(item, compute_uv=False)) for item in overlap]
                )
            state["last_vectors"] = vectors
            state["occupation_values"][:, point_index] = values
            state["entanglement_energies"][:, point_index] = np.log(
                np.clip(1.0 - values, 1e-14, 1.0) / np.clip(values, 1e-14, 1.0)
            )
            state["wall_weights"][:, point_index] = weights
            state["branch_log_probability"][:, point_index] = branch.total_log_probability
            state["minimum_event_probability"][:, point_index] = branch.minimum_probability
            state["reference_gap"][:, point_index] = reference_gap
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                state["point_peak_allocated_bytes"][point_index] = int(
                    torch.cuda.max_memory_allocated()
                )
                state["point_peak_reserved_bytes"][point_index] = int(
                    torch.cuda.max_memory_reserved()
                )
            state["point_elapsed_seconds"][point_index] = (
                time.perf_counter() - point_started
            )
            completed_times = state["point_elapsed_seconds"][: point_index + 1]
            mean_seconds = float(np.nanmean(completed_times))
            remaining_seconds = mean_seconds * float(points - point_index - 1)
            point_bar.set_postfix(
                point_s=f"{state['point_elapsed_seconds'][point_index]:.1f}",
                eta_h=f"{remaining_seconds / 3600.0:.2f}",
                peak_GiB=f"{state['point_peak_reserved_bytes'][point_index] / 1024**3:.2f}",
            )
            print(
                f"[H3 point {point_index + 1}/{points}] "
                f"phi={phi[point_index]:.8f}, elapsed={state['point_elapsed_seconds'][point_index]:.2f}s, "
                f"mean={mean_seconds:.2f}s/point, remaining ETA={remaining_seconds / 3600.0:.3f}h"
            )
            state["completed_points"] = point_index + 1
            save_npz_atomic(checkpoint_drive, **_checkpoint_payload(state))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    crossings, ambiguous = crossing_summary(
        state["entanglement_energies"], state["wall_weights"]
    )
    closure_last_vectors = state["last_vectors"]
    if str(case["gauge"]) == "uniform":
        subsystem_y = np.repeat(np.arange(int(case["Ny"]) // 2), 2 * int(case["Nx"]))
        large_gauge = np.exp(2j * math.pi * subsystem_y / float(case["Ny"]))
        closure_last_vectors = large_gauge.conj()[None, :, None] * closure_last_vectors
    closure = np.asarray(
        [
            np.linalg.norm(
                closure_last_vectors[sample] @ closure_last_vectors[sample].conj().T
                - state["first_vectors"][sample] @ state["first_vectors"][sample].conj().T,
                ord="fro",
            )
            for sample in range(shard_samples)
        ],
        dtype=np.float64,
    )
    expected_parent_log = replay_parent_log_probability
    phi0_log_residual = np.abs(state["branch_log_probability"][:, 0] - expected_parent_log)
    measured_point_seconds = np.asarray(state["point_elapsed_seconds"], dtype=np.float64)
    measured_point_seconds = measured_point_seconds[np.isfinite(measured_point_seconds)]
    total_replay_seconds = float(np.sum(measured_point_seconds))
    mean_point_seconds = (
        float(np.mean(measured_point_seconds)) if measured_point_seconds.size else float("nan")
    )
    mean_record_point_seconds = mean_point_seconds / float(shard_samples)
    product_path = shard_root / "H3_twist_flow.npz"
    save_npz_atomic(
        product_path,
        **_checkpoint_payload(state),
        crossing_count=crossings,
        ambiguous_crossing_count=ambiguous,
        closure_projector_frobenius=closure,
        phi0_log_probability_residual=phi0_log_residual,
        parent_record_sha256=np.asarray(record_sha256),
        replayed_parent_record_indices=record_indices,
        replayed_global_sample_indices=np.asarray(
            selected_global_sample_indices, dtype=np.int64
        ),
    )
    manifest.update(
        {
            "status": "complete_local",
            "completed_unix": time.time(),
            "products": {
                "H3_twist_flow": {
                    "path": str(product_path),
                    "sha256": sha256_file(product_path),
                    "bytes": product_path.stat().st_size,
                    "permanent_covariance_bytes": 0,
                }
            },
            "acceptance_diagnostics": {
                "max_phi0_log_probability_residual": float(np.max(phi0_log_residual)),
                "max_closure_projector_frobenius": float(np.max(closure)),
                "minimum_event_probability": float(np.min(state["minimum_event_probability"])),
                "crossing_count": crossings.tolist(),
                "ambiguous_crossings": ambiguous.tolist(),
            },
            "a100_replay_telemetry": {
                "completed_twist_points": int(state["completed_points"]),
                "total_synchronized_seconds": total_replay_seconds,
                "mean_seconds_per_twist_point": mean_point_seconds,
                "mean_seconds_per_replayed_record_per_twist_point": (
                    mean_record_point_seconds
                ),
                "twist_points_per_hour": (
                    3600.0 / mean_point_seconds
                    if np.isfinite(mean_point_seconds) and mean_point_seconds > 0.0
                    else None
                ),
                "projected_65_point_hours_from_mean": (
                    65.0 * mean_point_seconds / 3600.0
                    if np.isfinite(mean_point_seconds)
                    else None
                ),
                "projected_grid_hours_from_mean": {
                    str(grid): (
                        float(grid) * mean_point_seconds / 3600.0
                        if np.isfinite(mean_point_seconds)
                        else None
                    )
                    for grid in (17, 33, 65, 129)
                },
                "projected_promotions_hours_from_linear_record_scaling": {
                    "one_record_65_points": 65.0 * mean_record_point_seconds / 3600.0,
                    "one_record_129_points": 129.0 * mean_record_point_seconds / 3600.0,
                    "five_records_65_points": 5.0 * 65.0 * mean_record_point_seconds / 3600.0,
                    "five_records_129_points": 5.0 * 129.0 * mean_record_point_seconds / 3600.0,
                    "ten_records_65_points": 10.0 * 65.0 * mean_record_point_seconds / 3600.0,
                    "ten_records_129_points": 10.0 * 129.0 * mean_record_point_seconds / 3600.0,
                },
                "peak_allocated_bytes": int(
                    np.max(state["point_peak_allocated_bytes"], initial=0)
                ),
                "peak_reserved_bytes": int(
                    np.max(state["point_peak_reserved_bytes"], initial=0)
                ),
            },
        }
    )
    write_json_atomic(paths.manifest_path, manifest)
    receipt = archive_run_to_drive(paths)
    manifest["status"] = "archived_to_drive"
    manifest["drive_receipt"] = receipt
    write_json_atomic(paths.manifest_path, manifest)
    checkpoint_drive.unlink(missing_ok=True)
    return {"manifest": str(paths.manifest_path), "receipt": receipt, "products": manifest["products"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one complete H3 frozen-record twist shard")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--mode", choices=("production", "pilot", "smoke"), default="production"
    )
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--accepted-width-json")
    parser.add_argument(
        "--pilot-width",
        type=int,
        default=20,
        help="Non-gating provisional width used only in pilot mode.",
    )
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-cases-json", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def production_cases(config: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    contract = config["H3"]["production_contract"]
    parent_shards = [int(value) for value in contract["parent_shards"]]
    if parent_shards != [0, 1]:
        raise ValueError("H3 production parent shards must remain exactly [0, 1]")
    protocols = set(str(value) for value in contract["protocols"])
    selected = [
        case
        for case in cases
        if int(case["Ny"]) == int(contract["Ny"])
        and int(case["grid_points"]) == int(contract["grid_points"])
        and str(case["gauge"]) == str(contract["gauge"])
        and int(case["seam_shift"]) == int(contract["seam_shift"])
        and str(case["parent_protocol"]) in protocols
        and case.get("record_indices") is None
    ]
    if len(selected) != len(protocols):
        raise RuntimeError(
            f"H3 production contract selected {len(selected)} cases for {len(protocols)} protocols"
        )
    return selected, parent_shards


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    src = bundle_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    config = load_config(bundle_root)
    if args.mode == "pilot" and (
        args.accepted_width_json is None
        or not Path(args.accepted_width_json).is_file()
    ):
        accepted_width = int(args.pilot_width)
    else:
        accepted_width = _accepted_width(args.accepted_width_json)
    cases = expand_cases(config, accepted_width=accepted_width)
    production_parent_shards: list[int] | None = None
    if args.mode == "production":
        cases, production_parent_shards = production_cases(config, cases)
    if args.list_cases_json:
        print(json.dumps([
            {
                "case_id": case["case_id"],
                "shard_count": (
                    len(production_parent_shards)
                    if production_parent_shards is not None
                    else int(config["locked_contract"]["samples"])
                    // int(config["locked_contract"]["sample_shard_size"])
                ),
            }
            for case in cases
        ]))
        return 0
    if args.list_cases:
        print("\n".join(case["case_id"] for case in cases))
        return 0
    indexed = case_index(cases)
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in indexed:
        raise KeyError(f"unknown H3 case {case_id!r}")
    case = indexed[case_id]
    if production_parent_shards is not None and int(args.shard_index) not in production_parent_shards:
        raise IndexError(
            f"H3 production shard must be one of {production_parent_shards}, got {args.shard_index}"
        )
    preflight = {
        "case": case,
        "shard_index": int(args.shard_index),
        "twist_replays": min(int(case["grid_points"]), 5) if args.mode == "smoke" else int(case["grid_points"]),
        "replayed_parent_record_indices": (
            "all parent records"
            if case.get("record_indices") is None
            else list(case["record_indices"])
        ),
        "replayed_records": (
            5 if case.get("record_indices") is None else len(case["record_indices"])
        ),
        "permanent_covariance_bytes": 0,
        "checkpoint": "compact after every completed twist point",
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    result = run_h3(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=args.shard_index,
        drive_root=args.drive_root.resolve(),
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
