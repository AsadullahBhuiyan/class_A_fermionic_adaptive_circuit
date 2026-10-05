from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from production_runtime import (
    archive_run_to_drive,
    base_manifest,
    initialize_run_directory,
    load_config,
    make_run_paths,
    existing_archive_receipt,
    output_collection_for_mode,
    require_a100,
    save_npz_atomic,
    sha256_file,
    write_json_atomic,
)
from record_observables import OrderedBornRecordWriter
from tangent_observables import TangentFrameWriter


CASE_ID = "V0_engine_record_replay_validation"


def validation_case() -> dict[str, Any]:
    """Return the deterministic one-job identity consumed by the shared orchestrator."""
    return {
        "case_id": CASE_ID,
        "campaign": "V0",
        "kind": "deterministic_validation",
        "model": {"Nx": 1, "Ny": 2, "device": "cuda:0", "dtype": "complex128"},
        "run": {"samples": 1, "suite_version": 2},
    }


def validation_shard_seed(root_seed: int) -> int:
    raw = f"{int(root_seed)}:{CASE_ID}:0".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _model(engine: Any, device: str) -> Any:
    return engine(
        Nx=1,
        Ny=2,
        DW=False,
        nshell=None,
        device=device,
        dtype="complex128",
        backend="dense",
    )


def _site_ids(model: Any) -> list[int]:
    info = model._sequence_helper("random", meas_slab_only=False)
    return [int(x + model.Nx * y) for x, y in info["coords_for_len"]]


def _run_recorded(torch: Any, engine: Any, device: str, seed: int) -> tuple[dict, OrderedBornRecordWriter, np.ndarray]:
    model = _model(engine, device)
    sites = _site_ids(model)
    writer = OrderedBornRecordWriter(
        samples=2,
        cycles=2,
        sites_per_cycle=len(sites),
        expected_site_ids=sites,
        buffer_device=device,
    )
    initial: list[np.ndarray] = []

    def cycle_observer(*, cycle: int, G: Any, **_: Any) -> None:
        if int(cycle) == 0:
            initial.append(G.detach().cpu().numpy().copy())

    torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        init_mode="maxmix",
        save=False,
        return_data=True,
        sequence="random",
        meas_slab_only=False,
        perfect_correction=True,
        batch_size=2,
        record_observer=writer,
        cycle_observer=cycle_observer,
    )
    writer.validate()
    return result, writer, initial[0]


def _replay(
    torch: Any,
    engine: Any,
    *,
    device: str,
    schedule: np.ndarray,
    outcomes: np.ndarray,
    initial: np.ndarray,
    writer: OrderedBornRecordWriter | None = None,
    site_observer: Any | None = None,
) -> dict[str, Any]:
    model = _model(engine, device)
    return model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=int(schedule.shape[1]),
        samples=int(schedule.shape[0]),
        init_mode="maxmix",
        G_init=initial,
        save=False,
        return_data=True,
        sequence="random",
        meas_slab_only=False,
        perfect_correction=True,
        batch_size=int(schedule.shape[0]),
        frozen_schedule=schedule,
        frozen_outcomes=outcomes,
        record_observer=writer,
        site_observer=site_observer,
    )


def _benchmark_model(engine: Any, device: str, *, smoke: bool) -> Any:
    side = 4 if smoke else 20
    return engine(
        Nx=side,
        Ny=side,
        DW=False,
        nshell=2,
        device=device,
        dtype="complex128",
        backend="local",
    )


def _sitewise_batch_benchmark(
    torch: Any, engine: Any, device: str, *, smoke: bool
) -> dict[str, Any]:
    """Compare the frame-native batch path with frozen-record covariance evolution."""

    samples = 2 if smoke else 5
    source_model = _benchmark_model(engine, device, smoke=smoke)
    sites = _site_ids(source_model)
    writer = OrderedBornRecordWriter(
        samples=samples,
        cycles=1,
        sites_per_cycle=len(sites),
        expected_site_ids=sites,
        buffer_device=device,
    )
    initial: list[np.ndarray] = []

    def capture_initial(*, cycle: int, G: Any, **_: Any) -> None:
        if int(cycle) == 0:
            initial.append(G.detach().cpu().numpy().copy())

    torch.manual_seed(2026081799)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(2026081799)
    source_model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=1,
        samples=samples,
        init_mode="default",
        save=False,
        return_data=False,
        sequence="random",
        meas_slab_only=False,
        perfect_correction=True,
        batch_size=samples,
        record_observer=writer,
        cycle_observer=capture_initial,
    )
    writer.validate()

    def replay(*, grouped_reference: bool) -> tuple[dict[str, Any], float, int, int]:
        model = _benchmark_model(engine, device, smoke=smoke)

        def no_op_site_observer(**_: Any) -> None:
            return None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = model.run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            samples=samples,
            init_mode="default",
            G_init=initial[0],
            save=False,
            return_data=True,
            sequence="random",
            meas_slab_only=False,
            perfect_correction=True,
            batch_size=samples,
            frozen_schedule=writer.site_ids,
            frozen_outcomes=writer.outcomes,
            site_observer=no_op_site_observer if grouped_reference else None,
            state_representation=(
                "covariance" if grouped_reference else "physical_frame"
            ),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
        else:
            allocated = reserved = 0
        return result, time.perf_counter() - started, allocated, reserved

    reference, reference_seconds, reference_allocated, reference_reserved = replay(
        grouped_reference=True
    )
    fast, fast_seconds, fast_allocated, fast_reserved = replay(grouped_reference=False)
    error = float(np.max(np.abs(fast["G_final"] - reference["G_final"])))
    return {
        "geometry": [int(source_model.Nx), int(source_model.Ny)],
        "samples": samples,
        "cycles": 1,
        "fast_path": fast["site_update_batching"],
        "reference_path": reference["site_update_batching"],
        "max_abs_final_covariance_error": error,
        "fast_synchronized_seconds": float(fast_seconds),
        "grouped_reference_synchronized_seconds": float(reference_seconds),
        "speedup_over_grouped_reference": (
            float(reference_seconds / fast_seconds) if fast_seconds > 0.0 else None
        ),
        "fast_peak_allocated_bytes": fast_allocated,
        "fast_peak_reserved_bytes": fast_reserved,
        "reference_peak_allocated_bytes": reference_allocated,
        "reference_peak_reserved_bytes": reference_reserved,
    }


def _occupied_frame_campaign_validation(
    torch: Any, engine: Any, device: str, *, smoke: bool
) -> dict[str, Any]:
    """Frozen-record 16x16 campaign parity and A100 resource report."""

    side = 4 if smoke else 16
    cycles = 2 if smoke else 32
    samples = 2 if smoke else 10

    def make_model() -> Any:
        return engine(
            Nx=side,
            Ny=side,
            DW=False,
            nshell=1,
            alpha_1=1.0,
            alpha_2=1.0,
            device=device,
            dtype="complex128",
            backend="local",
        )

    initializer = make_model()
    torch.manual_seed(2026082201)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(2026082201)
    initial_frame = initializer._prepare_initial_frame_batch(
        samples, init_mode="default"
    )
    initial = initial_frame.centered_covariance(
        reason="validation_shared_initial_covariance"
    ).detach()
    sites = _site_ids(initializer)
    frame_record = OrderedBornRecordWriter(
        samples=samples,
        cycles=cycles,
        sites_per_cycle=len(sites),
        expected_site_ids=sites,
        buffer_device=device,
    )
    frame_history: list[np.ndarray] = []

    def capture_frame(*, state: Any, **_: Any) -> None:
        covariance = state.centered_covariance(
            reason="validation_every_cycle_parity_checkpoint"
        )
        frame_history.append(covariance.detach().cpu().numpy().copy())

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    torch.manual_seed(2026082202)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(2026082202)
    frame_result = make_model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=samples,
        init_mode="default",
        G_init=initial,
        save=False,
        return_data=False,
        sequence="random",
        perfect_correction=True,
        batch_size=samples,
        state_representation="physical_frame",
        native_cycle_observer=capture_frame,
        record_observer=frame_record,
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        frame_peak = int(torch.cuda.max_memory_allocated())
    else:
        frame_peak = 0
    frame_seconds = time.perf_counter() - started
    frame_record.validate()

    covariance_record = OrderedBornRecordWriter(
        samples=samples,
        cycles=cycles,
        sites_per_cycle=len(sites),
        expected_site_ids=sites,
        buffer_device=device,
    )
    covariance_history: list[np.ndarray] = []

    def capture_covariance(*, state: Any, **_: Any) -> None:
        covariance_history.append(state.detach().cpu().numpy().copy())

    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    covariance_result = make_model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=samples,
        init_mode="default",
        G_init=initial,
        save=False,
        return_data=False,
        sequence="random",
        perfect_correction=True,
        batch_size=samples,
        frozen_schedule=frame_record.site_ids,
        frozen_outcomes=frame_record.outcomes,
        state_representation="covariance",
        native_cycle_observer=capture_covariance,
        record_observer=covariance_record,
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
        covariance_peak = int(torch.cuda.max_memory_allocated())
    else:
        covariance_peak = 0
    covariance_seconds = time.perf_counter() - started
    covariance_record.validate()

    cycle_errors = np.asarray(
        [
            np.max(np.abs(frame_value - covariance_value))
            for frame_value, covariance_value in zip(
                frame_history, covariance_history, strict=True
            )
        ],
        dtype=np.float64,
    )
    probability_error = float(
        np.nanmax(
            np.abs(
                frame_record.realized_probability
                - covariance_record.realized_probability
            )
        )
    )
    branch_disagreements = int(
        np.count_nonzero(frame_record.outcomes != covariance_record.outcomes)
    )
    frame_diag = frame_result["frame_diagnostics"]
    gram_residual = max(
        float(value)
        for batch in frame_diag
        for value in batch["gram_residual"]
    )
    min_rank = min(
        int(value) for batch in frame_diag for value in batch["min_ranks"]
    )
    max_rank = max(
        int(value) for batch in frame_diag for value in batch["max_ranks"]
    )
    channel_updates = samples * cycles * len(sites) * 4
    return {
        "geometry": [side, side],
        "cycles": cycles,
        "samples": samples,
        "cycle_axis": list(range(cycles + 1)),
        "max_abs_covariance_error_by_cycle": cycle_errors.tolist(),
        "max_abs_covariance_error": float(np.max(cycle_errors)),
        "max_abs_branch_probability_error": probability_error,
        "branch_disagreement_count": branch_disagreements,
        "normalized_gram_residual_max": gram_residual,
        "rank_min": min_rank,
        "rank_max": max_rank,
        "frame_seconds": float(frame_seconds),
        "covariance_seconds": float(covariance_seconds),
        "frame_speedup_over_covariance": float(covariance_seconds / frame_seconds),
        "frame_peak_allocated_bytes": frame_peak,
        "covariance_peak_allocated_bytes": covariance_peak,
        "frame_peak_memory_ratio": (
            float(frame_peak / covariance_peak) if covariance_peak else None
        ),
        "frame_channel_updates_per_second": float(channel_updates / frame_seconds),
        "covariance_channel_updates_per_second": float(
            channel_updates / covariance_seconds
        ),
        "frame_covariance_materialization_count": int(
            frame_result["covariance_materialization_count"]
        ),
        "frame_materialization_reasons": frame_result[
            "covariance_materializations"
        ],
        "covariance_explicit_override": bool(
            covariance_result["explicit_covariance_override"]
        ),
    }


def run_suite(*, smoke: bool, output_root: Path) -> dict[str, Any]:
    import torch
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu

    gpu = require_a100(smoke=smoke)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started = time.time()
    original, record, initial = _run_recorded(
        torch, classA_U1FGTN_gpu, device, seed=2026081700
    )
    schedule = record.site_ids.astype(np.int64, copy=True)
    outcomes = record.outcomes.copy()

    replay_writer = OrderedBornRecordWriter(
        samples=2,
        cycles=2,
        sites_per_cycle=schedule.shape[2],
        expected_site_ids=_site_ids(_model(classA_U1FGTN_gpu, device)),
        buffer_device=device,
    )
    replay = _replay(
        torch,
        classA_U1FGTN_gpu,
        device=device,
        schedule=schedule,
        outcomes=outcomes,
        initial=initial,
        writer=replay_writer,
    )
    replay_error = float(np.max(np.abs(replay["G_final"] - original["G_final"])))
    # Production record writers intentionally retain their complete buffers on the
    # accelerator until a single explicit synchronization.  Materialize before NumPy
    # consumes the replay diagnostics.
    replay_writer.validate()
    log_error = float(
        np.nanmax(
            np.abs(
                replay_writer.conditional_log_probability
                - record.conditional_log_probability
            )
        )
    )

    # Observer neutrality: the replay callback is diagnostic-only.
    without_observer = _replay(
        torch,
        classA_U1FGTN_gpu,
        device=device,
        schedule=schedule,
        outcomes=outcomes,
        initial=initial,
    )
    observer_error = float(
        np.max(np.abs(without_observer["G_final"] - replay["G_final"]))
    )

    # Exhaust all 2^(4*2)=256 one-cycle Born words and verify total branch weight.
    branch_count = 2 ** (4 * schedule.shape[2])
    branch_schedule = np.repeat(schedule[:1, :1], branch_count, axis=0)
    branch_outcomes = np.zeros(
        (branch_count, 1, schedule.shape[2], 4), dtype=np.bool_
    )
    words = np.arange(branch_count, dtype=np.uint16)
    for bit in range(4 * schedule.shape[2]):
        branch_outcomes[:, 0, bit // 4, bit % 4] = ((words >> bit) & 1).astype(bool)
    branch_writer = OrderedBornRecordWriter(
        samples=branch_count,
        cycles=1,
        sites_per_cycle=schedule.shape[2],
        expected_site_ids=_site_ids(_model(classA_U1FGTN_gpu, device)),
        buffer_device=device,
    )
    _replay(
        torch,
        classA_U1FGTN_gpu,
        device=device,
        schedule=branch_schedule,
        outcomes=branch_outcomes,
        initial=np.repeat(initial[:1], branch_count, axis=0),
        writer=branch_writer,
    )
    branch_writer.validate()
    active = np.arange(4)[None, None, None, :] < branch_writer.channel_count[..., None]
    branch_log_weight = np.sum(
        np.where(active, branch_writer.conditional_log_probability, 0.0), axis=(1, 2, 3)
    )
    branch_probability_sum = float(np.exp(branch_log_weight).sum())

    # Same initial covariance and frozen word on CPU and GPU devices.
    device_parity_error = 0.0
    if torch.cuda.is_available():
        cpu_replay = _replay(
            torch,
            classA_U1FGTN_gpu,
            device="cpu",
            schedule=schedule,
            outcomes=outcomes,
            initial=initial,
        )
        device_parity_error = float(
            np.max(np.abs(cpu_replay["G_final"] - replay["G_final"]))
        )

    tangent = TangentFrameWriter(
        samples=2,
        physical_cycles=2,
        nlayer=4,
        nvec=2,
        alignment_cycles=0,
        frame_cycles=[2],
        nx=1,
        ny=2,
    )
    tangent_model = _model(classA_U1FGTN_gpu, device)
    tangent_model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        G_init=initial,
        save=False,
        return_data=False,
        sequence="random",
        meas_slab_only=False,
        perfect_correction=True,
        batch_size=2,
        frozen_schedule=schedule,
        frozen_outcomes=outcomes,
        lyapunov_frame_observer=tangent,
        lyapunov_nvec=2,
        lyapunov_start_cycle=1,
        lyapunov_track_restricted_core=True,
    )
    tangent_diag = tangent.validate()
    frame = tangent.frames[2]
    gram = np.swapaxes(frame.conj(), -1, -2) @ frame
    tangent_orthogonality_error = float(
        np.max(np.abs(gram - np.eye(2, dtype=np.complex128)[None]))
    )
    batch_benchmark = _sitewise_batch_benchmark(
        torch, classA_U1FGTN_gpu, device, smoke=smoke
    )
    occupied_frame_campaign = _occupied_frame_campaign_validation(
        torch, classA_U1FGTN_gpu, device, smoke=smoke
    )

    checks = {
        "fresh_complete_random_permutations": bool(record.validate()["event_count"] > 0),
        "record_replay": replay_error <= 2e-10,
        "record_log_likelihood_replay": log_error <= 2e-10,
        "record_observer_diagnostic_only": observer_error == 0.0,
        "exact_one_cycle_branch_normalization": abs(branch_probability_sum - 1.0) <= 2e-9,
        "cpu_device_gpu_device_frozen_word_parity": device_parity_error <= 5e-10,
        "tangent_qr_finite_and_orthonormal": tangent_orthogonality_error <= 5e-10,
        "frame_gpu_batch_matches_covariance_reference": (
            batch_benchmark["max_abs_final_covariance_error"] <= 5e-9
            and batch_benchmark["fast_path"]
            == "padded_variable_rank_frame_v2"
            and batch_benchmark["reference_path"] == "grouped_site_reference_v1"
        ),
        "occupied_frame_campaign_every_cycle_parity": (
            occupied_frame_campaign["max_abs_covariance_error"] < 1e-8
            and occupied_frame_campaign["max_abs_branch_probability_error"] < 1e-8
            and occupied_frame_campaign["normalized_gram_residual_max"] < 1e-9
            and occupied_frame_campaign["branch_disagreement_count"] == 0
            and occupied_frame_campaign["cycle_axis"]
            == list(range(occupied_frame_campaign["cycles"] + 1))
        ),
    }
    payload = {
        "schema": "classA_A100_validation_suite_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "diagnostics": {
            "replay_max_abs_error": replay_error,
            "log_likelihood_max_abs_error": log_error,
            "observer_max_abs_error": observer_error,
            "branch_probability_sum": branch_probability_sum,
            "device_parity_max_abs_error": device_parity_error,
            "tangent_orthogonality_max_abs_error": tangent_orthogonality_error,
            "tangent": tangent_diag,
            "sitewise_batch_benchmark": batch_benchmark,
            "occupied_frame_campaign": occupied_frame_campaign,
        },
        "gpu_preflight": gpu,
        "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        "elapsed_seconds": time.time() - started,
        "permanent_covariance_bytes": 0,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    save_npz_atomic(
        output_root / "validation_compact_diagnostics.npz",
        branch_log_weight=branch_log_weight,
        tangent_qr_log_increment=tangent.qr_log_increment,
        tangent_qr_r=tangent.qr_r,
        occupied_frame_cycle=np.asarray(
            occupied_frame_campaign["cycle_axis"], dtype=np.int64
        ),
        occupied_frame_max_abs_covariance_error=np.asarray(
            occupied_frame_campaign["max_abs_covariance_error_by_cycle"],
            dtype=np.float64,
        ),
    )
    write_json_atomic(output_root / "validation_summary.json", payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"validation failed: {checks}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the A100 production validation suite")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("production", "pilot", "smoke"), default="production"
    )
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-cases-json", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.list_cases_json:
        print(
            json.dumps(
                [{"case_id": CASE_ID, "shard_count": 1, "case": validation_case()}]
            )
        )
        return 0
    if args.list_cases:
        print(CASE_ID)
        return 0
    if args.case_id not in (None, CASE_ID):
        raise KeyError(f"the validation bundle has one case: {CASE_ID}")
    if int(args.shard_index) != 0:
        raise IndexError("the deterministic validation suite has only shard 0")
    preflight = {
        "case_id": CASE_ID,
        "device": "smoke override" if args.mode == "smoke" else "A100 40GB",
        "permanent_covariance_bytes_planned": 0,
        "checks": 9,
    }
    print(json.dumps(preflight, indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    bundle_root = args.bundle_root.resolve()
    config = load_config(bundle_root)
    run_config = {
        "case_id": CASE_ID,
        "case": validation_case(),
        "mode": args.mode,
        "suite_version": 2,
        "canonical_engine_sha256": sha256_file(
            bundle_root / "src" / "classA_U1FGTN_gpu.py"
        ),
        "audit_sha256": config["audit_sha256"],
    }
    paths = make_run_paths(
        bundle_root=bundle_root,
        bundle_name="00_validation",
        run_config=run_config,
        drive_root=args.drive_root,
        output_collection=output_collection_for_mode(args.mode),
    )
    existing = existing_archive_receipt(paths)
    if existing is not None:
        print(json.dumps({"status": "already_archived", "receipt": existing}, indent=2))
        return 0
    manifest = base_manifest(
        bundle_root=bundle_root,
        bundle_name="00_validation",
        run_config=run_config,
        root_seed=2026081700,
    )
    manifest.update(
        {
            "status": "running",
            "case_id": CASE_ID,
            "shard_index": 0,
            "shard_generator_seed": validation_shard_seed(int(config["root_seed"])),
            "global_sample_indices": [0],
        }
    )
    initialize_run_directory(paths, manifest)
    payload = run_suite(smoke=args.mode == "smoke", output_root=paths.run_root / "analysis")
    manifest.update({"status": "complete_local", "validation": payload})
    write_json_atomic(paths.manifest_path, manifest)
    receipt = archive_run_to_drive(paths)
    manifest.update({"status": "archived_to_drive", "drive_receipt": receipt})
    write_json_atomic(paths.manifest_path, manifest)
    print(json.dumps({"validation": payload, "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
