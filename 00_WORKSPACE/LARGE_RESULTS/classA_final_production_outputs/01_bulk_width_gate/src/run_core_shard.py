from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from campaign_cases import case_index, expand_cases
from production_runtime import (
    SHARD_SIZE,
    archive_run_to_drive,
    archive_compact_stage,
    base_manifest,
    load_config,
    make_run_paths,
    existing_archive_receipt,
    require_a100,
    sha256_json,
    sha256_file,
    shard_table,
    write_json_atomic,
)
from record_observables import OrderedBornRecordWriter
from tangent_observables import TangentFrameWriter
from noise_observables import OnsitePhaseNoiseWriter


def _accepted_width(path: str | None) -> int | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get("accepted_Nx", payload.get("accepted_width"))
    if value is None:
        raise ValueError(f"gate file {path} does not contain accepted_Nx")
    if payload.get("status") not in (None, "passed", "accepted"):
        raise RuntimeError(f"width gate is not passed: {payload.get('status')!r}")
    value = int(value)
    if payload.get("schema_version") != 1:
        raise ValueError("production descendants require the schema-v1 joint W1/B0 gate")
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


def _m3_wall_sigma(path: str | None) -> list[float] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") not in ("passed", "accepted"):
        raise RuntimeError("M3 bulk gate is not passed")
    values = payload.get("wall_sigma_bracket")
    if not values:
        raise ValueError("M3 bulk gate does not contain wall_sigma_bracket")
    return [float(value) for value in values]


def _validate_launch_gates(
    config: dict[str, Any], case: dict[str, Any], path: str | None
) -> dict[str, Any] | None:
    requirements = config.get("launch_gate_requirements", {}).get(case.get("campaign"), [])
    if not requirements:
        return None
    if path is None:
        raise ValueError(
            f"{case['case_id']} requires --gate-decisions-json with: {requirements}"
        )
    gate_path = Path(path)
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "accepted":
        raise RuntimeError(f"launch-gate file is not accepted: {gate_path}")
    decisions = payload.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("launch-gate file is missing its decisions object")
    failed = [name for name in requirements if decisions.get(name) is not True]
    if failed:
        raise RuntimeError(f"launch gates are not viable for {case['case_id']}: {failed}")
    return {
        "file_name": gate_path.name,
        "sha256": sha256_file(gate_path),
        "required_decisions": list(requirements),
    }


def _smoke_case(case: dict[str, Any]) -> dict[str, Any]:
    case = copy.deepcopy(case)
    if "model" not in case:
        return case
    case["case_id"] = f"SMOKE_{case['case_id']}"
    case["model"]["Nx"] = 4
    case["model"]["Ny"] = 6
    case["run"].update({"cycles": 4, "samples": 4})
    case["observation_cycles"] = [2, 4]
    case["covariance_cycles"] = [4]
    case["strip_entropy_cycles"] = []
    case["local_marker_cycles"] = [4]
    case["bott_cycles"] = []
    case.pop("h2_origin_cycles", None)
    case["inline_descendant_stages"] = []
    case["tangent_alignment_stop"] = 2
    if "lyapunov_nvec" in case["run"]:
        case["run"]["lyapunov_nvec"] = min(4, 2 * 4 * 6)
    if "onsite_phase_noise_sigma" in case["run"]:
        case["run"]["onsite_phase_noise_sigma"] = min(
            0.1, float(case["run"]["onsite_phase_noise_sigma"])
        )
    return case


def _shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _rng_payload(torch: Any) -> dict[str, np.ndarray]:
    payload = {"torch_cpu_rng_state": torch.get_rng_state().cpu().numpy()}
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            payload[f"torch_cuda_rng_state_{index}"] = state.cpu().numpy()
    return payload


def _reject_dense_choi_case_fields(case: dict[str, Any]) -> None:
    """Enforce the production policy that dense Choi tracking is never enabled."""
    stale: list[str] = []
    for key in case:
        lowered = str(key).lower()
        if "choi" in lowered or "bridge" in lowered:
            stale.append(str(key))
    run = case.get("run", {})
    if isinstance(run, dict):
        for key in run:
            lowered = str(key).lower()
            if "choi" in lowered or "bridge" in lowered:
                stale.append(f"run.{key}")
    if stale:
        raise ValueError(
            "dense Choi tracking is forbidden in production cases; remove stale fields: "
            + ", ".join(sorted(stale))
        )


def _enforce_no_dense_choi_run_args(run_args: dict[str, Any]) -> dict[str, Any]:
    forbidden = sorted(
        str(key)
        for key in run_args
        if (
            str(key).lower() == "track_choi" and bool(run_args[key])
        )
        or (
            str(key).lower() != "track_choi"
            and ("choi" in str(key).lower() or "bridge" in str(key).lower())
        )
    )
    if forbidden:
        raise ValueError(
            "dense Choi tracking is forbidden in production run arguments: "
            + ", ".join(forbidden)
        )
    run_args["track_choi"] = False
    return run_args


def _save_rng(path: Path, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **_rng_payload(torch))
    os.replace(tmp, path)


def preflight(case: dict[str, Any], *, shard_samples: int) -> dict[str, Any]:
    if "model" not in case:
        return {"case_id": case["case_id"], "kind": case["kind"], "derived_only": True}
    _reject_dense_choi_case_fields(case)
    nx = int(case["model"]["Nx"])
    ny = int(case["model"]["Ny"])
    replay_cycles = len(case.get("covariance_cycles", []))
    snapshots = 0
    transient_cycles: set[int] = set()
    if "H1" in case.get("inline_descendant_stages", []):
        transient_cycles.update((ny, 3 * ny // 2, 2 * ny))
    if "H2" in case.get("inline_descendant_stages", []):
        transient_cycles.update(int(value) for value in case.get("h2_origin_cycles", []))
    nlayer = 2 * nx * ny
    dense = shard_samples * snapshots * nlayer * nlayer * np.dtype(np.complex128).itemsize
    triangle = (
        shard_samples
        * snapshots
        * (nlayer * (nlayer + 1) // 2)
        * np.dtype(np.complex128).itemsize
    )
    # Conservative uncompressed bound: int32 site id, uint8 channel count, and one
    # packed byte each for four outcome and target bits per elementary update.
    events_upper = int(shard_samples) * int(case["run"]["cycles"]) * nx * ny
    compact_record_upper = events_upper * 7
    # The CUDA capture buffer retains unpacked per-channel diagnostics during the
    # run: 4+1+4+4+4+32+32+32 = 113 bytes per site event.
    record_device_buffer = events_upper * 113
    transient_state_bank = (
        int(shard_samples) * len(transient_cycles) * nlayer * nlayer
        * np.dtype(np.complex128).itemsize
    )
    convergence_transient = (
        int(shard_samples) * nlayer * nlayer * np.dtype(np.complex128).itemsize
    )
    convergence_archive = (
        int(shard_samples) * int(case["run"]["cycles"]) * np.dtype(np.float64).itemsize
    )
    h1_spectra_upper = 0
    if "H1" in case.get("inline_descendant_stages", []):
        h1_spectra_upper = (
            int(shard_samples) * 3 * ny * (nx * ny) * np.dtype(np.float64).itemsize
        )
    return {
        "case_id": case["case_id"],
        "Nx": nx,
        "Ny": ny,
        "cycles": int(case["run"]["cycles"]),
        "shard_samples": int(shard_samples),
        "selected_covariance_snapshots": snapshots,
        "replay_checkpoint_cycles": replay_cycles,
        "dense_covariance_bytes_if_saved": dense,
        "transient_triangle_covariance_bytes": triangle,
        "permanent_covariance_bytes_planned": 0,
        "compact_record_uncompressed_bytes_upper_bound": compact_record_upper,
        "record_device_buffer_bytes": record_device_buffer,
        "record_device_buffer_GiB": record_device_buffer / 1024**3,
        "h1_compact_spectra_bytes_upper_bound": h1_spectra_upper,
        "case_record_plus_h1_uncompressed_bytes_upper_bound": (
            compact_record_upper + h1_spectra_upper
        ) * max(1, int(case["run"].get("samples", shard_samples)) // int(shard_samples)),
        "drive_planning_ceiling_bytes": 12_000_000_000,
        "drive_absolute_edge_bytes": 14_000_000_000,
        "dense_GiB_if_saved": dense / 1024**3,
        "transient_triangle_GiB": triangle / 1024**3,
        "transient_state_bank_cycles": sorted(transient_cycles),
        "transient_state_bank_bytes": transient_state_bank,
        "transient_state_bank_GiB": transient_state_bank / 1024**3,
        "choi_tracking_enabled": False,
        "choi_bridge_transient_bytes": 0,
        "choi_bridge_transient_GiB": 0.0,
        "convergence_diagnostic": "frobenius(G_cycle-G_previous_cycle)/nlayer",
        "convergence_transient_previous_covariance_bytes": convergence_transient,
        "convergence_transient_previous_covariance_GiB": convergence_transient / 1024**3,
        "convergence_archive_bytes": convergence_archive,
        "transient_memory_planning_ceiling_bytes": 12_000_000_000,
    }


def run_case_shard(
    *,
    bundle_root: Path,
    config: dict[str, Any],
    case: dict[str, Any],
    shard_index: int,
    drive_root: Path,
    mode: str,
) -> dict[str, Any]:
    if "model" not in case:
        raise RuntimeError(
            f"{case['case_id']} is a deterministic descendant and must use its specialized replay runner"
        )
    import torch
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu
    from selected_observables import SelectedCovarianceObserver

    mode = str(mode)
    if mode not in ("production", "pilot", "smoke"):
        raise ValueError(f"unsupported run mode {mode!r}")
    smoke = mode == "smoke"
    gpu = require_a100(smoke=smoke)
    total_samples = int(case["run"]["samples"])
    if total_samples == 1:
        if int(shard_index) != 0:
            raise IndexError("a deterministic one-record case has only shard 0")
        sample_start, sample_stop = 0, 1
    else:
        allocation = shard_table(range(total_samples), SHARD_SIZE)
        if not (0 <= int(shard_index) < len(allocation)):
            raise IndexError(f"shard index must be in 0..{len(allocation)-1}")
        sample_start = allocation[int(shard_index)]["sample_start"]
        sample_stop = allocation[int(shard_index)]["sample_stop"]
    shard_samples = sample_stop - sample_start
    storage_plan = preflight(case, shard_samples=shard_samples)
    if storage_plan.get("transient_state_bank_bytes", 0) > 12_000_000_000:
        raise MemoryError(
            "the fused transient-state bank exceeds the 12 GB planning ceiling; "
            "split H1 and H2 into separate replay stages for this accepted width"
        )
    seed = _shard_seed(int(config["root_seed"]), case["case_id"], int(shard_index))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    run_config = {
        "mode": mode,
        "case": case,
        "shard_index": int(shard_index),
        "sample_start": sample_start,
        "sample_stop": sample_stop,
        "shard_generator_seed": seed,
        "canonical_engine_sha256": sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py"),
        "audit_sha256": config["audit_sha256"],
    }
    paths = make_run_paths(
        bundle_root=bundle_root,
        bundle_name=config["bundle"],
        run_config=run_config,
        drive_root=drive_root,
        output_collection=(
            "classA_pilot_outputs"
            if mode == "pilot"
            else "classA_final_production_outputs"
        ),
    )
    existing = existing_archive_receipt(paths)
    if existing is not None:
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
            "global_sample_indices": list(range(sample_start, sample_stop)),
            "shard_generator_seed": seed,
            "gpu_preflight": gpu,
            "storage_preflight": storage_plan,
        }
    )
    from production_runtime import initialize_run_directory

    initialize_run_directory(paths, manifest)
    shard_root = paths.run_root / "shards" / f"shard_{int(shard_index):03d}"
    shard_root.mkdir(parents=True, exist_ok=True)
    _save_rng(shard_root / "rng_before.npz", torch)

    model_config = dict(case["model"])
    init_mode = model_config.pop("init_mode")
    meas_slab_only = bool(model_config.pop("meas_slab_only"))
    model = classA_U1FGTN_gpu(**model_config)
    sequence_info = model._sequence_helper("random", meas_slab_only=meas_slab_only)
    expected_site_ids = [int(x + model.Nx * y) for x, y in sequence_info["coords_for_len"]]
    cycles = int(case["run"]["cycles"])

    record_writer = None
    postselect_probability = float(case["run"].get("postselect_probability", 0.0))
    if postselect_probability == 0.0:
        record_writer = OrderedBornRecordWriter(
            samples=shard_samples, cycles=cycles,
            sites_per_cycle=len(expected_site_ids), expected_site_ids=expected_site_ids,
            buffer_device=model.device,
        )
    selected = SelectedCovarianceObserver(
        nx=model.Nx,
        ny=model.Ny,
        samples=shard_samples,
        physical_cycles=cycles,
        observation_cycles=case.get("observation_cycles", [cycles]),
        covariance_cycles=(),
        strip_entropy_cycles=case.get("strip_entropy_cycles", []),
        local_marker_cycles=case.get("local_marker_cycles", []),
        bott_cycles=case.get("bott_cycles", []),
    )
    transient_bank = None
    inline_stages = list(case.get("inline_descendant_stages", []))
    cycle_observer: Any = selected
    if inline_stages:
        from fused_chirality_observables import CompositeCycleObserver, TransientStateBank

        transient_cycles: set[int] = set()
        if "H1" in inline_stages:
            transient_cycles.update((model.Ny, 3 * model.Ny // 2, 2 * model.Ny))
        if "H2" in inline_stages:
            transient_cycles.update(int(value) for value in case.get("h2_origin_cycles", []))
        transient_bank = TransientStateBank(samples=shard_samples, cycles=transient_cycles)
        cycle_observer = CompositeCycleObserver(selected, transient_bank)
    tangent = None
    if "lyapunov_nvec" in case["run"]:
        tangent_basis = model.active_top_layer_indices(
            meas_slab_only=meas_slab_only
        ).detach().cpu().numpy()
        tangent = TangentFrameWriter(
            samples=shard_samples,
            physical_cycles=cycles,
            nlayer=len(tangent_basis),
            nvec=int(case["run"]["lyapunov_nvec"]),
            alignment_cycles=int(case.get("tangent_alignment_stop", model.Ny)),
            frame_cycles=case.get("observation_cycles", [cycles]),
            nx=model.Nx,
            ny=model.Ny,
            basis_indices=tangent_basis,
        )
    noise = None
    if "onsite_phase_noise_sigma" in case["run"]:
        noise = OnsitePhaseNoiseWriter(
            samples=shard_samples,
            cycles=cycles,
            nlayer=model.Nlayer,
            sigma=float(case["run"]["onsite_phase_noise_sigma"]),
        )
    run_args = dict(case["run"])
    run_args.pop("samples", None)
    run_args["samples"] = shard_samples
    run_args["batch_size"] = shard_samples
    run_args["init_mode"] = init_mode
    run_args["meas_slab_only"] = meas_slab_only
    run_args["cycle_observer"] = cycle_observer
    run_args["record_observer"] = record_writer
    run_args["noise_observer"] = noise
    run_args = _enforce_no_dense_choi_run_args(run_args)
    if tangent is not None:
        run_args["lyapunov_frame_observer"] = tangent
        run_args["lyapunov_track_restricted_core"] = True
        run_args["lyapunov_failure_mode"] = "raise"
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.time()
    result = model.run_markov_circuit(**run_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    if bool(result.get("choi_tracked", False)):
        raise RuntimeError("canonical engine unexpectedly enabled forbidden Choi tracking")
    expects_sitewise_batch = (
        not bool(run_args.get("postselect", False))
        and float(run_args.get("postselect_probability", 0.0)) == 0.0
        and not bool(getattr(model, "triv_region_local_mode", False))
        and not bool(run_args.get("lyapunov_track_record_fisher", False))
    )
    if (
        expects_sitewise_batch
        and result.get("site_update_batching")
        != "sitewise_rank1_full_trajectory_batch_v1"
    ):
        raise RuntimeError(
            "ordinary shard did not use the required full-trajectory sitewise GPU batch"
        )
    print(
        f"[GPU batching] case={case['case_id']}, shard={shard_index}, "
        f"samples={shard_samples}, mode={result.get('site_update_batching')}, "
        f"peak_reserved_GiB={torch.cuda.max_memory_reserved(model.device) / 1024**3:.3f}"
        if torch.cuda.is_available()
        else f"[batching] mode={result.get('site_update_batching')} (CPU smoke)"
    )
    parent_elapsed = time.time() - started
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

    descendant_products: dict[str, Any] = {}
    stage_receipts: dict[str, Any] = {}
    if transient_bank is not None:
        from fused_chirality_observables import (
            compute_h1_live_products,
            compute_h2_live_products,
        )

        transient_bank.validate()
        if "H2" in inline_stages and record_writer is not None:
            # H2 forms NumPy replay slices.  The production record remained on CUDA
            # throughout the parent evolution and crosses to CPU once here.
            record_writer.materialize_cpu()
        if "H1" in inline_stages:
            descendant_products["H1_modular_transport"] = compute_h1_live_products(
                state_bank=transient_bank,
                nx=model.Nx,
                ny=model.Ny,
                cycles=(model.Ny, 3 * model.Ny // 2, 2 * model.Ny),
                config=config["H1"],
                path=shard_root / "H1_modular_transport.npz",
                device=str(model.device),
                smoke=smoke,
            )
            stage_receipts["H1"] = archive_compact_stage(
                paths,
                stage="H1_modular_transport",
                case_id=case["case_id"],
                shard_index=shard_index,
                product_path=shard_root / "H1_modular_transport.npz",
            )
        if "H2" in inline_stages:
            if not isinstance(record_writer, OrderedBornRecordWriter):
                raise RuntimeError("H2 requires the physical ordered Born record")
            descendant_products["H2_born_response"] = compute_h2_live_products(
                model=model,
                state_bank=transient_bank,
                record_writer=record_writer,
                origins=case.get("h2_origin_cycles", []),
                config=config["H2"],
                path=shard_root / "H2_born_response.npz",
                meas_slab_only=meas_slab_only,
                smoke=smoke,
            )
            stage_receipts["H2"] = archive_compact_stage(
                paths,
                stage="H2_born_response",
                case_id=case["case_id"],
                shard_index=shard_index,
                product_path=shard_root / "H2_born_response.npz",
            )
        transient_bank.clear()

    products = {
        "selected_observables": selected.save(
            shard_root / "selected_observables.npz",
            config=run_config,
            include_transient_covariances=False,
        )
    }
    if record_writer is not None:
        products["ordered_record"] = record_writer.save(shard_root / "ordered_record.npz")
    if tangent is not None:
        products["tangent"] = tangent.save(shard_root / "tangent_qr.npz")
    if noise is not None:
        products["onsite_phase_noise"] = noise.save(
            shard_root / "onsite_phase_noise.npz"
        )
    products.update(descendant_products)
    total_elapsed = time.time() - started
    manifest.update(
        {
            "status": "complete_local",
            "elapsed_seconds": total_elapsed,
            "parent_elapsed_seconds": parent_elapsed,
            "parent_trajectories_per_hour": (
                3600.0 * float(shard_samples) / max(parent_elapsed, 1e-12)
            ),
            "gpu_peak_allocated_bytes": gpu_peak_allocated,
            "gpu_peak_reserved_bytes": gpu_peak_reserved,
            "projected_20_shard_case_gpu_hours": 20.0 * total_elapsed / 3600.0,
            "projected_20_shard_parent_only_gpu_hours": 20.0 * parent_elapsed / 3600.0,
            "compact_stage_receipts": stage_receipts,
            "products": products,
            "canonical_result_metadata": {
                key: value
                for key, value in result.items()
                if key not in ("G_final", "G_final_avg", "G_hist", "G_hist_avg")
            },
            "completed_unix": time.time(),
        }
    )
    write_json_atomic(paths.manifest_path, manifest)
    receipt = archive_run_to_drive(paths)
    manifest["status"] = "archived_to_drive"
    manifest["drive_receipt"] = receipt
    write_json_atomic(paths.manifest_path, manifest)
    return {"manifest": str(paths.manifest_path), "receipt": receipt, "products": products}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one immutable production trajectory shard")
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
        help="Non-gating provisional width used only in pilot mode when no accepted-width JSON exists.",
    )
    parser.add_argument("--m3-bulk-gate-json")
    parser.add_argument("--gate-decisions-json")
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
    config = load_config(bundle_root)
    if args.mode == "pilot" and (
        args.accepted_width_json is None
        or not Path(args.accepted_width_json).is_file()
    ):
        accepted = int(args.pilot_width)
    else:
        accepted = _accepted_width(args.accepted_width_json)
    if args.mode == "smoke" and accepted is None:
        accepted = 4
    if args.mode == "pilot" and accepted is None:
        accepted = int(args.pilot_width)
        if accepted <= 0:
            raise ValueError("--pilot-width must be positive")
    cases = expand_cases(
        config,
        accepted_width=accepted,
        m3_wall_sigma=_m3_wall_sigma(args.m3_bulk_gate_json),
    )
    if args.list_cases_json:
        print(json.dumps([
            {
                "case_id": case["case_id"],
                "shard_count": 1
                if int(case.get("run", {}).get("samples", 1)) == 1
                else len(shard_table(range(int(case["run"]["samples"])), SHARD_SIZE)),
            }
            for case in cases
        ]))
        return 0
    if args.list_cases:
        for case in cases:
            print(case["case_id"])
        return 0
    selected_cases = case_index(cases)
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in selected_cases:
        raise KeyError(f"unknown case {case_id!r}; use --list-cases")
    case = selected_cases[case_id]
    launch_gates = None
    if args.mode == "production":
        launch_gates = _validate_launch_gates(config, case, args.gate_decisions_json)
    if args.mode == "smoke":
        case = _smoke_case(case)
    elif args.mode == "pilot":
        case = copy.deepcopy(case)
        case["pilot_contract"] = {
            "status": "non_gating_provisional_width",
            "provisional_Nx": accepted,
            "pool_with_production": False,
        }
    if launch_gates is not None:
        case = copy.deepcopy(case)
        case["launch_gate_receipt"] = launch_gates
    shard_samples = min(SHARD_SIZE, int(case.get("run", {}).get("samples", 1)))
    print(json.dumps(preflight(case, shard_samples=shard_samples), indent=2, sort_keys=True))
    if args.preflight_only:
        return 0
    result = run_case_shard(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=args.shard_index,
        drive_root=args.drive_root,
        mode=args.mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
