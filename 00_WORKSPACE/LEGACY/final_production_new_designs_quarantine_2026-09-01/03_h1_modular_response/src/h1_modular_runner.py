"""Immutable A100 runner for the standalone H1 modular-response campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu
except ModuleNotFoundError:
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

from h1_io import (
    json_ready, sha256_file, sha256_json, write_json_atomic,
)
from h1_modular_observables import H1ModularObserver, checkpoint_cycles
from h1_record import OrderedBornRecordWriter


BUNDLE = "03_h1_modular_response"
REVISION = "production_25sample_h1_modular_response_v1"
AUDIT = "f805295a2a22a4ee5b746b1996538d785aaeb546c94ded5d6c4361971c447440"
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
SHARD_SIZE = 5


def _contract_audit(config: dict[str, Any]) -> str:
    payload = {
        key: config[key]
        for key in (
            "locked_contract", "wall_protocols", "modular_observer", "analysis",
            "raw_products", "retired_products",
        )
    }
    return sha256_json(payload)


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    config = json.loads(
        (Path(bundle_root) / "production_config.json").read_text(encoding="utf-8")
    )
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("bundle") != BUNDLE or config.get("sampling_revision") != REVISION:
        raise ValueError("H1 bundle identity or revision changed")
    if config.get("audit_sha256") != AUDIT or _contract_audit(config) != AUDIT:
        raise ValueError("H1 locked contract audit hash changed")
    locked = config.get("locked_contract", {})
    expected = {
        "samples": 25,
        "sample_shard_size": 5,
        "Nx": 20,
        "Ny": 40,
        "cycles": 80,
        "checkpoints": [40, 48, 56, 64, 72, 80],
        "alpha_1_values": [1.0, 3.0],
        "alpha_2": 30.0,
        "nshell": 1,
        "filling_fraction": 0.5,
        "initial_state": "haar_random_half_filled_slater",
        "perfect_correction": True,
        "born_sampling": True,
        "sequence": "random",
        "ancilla_occupation": 0.5,
        "physical_burn_in_cycles": 0,
        "dtype": "complex128",
        "canonical_entry_point": ENTRY_POINT,
    }
    if locked != expected:
        raise ValueError("H1 locked hyperparameters differ from the approved plan")
    if checkpoint_cycles(40) != locked["checkpoints"]:
        raise ValueError("H1 inclusive checkpoint helper changed")
    if config.get("wall_protocols") != {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }:
        raise ValueError("hard/soft H1 wall definitions changed")
    if "static_susceptibility" not in config.get("retired_products", []):
        raise ValueError("static susceptibility must remain explicitly retired")


def expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    locked = config["locked_contract"]
    cases: list[dict[str, Any]] = []
    for alpha_1 in locked["alpha_1_values"]:
        for protocol in ("hard", "soft"):
            flags = config["wall_protocols"][protocol]
            alpha_tag = str(int(alpha_1)) if float(alpha_1).is_integer() else str(alpha_1).replace(".", "p")
            cases.append({
                "case_id": f"H1_N20x40_{protocol}_a1-{alpha_tag}",
                "campaign": "H1_MODULAR_RESPONSE",
                "kind": "stochastic",
                "protocol": protocol,
                "model": {
                    "Nx": 20,
                    "Ny": 40,
                    "DW": True,
                    "nshell": 1,
                    "filling_frac": 0.5,
                    "alpha_1": float(alpha_1),
                    "alpha_2": 30.0,
                    "trial_orbitals": "X",
                    "dw_truncation": bool(flags["dw_truncation"]),
                    "device": "cuda:0",
                    "dtype": "complex128",
                    "backend": "local",
                    "init_mode": "default",
                },
                "run": {
                    "cycles": 80,
                    "samples": 25,
                    "sequence": "random",
                    "perfect_correction": True,
                    "postselect": False,
                    "postselect_probability": 0.0,
                    "n_a": 0.5,
                    "meas_slab_only": bool(flags["meas_slab_only"]),
                },
                "observer": copy.deepcopy(config["modular_observer"]),
            })
    if len(cases) != 4 or len({case["case_id"] for case in cases}) != 4:
        raise AssertionError("H1 expansion must contain exactly four unique cases")
    return cases


def shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _require_a100(*, smoke: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if smoke:
            return {"device": "cpu", "smoke_override": True}
        raise RuntimeError("H1 production and preflight require an A100 GPU")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if "A100" not in name.upper() and not smoke:
        raise RuntimeError(f"H1 production requires an A100; detected {name!r}")
    return {
        "device": name,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_fraction": float(free_bytes / total_bytes),
        "smoke_override": bool(smoke),
    }


def _archive_paths(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, drive_root: Path, mode: str,
) -> tuple[Path, Path, str, dict[str, Any]]:
    engine_hash = sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py")
    run_config = {
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "canonical_engine_sha256": engine_hash,
        "shard_index": int(shard_index),
        "case": case,
    }
    run_id = f"{BUNDLE}_{sha256_json(run_config)[:16]}"
    scratch_base = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    scratch = scratch_base / "classA_final_production" / BUNDLE / run_id
    collection = config[
        "production_output_collection" if mode != "pilot" else "pilot_output_collection"
    ]
    archive = (
        drive_root.resolve() / collection / config["output_bundle"] / f"{run_id}.tar.gz"
    )
    return scratch, archive, run_id, run_config


def _verify_existing(archive: Path) -> dict[str, Any] | None:
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if not archive.exists() and not receipt_path.exists():
        return None
    if not archive.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete H1 archive/receipt pair: {archive}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != archive.name:
        raise RuntimeError("H1 receipt names another archive")
    if receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError("existing H1 archive failed checksum verification")
    return receipt


def _archive(scratch: Path, archive: Path, run_id: str) -> dict[str, Any]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_base = archive.parent / f".{run_id}.{os.getpid()}"
    temporary = Path(shutil.make_archive(str(temporary_base), "gztar", root_dir=scratch))
    os.replace(temporary, archive)
    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "created_unix": time.time(),
    }
    write_json_atomic(archive.with_suffix(archive.suffix + ".receipt.json"), receipt)
    return receipt


def _source_hashes(src_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(src_dir.glob("*.py"))}


def _rng_payload() -> dict[str, np.ndarray]:
    payload = {
        "torch_cpu_rng_state": torch.get_rng_state().cpu().numpy(),
        "numpy_state_json": np.asarray(json.dumps(json_ready(np.random.get_state()))),
    }
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            payload[f"torch_cuda_rng_state_{index}"] = state.cpu().numpy()
    return payload


def run_case(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, drive_root: Path, mode: str, archive_result: bool = True,
) -> dict[str, Any]:
    if not 0 <= int(shard_index) < 5:
        raise IndexError("H1 shard index must lie in 0..4")
    global_ids = list(
        range(int(shard_index) * SHARD_SIZE, (int(shard_index) + 1) * SHARD_SIZE)
    )
    scratch, archive, run_id, run_config = _archive_paths(
        bundle_root=bundle_root, config=config, case=case,
        shard_index=shard_index, drive_root=drive_root, mode=mode,
    )
    if archive_result:
        existing = _verify_existing(archive)
        if existing is not None:
            return {"status": "already_archived", "receipt": existing}
    smoke = mode == "smoke"
    gpu = _require_a100(smoke=smoke)
    seed = shard_seed(int(config["root_seed"]), case["case_id"], shard_index)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = dict(case["model"])
    init_mode = model_config.pop("init_mode")
    if smoke:
        model_config["device"] = "cpu"
    model = classA_U1FGTN_gpu(**model_config)
    observer_config = case["observer"]
    response_times = np.arange(
        observer_config["response_time_start"],
        observer_config["response_time_stop"] + 0.5 * observer_config["response_time_step"],
        observer_config["response_time_step"],
    )
    packet_times = np.arange(
        observer_config["packet_time_start"],
        observer_config["packet_time_stop"] + 0.5 * observer_config["packet_time_step"],
        observer_config["packet_time_step"],
    )
    observer = H1ModularObserver(
        nx=20,
        ny=40,
        checkpoints=case["run"]["cycles"] and config["locked_contract"]["checkpoints"],
        global_sample_ids=global_ids,
        root_seed=int(config["root_seed"]),
        source_count=int(observer_config["sources_per_wall_checkpoint"]),
        response_times=response_times,
        packet_times=packet_times,
        spectral_clip_eps=observer_config["spectral_clip_eps"],
        wall_half_widths=observer_config["wall_half_widths"],
        primary_epsilon=float(observer_config["primary_spectral_clip_eps"]),
        primary_wall_half_width=int(observer_config["primary_wall_half_width"]),
        fit_window=tuple(observer_config["retarded_fit_window"]),
        minimum_packet_retention=float(observer_config["minimum_packet_retention"]),
    )
    sequence_info = model._sequence_helper(
        "random", meas_slab_only=bool(case["run"]["meas_slab_only"])
    )
    expected_site_ids = [
        int(x + model.Nx * y) for x, y in sequence_info["coords_for_len"]
    ]
    record = OrderedBornRecordWriter(
        samples=SHARD_SIZE,
        cycles=80,
        sites_per_cycle=len(expected_site_ids),
        expected_site_ids=expected_site_ids,
        buffer_device=model.device,
    )

    if scratch.exists():
        shutil.rmtree(scratch)
    shard_root = scratch / "shards" / f"shard_{int(shard_index):03d}"
    shard_root.mkdir(parents=True)
    with (shard_root / "rng_before.npz").open("wb") as handle:
        np.savez_compressed(handle, **_rng_payload())
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    began = time.perf_counter()
    result = model.run_markov_circuit(
        cycles=80,
        samples=SHARD_SIZE,
        batch_size=SHARD_SIZE,
        init_mode=init_mode,
        sequence="random",
        perfect_correction=True,
        postselect=False,
        postselect_probability=0.0,
        n_a=0.5,
        meas_slab_only=bool(case["run"]["meas_slab_only"]),
        G_history=False,
        save=False,
        progress=True,
        return_data=False,
        state_representation="auto",
        native_cycle_observer=observer,
        record_observer=record,
        require_no_covariance_materialization=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - began
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("H1 random-pure run did not use the occupied-frame representation")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("H1 unexpectedly materialized a full covariance")
    with (shard_root / "rng_after.npz").open("wb") as handle:
        np.savez_compressed(handle, **_rng_payload())

    product = observer.save(shard_root / "h1_modular", config=run_config)
    record_product = record.save(shard_root / "ordered_born_record.npz")
    manifest = {
        "schema_version": 1,
        "status": "complete_local",
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "canonical_entry_point": ENTRY_POINT,
        "canonical_engine_sha256": run_config["canonical_engine_sha256"],
        "run_config": run_config,
        "run_config_hash": sha256_json(run_config),
        "root_seed": int(config["root_seed"]),
        "case_id": case["case_id"],
        "protocol": case["protocol"],
        "alpha_1": case["model"]["alpha_1"],
        "shard_index": int(shard_index),
        "global_sample_indices": global_ids,
        "shard_generator_seed": seed,
        "gpu_preflight": gpu,
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(model.device)) if torch.cuda.is_available() else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(model.device)) if torch.cuda.is_available() else 0,
        "products": {
            "h1_modular": product,
            "ordered_born_record": record_product,
        },
        "retired_products_absent": config["retired_products"],
        "source_hashes": _source_hashes(bundle_root / "src"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "created_unix": time.time(),
    }
    write_json_atomic(scratch / "manifest.json", manifest)
    if not archive_result:
        return {"status": "preflight_complete", "manifest": manifest, "scratch": str(scratch)}
    receipt = _archive(scratch, archive, run_id)
    return {"status": "archived_to_drive", "receipt": receipt, "manifest": manifest}


def _preflight_path(drive_root: Path, config: dict[str, Any]) -> Path:
    return (
        drive_root.resolve() / config["production_output_collection"]
        / config["output_bundle"] / "a100_preflight.json"
    )


def a100_preflight(
    *, bundle_root: Path, config: dict[str, Any], drive_root: Path,
) -> dict[str, Any]:
    case = next(
        row for row in expand_cases(config)
        if row["protocol"] == "soft" and row["model"]["alpha_1"] == 1.0
    )
    failure = None
    try:
        result = run_case(
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=0,
            drive_root=drive_root,
            mode="production",
            archive_result=False,
        )
        manifest = result["manifest"]
        total = int(manifest["gpu_preflight"]["total_bytes"])
        peak = int(manifest["gpu_peak_reserved_bytes"])
        shard_bytes = int(sum(row["bytes"] for row in manifest["products"].values()))
        projected_bytes = 20 * shard_bytes
        safe = (
            peak <= 0.8 * total
            and shard_bytes <= 1 * 1024**3
            and projected_bytes <= 12 * 1024**3
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        manifest = None
        total = int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0
        peak = int(torch.cuda.max_memory_reserved(0)) if torch.cuda.is_available() else 0
        shard_bytes = 0
        projected_bytes = 0
        safe = False
        failure = f"{type(exc).__name__}: {exc}"
    payload = {
        "schema": "h1_a100_full_five_trajectory_shard_preflight_v1",
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "case_id": case["case_id"],
        "measured_trajectories": SHARD_SIZE,
        "measured_cycles": 80,
        "measured_checkpoints": [40, 48, 56, 64, 72, 80],
        "peak_reserved_bytes": peak,
        "gpu_total_bytes": total,
        "peak_reserved_fraction": float(peak / total) if total else None,
        "measured_shard_product_bytes": shard_bytes,
        "projected_twenty_shard_bytes": projected_bytes,
        "safe": bool(safe),
        "safety_rule": "peak<=0.8*A100, shard<=1GiB, projected outputs<=12GiB",
        "contract_changed": False,
        "failure": failure,
        "created_unix": time.time(),
    }
    path = _preflight_path(drive_root, config)
    write_json_atomic(path, payload)
    payload["receipt_path"] = str(path)
    return payload


def require_safe_preflight(
    *, drive_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    path = _preflight_path(drive_root, config)
    if not path.is_file():
        raise RuntimeError(f"H1 production locked until --a100-preflight passes: missing {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "h1_a100_full_five_trajectory_shard_preflight_v1",
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "safe": True,
        "contract_changed": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"A100 H1 preflight is unsafe, incomplete, or stale: {mismatch}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("production", "pilot", "smoke"), default="production")
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--list-cases-json", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--a100-preflight", action="store_true")
    parser.add_argument("--pilot-width", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle_root = args.bundle_root.resolve()
    config = load_config(bundle_root)
    cases = expand_cases(config)
    if args.list_cases_json:
        print(json.dumps([
            {"case_id": row["case_id"], "shard_count": 5, "case": row}
            for row in cases
        ]))
        return 0
    if args.list_cases:
        print("\n".join(row["case_id"] for row in cases))
        return 0
    if args.a100_preflight:
        try:
            current = require_safe_preflight(
                drive_root=args.drive_root, config=config
            )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            current = None
        if current is not None:
            current = dict(current)
            current["status"] = "reused_current_safe_receipt"
            current["receipt_path"] = str(_preflight_path(args.drive_root, config))
            print(json.dumps(current, indent=2, sort_keys=True))
            return 0
        result = a100_preflight(
            bundle_root=bundle_root, config=config, drive_root=args.drive_root
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result.get("safe", False):
            raise RuntimeError("H1 A100 qualification completed but was not safe")
        return 0
    selected = {row["case_id"]: row for row in cases}
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in selected:
        raise KeyError(f"unknown H1 case {case_id!r}")
    case = selected[case_id]
    print(json.dumps({
        "bundle": BUNDLE,
        "case_id": case_id,
        "shard_index": args.shard_index,
        "global_sample_indices": list(
            range(args.shard_index * 5, args.shard_index * 5 + 5)
        ),
        "model": case["model"],
        "run": case["run"],
        "observer": case["observer"],
    }, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        _require_a100(smoke=args.mode == "smoke")
        return 0
    if args.mode == "production":
        require_safe_preflight(drive_root=args.drive_root, config=config)
    result = run_case(
        bundle_root=bundle_root,
        config=config,
        case=case,
        shard_index=args.shard_index,
        drive_root=args.drive_root,
        mode=args.mode,
        archive_result=True,
    )
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
