from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu
except ModuleNotFoundError:  # Repository tests import the canonical source package.
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu
from p1_chern_observables import P1ChernObserver


BUNDLE = "01_p1_chern_dynamics"
SAMPLING_REVISION = "production_25sample_p1_chern_v1"
SHARD_SIZE = 5
CONTRACT_AUDIT_SHA256 = (
    "9a905aac643e0ad5c1fca2d07038791905b4409dd5fcab0563d1dfec4a29391d"
)
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"


def json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return repr(value)


def write_json_atomic(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    path = Path(bundle_root) / "production_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    expected_locked = {
        "samples": 25,
        "sample_shard_size": 5,
        "cycles_rule": "L",
        "physical_burn_in_cycles": 0,
        "sequence": "random",
        "dtype": "complex128",
        "canonical_entry_point": ENTRY_POINT,
    }
    if config.get("bundle") != BUNDLE:
        raise ValueError("P1 bundle name is not immutable")
    if config.get("sampling_revision") != SAMPLING_REVISION:
        raise ValueError("P1 sampling revision is not immutable")
    if config.get("audit_sha256") != CONTRACT_AUDIT_SHA256:
        raise ValueError("P1 contract audit hash differs from the approved redesign")
    if config.get("locked_contract") != expected_locked:
        raise ValueError("P1 locked contract differs from the approved redesign")
    campaign = config.get("P1", {})
    expected = {
        "sizes": [16, 24, 32, 64],
        "nshell_values": [1, 2, None],
        "alpha_1": 1.0,
        "alpha_2": 1.0,
        "initial_state": "random_half_filled_slater",
        "ancilla_occupation": 0.5,
        "center_count": 10,
        "center_sampling": "fresh_per_sample_cycle_shared_across_shells",
        "chern_radius_fraction": 0.4,
        "chern_sign_target": 1.0,
        "declared_observables": ["periodic_trijunction_real_space_chern"],
    }
    mismatches = {
        key: (campaign.get(key), value)
        for key, value in expected.items()
        if campaign.get(key) != value
    }
    if mismatches:
        raise ValueError(f"P1 campaign differs from the approved redesign: {mismatches}")


def expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    campaign = config["P1"]
    cases: list[dict[str, Any]] = []
    for size in campaign["sizes"]:
        for nshell in campaign["nshell_values"]:
            shell_tag = "none" if nshell is None else str(int(nshell))
            cases.append(
                {
                    "case_id": f"P1_CHERN_L{int(size)}_nsh-{shell_tag}",
                    "campaign": "P1",
                    "kind": "stochastic",
                    "model": {
                        "Nx": int(size),
                        "Ny": int(size),
                        "DW": False,
                        "nshell": nshell,
                        "filling_frac": 0.5,
                        "alpha_1": 1.0,
                        "alpha_2": 1.0,
                        "trial_orbitals": "X",
                        "dw_truncation": False,
                        "device": "cuda:0",
                        "dtype": "complex128",
                        "backend": "dense" if nshell is None else "local",
                        "init_mode": "default",
                    },
                    "run": {
                        "cycles": int(size),
                        "samples": 25,
                        "sequence": "random",
                        "perfect_correction": True,
                        "postselect": False,
                        "postselect_probability": 0.0,
                        "n_a": 0.5,
                    },
                    "observer": {
                        "center_count": 10,
                        "radius_fraction": 0.4,
                        "cycles": list(range(int(size) + 1)),
                    },
                }
            )
    if len(cases) != 12:
        raise AssertionError("P1 expansion must contain exactly twelve cases")
    return cases


def case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(case["case_id"]): case for case in cases}
    if len(result) != len(cases):
        raise ValueError("duplicate P1 case ID")
    return result


def _source_hashes(src_dir: Path) -> dict[str, str]:
    return {
        str(path.relative_to(src_dir)): sha256_file(path)
        for path in sorted(src_dir.glob("*.py"))
    }


def _require_a100(*, smoke: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if smoke:
            return {"device": "cpu", "smoke_override": True}
        raise RuntimeError("P1 production and preflight require an A100 GPU")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if "A100" not in name.upper() and not smoke:
        raise RuntimeError(f"P1 production requires an A100; detected {name!r}")
    if free_bytes / total_bytes < 0.2 and not smoke:
        raise RuntimeError("P1 requires at least 20% free A100 memory")
    return {
        "device": name,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_fraction": float(free_bytes / total_bytes),
        "smoke_override": bool(smoke),
    }


def _archive_paths(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any], shard_index: int,
    drive_root: Path, mode: str,
) -> tuple[Path, Path, str, dict[str, Any]]:
    engine_hash = sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py")
    run_config = {
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "canonical_engine_sha256": engine_hash,
        "shard_index": int(shard_index),
        "case": case,
    }
    run_id = f"{BUNDLE}_{sha256_json(run_config)[:16]}"
    scratch_base = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    scratch = scratch_base / "classA_final_production" / BUNDLE / run_id
    collection_key = "production_output_collection" if mode != "pilot" else "pilot_output_collection"
    output = drive_root.resolve() / str(config[collection_key]) / str(config.get("output_bundle", BUNDLE))
    archive = output / f"{run_id}.tar.gz"
    return scratch, archive, run_id, run_config


def _verify_existing(archive: Path) -> dict[str, Any] | None:
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if not archive.exists() and not receipt_path.exists():
        return None
    if not archive.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete archive/receipt pair: {archive}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != archive.name:
        raise RuntimeError("archive receipt names a different file")
    if receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError("archive checksum mismatch")
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


def _run_case(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any], shard_index: int,
    drive_root: Path, mode: str, archive_result: bool = True,
) -> dict[str, Any]:
    samples = int(case["run"]["samples"])
    shard_count = samples // SHARD_SIZE
    if not 0 <= int(shard_index) < shard_count:
        raise IndexError(f"shard index must lie in 0..{shard_count - 1}")
    sample_start = int(shard_index) * SHARD_SIZE
    sample_stop = sample_start + SHARD_SIZE
    global_sample_ids = list(range(sample_start, sample_stop))
    scratch, archive, run_id, run_config = _archive_paths(
        bundle_root=bundle_root, config=config, case=case, shard_index=shard_index,
        drive_root=drive_root, mode=mode,
    )
    if archive_result:
        existing = _verify_existing(archive)
        if existing is not None:
            return {"status": "already_archived", "receipt": existing}

    smoke = mode == "smoke"
    gpu = _require_a100(smoke=smoke)
    seed = shard_seed(int(config["root_seed"]), str(case["case_id"]), int(shard_index))
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model_config = dict(case["model"])
    init_mode = str(model_config.pop("init_mode"))
    if smoke:
        model_config["device"] = "cpu"
    model = classA_U1FGTN_gpu(**model_config)
    observer = P1ChernObserver(
        size=int(case["model"]["Nx"]),
        physical_cycles=int(case["run"]["cycles"]),
        global_sample_ids=global_sample_ids,
        root_seed=int(config["root_seed"]),
        center_count=int(case["observer"]["center_count"]),
        radius_fraction=float(case["observer"]["radius_fraction"]),
    )
    if scratch.exists():
        shutil.rmtree(scratch)
    (scratch / "shards" / f"shard_{int(shard_index):03d}").mkdir(parents=True)
    shard_root = scratch / "shards" / f"shard_{int(shard_index):03d}"
    before_rng = {
        "numpy_state_json": np.asarray(json.dumps(json_ready(np.random.get_state()))),
        "torch_cpu": torch.get_rng_state().cpu().numpy(),
    }
    with (shard_root / "rng_before.npz").open("wb") as handle:
        np.savez_compressed(handle, **before_rng)

    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    began = time.perf_counter()
    result = model.run_markov_circuit(
        cycles=int(case["run"]["cycles"]),
        samples=SHARD_SIZE,
        batch_size=SHARD_SIZE,
        init_mode=init_mode,
        sequence=str(case["run"]["sequence"]),
        perfect_correction=bool(case["run"]["perfect_correction"]),
        postselect=False,
        postselect_probability=0.0,
        n_a=float(case["run"]["n_a"]),
        G_history=False,
        save=False,
        progress=True,
        return_data=False,
        state_representation="auto",
        native_cycle_observer=observer,
        require_no_covariance_materialization=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - began
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("random pure P1 did not use the occupied-frame representation")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("P1 unexpectedly materialized a dense covariance")

    product = observer.save(
        shard_root / "p1_chern.npz", config=run_config
    )
    manifest = {
        "schema_version": 1,
        "status": "complete_local",
        "bundle": BUNDLE,
        "audit_sha256": config["audit_sha256"],
        "canonical_entry_point": ENTRY_POINT,
        "canonical_engine_sha256": run_config["canonical_engine_sha256"],
        "run_config": run_config,
        "run_config_hash": sha256_json(run_config),
        "root_seed": int(config["root_seed"]),
        "case_id": str(case["case_id"]),
        "shard_index": int(shard_index),
        "global_sample_indices": global_sample_ids,
        "shard_generator_seed": seed,
        "gpu_preflight": gpu,
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(model.device)) if torch.cuda.is_available() else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(model.device)) if torch.cuda.is_available() else 0,
        "products": {"p1_chern": product},
        "retired_products_absent": [
            "covariance", "bott", "density", "entropy", "tangent",
            "convergence", "ordered_record", "purity",
        ],
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
        return {"status": "preflight_complete", "manifest": manifest}
    receipt = _archive(scratch, archive, run_id)
    return {"status": "archived_to_drive", "receipt": receipt, "manifest": manifest}


def _a100_preflight(
    *, bundle_root: Path, config: dict[str, Any], drive_root: Path
) -> dict[str, Any]:
    original = next(
        case for case in expand_cases(config)
        if case["model"]["Nx"] == 64 and case["model"]["nshell"] == 1
    )
    case = copy.deepcopy(original)
    case["run"]["samples"] = SHARD_SIZE
    began = time.perf_counter()
    try:
        result = _run_case(
            bundle_root=bundle_root, config=config, case=case, shard_index=0,
            drive_root=drive_root, mode="production", archive_result=False,
        )
    except torch.cuda.OutOfMemoryError as exc:
        gpu = _require_a100(smoke=False)
        total_bytes = int(gpu["total_bytes"])
        peak_reserved = int(torch.cuda.max_memory_reserved(0))
        payload = {
            "schema": "p1_a100_full_L64_shard_preflight_v1",
            "bundle": BUNDLE,
            "sampling_revision": config["sampling_revision"],
            "audit_sha256": config["audit_sha256"],
            "case_id": original["case_id"],
            "measured_trajectories": SHARD_SIZE,
            "measured_cycles_per_trajectory": 64,
            "measured_centers_per_sample_cycle": 10,
            "elapsed_seconds_before_failure": time.perf_counter() - began,
            "peak_reserved_bytes": peak_reserved,
            "gpu_total_bytes": total_bytes,
            "peak_reserved_fraction": float(peak_reserved / total_bytes),
            "projected_seconds_for_three_L64_cases": None,
            "safe": False,
            "failure": "cuda_out_of_memory",
            "failure_detail": str(exc),
            "contract_changed": False,
            "created_unix": time.time(),
        }
        receipt = _preflight_receipt_path(drive_root=drive_root, config=config)
        write_json_atomic(receipt, payload)
        payload["receipt_path"] = str(receipt)
        return payload
    manifest = result["manifest"]
    per_shard = float(manifest["elapsed_seconds"])
    total_bytes = int(manifest["gpu_preflight"]["total_bytes"])
    peak_reserved = int(manifest["gpu_peak_reserved_bytes"])
    payload = {
        "schema": "p1_a100_full_L64_shard_preflight_v1",
        "bundle": BUNDLE,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "case_id": original["case_id"],
        "measured_trajectories": SHARD_SIZE,
        "measured_cycles_per_trajectory": 64,
        "measured_centers_per_sample_cycle": 10,
        "elapsed_seconds": per_shard,
        "peak_allocated_bytes": manifest["gpu_peak_allocated_bytes"],
        "peak_reserved_bytes": peak_reserved,
        "gpu_total_bytes": total_bytes,
        "peak_reserved_fraction": float(peak_reserved / total_bytes),
        "projected_seconds_for_three_L64_cases": 15.0 * per_shard,
        "safe": bool(peak_reserved <= 0.8 * total_bytes),
        "safety_rule": "peak_reserved_bytes <= 0.8 * gpu_total_bytes",
        "contract_changed": False,
        "created_unix": time.time(),
    }
    receipt = _preflight_receipt_path(drive_root=drive_root, config=config)
    write_json_atomic(receipt, payload)
    payload["receipt_path"] = str(receipt)
    return payload


def _preflight_receipt_path(*, drive_root: Path, config: dict[str, Any]) -> Path:
    return (
        drive_root.resolve()
        / str(config["production_output_collection"])
        / str(config.get("output_bundle", BUNDLE))
        / "a100_preflight.json"
    )


def _require_safe_preflight(*, drive_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _preflight_receipt_path(drive_root=drive_root, config=config)
    if not path.is_file():
        raise RuntimeError(
            "P1 production is locked until --a100-preflight succeeds on an A100; "
            f"missing {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "p1_a100_full_L64_shard_preflight_v1",
        "bundle": BUNDLE,
        "sampling_revision": config["sampling_revision"],
        "audit_sha256": config["audit_sha256"],
        "contract_changed": False,
        "safe": True,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"P1 A100 preflight receipt is not safe/current: {mismatches}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one lean P1 Chern-dynamics shard")
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
            {"case_id": case["case_id"], "shard_count": 5, "case": case}
            for case in cases
        ]))
        return 0
    if args.list_cases:
        for case in cases:
            print(case["case_id"])
        return 0
    if args.a100_preflight:
        print(json.dumps(_a100_preflight(
            bundle_root=bundle_root, config=config, drive_root=args.drive_root.resolve()
        ), indent=2, sort_keys=True))
        return 0
    selected = case_index(cases)
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in selected:
        raise KeyError(f"unknown P1 case {case_id!r}")
    case = selected[case_id]
    if args.mode == "production":
        _require_safe_preflight(drive_root=args.drive_root.resolve(), config=config)
    preflight = {
        "bundle": BUNDLE,
        "case_id": case_id,
        "shard_index": int(args.shard_index),
        "samples": 5,
        "global_sample_indices": list(
            range(int(args.shard_index) * 5, int(args.shard_index) * 5 + 5)
        ),
        "model": case["model"],
        "run": case["run"],
        "observer": case["observer"],
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        _require_a100(smoke=args.mode == "smoke")
        return 0
    result = _run_case(
        bundle_root=bundle_root, config=config, case=case,
        shard_index=int(args.shard_index), drive_root=args.drive_root.resolve(),
        mode=str(args.mode), archive_result=True,
    )
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
