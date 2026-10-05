"""Immutable runner for the raw programmable-wall window campaign."""

from __future__ import annotations

import argparse
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
from wall_window_observables import WallWindowObserver, checkpoint_cycles


BUNDLE = "02_wall_cft_windows"
REVISION = "production_25sample_wall_windows_v1"
AUDIT = "a4a81421626ef933a5ff1e8dc82e3c9fd8678ce36fbfbb9e226bf5f43da59f70"
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
SHARD_SIZE = 5


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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    config = json.loads((Path(bundle_root) / "production_config.json").read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("bundle") != BUNDLE or config.get("sampling_revision") != REVISION:
        raise ValueError("wall-window bundle identity or revision changed")
    if config.get("audit_sha256") != AUDIT:
        raise ValueError("wall-window contract audit hash changed")
    locked = config.get("locked_contract", {})
    expected = {
        "samples": 25, "sample_shard_size": 5, "Nx": 20,
        "Ny_values": [20, 30, 40, 50, 60], "cycles_rule": "2*Ny",
        "checkpoint_rule": "Ny + nearest_integer(j*Ny/6), j=1..6",
        "alpha_1": 1.0, "alpha_2": 30.0, "nshell": 1,
        "filling_fraction": 0.5,
        "initial_state": "haar_random_half_filled_slater",
        "perfect_correction": True, "born_sampling": True, "sequence": "random",
        "ancilla_occupation": 0.5, "physical_burn_in_cycles": 0,
        "dtype": "complex128", "canonical_entry_point": ENTRY_POINT,
    }
    if locked != expected:
        raise ValueError("wall-window locked contract differs from the approved plan")
    if config.get("wall_protocols") != {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }:
        raise ValueError("wall protocol definitions changed")
    if config.get("raw_products") != [
        "periodic_window_occupation_spectrum",
        "periodic_window_entropy_contour",
        "periodic_window_charge_variance_contour",
        "x_resolved_square_correlator",
    ]:
        raise ValueError("declared raw product set changed")


def expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    cases = []
    for ny in config["locked_contract"]["Ny_values"]:
        for protocol in ("hard", "soft"):
            flags = config["wall_protocols"][protocol]
            cases.append({
                "case_id": f"WALL_N20x{int(ny)}_{protocol}",
                "campaign": "W1_G2_raw_windows",
                "kind": "stochastic",
                "protocol": protocol,
                "model": {
                    "Nx": 20, "Ny": int(ny), "DW": True, "nshell": 1,
                    "filling_frac": 0.5, "alpha_1": 1.0, "alpha_2": 30.0,
                    "trial_orbitals": "X", "dw_truncation": bool(flags["dw_truncation"]),
                    "device": "cuda:0", "dtype": "complex128", "backend": "local",
                    "init_mode": "default",
                },
                "run": {
                    "cycles": 2 * int(ny), "samples": 25, "sequence": "random",
                    "perfect_correction": True, "postselect": False,
                    "postselect_probability": 0.0, "n_a": 0.5,
                    "meas_slab_only": bool(flags["meas_slab_only"]),
                },
                "observer": {
                    "checkpoints": checkpoint_cycles(int(ny)),
                    "window_heights": list(range(int(ny) // 2 + 1)),
                    "periodic_y_origins": list(range(int(ny))),
                    "square_correlator_ry": list(range(1, int(ny) // 2 + 1)),
                },
            })
    if len(cases) != 10 or len({row["case_id"] for row in cases}) != 10:
        raise AssertionError("wall-window expansion must contain exactly ten unique cases")
    return cases


def _require_a100(*, smoke: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if smoke:
            return {"device": "cpu", "smoke_override": True}
        raise RuntimeError("wall-window production and preflight require an A100 GPU")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if "A100" not in name.upper() and not smoke:
        raise RuntimeError(f"wall-window production requires an A100; detected {name!r}")
    return {
        "device": name, "free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
        "free_fraction": float(free_bytes / total_bytes), "smoke_override": bool(smoke),
    }


def _archive_paths(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, drive_root: Path, mode: str,
) -> tuple[Path, Path, str, dict[str, Any]]:
    engine_hash = sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py")
    run_config = {
        "sampling_revision": REVISION, "audit_sha256": AUDIT,
        "canonical_engine_sha256": engine_hash,
        "shard_index": int(shard_index), "case": case,
    }
    run_id = f"{BUNDLE}_{sha256_json(run_config)[:16]}"
    scratch_base = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    scratch = scratch_base / "classA_final_production" / BUNDLE / run_id
    collection = config["production_output_collection" if mode != "pilot" else "pilot_output_collection"]
    archive = drive_root.resolve() / collection / config["output_bundle"] / f"{run_id}.tar.gz"
    return scratch, archive, run_id, run_config


def _verify_existing(archive: Path) -> dict[str, Any] | None:
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if not archive.exists() and not receipt_path.exists():
        return None
    if not archive.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete archive/receipt pair: {archive}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("archive") != archive.name or receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError("existing wall-window archive failed receipt verification")
    return receipt


def _archive(scratch: Path, archive: Path, run_id: str) -> dict[str, Any]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_base = archive.parent / f".{run_id}.{os.getpid()}"
    temporary = Path(shutil.make_archive(str(temporary_base), "gztar", root_dir=scratch))
    os.replace(temporary, archive)
    receipt = {
        "schema_version": 1, "run_id": run_id, "archive": archive.name,
        "archive_sha256": sha256_file(archive), "archive_bytes": archive.stat().st_size,
        "created_unix": time.time(),
    }
    write_json_atomic(archive.with_suffix(archive.suffix + ".receipt.json"), receipt)
    return receipt


def _source_hashes(src_dir: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(src_dir.glob("*.py"))}


def run_case(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any], shard_index: int,
    drive_root: Path, mode: str, archive_result: bool = True,
) -> dict[str, Any]:
    if not 0 <= int(shard_index) < 5:
        raise IndexError("shard index must lie in 0..4")
    global_ids = list(range(int(shard_index) * SHARD_SIZE, (int(shard_index) + 1) * SHARD_SIZE))
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
    observer = WallWindowObserver(
        nx=case["model"]["Nx"], ny=case["model"]["Ny"],
        checkpoints=case["observer"]["checkpoints"], global_sample_ids=global_ids,
    )
    if scratch.exists():
        shutil.rmtree(scratch)
    shard_root = scratch / "shards" / f"shard_{int(shard_index):03d}"
    shard_root.mkdir(parents=True)
    with (shard_root / "rng_before.npz").open("wb") as handle:
        np.savez_compressed(
            handle, torch_cpu=torch.get_rng_state().cpu().numpy(),
            numpy_state_json=np.asarray(json.dumps(json_ready(np.random.get_state()))),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    began = time.perf_counter()
    result = model.run_markov_circuit(
        cycles=case["run"]["cycles"], samples=SHARD_SIZE, batch_size=SHARD_SIZE,
        init_mode=init_mode, sequence=case["run"]["sequence"],
        perfect_correction=True, postselect=False, postselect_probability=0.0,
        n_a=case["run"]["n_a"], meas_slab_only=case["run"]["meas_slab_only"],
        G_history=False, save=False, progress=True, return_data=False,
        state_representation="auto", native_cycle_observer=observer,
        require_no_covariance_materialization=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - began
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("wall-window random pure run did not resolve to occupied frames")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("wall-window run unexpectedly materialized a covariance")
    product = observer.save(shard_root / "wall_windows", config=run_config)
    manifest = {
        "schema_version": 1, "status": "complete_local", "bundle": BUNDLE,
        "sampling_revision": REVISION, "audit_sha256": AUDIT,
        "canonical_entry_point": ENTRY_POINT,
        "canonical_engine_sha256": run_config["canonical_engine_sha256"],
        "run_config": run_config, "run_config_hash": sha256_json(run_config),
        "root_seed": int(config["root_seed"]), "case_id": case["case_id"],
        "protocol": case["protocol"], "shard_index": int(shard_index),
        "global_sample_indices": global_ids, "shard_generator_seed": seed,
        "gpu_preflight": gpu, "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(model.device)) if torch.cuda.is_available() else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(model.device)) if torch.cuda.is_available() else 0,
        "products": {"wall_windows": product},
        "retired_products_absent": [
            "covariance", "eigenvectors", "chern", "bott", "tangent", "event_records",
            "derived_entropy", "derived_charge_cumulants", "fits", "figures",
        ],
        "source_hashes": _source_hashes(bundle_root / "src"),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__},
        "created_unix": time.time(),
    }
    write_json_atomic(scratch / "manifest.json", manifest)
    if not archive_result:
        return {"status": "preflight_complete", "manifest": manifest, "scratch": str(scratch)}
    receipt = _archive(scratch, archive, run_id)
    return {"status": "archived_to_drive", "receipt": receipt, "manifest": manifest}


def _preflight_path(drive_root: Path, config: dict[str, Any]) -> Path:
    return drive_root.resolve() / config["production_output_collection"] / config["output_bundle"] / "a100_preflight.json"


def a100_preflight(*, bundle_root: Path, config: dict[str, Any], drive_root: Path) -> dict[str, Any]:
    cases = {row["case_id"]: row for row in expand_cases(config)}
    rows = []
    safe = True
    failure = None
    for protocol in ("hard", "soft"):
        case = cases[f"WALL_N20x60_{protocol}"]
        try:
            result = run_case(
                bundle_root=bundle_root, config=config, case=case, shard_index=0,
                drive_root=drive_root, mode="production", archive_result=False,
            )
            manifest = result["manifest"]
            total = int(manifest["gpu_preflight"]["total_bytes"])
            peak = int(manifest["gpu_peak_reserved_bytes"])
            product_bytes = int(manifest["products"]["wall_windows"]["bytes"])
            row_safe = peak <= 0.8 * total and product_bytes <= 8 * 1024**3
            rows.append({
                "case_id": case["case_id"], "elapsed_seconds": manifest["elapsed_seconds"],
                "peak_allocated_bytes": manifest["gpu_peak_allocated_bytes"],
                "peak_reserved_bytes": peak, "gpu_total_bytes": total,
                "peak_reserved_fraction": peak / total, "product_bytes": product_bytes,
                "safe": row_safe,
            })
            safe = safe and row_safe
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            safe = False
            failure = f"{type(exc).__name__}: {exc}"
            rows.append({"case_id": case["case_id"], "safe": False, "failure": failure})
            break
    payload = {
        "schema": "wall_windows_a100_ny60_both_protocols_v1", "bundle": BUNDLE,
        "sampling_revision": REVISION, "audit_sha256": AUDIT,
        "measured_trajectories_per_protocol": SHARD_SIZE,
        "measured_cycles": 120, "maximum_window_height": 30,
        "protocols": rows, "safe": bool(safe),
        "safety_rule": "each peak_reserved <= 0.8*A100 memory and product <= 8 GiB",
        "contract_changed": False, "failure": failure, "created_unix": time.time(),
    }
    path = _preflight_path(drive_root, config)
    write_json_atomic(path, payload)
    payload["receipt_path"] = str(path)
    return payload


def require_safe_preflight(*, drive_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _preflight_path(drive_root, config)
    if not path.is_file():
        raise RuntimeError(f"production locked until both Ny=60 A100 preflights pass: missing {path}")
    payload = json.loads(path.read_text())
    expected = {
        "schema": "wall_windows_a100_ny60_both_protocols_v1", "bundle": BUNDLE,
        "sampling_revision": REVISION, "audit_sha256": AUDIT,
        "safe": True, "contract_changed": False,
    }
    mismatch = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatch or len(payload.get("protocols", [])) != 2:
        raise RuntimeError(f"A100 preflight is unsafe, incomplete, or stale: {mismatch}")
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
        print(json.dumps([{"case_id": row["case_id"], "shard_count": 5, "case": row} for row in cases]))
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
            raise RuntimeError("wall-window A100 qualification completed but was not safe")
        return 0
    selected = {row["case_id"]: row for row in cases}
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in selected:
        raise KeyError(f"unknown wall-window case {case_id!r}")
    case = selected[case_id]
    preflight = {
        "bundle": BUNDLE, "case_id": case_id, "shard_index": args.shard_index,
        "global_sample_indices": list(range(args.shard_index * 5, args.shard_index * 5 + 5)),
        "model": case["model"], "run": case["run"], "observer": case["observer"],
    }
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        _require_a100(smoke=args.mode == "smoke")
        return 0
    if args.mode == "production":
        require_safe_preflight(drive_root=args.drive_root, config=config)
    result = run_case(
        bundle_root=bundle_root, config=config, case=case, shard_index=args.shard_index,
        drive_root=args.drive_root, mode=args.mode, archive_result=True,
    )
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
