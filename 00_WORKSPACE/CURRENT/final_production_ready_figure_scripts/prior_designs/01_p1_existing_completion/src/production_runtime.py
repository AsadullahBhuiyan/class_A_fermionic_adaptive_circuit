from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
AUDIT_SHA256 = "d23d313fd8d6b11074ae0a351b8c8f9fa720cbd0866dac739ba0beca9fc1338f"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
PRODUCTION_SAMPLES = 25
SHARD_SIZE = 5


def json_ready(value: Any) -> Any:
    """Convert runtime metadata to bounded, deterministic JSON data.

    Production arrays belong in NPZ products, not manifests.  Small arrays are retained
    verbatim; unexpectedly large arrays are represented by a content hash so a canonical
    runner result cannot accidentally turn the manifest into a raw-data archive.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size <= 10_000:
            return array.tolist()
        contiguous = np.ascontiguousarray(array)
        return {
            "array_summary": True,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
        }
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, set):
        return [json_ready(item) for item in sorted(value, key=repr)]
    return repr(value)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json_atomic(path: Path | str, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def save_npz_atomic(path: Path | str, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    path = Path(bundle_root) / "production_config.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_locked_contract(config)
    return config


def validate_locked_contract(config: dict[str, Any]) -> None:
    locked = config.get("locked_contract", {})
    expected = {
        "samples": PRODUCTION_SAMPLES,
        "sample_shard_size": SHARD_SIZE,
        "cycles_rule": "2*Ny",
        "physical_burn_in_cycles": 0,
        "sequence": "random",
        "dtype": "complex128",
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
    }
    mismatches = {
        key: (locked.get(key), value)
        for key, value in expected.items()
        if locked.get(key) != value
    }
    if mismatches:
        raise ValueError(f"production_config.json violates the locked audit contract: {mismatches}")
    if config.get("audit_sha256") != AUDIT_SHA256:
        raise ValueError("production_config.json was generated against a different audit PDF hash")


def deterministic_sample_seeds(root_seed: int, count: int = PRODUCTION_SAMPLES) -> list[int]:
    sequence = np.random.SeedSequence(int(root_seed))
    return [int(child.generate_state(1, dtype=np.uint64)[0]) for child in sequence.spawn(int(count))]


def shard_table(sample_seeds: Iterable[int], shard_size: int = SHARD_SIZE) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in sample_seeds]
    if not seeds or len(seeds) % int(shard_size):
        raise ValueError("sample list must be nonempty and divisible into fixed five-trajectory shards")
    rows = []
    for shard_index, start in enumerate(range(0, len(seeds), int(shard_size))):
        stop = start + int(shard_size)
        rows.append(
            {
                "shard_index": shard_index,
                "sample_start": start,
                "sample_stop": stop,
                "sample_indices": list(range(start, stop)),
                "sample_seeds": seeds[start:stop],
            }
        )
    return rows


def environment_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        )
    except Exception as exc:
        payload["torch_import_error"] = repr(exc)
    return payload


def require_a100(*, smoke: bool = False, minimum_free_fraction: float = 0.2) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        if smoke:
            return {"device": "cpu", "smoke_override": True}
        raise RuntimeError("Production mode requires a CUDA A100 runtime")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_fraction = float(free_bytes) / float(total_bytes)
    if "A100" not in name.upper() and not smoke:
        raise RuntimeError(f"Production mode requires an A100; detected {name!r}")
    if free_fraction < float(minimum_free_fraction):
        raise RuntimeError(
            f"Only {free_fraction:.1%} GPU memory is free; the audit requires at least "
            f"{float(minimum_free_fraction):.0%} headroom"
        )
    return {
        "device": name,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_fraction": free_fraction,
        "smoke_override": bool(smoke),
    }


def git_commit(path: Path | str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(path), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def source_hashes(src_dir: Path | str) -> dict[str, str]:
    src_dir = Path(src_dir)
    return {
        str(path.relative_to(src_dir)): sha256_file(path)
        for path in sorted(src_dir.rglob("*.py"))
    }


@dataclass(frozen=True)
class RunPaths:
    bundle_root: Path
    scratch_root: Path
    drive_output_root: Path
    run_id: str

    @property
    def run_root(self) -> Path:
        return self.scratch_root / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "manifest.json"


def make_run_paths(
    *,
    bundle_root: Path | str,
    bundle_name: str,
    run_config: dict[str, Any],
    drive_root: Path | str,
    output_collection: str = "classA_final_production_outputs",
) -> RunPaths:
    bundle_root = Path(bundle_root).resolve()
    config_hash = sha256_json(run_config)[:16]
    run_id = f"{bundle_name}_{config_hash}"
    scratch_base = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    scratch_root = scratch_base / "classA_final_production" / bundle_name
    output_collection = str(output_collection).strip()
    if output_collection not in (
        "classA_final_production_outputs",
        "classA_pilot_outputs",
    ):
        raise ValueError(f"unsupported output collection {output_collection!r}")
    drive_output_root = (
        Path(drive_root).resolve() / output_collection / bundle_name
    )
    return RunPaths(bundle_root, scratch_root, drive_output_root, run_id)


def initialize_run_directory(paths: RunPaths, manifest: dict[str, Any]) -> None:
    paths.run_root.mkdir(parents=True, exist_ok=True)
    for name in ("shards", "merged", "analysis", "figures", "logs"):
        (paths.run_root / name).mkdir(exist_ok=True)
    if paths.manifest_path.exists():
        with paths.manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("run_config_hash") != manifest.get("run_config_hash"):
            raise RuntimeError("existing scratch run has a different immutable configuration")
    else:
        write_json_atomic(paths.manifest_path, manifest)


def archive_run_to_drive(paths: RunPaths) -> dict[str, Any]:
    paths.drive_output_root.mkdir(parents=True, exist_ok=True)
    archive_base = paths.drive_output_root / paths.run_id
    tmp_base = paths.drive_output_root / f".{paths.run_id}.{os.getpid()}"
    tmp_archive = Path(shutil.make_archive(str(tmp_base), "gztar", root_dir=paths.run_root))
    final_archive = archive_base.with_suffix(".tar.gz")
    os.replace(tmp_archive, final_archive)
    checksum = sha256_file(final_archive)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_id": paths.run_id,
        "archive": final_archive.name,
        "archive_sha256": checksum,
        "archive_bytes": final_archive.stat().st_size,
        "created_unix": time.time(),
    }
    write_json_atomic(final_archive.with_suffix(final_archive.suffix + ".receipt.json"), receipt)
    return receipt


def verify_archive_receipt(archive: Path | str) -> dict[str, Any]:
    """Load and verify the completion receipt adjacent to one Drive archive."""
    archive = Path(archive)
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if not archive.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete archive/receipt pair for {archive}")
    with receipt_path.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("archive") != archive.name:
        raise RuntimeError(f"archive receipt names a different file for {archive}")
    if receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError(f"archive checksum mismatch for {archive}")
    return receipt


def existing_archive_receipt(paths: RunPaths) -> dict[str, Any] | None:
    """Return a verified completion receipt, or None when the run has not been archived."""
    archive = (paths.drive_output_root / paths.run_id).with_suffix(".tar.gz")
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if not archive.exists() and not receipt_path.exists():
        return None
    return verify_archive_receipt(archive)


def base_manifest(
    *,
    bundle_root: Path | str,
    bundle_name: str,
    run_config: dict[str, Any],
    root_seed: int,
) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    declared_samples = int(
        run_config.get("case", {}).get("run", {}).get("samples", PRODUCTION_SAMPLES)
    )
    if declared_samples <= 0:
        raise ValueError("the declared run sample count must be positive")
    sample_labels = list(range(declared_samples))
    allocation = shard_table(sample_labels)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "initialized",
        "bundle": bundle_name,
        "audit_sha256": AUDIT_SHA256,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "run_config": run_config,
        "run_config_hash": sha256_json(run_config),
        "root_seed": int(root_seed),
        "global_sample_labels": sample_labels,
        "planned_shard_allocation": [
            {key: value for key, value in row.items() if key != "sample_seeds"}
            for row in allocation
        ],
        "rng_contract": (
            "stateful_torch_stream_per_immutable_five_trajectory_shard;"
            "exact_pre_post_states_saved;repartitioning_changes_ensemble"
        ),
        "source_hashes": source_hashes(bundle_root / "src"),
        "environment": environment_manifest(),
        "git_commit": git_commit(bundle_root),
        "created_unix": time.time(),
    }


def archive_compact_stage(
    paths: RunPaths,
    *,
    stage: str,
    case_id: str,
    shard_index: int,
    product_path: Path | str,
) -> dict[str, Any]:
    """Atomically checkpoint one compact descendant product to Drive.

    This deliberately accepts only a regular file.  It is used between fused H1/H2
    stages so a later failure never forces the already-computed compact observable to
    be repeated.  Covariance arrays remain live scratch state and cannot enter this API.
    """
    product_path = Path(product_path)
    if not product_path.is_file():
        raise FileNotFoundError(product_path)
    if "covariance" in product_path.name.lower():
        raise ValueError("compact stage checkpoints must not archive covariance files")
    stage = str(stage).strip().replace("/", "_")
    target_dir = (
        paths.drive_output_root / "stage_checkpoints" / paths.run_id / str(case_id)
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"shard_{int(shard_index):03d}_{stage}{product_path.suffix}"
    temporary = target_dir / f".{target.name}.{os.getpid()}.tmp"
    shutil.copy2(product_path, temporary)
    os.replace(temporary, target)
    receipt = {
        "schema": "compact_stage_checkpoint_v1",
        "stage": stage,
        "case_id": str(case_id),
        "shard_index": int(shard_index),
        "path": str(target),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
        "permanent_covariance_bytes": 0,
    }
    write_json_atomic(target.with_suffix(target.suffix + ".receipt.json"), receipt)
    return receipt
