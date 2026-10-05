"""Immutable A100 runner for the standalone H1-v3 endpoint-packet campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
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
except ModuleNotFoundError:
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu

from h1_io import json_ready, sha256_file, sha256_json, write_json_atomic
from h1_packet_observables import H1EndpointPacketObserver, checkpoint_cycles
from h1_record import OrderedBornRecordWriter, load_ordered_record
from drive_remote_commit import (
    DEFAULT_DRIVE_ROOT,
    DriveRemoteCommitter,
    RemoteCommitError,
    publish_json,
    read_remote_json,
)


BUNDLE = "08_h1_endpoint_packet"
REVISION = "production_25sample_h1_endpoint_packet_v4"
AUDIT = "03b198496c93449ebc8a087e7bca1a7dccdb6b928caf923b136176f746dce687"
A100_PREFLIGHT_SCHEMA = "h1_endpoint_packet_server_verified_a100_preflight_v4"
FAILED_QUALIFICATION_DIRECTORY = "_failed_qualifications"
NUMERICAL_STATUSES = frozenset(("pass", "warning", "hard_failure"))
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
SHARD_SIZE = 5
H1_REMOTE_REQUIRED_HEADROOM_BYTES = 1_073_741_824


def _server_commit_required(path: Path | str) -> bool:
    try:
        Path(path).resolve().relative_to(DEFAULT_DRIVE_ROOT.resolve())
    except ValueError:
        return False
    return True


def _remote_committer(path: Path | str) -> DriveRemoteCommitter:
    if not _server_commit_required(path):
        raise RuntimeError(f"not a production Google Drive path: {path}")
    return DriveRemoteCommitter(drive_root=DEFAULT_DRIVE_ROOT)


def _contract_audit(config: dict[str, Any]) -> str:
    keys = (
        "root_seed",
        "output_bundle",
        "production_output_collection",
        "pilot_output_collection",
        "strict_source_hash_resume",
        "drive_commit",
        "v3_compatibility",
        "supersedes_for_interpretation",
        "locked_contract",
        "wall_protocols",
        "trajectory_reuse",
        "packet_observer",
        "analysis",
        "acceptance_gates",
        "raw_products",
        "retired_products",
    )
    return sha256_json({key: config[key] for key in keys})


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    config = json.loads(
        (Path(bundle_root) / "production_config.json").read_text(encoding="utf-8")
    )
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("bundle") != BUNDLE or config.get("sampling_revision") != REVISION:
        raise ValueError("H1-v4 bundle identity or revision changed")
    if config.get("audit_sha256") != AUDIT or _contract_audit(config) != AUDIT:
        raise ValueError("H1-v4 locked contract audit hash changed")
    if config.get("strict_source_hash_resume") is not True:
        raise ValueError("H1-v4 must checksum every source file during resume")
    if config.get("drive_commit") != {
        "schema": "classA_drive_api_commit_v1",
        "authoritative_backend": "google_drive_api_v3",
        "drivefs_role": "non_authoritative_cache",
        "required_headroom_bytes": H1_REMOTE_REQUIRED_HEADROOM_BYTES,
        "archive_publish_order": "archive_remote_verify_then_receipt_remote_verify",
    }:
        raise ValueError("H1-v4 Drive durability contract changed")
    expected_locked = {
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
    if config.get("locked_contract") != expected_locked:
        raise ValueError("H1-v3 locked physical hyperparameters changed")
    if checkpoint_cycles(40) != expected_locked["checkpoints"]:
        raise ValueError("H1-v3 inclusive checkpoint helper changed")
    if config.get("wall_protocols") != {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }:
        raise ValueError("H1-v3 hard/soft wall definitions changed")
    observer = config.get("packet_observer", {})
    expected_observer_fields = {
        "cut_origins": "range(Ny)",
        "spectral_clip_eps": [1e-8, 1e-10, 1e-12],
        "primary_spectral_clip_eps": 1e-10,
        "source_width_columns": [1, 3],
        "primary_source_width_columns": 3,
        "retention_width_columns": [1, 3, 5],
        "primary_retention_width_columns": 3,
        "fixed_time": 2.0,
        "primary_fit_window": [0.1, 2.0],
        "sensitivity_fit_windows": [[0.05, 1.5], [0.2, 2.0], [0.1, 2.5]],
        "wall_orientation_signs": [-1, 1],
    }
    changed = {
        key: (observer.get(key), value)
        for key, value in expected_observer_fields.items()
        if observer.get(key) != value
    }
    if changed:
        raise ValueError(f"H1-v3 endpoint estimator changed: {changed}")
    numerical_fields = (
        "raw_norm_warning_tolerance",
        "raw_norm_hard_failure_tolerance",
        "post_normalization_norm_tolerance",
        "gram_diagnostic_trigger",
    )
    missing_numerical = [name for name in numerical_fields if name not in observer]
    if missing_numerical:
        raise ValueError(
            f"H1-v3 numerical observer contract is incomplete: {missing_numerical}"
        )
    numerical_values = {name: float(observer[name]) for name in numerical_fields}
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in numerical_values.values()
    ):
        raise ValueError("H1-v3 numerical observer tolerances must be finite and positive")
    if not (
        numerical_values["post_normalization_norm_tolerance"]
        < numerical_values["gram_diagnostic_trigger"]
        <= numerical_values["raw_norm_warning_tolerance"]
        < numerical_values["raw_norm_hard_failure_tolerance"]
    ):
        raise ValueError("H1-v3 numerical observer tolerances are inconsistently ordered")
    if "signed_retarded_modular_response" not in config.get("retired_products", []):
        raise ValueError("the superseded retarded response must remain explicitly retired")


def expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_config(config)
    locked = config["locked_contract"]
    cases: list[dict[str, Any]] = []
    for alpha_1 in locked["alpha_1_values"]:
        for protocol in ("hard", "soft"):
            flags = config["wall_protocols"][protocol]
            alpha_tag = (
                str(int(alpha_1))
                if float(alpha_1).is_integer()
                else str(alpha_1).replace(".", "p")
            )
            cases.append({
                "case_id": f"H1_N20x40_{protocol}_a1-{alpha_tag}",
                "campaign": "H1_ENDPOINT_PACKET_V4",
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
                "observer": copy.deepcopy(config["packet_observer"]),
            })
    if len(cases) != 4 or len({case["case_id"] for case in cases}) != 4:
        raise AssertionError("H1-v3 expansion must contain four unique cases")
    return cases


def shard_seed(root_seed: int, case_id: str, shard_index: int) -> int:
    raw = f"{int(root_seed)}:{case_id}:{int(shard_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def _require_a100(*, smoke: bool = False) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if smoke:
            return {"device": "cpu", "smoke_override": True}
        raise RuntimeError("H1-v3 production and preflight require an A100 GPU")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if "A100" not in name.upper() and not smoke:
        raise RuntimeError(f"H1-v3 production requires an A100; detected {name!r}")
    total_gib = float(total_bytes / 1024**3)
    if not smoke and not 35.0 <= total_gib <= 45.0:
        raise RuntimeError(
            f"H1-v3 is qualified for an A100 40 GB runtime; detected {total_gib:.1f} GiB"
        )
    return {
        "device": name,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "total_gib": total_gib,
        "free_fraction": float(free_bytes / total_bytes),
        "smoke_override": bool(smoke),
    }


def _archive_paths(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, drive_root: Path, mode: str,
) -> tuple[Path, Path, str, dict[str, Any]]:
    engine_hash = sha256_file(bundle_root / "src" / "classA_U1FGTN_gpu.py")
    source_hashes = _source_hashes(bundle_root / "src")
    run_config = {
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "canonical_engine_sha256": engine_hash,
        "bundle_source_hashes_sha256": sha256_json(source_hashes),
        "shard_index": int(shard_index),
        "case": case,
        "trajectory_reuse_policy": config["trajectory_reuse"]["policy"],
    }
    run_id = f"{BUNDLE}_{sha256_json(run_config)[:16]}"
    scratch_base = Path("/content") if Path("/content").exists() else Path(tempfile.gettempdir())
    scratch = scratch_base / "classA_final_production" / BUNDLE / run_id
    collection = config[
        "production_output_collection"
        if mode == "production"
        else "pilot_output_collection"
    ]
    archive = drive_root.resolve() / collection / config["output_bundle"] / f"{run_id}.tar.gz"
    return scratch, archive, run_id, run_config


def _failed_qualification_archive(archive: Path) -> Path:
    return archive.parent / FAILED_QUALIFICATION_DIRECTORY / archive.name


def _numerical_summary(product: dict[str, Any]) -> dict[str, Any]:
    required = (
        "numerical_status",
        "raw_norm_warning",
        "raw_norm_hard_failure",
        "post_normalization_norm_failure",
        "hermiticity_failure",
        "dtype_contract_failure",
        "maximum_raw_packet_norm_error",
        "maximum_raw_packet_norm_drift",
        "maximum_post_normalization_norm_error",
    )
    missing = [name for name in required if name not in product]
    if missing:
        raise RuntimeError(f"H1-v3 numerical product is incomplete: {missing}")
    status = str(product["numerical_status"])
    if status not in NUMERICAL_STATUSES:
        raise RuntimeError(f"H1-v3 numerical product has invalid status {status!r}")
    maxima = {
        name: float(product[name])
        for name in (
            "maximum_raw_packet_norm_error",
            "maximum_raw_packet_norm_drift",
            "maximum_post_normalization_norm_error",
        )
    }
    if any(not np.isfinite(value) or value < 0.0 for value in maxima.values()):
        raise FloatingPointError("H1-v3 numerical product has an invalid norm maximum")
    hard_flags = bool(
        product["raw_norm_hard_failure"]
        or product["post_normalization_norm_failure"]
        or product["hermiticity_failure"]
        or product["dtype_contract_failure"]
    )
    if (status == "hard_failure") != hard_flags:
        raise RuntimeError("H1-v3 numerical status disagrees with its hard-failure flags")
    if status == "warning" and not bool(product["raw_norm_warning"]):
        raise RuntimeError("H1-v3 warning status lacks its raw-norm warning flag")
    optional = (
        "maximum_hermiticity_error",
        "maximum_conditional_eigenvector_gram_residual",
        "conditional_eigenvector_gram_diagnostics",
        "raw_norm_warning_tolerance",
        "raw_norm_hard_failure_tolerance",
        "post_normalization_norm_tolerance",
        "gram_diagnostic_trigger",
        "raw_norm_error_argmax",
        "actual_dtype",
        "actual_probability_dtype",
    )
    return {
        "numerical_status": status,
        "raw_norm_warning": bool(product["raw_norm_warning"]),
        "raw_norm_hard_failure": bool(product["raw_norm_hard_failure"]),
        "post_normalization_norm_failure": bool(
            product["post_normalization_norm_failure"]
        ),
        "hermiticity_failure": bool(product["hermiticity_failure"]),
        "dtype_contract_failure": bool(product["dtype_contract_failure"]),
        **maxima,
        **{name: json_ready(product.get(name)) for name in optional},
    }


def _emit_numerical_warning(summary: dict[str, Any]) -> None:
    if summary.get("numerical_status") != "warning":
        return
    payload = {
        "maximum_raw_packet_norm_error": summary[
            "maximum_raw_packet_norm_error"
        ],
        "raw_norm_warning_tolerance": summary["raw_norm_warning_tolerance"],
        "raw_norm_hard_failure_tolerance": summary[
            "raw_norm_hard_failure_tolerance"
        ],
        "action": "archive_and_continue",
    }
    print(
        "[H1-v3 NUMERICAL WARNING] "
        + json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")),
        flush=True,
    )


def _verify_existing(archive: Path) -> dict[str, Any] | None:
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if _server_commit_required(archive):
        committer = _remote_committer(archive)
        try:
            receipt = read_remote_json(committer, receipt_path)
            committer.path_commit_record(receipt_path)
        except RemoteCommitError as exc:
            if "absent" in str(exc):
                try:
                    committer.path_commit_record(archive)
                except RemoteCommitError as archive_exc:
                    if "absent" in str(archive_exc):
                        return None
                    raise
                raise RuntimeError(
                    f"server H1-v4 archive exists without its receipt: {archive}"
                ) from exc
            raise
        record = receipt.get("archive_remote_commit")
        if not isinstance(record, dict):
            raise RuntimeError("H1-v4 receipt lacks remote archive metadata")
        committer.verify_commit_record(record)
        if (
            receipt.get("archive") != archive.name
            or receipt.get("archive_sha256") != record.get("remote_sha256")
            or int(receipt.get("archive_bytes", -1))
            != int(record.get("remote_bytes", -2))
        ):
            raise RuntimeError("H1-v4 receipt disagrees with Drive metadata")
        return receipt
    if not archive.exists() and not receipt_path.exists():
        return None
    if not archive.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"incomplete H1-v3 archive/receipt pair: {archive}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != archive.name:
        raise RuntimeError("H1-v3 receipt names another archive")
    if receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError("existing H1-v3 archive failed checksum verification")
    return receipt


def _root_manifest_from_archive(path: Path) -> dict[str, Any]:
    path_for_read = path
    if _server_commit_required(path):
        committer = _remote_committer(path)
        receipt = read_remote_json(
            committer, path.with_suffix(path.suffix + ".receipt.json")
        )
        record = receipt.get("archive_remote_commit")
        if not isinstance(record, dict):
            raise RuntimeError("H1-v4 receipt lacks remote archive metadata")
        committer.verify_commit_record(record)
        cache = Path("/content/classA_remote_cache/h1_archives") / path.name
        if (
            not cache.is_file()
            or cache.stat().st_size != int(record["remote_bytes"])
            or sha256_file(cache) != record["remote_sha256"]
        ):
            committer.download_to(str(record["remote_file_id"]), cache)
        path_for_read = cache
    with tarfile.open(path_for_read, "r:gz") as archive:
        matches = [
            member for member in archive.getmembers()
            if member.name.lstrip("./") == "manifest.json"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one root manifest in {path}, found {len(matches)}"
            )
        handle = archive.extractfile(matches[0])
        if handle is None:
            raise RuntimeError(f"cannot read root manifest in {path}")
        return json.loads(handle.read().decode("utf-8"))


def _archive(scratch: Path, archive: Path, run_id: str) -> dict[str, Any]:
    remote_commit = _server_commit_required(archive)
    if not remote_commit:
        archive.parent.mkdir(parents=True, exist_ok=True)
    stage_root = (
        Path("/content/classA_remote_stage/h1_archives")
        if remote_commit
        else archive.parent
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    temporary_base = stage_root / f".{run_id}.{os.getpid()}"
    temporary = Path(shutil.make_archive(str(temporary_base), "gztar", root_dir=scratch))
    archive_sha256 = sha256_file(temporary)
    archive_bytes = temporary.stat().st_size
    archive_remote_commit: dict[str, Any] | None = None
    if remote_commit:
        committer = _remote_committer(archive)
        archive_remote_commit = committer.upload_verified(
            temporary,
            archive,
            replace=False,
            required_headroom_bytes=H1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
    else:
        os.replace(temporary, archive)
    receipt = {
        "schema_version": 2 if remote_commit else 1,
        "run_id": run_id,
        "archive": archive.name,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "created_unix": time.time(),
    }
    receipt_path = archive.with_suffix(archive.suffix + ".receipt.json")
    if archive_remote_commit is not None:
        receipt["archive_remote_commit"] = archive_remote_commit
        receipt_commit = publish_json(
            committer,
            receipt,
            receipt_path,
            replace=False,
            required_headroom_bytes=H1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
        receipt["receipt_remote_commit"] = receipt_commit
        temporary.unlink(missing_ok=True)
    else:
        write_json_atomic(receipt_path, receipt)
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


def _restore_rng_payload(payload: dict[str, np.ndarray]) -> None:
    torch.set_rng_state(
        torch.as_tensor(payload["torch_cpu_rng_state"], dtype=torch.uint8, device="cpu")
    )
    numpy_state = json.loads(str(np.asarray(payload["numpy_state_json"]).item()))
    np.random.set_state((
        str(numpy_state[0]),
        np.asarray(numpy_state[1], dtype=np.uint32),
        int(numpy_state[2]),
        int(numpy_state[3]),
        float(numpy_state[4]),
    ))
    if torch.cuda.is_available():
        cuda_keys = sorted(
            (key for key in payload if key.startswith("torch_cuda_rng_state_")),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        if len(cuda_keys) != torch.cuda.device_count():
            raise RuntimeError(
                "source RNG payload has a different CUDA device count from this runtime"
            )
        for index, key in enumerate(cuda_keys):
            torch.cuda.set_rng_state(
                torch.as_tensor(payload[key], dtype=torch.uint8, device="cpu"),
                device=index,
            )


def _tar_member_bytes(archive: tarfile.TarFile, suffix: str) -> bytes:
    matches = [
        member for member in archive.getmembers()
        if member.name.lstrip("./").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r} member, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError(f"cannot read {matches[0].name}")
    return handle.read()


def _npz_payload(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _compact_record_payload(raw: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        bit_shape = tuple(int(value) for value in np.asarray(data["bit_shape"]).tolist())
        outcomes = np.unpackbits(
            np.asarray(data["outcome_bits_packed"]), axis=-1,
            count=bit_shape[-1], bitorder="little",
        ).astype(np.bool_, copy=False).reshape(bit_shape)
        targets = np.unpackbits(
            np.asarray(data["target_bits_packed"]), axis=-1,
            count=bit_shape[-1], bitorder="little",
        ).astype(np.bool_, copy=False).reshape(bit_shape)
        return {
            "site_ids": np.asarray(data["site_ids"], dtype=np.int64),
            "channel_count": np.asarray(data["channel_count"], dtype=np.uint8),
            "outcomes": outcomes,
            "targets": targets,
            "cumulative_self_information": np.asarray(
                data["cumulative_self_information"], dtype=np.float64
            ),
        }


def _source_manifest(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        return json.loads(_tar_member_bytes(archive, "manifest.json").decode("utf-8"))


def _verify_source_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if not receipt_path.is_file():
        raise RuntimeError(f"source response archive lacks a receipt: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != path.name or receipt.get("archive_sha256") != sha256_file(path):
        raise RuntimeError(f"source response archive failed its receipt: {path}")
    return receipt


def _same_physical_case(source: dict[str, Any], target: dict[str, Any]) -> bool:
    return (
        source.get("case_id") == target.get("case_id")
        and source.get("protocol") == target.get("protocol")
        and source.get("model") == target.get("model")
        and source.get("run") == target.get("run")
    )


def _find_replay_source(
    *, drive_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, engine_hash: str, mode: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    policy = config["trajectory_reuse"]
    if mode != "production":
        return None, ["production_response_replay_is_disabled_for_pilot_or_smoke"]
    source_root = (
        drive_root.resolve() / policy["source_collection"] / policy["source_bundle"]
    )
    if not source_root.is_dir():
        return None, [f"source_archive_directory_missing:{source_root}"]
    expected_ids = list(
        range(int(shard_index) * SHARD_SIZE, (int(shard_index) + 1) * SHARD_SIZE)
    )
    expected_seed = shard_seed(int(config["root_seed"]), case["case_id"], shard_index)
    matches: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for path in sorted(source_root.glob("*.tar.gz")):
        try:
            manifest = _source_manifest(path)
        except (OSError, tarfile.TarError, RuntimeError, json.JSONDecodeError) as exc:
            rejected.append(f"unreadable:{path.name}:{type(exc).__name__}")
            continue
        if manifest.get("case_id") != case["case_id"] or int(manifest.get("shard_index", -1)) != int(shard_index):
            continue
        source_case = manifest.get("run_config", {}).get("case", {})
        source_engine = str(
            manifest.get("canonical_engine_sha256")
            or manifest.get("run_config", {}).get("canonical_engine_sha256", "")
        )
        reasons = []
        if manifest.get("bundle") != policy["source_bundle"]:
            reasons.append("bundle")
        if manifest.get("sampling_revision") != policy["source_sampling_revision"]:
            reasons.append("sampling_revision")
        if manifest.get("status") != "complete_local":
            reasons.append("status")
        if source_engine != engine_hash:
            reasons.append("canonical_engine_sha256")
        if int(manifest.get("root_seed", -1)) != int(config["root_seed"]):
            reasons.append("root_seed")
        if int(manifest.get("shard_generator_seed", -1)) != expected_seed:
            reasons.append("shard_generator_seed")
        if [int(value) for value in manifest.get("global_sample_indices", [])] != expected_ids:
            reasons.append("global_sample_indices")
        if not _same_physical_case(source_case, case):
            reasons.append("physical_case")
        if reasons:
            rejected.append(f"{path.name}:{','.join(reasons)}")
            continue
        matches.append((path, manifest))
    if len(matches) > 1:
        raise RuntimeError(
            "multiple checksum-eligible response archives match one H1-v3 shard: "
            + ", ".join(str(path) for path, _ in matches)
        )
    if not matches:
        return None, rejected or ["no_matching_response_archive"]
    path, manifest = matches[0]
    receipt = _verify_source_receipt(path)
    with tarfile.open(path, "r:gz") as archive:
        record_raw = _tar_member_bytes(archive, "ordered_born_record.npz")
        rng_raw = _tar_member_bytes(archive, "rng_before.npz")
    record = _compact_record_payload(record_raw)
    rng = _npz_payload(rng_raw)
    expected_shape = (SHARD_SIZE, 80)
    if record["site_ids"].shape[:2] != expected_shape:
        raise RuntimeError("source response record has the wrong sample/cycle axes")
    return {
        "path": path,
        "manifest": manifest,
        "receipt": receipt,
        "record": record,
        "rng": rng,
        "record_sha256": hashlib.sha256(record_raw).hexdigest(),
        "rng_before_sha256": hashlib.sha256(rng_raw).hexdigest(),
    }, rejected


def _validate_replayed_record(
    source: dict[str, np.ndarray], replay_path: Path,
) -> dict[str, Any]:
    replay = load_ordered_record(replay_path)
    for key in ("site_ids", "channel_count", "outcomes", "targets"):
        if not np.array_equal(source[key], replay[key]):
            raise RuntimeError(f"replayed ordered record differs from source field {key}")
    difference = float(np.max(np.abs(
        source["cumulative_self_information"] - replay["cumulative_self_information"]
    )))
    if difference > 1e-9:
        raise RuntimeError(
            f"replayed conditional log probabilities differ by {difference:.3e}"
        )
    return {
        "exact_discrete_record_match": True,
        "maximum_cumulative_self_information_difference": difference,
        "tolerance": 1e-9,
    }


def run_case(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, drive_root: Path, mode: str, archive_result: bool = True,
) -> dict[str, Any]:
    if not 0 <= int(shard_index) < 5:
        raise IndexError("H1-v3 shard index must lie in 0..4")
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
    actual_walls = sorted(int(value) % model.Nx for value in model.DW_loc)
    if len(actual_walls) != 2 or len(set(actual_walls)) != 2:
        raise RuntimeError(f"model did not expose two distinct DW_loc values: {model.DW_loc}")
    observer_config = case["observer"]
    times = np.arange(
        float(observer_config["modular_time_start"]),
        float(observer_config["modular_time_stop"])
        + 0.5 * float(observer_config["modular_time_step"]),
        float(observer_config["modular_time_step"]),
    )
    fit_windows = [observer_config["primary_fit_window"], *observer_config["sensitivity_fit_windows"]]
    observer = H1EndpointPacketObserver(
        nx=model.Nx,
        ny=model.Ny,
        wall_x=actual_walls,
        checkpoints=config["locked_contract"]["checkpoints"],
        global_sample_ids=global_ids,
        modular_times=times,
        spectral_clip_eps=observer_config["spectral_clip_eps"],
        source_widths=observer_config["source_width_columns"],
        retention_widths=observer_config["retention_width_columns"],
        primary_epsilon=float(observer_config["primary_spectral_clip_eps"]),
        primary_source_width=int(observer_config["primary_source_width_columns"]),
        primary_retention_width=int(observer_config["primary_retention_width_columns"]),
        fixed_time=float(observer_config["fixed_time"]),
        fit_windows=fit_windows,
        orientation_signs=observer_config["wall_orientation_signs"],
        minimum_primary_retention=float(observer_config["minimum_primary_retention"]),
        raw_norm_warning_tolerance=float(
            observer_config["raw_norm_warning_tolerance"]
        ),
        raw_norm_hard_failure_tolerance=float(
            observer_config["raw_norm_hard_failure_tolerance"]
        ),
        post_normalization_norm_tolerance=float(
            observer_config["post_normalization_norm_tolerance"]
        ),
        gram_diagnostic_trigger=float(observer_config["gram_diagnostic_trigger"]),
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
    replay_source, replay_rejections = _find_replay_source(
        drive_root=drive_root,
        config=config,
        case=case,
        shard_index=shard_index,
        engine_hash=run_config["canonical_engine_sha256"],
        mode=mode,
    )
    if replay_source is not None:
        _restore_rng_payload(replay_source["rng"])
        trajectory_provenance = {
            "mode": "checksum_verified_response_record_replay",
            "source_archive": str(replay_source["path"]),
            "source_archive_sha256": replay_source["receipt"]["archive_sha256"],
            "source_record_sha256": replay_source["record_sha256"],
            "source_rng_before_sha256": replay_source["rng_before_sha256"],
            "fallback_rejections": replay_rejections,
        }
        frozen_schedule = replay_source["record"]["site_ids"]
        frozen_outcomes = replay_source["record"]["outcomes"]
    else:
        trajectory_provenance = {
            "mode": "fresh_same_preregistered_seed",
            "source_archive": None,
            "fallback_reasons": replay_rejections,
        }
        frozen_schedule = None
        frozen_outcomes = None

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
        frozen_schedule=frozen_schedule,
        frozen_outcomes=frozen_outcomes,
        require_no_covariance_materialization=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - began
    if result.get("state_representation_resolved") != "physical_frame":
        raise RuntimeError("H1-v3 random-pure run did not use occupied-frame representation")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("H1-v3 unexpectedly materialized a full covariance")
    with (shard_root / "rng_after.npz").open("wb") as handle:
        np.savez_compressed(handle, **_rng_payload())

    product = observer.save(shard_root / "h1_endpoint_packet", config=run_config)
    numerical = _numerical_summary(product)
    _emit_numerical_warning(numerical)
    record_path = shard_root / "ordered_born_record.npz"
    record_product = record.save(record_path)
    replay_validation = None
    if replay_source is not None:
        replay_validation = _validate_replayed_record(
            replay_source["record"], record_path
        )
        trajectory_provenance["replay_validation"] = replay_validation
    hard_numerical_failure = numerical["numerical_status"] == "hard_failure"
    manifest = {
        "schema_version": 2,
        "status": (
            "numerical_hard_failure" if hard_numerical_failure else "complete_local"
        ),
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "bundle_source_hashes_sha256": sha256_json(
            _source_hashes(bundle_root / "src")
        ),
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
        "model_DW_loc": actual_walls,
        "trajectory_provenance": trajectory_provenance,
        "gpu_preflight": gpu,
        "elapsed_seconds": elapsed,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(model.device)) if torch.cuda.is_available() else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(model.device)) if torch.cuda.is_available() else 0,
        "numerical_status": numerical["numerical_status"],
        "numerical_diagnostics": numerical,
        "products": {
            "h1_endpoint_packet": product,
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
    if hard_numerical_failure:
        if not archive_result:
            return {
                "status": "numerical_hard_failure",
                "manifest": manifest,
                "scratch": str(scratch),
                "failed_qualification_archive": str(
                    _failed_qualification_archive(archive)
                ),
            }
        failed_archive = _failed_qualification_archive(archive)
        receipt = _archive(scratch, failed_archive, run_id)
        return {
            "status": "numerical_hard_failure_archived",
            "receipt": receipt,
            "archive_path": str(failed_archive),
            "manifest": manifest,
        }
    if not archive_result:
        return {
            "status": "preflight_complete",
            "manifest": manifest,
            "scratch": str(scratch),
        }
    receipt = _archive(scratch, archive, run_id)
    return {
        "status": "archived_to_drive",
        "receipt": receipt,
        "manifest": manifest,
        "numerical_status": numerical["numerical_status"],
    }


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
    failure: str | None = None
    manifest: dict[str, Any] | None = None
    qualification_receipt: dict[str, Any] | None = None
    failed_receipt: dict[str, Any] | None = None
    failed_archive: Path | None = None
    scratch: Path | None = None
    total = 0
    peak = 0
    shard_bytes = 0
    projected_bytes = 0
    numerical: dict[str, Any] = {
        "numerical_status": "unavailable",
        "raw_norm_warning": None,
        "raw_norm_hard_failure": None,
        "post_normalization_norm_failure": None,
        "hermiticity_failure": None,
        "dtype_contract_failure": None,
        "maximum_raw_packet_norm_error": None,
        "maximum_raw_packet_norm_drift": None,
        "maximum_post_normalization_norm_error": None,
    }
    safe = False
    try:
        scratch, archive, run_id, _ = _archive_paths(
            bundle_root=bundle_root, config=config, case=case,
            shard_index=0, drive_root=drive_root, mode="production",
        )
        existing = _verify_existing(archive)
        if existing is not None:
            manifest = _root_manifest_from_archive(archive)
            if (
                manifest.get("bundle") != BUNDLE
                or manifest.get("sampling_revision") != REVISION
                or manifest.get("audit_sha256") != AUDIT
                or manifest.get("case_id") != case["case_id"]
                or int(manifest.get("shard_index", -1)) != 0
                or manifest.get("status") != "complete_local"
            ):
                raise RuntimeError(
                    "existing qualification archive is not the current H1-v3 shard zero"
                )
            qualification_receipt = existing
        else:
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
            scratch = Path(result["scratch"])
            if result.get("status") == "numerical_hard_failure":
                failed_archive = _failed_qualification_archive(archive)
                failed_receipt = _archive(scratch, failed_archive, run_id)
                failure = (
                    "FloatingPointError: H1-v3 qualification reached a "
                    "numerical hard-failure ceiling"
                )
        if manifest is None:
            raise RuntimeError("H1-v3 qualification produced no manifest")
        product = manifest.get("products", {}).get("h1_endpoint_packet")
        if not isinstance(product, dict):
            raise RuntimeError("H1-v3 qualification manifest lacks its packet product")
        numerical = _numerical_summary(product)
        total = int(manifest["gpu_preflight"]["total_bytes"])
        peak = int(manifest["gpu_peak_reserved_bytes"])
        shard_bytes = int(sum(row["bytes"] for row in manifest["products"].values()))
        projected_bytes = 20 * shard_bytes
        safe = bool(
            numerical["numerical_status"] in {"pass", "warning"}
            and "A100" in str(manifest["gpu_preflight"]["device"]).upper()
            and 35.0 <= total / 1024**3 <= 45.0
            and peak <= 0.8 * total
            and shard_bytes <= 1 * 1024**3
            and projected_bytes <= 12 * 1024**3
        )
        if safe and qualification_receipt is None:
            if scratch is None:
                raise RuntimeError("H1-v3 qualification scratch directory is missing")
            qualification_receipt = _archive(scratch, archive, run_id)
        if numerical["numerical_status"] == "hard_failure":
            safe = False
            if failure is None:
                failure = (
                    "FloatingPointError: H1-v3 qualification manifest records a "
                    "numerical hard failure"
                )
    except FloatingPointError as exc:
        safe = False
        failure = f"{type(exc).__name__}: {exc}"
    except (torch.cuda.OutOfMemoryError, RuntimeError, OSError, tarfile.TarError) as exc:
        safe = False
        failure = f"{type(exc).__name__}: {exc}"

    payload = {
        "schema": A100_PREFLIGHT_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "canonical_engine_sha256": sha256_file(
            bundle_root / "src" / "classA_U1FGTN_gpu.py"
        ),
        "bundle_source_hashes_sha256": sha256_json(
            _source_hashes(bundle_root / "src")
        ),
        "case_id": case["case_id"],
        "measured_trajectories": SHARD_SIZE,
        "measured_cycles": 80,
        "measured_checkpoints": [40, 48, 56, 64, 72, 80],
        "measured_translated_cuts_per_checkpoint": 40,
        "peak_reserved_bytes": peak,
        "gpu_total_bytes": total,
        "peak_reserved_fraction": float(peak / total) if total else None,
        "measured_shard_product_bytes": shard_bytes,
        "projected_twenty_shard_bytes": projected_bytes,
        **numerical,
        "scientific_shard_archived": qualification_receipt is not None,
        "qualification_archive": (
            None if qualification_receipt is None else qualification_receipt["archive"]
        ),
        "qualification_archive_sha256": (
            None
            if qualification_receipt is None
            else qualification_receipt["archive_sha256"]
        ),
        "failed_qualification_archived": failed_receipt is not None,
        "failed_qualification_archive": (
            None if failed_archive is None else str(failed_archive)
        ),
        "failed_qualification_archive_sha256": (
            None if failed_receipt is None else failed_receipt["archive_sha256"]
        ),
        "safe": bool(safe),
        "safety_rule": (
            "A100 40GB, numerical_status in {pass,warning}, peak<=0.8*GPU, "
            "shard<=1GiB, projected outputs<=12GiB"
        ),
        "contract_changed": True,
        "failure": failure,
        "created_unix": time.time(),
    }
    path = _preflight_path(drive_root, config)
    if _server_commit_required(path):
        publish_json(
            _remote_committer(path),
            payload,
            path,
            replace=True,
            required_headroom_bytes=H1_REMOTE_REQUIRED_HEADROOM_BYTES,
        )
    else:
        write_json_atomic(path, payload)
    payload["receipt_path"] = str(path)
    return payload


def require_safe_preflight(
    *, drive_root: Path, config: dict[str, Any],
) -> dict[str, Any]:
    path = _preflight_path(drive_root, config)
    if _server_commit_required(path):
        try:
            committer = _remote_committer(path)
            payload = read_remote_json(committer, path)
            committer.path_commit_record(path)
        except RemoteCommitError as exc:
            raise RuntimeError(
                "H1-v4 production locked until --a100-preflight passes: "
                f"missing server-verified receipt {path}"
            ) from exc
    else:
        if not path.is_file():
            raise RuntimeError(
                f"H1-v4 production locked until --a100-preflight passes: missing {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": A100_PREFLIGHT_SCHEMA,
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "audit_sha256": AUDIT,
        "canonical_engine_sha256": sha256_file(
            Path(__file__).resolve().parent / "classA_U1FGTN_gpu.py"
        ),
        "bundle_source_hashes_sha256": sha256_json(
            _source_hashes(Path(__file__).resolve().parent)
        ),
        "case_id": "H1_N20x40_soft_a1-1",
        "safe": True,
        "contract_changed": True,
        "scientific_shard_archived": True,
        "failed_qualification_archived": False,
        "dtype_contract_failure": False,
    }
    mismatch = {
        key: (payload.get(key), value)
        for key, value in expected.items() if payload.get(key) != value
    }
    if payload.get("numerical_status") not in {"pass", "warning"}:
        mismatch["numerical_status"] = (
            payload.get("numerical_status"), "pass or warning"
        )
    for name in (
        "maximum_raw_packet_norm_error",
        "maximum_raw_packet_norm_drift",
        "maximum_post_normalization_norm_error",
    ):
        try:
            value = float(payload[name])
        except (KeyError, TypeError, ValueError):
            mismatch[name] = (payload.get(name), "finite nonnegative")
        else:
            if not np.isfinite(value) or value < 0.0:
                mismatch[name] = (payload.get(name), "finite nonnegative")
    qualification_name = payload.get("qualification_archive")
    if (
        not isinstance(qualification_name, str)
        or not qualification_name
        or Path(qualification_name).name != qualification_name
    ):
        mismatch["qualification_archive"] = (
            qualification_name, "one archive basename"
        )
    else:
        qualification_archive = path.parent / qualification_name
        try:
            qualification_receipt = _verify_existing(qualification_archive)
            if qualification_receipt is None:
                raise RuntimeError("qualification archive pair is missing")
            if qualification_receipt.get("archive_sha256") != payload.get(
                "qualification_archive_sha256"
            ):
                raise RuntimeError(
                    "qualification receipt hash differs from the preflight receipt"
                )
            qualification_manifest = _root_manifest_from_archive(
                qualification_archive
            )
            manifest_expected = {
                "bundle": BUNDLE,
                "sampling_revision": REVISION,
                "audit_sha256": AUDIT,
                "bundle_source_hashes_sha256": expected[
                    "bundle_source_hashes_sha256"
                ],
                "case_id": expected["case_id"],
                "shard_index": 0,
                "status": "complete_local",
            }
            manifest_mismatch = {
                key: (qualification_manifest.get(key), value)
                for key, value in manifest_expected.items()
                if qualification_manifest.get(key) != value
            }
            if manifest_mismatch:
                raise RuntimeError(
                    f"qualification archive manifest is stale: {manifest_mismatch}"
                )
        except (OSError, RuntimeError, ValueError, tarfile.TarError) as exc:
            mismatch["qualification_archive_pair"] = (
                str(exc), "present, checksummed, and current"
            )
    if mismatch:
        raise RuntimeError(
            f"A100 H1-v3 preflight is unsafe, incomplete, or stale: {mismatch}"
        )
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
    parser.add_argument("--remote-status", action="store_true")
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
            current = require_safe_preflight(drive_root=args.drive_root, config=config)
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
            raise RuntimeError("H1-v3 A100 qualification completed but was not safe")
        return 0
    selected = {row["case_id"]: row for row in cases}
    case_id = args.case_id or cases[0]["case_id"]
    if case_id not in selected:
        raise KeyError(f"unknown H1-v3 case {case_id!r}")
    case = selected[case_id]
    if args.remote_status:
        _, archive, _, _ = _archive_paths(
            bundle_root=bundle_root,
            config=config,
            case=case,
            shard_index=int(args.shard_index),
            drive_root=args.drive_root.resolve(),
            mode=str(args.mode),
        )
        receipt = _verify_existing(archive)
        if receipt is None:
            print(json.dumps({"exists": False, "archive": str(archive)}))
            return 0
        print(json.dumps({
            "exists": True,
            "archive": str(archive),
            "receipt": receipt,
            "manifest": _root_manifest_from_archive(archive),
        }, sort_keys=True))
        return 0
    print(json.dumps({
        "bundle": BUNDLE,
        "case_id": case_id,
        "shard_index": args.shard_index,
        "global_sample_indices": list(
            range(args.shard_index * SHARD_SIZE, args.shard_index * SHARD_SIZE + SHARD_SIZE)
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
    if result.get("status") == "numerical_hard_failure_archived":
        raise FloatingPointError(
            "H1-v3 numerical hard failure; complete diagnostics were archived at "
            f"{result.get('archive_path')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
