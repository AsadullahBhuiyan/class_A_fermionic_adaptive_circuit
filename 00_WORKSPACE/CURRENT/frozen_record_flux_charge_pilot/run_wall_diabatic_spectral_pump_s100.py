#!/usr/bin/env python3
"""Wall-diabatized spectral pump for the verified Ny=24 endpoint ensembles.

The finite-volume flattened parent has two wall modes which weakly hybridize at
the flux crossing.  Ordinary overlap continuation is used for the spectator
subspace.  Inside a strictly qualified isolated rank-two edge cluster, the
occupied member is instead continued by its left/right wall label.
"""

from __future__ import annotations

import os

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import argparse
import contextlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any, Iterable

import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
CAMPAIGN_SCHEMA = "wall_diabatic_spectral_pump_campaign_v1"
RESULT_SCHEMA = "wall_diabatic_spectral_pump_result_v1"
COMPLETION_SCHEMA = "wall_diabatic_spectral_pump_completion_v1"
SHARD_RESULT_SCHEMA = "wall_pump_width_endpoint_shard_v1"
SHARD_COMPLETION_SCHEMA = "wall_pump_width_endpoint_shard_completion_v1"
DEFAULT_CONFIG = PROJECT_ROOT / "campaign_config.wall_diabatic_spectral_pump_s100_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1"
DEFAULT_NEW_ENDPOINT_ROOT = PROJECT_ROOT / "imported_endpoints/wall_pump_width_endpoints_s100_v1"
SOURCE_PATHS = {"campaign_runner": Path(__file__).resolve()}
DIRECTIONS = ("ccw", "cw")

# These are source contracts, not user-tunable campaign settings.  Keeping a
# runner-owned copy serves two purposes: a source row passed directly to the
# public endpoint loader is still checked strictly, and an edited campaign
# configuration cannot silently bless a self-consistent but foreign endpoint
# shard.  ``validate_config`` below requires its serialized copy to match this
# table exactly.
LOCKED_SHARDED_SOURCE_CONTRACTS = {
    "gpu": {
        "sampling_revision": "wall_pump_width_endpoints_s100_v1",
        "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
        "execution_backend": "gpu",
        "config_hash_key": "base_config_sha256",
        "config_hash": "428d57fb89db7afc404ccbd0721292f31957f34910b2717817bfb8b31d43d61d",
        "source_hashes": {
            "run_campaign.py": "1beff3833d719d2c2cd9fc64d86858cd8f5c3fa4d584a5ddb35a8c2d911e9dad",
            "campaign_config.json": "5c538a3bcd84da8b523bb8e8679b5be41503d9c995d97c4710ccc8ed8e2bf31f",
            "src/classA_U1FGTN_gpu.py": "53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700",
            "src/occupied_frame_gpu.py": "bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0",
        },
    },
    "canonical_cpu": {
        "sampling_revision": "wall_pump_width_endpoints_s100_v1",
        "canonical_entry_point": "classA_U1FGTN.run_markov_circuit",
        "execution_backend": "canonical_cpu",
        "config_hash_key": "config_hash",
        "config_hash": "3376a120018b4e5102342cab1afe6187f6ea3e2ccda6774e12da979821db504f",
        "source_hashes": {
            "cpu_fallback_runner": "4eab8a947ec09c0b5410b707be9199ef5a0eba4a76859d4da4d368072efa442c",
            "cpu_fallback_config": "ee65c7038ff80ceee3e79d1ef4e07960d565a73fbcfbfad0a4b158a7380bf352",
            "canonical_cpu_engine": "e8bc0ea58b14f311aa64b4297d100183227a5ed3bcb819b3dd989221264d254e",
            "canonical_occupied_frame": "5e689503de89812546c98d4e474511ded13b8ac2b42d57b41c730ed444e2d4c1",
            "gpu_campaign_contract": "5c538a3bcd84da8b523bb8e8679b5be41503d9c995d97c4710ccc8ed8e2bf31f",
        },
    },
}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_config_hash(config: dict[str, Any]) -> str:
    keys = (
        "schema", "campaign_id", "sources", "bridge_sources", "sharded_source_contracts",
        "ensemble", "flux",
        "spectral_parent", "wall_diabatization", "chern_control", "acceptance",
    )
    return hashlib.sha256(canonical_json({key: config[key] for key in keys}).encode()).hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_path(path) for name, path in SOURCE_PATHS.items()}


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CAMPAIGN_SCHEMA:
        raise ValueError(f"expected schema {CAMPAIGN_SCHEMA!r}")
    if config.get("campaign_id") != "N20_24_28_32x24_wall_diabatic_spectral_pump_s100_v1":
        raise ValueError("unexpected campaign identity")
    expected_cells = {
        ("nsh1", 20, "per_sample"), ("nsh1", 24, "per_sample"),
        ("dense", 20, "per_sample"), ("nsh1", 28, "five_sample_shards"),
        ("nsh1", 32, "five_sample_shards"), ("dense", 24, "five_sample_shards"),
        ("dense", 28, "five_sample_shards"), ("dense", 32, "five_sample_shards"),
    }
    sources = config.get("sources", [])
    actual_cells = {(row["protocol"], int(row["Nx"]), row["kind"]) for row in sources}
    if len(sources) != 8 or actual_cells != expected_cells or any(int(row["Ny"]) != 24 for row in sources):
        raise ValueError("source matrix differs from the locked eight-cell campaign")
    if len({row["cell"] for row in sources}) != 8:
        raise ValueError("source cell IDs are not unique")
    for row in sources:
        if row["kind"] == "per_sample" and (
            len(str(row.get("expected_config_hash", ""))) != 64
            or not row.get("expected_source_hashes")
            or any(len(str(value)) != 64 for value in row["expected_source_hashes"].values())
        ):
            raise ValueError("per-sample source identities are not checksum-pinned")
    bridge = config.get("bridge_sources", [])
    expected_bridge = {
        ("nsh1", 20), ("nsh1", 24), ("dense", 20),
    }
    if (
        len(bridge) != 3
        or {(row["protocol"], int(row["Nx"])) for row in bridge} != expected_bridge
        or any(
            int(row["Ny"]) != 24 or row["kind"] != "five_sample_shards"
            or int(row["samples_per_wall"]) != 25 or row["source_backend"] != "gpu"
            or bool(row["is_primary"])
            for row in bridge
        )
    ):
        raise ValueError("GPU bridge table differs from the locked 150-endpoint contract")
    if len({row["cell"] for row in [*sources, *bridge]}) != 11:
        raise ValueError("primary and bridge source cell IDs are not unique")
    contracts = config.get("sharded_source_contracts", {})
    if contracts != LOCKED_SHARDED_SOURCE_CONTRACTS:
        raise ValueError("accepted sharded-source contracts differ from the locked identities")
    for backend, contract in contracts.items():
        if contract.get("execution_backend") != backend:
            raise ValueError("sharded-source backend identity is inconsistent")
        if contract.get("sampling_revision") != "wall_pump_width_endpoints_s100_v1":
            raise ValueError("sharded-source sampling revision changed")
        if len(str(contract.get("config_hash", ""))) != 64:
            raise ValueError("sharded-source configuration hash is invalid")
        if not contract.get("source_hashes") or any(
            len(str(value)) != 64 for value in contract["source_hashes"].values()
        ):
            raise ValueError("sharded-source hashes are invalid")
    if config["ensemble"] != {"samples_per_cell_wall": 100, "walls": ["soft", "hard"]}:
        raise ValueError("ensemble differs from the locked 1,600-endpoint campaign")
    flux = config["flux"]
    if flux != {
        "grid_intervals": 256, "regulator": 1e-7,
        "directions": {"ccw": 1, "cw": -1}, "twist_gauge": "uniform",
    }:
        raise ValueError("flux grid or signed regulator changed")
    edge = config["wall_diabatization"]
    expected_edge = {
        "edge_block_rank": 2, "wall_window_radius": 2,
        "minimum_external_gap": 0.1,
        "require_external_gap_larger_than_internal": True,
        "maximum_left_B_eigenvalue": -0.8,
        "minimum_right_B_eigenvalue": 0.8,
        "minimum_combined_wall_weight": 0.8,
        "minimum_neighboring_cluster_overlap": 0.8,
        "unresolved_policy": "retain_diagnostic_continuation_and_mark_unresolved",
    }
    if edge != expected_edge:
        raise ValueError("wall-diabatization thresholds changed")
    if config["chern_control"] != {
        "definition": "legacy_periodic_three_wedge_real_space_chern_from_occupied_frame",
        "xref": "Nx/2", "yref_values": "all", "radius": 4.0,
        "reported_statistic": "mean_over_yref",
    }:
        raise ValueError("source Chern control changed")
    if config["acceptance"] != {
        "input_frame_gram_tolerance": 1e-8,
        "projector_idempotency_tolerance": 1e-10,
        "charge_conservation_tolerance": 1e-10,
        "large_gauge_parent_tolerance": 1e-8,
    }:
        raise ValueError("numerical gates changed")


def source_rows(config: dict[str, Any], *, include_bridge: bool = True) -> dict[str, dict[str, Any]]:
    rows = list(config["sources"])
    if include_bridge:
        rows.extend(config["bridge_sources"])
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if row["kind"] == "five_sample_shards":
            row["accepted_source_contracts"] = config["sharded_source_contracts"]
        result[str(row["cell"])] = row
    return result


def tasks(config: dict[str, Any], *, include_bridge: bool = True) -> list[dict[str, Any]]:
    sources = list(config["sources"])
    if include_bridge:
        sources.extend(config["bridge_sources"])
    rows = [
        {
            "stage": "wall_diabatic_pump",
            "task_id": f"wall_diabatic_{source['cell']}_{wall}_sample_{sample_id:03d}",
            "cell": str(source["cell"]), "protocol": str(source["protocol"]),
            "size": f"N{int(source['Nx'])}x{int(source['Ny'])}",
            "Nx": int(source["Nx"]), "Ny": int(source["Ny"]),
            "wall": wall, "sample_id": sample_id,
            "grid_intervals": int(config["flux"]["grid_intervals"]),
            "edge_block_rank": 2,
            "wall_window": int(config["wall_diabatization"]["wall_window_radius"]),
            "is_primary": bool(source.get("is_primary", True)),
            "source_backend": str(source.get(
                "source_backend", "cpu" if source["kind"] == "per_sample" else "gpu"
            )),
            "result_collection": (
                "pump" if bool(source.get("is_primary", True)) else "bridge_pump"
            ),
            "control_kind": "wall_diabatic_with_ordinary_and_instantaneous",
        }
        for source in sources
        for wall in config["ensemble"]["walls"]
        for sample_id in range(int(source.get(
            "samples_per_wall", config["ensemble"]["samples_per_cell_wall"]
        )))
    ]
    expected = 1750 if include_bridge else 1600
    if len(rows) != expected or len({row["task_id"] for row in rows}) != expected:
        raise RuntimeError(f"campaign must expand to exactly {expected:,} unique endpoint tasks")
    if sum(bool(row["is_primary"]) for row in rows) != 1600:
        raise RuntimeError("campaign must retain exactly 1,600 primary endpoint tasks")
    if len({result_paths(Path("."), row)[0] for row in rows}) != expected:
        raise RuntimeError("campaign tasks do not resolve to unique result paths")
    return rows


def result_paths(output_root: Path, task: dict[str, Any]) -> tuple[Path, Path]:
    collection = str(task.get("result_collection", "pump"))
    variant = task.get("variant")
    prefix = Path(output_root) / collection
    if variant not in (None, "", "primary"):
        prefix = prefix / str(variant)
    result = (
        prefix / task["protocol"] / task["size"] / task["wall"]
        / f"sample_{int(task['sample_id']):03d}.npz"
    )
    return result, result.with_suffix(".completion.json")


def failure_path(output_root: Path, task: dict[str, Any]) -> Path:
    return Path(output_root) / "failures" / f"{task['task_id']}.json"


@dataclass(frozen=True)
class EndpointRef:
    frame_path: str
    completion_path: str
    member_index: int | None
    result_sha256: str
    result_bytes: int
    completion_sha256: str
    source_config_hash: str
    source_schema: str

    def dependency(self) -> dict[str, Any]:
        return {
            "frame_path": self.frame_path,
            "completion_path": self.completion_path,
            "member_index": self.member_index,
            "result_sha256": self.result_sha256,
            "result_bytes": self.result_bytes,
            "completion_sha256": self.completion_sha256,
            "source_config_hash": self.source_config_hash,
            "source_schema": self.source_schema,
        }


def _read_completion_pair(result: Path, completion_path: Path) -> tuple[dict[str, Any], str]:
    if not result.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"missing source result/completion pair: {result}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    record = completion.get("result", {})
    if record.get("name") != result.name or int(record.get("bytes", -1)) != result.stat().st_size:
        raise RuntimeError(f"source result name/size mismatch: {result}")
    digest = sha256_path(result)
    if record.get("sha256") != digest:
        raise RuntimeError(f"source result checksum mismatch: {result}")
    return completion, digest


def _existing_source_ref(source: dict[str, Any], wall: str, sample_id: int) -> EndpointRef:
    result = PROJECT_ROOT / source["relative_root"] / wall / f"sample_{sample_id:03d}.npz"
    completion_path = result.with_suffix(".completion.json")
    completion, digest = _read_completion_pair(result, completion_path)
    expected = {
        "stage": "burnin", "task_id": f"burnin_{wall}_sample_{sample_id:03d}",
        "wall": wall, "sample_id": sample_id,
        "config_hash": source["expected_config_hash"],
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(f"source completion {key} mismatch: {result}")
    if completion.get("source_hashes") != source["expected_source_hashes"]:
        raise RuntimeError(f"source completion source_hashes mismatch: {result}")
    with np.load(result, allow_pickle=False) as saved:
        frame = np.asarray(saved["frame"])
        rank = int(np.asarray(saved["rank"]).item())
        if frame.dtype != np.complex128 or frame.shape != (2 * source["Nx"] * source["Ny"], rank):
            raise RuntimeError(f"source frame dtype/shape mismatch: {result}")
        embedded = json.loads(str(np.asarray(saved["metadata_json"]).item()))
        for key in ("stage", "task_id", "wall", "sample_id", "config_hash", "source_hashes"):
            if embedded.get(key) != completion.get(key):
                raise RuntimeError(f"source embedded identity mismatch for {key}: {result}")
    return EndpointRef(
        str(result), str(completion_path), None, digest, result.stat().st_size,
        sha256_path(completion_path), str(completion["config_hash"]), str(completion["schema"]),
    )


def _sharded_source_ref(
    source: dict[str, Any], wall: str, sample_id: int, new_endpoint_root: Path
) -> EndpointRef:
    shard_index, member_index = divmod(sample_id, 5)
    root = Path(new_endpoint_root) / source["relative_root"] / wall
    result = root / f"shard_{shard_index:02d}.npz"
    completion_path = result.with_suffix(".completion.json")
    completion, digest = _read_completion_pair(result, completion_path)
    expected_ids = list(range(5 * shard_index, 5 * shard_index + 5))
    backend = str(completion.get("execution_backend", ""))
    contracts = source.get("accepted_source_contracts", LOCKED_SHARDED_SOURCE_CONTRACTS)
    if contracts != LOCKED_SHARDED_SOURCE_CONTRACTS:
        raise RuntimeError("endpoint source row carries an altered source-contract table")
    if backend not in contracts:
        raise RuntimeError(f"endpoint shard backend is not accepted: {backend!r}")
    declared_backend = source.get("source_backend")
    if declared_backend is not None and backend != str(declared_backend):
        raise RuntimeError(
            f"endpoint shard backend {backend!r} differs from the source lane "
            f"{declared_backend!r}"
        )
    contract = contracts[backend]
    source_cell = str(source.get("source_cell", source["cell"]))
    collection = "bridge" if str(source["relative_root"]).startswith("bridge/") else "endpoints"
    expected = {
        "schema": SHARD_COMPLETION_SCHEMA, "stage": "endpoint_shard",
        "collection": collection, "cell": source_cell, "protocol": source["protocol"],
        "Nx": int(source["Nx"]), "Ny": int(source["Ny"]), "wall": wall,
        "shard_index": shard_index, "sample_ids": expected_ids,
        "sampling_revision": contract["sampling_revision"],
        "canonical_entry_point": contract["canonical_entry_point"],
        "execution_backend": contract["execution_backend"],
        "cycles": 48, contract["config_hash_key"]: contract["config_hash"],
        "source_hashes": contract["source_hashes"],
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise RuntimeError(f"endpoint shard completion {key} mismatch: {result}")
    with np.load(result, allow_pickle=False) as saved:
        if str(np.asarray(saved["schema"]).item()) != SHARD_RESULT_SCHEMA:
            raise RuntimeError(f"endpoint shard result schema mismatch: {result}")
        ids = np.asarray(saved["sample_ids"], dtype=np.int64)
        ranks = np.asarray(saved["ranks"], dtype=np.int64)
        frames = np.asarray(saved["frames"])
        if not np.array_equal(ids, expected_ids) or ranks.shape != (5,):
            raise RuntimeError(f"endpoint shard member table mismatch: {result}")
        if frames.dtype != np.complex128 or frames.ndim != 3 or frames.shape[:2] != (
            5, 2 * int(source["Nx"]) * int(source["Ny"]),
        ):
            raise RuntimeError(f"endpoint shard frame dtype/shape mismatch: {result}")
        if np.any(ranks <= 0) or np.any(ranks > frames.shape[2]):
            raise RuntimeError(f"endpoint shard rank is invalid: {result}")
        metadata = json.loads(str(np.asarray(saved["metadata_json"]).item()))
        for key in (
            "collection", "cell", "protocol", "Nx", "Ny", "wall", "shard_index",
            "sample_ids", "sampling_revision", "canonical_entry_point",
            "execution_backend", "cycles", contract["config_hash_key"], "source_hashes",
        ):
            if metadata.get(key) != completion.get(key):
                raise RuntimeError(f"endpoint shard embedded identity mismatch for {key}: {result}")
        expected_wall_flags = {
            "dw_truncation": wall == "hard", "meas_slab_only": wall == "hard",
        }
        if metadata.get("wall_flags") != expected_wall_flags:
            raise RuntimeError(f"endpoint shard wall flags mismatch: {result}")
        if (
            metadata.get("dtype") != "complex128"
            or metadata.get("sequence") != "raster_y"
            or metadata.get("perfect_correction") is not True
            or metadata.get("postselect") is not False
        ):
            raise RuntimeError(f"endpoint shard scientific metadata mismatch: {result}")
        scalar_expected = {
            "sampling_revision": contract["sampling_revision"],
            "canonical_entry_point": contract["canonical_entry_point"],
            "execution_backend": contract["execution_backend"],
            "collection": collection, "protocol": source["protocol"], "wall": wall,
            "Nx": int(source["Nx"]), "Ny": int(source["Ny"]), "cycles_total": 48,
            "nshell_label": source["protocol"],
        }
        for key, value in scalar_expected.items():
            if key not in saved or np.asarray(saved[key]).item() != value:
                raise RuntimeError(f"endpoint shard NPZ field {key} mismatch: {result}")
        if not np.isclose(float(np.asarray(saved["alpha_1"]).item()), 1.0):
            raise RuntimeError(f"endpoint shard alpha_1 mismatch: {result}")
        if not np.isclose(float(np.asarray(saved["alpha_2"]).item()), 30.0):
            raise RuntimeError(f"endpoint shard alpha_2 mismatch: {result}")
    return EndpointRef(
        str(result), str(completion_path), member_index, digest, result.stat().st_size,
        sha256_path(completion_path), str(completion[contract["config_hash_key"]]),
        str(completion["schema"]),
    )


def endpoint_ref(
    source: dict[str, Any], wall: str, sample_id: int, new_endpoint_root: Path
) -> EndpointRef:
    if source["kind"] == "per_sample":
        return _existing_source_ref(source, wall, sample_id)
    if source["kind"] == "five_sample_shards":
        return _sharded_source_ref(source, wall, sample_id, new_endpoint_root)
    raise ValueError(f"unknown source kind {source['kind']!r}")


def load_endpoint(ref: EndpointRef, nx: int, ny: int) -> np.ndarray:
    result = Path(ref.frame_path)
    if result.stat().st_size != ref.result_bytes or sha256_path(result) != ref.result_sha256:
        raise RuntimeError("endpoint dependency changed after discovery")
    if sha256_path(Path(ref.completion_path)) != ref.completion_sha256:
        raise RuntimeError("endpoint completion changed after discovery")
    with np.load(result, allow_pickle=False) as saved:
        if ref.member_index is None:
            frame = np.array(saved["frame"], dtype=np.complex128, copy=True)
            rank = int(np.asarray(saved["rank"]).item())
        else:
            rank = int(np.asarray(saved["ranks"])[ref.member_index])
            frame = np.array(saved["frames"][ref.member_index, :, :rank], dtype=np.complex128, copy=True)
    if frame.shape != (2 * nx * ny, rank) or not np.all(np.isfinite(frame)):
        raise RuntimeError("loaded endpoint frame is invalid")
    return frame


def flux_grid(intervals: int, regulator: float, sigma: int) -> np.ndarray:
    return -sigma * regulator + sigma * np.linspace(0.0, 2.0 * np.pi, intervals + 1)


def _coordinates(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.tile(np.repeat(np.arange(nx, dtype=np.int64), 2), ny)
    y = np.repeat(np.arange(ny, dtype=np.int64), 2 * nx)
    dy = y[:, None] - y[None, :]
    dy = ((dy + ny // 2) % ny) - ny // 2
    return x, y, dy


def twisted_parent(
    h0: np.ndarray, dy: np.ndarray, phi: float, ny: int,
    *, twist_gauge: str = "uniform", y: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the threaded parent in uniform or single-seam gauge."""
    uniform = np.asarray(h0 * np.exp(1j * float(phi) * dy / ny), dtype=np.complex128)
    uniform = 0.5 * (uniform + uniform.conj().T)
    if twist_gauge == "uniform":
        return uniform
    if twist_gauge != "seam" or y is None:
        raise ValueError("twist_gauge must be 'uniform', or 'seam' with y coordinates")
    basis = np.exp(1j * float(phi) * y / ny)
    seam = basis.conj()[:, None] * uniform * basis[None, :]
    return 0.5 * (seam + seam.conj().T)


def _parent_eigensystem(
    h0: np.ndarray, dy: np.ndarray, phi: float, ny: int, y: np.ndarray,
    twist_gauge: str, subset: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize in the requested gauge and return vectors in common uniform gauge."""
    matrix = twisted_parent(h0, dy, phi, ny, twist_gauge=twist_gauge, y=y)
    if subset is None:
        values, vectors = np.linalg.eigh(matrix)
    else:
        values, vectors = scipy.linalg.eigh(
            matrix, subset_by_index=subset, driver="evr",
            check_finite=False, overwrite_a=True,
        )
    if twist_gauge == "seam":
        basis = np.exp(1j * float(phi) * y / ny)
        vectors = basis[:, None] * vectors
    return values, np.asarray(vectors, dtype=np.complex128)


def _frame_observables(frame: np.ndarray, mode_x: np.ndarray, nx: int) -> tuple[float, float, np.ndarray]:
    density = np.real(np.sum(np.abs(frame) ** 2, axis=1))
    density_x = np.bincount(mode_x, weights=density, minlength=nx).astype(np.float64)
    return float(density_x[: nx // 2].sum()), float(density_x[nx // 2 :].sum()), density_x


def _projector_observables(projector: np.ndarray, mode_x: np.ndarray, nx: int) -> tuple[float, float, np.ndarray]:
    density_x = np.bincount(mode_x, weights=np.real(np.diag(projector)), minlength=nx)
    return float(density_x[: nx // 2].sum()), float(density_x[nx // 2 :].sum()), density_x


def _ordinary_select(
    previous: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray, rank: int
) -> tuple[np.ndarray, float, float]:
    weights = np.real(np.sum(np.abs(previous.conj().T @ eigenvectors) ** 2, axis=0))
    order = np.lexsort((eigenvalues, -weights))
    selected = np.asarray(eigenvectors[:, order[:rank]], dtype=np.complex128)
    singular = np.linalg.svd(previous.conj().T @ selected, compute_uv=False)
    return selected, float(np.min(singular)), float(np.min(weights[order[:rank]]))


def _spectator_select(
    previous: np.ndarray, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
    rank: int, edge_indices: tuple[int, int],
) -> np.ndarray:
    keep = np.ones(eigenvalues.size, dtype=bool)
    keep[list(edge_indices)] = False
    candidates = eigenvectors[:, keep]
    values = eigenvalues[keep]
    weights = np.real(np.sum(np.abs(previous.conj().T @ candidates) ** 2, axis=0))
    order = np.lexsort((values, -weights))
    return np.asarray(candidates[:, order[:rank]], dtype=np.complex128)


def _wall_modes(
    cluster: np.ndarray, b_diagonal: np.ndarray,
    previous_cluster: np.ndarray | None, previous_modes: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = np.asarray(cluster, dtype=np.complex128)
    if previous_cluster is not None:
        u, _, vh = np.linalg.svd(previous_cluster.conj().T @ aligned)
        aligned = aligned @ (vh.conj().T @ u.conj().T)
    small_b = aligned.conj().T @ (b_diagonal[:, None] * aligned)
    values, rotation = np.linalg.eigh(0.5 * (small_b + small_b.conj().T))
    modes = np.asarray(aligned @ rotation, dtype=np.complex128)
    if previous_modes is not None:
        for index in range(modes.shape[1]):
            overlap = np.vdot(previous_modes[:, index], modes[:, index])
            if abs(overlap) > 0:
                modes[:, index] *= np.exp(-1j * np.angle(overlap))
    return values.astype(np.float64), modes


def _periodic_distance(x: np.ndarray, center: int, nx: int) -> np.ndarray:
    direct = np.abs(x - int(center))
    return np.minimum(direct, nx - direct)


def _edge_scan(
    h0: np.ndarray, dy: np.ndarray, phi: np.ndarray, rank: int, nx: int, ny: int,
    y: np.ndarray, b_diagonal: np.ndarray, wall_mask: np.ndarray,
    edge_block_rank: int, twist_gauge: str,
) -> dict[str, np.ndarray | int | bool | str]:
    count = phi.size
    internal = np.empty(count)
    external = np.empty(count)
    block_rank = int(edge_block_rank)
    occupied_in_block = block_rank // 2
    b_values = np.empty((count, block_rank))
    wall_weights = np.empty((count, block_rank))
    links = np.ones(count)
    previous_cluster: np.ndarray | None = None
    for point, value in enumerate(phi):
        lower = rank - occupied_in_block - 1
        upper = rank + occupied_in_block
        values, vectors = _parent_eigensystem(
            h0, dy, float(value), ny, y, twist_gauge, (lower, upper),
        )
        internal[point] = float(values[occupied_in_block + 1] - values[occupied_in_block])
        external[point] = float(min(values[1] - values[0], values[-1] - values[-2]))
        cluster = np.asarray(vectors[:, 1:-1], dtype=np.complex128)
        b_values[point], modes = _wall_modes(cluster, b_diagonal, previous_cluster, None)
        wall_weights[point] = np.real(np.sum(np.abs(modes[wall_mask, :]) ** 2, axis=0))
        if previous_cluster is not None:
            links[point] = float(np.min(np.linalg.svd(previous_cluster.conj().T @ cluster, compute_uv=False)))
        previous_cluster = cluster
    neighbor_links = links.copy()
    neighbor_links[:-1] = np.minimum(neighbor_links[:-1], links[1:])
    return {
        "internal": internal, "external": external, "b_values": b_values,
        "wall_weights": wall_weights, "links": links, "neighbor_links": neighbor_links,
    }


def _qualify_active_region(scan: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    edge = config["wall_diabatization"]
    internal = np.asarray(scan["internal"])
    external = np.asarray(scan["external"])
    b_values = np.asarray(scan["b_values"])
    wall_weights = np.asarray(scan["wall_weights"])
    neighbor = np.asarray(scan["neighbor_links"])
    minimum = int(np.argmin(internal))
    point_valid = (
        (external >= float(edge["minimum_external_gap"]))
        & (external > internal)
        & (b_values[:, 0] <= float(edge["maximum_left_B_eigenvalue"]))
        & (b_values[:, -1] >= float(edge["minimum_right_B_eigenvalue"]))
        & (np.min(wall_weights[:, [0, -1]], axis=1) >= float(edge["minimum_combined_wall_weight"]))
        & (neighbor >= float(edge["minimum_neighboring_cluster_overlap"]))
    )
    active = np.zeros_like(point_valid)
    reasons: list[str] = []
    if not point_valid[minimum]:
        if external[minimum] < float(edge["minimum_external_gap"]):
            reasons.append("external_gap_below_0.1")
        if external[minimum] <= internal[minimum]:
            reasons.append("edge_pair_not_isolated")
        if b_values[minimum, 0] > float(edge["maximum_left_B_eigenvalue"]):
            reasons.append("left_wall_character_below_threshold")
        if b_values[minimum, -1] < float(edge["minimum_right_B_eigenvalue"]):
            reasons.append("right_wall_character_below_threshold")
        if np.min(wall_weights[minimum, [0, -1]]) < float(edge["minimum_combined_wall_weight"]):
            reasons.append("radius2_wall_weight_below_threshold")
        if neighbor[minimum] < float(edge["minimum_neighboring_cluster_overlap"]):
            reasons.append("neighboring_cluster_overlap_below_threshold")
        return {
            "resolved": False, "reason": ";".join(reasons) or "minimum_gap_point_unresolved",
            "minimum": minimum, "start": -1, "end": -1, "active": active,
            "point_valid": point_valid,
        }
    start = minimum
    while start > 0 and point_valid[start - 1]:
        start -= 1
    end = minimum
    while end + 1 < point_valid.size and point_valid[end + 1]:
        end += 1
    active[start : end + 1] = True
    return {
        "resolved": True, "reason": "", "minimum": minimum,
        "start": start, "end": end, "active": active, "point_valid": point_valid,
    }


def _branch_path(
    frame: np.ndarray, h0: np.ndarray, dy: np.ndarray, phi: np.ndarray,
    rank: int, nx: int, ny: int, mode_x: np.ndarray, y: np.ndarray,
    b_diagonal: np.ndarray, qualification: dict[str, Any],
    edge_block_rank: int, twist_gauge: str, queue: Any = None,
) -> dict[str, Any]:
    count = phi.size
    left = np.empty(count)
    right = np.empty(count)
    density_x = np.empty((count, nx))
    ordinary_left = np.empty(count)
    ordinary_right = np.empty(count)
    instant_left = np.empty(count)
    instant_right = np.empty(count)
    principal = np.empty(count)
    weight_floor = np.empty(count)
    ordinary_principal = np.empty(count)
    ordinary_weight_floor = np.empty(count)
    projector_residual = np.empty(count)
    charge_residual = np.empty(count)
    previous_main = np.asarray(frame, dtype=np.complex128)
    previous_ordinary = np.asarray(frame, dtype=np.complex128)
    previous_spectator: np.ndarray | None = None
    previous_cluster: np.ndarray | None = None
    previous_modes: np.ndarray | None = None
    occupied_in_block = int(edge_block_rank) // 2
    entering_labels: np.ndarray | None = None
    start_main: np.ndarray | None = None
    endpoint_main: np.ndarray | None = None
    endpoint_ordinary: np.ndarray | None = None
    endpoint_instant: np.ndarray | None = None
    active = np.asarray(qualification["active"], dtype=bool)
    for point, value in enumerate(phi):
        eigenvalues, eigenvectors = _parent_eigensystem(
            h0, dy, float(value), ny, y, twist_gauge,
        )
        ordinary, ordinary_principal[point], ordinary_weight_floor[point] = _ordinary_select(
            previous_ordinary, eigenvalues, eigenvectors, rank
        )
        instantaneous = np.asarray(eigenvectors[:, :rank], dtype=np.complex128)
        if active[point]:
            edge_indices = tuple(range(rank - occupied_in_block, rank + occupied_in_block))
            cluster = np.asarray(eigenvectors[:, list(edge_indices)], dtype=np.complex128)
            _, modes = _wall_modes(cluster, b_diagonal, previous_cluster, previous_modes)
            reference = previous_main if previous_spectator is None else previous_spectator
            spectator = _spectator_select(
                reference, eigenvalues, eigenvectors, rank - occupied_in_block, edge_indices
            )
            if entering_labels is None:
                occupations = np.real(np.sum(np.abs(previous_main.conj().T @ modes) ** 2, axis=0))
                entering_labels = np.argsort(-occupations, kind="stable")[:occupied_in_block]
                entering_labels = np.sort(entering_labels)
            main = np.column_stack((spectator, modes[:, entering_labels]))
            previous_spectator, previous_cluster, previous_modes = spectator, cluster, modes
            singular = np.linalg.svd(previous_main.conj().T @ main, compute_uv=False)
            weights = np.real(np.sum(np.abs(previous_main.conj().T @ main) ** 2, axis=0))
            principal[point], weight_floor[point] = float(np.min(singular)), float(np.min(weights))
        else:
            main, principal[point], weight_floor[point] = _ordinary_select(
                previous_main, eigenvalues, eigenvectors, rank
            )
            previous_spectator = previous_cluster = previous_modes = None
        left[point], right[point], density_x[point] = _frame_observables(main, mode_x, nx)
        ordinary_left[point], ordinary_right[point], _ = _frame_observables(ordinary, mode_x, nx)
        instant_left[point], instant_right[point], _ = _frame_observables(instantaneous, mode_x, nx)
        gram = main.conj().T @ main
        projector_residual[point] = float(np.max(np.abs(gram - np.eye(rank))))
        charge_residual[point] = abs(left[point] + right[point] - rank)
        previous_main, previous_ordinary = main, ordinary
        start_main = main if point == 0 else start_main
        endpoint_main, endpoint_ordinary, endpoint_instant = main, ordinary, instantaneous
        if queue is not None:
            queue.put(1)
    assert start_main is not None and endpoint_main is not None
    assert endpoint_ordinary is not None and endpoint_instant is not None
    return {
        "left": left, "right": right, "density_x": density_x,
        "ordinary_left": ordinary_left, "ordinary_right": ordinary_right,
        "instant_left": instant_left, "instant_right": instant_right,
        "principal": principal, "weight_floor": weight_floor,
        "ordinary_principal": ordinary_principal,
        "ordinary_weight_floor": ordinary_weight_floor,
        "projector_residual": projector_residual, "charge_residual": charge_residual,
        "entering_labels": (
            np.full(occupied_in_block, -1, dtype=np.int8)
            if entering_labels is None else entering_labels.astype(np.int8)
        ),
        "start_main": start_main,
        "endpoint_main": endpoint_main,
        "endpoint_ordinary": endpoint_ordinary, "endpoint_instant": endpoint_instant,
    }


def _partition_indices(nx: int, ny: int, xref: int, yref: int, radius: float) -> tuple[np.ndarray, ...]:
    xs = np.arange(nx)
    ys = np.arange(ny)
    dx = (xs - xref + nx // 2) % nx - nx // 2
    ddy = (ys - yref + ny // 2) % ny - ny // 2
    dx_grid, dy_grid = np.meshgrid(dx, ddy, indexing="ij")
    inside = dx_grid * dx_grid + dy_grid * dy_grid <= radius * radius
    theta = np.mod(np.arctan2(dy_grid, dx_grid), 2.0 * np.pi)
    bounds = (0.0, 2 * np.pi / 3, 4 * np.pi / 3, 2 * np.pi)
    result: list[np.ndarray] = []
    for index in range(3):
        mask = inside & (theta >= bounds[index]) & (theta < bounds[index + 1])
        xx, yy = np.nonzero(mask)
        first = 2 * xx + 2 * nx * yy
        result.append(np.sort(np.concatenate((first, first + 1))).astype(np.int64))
    if min(map(len, result)) == 0:
        raise RuntimeError("source Chern partition has an empty wedge")
    return tuple(result)


def real_space_chern_by_y0(frame: np.ndarray, nx: int, ny: int, radius: float) -> np.ndarray:
    """Exact legacy three-wedge estimator, evaluated from the occupied frame."""
    occupied = np.asarray(frame, dtype=np.complex128).conj()
    values = np.empty(ny)
    for yref in range(ny):
        a, b, c = _partition_indices(nx, ny, nx // 2, yref, radius)
        wa, wb, wc = occupied[a], occupied[b], occupied[c]
        pca, pab, pbc = wc @ wa.conj().T, wa @ wb.conj().T, wb @ wc.conj().T
        pac, pcb, pba = wa @ wc.conj().T, wc @ wb.conj().T, wb @ wa.conj().T
        first = np.trace(pca @ pab @ pbc)
        second = np.trace(pac @ pcb @ pba)
        values[yref] = float(np.real(12.0 * np.pi * 1j * (first - second)))
    return values


def _defect_diagnostics(
    endpoint_frame: np.ndarray, projector0: np.ndarray, sigma: int,
    mode_x: np.ndarray, y: np.ndarray, nx: int, ny: int,
) -> dict[str, Any]:
    gauge = np.exp(1j * sigma * 2.0 * np.pi * y / ny)
    projector_end = endpoint_frame @ endpoint_frame.conj().T
    unwrapped = gauge.conj()[:, None] * projector_end * gauge[None, :]
    defect = 0.5 * (unwrapped - projector0 + (unwrapped - projector0).conj().T)
    values, vectors = np.linalg.eigh(defect)
    positive = values > 0.0
    negative = values < 0.0
    particle_mode = np.sum(np.abs(vectors[:, positive]) ** 2 * values[positive][None, :], axis=1)
    hole_mode = np.sum(np.abs(vectors[:, negative]) ** 2 * (-values[negative])[None, :], axis=1)
    particle_x = np.bincount(mode_x, weights=particle_mode, minlength=nx)
    hole_x = np.bincount(mode_x, weights=hole_mode, minlength=nx)
    leading_positive = int(np.argmax(values))
    leading_negative = int(np.argmin(values))
    leading_particle_x = np.bincount(
        mode_x, weights=np.abs(vectors[:, leading_positive]) ** 2, minlength=nx
    )
    leading_hole_x = np.bincount(
        mode_x, weights=np.abs(vectors[:, leading_negative]) ** 2, minlength=nx
    )
    wall_centers = (nx // 4, 3 * nx // 4)
    wall_masks_x = np.stack([
        _periodic_distance(np.arange(nx), center, nx) <= 2 for center in wall_centers
    ])
    particle_wall_weights = wall_masks_x @ leading_particle_x
    hole_wall_weights = wall_masks_x @ leading_hole_x
    remaining = np.delete(values, (leading_negative, leading_positive))
    _, _, defect_density_x = _projector_observables(defect, mode_x, nx)
    cuts = np.arange(1, nx, dtype=np.int64)
    multicut = np.asarray([
        0.5 * (defect_density_x[cut:].sum() - defect_density_x[:cut].sum()) for cut in cuts
    ])
    centered_x = np.arange(nx, dtype=np.float64) - 0.5 * (nx - 1)
    return {
        "eigenvalues": values.astype(np.float64),
        "particle_density_x": particle_x.astype(np.float64),
        "hole_density_x": hole_x.astype(np.float64),
        "leading_positive_eigenvalue": float(values[leading_positive]),
        "leading_negative_eigenvalue": float(values[leading_negative]),
        "leading_particle_mode_density_x": leading_particle_x.astype(np.float64),
        "leading_hole_mode_density_x": leading_hole_x.astype(np.float64),
        "leading_particle_wall_weights": particle_wall_weights.astype(np.float64),
        "leading_hole_wall_weights": hole_wall_weights.astype(np.float64),
        "positive_count_above_0p9": int(np.count_nonzero(values > 0.9)),
        "negative_count_below_minus_0p9": int(np.count_nonzero(values < -0.9)),
        "maximum_remaining_abs_eigenvalue": float(np.max(np.abs(remaining), initial=0.0)),
        "multicut_positions": cuts,
        "multicut_q_x": multicut,
        "center_displacement": float(centered_x @ defect_density_x),
        "defect": defect,
    }


def compute_wall_diabatic_pump(
    frame: np.ndarray, task: dict[str, Any], config: dict[str, Any],
    *, grid_intervals: int | None = None, queue: Any = None,
) -> dict[str, Any]:
    nx, ny = int(task["Nx"]), int(task["Ny"])
    dimension, rank = frame.shape
    if dimension != 2 * nx * ny or not (2 <= rank <= dimension - 2) or frame.dtype != np.complex128:
        raise RuntimeError("endpoint frame has invalid dimension, rank, or dtype")
    gram = float(np.max(np.abs(frame.conj().T @ frame - np.eye(rank))))
    if gram > float(config["acceptance"]["input_frame_gram_tolerance"]):
        raise RuntimeError(f"endpoint frame Gram residual failed: {gram:.3e}")
    projector0 = frame @ frame.conj().T
    input_projector_residual = float(np.max(np.abs(projector0 @ projector0 - projector0)))
    if input_projector_residual > float(config["acceptance"]["projector_idempotency_tolerance"]):
        raise RuntimeError(f"endpoint projector residual failed: {input_projector_residual:.3e}")
    h0 = np.eye(dimension, dtype=np.complex128) - 2.0 * projector0
    mode_x, y, dy = _coordinates(nx, ny)
    b_diagonal = np.where(mode_x < nx // 2, -1.0, 1.0)
    wall_centers = (nx // 4, 3 * nx // 4)
    radius = int(task.get("wall_window", config["wall_diabatization"]["wall_window_radius"]))
    wall_mask = (
        (_periodic_distance(mode_x, wall_centers[0], nx) <= radius)
        | (_periodic_distance(mode_x, wall_centers[1], nx) <= radius)
    )
    source_left, source_right, source_density_x = _frame_observables(frame, mode_x, nx)
    intervals = int(
        task.get("grid_intervals", config["flux"]["grid_intervals"])
        if grid_intervals is None else grid_intervals
    )
    regulator = float(config["flux"]["regulator"])
    edge_block_rank = int(task.get("edge_block_rank", 2))
    if edge_block_rank not in (2, 4) or edge_block_rank % 2:
        raise ValueError("edge_block_rank must be 2 or 4")
    twist_gauge = str(task.get("twist_gauge", config["flux"]["twist_gauge"]))
    if twist_gauge not in ("uniform", "seam"):
        raise ValueError("twist_gauge must be uniform or seam")
    c_by_y0 = real_space_chern_by_y0(frame, nx, ny, float(config["chern_control"]["radius"]))
    direction_payloads: list[dict[str, Any]] = []
    for direction in DIRECTIONS:
        sigma = int(config["flux"]["directions"][direction])
        phi = flux_grid(intervals, regulator, sigma)
        scan = _edge_scan(
            h0, dy, phi, rank, nx, ny, y, b_diagonal, wall_mask,
            edge_block_rank, twist_gauge,
        )
        qualification = _qualify_active_region(scan, config)
        branch = _branch_path(
            frame, h0, dy, phi, rank, nx, ny, mode_x, y, b_diagonal,
            qualification, edge_block_rank, twist_gauge, queue,
        )
        reverse_qualification = dict(qualification)
        reverse_qualification["active"] = np.asarray(qualification["active"])[::-1]
        reverse = _branch_path(
            branch["endpoint_main"], h0, dy, phi[::-1], rank, nx, ny,
            mode_x, y, b_diagonal, reverse_qualification, edge_block_rank,
            twist_gauge, queue,
        )
        reverse_projector = reverse["endpoint_main"] @ reverse["endpoint_main"].conj().T
        forward_start_projector = branch["start_main"] @ branch["start_main"].conj().T
        undo_error = float(np.max(np.abs(reverse_projector - forward_start_projector)))
        endpoint = _defect_diagnostics(
            branch["endpoint_main"], projector0, sigma, mode_x, y, nx, ny
        )
        h_start = twisted_parent(
            h0, dy, float(phi[0]), ny, twist_gauge=twist_gauge, y=y
        )
        h_end = twisted_parent(
            h0, dy, float(phi[-1]), ny, twist_gauge=twist_gauge, y=y
        )
        gauge = np.exp(1j * sigma * 2.0 * np.pi * y / ny)
        gauged_start = (
            gauge[:, None] * h_start * gauge.conj()[None, :]
            if twist_gauge == "uniform" else h_start
        )
        parent_error = float(np.max(np.abs(h_end - gauged_start)))
        dl = branch["left"] - source_left
        dr = branch["right"] - source_right
        odl = branch["ordinary_left"] - source_left
        odr = branch["ordinary_right"] - source_right
        idl = branch["instant_left"] - source_left
        idr = branch["instant_right"] - source_right
        direction_payloads.append({
            "phi": phi,
            "N_left": branch["left"], "N_right": branch["right"],
            "delta_N_left": dl, "delta_N_right": dr,
            "density_x": branch["density_x"],
            "ordinary_delta_N_left": odl, "ordinary_delta_N_right": odr,
            "instantaneous_delta_N_left": idl, "instantaneous_delta_N_right": idr,
            "principal_overlap": branch["principal"],
            "selected_weight_floor": branch["weight_floor"],
            "ordinary_principal_overlap": branch["ordinary_principal"],
            "ordinary_selected_weight_floor": branch["ordinary_weight_floor"],
            "projector_residual": branch["projector_residual"],
            "total_charge_residual": branch["charge_residual"],
            "edge_internal_gap": scan["internal"], "edge_external_gap": scan["external"],
            "edge_B_eigenvalues": scan["b_values"],
            "edge_combined_wall_weight": scan["wall_weights"],
            "edge_link_min_singular": scan["neighbor_links"],
            "edge_active_mask": qualification["active"],
            "edge_point_valid": qualification["point_valid"],
            "resolved": qualification["resolved"], "reason": qualification["reason"],
            "minimum": qualification["minimum"], "start": qualification["start"],
            "end": qualification["end"], "entering_labels": branch["entering_labels"],
            "parent_error": parent_error, "undo_error": undo_error,
            "endpoint": endpoint,
        })
    stack = lambda key: np.stack([np.asarray(row[key]) for row in direction_payloads])
    n_left, n_right = stack("N_left"), stack("N_right")
    dl, dr = stack("delta_N_left"), stack("delta_N_right")
    odl, odr = stack("ordinary_delta_N_left"), stack("ordinary_delta_N_right")
    idl, idr = stack("instantaneous_delta_N_left"), stack("instantaneous_delta_N_right")
    arrays: dict[str, Any] = {
        "schema": np.asarray(RESULT_SCHEMA),
        "directions": np.asarray(DIRECTIONS),
        "sigma": np.asarray([1, -1], dtype=np.int8),
        "phi": stack("phi"),
        "N_left": n_left, "N_right": n_right, "N_total": n_left + n_right,
        "delta_N_left": dl, "delta_N_right": dr, "delta_N_total": dl + dr,
        "q_x": 0.5 * (dr - dl), "density_x": stack("density_x"),
        "ordinary_delta_N_left": odl, "ordinary_delta_N_right": odr,
        "ordinary_delta_N_total": odl + odr, "ordinary_q_x": 0.5 * (odr - odl),
        "instantaneous_delta_N_left": idl, "instantaneous_delta_N_right": idr,
        "instantaneous_delta_N_total": idl + idr,
        "instantaneous_q_x": 0.5 * (idr - idl),
        "source_N_left": np.asarray(source_left), "source_N_right": np.asarray(source_right),
        "source_density_x": source_density_x,
        "source_real_space_chern_by_y0": c_by_y0,
        "source_real_space_chern_mean": np.asarray(float(np.mean(c_by_y0))),
        "source_real_space_chern_std": np.asarray(float(np.std(c_by_y0, ddof=1))),
        "source_real_space_chern_xref": np.asarray(nx // 2, dtype=np.int64),
        "source_real_space_chern_radius": np.asarray(float(config["chern_control"]["radius"])),
        "edge_internal_gap": stack("edge_internal_gap"),
        "edge_external_gap": stack("edge_external_gap"),
        "edge_B_eigenvalues": stack("edge_B_eigenvalues"),
        "edge_combined_wall_weight": stack("edge_combined_wall_weight"),
        "edge_link_min_singular": stack("edge_link_min_singular"),
        "edge_active_mask": stack("edge_active_mask").astype(bool),
        "edge_point_valid": stack("edge_point_valid").astype(bool),
        "edge_minimum_gap_index": np.asarray([row["minimum"] for row in direction_payloads], dtype=np.int64),
        "edge_active_start_index": np.asarray([row["start"] for row in direction_payloads], dtype=np.int64),
        "edge_active_end_index": np.asarray([row["end"] for row in direction_payloads], dtype=np.int64),
        "edge_entering_wall_label": np.asarray([
            row["entering_labels"][0] for row in direction_payloads
        ], dtype=np.int8),
        "edge_entering_wall_labels": np.stack([
            row["entering_labels"] for row in direction_payloads
        ]).astype(np.int8),
        "resolved": np.asarray([row["resolved"] for row in direction_payloads], dtype=bool),
        "unresolved_reason": np.asarray([row["reason"] for row in direction_payloads]),
        "principal_overlap": stack("principal_overlap"),
        "selected_weight_floor": stack("selected_weight_floor"),
        "ordinary_principal_overlap": stack("ordinary_principal_overlap"),
        "ordinary_selected_weight_floor": stack("ordinary_selected_weight_floor"),
        "endpoint_defect_eigenvalues": np.stack([row["endpoint"]["eigenvalues"] for row in direction_payloads]),
        "endpoint_particle_density_x": np.stack([row["endpoint"]["particle_density_x"] for row in direction_payloads]),
        "endpoint_hole_density_x": np.stack([row["endpoint"]["hole_density_x"] for row in direction_payloads]),
        "endpoint_leading_positive_eigenvalue": np.asarray([
            row["endpoint"]["leading_positive_eigenvalue"] for row in direction_payloads
        ]),
        "endpoint_leading_negative_eigenvalue": np.asarray([
            row["endpoint"]["leading_negative_eigenvalue"] for row in direction_payloads
        ]),
        "endpoint_leading_particle_mode_density_x": np.stack([
            row["endpoint"]["leading_particle_mode_density_x"] for row in direction_payloads
        ]),
        "endpoint_leading_hole_mode_density_x": np.stack([
            row["endpoint"]["leading_hole_mode_density_x"] for row in direction_payloads
        ]),
        "endpoint_leading_particle_wall_weights": np.stack([
            row["endpoint"]["leading_particle_wall_weights"] for row in direction_payloads
        ]),
        "endpoint_leading_hole_wall_weights": np.stack([
            row["endpoint"]["leading_hole_wall_weights"] for row in direction_payloads
        ]),
        "endpoint_positive_defect_count_above_0p9": np.asarray([
            row["endpoint"]["positive_count_above_0p9"] for row in direction_payloads
        ], dtype=np.int8),
        "endpoint_negative_defect_count_below_minus_0p9": np.asarray([
            row["endpoint"]["negative_count_below_minus_0p9"] for row in direction_payloads
        ], dtype=np.int8),
        "endpoint_maximum_remaining_abs_defect_eigenvalue": np.asarray([
            row["endpoint"]["maximum_remaining_abs_eigenvalue"] for row in direction_payloads
        ]),
        "multicut_positions": direction_payloads[0]["endpoint"]["multicut_positions"],
        "multicut_q_x": np.stack([row["endpoint"]["multicut_q_x"] for row in direction_payloads]),
        "center_of_charge_displacement": np.asarray([row["endpoint"]["center_displacement"] for row in direction_payloads]),
        "total_charge_residual": stack("total_charge_residual"),
        "projector_residual": stack("projector_residual"),
        "input_frame_gram_residual": np.asarray(gram),
        "input_projector_residual": np.asarray(input_projector_residual),
        "large_gauge_parent_error": np.asarray([row["parent_error"] for row in direction_payloads]),
        "continuation_undo_error": np.asarray([row["undo_error"] for row in direction_payloads]),
        "rank": np.asarray(rank, dtype=np.int64),
    }
    near_unit = np.abs(arrays["q_x"][:, -1]) > 0.9
    single_pair = (
        (arrays["endpoint_positive_defect_count_above_0p9"] == 1)
        & (arrays["endpoint_negative_defect_count_below_minus_0p9"] == 1)
        & (arrays["endpoint_maximum_remaining_abs_defect_eigenvalue"] < 1e-6)
    )
    particle_wall = np.asarray(arrays["endpoint_leading_particle_wall_weights"])
    hole_wall = np.asarray(arrays["endpoint_leading_hole_wall_weights"])
    localized_opposite = (
        (np.max(particle_wall, axis=1) > 0.8)
        & (np.max(hole_wall, axis=1) > 0.8)
        & (np.argmax(particle_wall, axis=1) != np.argmax(hole_wall, axis=1))
    )
    arrays["endpoint_near_unit_event"] = near_unit
    arrays["endpoint_single_defect_pair"] = single_pair
    arrays["endpoint_defect_modes_opposite_wall_localized"] = localized_opposite
    arrays["endpoint_near_unit_defect_validation_pass"] = (~near_unit) | (single_pair & localized_opposite)
    validate_result_arrays(arrays, task, config, intervals)
    return arrays


def validate_result_arrays(
    arrays: dict[str, Any], task: dict[str, Any], config: dict[str, Any], intervals: int | None = None
) -> None:
    count = int(intervals or task["grid_intervals"]) + 1
    history = (
        "phi", "N_left", "N_right", "N_total", "delta_N_left", "delta_N_right",
        "delta_N_total", "q_x", "ordinary_delta_N_left", "ordinary_delta_N_right",
        "ordinary_delta_N_total", "ordinary_q_x", "instantaneous_delta_N_left",
        "instantaneous_delta_N_right", "instantaneous_delta_N_total", "instantaneous_q_x",
        "edge_internal_gap", "edge_external_gap", "edge_link_min_singular",
        "principal_overlap", "selected_weight_floor", "ordinary_principal_overlap",
        "ordinary_selected_weight_floor", "total_charge_residual", "projector_residual",
    )
    for key in history:
        value = np.asarray(arrays[key])
        if value.shape != (2, count) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid directional history {key}: {value.shape}")
    nx, ny = int(task["Nx"]), int(task["Ny"])
    dimension = 2 * nx * ny
    density_x = np.asarray(arrays["density_x"])
    if density_x.shape != (2, count, nx) or not np.all(np.isfinite(density_x)):
        raise RuntimeError("density_x shape mismatch")
    edge_rank = int(task.get("edge_block_rank", 2))
    edge_b = np.asarray(arrays["edge_B_eigenvalues"])
    if edge_b.shape != (2, count, edge_rank) or not np.all(np.isfinite(edge_b)):
        raise RuntimeError("edge_B_eigenvalues shape mismatch")
    edge_wall = np.asarray(arrays["edge_combined_wall_weight"])
    if edge_wall.shape != (2, count, edge_rank) or not np.all(np.isfinite(edge_wall)):
        raise RuntimeError("edge_combined_wall_weight shape mismatch")
    numeric_shapes = {
        "source_density_x": (nx,),
        "source_real_space_chern_by_y0": (ny,),
        "endpoint_defect_eigenvalues": (2, dimension),
        "endpoint_particle_density_x": (2, nx),
        "endpoint_hole_density_x": (2, nx),
        "endpoint_leading_particle_mode_density_x": (2, nx),
        "endpoint_leading_hole_mode_density_x": (2, nx),
        "endpoint_leading_particle_wall_weights": (2, 2),
        "endpoint_leading_hole_wall_weights": (2, 2),
        "multicut_q_x": (2, nx - 1),
        "center_of_charge_displacement": (2,),
        "large_gauge_parent_error": (2,),
        "continuation_undo_error": (2,),
    }
    for key, shape in numeric_shapes.items():
        value = np.asarray(arrays[key])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid finite diagnostic {key}: {value.shape}")
    scalar_finite = (
        "source_N_left", "source_N_right", "source_real_space_chern_mean",
        "source_real_space_chern_std", "source_real_space_chern_xref",
        "source_real_space_chern_radius", "input_frame_gram_residual",
        "input_projector_residual", "rank",
    )
    if any(np.asarray(arrays[key]).shape != () or not np.isfinite(np.asarray(arrays[key]).item())
           for key in scalar_finite):
        raise RuntimeError("a scalar pump diagnostic is missing or nonfinite")
    if not np.array_equal(np.asarray(arrays["multicut_positions"]), np.arange(1, nx)):
        raise RuntimeError("multi-cut positions changed")
    for key in (
        "edge_active_mask", "edge_point_valid", "resolved",
        "endpoint_near_unit_event", "endpoint_single_defect_pair",
        "endpoint_defect_modes_opposite_wall_localized",
        "endpoint_near_unit_defect_validation_pass",
    ):
        expected_shape = (2, count) if key in {"edge_active_mask", "edge_point_valid"} else (2,)
        if np.asarray(arrays[key]).shape != expected_shape:
            raise RuntimeError(f"boolean diagnostic {key} shape mismatch")
    if np.asarray(arrays["edge_entering_wall_labels"]).shape != (2, edge_rank // 2):
        raise RuntimeError("edge entering-label shape mismatch")
    if not np.array_equal(arrays["directions"], DIRECTIONS) or not np.array_equal(arrays["sigma"], (1, -1)):
        raise RuntimeError("direction axis changed")
    if not np.allclose(arrays["q_x"], 0.5 * (arrays["delta_N_right"] - arrays["delta_N_left"]), atol=1e-12):
        raise RuntimeError("q_x is inconsistent with regional charges")
    acceptance = config["acceptance"]
    if float(arrays["input_frame_gram_residual"]) > float(acceptance["input_frame_gram_tolerance"]):
        raise RuntimeError("input frame violates the Gram gate")
    if float(arrays["input_projector_residual"]) > float(acceptance["projector_idempotency_tolerance"]):
        raise RuntimeError("input projector violates the idempotency gate")
    if float(np.max(arrays["total_charge_residual"])) > float(acceptance["charge_conservation_tolerance"]):
        raise RuntimeError("transported projector violates charge conservation")
    if float(np.max(arrays["projector_residual"])) > float(acceptance["projector_idempotency_tolerance"]):
        raise RuntimeError("transported frame violates projector/idempotency gate")
    if float(np.max(arrays["large_gauge_parent_error"])) > float(acceptance["large_gauge_parent_tolerance"]):
        raise RuntimeError("uniform-twist parent fails large-gauge closure")
    if float(np.max(arrays["continuation_undo_error"])) > float(acceptance["large_gauge_parent_tolerance"]):
        raise RuntimeError("forward/reverse continuation fails the undo gate")


def _metadata(
    task: dict[str, Any], config_hash: str, hashes: dict[str, str], ref: EndpointRef
) -> dict[str, Any]:
    return {**task, "config_hash": config_hash, "source_hashes": hashes,
            "endpoint_dependency": ref.dependency()}


def publish_pair(
    output_root: Path, task: dict[str, Any], arrays: dict[str, Any],
    config_hash: str, hashes: dict[str, str], ref: EndpointRef, elapsed: float,
) -> None:
    result, completion = result_paths(output_root, task)
    metadata = _metadata(task, config_hash, hashes, ref)
    payload = dict(arrays)
    payload["metadata_json"] = np.asarray(canonical_json(metadata))
    _atomic_npz(result, payload)
    record = {"name": result.name, "bytes": result.stat().st_size, "sha256": sha256_path(result)}
    _atomic_json(completion, {
        "schema": COMPLETION_SCHEMA, **metadata, "result": record,
        "elapsed_seconds": float(elapsed), "completed_unix": time.time(),
    })


def verify_pair(
    output_root: Path, task: dict[str, Any], config: dict[str, Any],
    config_hash: str, hashes: dict[str, str], ref: EndpointRef,
) -> tuple[bool, str, dict[str, Any] | None]:
    result, completion_path = result_paths(output_root, task)
    if not result.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair", None
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {"schema": COMPLETION_SCHEMA, **_metadata(task, config_hash, hashes, ref)}
        for key, value in expected.items():
            if completion.get(key) != value:
                return False, f"completion {key} mismatch", None
        record = completion["result"]
        if record.get("name") != result.name or int(record.get("bytes", -1)) != result.stat().st_size:
            return False, "completion result name/size mismatch", None
        if record.get("sha256") != sha256_path(result):
            return False, "result checksum mismatch", None
        with np.load(result, allow_pickle=False) as saved:
            if str(np.asarray(saved["schema"]).item()) != RESULT_SCHEMA:
                return False, "result schema mismatch", None
            embedded = json.loads(str(np.asarray(saved["metadata_json"]).item()))
            if embedded != _metadata(task, config_hash, hashes, ref):
                return False, "embedded result identity mismatch", None
            arrays = {key: np.array(saved[key], copy=True) for key in saved.files if key not in {"schema", "metadata_json"}}
        validate_result_arrays(arrays, task, config)
        return True, "verified", completion
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", None


def _record_failure(output_root: Path, task: dict[str, Any], exc: BaseException) -> None:
    _atomic_json(failure_path(output_root, task), {
        "task_id": task["task_id"], "failed_unix": time.time(),
        "error_type": type(exc).__name__, "message": str(exc),
        "traceback": traceback.format_exc(),
    })


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    task, source, ref, config, output_text, config_hash, hashes, queue = payload
    started = time.perf_counter()
    output_root = Path(output_text)
    log_path = output_root / "logs/tasks" / f"{task['task_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame = load_endpoint(ref, int(task["Nx"]), int(task["Ny"]))
        with threadpool_limits(limits=int(config["execution"]["blas_threads"])):
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                arrays = compute_wall_diabatic_pump(frame, task, config, queue=queue)
        publish_pair(output_root, task, arrays, config_hash, hashes, ref, time.perf_counter() - started)
        failure_path(output_root, task).unlink(missing_ok=True)
        return {"ok": True, "task_id": task["task_id"]}
    except BaseException as exc:
        _record_failure(output_root, task, exc)
        return {"ok": False, "task_id": task["task_id"], "error": f"{type(exc).__name__}: {exc}"}


def _selected(
    rows: Iterable[dict[str, Any]], cells: set[str] | None, walls: set[str] | None,
    sample_ids: set[int] | None,
) -> list[dict[str, Any]]:
    return [row for row in rows if (
        (cells is None or row["cell"] in cells)
        and (walls is None or row["wall"] in walls)
        and (sample_ids is None or int(row["sample_id"]) in sample_ids)
    )]


def _discover(
    selected: list[dict[str, Any]], config: dict[str, Any], new_root: Path,
) -> tuple[dict[str, EndpointRef], dict[str, str]]:
    sources = source_rows(config)
    refs: dict[str, EndpointRef] = {}
    missing: dict[str, str] = {}
    for task in selected:
        try:
            refs[task["task_id"]] = endpoint_ref(
                sources[task["cell"]], task["wall"], int(task["sample_id"]), new_root
            )
        except Exception as exc:
            missing[task["task_id"]] = f"{type(exc).__name__}: {exc}"
    return refs, missing


def write_identity(config: dict[str, Any], output_root: Path) -> None:
    _atomic_json(output_root / "campaign_identity.json", {
        "schema": CAMPAIGN_SCHEMA, "campaign_id": config["campaign_id"],
        "config_hash": scientific_config_hash(config), "source_hashes": source_hashes(),
        "configuration": config,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("report", "run"), nargs="?", default="report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--new-endpoint-root", type=Path, default=DEFAULT_NEW_ENDPOINT_ROOT)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--cells", nargs="*")
    parser.add_argument(
        "--include-bridge", action=argparse.BooleanOptionalAction, default=None,
        help="include the optional 150-pair GPU bridge (default: auto-detect bridge/)",
    )
    parser.add_argument("--walls", nargs="*", choices=("soft", "hard"))
    parser.add_argument("--sample-ids", nargs="*", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    validate_config(config)
    output_root = args.output_root.resolve()
    new_root = args.new_endpoint_root.resolve()
    workers = int(args.workers or config["execution"]["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    include_bridge = (
        bool(args.include_bridge)
        if args.include_bridge is not None
        else (new_root / "bridge").is_dir()
    )
    all_cells = set(source_rows(config, include_bridge=include_bridge))
    cells = None if args.cells is None else set(args.cells)
    walls = None if args.walls is None else set(args.walls)
    sample_ids = None if args.sample_ids is None else set(args.sample_ids)
    if cells is not None and not cells.issubset(all_cells):
        raise ValueError(f"unknown cells: {sorted(cells - all_cells)}")
    if sample_ids is not None and not sample_ids.issubset(set(range(100))):
        raise ValueError("sample IDs must lie in 0,...,99")
    rows = _selected(tasks(config, include_bridge=include_bridge), cells, walls, sample_ids)
    refs, missing = _discover(rows, config, new_root)
    hashes, config_hash = source_hashes(), scientific_config_hash(config)
    verified: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    invalid: dict[str, str] = {}
    for task in rows:
        ref = refs.get(task["task_id"])
        if ref is None:
            continue
        ok, reason, _ = verify_pair(output_root, task, config, config_hash, hashes, ref)
        if ok:
            verified.append(task)
        else:
            pending.append(task)
            if "missing result/completion" not in reason:
                invalid[task["task_id"]] = reason
    print(f"[campaign] {config['campaign_id']}")
    print("[contract] 1,600 primary pairs + optional 150-pair GPU bridge; each contains CCW and CW")
    print(f"[bridge] included={include_bridge}")
    print("[pump] M=256, signed 1e-7 regulator, uniform twist, rank-two wall diabatization")
    print(f"[sources] verified={len(refs)}/{len(rows)} unavailable={len(missing)} new_root={new_root}")
    print(f"[resume] verified={len(verified)}/{len(rows)} pending={len(pending)} invalid={len(invalid)}")
    print(f"[output] {output_root}")
    if missing:
        for task_id, reason in list(missing.items())[:8]:
            print(f"[source unavailable] {task_id}: {reason}")
        if len(missing) > 8:
            print(f"[source unavailable] ... and {len(missing) - 8} more")
    if args.stage == "report":
        return 0
    if missing:
        raise RuntimeError(f"run selection has {len(missing)} unavailable/invalid endpoint sources")
    output_root.mkdir(parents=True, exist_ok=True)
    write_identity(config, output_root)
    run_rows = pending if args.resume else rows
    if not run_rows:
        print("[complete] all selected endpoint tasks are verified")
        return 0
    source_map = source_rows(config)
    context = mp.get_context("spawn")
    points_per_task = 4 * (int(config["flux"]["grid_intervals"]) + 1)
    failures: list[dict[str, Any]] = []
    with context.Manager() as manager:
        queue = manager.Queue()
        with tqdm(total=len(rows), initial=(len(verified) if args.resume else 0), desc="endpoint tasks", unit="task", position=0) as task_bar, \
                tqdm(total=len(rows) * points_per_task,
                     initial=(len(verified) * points_per_task if args.resume else 0),
                     desc="continued flux points", unit="point", position=1) as point_bar, \
                ProcessPoolExecutor(max_workers=min(workers, len(run_rows)), mp_context=context) as pool:
            futures = {
                pool.submit(_worker, (
                    task, source_map[task["cell"]], refs[task["task_id"]], config,
                    str(output_root), config_hash, hashes, queue,
                ))
                for task in run_rows
            }
            while futures:
                done, futures = wait(futures, timeout=0.25, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        point_bar.update(int(queue.get_nowait()))
                    except Exception:
                        break
                for future in done:
                    row = future.result()
                    task_bar.update(1)
                    if not row["ok"]:
                        failures.append(row)
                        task_bar.write(f"[failure] {row['task_id']}: {row['error']}")
            while True:
                try:
                    point_bar.update(int(queue.get_nowait()))
                except Exception:
                    break
    if failures:
        raise RuntimeError(f"{len(failures)} endpoint pump tasks failed; inspect failures/")
    print("[complete] all selected wall-diabatized endpoint tasks finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
