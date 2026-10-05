"""Resumable A100 runner for the ambient log-Gram alpha scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

try:
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu
except ModuleNotFoundError:
    from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu
from log_gram_observer import LogGramObserver, checkpoint_cycles


BUNDLE = "07_log_gram_alpha_scan"
REVISION = "ambient_log_gram_alpha_scan_v1"
ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
HEARTBEAT_SECONDS = 60.0
FORBIDDEN_OUTPUT_KEYS = {
    "tangent_product",
    "covariance",
    "choi",
    "entropy",
    "topology",
    "correlations",
    "transfer_matrix",
}


@contextmanager
def _shard_heartbeat(case_id, shard_index, *, stream=None):
    """Emit newline-delimited liveness updates while one shard is running."""
    interval = float(HEARTBEAT_SECONDS)
    if interval <= 0:
        raise ValueError("HEARTBEAT_SECONDS must be positive")
    stream = sys.stderr if stream is None else stream
    stopped = threading.Event()
    started = time.monotonic()

    def emit():
        while not stopped.wait(interval):
            elapsed = time.monotonic() - started
            tqdm.write(
                f"[{BUNDLE}] heartbeat case={case_id} "
                f"shard={int(shard_index):02d} elapsed={elapsed:.0f}s",
                file=stream,
            )

    thread = threading.Thread(
        target=emit, name=f"{BUNDLE}-heartbeat", daemon=True
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return repr(value)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
        suffix=".partial", delete=False
    ) as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(root_seed: int, *parts: Any) -> int:
    raw = ":".join([str(int(root_seed)), *[str(value) for value in parts]]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


def load_config(bundle_root: Path | str) -> dict[str, Any]:
    config = json.loads((Path(bundle_root) / "production_config.json").read_text())
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("bundle") != BUNDLE or config.get("campaign_revision") != REVISION:
        raise ValueError("bundle identity or revision changed")
    contract = config.get("contract", {})
    expected = {
        "Nx": 20,
        "Ny_values": [20, 30, 40, 50, 60],
        "alpha_1_values": [round(value, 1) for value in np.linspace(1.0, 3.0, 11)],
        "alpha_2": 30.0,
        "initializations": ["maxmix", "pure"],
        "samples_per_case": 25,
        "samples_per_shard": 5,
        "cycles_rule": "2*Ny",
        "checkpoint_rule": "round(linspace(Ny,2*Ny,6))",
        "n_modes": 16,
        "nshell": 1,
        "filling_fraction": 0.5,
        "sequence": "random",
        "perfect_correction": True,
        "postselection": False,
        "ancilla_occupation": 0.5,
        "dtype": "complex128",
        "canonical_entry_point": ENTRY_POINT,
        "choi_tracked": False,
    }
    if contract != expected:
        raise ValueError("resolved campaign contract differs from the approved plan")
    if config.get("protocols") != {
        "hard": {"DW": True, "dw_truncation": True, "meas_slab_only": True},
        "soft": {"DW": True, "dw_truncation": False, "meas_slab_only": False},
    }:
        raise ValueError("hard/soft protocol definitions changed")
    expected_selection = {
        "finite_singular_value_tolerance": 1e-12,
        "degeneracy_tolerance": 64.0 * np.finfo(np.float64).eps,
        "eigenpair_residual_tolerance": 1e-8,
        "eigenvector_gram_tolerance": 1e-8,
        "block_label_encoding": {"mixed": -1, "occupied": 0, "empty": 1},
    }
    if config.get("selection") != expected_selection:
        raise ValueError("finite-mode selection contract changed")


def expand_cases(config: dict[str, Any], *, pilot: bool = False) -> list[dict[str, Any]]:
    validate_config(config)
    contract = config["contract"]
    ny_values = config["pilot"]["Ny_values"] if pilot else contract["Ny_values"]
    alpha_values = (
        config["pilot"]["alpha_1_values"] if pilot else contract["alpha_1_values"]
    )
    samples = config["pilot"]["samples_per_case"] if pilot else contract["samples_per_case"]
    cases = []
    for ny in ny_values:
        for alpha_1 in alpha_values:
            for protocol in ("hard", "soft"):
                for initialization in ("maxmix", "pure"):
                    flags = config["protocols"][protocol]
                    case_id = (
                        f"LG_N20x{int(ny)}_a{float(alpha_1):.1f}_"
                        f"{protocol}_{initialization}"
                    )
                    cases.append(
                        {
                            "case_id": case_id,
                            "case_ordinal": len(cases),
                            "protocol": protocol,
                            "initialization": initialization,
                            "samples": int(samples),
                            "model": {
                                "Nx": 20,
                                "Ny": int(ny),
                                "DW": True,
                                "nshell": 1,
                                "filling_frac": 0.5,
                                "alpha_1": float(alpha_1),
                                "alpha_2": 30.0,
                                "trial_orbitals": "X",
                                "dw_truncation": bool(flags["dw_truncation"]),
                                "dtype": "complex128",
                                "backend": "local",
                            },
                            "run": {
                                "cycles": 2 * int(ny),
                                "sequence": "random",
                                "perfect_correction": True,
                                "postselect": False,
                                "postselect_probability": 0.0,
                                "n_a": 0.5,
                                "meas_slab_only": bool(flags["meas_slab_only"]),
                                "checkpoints": checkpoint_cycles(int(ny)),
                            },
                        }
                    )
    expected = 12 if pilot else 220
    if len(cases) != expected or len({case["case_id"] for case in cases}) != expected:
        raise AssertionError(f"case expansion must contain {expected} unique cases")
    return cases


def production_queue(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    shard_size = int(config["contract"]["samples_per_shard"])
    for case in expand_cases(config):
        if case["samples"] % shard_size:
            raise AssertionError("production samples must divide into immutable shards")
        for shard_index in range(case["samples"] // shard_size):
            rows.append(
                {
                    "queue_index": len(rows),
                    "case_id": case["case_id"],
                    "case_ordinal": case["case_ordinal"],
                    "shard_index": shard_index,
                    "sample_start": shard_index * shard_size,
                    "sample_stop": (shard_index + 1) * shard_size,
                }
            )
    if len(rows) != 1100:
        raise AssertionError("production queue must contain exactly 1,100 shards")
    return rows


class CompactRecordCollector:
    def __init__(
        self, samples: int, cycles: int, sites_per_cycle: int,
        exterior_orbital_count: int = 0
    ):
        self.site_order = np.full(
            (samples, cycles, sites_per_cycle), -1, dtype=np.int16
        )
        self.outcomes = np.zeros(
            (samples, cycles, sites_per_cycle, 4), dtype=np.bool_
        )
        self.exterior_orbital_indices = np.full(
            (int(exterior_orbital_count),), -1, dtype=np.int32
        )
        self.exterior_outcomes = np.zeros(
            (samples, int(exterior_orbital_count)), dtype=np.bool_
        )

    def __call__(
        self, *, cycle, update_index, site_ids, sample_indices,
        outcome_occupied, **_
    ):
        sample_indices = sample_indices.detach().cpu().numpy().astype(np.int64)
        site_ids = site_ids.detach().cpu().numpy().astype(np.int16)
        outcomes = outcome_occupied.detach().cpu().numpy().astype(np.bool_)
        if outcomes.shape[1] != 4:
            raise RuntimeError("campaign requires four OW outcomes per visited site")
        self.site_order[sample_indices, int(cycle) - 1, int(update_index)] = site_ids
        self.outcomes[sample_indices, int(cycle) - 1, int(update_index)] = outcomes

    def exterior_callback(
        self, *, orbital_indices, outcome_occupied, sample_indices, **_
    ):
        indices = orbital_indices.detach().cpu().numpy().astype(np.int32)
        outcomes = outcome_occupied.detach().cpu().numpy().astype(np.bool_)
        sample_indices = sample_indices.detach().cpu().numpy().astype(np.int64)
        if indices.shape != self.exterior_orbital_indices.shape:
            raise RuntimeError("exterior preparation record shape changed")
        if np.any(self.exterior_orbital_indices >= 0):
            np.testing.assert_array_equal(self.exterior_orbital_indices, indices)
        else:
            self.exterior_orbital_indices[:] = indices
        self.exterior_outcomes[sample_indices] = outcomes

    def validate(self):
        if np.any(self.site_order < 0):
            raise RuntimeError("measurement record is incomplete")
        if self.exterior_orbital_indices.size and np.any(
            self.exterior_orbital_indices < 0
        ):
            raise RuntimeError("exterior preparation record is incomplete")

    def arrays(self):
        self.validate()
        packed = np.packbits(self.outcomes.reshape(self.outcomes.shape[0], -1), axis=1)
        return {
            "site_order_records": self.site_order,
            "measurement_outcomes_packed": packed,
            "measurement_outcomes_shape": np.asarray(self.outcomes.shape, dtype=np.int64),
            "exterior_orbital_indices": self.exterior_orbital_indices,
            "exterior_outcomes_packed": np.packbits(
                self.exterior_outcomes, axis=1
            ),
            "exterior_outcomes_shape": np.asarray(
                self.exterior_outcomes.shape, dtype=np.int64
            ),
        }


def _require_runtime(*, allow_cpu: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        if allow_cpu:
            return {"device": "cpu", "allow_cpu": True}
        raise RuntimeError("pilot and production runs require an A100 GPU")
    name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if "A100" not in name.upper() and not allow_cpu:
        raise RuntimeError(f"campaign requires an A100; detected {name!r}")
    return {
        "device": name,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "free_fraction": float(free_bytes / total_bytes),
        "allow_cpu": bool(allow_cpu),
    }


def _memory_preflight(
    *, model, case: dict[str, Any], sample_count: int,
    active_dimension: int, runtime: dict[str, Any]
) -> dict[str, Any]:
    """Conservative admission estimate for the requested A100 batch."""
    complex_bytes = torch.empty((), dtype=torch.complex128).element_size()
    nlayer = int(model.Nlayer)
    active_dimension = int(active_dimension)
    if case["initialization"] == "maxmix":
        batch = min(5, int(sample_count))
        covariance_bytes = batch * nlayer * nlayer * complex_bytes
        checkpoint_bytes = active_dimension * active_dimension * complex_bytes
        estimated_peak = 4 * covariance_bytes + 6 * checkpoint_bytes
        components = {
            "covariance_batch_bytes": covariance_bytes,
            "checkpoint_eigensolver_bytes": 6 * checkpoint_bytes,
        }
    else:
        rank = nlayer // 2
        initial_frames = int(sample_count) * nlayer * rank * complex_bytes
        physical_frame = nlayer * rank * complex_bytes
        tangent_frame = nlayer * active_dimension * complex_bytes
        occupied = active_dimension // 2
        empty = active_dimension - occupied
        core_bytes = (occupied * occupied + empty * empty) * complex_bytes
        stabilized_working_set = physical_frame + tangent_frame + core_bytes
        checkpoint_bytes = active_dimension * active_dimension * complex_bytes
        estimated_peak = (
            initial_frames + 8 * stabilized_working_set + 4 * checkpoint_bytes
        )
        components = {
            "initial_frame_shard_bytes": initial_frames,
            "stabilized_working_set_bytes": 8 * stabilized_working_set,
            "checkpoint_svd_bytes": 4 * checkpoint_bytes,
        }
    admission_fraction = 0.75
    free_bytes = runtime.get("free_bytes")
    admitted = (
        free_bytes is None
        or int(estimated_peak) <= admission_fraction * int(free_bytes)
    )
    result = {
        "formula": "campaign_conservative_peak_v1",
        "estimated_peak_bytes": int(estimated_peak),
        "admission_fraction_of_free_memory": admission_fraction,
        "free_bytes_at_preflight": None if free_bytes is None else int(free_bytes),
        "admitted": bool(admitted),
        "components": components,
    }
    if not admitted:
        raise MemoryError(
            "A100 memory preflight rejected the requested batch: "
            f"estimated_peak={estimated_peak} bytes, free={free_bytes} bytes, "
            f"admission_fraction={admission_fraction}."
        )
    return result


def _source_hashes(bundle_root: Path) -> dict[str, str]:
    names = (
        "classA_U1FGTN_gpu.py",
        "occupied_frame_gpu.py",
        "log_gram_observer.py",
        "log_gram_runner.py",
    )
    return {name: _sha256_file(bundle_root / "src" / name) for name in names}


def _make_pure_frames(model, seeds: np.ndarray) -> torch.Tensor:
    rank = model.Nlayer // 2
    rows = []
    for seed in seeds.tolist():
        generator = torch.Generator(device=model.device)
        generator.manual_seed(int(seed))
        real = torch.randn(
            (model.Nlayer, rank), dtype=torch.float64,
            device=model.device, generator=generator
        )
        imag = torch.randn(
            (model.Nlayer, rank), dtype=torch.float64,
            device=model.device, generator=generator
        )
        q, r = torch.linalg.qr(torch.complex(real, imag), mode="reduced")
        diagonal = torch.diagonal(r)
        phase = torch.where(
            torch.abs(diagonal) > 0.0,
            diagonal / torch.abs(diagonal).clamp_min(torch.finfo(torch.float64).tiny),
            torch.ones_like(diagonal),
        )
        rows.append((q * phase.conj().unsqueeze(0)).unsqueeze(0))
    return torch.cat(rows, dim=0)


def _run_engine(
    *, model, case, sample_count, observer, records, frame_init,
    frozen_schedule=None, frozen_outcomes=None, frozen_exterior_outcomes=None
):
    common = {
        "G_history": False,
        "progress": False,
        "cycles": case["run"]["cycles"],
        "samples": int(sample_count),
        "save": False,
        "save_init": False,
        "return_data": False,
        "sequence": case["run"]["sequence"],
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "n_a": 0.5,
        "meas_slab_only": case["run"]["meas_slab_only"],
        "record_observer": records,
        "exterior_outcome_observer": (
            records.exterior_callback if case["protocol"] == "hard" else None
        ),
        "frozen_schedule": frozen_schedule,
        "frozen_outcomes": frozen_outcomes,
        "frozen_exterior_outcomes": frozen_exterior_outcomes,
        "track_choi": False,
        "choi_observer": None,
    }
    if case["initialization"] == "maxmix":
        return model.run_markov_circuit(
            **common,
            init_mode="maxmix",
            state_representation="covariance",
            batch_size=min(5, int(sample_count)),
            cycle_observer=observer.maxmix_callback,
            cycle_observer_cycles=case["run"]["checkpoints"],
        )
    return model.run_markov_circuit(
        **common,
        init_mode="default",
        frame_init=frame_init,
        state_representation="physical_frame",
        batch_size=1,
        require_no_covariance_materialization=True,
        lyapunov_basis_mode="pure_occupied_empty",
        lyapunov_frame_observer=observer.pure_callback,
        lyapunov_observer_cycles=case["run"]["checkpoints"],
        lyapunov_start_cycle=1,
        lyapunov_track_restricted_core=True,
    )


def validate_shard_arrays(
    arrays: dict[str, np.ndarray], *, sample_count, active_dim,
    residual_tolerance=1e-8, gram_tolerance=1e-8
):
    expected = (int(sample_count), 6, 16)
    if arrays["log_gram_eigenvalues"].shape != expected:
        raise ValueError("log-Gram eigenvalue shape mismatch")
    if arrays["log_gram_eigenvectors"].shape != expected[:2] + (int(active_dim), 16):
        raise ValueError("log-Gram eigenvector shape mismatch")
    if not np.all(np.isfinite(arrays["log_gram_eigenvalues"])):
        raise ValueError("saved log-Gram eigenvalues must be finite")
    if not np.all(np.isfinite(arrays["log_gram_eigenvectors"])):
        raise ValueError("saved log-Gram eigenvectors must be finite")
    if np.max(arrays["eigenpair_residuals"]) > float(residual_tolerance):
        raise ValueError(
            f"eigenpair residual exceeds {float(residual_tolerance):.3e}"
        )
    if np.max(arrays["eigenvector_gram_errors"]) > float(gram_tolerance):
        raise ValueError(
            f"saved eigenvector Gram error exceeds {float(gram_tolerance):.3e}"
        )
    if np.any(arrays["boundary_spectral_gaps"] < -1e-12):
        raise ValueError("boundary spectral gap is negative")
    absolute_values = np.abs(arrays["log_gram_eigenvalues"])
    if np.any(np.diff(absolute_values, axis=-1) < -1e-12):
        raise ValueError("finite modes are not ordered by increasing |h|")
    if np.any(np.sum(arrays["finite_ranks"], axis=-1) < 16):
        raise ValueError("fewer than 16 finite modes were retained")
    forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(arrays)
    if forbidden:
        raise ValueError(f"forbidden state products present: {sorted(forbidden)}")


def _case_output_paths(output_root: Path, case_id: str, shard_index: int):
    directory = output_root / case_id
    stem = f"shard_{int(shard_index):02d}"
    return directory / f"{stem}.npz", directory / f"{stem}.manifest.json"


def _existing_valid(
    data_path: Path, manifest_path: Path, *, expected_identity: dict[str, Any]
) -> bool:
    data_exists = data_path.exists()
    manifest_exists = manifest_path.exists()
    if data_exists != manifest_exists:
        raise RuntimeError(
            "refusing to overwrite an orphan immutable shard artifact: "
            f"data_exists={data_exists}, manifest_exists={manifest_exists}, "
            f"data_path={data_path}"
        )
    if not data_exists:
        return False
    manifest = json.loads(manifest_path.read_text())
    for key, expected_value in expected_identity.items():
        if manifest.get(key) != _json_ready(expected_value):
            raise RuntimeError(
                f"immutable shard identity mismatch for {key!r} at {data_path}"
            )
    expected_checksum = manifest.get("output_sha256")
    if not expected_checksum or _sha256_file(data_path) != expected_checksum:
        raise RuntimeError(f"checksum mismatch for immutable shard {data_path}")
    with np.load(data_path, allow_pickle=False) as archive:
        values = archive["log_gram_eigenvalues"]
        vectors = archive["log_gram_eigenvectors"]
        if values.shape != (
            int(expected_identity["sample_count"]), 6, 16
        ):
            raise RuntimeError("immutable shard eigenvalue shape mismatch")
        if vectors.shape[0:2] != values.shape[0:2] or vectors.shape[-1] != 16:
            raise RuntimeError("immutable shard eigenvector shape mismatch")
    return True


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=f".{path.name}.",
        suffix=".partial", delete=False
    ) as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    checksum = _sha256_file(temporary)
    os.replace(temporary, path)
    if _sha256_file(path) != checksum:
        raise IOError(f"post-write checksum verification failed for {path}")
    return checksum


def run_shard(
    *, bundle_root: Path, config: dict[str, Any], case: dict[str, Any],
    shard_index: int, output_root: Path, sample_count: int | None = None,
    allow_cpu: bool = False, replay_check: bool = False
) -> dict[str, Any]:
    runtime = _require_runtime(allow_cpu=allow_cpu)
    shard_size = int(config["contract"]["samples_per_shard"])
    sample_count = shard_size if sample_count is None else int(sample_count)
    shard_index = int(shard_index)
    if shard_index < 0:
        raise ValueError("shard_index must be nonnegative")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    case_samples = int(case["samples"])
    shard_count = (case_samples + shard_size - 1) // shard_size
    if shard_index >= shard_count:
        raise ValueError(
            f"shard_index must lie in 0..{shard_count - 1} for {case['case_id']}"
        )
    sample_start = shard_index * shard_size
    expected_sample_count = min(shard_size, case_samples - sample_start)
    if sample_count != expected_sample_count:
        raise ValueError(
            f"shard {shard_index} requires {expected_sample_count} samples; "
            f"got {sample_count}"
        )
    data_path, manifest_path = _case_output_paths(
        output_root, case["case_id"], shard_index
    )
    source_hashes = _source_hashes(bundle_root)
    production_config_sha256 = _sha256_file(
        bundle_root / "production_config.json"
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_parameters = dict(case["model"])
    model_parameters["device"] = device
    model = classA_U1FGTN_gpu(**model_parameters)
    active_indices = model.active_top_layer_indices(
        meas_slab_only=case["run"]["meas_slab_only"]
    ).detach().cpu().numpy()
    checkpoints = tuple(case["run"]["checkpoints"])
    active_basis_sha256 = hashlib.sha256(
        np.ascontiguousarray(active_indices).view(np.uint8)
    ).hexdigest()
    memory_preflight = _memory_preflight(
        model=model,
        case=case,
        sample_count=sample_count,
        active_dimension=len(active_indices),
        runtime=runtime,
    )
    expected_identity = {
        "bundle": BUNDLE,
        "campaign_revision": REVISION,
        "root_seed": int(config["root_seed"]),
        "case": case,
        "shard_index": shard_index,
        "sample_start": sample_start,
        "sample_count": sample_count,
        "active_dimension": int(len(active_indices)),
        "active_basis_indices_sha256": active_basis_sha256,
        "source_sha256": source_hashes,
        "production_config_sha256": production_config_sha256,
    }
    if _existing_valid(
        data_path, manifest_path, expected_identity=expected_identity
    ):
        return {"status": "verified_existing", "data_path": str(data_path)}
    sites_per_cycle = len(
        model._sequence_helper(
            case["run"]["sequence"],
            meas_slab_only=case["run"]["meas_slab_only"],
        )["coords_for_len"]
    )
    observer = LogGramObserver(
        checkpoints=checkpoints,
        active_indices=active_indices,
        samples=sample_count,
        arm=case["initialization"],
        n_modes=16,
        singular_tolerance=float(
            config["selection"]["finite_singular_value_tolerance"]
        ),
        degeneracy_tolerance=float(
            config["selection"]["degeneracy_tolerance"]
        ),
    )
    records = CompactRecordCollector(
        sample_count,
        case["run"]["cycles"],
        sites_per_cycle,
        2 * len(model._exterior_site_ids()) if case["protocol"] == "hard" else 0,
    )
    root_seed = int(config["root_seed"])
    trajectory_seed = _stable_seed(root_seed, case["case_id"], shard_index, "born")
    initial_state_seeds = np.asarray(
        [
            _stable_seed(
                root_seed, case["case_id"], shard_index, sample, "initial"
            )
            for sample in range(sample_count)
        ],
        dtype=np.uint64,
    )
    frame_init = (
        _make_pure_frames(model, initial_state_seeds)
        if case["initialization"] == "pure"
        else None
    )
    torch.manual_seed(int(trajectory_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(trajectory_seed))
    started = time.time()
    engine_metadata = _run_engine(
        model=model,
        case=case,
        sample_count=sample_count,
        observer=observer,
        records=records,
        frame_init=frame_init,
    )
    elapsed = time.time() - started
    arrays = observer.arrays(range(sample_count))
    arrays.update(records.arrays())
    arrays.update(
        {
            "checkpoint_cycles": np.asarray(checkpoints, dtype=np.int32),
            "active_basis_indices": np.asarray(active_indices, dtype=np.int32),
            "initial_state_seeds": initial_state_seeds,
            "trajectory_stream_seed": np.asarray(trajectory_seed, dtype=np.uint64),
            "global_sample_ids": np.asarray(
                [
                    int(case["case_ordinal"]) * 25 + sample_start + offset
                    for offset in range(sample_count)
                ],
                dtype=np.int64,
            ),
        }
    )
    validate_shard_arrays(
        arrays,
        sample_count=sample_count,
        active_dim=len(active_indices),
        residual_tolerance=config["selection"]["eigenpair_residual_tolerance"],
        gram_tolerance=config["selection"]["eigenvector_gram_tolerance"],
    )

    replay_status = "not_requested"
    if replay_check:
        replay_observer = LogGramObserver(
            checkpoints=checkpoints,
            active_indices=active_indices,
            samples=sample_count,
            arm=case["initialization"],
            n_modes=16,
            singular_tolerance=float(
                config["selection"]["finite_singular_value_tolerance"]
            ),
            degeneracy_tolerance=float(
                config["selection"]["degeneracy_tolerance"]
            ),
        )
        replay_records = CompactRecordCollector(
            sample_count,
            case["run"]["cycles"],
            sites_per_cycle,
            2 * len(model._exterior_site_ids()) if case["protocol"] == "hard" else 0,
        )
        unpacked = np.unpackbits(
            arrays["measurement_outcomes_packed"], axis=1
        )[:, : np.prod(arrays["measurement_outcomes_shape"][1:])].reshape(
            tuple(arrays["measurement_outcomes_shape"])
        )
        exterior_shape = tuple(arrays["exterior_outcomes_shape"])
        exterior_unpacked = (
            np.unpackbits(arrays["exterior_outcomes_packed"], axis=1)[
                :, : exterior_shape[1]
            ].reshape(exterior_shape)
            if exterior_shape[1]
            else None
        )
        _run_engine(
            model=model,
            case=case,
            sample_count=sample_count,
            observer=replay_observer,
            records=replay_records,
            frame_init=frame_init,
            frozen_schedule=arrays["site_order_records"].astype(np.int64),
            frozen_outcomes=unpacked.astype(np.bool_),
            frozen_exterior_outcomes=(
                None
                if exterior_unpacked is None
                else exterior_unpacked.astype(np.bool_)
            ),
        )
        replay_arrays = replay_observer.arrays(range(sample_count))
        for key in ("log_gram_eigenvalues", "log_gram_eigenvectors"):
            np.testing.assert_allclose(
                replay_arrays[key], arrays[key], rtol=2e-10, atol=2e-10
            )
        replay_status = "passed"

    checksum = _save_npz_atomic(data_path, arrays)
    manifest = {
        "bundle": BUNDLE,
        "campaign_revision": REVISION,
        "canonical_entry_point": ENTRY_POINT,
        "root_seed": root_seed,
        "case": case,
        "shard_index": int(shard_index),
        "sample_start": sample_start,
        "sample_count": sample_count,
        "global_sample_ids": arrays["global_sample_ids"],
        "resolved_parameters": {
            **case["model"],
            **case["run"],
            "batch_size": 1 if case["initialization"] == "pure" else min(5, sample_count),
            "tangent_basis_mode": (
                "pure_occupied_empty" if case["initialization"] == "pure" else None
            ),
            "maxmix_identity": (
                "J_dagger_J=4*C_t*(1-C_t)" if case["initialization"] == "maxmix" else None
            ),
            "choi_tracked": False,
            "choi_observer": False,
            "selection": config["selection"],
        },
        "active_dimension": int(len(active_indices)),
        "active_basis_indices_sha256": active_basis_sha256,
        "exterior_preparation_excluded_from_cocycle": bool(
            case["protocol"] == "hard"
        ),
        "saved_products": sorted(arrays),
        "forbidden_state_products_saved": False,
        "choi_tracked": False,
        "source_sha256": source_hashes,
        "production_config_sha256": production_config_sha256,
        "mode_block_label_encoding": config["selection"]["block_label_encoding"],
        "memory_preflight": memory_preflight,
        "runtime_seconds": float(elapsed),
        "runtime": runtime,
        "host": platform.node(),
        "torch_version": torch.__version__,
        "engine_metadata": _json_ready(engine_metadata),
        "record_replay_validation": replay_status,
        "output_file": data_path.name,
        "output_sha256": checksum,
        "completed_at_unix": time.time(),
    }
    _write_json_atomic(manifest_path, manifest)
    return {
        "status": "completed",
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
        "sha256": checksum,
    }


def _find_case(cases, case_id):
    matches = [case for case in cases if case["case_id"] == case_id]
    if len(matches) != 1:
        raise KeyError(f"unknown case_id {case_id!r}")
    return matches[0]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("pilot", "production", "queue", "preflight")
    )
    parser.add_argument("--case-id")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--output-root", type=Path, default=Path("gpu_data"))
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--replay-check", action="store_true")
    args = parser.parse_args(argv)
    bundle_root = Path(__file__).resolve().parents[1]
    config = load_config(bundle_root)
    if args.mode == "preflight":
        print(json.dumps(_require_runtime(allow_cpu=args.allow_cpu), indent=2))
        return 0
    if args.mode == "queue":
        queue_path = bundle_root / "production_queue.json"
        _write_json_atomic(
            queue_path,
            {
                "bundle": BUNDLE,
                "revision": REVISION,
                "root_seed": config["root_seed"],
                "shard_count": 1100,
                "queue": production_queue(config),
            },
        )
        print(queue_path)
        return 0

    pilot = args.mode == "pilot"
    if args.mode == "production" and args.shard_index is not None:
        if not 0 <= int(args.shard_index) < 5:
            parser.error("--shard-index must be in 0..4 for production")
    if pilot and args.shard_index is not None:
        parser.error("--shard-index is not used in pilot mode")
    cases = expand_cases(config, pilot=pilot)
    if args.case_id is not None:
        cases = [_find_case(cases, args.case_id)]
    jobs = []
    for case in cases:
        if pilot:
            shard_indices = [0]
            sample_count = 2
        else:
            shard_indices = (
                [args.shard_index]
                if args.shard_index is not None
                else list(range(5))
            )
            sample_count = 5
        jobs.extend((case, int(shard_index), sample_count) for shard_index in shard_indices)
    bar = tqdm(
        total=len(jobs),
        desc=f"{BUNDLE} {args.mode}",
        unit="shard",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )
    try:
        for case, shard_index, sample_count in jobs:
            bar.set_postfix_str(
                f"case={case['case_id']} shard={shard_index:02d}", refresh=True
            )
            with _shard_heartbeat(case["case_id"], shard_index):
                result = run_shard(
                    bundle_root=bundle_root,
                    config=config,
                    case=case,
                    shard_index=shard_index,
                    output_root=args.output_root,
                    sample_count=sample_count,
                    allow_cpu=args.allow_cpu,
                    replay_check=args.replay_check,
                )
            if result.get("status") not in {"completed", "verified_existing"}:
                raise RuntimeError(
                    "log-Gram shard did not reach a durable state: " f"{result!r}"
                )
            print(json.dumps(result, sort_keys=True), flush=True)
            bar.update(1)
    finally:
        bar.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
