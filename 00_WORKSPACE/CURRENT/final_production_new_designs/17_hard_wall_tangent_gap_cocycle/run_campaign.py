#!/usr/bin/env python3
"""Two-lane A100 hard-wall pure-tangent gap and endpoint-cocycle campaign."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import site
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "src"
for candidate in (BUNDLE_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from replay_record_observer import (  # noqa: E402
    ReplayRecordObserver,
    native_frame_arrays,
    unpack_boolean_record,
)


BUNDLE = "17_hard_wall_tangent_gap_cocycle"
REVISION = "hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v2"
CONFIG_PATH = BUNDLE_ROOT / "campaign_config.json"
RESULT_SCHEMA = "hard_wall_tangent_gap_cocycle_result_v1"
COMPLETION_SCHEMA = "hard_wall_tangent_gap_cocycle_completion_v1"
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
REUSED_REVISION = "pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
V1_REVISION = "hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v1"
V1_CONFIGURATION_SHA256 = "1437f965c84e72744869c146e1021eeb832a059009c976be22b51a18cb5c271b"
V1_RESULT_SHA256 = "ba7579994dd3864fa5577aefa09588fc4f8efc1bdf200bbac3a0c62b02183e54"
V1_RESULT_BYTES = 563231101
V1_BATCH_SEED = 9217368992356111699
PRE_PADDING_HOTFIX_RUNNER_SHA256 = "65c03f5ae85d8d182819d03db747ab9fd341fc9d0dd89debd0dcbc1bbfb7d335"
PRE_NVRTC_HOTFIX_RUNNER_SHA256 = "15844b51dca8b8389d27fd18bf1911fa1333ee33d64f6ff49670c03094fb5271"
PRE_NVRTC_LAYOUT_HOTFIX_RUNNER_SHA256 = "3489127fc66bbedd57ea323c552ec397e37ff07ca10b6f4ca2a64038b860b88d"
NVRTC_CUDA13_REQUIREMENT = "nvidia-cuda-nvrtc==13.0.88"
NVRTC_REEXEC_MARKER = "CLASSA_NVRTC_RUNTIME_READY"
V1_SOURCE_HASHES = {
    "config": "03b23faea413d5f62357c30f5adf502f237f526421036054463b827115be6ab4",
    "gpu_engine": "53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700",
    "occupied_frame": "bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0",
    "record_helper": "31b418410be43cc3996b0537ae258e08efae3a193ba3376f2d7dc0ae446038f5",
    "runner": "a9818d1629059a88c62b35b499103554eb814b02388f547f9021656f24a6f476",
}
REUSED_CONFIG_SHA256 = "d1a7c212d8f2c06b0f774af14272a2c71b9d0482240f0c4c43be75a9cd494be4"
REUSED_SOURCE_HASHES = {
    "replay_record_observer.py": "31b418410be43cc3996b0537ae258e08efae3a193ba3376f2d7dc0ae446038f5",
    "run_campaign.py": "71e01f437b7f3e3055922bdb20828add826fc2d7ae751044f639e9d027ed448b",
    "src/classA_U1FGTN_gpu.py": "53de96bced6839b485afe04fe4aaa15d2e42c249a9cf14f6ca0c931f55409700",
    "src/occupied_frame_gpu.py": "bfc10cefea98ce00184a88b5c375eadfcda8e3dc6954d9f66183d566008951b0",
}
DEFAULT_REUSED_ROOT = (
    Path("/content/drive/MyDrive/classA_final_production_outputs") / REUSED_REVISION
)
DEFAULT_V1_ROOT = (
    Path("/content/drive/MyDrive/classA_final_production_outputs") / V1_REVISION
)
DEFAULT_OUTPUT_ROOT = (
    Path("/content/drive/MyDrive/classA_final_production_outputs") / REVISION
)
ALPHA_SWEEP = (
    3.0,
    2.75,
    2.5,
    2.3,
    2.2,
    2.15,
    2.1,
    2.075,
    2.05,
    2.025,
    2.0,
    1.975,
    1.95,
    1.925,
    1.9,
    1.85,
    1.8,
    1.7,
    1.5,
    1.25,
    1.0,
)
ENDPOINT_ALPHAS = (3.0, 1.0)
SWEEP_NY = (24, 28, 32)
ENDPOINT_NY = (36, 40)
BATCH_SIZE_BY_NY = {24: 25, 28: 20, 32: 15, 36: 10, 40: 10}
SAMPLES_PER_CASE = 100
SLOW_GAP_COUNT = 5
EXPECTED_CASES = 67
EXPECTED_TASKS = 375
EXPECTED_SAMPLES = 6700
LANES = ("A", "B")
SOURCE_PATHS = {
    "runner": Path(__file__).resolve(),
    "config": CONFIG_PATH,
    "record_helper": BUNDLE_ROOT / "replay_record_observer.py",
    "gpu_engine": SRC_ROOT / "classA_U1FGTN_gpu.py",
    "occupied_frame": SRC_ROOT / "occupied_frame_gpu.py",
}


@dataclass(frozen=True)
class Task:
    ny: int
    alpha_1: float
    alpha_index: int
    case_index: int
    batch_index: int
    sample_start: int
    sample_stop: int
    seed: int
    lane: str

    @property
    def sample_count(self) -> int:
        return self.sample_stop - self.sample_start

    @property
    def cycles(self) -> int:
        return 2 * self.ny

    @property
    def case_sample_indices(self) -> tuple[int, ...]:
        return tuple(range(self.sample_start, self.sample_stop))

    @property
    def global_sample_indices(self) -> tuple[int, ...]:
        first = self.case_index * SAMPLES_PER_CASE + self.sample_start
        return tuple(range(first, first + self.sample_count))

    @property
    def alpha_label(self) -> str:
        return alpha_tag(self.alpha_1)

    @property
    def reuse_record(self) -> bool:
        return self.ny in SWEEP_NY and self.alpha_1 in ENDPOINT_ALPHAS

    @property
    def import_v1(self) -> bool:
        return self.ny == 40 and self.alpha_1 == 1.0 and self.sample_start == 0 and self.sample_stop == 25

    @property
    def save_cocycle(self) -> bool:
        return self.ny == 40

    @property
    def task_id(self) -> str:
        return (
            f"Ny{self.ny:03d}_a1-{self.alpha_label}_batch-{self.batch_index:03d}_"
            f"samples-{self.sample_start:03d}-{self.sample_stop - 1:03d}"
        )


def alpha_tag(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_payload_sha256(values: Mapping[str, np.ndarray]) -> str:
    """Bind every replay-relevant array without retaining the transient record."""

    digest = hashlib.sha256()
    for name in (
        "initial_frame",
        "initial_ranks",
        "final_frame",
        "final_ranks",
        "schedule",
        "outcomes",
        "measurement_log_probability",
    ):
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(canonical_json(list(array.shape)).encode("ascii") + b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(path) for name, path in SOURCE_PATHS.items()}


def nvrtc_builtins_soname(cuda_version: str | None) -> str | None:
    if cuda_version is None:
        return None
    pieces = str(cuda_version).split(".")
    if len(pieces) < 2 or not pieces[0].isdigit() or not pieces[1].isdigit():
        return None
    return f"libnvrtc-builtins.so.{int(pieces[0])}.{int(pieces[1])}"


def find_nvrtc_library_dir(
    soname: str,
    search_roots: tuple[Path, ...] | None = None,
) -> Path | None:
    if search_roots is None:
        roots = [Path(value) for value in site.getsitepackages()]
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
        roots.extend(
            (
                Path(torch.__file__).resolve().parent,
                Path("/usr/local/cuda/lib64"),
                Path("/usr/local/cuda/targets/x86_64-linux/lib"),
            )
        )
    else:
        roots = list(search_roots)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            (
                root,
                root / "lib",
                root / "nvidia" / "cuda_nvrtc" / "lib",
                root / "nvidia" / "cu13" / "lib",
            )
        )
    for directory in candidates:
        if (directory / soname).is_file():
            return directory.resolve()
    return None


def ensure_nvrtc_runtime_or_reexec() -> None:
    """Repair the current Colab CUDA-13 NVRTC packaging mismatch once."""

    soname = nvrtc_builtins_soname(torch.version.cuda)
    if soname != "libnvrtc-builtins.so.13.0":
        return
    try:
        ctypes.CDLL(soname, mode=ctypes.RTLD_GLOBAL)
        print(f"[nvrtc] {soname} is available", flush=True)
        return
    except OSError:
        pass
    if os.environ.get(NVRTC_REEXEC_MARKER) == "1":
        raise RuntimeError(
            f"{soname} is still unavailable after installing {NVRTC_CUDA13_REQUIREMENT}; "
            "restart with a Colab 26.07/26.04 runtime or report the runtime image"
        )
    library_dir = find_nvrtc_library_dir(soname)
    if library_dir is None:
        print(
            f"[nvrtc] missing {soname}; installing pinned {NVRTC_CUDA13_REQUIREMENT}",
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--no-cache-dir",
                NVRTC_CUDA13_REQUIREMENT,
            ],
            check=True,
        )
        library_dir = find_nvrtc_library_dir(soname)
    if library_dir is None:
        raise RuntimeError(
            f"{NVRTC_CUDA13_REQUIREMENT} did not provide the required {soname}"
        )
    environment = dict(os.environ)
    old_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(library_dir) if not old_path else f"{library_dir}:{old_path}"
    )
    environment[NVRTC_REEXEC_MARKER] = "1"
    print(
        f"[nvrtc] re-executing with {library_dir} on LD_LIBRARY_PATH",
        flush=True,
    )
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def compatible_v2_source_hashes(
    observed: Mapping[str, str], current: Mapping[str, str]
) -> bool:
    """Accept pinned orchestration-only predecessors without weakening science pins."""

    if set(observed) != set(current):
        return False
    for name, expected in current.items():
        value = observed.get(name)
        if name == "runner":
            if value not in {
                expected,
                PRE_PADDING_HOTFIX_RUNNER_SHA256,
                PRE_NVRTC_HOTFIX_RUNNER_SHA256,
                PRE_NVRTC_LAYOUT_HOTFIX_RUNNER_SHA256,
            }:
                return False
        elif value != expected:
            return False
    return True


def expected_config() -> dict[str, Any]:
    return {
        "schema": "hard_wall_tangent_gap_cocycle_config_v1",
        "sampling_revision": REVISION,
        "root_seed": 2026091501,
        "Nx": 20,
        "alpha_1_sweep_values": list(ALPHA_SWEEP),
        "alpha_1_endpoint_values": list(ENDPOINT_ALPHAS),
        "alpha_2": 30.0,
        "sweep_Ny_values": list(SWEEP_NY),
        "endpoint_Ny_values": list(ENDPOINT_NY),
        "nshell": 1,
        "samples_per_case": SAMPLES_PER_CASE,
        "batch_size_by_Ny": {str(key): value for key, value in BATCH_SIZE_BY_NY.items()},
        "verified_v1_import": {
            "revision": V1_REVISION,
            "Ny": 40,
            "alpha_1": 1.0,
            "case_sample_indices": list(range(25)),
            "result_sha256": V1_RESULT_SHA256,
        },
        "cycles_multiplier": 2,
        "trial_orbitals": "X",
        "filling_fraction": 0.5,
        "sequence": "raster_y",
        "perfect_correction": True,
        "postselect": False,
        "dtype": "complex128",
        "device": "cuda:0",
        "state_representation": "physical_frame",
        "wall_construction": "hard_support_truncated",
        "tangent_basis_mode": "batched_canonical_from_occupied_empty_basis",
        "slow_gap_count": SLOW_GAP_COUNT,
        "singular_tolerance": 1e-14,
        "initial_purity_tolerance": 2e-9,
        "replay_probability_tolerance": 1e-9,
        "endpoint_projector_relative_frobenius_tolerance": 1e-6,
        "maximum_task_runtime_seconds": 3600.0,
        "maximum_peak_cuda_reserved_gib": 38.0,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 2026091402,
        "full_cocycle_Ny_values": [40],
        "reuse_acquisition_revision": REUSED_REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "saved_products": {
            "five_slowest_finite_time_gaps_every_sample": True,
            "ny40_scale_separated_chronological_cocycle": True,
            "per_cycle_products": False,
            "intermediate_frames": False,
            "choi_covariance": False,
            "covariance_history": False,
        },
    }


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected_config():
        raise ValueError("configuration differs from the locked slot-17 contract")
    return observed


def _task_seed(config: Mapping[str, Any], identity: str) -> int:
    raw = f"{config['root_seed']}|{config['sampling_revision']}|{identity}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") & ((1 << 63) - 1)


def _lane_for(ny: int, alpha_index: int) -> str:
    if ny == 24:
        return "A" if alpha_index % 2 == 0 else "B"
    if ny == 28:
        return "A" if alpha_index % 2 == 1 else "B"
    if ny == 32:
        return "A" if alpha_index % 2 == 0 else "B"
    return "A" if alpha_index == 1 else "B"


def expand_tasks(config: Mapping[str, Any] | None = None) -> list[Task]:
    config = expected_config() if config is None else dict(config)
    if config != expected_config():
        raise ValueError("cannot expand an unlocked configuration")
    cases: list[tuple[int, float, int]] = []
    for ny in SWEEP_NY:
        cases.extend((ny, alpha, index) for index, alpha in enumerate(ALPHA_SWEEP))
    for ny in ENDPOINT_NY:
        cases.extend((ny, alpha, index) for index, alpha in enumerate(ENDPOINT_ALPHAS))
    tasks: list[Task] = []
    for case_index, (ny, alpha, alpha_index) in enumerate(cases):
        batch_size = BATCH_SIZE_BY_NY[ny]
        if ny == 40 and alpha == 1.0:
            boundaries = [(0, 25)] + [
                (start, min(SAMPLES_PER_CASE, start + batch_size))
                for start in range(25, SAMPLES_PER_CASE, batch_size)
            ]
        else:
            boundaries = [
                (start, min(SAMPLES_PER_CASE, start + batch_size))
                for start in range(0, SAMPLES_PER_CASE, batch_size)
            ]
        for batch_index, (start, stop) in enumerate(boundaries):
            identity = f"Ny={ny}|a1={alpha:.12g}|batch={batch_index}|samples={start}:{stop}"
            tasks.append(
                Task(
                    ny=ny,
                    alpha_1=alpha,
                    alpha_index=alpha_index,
                    case_index=case_index,
                    batch_index=batch_index,
                    sample_start=start,
                    sample_stop=stop,
                    seed=(
                        V1_BATCH_SEED
                        if ny == 40 and alpha == 1.0 and start == 0 and stop == 25
                        else _task_seed(config, identity)
                    ),
                    lane=_lane_for(ny, alpha_index),
                )
            )
    if len(cases) != EXPECTED_CASES or len(tasks) != EXPECTED_TASKS:
        raise RuntimeError("task expansion count differs from the locked campaign")
    if sum(task.sample_count for task in tasks) != EXPECTED_SAMPLES:
        raise RuntimeError("task expansion does not contain 6,700 sample rows")
    if len({task.task_id for task in tasks}) != len(tasks):
        raise RuntimeError("task IDs are not unique")
    if len({task.seed for task in tasks}) != len(tasks):
        raise RuntimeError("task seeds are not unique")
    if sorted(index for task in tasks for index in task.global_sample_indices) != list(
        range(EXPECTED_SAMPLES)
    ):
        raise RuntimeError("global sample indices are not an exact partition")
    lane_counts = {lane: sum(task.lane == lane for task in tasks) for lane in LANES}
    if lane_counts != {"A": 190, "B": 185}:
        raise RuntimeError(f"unexpected lane partition {lane_counts}")
    return tasks


def tasks_for_lane(lane: str, config: Mapping[str, Any] | None = None) -> list[Task]:
    lane = lane.upper()
    if lane not in LANES:
        raise ValueError("lane must be A or B")
    tasks = [task for task in expand_tasks(config) if task.lane == lane]
    return sorted(tasks, key=lambda task: (0 if task.ny == 40 else 1, -task.ny, task.case_index, task.batch_index))


def result_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = output_root / "results" / f"Ny{task.ny:03d}" / f"alpha1_{task.alpha_label}"
    stem = f"batch_{task.batch_index:03d}_samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    return directory / f"{stem}.npz", directory / f"{stem}.complete.json"


def verify_v1_import(root: Path, task: Task) -> tuple[Path, Mapping[str, Any]]:
    """Reuse only the exact 25-row v1 archive that exceeded the runtime gate."""

    if not task.import_v1:
        raise ValueError("task is not the pinned v1 import")
    result_path, completion_path = result_paths(root, task)
    if not result_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError("the saved Ny=40, alpha1=1 v1 archive/completion pair is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": V1_REVISION,
        "task_id": task.task_id,
        "lane": "A",
        "Nx": 20,
        "Ny": 40,
        "alpha_1": 1.0,
        "alpha_2": 30.0,
        "nshell": 1,
        "cycles": 80,
        "case_sample_indices": list(range(25)),
        "global_sample_indices": list(range(6600, 6625)),
        "sample_count": 25,
        "batch_seed": V1_BATCH_SEED,
        "configuration_sha256": V1_CONFIGURATION_SHA256,
        "source_hashes": V1_SOURCE_HASHES,
        "full_cocycle_saved": True,
        "result_filename": result_path.name,
        "result_bytes": V1_RESULT_BYTES,
        "result_sha256": V1_RESULT_SHA256,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"the saved v1 completion differs from the pinned import: {key}")
    if result_path.stat().st_size != V1_RESULT_BYTES or sha256_file(result_path) != V1_RESULT_SHA256:
        raise ValueError("the saved v1 archive byte count or SHA-256 differs from the pinned import")
    with np.load(result_path, allow_pickle=False) as archive:
        for key, value in {
            "schema": RESULT_SCHEMA,
            "sampling_revision": V1_REVISION,
            "task_id": task.task_id,
            "Nx": 20,
            "Ny": 40,
            "alpha_1": 1.0,
            "cycles": 80,
            "sample_count": 25,
            "dtype": "complex128",
            "full_cocycle_saved": True,
        }.items():
            if archive[key].item() != value:
                raise ValueError(f"the saved v1 NPZ scientific identity differs: {key}")
        if json.loads(str(archive["source_hashes_json"].item())) != V1_SOURCE_HASHES:
            raise ValueError("the saved v1 NPZ source hashes differ")
        if not np.array_equal(archive["case_sample_indices"], np.arange(25)):
            raise ValueError("the saved v1 NPZ sample indices differ")
        gaps = np.asarray(archive["slow_effective_gaps_per_cycle"], dtype=np.float64)
        if gaps.shape != (25, SLOW_GAP_COUNT) or not np.all(np.isfinite(gaps)):
            raise ValueError("the saved v1 NPZ gaps are missing or nonfinite")
    return result_path, completion


def task_identity(task: Task, *, config_sha256: str, hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "bundle": BUNDLE,
        "sampling_revision": REVISION,
        "canonical_entry_point": CANONICAL_ENTRY_POINT,
        "task_id": task.task_id,
        "lane": task.lane,
        "Nx": 20,
        "Ny": task.ny,
        "alpha_1": task.alpha_1,
        "alpha_2": 30.0,
        "nshell": 1,
        "cycles": task.cycles,
        "case_sample_indices": list(task.case_sample_indices),
        "global_sample_indices": list(task.global_sample_indices),
        "sample_count": task.sample_count,
        "batch_seed": task.seed,
        "record_source": "reused_slot09" if task.reuse_record else "ephemeral_acquisition",
        "full_cocycle_saved": task.save_cocycle,
        "configuration_sha256": config_sha256,
        "source_hashes": dict(hashes),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    expected_bytes = int(local_path.stat().st_size)
    expected_sha = sha256_file(local_path)
    try:
        shutil.copyfile(local_path, temporary)
        if temporary.stat().st_size != expected_bytes or sha256_file(temporary) != expected_sha:
            raise OSError(f"Drive temporary readback mismatch: {temporary}")
        os.replace(temporary, final_path)
        if final_path.stat().st_size != expected_bytes or sha256_file(final_path) != expected_sha:
            raise OSError(f"Drive final readback mismatch: {final_path}")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"filename": final_path.name, "bytes": expected_bytes, "sha256": expected_sha}


def reused_result_paths(root: Path, task: Task) -> tuple[Path, Path]:
    if not task.reuse_record:
        raise ValueError("task does not use a slot-09 record")
    alpha = int(task.alpha_1)
    directory = root / "results" / "hard" / f"Ny{task.ny:03d}" / f"alpha1_{alpha}"
    stem = f"batch_{task.batch_index:03d}_samples_{task.sample_start:03d}-{task.sample_stop - 1:03d}"
    canonical = (directory / f"{stem}.npz", directory / f"{stem}.complete.json")
    if canonical[0].is_file() or canonical[1].is_file():
        return canonical
    flattened = tuple(root / path.relative_to(root / "results") for path in canonical)
    if flattened[0].is_file() or flattened[1].is_file():
        return flattened  # type: ignore[return-value]
    return canonical


def verify_reused_pair(root: Path, task: Task, *, checksum: bool = True) -> tuple[Path, Mapping[str, Any]]:
    result_path, completion_path = reused_result_paths(root, task)
    if not result_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"missing slot-09 acquisition pair for {task.task_id}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected = {
        "schema": "pure_tangent_replay_acquisition_completion_v1",
        "status": "complete",
        "sampling_revision": REUSED_REVISION,
        "construction": "hard",
        "Nx": 20,
        "Ny": task.ny,
        "alpha_1": task.alpha_1,
        "alpha_2": 30.0,
        "nshell": 1,
        "cycles": task.cycles,
        "case_sample_indices": list(task.case_sample_indices),
        "configuration_sha256": REUSED_CONFIG_SHA256,
        "source_hashes": REUSED_SOURCE_HASHES,
    }
    for key, value in expected.items():
        if completion.get(key) != value:
            raise ValueError(f"slot-09 completion mismatch for {task.task_id}: {key}")
    if completion.get("result_filename") != result_path.name:
        raise ValueError(f"slot-09 filename mismatch for {task.task_id}")
    if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
        raise ValueError(f"slot-09 byte-count mismatch for {task.task_id}")
    if checksum and completion.get("result_sha256") != sha256_file(result_path):
        raise ValueError(f"slot-09 checksum mismatch for {task.task_id}")
    with np.load(result_path, allow_pickle=False) as archive:
        if str(archive["sampling_revision"].item()) != REUSED_REVISION:
            raise ValueError("slot-09 NPZ revision mismatch")
        if str(archive["construction"].item()) != "hard":
            raise ValueError("slot-09 NPZ is not hard-wall data")
        if int(archive["Ny"].item()) != task.ny or float(archive["alpha_1"].item()) != task.alpha_1:
            raise ValueError("slot-09 NPZ geometry mismatch")
        if not np.array_equal(archive["case_sample_indices"], np.asarray(task.case_sample_indices)):
            raise ValueError("slot-09 NPZ sample-index mismatch")
    return result_path, completion


def stage_reused_record(root: Path, task: Task, scratch_root: Path) -> tuple[Path, Mapping[str, Any]]:
    result_path, completion = verify_reused_pair(root, task, checksum=True)
    cache = scratch_root / "input_cache"
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / f"slot09_{task.task_id}.npz"
    expected_bytes = int(completion["result_bytes"])
    expected_sha = str(completion["result_sha256"])
    if local.is_file() and local.stat().st_size == expected_bytes and sha256_file(local) == expected_sha:
        print(f"[input cache] {task.task_id}", flush=True)
        return local, completion
    temporary = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(result_path, temporary)
        if temporary.stat().st_size != expected_bytes or sha256_file(temporary) != expected_sha:
            raise OSError(f"staged slot-09 checksum mismatch: {task.task_id}")
        os.replace(temporary, local)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[input staged] {task.task_id}: {expected_bytes / 1024**3:.2f} GiB", flush=True)
    return local, completion


def source_slot09_tasks(task: Task) -> list[Task]:
    """Locate immutable slot-09 archives covering a smaller v2 task."""

    if not task.reuse_record:
        raise ValueError("the task does not use slot-09 records")
    old_size = {24: 50, 28: 50, 32: 25}[task.ny]
    first = task.sample_start // old_size
    last = (task.sample_stop - 1) // old_size
    return [
        Task(
            ny=task.ny,
            alpha_1=task.alpha_1,
            alpha_index=task.alpha_index,
            case_index=task.case_index,
            batch_index=index,
            sample_start=index * old_size,
            sample_stop=min(SAMPLES_PER_CASE, (index + 1) * old_size),
            seed=0,
            lane=task.lane,
        )
        for index in range(first, last + 1)
    ]


def load_reused_record(
    root: Path, task: Task, scratch_root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pieces: list[dict[str, np.ndarray]] = []
    source_rows: list[dict[str, Any]] = []
    original_seeds: list[int] = []
    for source in source_slot09_tasks(task):
        local, completion = stage_reused_record(root, source, scratch_root)
        loaded = _loaded_from_npz(local, source)
        first = max(task.sample_start, source.sample_start)
        last = min(task.sample_stop, source.sample_stop)
        selection = slice(first - source.sample_start, last - source.sample_start)
        pieces.append({name: np.array(value[selection], copy=True) for name, value in loaded.items()})
        source_rows.append(
            {
                "task_id": str(completion["task_id"]),
                "result_sha256": str(completion["result_sha256"]),
                "seed": int(completion["seed"]),
                "case_sample_indices": list(range(first, last)),
            }
        )
        original_seeds.extend([int(completion["seed"])] * (last - first))
        local.unlink(missing_ok=True)
    combined = concatenate_record_pieces(pieces)
    if combined["initial_frame"].shape[0] != task.sample_count:
        raise RuntimeError("the slot-09 source slices do not exactly cover this v2 task")
    return combined, {
        "record_source": "reused_slot09",
        "record_seed": -1,
        "record_original_batch_seeds": original_seeds,
        "record_source_batches": source_rows,
        "record_sha256": record_payload_sha256(combined),
    }


def concatenate_record_pieces(
    pieces: list[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate slot-09 slices with zero-padding for variable frame widths.

    Occupied frames are stored as ``(samples, modes, padded_rank)``. Independent
    slot-09 batches can have different final padded widths because their maximum
    realized particle ranks differ. The ranks identify the live columns, so
    adding zero padding is an exact representation change.
    """

    if not pieces:
        raise ValueError("cannot concatenate an empty record-piece list")
    names = set(pieces[0])
    if any(set(piece) != names for piece in pieces[1:]):
        raise ValueError("slot-09 record pieces have different fields")
    combined: dict[str, np.ndarray] = {}
    frame_ranks = {"initial_frame": "initial_ranks", "final_frame": "final_ranks"}
    for name in pieces[0]:
        arrays = [np.asarray(piece[name]) for piece in pieces]
        if name in frame_ranks:
            if any(array.ndim != 3 for array in arrays):
                raise ValueError(f"{name} must have shape (samples,modes,padded_rank)")
            mode_counts = {int(array.shape[1]) for array in arrays}
            if len(mode_counts) != 1:
                raise ValueError(f"slot-09 {name} mode dimensions differ")
            target_width = max(int(array.shape[2]) for array in arrays)
            padded = []
            ranks_name = frame_ranks[name]
            for piece, array in zip(pieces, arrays, strict=True):
                ranks = np.asarray(piece[ranks_name], dtype=np.int64)
                if ranks.shape != (array.shape[0],):
                    raise ValueError(f"slot-09 {ranks_name} shape mismatch")
                if np.any(ranks < 0) or np.any(ranks > array.shape[2]):
                    raise ValueError(f"slot-09 {name} rank exceeds its padded width")
                if array.shape[2] == target_width:
                    padded.append(array)
                else:
                    normalized = np.zeros(
                        (array.shape[0], array.shape[1], target_width), dtype=array.dtype
                    )
                    normalized[:, :, : array.shape[2]] = array
                    padded.append(normalized)
            arrays = padded
        else:
            trailing_shapes = {array.shape[1:] for array in arrays}
            if len(trailing_shapes) != 1:
                raise ValueError(f"slot-09 {name} trailing dimensions differ")
        combined[name] = np.concatenate(arrays, axis=0)
    return combined


def _loaded_from_npz(path: Path, task: Task) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        initial_frame = np.asarray(archive["initial_frame"], dtype=np.complex128)
        initial_ranks = np.asarray(archive["initial_ranks"], dtype=np.int64)
        final_frame = np.asarray(archive["final_frame"], dtype=np.complex128)
        final_ranks = np.asarray(archive["final_ranks"], dtype=np.int64)
        schedule = np.asarray(archive["record_schedule"], dtype=np.int32)
        packed = np.asarray(archive["record_outcomes_packed"], dtype=np.uint8)
        shape = np.asarray(archive["record_outcomes_shape"], dtype=np.int64)
        log_probability = np.asarray(archive["measurement_log_probability"], dtype=np.float64)
    outcomes = unpack_boolean_record(packed, shape)
    if initial_frame.shape[0] != task.sample_count or final_frame.shape[0] != task.sample_count:
        raise ValueError("record sample dimension mismatch")
    return {
        "initial_frame": initial_frame,
        "initial_ranks": initial_ranks,
        "final_frame": final_frame,
        "final_ranks": final_ranks,
        "schedule": schedule,
        "outcomes": outcomes,
        "measurement_log_probability": log_probability,
    }


def _seed_rng(seed: int) -> None:
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_model(task: Task, config: Mapping[str, Any]) -> classA_U1FGTN_gpu:
    model = classA_U1FGTN_gpu(
        Nx=20,
        Ny=task.ny,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=task.alpha_1,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
        triv_region_local_mode=False,
        device=str(config["device"]),
        dtype=str(config["dtype"]),
        backend="local",
    )
    if tuple(int(value) for value in model.DW_loc) != (5, 15):
        raise RuntimeError(f"unexpected domain-wall locations {model.DW_loc}")
    if model.dtype != torch.complex128:
        raise RuntimeError("constructed model is not complex128")
    active = model.active_top_layer_indices(meas_slab_only=True)
    if int(active.numel()) != 22 * task.ny:
        raise RuntimeError("hard-wall active dimension is not 22*Ny")
    return model


def acquire_ephemeral_record(
    model: classA_U1FGTN_gpu, task: Task
) -> tuple[dict[str, np.ndarray], Mapping[str, Any]]:
    _seed_rng(task.seed)
    updates = 11 * task.ny
    observer = ReplayRecordObserver(samples=task.sample_count, cycles=task.cycles, updates_per_cycle=updates)
    print(
        f"[acquisition] {task.task_id}: samples={task.sample_count}, cycles={task.cycles}, seed={task.seed}",
        flush=True,
    )
    result = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=task.cycles,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=task.sample_count,
        init_mode="default",
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=True,
        batch_size=task.sample_count,
        return_data=True,
        state_representation="physical_frame",
        native_cycle_observer=observer.capture_native_cycle,
        record_observer=observer.record_event,
        track_choi=False,
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )
    if bool(result.get("choi_tracked", False)) or bool(result.get("lyapunov_tracked", False)):
        raise RuntimeError("acquisition unexpectedly enabled Choi/tangent tracking")
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("acquisition materialized a covariance")
    final_frame, final_ranks = native_frame_arrays(result["native_final"])
    arrays = observer.result_arrays()
    loaded = {
        "initial_frame": np.asarray(arrays["initial_frame"], dtype=np.complex128),
        "initial_ranks": np.asarray(arrays["initial_ranks"], dtype=np.int64),
        "final_frame": final_frame,
        "final_ranks": final_ranks,
        "schedule": np.asarray(arrays["record_schedule"], dtype=np.int32),
        "outcomes": unpack_boolean_record(
            np.asarray(arrays["record_outcomes_packed"], dtype=np.uint8),
            np.asarray(arrays["record_outcomes_shape"], dtype=np.int64),
        ),
        "measurement_log_probability": np.asarray(
            arrays["measurement_log_probability"], dtype=np.float64
        ),
    }
    provenance = {
        "record_source": "ephemeral_acquisition",
        "record_seed": task.seed,
        "record_original_batch_seeds": [task.seed] * task.sample_count,
        "record_sha256": record_payload_sha256(loaded),
    }
    return loaded, provenance


def occupied_empty_basis(
    model: classA_U1FGTN_gpu,
    frames: np.ndarray,
    ranks: np.ndarray,
    *,
    purity_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame = torch.as_tensor(frames, dtype=torch.complex128, device=model.device)
    rank_values = torch.as_tensor(ranks, dtype=torch.int64, device=model.device)
    active = model.active_top_layer_indices(meas_slab_only=True).to(model.device)
    mask = torch.arange(frame.shape[-1], device=model.device)[None, :] < rank_values[:, None]
    occupied = (frame * mask[:, None, :]).index_select(1, active)
    correlation = occupied @ occupied.mH
    correlation = 0.5 * (correlation + correlation.mH)
    occupations, eigenvectors = torch.linalg.eigh(correlation)
    defects = torch.minimum(occupations.abs(), (1.0 - occupations).abs()).amax(dim=1)
    if bool(torch.any(~torch.isfinite(defects)).item()) or float(defects.max().item()) > purity_tolerance:
        raise FloatingPointError(f"active prepared-state purity defect {float(defects.max()):.3e}")
    occupied_masks = occupations > 0.5
    occupied_counts = torch.count_nonzero(occupied_masks, dim=1)
    empty_counts = occupations.shape[1] - occupied_counts
    block_sizes = torch.stack((occupied_counts, empty_counts), dim=1).to(torch.int64)
    ordered = []
    for row in range(frame.shape[0]):
        row_mask = occupied_masks[row]
        ordered.append(torch.cat((eigenvectors[row, :, row_mask], eigenvectors[row, :, ~row_mask]), dim=1))
    basis = torch.stack(ordered, dim=0)
    target = torch.eye(basis.shape[-1], dtype=basis.dtype, device=basis.device)
    error = torch.abs(basis.mH @ basis - target).amax()
    if not bool(torch.isfinite(error).item()) or float(error.item()) > 1e-9:
        raise FloatingPointError(f"tangent basis Gram error {float(error.item()):.3e}")
    return basis, active, block_sizes, defects


class ProbabilityAccumulator:
    def __init__(self, samples: int, cycles: int, device: torch.device) -> None:
        self.values = torch.zeros((samples, cycles + 1), dtype=torch.float64, device=device)

    def __call__(
        self,
        *,
        cycle: int,
        sample_offsets: torch.Tensor,
        conditional_log_probability: torch.Tensor,
        **_: Any,
    ) -> None:
        rows = sample_offsets.to(dtype=torch.long, device=self.values.device)
        self.values[rows, int(cycle)] += conditional_log_probability.to(torch.float64).sum(dim=1)


class EndpointTangentCapture:
    def __init__(self, final_cycle: int) -> None:
        self.final_cycle = int(final_cycle)
        self.frame: torch.Tensor | None = None
        self.core: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.core_null_count: torch.Tensor | None = None
        self.null_counts: torch.Tensor | None = None
        self.min_probability: torch.Tensor | None = None
        self.min_denominator: torch.Tensor | None = None
        self.invalid_count: torch.Tensor | None = None
        self.physical_frame: torch.Tensor | None = None
        self.physical_ranks: torch.Tensor | None = None

    def __call__(
        self,
        *,
        cycle: int,
        G: Any,
        lyapunov_frame: torch.Tensor,
        lyapunov_core_hat: torch.Tensor,
        lyapunov_core_log_scale: torch.Tensor,
        lyapunov_core_null_count: torch.Tensor,
        lyapunov_null_counts: torch.Tensor,
        lyapunov_min_branch_probability: torch.Tensor,
        lyapunov_min_abs_born_denominator: torch.Tensor,
        lyapunov_invalid_branch_count: torch.Tensor,
        **_: Any,
    ) -> None:
        if int(cycle) != self.final_cycle:
            return
        self.frame = lyapunov_frame.detach().clone()
        self.core = lyapunov_core_hat.detach().clone()
        self.scale = lyapunov_core_log_scale.detach().clone()
        self.core_null_count = lyapunov_core_null_count.detach().clone()
        self.null_counts = lyapunov_null_counts.detach().clone()
        self.min_probability = lyapunov_min_branch_probability.detach().clone()
        self.min_denominator = lyapunov_min_abs_born_denominator.detach().clone()
        self.invalid_count = lyapunov_invalid_branch_count.detach().clone()
        self.physical_frame = G.frame.detach().clone()
        self.physical_ranks = G.ranks.detach().clone()


def projector_errors(
    actual_frames: torch.Tensor,
    actual_ranks: torch.Tensor,
    reference_frames: np.ndarray,
    reference_ranks: np.ndarray,
) -> np.ndarray:
    reference = torch.as_tensor(reference_frames, dtype=torch.complex128, device=actual_frames.device)
    ranks = torch.as_tensor(reference_ranks, dtype=torch.int64, device=actual_frames.device)
    if not bool(torch.all(actual_ranks == ranks).item()):
        raise RuntimeError("replayed endpoint ranks differ from acquisition")
    values = []
    for row in range(actual_frames.shape[0]):
        rank = int(ranks[row].item())
        overlap = actual_frames[row, :, :rank].mH @ reference[row, :, :rank]
        squared = 2.0 * rank - 2.0 * torch.linalg.matrix_norm(overlap, ord="fro") ** 2
        values.append(torch.sqrt(torch.clamp(squared.real / max(rank, 1), min=0.0)))
    return np.asarray(torch.stack(values).detach().cpu().numpy(), dtype=np.float64)


def finalize_tangent(
    task: Task,
    capture: EndpointTangentCapture,
    initial_basis: torch.Tensor,
    active_indices: torch.Tensor,
    block_sizes: torch.Tensor,
    *,
    singular_tolerance: float,
) -> dict[str, np.ndarray]:
    if capture.frame is None or capture.core is None or capture.scale is None:
        raise RuntimeError("endpoint tangent capture is incomplete")
    image = capture.frame @ capture.core
    batch, full_dimension, active_dimension = image.shape
    rates_rows: list[torch.Tensor] = []
    gaps_rows: list[torch.Tensor] = []
    pair_rows: list[torch.Tensor] = []
    singular_null_rows: list[torch.Tensor] = []
    leading_logs_rows: list[torch.Tensor] = []
    for row in range(batch):
        occupied_dim, empty_dim = (int(value) for value in block_sizes[row].tolist())
        parts = []
        nulls = []
        leading = []
        start = 0
        for size in (occupied_dim, empty_dim):
            singular = torch.linalg.svdvals(image[row, :, start : start + size]).to(torch.float64)
            threshold = max(
                float(singular_tolerance),
                float(torch.finfo(torch.float64).eps * max(full_dimension, size) * singular[0].item()),
            )
            finite = torch.isfinite(singular) & (singular > threshold)
            logs = torch.full_like(singular, -torch.inf)
            logs[finite] = torch.log(singular[finite]) + capture.scale[row]
            parts.append(logs)
            nulls.append(torch.count_nonzero(~finite))
            leading.append(logs[0])
            start += size
        pair_rates = (parts[0][:, None] + parts[1][None, :]) / float(task.cycles)
        sortable = torch.where(
            torch.isfinite(pair_rates),
            torch.abs(pair_rates),
            torch.full_like(pair_rates, torch.inf),
        )
        values, flat = torch.topk(sortable.reshape(-1), k=SLOW_GAP_COUNT, largest=False, sorted=True)
        if bool(torch.any(~torch.isfinite(values)).item()):
            raise FloatingPointError("fewer than five finite physical tangent gaps")
        occupied_index = torch.div(flat, empty_dim, rounding_mode="floor")
        empty_index = flat % empty_dim
        rates = pair_rates.reshape(-1).index_select(0, flat)
        rates_rows.append(rates)
        gaps_rows.append(-2.0 * rates)
        pair_rows.append(torch.stack((occupied_index, empty_index), dim=1))
        singular_null_rows.append(torch.stack(nulls))
        leading_logs_rows.append(torch.stack(leading))
    arrays = {
        "slow_pair_rates_per_cycle": np.asarray(torch.stack(rates_rows).cpu(), dtype=np.float64),
        "slow_effective_gaps_per_cycle": np.asarray(torch.stack(gaps_rows).cpu(), dtype=np.float64),
        "slow_pair_indices": np.asarray(torch.stack(pair_rows).cpu(), dtype=np.int32),
        "one_leg_singular_null_counts": np.asarray(torch.stack(singular_null_rows).cpu(), dtype=np.int64),
        "one_leg_leading_log_singular_values": np.asarray(torch.stack(leading_logs_rows).cpu(), dtype=np.float64),
        "occupied_empty_block_sizes": np.asarray(block_sizes.detach().cpu(), dtype=np.int64),
        "active_input_indices": np.asarray(active_indices.detach().cpu(), dtype=np.int64),
        "active_dimension": np.asarray(active_dimension, dtype=np.int64),
        "full_dimension": np.asarray(full_dimension, dtype=np.int64),
    }
    if task.save_cocycle:
        # Column propagation gives J~^dagger U0.  Rotate back to the canonical
        # active input basis and store J~ with inactive hard-wall input rows omitted.
        adjoint_active = image @ initial_basis.mH
        chronological = adjoint_active.mH
        norms = torch.linalg.matrix_norm(chronological, ord="fro", dim=(-2, -1)).to(torch.float64)
        if bool(torch.any(~torch.isfinite(norms)).item()) or bool(torch.any(norms <= 0).item()):
            raise FloatingPointError("chronological cocycle normalization failed")
        arrays["chronological_cocycle_hat"] = np.asarray(
            (chronological / norms[:, None, None]).detach().cpu(), dtype=np.complex128
        )
        arrays["chronological_cocycle_log_scale"] = np.asarray(
            (capture.scale + torch.log(norms)).detach().cpu(), dtype=np.float64
        )
    return arrays


def replay_tangent(
    model: classA_U1FGTN_gpu,
    task: Task,
    loaded: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    print(f"[basis] {task.task_id}: constructing occupied/empty tangent basis", flush=True)
    basis, active, blocks, purity_defects = occupied_empty_basis(
        model,
        loaded["initial_frame"],
        loaded["initial_ranks"],
        purity_tolerance=float(config["initial_purity_tolerance"]),
    )
    capture = EndpointTangentCapture(task.cycles)
    probability = ProbabilityAccumulator(task.sample_count, task.cycles, model.device)
    print(
        f"[tangent replay] {task.task_id}: samples={task.sample_count}, cycles=1..{task.cycles}, "
        f"active_dim={basis.shape[-1]}",
        flush=True,
    )
    result = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=task.cycles,
        samples=task.sample_count,
        frame_init=loaded["initial_frame"],
        frame_ranks=loaded["initial_ranks"],
        frame_init_prepared=True,
        initial_purity_tolerance=float(config["initial_purity_tolerance"]),
        save=False,
        return_data=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=True,
        batch_size=task.sample_count,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        state_representation="physical_frame",
        frozen_schedule=loaded["schedule"],
        frozen_outcomes=loaded["outcomes"],
        record_observer=probability,
        return_native_state=False,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
        lyapunov_frame_observer=capture,
        lyapunov_initial_frame=basis,
        lyapunov_nvec=basis.shape[-1],
        lyapunov_basis_mode="canonical",
        lyapunov_start_cycle=1,
        lyapunov_track_restricted_core=True,
        lyapunov_singular_tol=float(config["singular_tolerance"]),
        lyapunov_failure_mode="raise",
    )
    if int(result.get("covariance_materialization_count", -1)) != 0:
        raise RuntimeError("tangent replay materialized a covariance")
    assert capture.physical_frame is not None and capture.physical_ranks is not None
    endpoint_errors = projector_errors(
        capture.physical_frame,
        capture.physical_ranks,
        loaded["final_frame"],
        loaded["final_ranks"],
    )
    if float(endpoint_errors.max()) > float(config["endpoint_projector_relative_frobenius_tolerance"]):
        raise FloatingPointError("tangent replay endpoint projector tolerance exceeded")
    replay_probability = np.asarray(probability.values.detach().cpu(), dtype=np.float64)
    probability_errors = np.max(
        np.abs(replay_probability - loaded["measurement_log_probability"]), axis=1
    )
    if float(probability_errors.max()) > float(config["replay_probability_tolerance"]):
        raise FloatingPointError("tangent replay probability tolerance exceeded")
    arrays = finalize_tangent(
        task,
        capture,
        basis,
        active,
        blocks,
        singular_tolerance=float(config["singular_tolerance"]),
    )
    diagnostics = {
        "initial_active_purity_defect": np.asarray(purity_defects.detach().cpu(), dtype=np.float64),
        "endpoint_projector_relative_frobenius_error": endpoint_errors,
        "replay_log_probability_max_abs_error": probability_errors,
        "lyapunov_cumulative_null_count": np.asarray(capture.null_counts.detach().cpu(), dtype=np.int64),
        "lyapunov_core_null_count": np.asarray(capture.core_null_count.detach().cpu(), dtype=np.int64),
        "lyapunov_min_branch_probability": np.asarray(capture.min_probability.detach().cpu(), dtype=np.float64),
        "lyapunov_min_abs_born_denominator": np.asarray(capture.min_denominator.detach().cpu(), dtype=np.float64),
        "lyapunov_invalid_branch_count": np.asarray(capture.invalid_count.detach().cpu(), dtype=np.int64),
    }
    arrays.update(diagnostics)
    return arrays, diagnostics


def run_task(
    task: Task,
    *,
    config: Mapping[str, Any],
    reused_root: Path,
    scratch_root: Path,
    config_sha256: str,
    hashes: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if task.import_v1:
        raise RuntimeError("the pinned v1 task is imported read-only, never recomputed in v2")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    model = build_model(task, config)
    if task.reuse_record:
        loaded, provenance = load_reused_record(reused_root, task, scratch_root)
    else:
        loaded, provenance = acquire_ephemeral_record(model, task)
    tangent, _diagnostics = replay_tangent(model, task, loaded, config)
    torch.cuda.synchronize(model.device)
    elapsed = time.monotonic() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(model.device))
    peak_reserved = int(torch.cuda.max_memory_reserved(model.device))
    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(RESULT_SCHEMA),
        "bundle": np.asarray(BUNDLE),
        "sampling_revision": np.asarray(REVISION),
        "canonical_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
        "task_id": np.asarray(task.task_id),
        "lane": np.asarray(task.lane),
        "Nx": np.asarray(20, dtype=np.int64),
        "Ny": np.asarray(task.ny, dtype=np.int64),
        "alpha_1": np.asarray(task.alpha_1, dtype=np.float64),
        "alpha_2": np.asarray(30.0, dtype=np.float64),
        "nshell": np.asarray(1, dtype=np.int64),
        "cycles": np.asarray(task.cycles, dtype=np.int64),
        "burn_in_cycles": np.asarray(0, dtype=np.int64),
        "sample_count": np.asarray(task.sample_count, dtype=np.int64),
        "case_sample_indices": np.asarray(task.case_sample_indices, dtype=np.int64),
        "global_sample_indices": np.asarray(task.global_sample_indices, dtype=np.int64),
        "batch_seed": np.asarray(task.seed, dtype=np.int64),
        "record_source": np.asarray(provenance["record_source"]),
        "record_seed": np.asarray(provenance["record_seed"], dtype=np.int64),
        "record_original_batch_seeds": np.asarray(
            provenance["record_original_batch_seeds"], dtype=np.int64
        ),
        "record_sha256": np.asarray(provenance["record_sha256"]),
        "full_cocycle_saved": np.asarray(task.save_cocycle, dtype=np.bool_),
        "gap_estimator": np.asarray("trajectory_first_quenched_finite_time_endpoint"),
        "gap_convention": np.asarray("Delta_ij=-2*(log_sigma_o_i+log_sigma_e_j)/T"),
        "chronology_convention": np.asarray("J_tilde_T=J_1 J_2 ... J_T; first applied factor leftmost"),
        "covariance_action_convention": np.asarray("deltaG_T=J_tilde_T^dagger deltaG_0 J_tilde_T"),
        "dtype": np.asarray("complex128"),
        "configuration_sha256": np.asarray(config_sha256),
        "source_hashes_json": np.asarray(canonical_json(hashes)),
        "choi_covariance_constructed": np.asarray(False, dtype=np.bool_),
        "covariance_history_constructed": np.asarray(False, dtype=np.bool_),
        "intermediate_frames_saved": np.asarray(False, dtype=np.bool_),
        "per_cycle_products_saved": np.asarray(False, dtype=np.bool_),
        **tangent,
    }
    performance = {
        "elapsed_seconds": float(elapsed),
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        **dict(provenance),
    }
    del loaded, tangent, model
    torch.cuda.empty_cache()
    print(
        f"[task computed] {task.task_id}: elapsed={elapsed / 60:.2f} min, "
        f"peak_reserved={peak_reserved / 1024**3:.2f} GiB",
        flush=True,
    )
    return arrays, performance


def validate_result_npz(
    path: Path,
    task: Task,
    *,
    config_sha256: str,
    hashes: Mapping[str, str],
) -> None:
    with np.load(path, allow_pickle=False) as archive:
        scalar_expected = {
            "schema": RESULT_SCHEMA,
            "sampling_revision": REVISION,
            "canonical_entry_point": CANONICAL_ENTRY_POINT,
            "task_id": task.task_id,
            "lane": task.lane,
            "Nx": 20,
            "Ny": task.ny,
            "alpha_1": task.alpha_1,
            "cycles": task.cycles,
            "sample_count": task.sample_count,
            "configuration_sha256": config_sha256,
            "full_cocycle_saved": task.save_cocycle,
        }
        for key, expected in scalar_expected.items():
            if archive[key].item() != expected:
                raise ValueError(f"result identity mismatch: {key}")
        if json.loads(str(archive["source_hashes_json"].item())) != dict(hashes):
            raise ValueError("result source hashes mismatch")
        if not np.array_equal(archive["case_sample_indices"], np.asarray(task.case_sample_indices)):
            raise ValueError("result case sample indices mismatch")
        gaps = np.asarray(archive["slow_effective_gaps_per_cycle"], dtype=np.float64)
        rates = np.asarray(archive["slow_pair_rates_per_cycle"], dtype=np.float64)
        pairs = np.asarray(archive["slow_pair_indices"], dtype=np.int32)
        if gaps.shape != (task.sample_count, SLOW_GAP_COUNT) or rates.shape != gaps.shape:
            raise ValueError("gap array shape mismatch")
        if pairs.shape != (task.sample_count, SLOW_GAP_COUNT, 2):
            raise ValueError("gap pair-index shape mismatch")
        original_seeds = np.asarray(archive["record_original_batch_seeds"], dtype=np.int64)
        if original_seeds.shape != (task.sample_count,) or np.any(original_seeds < 0):
            raise ValueError("original record batch seeds are missing or invalid")
        if not np.all(np.isfinite(gaps)) or not np.all(np.isfinite(rates)):
            raise FloatingPointError("nonfinite gap values")
        if not np.array_equal(gaps, -2.0 * rates):
            raise ValueError("saved gaps do not exactly match -2 times the pair rates")
        active_dimension = 22 * task.ny
        full_dimension = 40 * task.ny
        blocks = np.asarray(archive["occupied_empty_block_sizes"], dtype=np.int64)
        if blocks.shape != (task.sample_count, 2) or np.any(blocks.sum(axis=1) != active_dimension):
            raise ValueError("occupied/empty block-size mismatch")
        if task.save_cocycle:
            cocycle = np.asarray(archive["chronological_cocycle_hat"], dtype=np.complex128)
            scales = np.asarray(archive["chronological_cocycle_log_scale"], dtype=np.float64)
            if cocycle.shape != (task.sample_count, active_dimension, full_dimension):
                raise ValueError("Ny=40 chronological cocycle shape mismatch")
            if scales.shape != (task.sample_count,) or not np.all(np.isfinite(scales)):
                raise FloatingPointError("chronological cocycle scale mismatch")
            norms = np.linalg.norm(cocycle.reshape(task.sample_count, -1), axis=1)
            if not np.allclose(norms, 1.0, atol=2e-12, rtol=2e-12):
                raise FloatingPointError("chronological cocycle is not normalized")
        elif "chronological_cocycle_hat" in archive.files or "chronological_cocycle_log_scale" in archive.files:
            raise ValueError("gap-only task unexpectedly saved a cocycle")
        for field in (
            "choi_covariance_constructed",
            "covariance_history_constructed",
            "intermediate_frames_saved",
            "per_cycle_products_saved",
        ):
            if bool(archive[field].item()):
                raise ValueError(f"forbidden saved product: {field}")


def verified_complete(
    output_root: Path,
    task: Task,
    *,
    config_sha256: str,
    hashes: Mapping[str, str],
    v1_root: Path = DEFAULT_V1_ROOT,
) -> tuple[bool, str]:
    if task.import_v1:
        try:
            verify_v1_import(v1_root, task)
        except (OSError, ValueError, KeyError, TypeError, FloatingPointError, json.JSONDecodeError) as exc:
            return False, f"pinned v1 import unavailable: {exc}"
        return True, "verified pinned v1 import"
    result_path, completion_path = result_paths(output_root, task)
    if not result_path.exists() and not completion_path.exists():
        return False, "missing"
    if not result_path.is_file() or not completion_path.is_file():
        return False, "incomplete result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion_hashes = completion.get("source_hashes")
        if not isinstance(completion_hashes, dict) or not compatible_v2_source_hashes(
            completion_hashes, hashes
        ):
            return False, "completion identity mismatch: source_hashes"
        for key, expected in task_identity(
            task, config_sha256=config_sha256, hashes=completion_hashes
        ).items():
            if completion.get(key) != expected:
                return False, f"completion identity mismatch: {key}"
        if completion.get("result_filename") != result_path.name:
            return False, "result filename mismatch"
        if int(completion.get("result_bytes", -1)) != result_path.stat().st_size:
            return False, "result byte-count mismatch"
        if completion.get("result_sha256") != sha256_file(result_path):
            return False, "result checksum mismatch"
        validate_result_npz(
            result_path, task, config_sha256=config_sha256, hashes=completion_hashes
        )
    except (OSError, ValueError, KeyError, TypeError, FloatingPointError, json.JSONDecodeError) as exc:
        return False, f"invalid output: {exc}"
    return True, "verified"


def estimated_result_bytes(task: Task) -> int:
    if task.save_cocycle:
        return int(task.sample_count * (22 * task.ny) * (40 * task.ny) * 16 + 128 * 1024**2)
    return 64 * 1024**2


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise OSError(f"no existing ancestor for {path}")
    return candidate


def require_storage(task: Task, scratch_root: Path, output_root: Path) -> None:
    estimate = estimated_result_bytes(task)
    local_free = shutil.disk_usage(_existing_ancestor(scratch_root)).free
    drive_free = shutil.disk_usage(_existing_ancestor(output_root)).free
    if local_free < estimate + 4 * 1024**3:
        raise OSError("insufficient local scratch headroom")
    if drive_free < 2 * estimate + 2 * 1024**3:
        raise OSError("insufficient DriveFS headroom")
    print(
        f"[storage] result_estimate={estimate / 1024**3:.2f} GiB, "
        f"local_free={local_free / 1024**3:.1f} GiB, drive_free={drive_free / 1024**3:.1f} GiB",
        flush=True,
    )


def save_task(
    task: Task,
    arrays: Mapping[str, Any],
    performance: Mapping[str, Any],
    *,
    output_root: Path,
    scratch_root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    hashes: Mapping[str, str],
) -> bool:
    local_dir = scratch_root / "outputs" / task.task_id
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir(parents=True)
    local_result = local_dir / "result.npz"
    _atomic_npz(local_result, arrays)
    validate_result_npz(local_result, task, config_sha256=config_sha256, hashes=hashes)
    final_result, final_completion = result_paths(output_root, task)
    published = publish_file(local_result, final_result)
    runtime_ok = float(performance["elapsed_seconds"]) <= float(config["maximum_task_runtime_seconds"])
    memory_ok = int(performance["peak_cuda_reserved_bytes"]) <= int(
        float(config["maximum_peak_cuda_reserved_gib"]) * 1024**3
    )
    gate_passed = bool(runtime_ok and memory_ok)
    completion = task_identity(task, config_sha256=config_sha256, hashes=hashes)
    completion.update(
        {
            "result_filename": published["filename"],
            "result_bytes": published["bytes"],
            "result_sha256": published["sha256"],
            **dict(performance),
            "performance_gate_passed": gate_passed,
            "maximum_task_runtime_seconds": float(config["maximum_task_runtime_seconds"]),
            "maximum_peak_cuda_reserved_bytes": int(
                float(config["maximum_peak_cuda_reserved_gib"]) * 1024**3
            ),
            "completed_utc": utc_now(),
        }
    )
    local_completion = local_dir / "completion.json"
    _atomic_json(local_completion, completion)
    publish_file(local_completion, final_completion)
    shutil.rmtree(local_dir)
    print(
        f"[task durable] {task.task_id}: {published['bytes'] / 1024**3:.2f} GiB, "
        f"sha256={published['sha256'][:16]}...",
        flush=True,
    )
    return gate_passed


def require_a100_and_set_limit(config: Mapping[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select an A100 runtime")
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name.upper() or int(properties.total_memory) < 38 * 1024**3:
        raise RuntimeError(
            f"need an A100 with 40-GB-class memory; found {properties.name!r}, "
            f"{properties.total_memory / 1024**3:.2f} GiB"
        )
    ceiling = float(config["maximum_peak_cuda_reserved_gib"]) * 1024**3
    fraction = min(1.0, ceiling / float(properties.total_memory))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    print(
        f"[device] {properties.name}; total={properties.total_memory / 1024**3:.2f} GiB; "
        f"allocator_ceiling={ceiling / 1024**3:.2f} GiB; dtype=complex128",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--reused-root", type=Path, default=DEFAULT_REUSED_ROOT)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=Path("/content/hard_wall_tangent_gap_cocycle"))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-tasks", type=int)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    hashes = source_hashes()
    config_sha256 = canonical_hash(config)
    tasks = tasks_for_lane(args.lane, config)
    output_root = args.output_root.resolve()
    scratch_root = args.scratch_root.resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    print("[configuration] " + json.dumps(config, indent=2, sort_keys=True), flush=True)
    print(f"[bundle] {BUNDLE_ROOT}", flush=True)
    print(f"[lane] {args.lane}", flush=True)
    print(f"[reused input] {args.reused_root}", flush=True)
    print(f"[saved v1 import] {args.v1_root}", flush=True)
    print(f"[output] {output_root}", flush=True)
    print(f"[scratch] {scratch_root}", flush=True)
    print(
        f"[workload] lane_tasks={len(tasks)}, lane_samples={sum(t.sample_count for t in tasks)}, "
        f"campaign_tasks={EXPECTED_TASKS}, campaign_samples={EXPECTED_SAMPLES}, "
        f"batch_sizes={BATCH_SIZE_BY_NY}",
        flush=True,
    )
    print(f"[sources] {json.dumps(hashes, sort_keys=True)}", flush=True)

    inventory = [
        (
            task,
            *verified_complete(
                output_root,
                task,
                config_sha256=config_sha256,
                hashes=hashes,
                v1_root=args.v1_root,
            ),
        )
        for task in tasks
    ]
    complete_count = sum(complete for _, complete, _ in inventory)
    print(
        f"[resume] completed={complete_count}, pending={len(tasks) - complete_count}, total={len(tasks)}",
        flush=True,
    )
    for task, complete, reason in inventory:
        if not complete and reason != "missing":
            print(f"[resume warning] {task.task_id}: {reason}", flush=True)
    if args.report_only:
        return 0
    if args.max_new_tasks is not None and args.max_new_tasks < 0:
        raise ValueError("--max-new-tasks must be nonnegative")
    missing_imports = [task.task_id for task, complete, _ in inventory if task.import_v1 and not complete]
    if missing_imports:
        raise RuntimeError(
            "the pinned v1 batch is missing or invalid; restore it before v2 production: "
            + ", ".join(missing_imports)
        )
    performance_blocks = []
    for task, complete, _ in inventory:
        if complete and not task.import_v1:
            _, completion_path = result_paths(output_root, task)
            if json.loads(completion_path.read_text())["performance_gate_passed"] is not True:
                performance_blocks.append(task.task_id)
    if performance_blocks:
        raise RuntimeError("completed task exceeded the performance gate: " + ", ".join(performance_blocks))
    ensure_nvrtc_runtime_or_reexec()
    require_a100_and_set_limit(config)

    pending = [task for task, complete, _ in inventory if not complete and not task.import_v1]
    limit = len(pending) if args.max_new_tasks is None else int(args.max_new_tasks)
    completed_new = 0
    with tqdm(
        total=len(tasks),
        initial=complete_count,
        desc=f"A100 tangent lane {args.lane}",
        unit="batch",
    ) as bar:
        for task in pending:
            if completed_new >= limit:
                break
            require_storage(task, scratch_root, output_root)
            arrays, performance = run_task(
                task,
                config=config,
                reused_root=args.reused_root,
                scratch_root=scratch_root,
                config_sha256=config_sha256,
                hashes=hashes,
            )
            gate_passed = save_task(
                task,
                arrays,
                performance,
                output_root=output_root,
                scratch_root=scratch_root,
                config=config,
                config_sha256=config_sha256,
                hashes=hashes,
            )
            del arrays
            torch.cuda.empty_cache()
            completed_new += 1
            bar.update(1)
            bar.set_postfix(
                completed=complete_count + completed_new,
                pending=len(tasks) - complete_count - completed_new,
            )
            if not gate_passed:
                raise RuntimeError(
                    f"{task.task_id} is durable but exceeded the one-hour or 38-GiB gate; "
                    "the queue stopped before another task"
                )
    print(
        f"[summary] lane={args.lane}, verified_before={complete_count}, "
        f"newly_completed={completed_new}, remaining={len(tasks) - complete_count - completed_new}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
