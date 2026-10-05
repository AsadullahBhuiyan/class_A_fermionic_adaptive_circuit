#!/usr/bin/env python3
"""Staged, resumable CPU campaign for the L=24 soft-wall reference anisotropy."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import multiprocessing
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path[:0] = [str(SRC_ROOT), str(HERE)]

from fgtn.classA_U1FGTN import classA_U1FGTN
from reference_probe import ReferencePairObserver, anisotropy_from_matching_time, matching_time


CONFIG_PATH = HERE / "campaign_config.l24.v3.json"
CANONICAL_ENTRY = "classA_U1FGTN.run_markov_circuit"
SOURCE_PATHS = (
    REPO_ROOT / "src/fgtn/classA_U1FGTN.py",
    REPO_ROOT / "src/fgtn/occupied_frame.py",
    HERE / "reference_probe.py",
    Path(__file__),
    CONFIG_PATH,
    HERE / "launch_tmux.py",
)
STREAM_CODES = {"engine": 11, "post": 13, "probe": 17, "position": 19, "bootstrap": 23}
_WORKER_CPU_IDS: list[int] = []
_WORKER_MODEL_TEMPLATE: classA_U1FGTN | None = None
_WORKER_MODEL_SIGNATURE: dict[str, Any] | None = None


class ScienceUnresolved(RuntimeError):
    """A scientifically required gate was not resolved; this is not a code crash."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def payload_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_pickle_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pkl.tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in SOURCE_PATHS
        if path.exists()
    }


def git_metadata() -> dict[str, Any]:
    def call(*args: str) -> str:
        return subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": call("git", "rev-parse", "HEAD"), "dirty": bool(call("git", "status", "--short"))}


def initialize_run(root: Path, *, resume: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    for relative in ("status", "logs", "checkpoints/audit", "checkpoints/final", "raw/audit", "raw/final", "processed", "figures"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    config = load_config()
    config_hash = payload_hash(config)
    hashes = source_hashes()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise RuntimeError(f"Run root already contains a manifest; use --resume: {root}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("config_sha256") != config_hash:
            raise RuntimeError("Resume configuration differs from the immutable campaign configuration.")
        if manifest.get("source_sha256") != hashes:
            raise RuntimeError("Resume source hashes differ from the campaign manifest.")
        return manifest
    write_json_atomic(root / CONFIG_PATH.name, config)
    manifest = {
        "schema_version": 3,
        "campaign": config["campaign"],
        "matched_hard_wall_comparison_runs": [
            str(parent_run_path(config)),
            str(REPO_ROOT / config["reuse"]["failed_v2_run_relative"]),
        ],
        "campaign_reason": "Matched soft-domain-wall comparison: retain DW=True and alpha_top=1, alpha_triv=30 while setting dw_truncation=False and meas_slab_only=False.",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "run_root": str(root.resolve()),
        "config_sha256": config_hash,
        "source_sha256": hashes,
        "git": git_metadata(),
        "canonical_dynamics_entry_point": CANONICAL_ENTRY,
        "observable_cycle_coordinate": "u=0..T_follow after the second reference insertion",
        "checkpoint_schedule": "sparse physical-state checkpoints at T_eq-1; not a cycle-resolved observable",
        "stages": {},
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def update_manifest(root: Path, stage: str, status: str, **details: Any) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["updated_utc"] = utc_now()
    manifest["stages"][stage] = {"status": status, "recorded_utc": utc_now(), **jsonable(details)}
    write_json_atomic(path, manifest)


def record_execution(root: Path, cpu_ids: Sequence[int], workers: int) -> None:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["updated_utc"] = utc_now()
    manifest["execution"] = {
        "cpu_ids": [int(value) for value in cpu_ids],
        "workers": int(workers),
        "blas_threads": 1,
        "same_numa_node": True,
        "cpu_topology": {
            str(cpu): {"node": cpu_topology(int(cpu))[0], "package": cpu_topology(int(cpu))[1], "core": cpu_topology(int(cpu))[2]}
            for cpu in cpu_ids[:workers]
        },
    }
    write_json_atomic(path, manifest)


def stage_marker(root: Path, stage: str) -> Path:
    return root / "status" / stage / "SUCCESS"


def run_stage(root: Path, name: str, function: Any, *, resume: bool) -> Any:
    marker = stage_marker(root, name)
    if resume and marker.is_file():
        print(f"[resume] {name}: already complete", flush=True)
        return None
    marker.parent.mkdir(parents=True, exist_ok=True)
    update_manifest(root, name, "running")
    started = time.perf_counter()
    result = function()
    marker.write_text(utc_now() + "\n")
    update_manifest(root, name, "complete", elapsed_seconds=time.perf_counter() - started)
    return result


def build_model(nx: int, ny: int) -> classA_U1FGTN:
    config = load_config()
    geometry = config["geometry"]
    dynamics = config["dynamics"]
    if geometry["dw_truncation"] is not False or dynamics["meas_slab_only"] is not False:
        raise RuntimeError("This campaign requires dw_truncation=False and meas_slab_only=False.")
    model = classA_U1FGTN(
        nx,
        ny,
        DW=bool(geometry["DW"]),
        nshell=int(geometry["nshell"]),
        filling_frac=float(geometry["filling_frac"]),
        alpha_1=float(geometry["alpha_1"]),
        alpha_2=float(geometry["alpha_2"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_truncation=bool(geometry["dw_truncation"]),
    )
    model.construct_OW_projectors(
        nshell=int(geometry["nshell"]),
        DW=bool(geometry["DW"]),
        trial_orbitals=str(geometry["trial_orbitals"]),
        dw_truncation=bool(geometry["dw_truncation"]),
    )
    if bool(model.dw_truncation) != bool(geometry["dw_truncation"]):
        raise RuntimeError("Model construction changed the locked domain-wall truncation flag.")
    return model


def model_checkpoint_signature(model: classA_U1FGTN) -> dict[str, Any]:
    """Return the exact signature used by the locked production protocol."""
    config = load_config()
    meas_slab_only = bool(config["dynamics"]["meas_slab_only"])
    active = model.active_top_layer_indices(meas_slab_only=meas_slab_only)
    return model._markov_checkpoint_signature(
        sequence="random",
        active_top_layer_indices=active,
        meas_slab_only=model._meas_slab_only_effective(meas_slab_only),
        dw_exclude=None,
        perfect_correction=True,
        postselect_probability=0.0,
        n_a=0.5,
        p_gain=0.5,
        p_loss=0.5,
        physical_covariance_update=model._physical_covariance_update_label(
            model._normalize_physical_covariance_update("rank1")
        ),
        state_representation="physical_frame",
    )


def signature_diff(saved: Any, current: Any) -> dict[str, Any]:
    """Return a concise, JSON-safe field-level checkpoint-signature diff."""
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return {"signature": {"saved": jsonable(saved), "current": jsonable(current)}}
    normalized_saved = copy.deepcopy(saved)
    normalized_saved.setdefault("twist_x", 0.0)
    differences: dict[str, Any] = {}
    for key in sorted(set(normalized_saved) | set(current)):
        left, right = normalized_saved.get(key), current.get(key)
        if left != right:
            differences[key] = {"saved": jsonable(left), "current": jsonable(right)}
    return differences


def prepare_model_template() -> dict[str, Any]:
    """Build the canonical model once; every later stage reuses that exact object."""
    global _WORKER_MODEL_TEMPLATE, _WORKER_MODEL_SIGNATURE
    if _WORKER_MODEL_TEMPLATE is not None and _WORKER_MODEL_SIGNATURE is not None:
        current = model_checkpoint_signature(_WORKER_MODEL_TEMPLATE)
        differences = signature_diff(_WORKER_MODEL_SIGNATURE, current)
        if differences:
            raise RuntimeError(
                f"Canonical parent model template mutated between stages: {json.dumps(differences, sort_keys=True)}"
            )
        return copy.deepcopy(_WORKER_MODEL_SIGNATURE)
    config = load_config()
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    _WORKER_MODEL_TEMPLATE = build_model(nx, ny)
    _WORKER_MODEL_SIGNATURE = model_checkpoint_signature(_WORKER_MODEL_TEMPLATE)
    return copy.deepcopy(_WORKER_MODEL_SIGNATURE)


def model_from_template() -> classA_U1FGTN:
    if _WORKER_MODEL_TEMPLATE is None or _WORKER_MODEL_SIGNATURE is None:
        raise RuntimeError("Canonical worker model template was not prepared before task launch.")
    model = copy.deepcopy(_WORKER_MODEL_TEMPLATE)
    current = model_checkpoint_signature(model)
    differences = signature_diff(_WORKER_MODEL_SIGNATURE, current)
    if differences:
        raise RuntimeError(f"Deep-copied model template changed signature: {json.dumps(differences, sort_keys=True)}")
    return model


def assert_checkpoint_compatible(checkpoint: dict[str, Any], model: classA_U1FGTN, task_key: str) -> None:
    current = model_checkpoint_signature(model)
    differences = signature_diff(checkpoint.get("signature"), current)
    if differences:
        raise RuntimeError(
            f"Checkpoint signature mismatch before canonical engine resume for {task_key}: "
            f"{json.dumps(differences, sort_keys=True)}"
        )


def frame_diagnostics(checkpoint: dict[str, Any]) -> dict[str, Any]:
    native = checkpoint.get("native_state")
    if not isinstance(native, dict):
        raise RuntimeError("Frame checkpoint is missing native_state.")
    frame = np.asarray(native.get("frame"), dtype=np.complex128)
    if frame.ndim != 2 or not np.all(np.isfinite(frame)):
        raise RuntimeError("Checkpoint occupied frame is invalid or non-finite.")
    gram = frame.conj().T @ frame
    residual = float(np.linalg.norm(gram - np.eye(frame.shape[1]), ord="fro"))
    return {"frame_shape": list(frame.shape), "gram_residual": residual}


def seed_for(root_seed: int, sample: int, stream: str) -> int:
    sequence = np.random.SeedSequence([int(root_seed), int(sample), STREAM_CODES[stream]])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def parse_cpu_list(value: str) -> list[int]:
    result: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = (int(item) for item in token.split("-", 1))
            result.extend(range(start, stop + 1))
        else:
            result.append(int(token))
    if len(result) != len(set(result)):
        raise ValueError("CPU list contains duplicates.")
    return result


def cpu_topology(cpu: int) -> tuple[int, int, int]:
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    package = int((base / "physical_package_id").read_text())
    core = int((base / "core_id").read_text())
    node_paths = list(Path(f"/sys/devices/system/cpu/cpu{cpu}").glob("node[0-9]*"))
    node = int(node_paths[0].name[4:]) if node_paths else package
    return node, package, core


def validate_cpu_ids(cpu_ids: list[int], required: int) -> list[int]:
    if len(cpu_ids) < required:
        raise ValueError(f"At least {required} CPU IDs are required; received {len(cpu_ids)}.")
    selected = cpu_ids[:required]
    available = os.sched_getaffinity(0)
    if any(cpu not in available for cpu in selected):
        raise ValueError("Requested CPU lies outside the current affinity mask.")
    topology = [cpu_topology(cpu) for cpu in selected]
    if len({(package, core) for _, package, core in topology}) != required:
        raise ValueError("CPU list contains hyperthread siblings.")
    if len({node for node, _, _ in topology}) != 1:
        raise ValueError("All campaign CPUs must lie on one NUMA node.")
    return selected


def worker_initializer(cpu_ids: list[int]) -> None:
    global _WORKER_CPU_IDS
    _WORKER_CPU_IDS = list(cpu_ids)
    identity = multiprocessing.current_process()._identity
    ordinal = (identity[-1] - 1) if identity else os.getpid()
    cpu = _WORKER_CPU_IDS[ordinal % len(_WORKER_CPU_IDS)]
    os.sched_setaffinity(0, {int(cpu)})
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


def run_tasks(tasks: Sequence[dict[str, Any]], function: Any, workers: int, cpu_ids: list[int], label: str) -> list[dict[str, Any]]:
    if not tasks:
        return []
    rows: list[dict[str, Any]] = []
    worker_count = min(int(workers), len(tasks))
    if worker_count == 1:
        worker_initializer(cpu_ids[:1])
        for index, task in enumerate(tasks, 1):
            rows.append(function(task))
            print(f"[{label}] {index}/{len(tasks)}", flush=True)
        return rows
    if multiprocessing.get_start_method(allow_none=True) not in (None, "fork"):
        raise RuntimeError("This campaign requires POSIX fork for canonical model-template inheritance.")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("fork"),
        initializer=worker_initializer,
        initargs=(cpu_ids[:worker_count],),
    ) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index == len(tasks) or index % max(1, len(tasks) // 20) == 0:
                print(f"[{label}] {index}/{len(tasks)}", flush=True)
    return rows


def checkpoint_path(root: Path, stage: str, sample: int) -> Path:
    return root / "checkpoints" / stage / f"sample_{sample:04d}.pkl"


def branch_path(root: Path, stage: str, eq_multiplier: int, follow_cycles: int, separation: int, sample: int) -> Path:
    label = "space" if separation == 0 else f"time_dt{separation:02d}"
    return root / "raw" / stage / "branches" / f"eq_{eq_multiplier:02d}" / f"follow_{follow_cycles:03d}" / label / f"sample_{sample:04d}.npz"


def base_checkpoint_task(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    stage = str(task["stage"])
    sample = int(task["sample"])
    eq_multipliers = sorted(set(int(value) for value in task["eq_multipliers"]))
    config_hash = str(task["config_hash"])
    accepted_hashes = set(str(value) for value in task.get("accepted_config_hashes", [config_hash]))
    path = checkpoint_path(root, stage, sample)
    existing: dict[str, Any] | None = None
    if path.exists():
        with path.open("rb") as handle:
            existing = pickle.load(handle)
        if existing.get("config_sha256") not in accepted_hashes or existing.get("stage") != stage or int(existing.get("sample", -1)) != sample:
            raise RuntimeError(f"Checkpoint metadata mismatch: {path}")
        if all(str(value) in existing["checkpoints"] for value in eq_multipliers):
            return {"path": str(path), "sha256": sha256_file(path), "sample": sample, "resumed": True}

    config = load_config()
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    root_seed = int(config[stage]["root_seed"])
    engine_seed = seed_for(root_seed, sample, "engine")
    checkpoints = {} if existing is None else dict(existing["checkpoints"])
    missing = [value for value in eq_multipliers if str(value) not in checkpoints]
    if not missing:
        return {"path": str(path), "sha256": sha256_file(path), "sample": sample, "resumed": True}

    model = model_from_template()
    resume_state = None
    earlier = [int(key) for key in checkpoints if int(key) < max(missing)]
    if earlier:
        earlier_key = max(earlier)
        resume_state = copy.deepcopy(checkpoints[str(earlier_key)])
        assert_checkpoint_compatible(
            resume_state,
            model,
            f"stage={stage},sample={sample},eq={earlier_key},operation=extend",
        )
    target_cycles = {value * ny - 1: value for value in missing}

    def capture(*, cycle: int, state: dict[str, Any], **_: Any) -> None:
        if int(cycle) in target_cycles:
            checkpoints[str(target_cycles[int(cycle)])] = state

    model.run_markov_circuit(
        cycles=max(missing) * ny - 1,
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=engine_seed,
        state_representation="physical_frame",
        return_native_state=True,
        meas_slab_only=bool(config["dynamics"]["meas_slab_only"]),
        parallelize_samples=False,
        init_mode="default",
        checkpoint_state=resume_state,
        checkpoint_observer=capture,
    )
    if any(str(value) not in checkpoints for value in eq_multipliers):
        raise RuntimeError(f"Failed to capture every requested checkpoint for sample {sample}.")
    payload = {
        "schema": "reference_anisotropy_base_checkpoints_v1",
        "config_sha256": config_hash,
        "stage": stage,
        "sample": sample,
        "engine_seed": engine_seed,
        "checkpoint_cycles": {key: int(key) * ny - 1 for key in checkpoints},
        "checkpoints": checkpoints,
    }
    save_pickle_atomic(path, payload)
    return {"path": str(path), "sha256": sha256_file(path), "sample": sample, "resumed": False}


def reset_post_insertion_rngs(checkpoint: dict[str, Any], post_seed: int) -> dict[str, Any]:
    state = copy.deepcopy(checkpoint)
    children = np.random.SeedSequence(int(post_seed)).spawn(2)
    state["rng_states"]["schedule"] = copy.deepcopy(np.random.default_rng(children[0]).bit_generator.state)
    state["rng_states"]["dynamics"] = copy.deepcopy(np.random.default_rng(children[1]).bit_generator.state)
    return state


def validate_branch_file(path: Path, task: dict[str, Any]) -> None:
    config = load_config()
    expected_dw_truncation = bool(config["geometry"]["dw_truncation"])
    expected_meas_slab_only = bool(config["dynamics"]["meas_slab_only"])
    with np.load(path, allow_pickle=False) as data:
        expected = int(task["follow_cycles"]) + 1
        cycles = np.asarray(data["relative_cycles"])
        if cycles.shape != (expected,) or not np.array_equal(cycles, np.arange(expected)):
            raise RuntimeError(f"Incomplete relative-cycle coordinate in {path}")
        for key in ("mutual_information", "entropy_r1", "entropy_r2", "entropy_r12"):
            values = np.asarray(data[key])
            if values.shape != (expected,) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"Invalid {key} in {path}")
        if int(data["sample"]) != int(task["sample"]) or int(data["delta_tau"]) != int(task["separation"]):
            raise RuntimeError(f"Task identity mismatch in {path}")
        if str(data["config_sha256"]) != str(task["config_hash"]):
            raise RuntimeError(f"Configuration hash mismatch in {path}")
        if bool(data["dw_truncation"]) != expected_dw_truncation or bool(data["meas_slab_only"]) != expected_meas_slab_only:
            raise RuntimeError(f"Soft-wall protocol invariant failed in {path}")


def branch_task(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    stage = str(task["stage"])
    sample = int(task["sample"])
    eq_multiplier = int(task["eq_multiplier"])
    separation = int(task["separation"])
    follow_cycles = int(task["follow_cycles"])
    output = branch_path(root, stage, eq_multiplier, follow_cycles, separation, sample)
    if output.exists():
        validate_branch_file(output, task)
        return {"path": str(output), "sha256": sha256_file(output), "resumed": True}

    config = load_config()
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    root_seed = int(config[stage]["root_seed"])
    with checkpoint_path(root, stage, sample).open("rb") as handle:
        base = pickle.load(handle)
    accepted_hashes = set(str(value) for value in task.get("accepted_config_hashes", [task["config_hash"]]))
    if base.get("config_sha256") not in accepted_hashes:
        raise RuntimeError(
            f"Base-checkpoint configuration mismatch for stage={stage},sample={sample}: "
            f"saved={base.get('config_sha256')}, accepted={sorted(accepted_hashes)}"
        )
    checkpoint = reset_post_insertion_rngs(base["checkpoints"][str(eq_multiplier)], seed_for(root_seed, sample, "post"))

    model = model_from_template()
    assert_checkpoint_compatible(
        checkpoint,
        model,
        f"stage={stage},sample={sample},eq={eq_multiplier},separation={separation},follow={follow_cycles}",
    )
    walls = tuple(int(value) for value in model.DW_loc)
    wall_x = walls[sample % len(walls)]
    y1 = int(np.random.default_rng(seed_for(root_seed, sample, "position")).integers(0, ny))
    y2 = y1 if separation > 0 else (y1 + ny // 2) % ny
    tau1 = eq_multiplier * ny
    tau2 = tau1 + separation
    observer = ReferencePairObserver(
        nx=nx,
        ny=ny,
        tau1=tau1,
        tau2=tau2,
        follow_cycles=follow_cycles,
        first_site=(wall_x, y1),
        second_site=(wall_x, y2),
        rng=np.random.default_rng(seed_for(root_seed, sample, "probe")),
    )
    started = time.perf_counter()
    result = model.run_markov_circuit(
        cycles=tau2 + follow_cycles,
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=int(checkpoint["random_seed"]),
        state_representation="physical_frame",
        return_native_state=True,
        native_cycle_observer=observer,
        meas_slab_only=bool(config["dynamics"]["meas_slab_only"]),
        parallelize_samples=False,
        init_mode="default",
        checkpoint_state=checkpoint,
    )
    observer.assert_complete()
    payload = observer.payload()
    final = result["native_final"]
    gram_residual = float(final["gram_residual"])
    if int(final["physical_dimension"]) != 2 * nx * ny + 4 or gram_residual > float(config["validation"]["gram_residual_max"]):
        raise RuntimeError("Final augmented frame failed its dimension or Gram-residual check.")
    insertion_probability = np.empty((2, 2), dtype=np.float64)
    insertion_selected_probability = np.empty((2, 2), dtype=np.float64)
    insertion_outcome = np.empty((2, 2), dtype=np.bool_)
    sites = np.empty((2, 2), dtype=np.int64)
    for reference_index, reference in enumerate((payload["reference_one"], payload["reference_two"])):
        sites[reference_index] = (reference["x"], reference["y"])
        for orbital_index, event in enumerate(reference["events"]):
            insertion_probability[reference_index, orbital_index] = event["probability_occupied"]
            insertion_selected_probability[reference_index, orbital_index] = event["selected_probability"]
            insertion_outcome[reference_index, orbital_index] = event["outcome_occupied"]
    absolute_cycles = np.asarray(payload["cycles"], dtype=np.int64)
    save_npz_atomic(
        output,
        schema=np.asarray("gaussian_reference_pair_anisotropy_l24_v2"),
        config_sha256=np.asarray(task["config_hash"]),
        canonical_dynamics_entry_point=np.asarray(CANONICAL_ENTRY),
        stage=np.asarray(stage),
        sample=np.asarray(sample),
        nx=np.asarray(nx),
        ny=np.asarray(ny),
        eq_multiplier=np.asarray(eq_multiplier),
        tau1=np.asarray(tau1),
        tau2=np.asarray(tau2),
        delta_tau=np.asarray(separation),
        follow_cycles=np.asarray(follow_cycles),
        relative_cycles=absolute_cycles - tau2,
        absolute_cycles=absolute_cycles,
        mutual_information=np.asarray(payload["mutual_information"], dtype=np.float64),
        entropy_r1=np.asarray(payload["entropy_r1"], dtype=np.float64),
        entropy_r2=np.asarray(payload["entropy_r2"], dtype=np.float64),
        entropy_r12=np.asarray(payload["entropy_r12"], dtype=np.float64),
        insertion_probability=insertion_probability,
        insertion_selected_probability=insertion_selected_probability,
        insertion_outcome=insertion_outcome,
        sites=sites,
        wall_x=np.asarray(wall_x),
        y1=np.asarray(y1),
        engine_seed=np.asarray(base["engine_seed"], dtype=np.uint64),
        post_seed=np.asarray(seed_for(root_seed, sample, "post"), dtype=np.uint64),
        probe_seed=np.asarray(seed_for(root_seed, sample, "probe"), dtype=np.uint64),
        position_seed=np.asarray(seed_for(root_seed, sample, "position"), dtype=np.uint64),
        gram_residual=np.asarray(gram_residual),
        physical_dimension=np.asarray(final["physical_dimension"]),
        wall_seconds=np.asarray(time.perf_counter() - started),
        dw_truncation=np.asarray(bool(config["geometry"]["dw_truncation"])),
        meas_slab_only=np.asarray(bool(config["dynamics"]["meas_slab_only"])),
    )
    validate_branch_file(output, task)
    return {"path": str(output), "sha256": sha256_file(output), "resumed": False}


def benchmark_task(task: dict[str, Any]) -> dict[str, Any]:
    config = load_config()
    nx, ny = int(config["geometry"]["Nx"]), int(config["geometry"]["Ny"])
    model = build_model(nx, ny)
    started = time.perf_counter()
    result = model.run_markov_circuit(
        cycles=int(config["benchmark"]["cycles"]),
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=seed_for(int(config["benchmark"]["root_seed"]), int(task["sample"]), "engine"),
        state_representation="physical_frame",
        return_native_state=True,
        meas_slab_only=bool(config["dynamics"]["meas_slab_only"]),
        parallelize_samples=False,
        init_mode="default",
    )
    final = result["native_final"]
    residual = float(final["gram_residual"])
    if residual > float(config["validation"]["gram_residual_max"]):
        raise RuntimeError("Benchmark trajectory violated the Gram-residual ceiling.")
    return {"sample": int(task["sample"]), "elapsed_seconds": time.perf_counter() - started, "gram_residual": residual}


def parent_run_path(config: dict[str, Any]) -> Path:
    relative = Path(str(config["reuse"]["parent_run_relative"]))
    return relative if relative.is_absolute() else REPO_ROOT / relative


def validate_parent_run(config: dict[str, Any]) -> dict[str, Any]:
    parent = parent_run_path(config)
    manifest_path = parent / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Configured parent run is missing its manifest: {parent}")
    manifest = json.loads(manifest_path.read_text())
    parent_config_path = parent / "campaign_config.l24.v1.json"
    if not parent_config_path.exists():
        raise RuntimeError(f"Parent run is missing its immutable configuration copy: {parent_config_path}")
    parent_config = json.loads(parent_config_path.read_text())
    for section in ("canonical_dynamics_entry_point", "geometry", "dynamics", "audit", "final", "statistics", "benchmark", "parallel"):
        if parent_config.get(section) != config.get(section):
            raise RuntimeError(f"Parent-run physics/protocol mismatch in section {section!r}.")
    current_sources = source_hashes()
    for relative in ("src/fgtn/classA_U1FGTN.py", "src/fgtn/occupied_frame.py", "00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot/reference_probe.py"):
        if manifest.get("source_sha256", {}).get(relative) != current_sources.get(relative):
            raise RuntimeError(f"Parent-run source mismatch for {relative}.")
    return {"path": parent, "manifest": manifest, "config": parent_config}


def accepted_checkpoint_hashes(root: Path) -> list[str]:
    hashes = [payload_hash(load_config())]
    record = root / "processed" / "reused_audit_checkpoints.json"
    if record.exists():
        hashes.append(str(json.loads(record.read_text())["parent_config_sha256"]))
    return sorted(set(hashes))


def import_parent_benchmark(root: Path) -> dict[str, Any]:
    config = load_config()
    parent_info = validate_parent_run(config)
    source = parent_info["path"] / "processed" / "benchmark.json"
    if not source.exists():
        raise RuntimeError(f"Reusable parent benchmark is missing: {source}")
    payload = json.loads(source.read_text())
    if int(payload.get("recommended_workers", -1)) != 20:
        raise RuntimeError("Validated parent benchmark did not select the expected 20 workers.")
    destination = root / "processed" / "benchmark.json"
    write_json_atomic(destination, payload)
    write_json_atomic(
        root / "processed" / "reused_benchmark.json",
        {
            "parent_run": str(parent_info["path"]),
            "source": str(source),
            "source_sha256": sha256_file(source),
            "destination_sha256": sha256_file(destination),
            "reason": "v1 throughput benchmark is upstream of the checkpoint-resume failure",
        },
    )
    return payload


def import_parent_audit_checkpoints(root: Path) -> dict[str, Any]:
    config = load_config()
    parent_info = validate_parent_run(config)
    expected_signature = prepare_model_template()
    parent_hash = str(parent_info["manifest"]["config_sha256"])
    rows = []
    for sample in range(int(config["audit"]["samples"])):
        source = checkpoint_path(parent_info["path"], "audit", sample)
        if not source.exists():
            raise RuntimeError(f"Reusable parent checkpoint is missing: {source}")
        with source.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("schema") != "reference_anisotropy_base_checkpoints_v1":
            raise RuntimeError(f"Unexpected parent checkpoint schema: {source}")
        if payload.get("config_sha256") != parent_hash or payload.get("stage") != "audit" or int(payload.get("sample", -1)) != sample:
            raise RuntimeError(f"Parent checkpoint metadata mismatch: {source}")
        for multiplier in config["audit"]["equilibration_multipliers"]:
            checkpoint = payload.get("checkpoints", {}).get(str(multiplier))
            if checkpoint is None:
                raise RuntimeError(f"Parent checkpoint lacks {multiplier}L state: {source}")
            differences = signature_diff(checkpoint.get("signature"), expected_signature)
            if differences:
                raise RuntimeError(f"Parent checkpoint signature mismatch {source} eq={multiplier}: {json.dumps(differences, sort_keys=True)}")
            expected_cycle = int(multiplier) * int(config["geometry"]["Ny"]) - 1
            if int(checkpoint.get("completed_cycles", -1)) != expected_cycle:
                raise RuntimeError(f"Parent checkpoint cycle mismatch {source} eq={multiplier}.")
            if set(checkpoint.get("rng_states", {})) != {"initialization", "exterior", "schedule", "dynamics"}:
                raise RuntimeError(f"Parent checkpoint RNG state is incomplete: {source}")
            diagnostic = frame_diagnostics(checkpoint)
            if diagnostic["frame_shape"][0] != 2 * int(config["geometry"]["Nx"]) * int(config["geometry"]["Ny"]):
                raise RuntimeError(f"Parent checkpoint physical frame dimension is wrong: {source}")
            if diagnostic["gram_residual"] > float(config["validation"]["gram_residual_max"]):
                raise RuntimeError(f"Parent checkpoint Gram residual is too large: {source}")
        destination = checkpoint_path(root, "audit", sample)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise RuntimeError(f"Existing imported checkpoint differs from parent: {destination}")
        else:
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
        rows.append({"sample": sample, "source": str(source), "destination": str(destination), "sha256": sha256_file(destination)})
    record = {
        "parent_run": str(parent_info["path"]),
        "parent_config_sha256": parent_hash,
        "files": rows,
        "validated_signature": expected_signature,
        "partial_reference_branches_reused": False,
        "reason": "v1 burn-in checkpoints are upstream of the failed branch-resume task and pass strict v2 validation",
    }
    write_json_atomic(root / "processed" / "reused_audit_checkpoints.json", record)
    return record


def resume_preflight_task(task: dict[str, Any]) -> dict[str, Any]:
    root = Path(task["root"])
    sample = int(task["sample"])
    config = load_config()
    with checkpoint_path(root, "audit", sample).open("rb") as handle:
        payload = pickle.load(handle)
    checkpoint = copy.deepcopy(payload["checkpoints"]["2"])
    model = model_from_template()
    task_key = f"stage=preflight,sample={sample},eq=2,operation=one-cycle-resume"
    assert_checkpoint_compatible(checkpoint, model, task_key)
    result = model.run_markov_circuit(
        cycles=int(checkpoint["completed_cycles"]) + 1,
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=int(checkpoint["random_seed"]),
        state_representation="physical_frame",
        return_native_state=True,
        meas_slab_only=bool(config["dynamics"]["meas_slab_only"]),
        parallelize_samples=False,
        init_mode="default",
        checkpoint_state=checkpoint,
    )
    final = result["native_final"]
    residual = float(final["gram_residual"])
    if int(final["physical_dimension"]) != 2 * int(config["geometry"]["Nx"]) * int(config["geometry"]["Ny"]) or residual > float(config["validation"]["gram_residual_max"]):
        raise RuntimeError(f"Production-shaped resume failed frame diagnostics for {task_key}.")
    return {"sample": sample, "gram_residual": residual, "checkpoint_sha256": sha256_file(checkpoint_path(root, "audit", sample))}


def ensure_base_checkpoints(root: Path, stage: str, samples: int, eq_multipliers: Sequence[int], workers: int, cpu_ids: list[int]) -> list[dict[str, Any]]:
    config_hash = payload_hash(load_config())
    tasks = [
        {
            "root": str(root),
            "stage": stage,
            "sample": sample,
            "eq_multipliers": list(eq_multipliers),
            "config_hash": config_hash,
            "accepted_config_hashes": accepted_checkpoint_hashes(root),
        }
        for sample in range(samples)
    ]
    rows = run_tasks(tasks, base_checkpoint_task, workers, cpu_ids, f"{stage}-checkpoints")
    write_json_atomic(root / "checkpoints" / stage / "index.json", {"eq_multipliers": list(eq_multipliers), "files": rows})
    return rows


def branch_tasks(root: Path, stage: str, samples: int, eq_multipliers: Sequence[int], follow_cycles: int, separations: Sequence[int], workers: int, cpu_ids: list[int]) -> list[dict[str, Any]]:
    config_hash = payload_hash(load_config())
    tasks = [
        {
            "root": str(root),
            "stage": stage,
            "sample": sample,
            "eq_multiplier": eq_multiplier,
            "follow_cycles": follow_cycles,
            "separation": separation,
            "config_hash": config_hash,
            "accepted_config_hashes": accepted_checkpoint_hashes(root),
        }
        for eq_multiplier in eq_multipliers
        for separation in sorted(set(int(value) for value in separations))
        for sample in range(samples)
    ]
    rows = run_tasks(tasks, branch_task, workers, cpu_ids, f"{stage}-branches")
    index_path = root / "raw" / stage / "raw_index.json"
    existing = json.loads(index_path.read_text()).get("files", []) if index_path.exists() else []
    merged = {row["path"]: row for row in existing}
    merged.update({row["path"]: row for row in rows})
    write_json_atomic(index_path, {"files": sorted(merged.values(), key=lambda row: row["path"])})
    return rows


def load_series(root: Path, stage: str, eq_multiplier: int, follow_cycles: int, separation: int, samples: int) -> np.ndarray:
    rows = []
    config_hash = payload_hash(load_config())
    for sample in range(samples):
        path = branch_path(root, stage, eq_multiplier, follow_cycles, separation, sample)
        task = {"follow_cycles": follow_cycles, "sample": sample, "separation": separation, "config_hash": config_hash}
        validate_branch_file(path, task)
        with np.load(path, allow_pickle=False) as data:
            rows.append(np.asarray(data["mutual_information"], dtype=np.float64))
    return np.stack(rows, axis=0)


def window_rows(series: np.ndarray, ny: int, window_number: int) -> np.ndarray:
    start = 1 + (int(window_number) - 1) * int(ny)
    stop = 1 + int(window_number) * int(ny)
    if stop > series.shape[1]:
        raise ValueError(f"Window {window_number} is unavailable for series shape {series.shape}.")
    return np.mean(series[:, start:stop], axis=1)


def bracket_details(separations: Sequence[int], temporal_values: Sequence[float], spatial_value: float) -> dict[str, Any] | None:
    seps = np.asarray(separations, dtype=int)
    vals = np.asarray(temporal_values, dtype=float)
    order = np.argsort(seps)
    seps, vals = seps[order], vals[order]
    if seps.size < 2 or not np.all(np.isfinite(vals)) or not np.isfinite(spatial_value):
        return None
    difference = vals - float(spatial_value)
    if difference[0] <= 0:
        return None
    for index in range(seps.size - 1):
        if difference[index] == 0:
            return {"lower": int(seps[index]), "upper": int(seps[index]), "time_star": float(seps[index])}
        if difference[index] > 0 and difference[index + 1] < 0:
            fraction = difference[index] / (difference[index] - difference[index + 1])
            return {
                "lower": int(seps[index]),
                "upper": int(seps[index + 1]),
                "time_star": float(seps[index] + fraction * (seps[index + 1] - seps[index])),
            }
    if difference[-1] == 0:
        return {"lower": int(seps[-1]), "upper": int(seps[-1]), "time_star": float(seps[-1])}
    return None


def grid_rows(root: Path, stage: str, eq_multiplier: int, follow_cycles: int, separations: Sequence[int], samples: int, window_number: int) -> dict[int, np.ndarray]:
    ny = int(load_config()["geometry"]["Ny"])
    return {
        int(separation): window_rows(load_series(root, stage, eq_multiplier, follow_cycles, int(separation), samples), ny, window_number)
        for separation in sorted(set(int(value) for value in separations))
    }


def grid_summary(rows: dict[int, np.ndarray]) -> dict[str, Any]:
    separations = sorted(value for value in rows if value > 0)
    spatial = float(np.mean(rows[0]))
    temporal = [float(np.mean(rows[value])) for value in separations]
    bracket = bracket_details(separations, temporal, spatial)
    return {"spatial_mean": spatial, "separations": separations, "temporal_means": temporal, "bracket": bracket}


def paired_point_gate(smaller: np.ndarray, larger: np.ndarray, rng: np.random.Generator, draws: int, shift_fraction: float) -> dict[str, Any]:
    smaller = np.asarray(smaller, dtype=float)
    larger = np.asarray(larger, dtype=float)
    if smaller.shape != larger.shape or smaller.ndim != 1:
        raise ValueError("Paired gate inputs must be matching vectors.")
    difference = larger - smaller
    indices = rng.integers(0, difference.size, size=(int(draws), difference.size))
    bootstrap = np.mean(difference[indices], axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    larger_sem = float(np.std(larger, ddof=1) / np.sqrt(larger.size))
    point_shift = float(np.mean(difference))
    threshold = float(shift_fraction) * larger_sem
    passed = bool(low <= 0 <= high and abs(point_shift) <= threshold)
    return {"passed": passed, "point_shift": point_shift, "ci_low": low, "ci_high": high, "larger_sem": larger_sem, "threshold": threshold}


def bootstrap_tstar(rows: dict[int, np.ndarray], draws: int, rng: np.random.Generator) -> np.ndarray:
    separations = np.asarray(sorted(value for value in rows if value > 0), dtype=float)
    temporal = np.stack([rows[int(value)] for value in separations], axis=1)
    spatial = rows[0]
    output = np.full(int(draws), np.nan, dtype=float)
    for draw in range(int(draws)):
        indices = rng.integers(0, spatial.size, size=spatial.size)
        output[draw] = matching_time(separations, np.mean(temporal[indices], axis=0), float(np.mean(spatial[indices])))
    return output


def tstar_gate(smaller: dict[int, np.ndarray], larger: dict[int, np.ndarray], rng: np.random.Generator, draws: int, shift_fraction: float, minimum_resolved: float) -> dict[str, Any]:
    seps = sorted(set(smaller) & set(larger))
    small = {key: smaller[key] for key in seps}
    large = {key: larger[key] for key in seps}
    small_boot = bootstrap_tstar(small, draws, rng)
    large_boot = bootstrap_tstar(large, draws, rng)
    finite = np.isfinite(small_boot) & np.isfinite(large_boot)
    resolved = float(np.mean(finite))
    small_summary, large_summary = grid_summary(small), grid_summary(large)
    small_point = None if small_summary["bracket"] is None else float(small_summary["bracket"]["time_star"])
    large_point = None if large_summary["bracket"] is None else float(large_summary["bracket"]["time_star"])
    if not np.any(finite) or small_point is None or large_point is None:
        return {"passed": False, "resolved_fraction": resolved, "reason": "unresolved crossing"}
    differences = large_boot[finite] - small_boot[finite]
    low, high = np.percentile(differences, [2.5, 97.5])
    larger_sem = float(np.std(large_boot[finite], ddof=1))
    shift = large_point - small_point
    threshold = float(shift_fraction) * larger_sem
    passed = bool(resolved >= minimum_resolved and low <= 0 <= high and abs(shift) <= threshold)
    return {"passed": passed, "resolved_fraction": resolved, "point_shift": shift, "ci_low": low, "ci_high": high, "larger_sem": larger_sem, "threshold": threshold}


def setting_gate(smaller: dict[int, np.ndarray], larger: dict[int, np.ndarray], rng: np.random.Generator, config: dict[str, Any]) -> dict[str, Any]:
    draws = int(config["statistics"]["audit_bootstrap_draws"])
    shift_fraction = float(config["statistics"]["maximum_shift_in_larger_sem"])
    minimum_resolved = float(config["audit"]["minimum_joint_bootstrap_resolved_fraction"])
    larger_summary = grid_summary(larger)
    if larger_summary["bracket"] is None:
        return {"passed": False, "reason": "larger setting has no ordered crossing"}
    endpoints = sorted(set((0, int(larger_summary["bracket"]["lower"]), int(larger_summary["bracket"]["upper"]))))
    if any(endpoint not in smaller or endpoint not in larger for endpoint in endpoints):
        return {"passed": False, "reason": "crossing endpoint absent from paired grids"}
    points = {
        str(endpoint): paired_point_gate(smaller[endpoint], larger[endpoint], rng, draws, shift_fraction)
        for endpoint in endpoints
    }
    time_gate = tstar_gate(smaller, larger, rng, draws, shift_fraction, minimum_resolved)
    return {"passed": bool(all(value["passed"] for value in points.values()) and time_gate["passed"]), "point_gates": points, "time_star_gate": time_gate}


def stage_preflight(root: Path, cpu_ids: list[int]) -> None:
    config = load_config()
    if config["canonical_dynamics_entry_point"] != CANONICAL_ENTRY:
        raise RuntimeError("Locked canonical dynamics entry point is incorrect.")
    if config["geometry"]["dw_truncation"] is not False or config["dynamics"]["meas_slab_only"] is not False:
        raise RuntimeError("Soft-wall invariants are not locked false.")
    command = [sys.executable, "-m", "pytest", "-q", str(HERE / "tests")]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    (root / "logs" / "preflight_pytest.log").write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode:
        raise RuntimeError("Preflight pytest failed; see logs/preflight_pytest.log.")

    model = build_model(4, 4)
    observer = ReferencePairObserver(nx=4, ny=4, tau1=2, tau2=3, follow_cycles=2, first_site=(model.DW_loc[0], 0), second_site=(model.DW_loc[0], 0), rng=np.random.default_rng(91))
    result = model.run_markov_circuit(
        cycles=5,
        samples=1,
        sequence="random",
        perfect_correction=True,
        G_history=False,
        save=False,
        progress=False,
        random_seed=90,
        state_representation="physical_frame",
        return_native_state=True,
        native_cycle_observer=observer,
        meas_slab_only=bool(config["dynamics"]["meas_slab_only"]),
        parallelize_samples=False,
        init_mode="default",
    )
    observer.assert_complete()
    payload = observer.payload()
    if not np.all(np.isfinite(payload["mutual_information"])) or int(result["native_final"]["physical_dimension"]) != 36:
        raise RuntimeError("Soft-wall preflight trajectory failed reference-observable validation.")
    prepare_model_template()
    resume_workers = int(config["validation"]["production_resume_workers"])
    ensure_base_checkpoints(root, "audit", resume_workers, [2], resume_workers, cpu_ids)
    resume_rows = run_tasks(
        [{"root": str(root), "sample": sample} for sample in range(resume_workers)],
        resume_preflight_task,
        resume_workers,
        cpu_ids,
        "preflight-l24-resume",
    )
    write_json_atomic(
        root / "status" / "preflight.json",
        {
            "pytest_command": command,
            "smoke": "passed",
            "production_l24_parallel_resume": "passed",
            "production_resume_workers": resume_workers,
            "resume_rows": resume_rows,
            "checkpoint_source": "generated_in_this_soft_wall_run",
            "dw_truncation": False,
            "meas_slab_only": False,
        },
    )


def stage_benchmark(root: Path, cpu_ids: list[int]) -> dict[str, Any]:
    config = load_config()
    if bool(config.get("reuse", {}).get("benchmark")):
        return import_parent_benchmark(root)
    benchmark = config["benchmark"]
    rows = []
    for requested in benchmark["worker_counts"]:
        workers = int(requested)
        tasks = [{"sample": sample} for sample in range(int(benchmark["samples"]))]
        started = time.perf_counter()
        results = run_tasks(tasks, benchmark_task, workers, cpu_ids, f"benchmark-{workers}")
        elapsed = time.perf_counter() - started
        rows.append({
            "requested_workers": workers,
            "elapsed_seconds": elapsed,
            "samples_per_hour": 3600.0 * len(results) / elapsed,
            "task_elapsed_mean": float(np.mean([row["elapsed_seconds"] for row in results])),
            "task_elapsed_max": float(np.max([row["elapsed_seconds"] for row in results])),
        })
    best = max(row["samples_per_hour"] for row in rows)
    viable = [row for row in rows if row["samples_per_hour"] >= float(benchmark["within_best_fraction"]) * best]
    chosen = min(viable, key=lambda row: row["requested_workers"])
    payload = {"rows": rows, "selection_rule": "smallest worker count within 5% of best throughput", "recommended_workers": chosen["requested_workers"], "cpu_ids": cpu_ids}
    write_json_atomic(root / "processed" / "benchmark.json", payload)
    return payload


def refinement_separations(summaries: dict[int, dict[str, Any]]) -> list[int]:
    missing: set[int] = set()
    for summary in summaries.values():
        bracket = summary.get("bracket")
        if bracket is not None:
            missing.update(range(int(bracket["lower"]) + 1, int(bracket["upper"])))
    return sorted(missing)


def select_follow_multiplier(gates: dict[str, dict[str, Any]], extension_multiplier: int = 4) -> int | None:
    if gates.get("1_vs_2", {}).get("passed"):
        return 2
    if gates.get("2_vs_3", {}).get("passed"):
        return 3
    if gates.get("3_vs_4", {}).get("passed"):
        return int(extension_multiplier)
    return None


def select_equilibration_multiplier(eq_multipliers: Sequence[int], gates: dict[str, dict[str, Any]], extension_gate: dict[str, Any] | None = None) -> int | None:
    ordered = [int(value) for value in eq_multipliers]
    for smaller, larger in zip(ordered[:-1], ordered[1:]):
        if gates.get(f"{smaller}_vs_{larger}", {}).get("passed"):
            return smaller
    if extension_gate is not None and extension_gate.get("passed"):
        return ordered[-1]
    return None


def audit_rows_for_eq(root: Path, eq_multiplier: int, follow_cycles: int, separations: Sequence[int], window_number: int) -> dict[int, np.ndarray]:
    samples = int(load_config()["audit"]["samples"])
    return grid_rows(root, "audit", eq_multiplier, follow_cycles, separations, samples, window_number)


def evaluate_equilibration(root: Path, eq_multipliers: Sequence[int], follow_cycles: int, separations: Sequence[int]) -> tuple[int | None, dict[str, Any]]:
    config = load_config()
    window_number = int(config["audit"]["initial_follow_multiplier"])
    rows = {eq: audit_rows_for_eq(root, eq, follow_cycles, separations, window_number) for eq in eq_multipliers}
    summaries = {str(eq): grid_summary(values) for eq, values in rows.items()}
    gates: dict[str, Any] = {}
    for index in range(len(eq_multipliers) - 1):
        smaller, larger = int(eq_multipliers[index]), int(eq_multipliers[index + 1])
        rng = np.random.default_rng(seed_for(int(config["audit"]["root_seed"]), 1000 + smaller * 10 + larger, "bootstrap"))
        gate = setting_gate(rows[smaller], rows[larger], rng, config)
        gates[f"{smaller}_vs_{larger}"] = gate
    selected = select_equilibration_multiplier(eq_multipliers, gates)
    return selected, {"summaries": summaries, "gates": gates}


def evaluate_follow(root: Path, eq_multiplier: int, follow_cycles: int, separations: Sequence[int], smaller_window: int, larger_window: int) -> dict[str, Any]:
    config = load_config()
    smaller = audit_rows_for_eq(root, eq_multiplier, follow_cycles, separations, smaller_window)
    larger = audit_rows_for_eq(root, eq_multiplier, follow_cycles, separations, larger_window)
    rng = np.random.default_rng(seed_for(int(config["audit"]["root_seed"]), 2000 + 10 * smaller_window + larger_window, "bootstrap"))
    return setting_gate(smaller, larger, rng, config)


def stage_audit(root: Path, workers: int, cpu_ids: list[int]) -> dict[str, Any]:
    config = load_config()
    prepare_model_template()
    if bool(config.get("reuse", {}).get("audit_checkpoints")):
        import_parent_audit_checkpoints(root)
    audit = config["audit"]
    ny = int(config["geometry"]["Ny"])
    samples = int(audit["samples"])
    eqs = [int(value) for value in audit["equilibration_multipliers"]]
    follow_cycles = int(audit["initial_follow_multiplier"]) * ny
    coarse = [0] + [int(value) for value in audit["coarse_temporal_separations"]]

    ensure_base_checkpoints(root, "audit", samples, eqs, workers, cpu_ids)
    branch_tasks(root, "audit", samples, eqs, follow_cycles, coarse, workers, cpu_ids)
    coarse_summaries = {
        eq: grid_summary(audit_rows_for_eq(root, eq, follow_cycles, coarse, int(audit["initial_follow_multiplier"])))
        for eq in eqs
    }
    refinement = refinement_separations(coarse_summaries)
    if refinement:
        branch_tasks(root, "audit", samples, eqs, follow_cycles, refinement, workers, cpu_ids)
    separations = sorted(set(coarse + refinement))
    selected_eq, equilibration_diagnostics = evaluate_equilibration(root, eqs, follow_cycles, separations)

    if selected_eq is None:
        extension = int(audit["equilibration_extension_multiplier"])
        print(f"[audit] extending equilibration to {extension}L", flush=True)
        ensure_base_checkpoints(root, "audit", samples, [extension], workers, cpu_ids)
        branch_tasks(root, "audit", samples, [extension], follow_cycles, coarse, workers, cpu_ids)
        extension_summary = grid_summary(audit_rows_for_eq(root, extension, follow_cycles, coarse, int(audit["initial_follow_multiplier"])))
        extension_refinement = refinement_separations({extension: extension_summary})
        all_extension_refinement = sorted(set(extension_refinement) | set(refinement))
        if all_extension_refinement:
            branch_tasks(root, "audit", samples, [eqs[-1], extension], follow_cycles, all_extension_refinement, workers, cpu_ids)
        extension_separations = sorted(set(coarse + all_extension_refinement))
        rng = np.random.default_rng(seed_for(int(audit["root_seed"]), 1000 + eqs[-1] * 10 + extension, "bootstrap"))
        smaller_rows = audit_rows_for_eq(root, eqs[-1], follow_cycles, extension_separations, int(audit["initial_follow_multiplier"]))
        larger_rows = audit_rows_for_eq(root, extension, follow_cycles, extension_separations, int(audit["initial_follow_multiplier"]))
        extension_gate = setting_gate(smaller_rows, larger_rows, rng, config)
        equilibration_diagnostics["extension"] = {"multiplier": extension, "summary": grid_summary(larger_rows), "gate": extension_gate}
        selected_eq = select_equilibration_multiplier(eqs, equilibration_diagnostics["gates"], extension_gate)
        if selected_eq is not None:
            separations = extension_separations

    if selected_eq is None:
        decision = {"status": "needs_attention", "reason": "equilibration did not converge through the 8L extension", "equilibration": equilibration_diagnostics}
        write_json_atomic(root / "processed" / "audit_decision.json", decision)
        return decision

    follow_2 = evaluate_follow(root, selected_eq, follow_cycles, separations, 1, 2)
    follow_3 = evaluate_follow(root, selected_eq, follow_cycles, separations, 2, 3)
    follow_diagnostics: dict[str, Any] = {"1_vs_2": follow_2, "2_vs_3": follow_3}
    selected_follow = select_follow_multiplier(follow_diagnostics, int(audit["follow_extension_multiplier"]))
    if selected_follow is None:
        extension_follow = int(audit["follow_extension_multiplier"])
        extended_cycles = extension_follow * ny
        print(f"[audit] extending post-insertion follow to {extension_follow}L", flush=True)
        branch_tasks(root, "audit", samples, [selected_eq], extended_cycles, separations, workers, cpu_ids)
        follow_4 = evaluate_follow(root, selected_eq, extended_cycles, separations, 3, 4)
        follow_diagnostics["3_vs_4"] = follow_4
        selected_follow = select_follow_multiplier(follow_diagnostics, extension_follow)

    if selected_follow is None:
        decision = {
            "status": "needs_attention",
            "reason": "post-insertion plateau did not converge through the 4L extension",
            "selected_equilibration_multiplier": selected_eq,
            "equilibration": equilibration_diagnostics,
            "follow": follow_diagnostics,
        }
        write_json_atomic(root / "processed" / "audit_decision.json", decision)
        return decision

    selected_follow_cycles = selected_follow * ny
    selected_rows = audit_rows_for_eq(
        root,
        selected_eq,
        selected_follow_cycles if selected_follow == int(audit["follow_extension_multiplier"]) else follow_cycles,
        separations,
        selected_follow,
    )
    selected_summary = grid_summary(selected_rows)
    if selected_summary["bracket"] is None:
        decision = {"status": "needs_attention", "reason": "accepted cycle setting has no ordered t_star bracket", "equilibration": equilibration_diagnostics, "follow": follow_diagnostics}
        write_json_atomic(root / "processed" / "audit_decision.json", decision)
        return decision
    decision = {
        "status": "passed",
        "selected_equilibration_multiplier": selected_eq,
        "selected_equilibration_cycles": selected_eq * ny,
        "selected_follow_multiplier": selected_follow,
        "selected_follow_cycles": selected_follow_cycles,
        "audit_temporal_separations": [value for value in separations if value > 0],
        "audit_time_star": selected_summary["bracket"]["time_star"],
        "audit_alpha": anisotropy_from_matching_time(ny, selected_summary["bracket"]["time_star"]),
        "equilibration": equilibration_diagnostics,
        "follow": follow_diagnostics,
    }
    write_json_atomic(root / "processed" / "audit_decision.json", decision)
    return decision


def final_statistics(root: Path, eq_multiplier: int, follow_cycles: int, separations: Sequence[int]) -> dict[str, Any]:
    config = load_config()
    final = config["final"]
    ny = int(config["geometry"]["Ny"])
    samples = int(final["samples"])
    window_number = follow_cycles // ny
    rows = grid_rows(root, "final", eq_multiplier, follow_cycles, separations, samples, window_number)
    summary = grid_summary(rows)
    if summary["bracket"] is None:
        return {"status": "unresolved", "reason": "no ordered downward crossing", "grid": summary}
    bracket = summary["bracket"]
    if int(bracket["upper"]) - int(bracket["lower"]) > 1:
        return {"status": "unresolved", "reason": "crossing is not bracketed by adjacent integer cycles", "grid": summary}
    draws = int(final["bootstrap_draws"])
    rng = np.random.default_rng(seed_for(int(final["root_seed"]), 0, "bootstrap"))
    boot_time = bootstrap_tstar(rows, draws, rng)
    boot_alpha = np.asarray([anisotropy_from_matching_time(ny, value) for value in boot_time])
    finite = np.isfinite(boot_time) & np.isfinite(boot_alpha)
    resolved = float(np.mean(finite))
    time_star = float(bracket["time_star"])
    alpha = anisotropy_from_matching_time(ny, time_star)
    distribution_path = root / "processed" / "final_distributions.npz"
    temporal_seps = np.asarray(sorted(value for value in rows if value > 0), dtype=int)
    temporal_rows = np.stack([rows[int(value)] for value in temporal_seps], axis=1)
    save_npz_atomic(
        distribution_path,
        spatial_by_sample=rows[0],
        temporal_separations=temporal_seps,
        temporal_by_sample=temporal_rows,
        bootstrap_time_star=boot_time,
        bootstrap_alpha=boot_alpha,
    )

    def stats(values: np.ndarray) -> dict[str, Any]:
        mean = float(np.mean(values))
        sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
        cv = None if mean <= 0 else float(np.std(values, ddof=1) / mean)
        return {"mean": mean, "sem": sem, "coefficient_of_variation": cv, "minimum": float(np.min(values)), "maximum": float(np.max(values))}

    config_stats = {"space": stats(rows[0])}
    config_stats.update({f"time_dt{value}": stats(rows[value]) for value in temporal_seps})
    plateau_drift = {}
    for separation in [0] + temporal_seps.tolist():
        full = load_series(root, "final", eq_multiplier, follow_cycles, int(separation), samples)
        if window_number >= 2:
            previous = window_rows(full, ny, window_number - 1)
            current = window_rows(full, ny, window_number)
            plateau_drift[str(separation)] = stats(current - previous)
    wall_diagnostics = {
        "wall_0_spatial": stats(rows[0][0::2]),
        "wall_1_spatial": stats(rows[0][1::2]),
    }
    calibrated = resolved >= float(final["minimum_bootstrap_resolved_fraction"])
    return {
        "status": "calibrated" if calibrated else "unresolved",
        "reason": None if calibrated else "bootstrap crossing resolution is below the locked 95% threshold; use S=200 before quoting production uncertainty",
        "samples": samples,
        "independent_sampling_unit": "complete Born-rule circuit trajectory",
        "time_star": time_star,
        "time_star_ci_low": float(np.percentile(boot_time[finite], 2.5)) if np.any(finite) else None,
        "time_star_ci_high": float(np.percentile(boot_time[finite], 97.5)) if np.any(finite) else None,
        "alpha": alpha,
        "alpha_ci_low": float(np.percentile(boot_alpha[finite], 2.5)) if np.any(finite) else None,
        "alpha_ci_high": float(np.percentile(boot_alpha[finite], 97.5)) if np.any(finite) else None,
        "bootstrap_resolved_fraction": resolved,
        "bracket": bracket,
        "spatial_separation": ny // 2,
        "temporal_separations": temporal_seps,
        "configuration_statistics": config_stats,
        "plateau_drift": plateau_drift,
        "wall_diagnostics": wall_diagnostics,
        "distribution_file": str(distribution_path.relative_to(root)),
        "distribution_sha256": sha256_file(distribution_path),
    }


def stage_final(root: Path, workers: int, cpu_ids: list[int]) -> dict[str, Any]:
    config = load_config()
    prepare_model_template()
    decision_path = root / "processed" / "audit_decision.json"
    if not decision_path.exists():
        raise ScienceUnresolved("The final campaign requires a completed cycle audit.")
    decision = json.loads(decision_path.read_text())
    if decision.get("status") != "passed":
        raise ScienceUnresolved(decision.get("reason", "Cycle audit did not pass."))
    final = config["final"]
    ny = int(config["geometry"]["Ny"])
    samples = int(final["samples"])
    eq_multiplier = int(decision["selected_equilibration_multiplier"])
    follow_cycles = int(decision["selected_follow_multiplier"]) * ny
    coarse = [0] + [int(value) for value in final["coarse_temporal_separations"]]
    ensure_base_checkpoints(root, "final", samples, [eq_multiplier], workers, cpu_ids)
    branch_tasks(root, "final", samples, [eq_multiplier], follow_cycles, coarse, workers, cpu_ids)
    coarse_rows = grid_rows(root, "final", eq_multiplier, follow_cycles, coarse, samples, follow_cycles // ny)
    coarse_summary = grid_summary(coarse_rows)
    if coarse_summary["bracket"] is None:
        result = {"status": "unresolved", "reason": "independent S=100 coarse grid has no ordered crossing through delta_tau=18", "grid": coarse_summary}
        write_json_atomic(root / "processed" / "final_results.json", result)
        return result
    missing = list(range(int(coarse_summary["bracket"]["lower"]) + 1, int(coarse_summary["bracket"]["upper"])))
    if missing:
        branch_tasks(root, "final", samples, [eq_multiplier], follow_cycles, missing, workers, cpu_ids)
    separations = sorted(set(coarse + missing))
    result = final_statistics(root, eq_multiplier, follow_cycles, separations)
    result.update({
        "equilibration_multiplier": eq_multiplier,
        "equilibration_cycles": eq_multiplier * ny,
        "follow_multiplier": follow_cycles // ny,
        "follow_cycles": follow_cycles,
        "maximum_temporal_separation": max(value for value in separations if value > 0),
        "maximum_total_cycles": eq_multiplier * ny + max(value for value in separations if value > 0) + follow_cycles,
        "dw_truncation": bool(config["geometry"]["dw_truncation"]),
        "meas_slab_only": bool(config["dynamics"]["meas_slab_only"]),
    })
    write_json_atomic(root / "processed" / "final_results.json", result)
    return result


def stage_analyze(root: Path) -> None:
    audit_path = root / "processed" / "audit_decision.json"
    final_path = root / "processed" / "final_results.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"status": "missing"}
    final = json.loads(final_path.read_text()) if final_path.exists() else {"status": "not_run"}
    lines = [
        "# L=24 soft-domain-wall reference-anisotropy campaign",
        "",
        f"- Audit status: `{audit.get('status')}`",
        f"- Final status: `{final.get('status')}`",
        "- Protocol invariants: `DW=True`, `alpha_top=1`, `alpha_triv=30`, `dw_truncation=False`, `meas_slab_only=False`.",
        f"- Canonical dynamics: `{CANONICAL_ENTRY}`.",
        "- Independent sampling unit: one complete Born-rule circuit trajectory.",
        "",
        "The matching time is defined by the arithmetic Born means",
        "",
        "$$\\overline I_{\\rm time}(0,t_*)=\\overline I_{\\rm space}(12,0),$$",
        "",
        "and the L=24 conversion is $\\alpha=6.7332/t_*$. The interpolation uses the first valid adjacent downward crossing.",
        "",
    ]
    if audit.get("status") == "passed":
        lines.extend([
            "## Cycle audit",
            "",
            f"- Accepted equilibration: `{audit['selected_equilibration_cycles']}` cycles (`{audit['selected_equilibration_multiplier']}L`).",
            f"- Accepted post-insertion follow: `{audit['selected_follow_cycles']}` cycles (`{audit['selected_follow_multiplier']}L`).",
            f"- Audit estimate: `t_*={audit['audit_time_star']:.6g}`, `alpha={audit['audit_alpha']:.6g}`.",
            "",
        ])
    else:
        lines.extend(["## Cycle audit", "", f"The convergence gate remains unresolved: {audit.get('reason', 'no reason recorded')}", ""])
    def display(value: Any) -> str:
        return "unresolved" if value is None else f"{float(value):.6g}"

    if final.get("time_star") is not None:
        lines.extend([
            "## Independent S=100 result",
            "",
            f"- $t_*={display(final['time_star'])}$, 95% bootstrap interval `[{display(final.get('time_star_ci_low'))}, {display(final.get('time_star_ci_high'))}]`.",
            f"- $\\alpha={display(final['alpha'])}$, 95% bootstrap interval `[{display(final.get('alpha_ci_low'))}, {display(final.get('alpha_ci_high'))}]`.",
            f"- Resolved bootstrap fraction: `{final['bootstrap_resolved_fraction']:.3f}`.",
            f"- Longest trajectory: `{final['maximum_total_cycles']}` cycles.",
            "",
        ])
    elif final.get("status") not in ("not_run",):
        lines.extend(["## Independent S=100 result", "", f"The matching-time estimate is unresolved: {final.get('reason')}", ""])
    lines.extend([
        "## Provenance",
        "",
        "All reference mutual informations and component entropies are retained for every sample and every relative follow cycle. Derived plateau means and bootstrap intervals do not replace those raw arrays.",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines))

    distribution_path = root / "processed" / "final_distributions.npz"
    if distribution_path.exists():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with np.load(distribution_path, allow_pickle=False) as data:
            spatial = np.asarray(data["spatial_by_sample"], dtype=float)
            separations = np.asarray(data["temporal_separations"], dtype=float)
            temporal = np.asarray(data["temporal_by_sample"], dtype=float)
            alpha_boot = np.asarray(data["bootstrap_alpha"], dtype=float)
        plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in"})
        figure, axes = plt.subplots(1, 2, figsize=(7.05, 2.7), constrained_layout=True)
        temporal_mean = np.mean(temporal, axis=0)
        temporal_sem = np.std(temporal, axis=0, ddof=1) / np.sqrt(temporal.shape[0])
        axes[0].errorbar(separations, temporal_mean, yerr=temporal_sem, color="#2166ac", marker="o", ls="-", capsize=2, label="temporal")
        axes[0].axhline(np.mean(spatial), color="black", ls="--", label="spatial $L/2$")
        if final.get("time_star") is not None:
            axes[0].axvline(final["time_star"], color="#b2182b", ls=":", label="$t_*$")
        axes[0].set(xlabel=r"insertion separation $\delta\tau$ (cycles)", ylabel=r"plateau $\overline I_{R_1,R_2}$")
        axes[0].legend(frameon=False)
        axes[0].text(-0.14, 1.03, "(a)", transform=axes[0].transAxes)
        finite_alpha = alpha_boot[np.isfinite(alpha_boot)]
        axes[1].hist(finite_alpha, bins=30, color="#4d9221", alpha=0.8)
        axes[1].set(xlabel=r"anisotropy $\alpha$", ylabel="bootstrap count")
        axes[1].text(-0.14, 1.03, "(b)", transform=axes[1].transAxes)
        for axis in axes:
            axis.tick_params(top=True, right=True)
        figure.savefig(root / "figures" / "l24_reference_anisotropy.pdf")
        figure.savefig(root / "figures" / "l24_reference_anisotropy.png", dpi=300)
        plt.close(figure)


def stage_validate(root: Path) -> None:
    config = load_config()
    if config["geometry"]["dw_truncation"] is not False or config["dynamics"]["meas_slab_only"] is not False:
        raise RuntimeError("Locked soft-wall flags are not false.")
    for index_path in (root / "raw" / "audit" / "raw_index.json", root / "raw" / "final" / "raw_index.json"):
        if not index_path.exists():
            raise RuntimeError(f"Missing raw index: {index_path}")
        for row in json.loads(index_path.read_text())["files"]:
            path = Path(row["path"])
            if not path.exists() or sha256_file(path) != row["sha256"]:
                raise RuntimeError(f"Missing or changed raw shard: {path}")
            with np.load(path, allow_pickle=False) as data:
                follow = int(data["follow_cycles"])
                if not np.array_equal(data["relative_cycles"], np.arange(follow + 1)):
                    raise RuntimeError(f"Incomplete cycle coordinate: {path}")
                for key in ("mutual_information", "entropy_r1", "entropy_r2", "entropy_r12"):
                    values = np.asarray(data[key])
                    if values.shape != (follow + 1,) or not np.all(np.isfinite(values)):
                        raise RuntimeError(f"Invalid cycle-resolved observable {key}: {path}")
    audit = json.loads((root / "processed" / "audit_decision.json").read_text())
    final = json.loads((root / "processed" / "final_results.json").read_text())
    if audit.get("status") != "passed":
        raise ScienceUnresolved(audit.get("reason", "Cycle audit is unresolved."))
    if final.get("status") != "calibrated":
        raise ScienceUnresolved(final.get("reason", "Final anisotropy is unresolved."))
    write_json_atomic(root / "processed" / "validation.json", {"status": "passed", "validated_utc": utc_now(), "raw_audit_sha256": sha256_file(root / "raw" / "audit" / "raw_index.json"), "raw_final_sha256": sha256_file(root / "raw" / "final" / "raw_index.json")})


def resolve_workers(root: Path, requested: str) -> int:
    if requested != "auto":
        workers = int(requested)
        if workers <= 0:
            raise ValueError("--workers must be positive or 'auto'.")
        return workers
    path = root / "processed" / "benchmark.json"
    if not path.exists():
        raise RuntimeError("--workers=auto requires a completed benchmark stage.")
    return int(json.loads(path.read_text())["recommended_workers"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "benchmark", "audit", "final", "analyze", "validate", "all"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cpu-list", default="")
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.run_root.resolve()
    try:
        initialize_run(root, resume=bool(args.resume))
        config = load_config()
        required = int(config["parallel"]["required_physical_cpus"])
        cpu_ids = parse_cpu_list(args.cpu_list)
        if args.stage in ("preflight", "benchmark", "audit", "final", "all"):
            cpu_ids = validate_cpu_ids(cpu_ids, required)
        if args.stage in ("preflight", "all"):
            run_stage(root, "preflight", lambda: stage_preflight(root, cpu_ids), resume=args.resume)
        if args.stage in ("benchmark", "all"):
            run_stage(root, "benchmark", lambda: stage_benchmark(root, cpu_ids), resume=args.resume)
        workers = resolve_workers(root, args.workers) if args.stage in ("audit", "final", "all") else 1
        if workers > len(cpu_ids) and args.stage in ("audit", "final", "all"):
            raise ValueError("Chosen worker count exceeds the validated CPU list.")
        if args.stage in ("audit", "final", "all"):
            record_execution(root, cpu_ids, workers)
        if args.stage in ("audit", "all"):
            run_stage(root, "audit", lambda: stage_audit(root, workers, cpu_ids), resume=args.resume)
            audit = json.loads((root / "processed" / "audit_decision.json").read_text())
            if audit.get("status") != "passed":
                raise ScienceUnresolved(audit.get("reason", "Cycle audit is unresolved."))
        if args.stage in ("final", "all"):
            run_stage(root, "final", lambda: stage_final(root, workers, cpu_ids), resume=args.resume)
        if args.stage in ("analyze", "all"):
            run_stage(root, "analyze", lambda: stage_analyze(root), resume=args.resume)
        if args.stage in ("validate", "all"):
            run_stage(root, "validate", lambda: stage_validate(root), resume=args.resume)
        if args.stage == "all":
            (root / "SUCCESS").write_text(utc_now() + "\n")
        print(f"[done] {args.stage}: {root}", flush=True)
        return 0
    except ScienceUnresolved as exc:
        write_json_atomic(root / "NEEDS_ATTENTION", {"recorded_utc": utc_now(), "reason": str(exc)})
        print(f"[needs-attention] {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        write_json_atomic(root / "FAILED", {"recorded_utc": utc_now(), "error": str(exc), "traceback": traceback.format_exc()})
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
