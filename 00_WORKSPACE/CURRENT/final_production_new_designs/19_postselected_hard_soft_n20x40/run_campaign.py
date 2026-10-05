#!/usr/bin/env python3
"""Sequential A100 hard/soft postselected max-mix campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu  # noqa: E402
from postselected_observer import PostselectedObserver  # noqa: E402


CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"
RESULT_SCHEMA = "postselected_maxmix_hard_soft_result_v2"
COMPLETION_SCHEMA = "postselected_maxmix_hard_soft_completion_v1"
CHECKPOINT_SCHEMA = "postselected_maxmix_hard_soft_checkpoint_v2"
SOURCE_FILES = (
    "campaign_config.json",
    "run_campaign.py",
    "postselected_observer.py",
    "src/classA_U1FGTN_gpu.py",
    "src/occupied_frame_gpu.py",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    return {relative: sha256_file(BUNDLE_ROOT / relative) for relative in SOURCE_FILES}


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode("utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def publish_file(local_path: Path, final_path: Path) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_name(f".{final_path.name}.uploading")
    temporary.unlink(missing_ok=True)
    shutil.copy2(local_path, temporary)
    expected_bytes = local_path.stat().st_size
    expected_hash = sha256_file(local_path)
    if temporary.stat().st_size != expected_bytes or sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise OSError(f"DriveFS readback failed for {final_path}")
    os.replace(temporary, final_path)
    if final_path.stat().st_size != expected_bytes or sha256_file(final_path) != expected_hash:
        raise OSError(f"DriveFS stable-file readback failed for {final_path}")
    return {"bytes": expected_bytes, "sha256": expected_hash}


def result_paths(output_root: Path, construction: str) -> tuple[Path, Path]:
    directory = output_root / "results" / construction
    return directory / "postselected_trajectory.npz", directory / "completion.json"


def checkpoint_paths(output_root: Path, construction: str) -> tuple[Path, Path]:
    directory = output_root / "checkpoints" / construction
    return directory / "checkpoint.npz", directory / "checkpoint.json"


def verified_complete(
    output_root: Path,
    construction: str,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
) -> tuple[bool, str]:
    result_path, completion_path = result_paths(output_root, construction)
    if not result_path.is_file() or not completion_path.is_file():
        return False, "missing result/completion pair"
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "completion_schema": COMPLETION_SCHEMA,
            "construction": construction,
            "configuration_hash": cfg_hash,
            "source_hashes": hashes,
            "result_filename": result_path.name,
            "result_bytes": result_path.stat().st_size,
            "result_sha256": sha256_file(result_path),
        }
        for key, value in expected.items():
            if completion.get(key) != value:
                return False, f"completion mismatch for {key}"
        with np.load(result_path, allow_pickle=False) as payload:
            if payload["result_schema"].item() != RESULT_SCHEMA:
                return False, "result schema mismatch"
            if payload["construction"].item() != construction:
                return False, "result construction mismatch"
        return True, "verified"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return False, f"verification error: {exc}"


def capture_rng() -> dict[str, np.ndarray]:
    np_state = np.random.get_state()
    payload = {
        "numpy_algorithm": np.asarray(np_state[0]),
        "numpy_keys": np.asarray(np_state[1], dtype=np.uint32),
        "numpy_position": np.asarray(np_state[2], dtype=np.int64),
        "numpy_has_gauss": np.asarray(np_state[3], dtype=np.int64),
        "numpy_cached_gaussian": np.asarray(np_state[4], dtype=np.float64),
        "torch_cpu_rng": torch.get_rng_state().cpu().numpy(),
    }
    for index, state in enumerate(torch.cuda.get_rng_state_all()):
        payload[f"torch_cuda_rng_{index}"] = state.cpu().numpy()
    payload["torch_cuda_rng_count"] = np.asarray(
        len(torch.cuda.get_rng_state_all()), dtype=np.int64
    )
    return payload


def restore_rng(payload: dict[str, np.ndarray]) -> None:
    np.random.set_state(
        (
            str(np.asarray(payload["numpy_algorithm"]).item()),
            np.asarray(payload["numpy_keys"], dtype=np.uint32),
            int(np.asarray(payload["numpy_position"]).item()),
            int(np.asarray(payload["numpy_has_gauss"]).item()),
            float(np.asarray(payload["numpy_cached_gaussian"]).item()),
        )
    )
    torch.set_rng_state(torch.as_tensor(payload["torch_cpu_rng"], dtype=torch.uint8))
    count = int(np.asarray(payload["torch_cuda_rng_count"]).item())
    torch.cuda.set_rng_state_all(
        [torch.as_tensor(payload[f"torch_cuda_rng_{index}"], dtype=torch.uint8) for index in range(count)]
    )


def save_checkpoint(
    output_root: Path,
    scratch_root: Path,
    construction: str,
    *,
    completed_cycle: int,
    elapsed_seconds: float,
    G: np.ndarray,
    observer: PostselectedObserver,
    cfg_hash: str,
    hashes: dict[str, str],
) -> dict[str, np.ndarray]:
    rng = capture_rng()
    payload = {
        "checkpoint_schema": np.asarray(CHECKPOINT_SCHEMA),
        "construction": np.asarray(construction),
        "configuration_hash": np.asarray(cfg_hash),
        "completed_cycle": np.asarray(completed_cycle, dtype=np.int64),
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
        "G": np.asarray(G, dtype=np.complex128),
        **{f"observer__{key}": value for key, value in observer.checkpoint_payload(completed_cycle).items()},
        **{f"rng__{key}": value for key, value in rng.items()},
    }
    local_dir = scratch_root / construction
    local_npz = local_dir / "checkpoint.npz"
    local_json = local_dir / "checkpoint.json"
    save_npz(local_npz, payload)
    final_npz, final_json = checkpoint_paths(output_root, construction)
    commit = publish_file(local_npz, final_npz)
    metadata = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "construction": construction,
        "configuration_hash": cfg_hash,
        "source_hashes": hashes,
        "completed_cycle": completed_cycle,
        "checkpoint_filename": final_npz.name,
        "checkpoint_bytes": commit["bytes"],
        "checkpoint_sha256": commit["sha256"],
    }
    write_json(local_json, metadata)
    publish_file(local_json, final_json)
    return rng


def load_checkpoint(
    output_root: Path,
    construction: str,
    *,
    cycles: int,
    observer: PostselectedObserver,
    cfg_hash: str,
    hashes: dict[str, str],
) -> tuple[int, float, np.ndarray | None, dict[str, np.ndarray] | None, str]:
    npz_path, json_path = checkpoint_paths(output_root, construction)
    if not npz_path.is_file() or not json_path.is_file():
        return 0, 0.0, None, None, "no checkpoint"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    expected = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "construction": construction,
        "configuration_hash": cfg_hash,
        "source_hashes": hashes,
        "checkpoint_filename": npz_path.name,
        "checkpoint_bytes": npz_path.stat().st_size,
        "checkpoint_sha256": sha256_file(npz_path),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"{construction} checkpoint mismatch for {key}")
    with np.load(npz_path, allow_pickle=False) as payload:
        completed = int(payload["completed_cycle"].item())
        if not 0 < completed <= cycles:
            raise RuntimeError(f"invalid checkpoint cycle {completed}")
        observer_payload = {
            key.removeprefix("observer__"): np.array(payload[key], copy=True)
            for key in payload.files
            if key.startswith("observer__")
        }
        observer.restore(observer_payload, completed_cycle=completed)
        rng = {
            key.removeprefix("rng__"): np.array(payload[key], copy=True)
            for key in payload.files
            if key.startswith("rng__")
        }
        return (
            completed,
            float(payload["elapsed_seconds"].item()),
            np.array(payload["G"], dtype=np.complex128, copy=True),
            rng,
            f"verified cycle {completed}",
        )


def build_model(config: dict[str, Any], construction: str) -> classA_U1FGTN_gpu:
    flags = config["constructions"][construction]
    return classA_U1FGTN_gpu(
        Nx=int(config["Nx"]),
        Ny=int(config["Ny"]),
        DW=True,
        nshell=int(config["nshell"]),
        filling_frac=0.5,
        alpha_1=float(config["alpha_1"]),
        alpha_2=float(config["alpha_2"]),
        trial_orbitals=config["trial_orbitals"],
        dw_truncation=bool(flags["dw_truncation"]),
        triv_region_local_mode=False,
        device=config["device"],
        dtype=config["dtype"],
        backend=config["backend"],
    )


def run_segment(
    model: classA_U1FGTN_gpu,
    config: dict[str, Any],
    construction: str,
    observer: PostselectedObserver,
    *,
    completed_cycle: int,
    segment_cycles: int,
    G_init: np.ndarray | None,
    progress_bar: tqdm,
) -> np.ndarray:
    hard = construction == "hard"
    continuing = completed_cycle > 0

    def cycle_observer(*, cycle: int, G: torch.Tensor, **_: Any) -> None:
        local_cycle = int(cycle)
        if continuing and local_cycle == 0:
            return
        global_cycle = completed_cycle + local_cycle
        observer.observe(cycle=global_cycle, G=G)
        if local_cycle > 0:
            progress_bar.update(1)
            progress_bar.set_postfix(
                entropy=f"{observer.total_entropy_nats[global_cycle]:.4g}",
                gap=f"{observer.lyapunov_gap[global_cycle]:.3g}",
            )

    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=int(segment_cycles),
        postselect=True,
        postselect_probability=1.0,
        perfect_correction=False,
        samples=1,
        init_mode="maxmix",
        G_init=G_init,
        G_init_prepared=bool(continuing and hard),
        save=False,
        save_init=False,
        n_a=0.5,
        sequence=config["sequence"],
        meas_slab_only=hard,
        batch_size=1,
        return_data=True,
        state_representation="covariance",
        initial_purity_tolerance=0.50000001,
        cycle_observer=cycle_observer,
    )
    if result.get("state_representation_resolved") != "covariance":
        raise RuntimeError("canonical engine did not use covariance representation")
    if bool(result.get("meas_slab_only_effective")) is not hard:
        raise RuntimeError("canonical engine slab-only resolution mismatch")
    if (
        float(result.get("postselect_probability", -1.0)) != 1.0
        or result.get("batch_size_mode") != "postselect"
    ):
        raise RuntimeError("canonical engine did not resolve full postselection")
    final = np.asarray(result["G_final"], dtype=np.complex128)
    if final.shape != (1, 2 * config["Nx"] * config["Ny"], 2 * config["Nx"] * config["Ny"]):
        raise RuntimeError(f"unexpected final covariance shape {final.shape}")
    return final


def execute(
    config: dict[str, Any],
    output_root: Path,
    scratch_root: Path,
    construction: str,
    *,
    cfg_hash: str,
    hashes: dict[str, str],
) -> None:
    hard = construction == "hard"
    seed = int(config["root_seed"]) + (0 if hard else 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model(config, construction)
    active_indices = model.active_top_layer_indices(meas_slab_only=hard)
    expected = int(config["constructions"][construction]["expected_active_modes"])
    if int(active_indices.numel()) != expected:
        raise RuntimeError(f"{construction}: expected {expected} active modes, got {active_indices.numel()}")
    observer = PostselectedObserver(cycles=int(config["cycles"]), active_indices=active_indices,
                                   nx=int(config["Nx"]), ny=int(config["Ny"]))
    completed, elapsed, G, rng, reason = load_checkpoint(
        output_root,
        construction,
        cycles=int(config["cycles"]),
        observer=observer,
        cfg_hash=cfg_hash,
        hashes=hashes,
    )
    print(f"[{construction}] {reason}; active_modes={expected}", flush=True)
    with tqdm(
        total=int(config["cycles"]),
        initial=completed,
        desc=f"{construction} postselect",
        unit="cycle",
        leave=True,
    ) as cycle_bar:
        while completed < int(config["cycles"]):
            segment = min(
                int(config["checkpoint_stride_cycles"]),
                int(config["cycles"]) - completed,
            )
            if rng is not None:
                restore_rng(rng)
            started = time.perf_counter()
            G = run_segment(
                model,
                config,
                construction,
                observer,
                completed_cycle=completed,
                segment_cycles=segment,
                G_init=G,
                progress_bar=cycle_bar,
            )
            torch.cuda.synchronize()
            elapsed += time.perf_counter() - started
            completed += segment
            rng = save_checkpoint(
                output_root,
                scratch_root,
                construction,
                completed_cycle=completed,
                elapsed_seconds=elapsed,
                G=G,
                observer=observer,
                cfg_hash=cfg_hash,
                hashes=hashes,
            )
            cycle_bar.set_postfix(durable_cycle=completed)

    if G is None:
        raise RuntimeError("missing final covariance")
    observer.validate()
    G_tensor = torch.as_tensor(G, device=config["device"], dtype=torch.complex128)
    idx = active_indices.to(device=G_tensor.device)
    active = G_tensor.index_select(-2, idx).index_select(-1, idx)[0]
    with torch.inference_mode():
        centered, vectors = torch.linalg.eigh(0.5 * (active + active.mH))
    centered_np = np.clip(centered.detach().cpu().numpy(), -1.0, 1.0)
    vectors_np = vectors.detach().cpu().numpy().astype(np.complex128, copy=False)
    active_np = active.detach().cpu().numpy().astype(np.complex128, copy=False)
    local_dir = scratch_root / construction / "result"
    local_result = local_dir / "postselected_trajectory.npz"
    local_completion = local_dir / "completion.json"
    payload = {
        "result_schema": np.asarray(RESULT_SCHEMA),
        "sampling_revision": np.asarray(config["sampling_revision"]),
        "configuration_hash": np.asarray(cfg_hash),
        "canonical_dynamics_entry_point": np.asarray(CANONICAL_ENTRY_POINT),
        "construction": np.asarray(construction),
        "Nx": np.asarray(config["Nx"], dtype=np.int64),
        "Ny": np.asarray(config["Ny"], dtype=np.int64),
        "alpha_1": np.asarray(config["alpha_1"], dtype=np.float64),
        "alpha_2": np.asarray(config["alpha_2"], dtype=np.float64),
        "seed": np.asarray(seed, dtype=np.uint64),
        "elapsed_seconds": np.asarray(elapsed, dtype=np.float64),
        "active_basis_indices": active_indices.detach().cpu().numpy(),
        "endpoint_centered_covariance": active_np,
        "endpoint_centered_spectrum": centered_np,
        "endpoint_occupations": 0.5 * (1.0 + centered_np),
        "endpoint_eigenvectors": vectors_np,
        **observer.result_payload(),
    }
    save_npz(local_result, payload)
    result_path, completion_path = result_paths(output_root, construction)
    commit = publish_file(local_result, result_path)
    completion = {
        "completion_schema": COMPLETION_SCHEMA,
        "construction": construction,
        "configuration_hash": cfg_hash,
        "source_hashes": hashes,
        "result_filename": result_path.name,
        "result_bytes": commit["bytes"],
        "result_sha256": commit["sha256"],
        "completed_cycles": int(config["cycles"]),
        "elapsed_seconds": elapsed,
    }
    write_json(local_completion, completion)
    publish_file(local_completion, completion_path)
    valid, why = verified_complete(
        output_root, construction, cfg_hash=cfg_hash, hashes=hashes
    )
    if not valid:
        raise RuntimeError(f"{construction} result failed verification: {why}")
    for path in checkpoint_paths(output_root, construction):
        path.unlink(missing_ok=True)
    print(
        f"[{construction}] complete in {elapsed / 60:.2f} min; "
        f"endpoint gap={observer.lyapunov_gap[-1]:.9g}",
        flush=True,
    )
    del model, observer, G_tensor, active, centered, vectors
    gc.collect()
    torch.cuda.empty_cache()


def require_runtime(config: dict[str, Any]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(config["device"])
    name = torch.cuda.get_device_name(device)
    gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if "A100" not in name or gib < float(config["gpu_memory_minimum_gib"]):
        raise RuntimeError(f"expected A100 40-GB-class GPU, got {name} ({gib:.2f} GiB)")
    if torch.complex128 != getattr(torch, config["dtype"]):
        raise RuntimeError("locked dtype is not complex128")
    print(json.dumps({"gpu": name, "gpu_GiB": gib}, indent=2), flush=True)


def campaign_tasks(config: dict[str, Any]):
    """Separate durable directories and configuration identities for each alpha."""
    values = [float(value) for value in config["alpha_1_values"]]
    if not values or len(set(values)) != len(values) or not all(np.isfinite(values)):
        raise ValueError("alpha_1_values must be nonempty, finite, and unique")
    for alpha in values:
        resolved = {**config, "alpha_1": alpha}
        for construction in config["construction_order"]:
            if construction not in ("hard", "soft"):
                raise ValueError(f"unknown construction: {construction}")
            yield f"alpha1_{alpha:g}", construction, resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BUNDLE_ROOT / "campaign_config.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-new-constructions", type=int)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hashes = source_hashes()
    output_root = args.output_root.resolve()
    scratch_root = args.scratch_root.resolve()
    if output_root.name != config["sampling_revision"] or not config["sampling_revision"].endswith("_v2"):
        raise ValueError("Use the new v2 sampling-revision output folder; v1 results must remain untouched")
    output_root.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    inventory = {}
    tasks = list(campaign_tasks(config))
    for alpha_dir, construction, resolved in tasks:
        complete, reason = verified_complete(
            output_root / alpha_dir, construction, cfg_hash=config_hash(resolved), hashes=hashes
        )
        checkpoint_npz, _ = checkpoint_paths(output_root / alpha_dir, construction)
        inventory[f"{alpha_dir}/{construction}"] = {
            "complete": complete,
            "reason": reason,
            "checkpoint_present": checkpoint_npz.is_file(),
        }
    print(
        json.dumps(
            {
                "config": config,
                "configuration_hash": config_hash(config),
                "source_hashes": hashes,
                "output_root": str(output_root),
                "scratch_root": str(scratch_root),
                "resume_inventory": inventory,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.report_only:
        return 0
    require_runtime(config)
    launched = 0
    with tqdm(total=len(tasks), desc="alpha / wall runs", unit="run") as outer:
        for alpha_dir, construction, resolved in tasks:
            task_output = output_root / alpha_dir
            cfg_hash = config_hash(resolved)
            valid, reason = verified_complete(
                task_output, construction, cfg_hash=cfg_hash, hashes=hashes
            )
            if valid:
                print(f"[{construction}] skipped: {reason}", flush=True)
                outer.update(1)
                continue
            if args.max_new_constructions is not None and launched >= args.max_new_constructions:
                print(f"[{construction}] pending due to MAX_NEW_CONSTRUCTIONS", flush=True)
                continue
            print(f"[start] {alpha_dir}/{construction}; output={task_output}", flush=True)
            execute(
                resolved,
                task_output,
                scratch_root / alpha_dir,
                construction,
                cfg_hash=cfg_hash,
                hashes=hashes,
            )
            launched += 1
            outer.update(1)
    print(f"[campaign] invocation finished; {launched} new runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
