"""
OSG job: one sample block for the slope-vs-cycle campaign.

Runs SAMPLE_BLOCK_SIZE samples (default 10) for ALL cycles 0..100,
saves a full accumulator checkpoint. Run 10 of these in parallel
(job IDs 0-9), then run merge_slope_vs_cycle.py locally to combine.

Usage:
    python run_slope_vs_cycle_block.py --job-id 0 --output-dir ./output
    # job-id 0  -> samples 0:10
    # job-id 1  -> samples 10:20
    # ...
    # job-id 9  -> samples 90:100
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu
from strip_entropy_streaming_gpu import (
    HELPER_VERSION,
    StripContourAccumulator,
    rel_to_root,
    write_json_atomic,
)

# ── Campaign config (mirrors run_slope_vs_cycle_resumable.ipynb exactly) ─────
NX = 20
NY = 40
NSHELL = 1
SAMPLES = 100
CYCLES = 100
FIT_AY_MIN = 8
ALPHA_1 = 1.0
ALPHA_2 = 30.0
DW_TRUNCATION = True
INIT_MODE = "default"
DTYPE = "complex128"
BACKEND = "local"
SEQUENCE = "raster_y"
POSTSELECT = False
PERFECT_CORRECTION = True
N_A = 0.5
SAMPLE_BLOCK_SIZE = 10
AUTOTUNE_FOR_A100_40GB = True
AUTOTUNE_REPEAT = 2
MEMORY_SAFETY_FRACTION = 0.85
EIGH_MEMORY_MULTIPLIER = 8.0
SAMPLE_CHUNK_CANDIDATES = [1, 2, 4, 8, 16]
Y0_CHUNK_CANDIDATES = [1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"]
TARGET_SLOPE_FULL_X = 1.0 / 3.0
CANONICAL_ENTRY_POINT = "classA_U1FGTN_gpu.run_markov_circuit"

RUN_ID = f"N{NX}x{NY}_nsh{NSHELL}_dwtrunc{int(DW_TRUNCATION)}_C{CYCLES}_S{SAMPLES}"

CAMPAIGN_CONFIG = {
    "Nx": NX,
    "Ny": NY,
    "nshell": NSHELL,
    "samples": SAMPLES,
    "cycles": CYCLES,
    "fit_Ay_min": FIT_AY_MIN,
    "alpha_1": ALPHA_1,
    "alpha_2": ALPHA_2,
    "dw_truncation": DW_TRUNCATION,
    "init_mode": INIT_MODE,
    "dtype": DTYPE,
    "sequence": SEQUENCE,
    "perfect_correction": PERFECT_CORRECTION,
    "postselect": POSTSELECT,
    "canonical_dynamics_entry_point": CANONICAL_ENTRY_POINT,
    "gpu_target": "A100 40GB",
    "sample_block_size": SAMPLE_BLOCK_SIZE,
    "checkpoint_format": "full_contour",
}


def validate_block_checkpoint(path: Path, *, block_count: int) -> None:
    acc = StripContourAccumulator.load_checkpoint(path, require_contours=True)
    if acc.nx != NX or acc.ny != NY:
        raise RuntimeError(f"Checkpoint {path} has wrong nx/ny.")
    if acc.ay_values != list(range(NY // 2 + 1)):
        raise RuntimeError(f"Checkpoint {path} has wrong ay_values.")
    if acc.cycles != list(range(CYCLES + 1)):
        raise RuntimeError(f"Checkpoint {path} has wrong cycles.")
    expected = int(block_count) * NY
    for ay in acc.ay_values:
        count = acc.count[ay]
        if not np.all(count == expected):
            bad = [int(acc.cycles[i]) for i, v in enumerate(count) if int(v) != expected]
            raise RuntimeError(
                f"Block checkpoint {path} incomplete for Ay={ay}. "
                f"Expected count {expected}. Bad cycles: {bad[:10]}"
            )


def run_block(block_start: int, output_dir: Path) -> dict:
    block_stop = min(SAMPLES, block_start + SAMPLE_BLOCK_SIZE)
    block_count = block_stop - block_start

    block_root = output_dir / "runs" / RUN_ID / "sample_block_checkpoints"
    block_root.mkdir(parents=True, exist_ok=True)
    block_path = block_root / f"block_{block_start:03d}_{block_stop:03d}.npz"
    status_path = block_root / f"block_{block_start:03d}_{block_stop:03d}_status.json"

    if block_path.exists():
        try:
            validate_block_checkpoint(block_path, block_count=block_count)
            print(f"[skip] block {block_start}:{block_stop} already complete", flush=True)
            return {"block_start": block_start, "block_stop": block_stop,
                    "checkpoint": str(block_path), "status": "existing"}
        except Exception as e:
            print(f"[warn] existing checkpoint invalid ({e}), re-running block", flush=True)

    block_config = {
        **CAMPAIGN_CONFIG,
        "run_id": RUN_ID,
        "Ay_values": list(range(NY // 2 + 1)),
        "block_start": int(block_start),
        "block_stop": int(block_stop),
        "block_count": int(block_count),
    }
    block_accumulator = StripContourAccumulator(
        nx=NX,
        ny=NY,
        ay_values=range(NY // 2 + 1),
        cycles=range(CYCLES + 1),
        config=block_config,
    )

    def cycle_observer(*, cycle, G, batch_index, batch_start, batch_count):
        print(
            f"[observer] block={block_start}:{block_stop} cycle={cycle} "
            f"batch={batch_index} samples {batch_start}:{batch_start + batch_count}",
            flush=True,
        )
        block_accumulator.update(
            cycle=cycle,
            G=G,
            eps=1e-12,
            autotune=AUTOTUNE_FOR_A100_40GB,
            sample_chunk_candidates=SAMPLE_CHUNK_CANDIDATES,
            y0_chunk_candidates=Y0_CHUNK_CANDIDATES,
            autotune_repeat=AUTOTUNE_REPEAT,
            memory_safety_fraction=MEMORY_SAFETY_FRACTION,
            eigh_memory_multiplier=EIGH_MEMORY_MULTIPLIER,
        )

    print(f"[run] block {block_start}:{block_stop} ({block_count} samples, {CYCLES} cycles)", flush=True)
    print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free/total GiB: {free/1024**3:.2f}/{total/1024**3:.2f}", flush=True)

    model = classA_U1FGTN_gpu(
        Nx=NX,
        Ny=NY,
        DW=True,
        nshell=NSHELL,
        filling_frac=0.5,
        alpha_1=ALPHA_1,
        alpha_2=ALPHA_2,
        trial_orbitals="X",
        dw_truncation=DW_TRUNCATION,
        triv_region_local_mode=False,
        device="cuda:0",
        dtype=DTYPE,
        backend=BACKEND,
    )
    t0 = time.time()
    result = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=CYCLES,
        postselect=POSTSELECT,
        perfect_correction=PERFECT_CORRECTION,
        samples=block_count,
        init_mode=INIT_MODE,
        save=False,
        n_a=N_A,
        sequence=SEQUENCE,
        batch_size=block_count,  # run all block samples in one batch
        return_data=False,
        cycle_observer=cycle_observer,
    )
    elapsed = time.time() - t0
    print(f"[done] block {block_start}:{block_stop} in {elapsed/3600:.2f}h", flush=True)

    block_accumulator.save_checkpoint(block_path, include_contours=True)
    validate_block_checkpoint(block_path, block_count=block_count)
    print(f"[saved] {block_path}", flush=True)

    record = {
        "block_start": int(block_start),
        "block_stop": int(block_stop),
        "checkpoint": str(block_path),
        "status": "created",
        "elapsed_hours": elapsed / 3600,
        "markov_result": result,
        "created_unix": time.time(),
    }
    write_json_atomic(status_path, record)

    del model, block_accumulator
    gc.collect()
    torch.cuda.empty_cache()

    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True,
                        help="Job index 0-9. Block start = job_id * SAMPLE_BLOCK_SIZE.")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"),
                        help="Root output directory (same for all blocks; must be on shared storage)")
    args = parser.parse_args()

    n_blocks = SAMPLES // SAMPLE_BLOCK_SIZE
    if args.job_id < 0 or args.job_id >= n_blocks:
        raise ValueError(f"--job-id must be 0..{n_blocks - 1}, got {args.job_id}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Is request_gpus = 1 in the submit file?")
    torch.set_grad_enabled(False)

    block_start = args.job_id * SAMPLE_BLOCK_SIZE
    print(f"[start] job_id={args.job_id} block={block_start}:{block_start+SAMPLE_BLOCK_SIZE} output={args.output_dir}", flush=True)
    run_block(block_start, args.output_dir)


if __name__ == "__main__":
    main()
