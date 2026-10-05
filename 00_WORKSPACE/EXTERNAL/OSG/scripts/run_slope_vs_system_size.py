"""
OSG job: one sample block for the slope-vs-system-size campaign.

Reads params.json[job_id] to get {Ny, block_start, block_stop, cycles, fit_cycle}.
Runs that block, saves a full accumulator checkpoint.
The local merge_slope_vs_system_size.py script combines all blocks per Ny.

Usage:
    python run_slope_vs_system_size.py --job-id 0 --output-dir ./output
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from classA_U1FGTN_gpu import classA_U1FGTN_gpu
from strip_entropy_streaming_gpu import (
    StripContourAccumulator,
    write_json_atomic,
)

# ── Fixed campaign settings ───────────────────────────────────────────────────
NX              = 20
NSHELL          = 1
ALPHA_1         = 1.0
ALPHA_2         = 30.0
DW_TRUNCATION   = True
INIT_MODE       = "default"
DTYPE           = "complex128"
BACKEND         = "local"
SEQUENCE        = "raster_y"
POSTSELECT      = False
PERFECT_CORRECTION = True
N_A             = 0.5
FIT_AY_MIN      = 8

AUTOTUNE        = True
AUTOTUNE_REPEAT = 2
MEMORY_SAFETY   = 0.85
EIGH_MULTIPLIER = 8.0
SAMPLE_CHUNKS   = [1, 2, 4, 8, 16]
Y0_CHUNKS       = [1, 2, 4, 6, 8, 12, 16, 24, 32, "ny"]

CANONICAL_ENTRY = "classA_U1FGTN_gpu.run_markov_circuit"


def run_block(Ny: int, block_start: int, block_stop: int,
              cycles: int, fit_cycle: int, output_dir: Path) -> dict:
    Ny          = int(Ny)
    block_count = block_stop - block_start
    run_id      = f"N{NX}x{Ny}_nsh{NSHELL}_dwtrunc{int(DW_TRUNCATION)}_C{cycles}_S100"
    block_dir   = output_dir / "blocks" / f"N{NX}x{Ny}"
    block_dir.mkdir(parents=True, exist_ok=True)
    block_path  = block_dir / f"block_{block_start:03d}_{block_stop:03d}.npz"
    status_path = block_dir / f"block_{block_start:03d}_{block_stop:03d}_status.json"

    if block_path.exists():
        print(f"[skip] block already exists: {block_path}", flush=True)
        return {"status": "existing", "checkpoint": str(block_path)}

    config = {
        "Nx": NX, "Ny": Ny, "nshell": NSHELL, "cycles": cycles,
        "fit_cycle": fit_cycle, "fit_Ay_min": FIT_AY_MIN,
        "alpha_1": ALPHA_1, "alpha_2": ALPHA_2,
        "dw_truncation": DW_TRUNCATION, "init_mode": INIT_MODE,
        "dtype": DTYPE, "sequence": SEQUENCE,
        "perfect_correction": PERFECT_CORRECTION, "postselect": POSTSELECT,
        "canonical_dynamics_entry_point": CANONICAL_ENTRY,
        "block_start": block_start, "block_stop": block_stop,
        "block_count": block_count, "run_id": run_id,
        "Ay_values": list(range(Ny // 2 + 1)),
    }

    accumulator = StripContourAccumulator(
        nx=NX,
        ny=Ny,
        ay_values=range(Ny // 2 + 1),
        cycles=[fit_cycle],
        config=config,
    )

    def cycle_observer(*, cycle, G, batch_index, batch_start, batch_count):
        if int(cycle) != fit_cycle:
            return
        print(
            f"[observer] Ny={Ny} block={block_start}:{block_stop} "
            f"cycle={cycle} batch={batch_index} "
            f"samples {batch_start}:{batch_start + batch_count}",
            flush=True,
        )
        accumulator.update(
            cycle=cycle, G=G, eps=1e-12,
            autotune=AUTOTUNE,
            sample_chunk_candidates=SAMPLE_CHUNKS,
            y0_chunk_candidates=Y0_CHUNKS,
            autotune_repeat=AUTOTUNE_REPEAT,
            memory_safety_fraction=MEMORY_SAFETY,
            eigh_memory_multiplier=EIGH_MULTIPLIER,
        )

    print(f"[run] Ny={Ny} block={block_start}:{block_stop} cycles={cycles}", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] {torch.cuda.get_device_name(0)}  "
          f"free/total: {free/1024**3:.1f}/{total/1024**3:.1f} GiB", flush=True)

    model = classA_U1FGTN_gpu(
        Nx=NX, Ny=Ny, DW=True, nshell=NSHELL, filling_frac=0.5,
        alpha_1=ALPHA_1, alpha_2=ALPHA_2, trial_orbitals="X",
        dw_truncation=DW_TRUNCATION, triv_region_local_mode=False,
        device="cuda:0", dtype=DTYPE, backend=BACKEND,
    )
    t0 = time.time()
    result = model.run_markov_circuit(
        G_history=False, progress=True,
        cycles=cycles, postselect=POSTSELECT,
        perfect_correction=PERFECT_CORRECTION,
        samples=block_count, init_mode=INIT_MODE,
        save=False, n_a=N_A, sequence=SEQUENCE,
        batch_size=block_count,
        return_data=False, cycle_observer=cycle_observer,
    )
    elapsed = time.time() - t0
    print(f"[done] Ny={Ny} block={block_start}:{block_stop} "
          f"in {elapsed/3600:.2f}h", flush=True)

    accumulator.save_checkpoint(block_path, include_contours=True)
    print(f"[saved] {block_path}", flush=True)

    record = {
        "status": "created", "Ny": Ny,
        "block_start": block_start, "block_stop": block_stop,
        "cycles": cycles, "fit_cycle": fit_cycle,
        "checkpoint": str(block_path),
        "elapsed_hours": elapsed / 3600,
        "markov_result": result,
        "created_unix": time.time(),
    }
    write_json_atomic(status_path, record)

    del model, accumulator
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    args = parser.parse_args()

    with open("params.json") as f:
        params = json.load(f)

    if args.job_id < 0 or args.job_id >= len(params):
        raise ValueError(f"--job-id must be 0..{len(params)-1}, got {args.job_id}")

    p = params[args.job_id]
    print(f"[start] job_id={args.job_id} params={p}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    torch.set_grad_enabled(False)

    run_block(
        Ny=p["Ny"], block_start=p["block_start"], block_stop=p["block_stop"],
        cycles=p["cycles"], fit_cycle=p["fit_cycle"],
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
