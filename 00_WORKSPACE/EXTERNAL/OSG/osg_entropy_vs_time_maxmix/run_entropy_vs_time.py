"""
OSG job: total entropy vs cycle for maxmix init, one sample block.

At every cycle t=1..Ny, computes S(t) = -Tr[G log G + (I-G) log(I-G)]
from eigenvalues of the full covariance matrix G. No subsystem cut.

Reads params.json[job_id] for {Ny, block_start, block_stop, cycles}.
Saves output/blocks/N20x{Ny}/block_{start}_{stop}.npz with sum/sumsq/count
arrays indexed by cycle — additively mergeable across blocks.

Usage:
    python run_entropy_vs_time.py --job-id 0 --output-dir ./output
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

NX               = 20
NSHELL           = 1
ALPHA_1          = 1.0
ALPHA_2          = 30.0
DW_TRUNCATION    = True
INIT_MODE        = "maxmix"
DTYPE            = "complex128"
BACKEND          = "local"
SEQUENCE         = "raster_y"
POSTSELECT       = False
PERFECT_CORRECTION = True
N_A              = 0.5


def total_entropy_batch(G: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    """S = -Tr[G log G + (I-G) log(I-G)] for each sample in the batch.

    G: (B, N, N) complex Hermitian, eigenvalues in [0, 1]
    returns: (B,) float64 numpy array
    """
    nu = torch.linalg.eigvalsh(G)          # (B, N) real
    nu = nu.clamp(eps, 1.0 - eps)
    S  = -(nu * torch.log(nu) + (1.0 - nu) * torch.log(1.0 - nu))
    return S.sum(dim=-1).detach().cpu().to(torch.float64).numpy()


def run_block(Ny: int, block_start: int, block_stop: int,
              cycles: int, output_dir: Path) -> None:
    Ny          = int(Ny)
    block_count = block_stop - block_start
    block_dir   = output_dir / "blocks" / f"N{NX}x{Ny}"
    block_dir.mkdir(parents=True, exist_ok=True)
    block_path  = block_dir / f"block_{block_start:03d}_{block_stop:03d}.npz"

    if block_path.exists():
        print(f"[skip] {block_path}", flush=True)
        return

    record_cycles = list(range(1, cycles + 1))   # 1 .. Ny
    n_t           = len(record_cycles)
    cycle_to_idx  = {c: i for i, c in enumerate(record_cycles)}

    sum_S   = np.zeros(n_t, dtype=np.float64)
    sumsq_S = np.zeros(n_t, dtype=np.float64)
    count   = np.zeros(n_t, dtype=np.int64)

    log_every = max(1, cycles // 10)

    def cycle_observer(*, cycle, G, batch_index, batch_start, batch_count):
        c = int(cycle)
        if c not in cycle_to_idx:
            return
        idx    = cycle_to_idx[c]
        S      = total_entropy_batch(G)      # (B,)
        sum_S[idx]   += S.sum()
        sumsq_S[idx] += (S ** 2).sum()
        count[idx]   += len(S)
        if c % log_every == 0:
            print(f"[obs] Ny={Ny} block={block_start}:{block_stop} "
                  f"cycle={c}/{cycles} S_mean={S.mean():.4f}", flush=True)

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
    model.run_markov_circuit(
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

    config = {
        "Nx": NX, "Ny": Ny, "nshell": NSHELL, "cycles": cycles,
        "alpha_1": ALPHA_1, "alpha_2": ALPHA_2,
        "dw_truncation": DW_TRUNCATION, "init_mode": INIT_MODE,
        "dtype": DTYPE, "sequence": SEQUENCE,
        "perfect_correction": PERFECT_CORRECTION, "postselect": POSTSELECT,
        "block_start": block_start, "block_stop": block_stop,
        "block_count": block_count,
    }

    np.savez_compressed(
        block_path,
        cycles   = np.array(record_cycles, dtype=np.int64),
        sum_S    = sum_S,
        sumsq_S  = sumsq_S,
        count    = count,
        ny       = np.array(Ny, dtype=np.int64),
        config_json = np.array(json.dumps(config, sort_keys=True)),
    )
    print(f"[saved] {block_path}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",    type=int,  required=True)
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
        cycles=p["cycles"], output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
