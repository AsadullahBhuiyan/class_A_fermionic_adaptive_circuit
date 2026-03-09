# --- project bootstrap ---
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
# -------------------------

import os
import time
from datetime import datetime

# CPU allocation (keep consistent with other scripts)
cpu_cap = 30
os.environ["MY_CPU_COUNT"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_MAX_THREADS"] = str(1)
try:
    os.sched_setaffinity(0, set(range(60, 60+cpu_cap)))
except Exception as exc:
    print(f"CPU affinity not set: {exc}")
try:
    print(os.sched_getaffinity(0))
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fgtn.classA_U1FGTN import classA_U1FGTN
def entanglement_contour_batch(Gtt_batch, Nx, Ny):
    arr = np.asarray(Gtt_batch, dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError(f"expected (B,N,N); got {arr.shape}")

    B, Nlayer, _ = arr.shape
    I = np.eye(Nlayer, dtype=np.complex128)
    G2 = 0.5 * (I + arr)
    evals, vecs = np.linalg.eigh(G2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
    f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    diagF = np.einsum("bik,bk,bik->bi", vecs, f_eigs, vecs.conj(), optimize=True).real
    diagF = diagF.reshape(B, 2, Nx, Ny, order="F")
    return diagF.sum(axis=1)


def _stderr(arr, axis=0):
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros(arr.shape[1 - axis], dtype=float)
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(n)


def main():
    t0 = time.time()

    data_path = (
        "/home/abhuiyan/class_A_fermionic_adaptive_circuit/cache/G_history_samples/N12x31/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit.npz"
    )

    with tqdm(total=1, desc="load") as pbar:
        with np.load(data_path) as data:
            G_hist = data["G_hist"]
        pbar.update(1)

    S, T, N, N2 = G_hist.shape
    if N != N2:
        raise ValueError(f"G_hist should be square in last dims, got {N}x{N2}.")

    Nx, Ny = 12, 31
    if N != 2 * Nx * Ny:
        raise ValueError(f"Expected N={2*Nx*Ny} for Nx={Nx}, Ny={Ny}; got N={N}.")

    model = classA_U1FGTN(Nx, Ny, DW=True, alpha_1=30, alpha_2=1)
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc not set; need DW=True with valid alpha profile")
    x0 = int(model.DW_loc[0]) % Nx
    x1 = int(model.DW_loc[1]) % Nx
    left_pair = [x0 % Nx]#, (x0 + 1) % Nx]
    right_pair = [x1 % Nx]#, (x1 - 1) % Nx]

    extra_xs = [2, 6, 10]

    # Half-system: keep y >= Ny//2 (top half)
    y_cut = Ny // 2
    yA = np.arange(y_cut, Ny)
    Ny_sub = len(yA)
    sub_indices = []
    for y in yA:
        base = 2 * Nx * y
        for x in range(Nx):
            sub_indices.append(base + 2 * x)
            sub_indices.append(base + 2 * x + 1)
    sub_indices = np.array(sub_indices, dtype=int)

    left_by_sample = np.zeros((S, T), dtype=float)
    right_by_sample = np.zeros((S, T), dtype=float)
    extra_by_sample = {x: np.zeros((S, T), dtype=float) for x in extra_xs}

    for t_idx in tqdm(range(T), desc="cycles"):
        G_sub = G_hist[:, t_idx][:, sub_indices][:, :, sub_indices]
        s_batch = entanglement_contour_batch(G_sub, Nx, Ny_sub)
        left_by_sample[:, t_idx] = np.sum(s_batch[:, left_pair, :], axis=(1, 2))
        right_by_sample[:, t_idx] = np.sum(s_batch[:, right_pair, :], axis=(1, 2))
        for x in extra_xs:
            extra_by_sample[x][:, t_idx] = np.sum(s_batch[:, x, :], axis=1)

    left_avg = np.mean(left_by_sample, axis=0)
    right_avg = np.mean(right_by_sample, axis=0)
    left_stderr = _stderr(left_by_sample, axis=0)
    right_stderr = _stderr(right_by_sample, axis=0)

    extra_avg = {x: np.mean(extra_by_sample[x], axis=0) for x in extra_xs}
    extra_stderr = {x: _stderr(extra_by_sample[x], axis=0) for x in extra_xs}

    t_vals = np.arange(T, dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)

    axes[0].errorbar(t_vals, left_avg, yerr=left_stderr, marker="o", ms=3, lw=1.2, label=f"x={x0} (left DW)")
    axes[0].errorbar(t_vals, right_avg, yerr=right_stderr, marker="o", ms=3, lw=1.2, label=f"x={x1} (right DW)")
    for x in extra_xs:
        axes[0].errorbar(t_vals, extra_avg[x], yerr=extra_stderr[x], marker="o", ms=3, lw=1.0, label=f"x={x}")
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel("integrated contour (top half)")
    axes[0].set_title("Linear scale (top half)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].errorbar(t_vals, left_avg, yerr=left_stderr, marker="o", ms=3, lw=1.2, label=f"x={x0} (left DW)")
    axes[1].errorbar(t_vals, right_avg, yerr=right_stderr, marker="o", ms=3, lw=1.2, label=f"x={x1} (right DW)")
    for x in extra_xs:
        axes[1].errorbar(t_vals, extra_avg[x], yerr=extra_stderr[x], marker="o", ms=3, lw=1.0, label=f"x={x}")
    axes[1].set_xlabel("cycle")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("integrated contour (log, top half)")
    axes[1].set_title("Log scale (top half)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    figs_dir = os.path.join("figs")
    os.makedirs(figs_dir, exist_ok=True)
    cache_key = os.path.splitext(os.path.basename(data_path))[0]
    pdf_path = os.path.join(figs_dir, f"{cache_key}_boundary_entropy_with_xcols_tophalf.pdf")
    fig.savefig(pdf_path, dpi=200)
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"Saved: {pdf_path}")
    print(f"Elapsed: {elapsed:0.2f}s")


if __name__ == "__main__":
    main()
