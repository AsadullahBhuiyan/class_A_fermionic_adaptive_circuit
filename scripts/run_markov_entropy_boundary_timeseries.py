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
import multiprocessing as mp
import time
from datetime import datetime

# Keep CPU allocation behavior aligned with run_markov_p_sweep.py
cpu_cap = 52
os.environ["MY_CPU_COUNT"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_MAX_THREADS"] = str(1)
try:
    os.sched_setaffinity(0, set(range(cpu_cap)))
except Exception as exc:
    print(f"CPU affinity not set: {exc}")
try:
    print(os.sched_getaffinity(0))
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

import fgtn.classA_U1FGTN as classA_U1FGTN
import importlib

importlib.reload(classA_U1FGTN)
from fgtn.classA_U1FGTN import classA_U1FGTN
def entanglement_contour_batch(Gtt_batch, Nx, Ny):
    """
    Gtt_batch: (B, Nlayer, Nlayer)
    Returns: (B, Nx, Ny)
    """
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
        return np.zeros(arr.shape[1-axis], dtype=float)
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(n)


def _safe_log_yerr(mean_vals, err_vals, eps=1e-12):
    lower = np.maximum(mean_vals - err_vals, eps)
    upper = mean_vals + err_vals
    yerr_low = mean_vals - lower
    yerr_high = np.maximum(upper - mean_vals, 0.0)
    return np.vstack([yerr_low, yerr_high])


def main():
    t0 = time.time()

    # Lattice + DW parameters
    Nx, Ny = 12, 31
    alpha_1, alpha_2 = 30, 1
    cycles = 50
    p_meas = 1.0
    samples = 100

    model = classA_U1FGTN(Nx, Ny, nshell=2, DW=True, alpha_1=alpha_1, alpha_2=alpha_2)
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc not set; need DW=True with valid alpha profile")

    x0 = int(model.DW_loc[0]) % Nx
    x1 = int(model.DW_loc[1]) % Nx
    left_pair = [x0 % Nx, (x0 + 1) % Nx]
    right_pair = [x1 % Nx, (x1 - 1) % Nx]

    # 1) Run Markov circuit from maximally mixed initial state
    init_mode = "maxmix"
    strict_maxmix_check = False  # set True to hard-fail if t=0 is not maxmix
    res = model.run_markov_circuit(
        G_history=True,
        cycles=cycles,
        progress=True,
        init_mode=init_mode,
        save=True,
        samples=samples,
        p_meas=p_meas,
        n_jobs=cpu_cap,
        parallelize_samples=True,
        sequence="random",
        top_triv_back_forth=True,
        max_in_flight=9*cpu_cap//10,
        save_suffix="_entropy_boundary_timeseries",
    )
    print(f"Completed p_meas={p_meas:.2f}")

    G_hist = res.get("G_hist")
    if G_hist is None:
        save_path = res.get("save_path")
        if save_path and os.path.exists(save_path):
            with np.load(save_path) as data:
                G_hist = data["G_hist"]
        else:
            raise RuntimeError("G_hist not available; enable G_history or provide saved path.")

    S, T, N, N2 = G_hist.shape
    if N != N2:
        raise ValueError(f"G_hist should be square in last dims, got {N}x{N2}.")
    if N != 2 * Nx * Ny:
        raise ValueError(f"Expected N={2*Nx*Ny} for Nx={Nx}, Ny={Ny}; got N={N}.")

    # Sanity check: maxmix init should be zero at t=0 for top layer
    if init_mode == "maxmix":
        max_abs_init = float(np.max(np.abs(G_hist[:, 0])))
        if max_abs_init > 1e-6:
            max_abs_t1 = float(np.max(np.abs(G_hist[:, 1]))) if T > 1 else float("nan")
            max_abs_d01 = (
                float(np.max(np.abs(G_hist[:, 1] - G_hist[:, 0]))) if T > 1 else float("nan")
            )
            msg = (
                f"Expected maxmix init (G=0) at t=0; max |G|={max_abs_init:.3g}. "
                f"(t1 max |G|={max_abs_t1:.3g}, max |Δ01|={max_abs_d01:.3g}) "
                "Check save_init or dataset mismatch."
            )
            if strict_maxmix_check:
                raise RuntimeError(msg)
            print(f"[warn] {msg}")

    # 2,3,4) Whole-system contour, boundary sums, trajectory averages
    left_by_sample = np.zeros((S, T), dtype=float)
    right_by_sample = np.zeros((S, T), dtype=float)

    tbar = tqdm(range(T), desc="Boundary contour vs cycle", unit="t")
    for t_idx in tbar:
        tbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
        s_batch = entanglement_contour_batch(G_hist[:, t_idx], Nx, Ny)  # (S, Nx, Ny)
        left_by_sample[:, t_idx] = np.sum(s_batch[:, left_pair, :], axis=(1, 2))
        right_by_sample[:, t_idx] = np.sum(s_batch[:, right_pair, :], axis=(1, 2))

    left_avg = np.mean(left_by_sample, axis=0)
    right_avg = np.mean(right_by_sample, axis=0)
    left_stderr = _stderr(left_by_sample, axis=0)
    right_stderr = _stderr(right_by_sample, axis=0)

    # 5,6) Plot integrated contour vs cycle (linear and log-y subplots)
    t_vals = np.arange(T, dtype=int)

    figs_dir = os.path.join("figs")
    os.makedirs(figs_dir, exist_ok=True)
    save_path = res.get("save_path")
    if save_path:
        cache_key = os.path.splitext(os.path.basename(save_path))[0]
        pdf_name = f"{cache_key}_boundary_entropy_vs_cycle.pdf"
    else:
        pdf_name = f"markov_p_sweep_boundary_entropy_pm{p_meas:.2f}.pdf"
    pdf_path = os.path.join(figs_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

        ax_lin.errorbar(
            t_vals, left_avg, yerr=left_stderr, marker="o", lw=1.5, capsize=3, label=f"Left boundary (x={left_pair})"
        )
        ax_lin.errorbar(
            t_vals, right_avg, yerr=right_stderr, marker="o", lw=1.5, capsize=3, label=f"Right boundary (x={right_pair})"
        )
        ax_lin.set_xlabel("cycle t")
        ax_lin.set_ylabel(r"$\sum_y s(x,y)$")
        ax_lin.set_title("Integrated contour vs cycle (linear y)")
        ax_lin.grid(alpha=0.3)
        ax_lin.legend(fontsize=8)

        ax_log.errorbar(
            t_vals,
            left_avg,
            yerr=_safe_log_yerr(left_avg, left_stderr),
            marker="o",
            lw=1.5,
            capsize=3,
            label=f"Left boundary (x={left_pair})",
        )
        ax_log.errorbar(
            t_vals,
            right_avg,
            yerr=_safe_log_yerr(right_avg, right_stderr),
            marker="o",
            lw=1.5,
            capsize=3,
            label=f"Right boundary (x={right_pair})",
        )
        ax_log.set_yscale("log")
        ax_log.set_xlabel("cycle t")
        ax_log.set_ylabel(r"$\sum_y s(x,y)$")
        ax_log.set_title("Integrated contour vs cycle (log y)")
        ax_log.grid(alpha=0.3)
        ax_log.legend(fontsize=8)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    elapsed = time.time() - t0
    print(f"Saved plots to {pdf_path}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
