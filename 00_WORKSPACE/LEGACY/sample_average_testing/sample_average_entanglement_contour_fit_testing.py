#!/usr/bin/env python3
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

# Keep BLAS threading bounded; sample-level parallelism is handled explicitly below.
CPU_CAP = 20
N_JOBS = CPU_CAP
AFFINITY_START = 90
for env_name in [
    "MY_CPU_COUNT",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]:
    os.environ[env_name] = "1"

try:
    os.sched_setaffinity(0, set(range(AFFINITY_START, AFFINITY_START + CPU_CAP)))
except Exception as exc:
    print(f"[info] CPU affinity not set: {exc}")
try:
    print(f"[info] CPU affinity = {sorted(os.sched_getaffinity(0))}")
except Exception:
    pass

import time
import zipfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.lib import format as npformat
from tqdm import tqdm

from fgtn.classA_U1FGTN import classA_U1FGTN

DATA_PATH = ROOT / (
    "cache/G_history_samples/N12x31/"
    "N12x31_C100_S100_nshNone_DW1_init-default_n_a0.5_"
    "seq-dw_symmetric_exclNone_ps0_pst0_fm0_pc0_markov_circuit.npz"
)
OUT_DIR = ROOT / "sample_average_testing"
FIG_DIR = OUT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_COUNTS = [1, 10, 50, 100]
Y0_BATCH = 30
SAMPLE_BATCH = 8
AY_MIN = 4
ALPHA_1 = 30.0
ALPHA_2 = 1.0


def contiguous_x_block(nx, x_start, x_stop_inclusive):
    x_start = int(x_start) % nx
    x_stop_inclusive = int(x_stop_inclusive) % nx
    if x_start <= x_stop_inclusive:
        return np.arange(x_start, x_stop_inclusive + 1, dtype=int)
    return np.concatenate(
        [
            np.arange(x_start, nx, dtype=int),
            np.arange(0, x_stop_inclusive + 1, dtype=int),
        ]
    )


def wrap_y_block(y0, ay, ny):
    return (np.arange(int(ay), dtype=int) + int(y0)) % int(ny)


def subsystem_indices_full_x(nx, ny, y_vals):
    idx = []
    for y in np.asarray(y_vals, dtype=int):
        base = 2 * nx * int(y)
        for x in range(nx):
            idx.append(base + 2 * x)
            idx.append(base + 2 * x + 1)
    return np.asarray(idx, dtype=int)


def entanglement_contour_batch(g_batch, nx, ny_sub):
    arr = np.asarray(g_batch, dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError(f"expected (B,N,N); got {arr.shape}")
    bsz, nlayer, nlayer2 = arr.shape
    if nlayer != nlayer2:
        raise ValueError(f"subsystem matrices must be square; got {arr.shape}")
    eye = np.eye(nlayer, dtype=np.complex128)
    g2 = 0.5 * (eye + arr)
    g2 = 0.5 * (g2 + np.swapaxes(g2.conj(), -1, -2))
    evals, vecs = np.linalg.eigh(g2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
    f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    diagf = np.einsum("bik,bk,bik->bi", vecs, f_eigs, vecs.conj(), optimize=True).real
    diagf = diagf.reshape(bsz, 2, nx, ny_sub, order="F")
    return diagf.sum(axis=1)


def fit_line_with_error(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size:
        order = np.argsort(x)
        x = x[order]
        y = y[order]
    if x.size < 3:
        return {
            "m": np.nan,
            "m_err": np.nan,
            "b": np.nan,
            "r2": np.nan,
            "x": x,
            "y": y,
            "y_fit": np.full_like(y, np.nan),
        }

    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        m, b = float(coeffs[0]), float(coeffs[1])
        m_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    except Exception:
        m, b, m_err = np.nan, np.nan, np.nan

    if not np.isfinite(m):
        return {
            "m": np.nan,
            "m_err": np.nan,
            "b": np.nan,
            "r2": np.nan,
            "x": x,
            "y": y,
            "y_fit": np.full_like(y, np.nan),
        }

    y_fit = m * x + b
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    return {"m": m, "m_err": m_err, "b": b, "r2": r2, "x": x, "y": y, "y_fit": y_fit}


def _read_npy_header(fp):
    version = npformat.read_magic(fp)
    if version == (1, 0):
        return npformat.read_array_header_1_0(fp)
    if version in {(2, 0), (3, 0)}:
        return npformat.read_array_header_2_0(fp)
    raise ValueError(f"unsupported .npy version {version}")


def _skip_exact(fp, nbytes, buffer_size=8 * 1024 * 1024):
    remaining = int(nbytes)
    while remaining > 0:
        chunk = fp.read(min(buffer_size, remaining))
        if not chunk:
            raise EOFError(f"unexpected EOF while skipping {nbytes} bytes")
        remaining -= len(chunk)


def _read_exact(fp, nbytes):
    remaining = int(nbytes)
    chunks = []
    while remaining > 0:
        chunk = fp.read(remaining)
        if not chunk:
            raise EOFError(f"unexpected EOF while reading {nbytes} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def load_final_batch(path, max_samples):
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as data:
        if "G_final" in data:
            g_final = np.asarray(data["G_final"], dtype=np.complex128)
            if g_final.ndim == 2:
                g_final = g_final[None, ...]
            if g_final.ndim != 3 or g_final.shape[1] != g_final.shape[2]:
                raise ValueError(f"G_final must have shape (S,N,N); got {g_final.shape}")
            keep = min(g_final.shape[0], max_samples)
            return g_final[:keep]

    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        if "G_hist.npy" not in names:
            raise KeyError(f"'G_final' or 'G_hist' not found in {path}")

        with zf.open("G_hist.npy") as fp:
            shape, fortran_order, dtype_descr = _read_npy_header(fp)
            dtype = np.dtype(dtype_descr)
            if fortran_order:
                raise ValueError("Fortran-order G_hist is not supported by the streaming loader")
            if len(shape) == 4:
                total_samples, time_steps, nlayer, nlayer2 = shape
            elif len(shape) == 3:
                total_samples, nlayer, nlayer2 = shape
                time_steps = 1
            else:
                raise ValueError(f"G_hist must have shape (S,T,N,N) or (S,N,N); got {shape}")
            if nlayer != nlayer2:
                raise ValueError(f"G_hist must be square in the last dims; got {shape}")

            keep = min(int(total_samples), int(max_samples))
            slice_bytes = nlayer * nlayer * dtype.itemsize
            skip_bytes = (time_steps - 1) * slice_bytes
            g_final = np.empty((keep, nlayer, nlayer), dtype=dtype)

            bar = tqdm(
                range(keep),
                desc="stream final states from G_hist",
                unit="sample",
            )
            for s_idx in bar:
                if skip_bytes > 0:
                    _skip_exact(fp, skip_bytes)
                buf = _read_exact(fp, slice_bytes)
                g_final[s_idx] = np.frombuffer(buf, dtype=dtype).reshape(nlayer, nlayer)
                bar.set_postfix(
                    {
                        "sample": f"{s_idx + 1}/{keep}",
                        "time": datetime.now().strftime("%H:%M:%S"),
                    },
                    refresh=False,
                )

    return np.asarray(g_final, dtype=np.complex128)


def build_subsystem_cache(nx, ny, ay_values):
    cache = {}
    bar = tqdm(ay_values, desc="precompute subsystem indices", unit="Ay")
    for ay in bar:
        cache[int(ay)] = [
            subsystem_indices_full_x(nx, ny, wrap_y_block(y0, ay, ny))
            for y0 in range(ny)
        ]
    return cache


def compute_sample_first_mean_curves(
    g_final, ay_values, subsystem_cache, left_xs, right_xs, nx, ny, y0_batch, sample_counts, sample_batch
):
    sample_counts = sorted(int(n) for n in sample_counts if int(n) <= g_final.shape[0])
    if not sample_counts:
        raise ValueError("No requested sample counts are available in the loaded data")

    y0_batch = max(1, min(int(y0_batch), int(ny)))
    sample_batch = max(1, min(int(sample_batch), int(g_final.shape[0])))
    left_means = {n: np.zeros(len(ay_values), dtype=float) for n in sample_counts}
    right_means = {n: np.zeros(len(ay_values), dtype=float) for n in sample_counts}

    ay_bar = tqdm(ay_values, desc="Ay scan (sample-first accumulation)", unit="Ay")
    for i, ay in enumerate(ay_bar):
        sub_idx_list = subsystem_cache[int(ay)]
        left_y0_totals = {n: 0.0 for n in sample_counts}
        right_y0_totals = {n: 0.0 for n in sample_counts}

        for y0_start in range(0, ny, y0_batch):
            idx_chunk = sub_idx_list[y0_start : y0_start + y0_batch]
            chunk_y0 = len(idx_chunk)
            left_vals = np.empty((g_final.shape[0], chunk_y0), dtype=float)
            right_vals = np.empty((g_final.shape[0], chunk_y0), dtype=float)

            for s_start in range(0, g_final.shape[0], sample_batch):
                s_stop = min(s_start + sample_batch, g_final.shape[0])
                g_chunk = g_final[s_start:s_stop]
                g_sub_batch = np.stack(
                    [g_sample[np.ix_(idx, idx)] for g_sample in g_chunk for idx in idx_chunk],
                    axis=0,
                )
                s_batch = entanglement_contour_batch(g_sub_batch, nx, int(ay)).reshape(
                    s_stop - s_start, chunk_y0, nx, int(ay)
                )
                left_vals[s_start:s_stop] = np.sum(s_batch[:, :, left_xs, :], axis=(2, 3))
                right_vals[s_start:s_stop] = np.sum(s_batch[:, :, right_xs, :], axis=(2, 3))

            left_prefix = np.cumsum(left_vals, axis=0)
            right_prefix = np.cumsum(right_vals, axis=0)
            for n in sample_counts:
                left_y0_totals[n] += float(np.sum(left_prefix[n - 1] / n))
                right_y0_totals[n] += float(np.sum(right_prefix[n - 1] / n))

        for n in sample_counts:
            left_means[n][i] = left_y0_totals[n] / ny
            right_means[n][i] = right_y0_totals[n] / ny

        ay_bar.set_postfix(
            {
                "Ay": int(ay),
                "subdim": int(2 * nx * ay),
                "y0_batch": y0_batch,
                "sample_batch": sample_batch,
                "time": datetime.now().strftime("%H:%M:%S"),
            },
            refresh=False,
        )

    return {n: {"left": left_means[n], "right": right_means[n]} for n in sample_counts}


def build_results_for_sample_counts(mean_curves_by_count, ay_values, ny):
    mid = ny / 2.0
    mask_lower = ay_values < mid
    mask_upper = ay_values > mid
    x_lower = np.log(np.sin(np.pi * ay_values[mask_lower] / ny))
    x_upper = np.log(np.sin(np.pi * ay_values[mask_upper] / ny))

    results = {}
    for n, mean_curves in mean_curves_by_count.items():
        left_mean = mean_curves["left"]
        right_mean = mean_curves["right"]

        curves = {
            "Left DW, lower branch": left_mean[mask_lower],
            "Left DW, upper branch": left_mean[mask_upper],
            "Right DW, lower branch": right_mean[mask_lower],
            "Right DW, upper branch": right_mean[mask_upper],
        }
        fits = {
            "Left DW, lower branch": fit_line_with_error(x_lower, curves["Left DW, lower branch"]),
            "Left DW, upper branch": fit_line_with_error(x_upper, curves["Left DW, upper branch"]),
            "Right DW, lower branch": fit_line_with_error(x_lower, curves["Right DW, lower branch"]),
            "Right DW, upper branch": fit_line_with_error(x_upper, curves["Right DW, upper branch"]),
        }
        results[int(n)] = {
            "curves": curves,
            "fits": fits,
            "x_lower": x_lower,
            "x_upper": x_upper,
            "ay_lower": ay_values[mask_lower],
            "ay_upper": ay_values[mask_upper],
        }
    return results


def save_pdf(pdf_path, results, left_xs, right_xs, ay_values):
    styles = {
        "Left DW, lower branch": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
        "Left DW, upper branch": {"color": "#1f77b4", "marker": "s", "linestyle": "--"},
        "Right DW, lower branch": {"color": "#ff7f0e", "marker": "o", "linestyle": "-"},
        "Right DW, upper branch": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    }

    ordered_counts = list(results.keys())
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        axes = axes.ravel()

        for ax, n in zip(axes, ordered_counts):
            res = results[n]
            for name, style in styles.items():
                fit = res["fits"][name]
                x_vals = res["x_lower"] if "lower" in name else res["x_upper"]
                y_vals = res["curves"][name]
                ax.plot(
                    x_vals,
                    y_vals,
                    linestyle="None",
                    marker=style["marker"],
                    ms=4,
                    color=style["color"],
                    label=(
                        f"{name}: m={fit['m']:.4g} +/- {fit['m_err']:.2g}, "
                        f"R^2={fit['r2']:.4f}"
                    ),
                )
                if fit["x"].size > 1 and np.all(np.isfinite(fit["y_fit"])):
                    ax.plot(
                        fit["x"],
                        fit["y_fit"],
                        style["linestyle"],
                        lw=1.5,
                        color=style["color"],
                    )

            ax.set_title(
                f"samples={n} | y0-averaged full-width subsystem\n"
                f"left x={left_xs.tolist()}, right x={right_xs.tolist()}"
            )
            ax.set_xlabel(r"$\log[\sin(\pi A_y/N_y)]$")
            ax.set_ylabel(r"$\sum_{x \in I_x}\sum_{y \in A(y_0,A_y)} s_A(x,y)$")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

        for ax in axes[len(ordered_counts) :]:
            ax.axis("off")

        fig.suptitle(
            "Sample-averaged + subsystem-origin-averaged integrated contour fits\n"
            f"{DATA_PATH.name} | Ay scan={int(ay_values[0])}..{int(ay_values[-1])}",
            fontsize=14,
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig_table, ax_table = plt.subplots(1, 1, figsize=(15, 10))
        ax_table.axis("off")
        rows = []
        for n in ordered_counts:
            for name in [
                "Left DW, lower branch",
                "Left DW, upper branch",
                "Right DW, lower branch",
                "Right DW, upper branch",
            ]:
                fit = results[n]["fits"][name]
                rows.append(
                    [
                        str(n),
                        name,
                        f"{fit['m']:.8e}",
                        f"{fit['m_err']:.8e}",
                        f"{fit['b']:.8e}",
                        f"{fit['r2']:.8f}",
                    ]
                )

        table = ax_table.table(
            cellText=rows,
            colLabels=["samples", "curve", "slope", "slope_err", "intercept", "R^2"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax_table.set_title(
            "Fit summary table\n"
            f"{DATA_PATH.name}",
            fontsize=14,
            pad=20,
        )
        pdf.savefig(fig_table)
        plt.close(fig_table)


def main():
    t0 = time.time()
    print(f"[info] input cache = {DATA_PATH}")
    print(f"[info] output dir  = {FIG_DIR}")
    print(f"[info] cpu cap     = {CPU_CAP}")
    print(f"[info] thread jobs = {N_JOBS}")
    print(f"[info] y0 batch    = {Y0_BATCH}")
    print(f"[info] sample batch= {SAMPLE_BATCH}")

    max_samples = max(SAMPLE_COUNTS)
    g_final = load_final_batch(DATA_PATH, max_samples=max_samples)
    n_samples, nlayer, nlayer2 = g_final.shape
    if nlayer != nlayer2:
        raise ValueError(f"final states must be square; got {g_final.shape}")

    model = classA_U1FGTN(12, 31, nshell=None, DW=True, alpha_1=ALPHA_1, alpha_2=ALPHA_2)
    nx = model.Nx
    ny = model.Ny
    expected_n = 2 * nx * ny
    if nlayer != expected_n:
        raise ValueError(f"expected Nlayer={expected_n} from model, got {nlayer}")

    x0 = int(model.DW_loc[0]) % nx
    x1 = int(model.DW_loc[1]) % nx
    left_xs = contiguous_x_block(nx, 0, nx-1)
    right_xs = contiguous_x_block(nx, 0, nx-1)

    ay_values = np.arange(AY_MIN, (ny + 1) - AY_MIN, dtype=int)
    subsystem_cache = build_subsystem_cache(nx, ny, ay_values)

    print(f"[info] DW locations       = ({x0}, {x1})")
    print(f"[info] left x interval    = {left_xs.tolist()}")
    print(f"[info] right x interval   = {right_xs.tolist()}")
    print(f"[info] available samples  = {n_samples}")
    print(f"[info] Ay scan            = {int(ay_values[0])}..{int(ay_values[-1])}")
    print("[info] averaging order   = sample-first within each y0, then y0-average")

    mean_curves_by_count = compute_sample_first_mean_curves(
        g_final,
        ay_values,
        subsystem_cache,
        left_xs,
        right_xs,
        nx,
        ny,
        Y0_BATCH,
        SAMPLE_COUNTS,
        SAMPLE_BATCH,
    )
    results = build_results_for_sample_counts(mean_curves_by_count, ay_values, ny)

    pdf_path = FIG_DIR / (
        DATA_PATH.stem + "_sample_average_y0_average_integrated_contour_fits.pdf"
    )
    save_pdf(pdf_path, results, left_xs, right_xs, ay_values)

    print("[info] saved pdf:")
    print(f"  {pdf_path}")
    print("[info] fit summary")
    for n, res in results.items():
        print(f"  samples={n}")
        for name, fit in res["fits"].items():
            print(
                f"    - {name}: slope={fit['m']:.8e}, "
                f"slope_err={fit['m_err']:.8e}, intercept={fit['b']:.8e}, R^2={fit['r2']:.8f}"
            )

    print(f"[info] elapsed = {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
