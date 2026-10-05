#!/usr/bin/env python3
# --- project bootstrap ---
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
# -------------------------

# Keep BLAS threading bounded; the script batches dense linear algebra explicitly.
CPU_CAP = 22
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
AY_MIN = 3
ALPHA_1 = 30.0
ALPHA_2 = 1.0
L_A = 4
L_C = 4
# Allowed symbolic entries:
# - "left_dw_strip": x in [x_dw-1, x_dw, x_dw+1]
# - "full_width": x in [0, Nx-1]
x_sum_windows = [
    "left_dw_strip",
    "full_width",
]


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


def wrap_y_block(y0, length, ny):
    return (np.arange(int(length), dtype=int) + int(y0)) % int(ny)


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


def parse_run_metadata(path):
    stem = Path(path).stem
    m_n = re.search(r"N(\d+)x(\d+)", stem)
    m_c = re.search(r"_C(\d+)", stem)
    m_s = re.search(r"_S(\d+)", stem)
    if m_n is None or m_c is None or m_s is None:
        raise ValueError(f"Could not parse N/C/S metadata from {path}")
    return {
        "Nx": int(m_n.group(1)),
        "Ny": int(m_n.group(2)),
        "cycles": int(m_c.group(1)),
        "samples_total": int(m_s.group(1)),
    }


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

            bar = tqdm(range(keep), desc="stream final states from G_hist", unit="sample")
            for s_idx in bar:
                if skip_bytes > 0:
                    _skip_exact(fp, skip_bytes)
                buf = _read_exact(fp, slice_bytes)
                g_final[s_idx] = np.frombuffer(buf, dtype=dtype).reshape(nlayer, nlayer)
                bar.set_postfix(
                    {"sample": f"{s_idx + 1}/{keep}", "time": datetime.now().strftime("%H:%M:%S")},
                    refresh=False,
                )

    return np.asarray(g_final, dtype=np.complex128)


def normalize_x_sum_windows(nx, window_specs, dw_loc):
    out = []
    seen = set()
    for spec in window_specs:
        if isinstance(spec, str):
            if spec == "full_width":
                xs = contiguous_x_block(nx, 0, nx - 1)
                spec_tuple = (0, nx - 1)
                label = f"x=[0,{nx - 1}]"
            elif spec == "left_dw_strip":
                x0 = int(dw_loc[0]) % nx
                xs = contiguous_x_block(nx, x0 - 1, x0 + 1)
                spec_tuple = (x0 - 1, x0 + 1)
                label = f"left DW strip x={xs.tolist()}"
            else:
                raise ValueError(f"Unknown symbolic x_sum_window spec: {spec}")
        else:
            if len(spec) != 2:
                raise ValueError(f"Each x_sum_window must be a 2-tuple; got {spec}")
            xs = contiguous_x_block(nx, spec[0], spec[1])
            spec_tuple = tuple(map(int, spec))
            if xs.size == nx:
                label = f"x=[0,{nx - 1}]"
            else:
                label = f"x={xs.tolist()}"
        key = tuple(int(x) for x in xs.tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append({"spec": spec_tuple, "xs": xs, "key": key, "label": label})
    if not out:
        raise ValueError("No valid x_sum_windows were provided")
    return out


def build_contour_subsystem_cache(nx, ny, ay_values):
    cache = {}
    bar = tqdm(ay_values, desc="precompute contour subsystem indices", unit="Ay")
    for ay in bar:
        cache[int(ay)] = [
            subsystem_indices_full_x(nx, ny, wrap_y_block(y0, ay, ny))
            for y0 in range(ny)
        ]
    return cache


def build_mi_subsystem_cache(nx, ny, lb_values):
    cache = {}
    bar = tqdm(lb_values, desc="precompute MI subsystem indices", unit="lB")
    for l_b in bar:
        y0_entries = []
        for y0 in range(ny):
            y_a = wrap_y_block(y0, L_A, ny)
            y_b = wrap_y_block(y0 + L_A, int(l_b), ny)
            y_c = wrap_y_block(y0 + L_A + int(l_b), L_C, ny)
            y_sets = {
                "A": y_a,
                "B": y_b,
                "C": y_c,
                "AB": np.concatenate([y_a, y_b]),
                "AC": np.concatenate([y_a, y_c]),
                "BC": np.concatenate([y_b, y_c]),
                "ABC": np.concatenate([y_a, y_b, y_c]),
            }
            y0_entries.append(
                {
                    key: {
                        "idx": subsystem_indices_full_x(nx, ny, y_vals),
                        "ny_sub": len(y_vals),
                    }
                    for key, y_vals in y_sets.items()
                }
            )
        cache[int(l_b)] = y0_entries
    return cache


def contour_entropy_sample_first_mean(
    g_final,
    ay_values,
    contour_cache,
    x_windows,
    nx,
    ny,
    y0_batch,
    sample_counts,
    sample_batch,
):
    sample_counts = sorted(int(n) for n in sample_counts if int(n) <= g_final.shape[0])
    if not sample_counts:
        raise ValueError("No requested sample counts are available in the loaded data")

    y0_batch = max(1, min(int(y0_batch), int(ny)))
    sample_batch = max(1, min(int(sample_batch), int(g_final.shape[0])))
    mean_entropy = {
        window["key"]: {n: np.zeros(len(ay_values), dtype=float) for n in sample_counts}
        for window in x_windows
    }

    ay_bar = tqdm(ay_values, desc="Ay scan (contour)", unit="Ay")
    for i, ay in enumerate(ay_bar):
        sub_idx_list = contour_cache[int(ay)]
        y0_totals = {
            window["key"]: {n: 0.0 for n in sample_counts}
            for window in x_windows
        }

        for y0_start in range(0, ny, y0_batch):
            idx_chunk = sub_idx_list[y0_start : y0_start + y0_batch]
            chunk_y0 = len(idx_chunk)
            entropy_vals = {
                window["key"]: np.empty((g_final.shape[0], chunk_y0), dtype=float)
                for window in x_windows
            }

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
                for window in x_windows:
                    entropy_vals[window["key"]][s_start:s_stop] = np.sum(
                        s_batch[:, :, window["xs"], :], axis=(2, 3)
                    )

            for window in x_windows:
                prefix = np.cumsum(entropy_vals[window["key"]], axis=0)
                for n in sample_counts:
                    y0_totals[window["key"]][n] += float(np.sum(prefix[n - 1] / n))

        for window in x_windows:
            for n in sample_counts:
                mean_entropy[window["key"]][n][i] = y0_totals[window["key"]][n] / ny

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

    return mean_entropy


def subsystem_partial_entropy_batch(g_chunk, idx, nx, ny_sub, x_windows):
    g_sub_batch = np.stack([g_sample[np.ix_(idx, idx)] for g_sample in g_chunk], axis=0)
    s_batch = entanglement_contour_batch(g_sub_batch, nx, ny_sub)
    return {
        window["key"]: np.sum(s_batch[:, window["xs"], :], axis=(1, 2))
        for window in x_windows
    }


def mi_i2_minus_i3_sample_first_mean(g_final, lb_values, mi_cache, x_windows, nx, ny, sample_counts, sample_batch):
    sample_counts = sorted(int(n) for n in sample_counts if int(n) <= g_final.shape[0])
    if not sample_counts:
        raise ValueError("No requested sample counts are available in the loaded data")

    sample_batch = max(1, min(int(sample_batch), int(g_final.shape[0])))
    mean_d = {
        window["key"]: {n: np.zeros(len(lb_values), dtype=float) for n in sample_counts}
        for window in x_windows
    }

    lb_bar = tqdm(lb_values, desc="lB scan (MI)", unit="lB")
    for i, l_b in enumerate(lb_bar):
        y0_entries = mi_cache[int(l_b)]
        y0_totals = {
            window["key"]: {n: 0.0 for n in sample_counts}
            for window in x_windows
        }

        for y0_entry in y0_entries:
            d_vals = {
                window["key"]: np.empty(g_final.shape[0], dtype=float)
                for window in x_windows
            }
            for s_start in range(0, g_final.shape[0], sample_batch):
                s_stop = min(s_start + sample_batch, g_final.shape[0])
                g_chunk = g_final[s_start:s_stop]
                s_a = subsystem_partial_entropy_batch(g_chunk, y0_entry["A"]["idx"], nx, y0_entry["A"]["ny_sub"], x_windows)
                s_b = subsystem_partial_entropy_batch(g_chunk, y0_entry["B"]["idx"], nx, y0_entry["B"]["ny_sub"], x_windows)
                s_c = subsystem_partial_entropy_batch(g_chunk, y0_entry["C"]["idx"], nx, y0_entry["C"]["ny_sub"], x_windows)
                s_ab = subsystem_partial_entropy_batch(g_chunk, y0_entry["AB"]["idx"], nx, y0_entry["AB"]["ny_sub"], x_windows)
                s_ac = subsystem_partial_entropy_batch(g_chunk, y0_entry["AC"]["idx"], nx, y0_entry["AC"]["ny_sub"], x_windows)
                s_bc = subsystem_partial_entropy_batch(g_chunk, y0_entry["BC"]["idx"], nx, y0_entry["BC"]["ny_sub"], x_windows)
                s_abc = subsystem_partial_entropy_batch(g_chunk, y0_entry["ABC"]["idx"], nx, y0_entry["ABC"]["ny_sub"], x_windows)

                for window in x_windows:
                    key = window["key"]
                    i2 = s_a[key] + s_c[key] - s_ac[key]
                    i3 = s_a[key] + s_b[key] + s_c[key] - s_ab[key] - s_ac[key] - s_bc[key] + s_abc[key]
                    d_vals[key][s_start:s_stop] = i2 - i3

            for window in x_windows:
                prefix = np.cumsum(d_vals[window["key"]])
                for n in sample_counts:
                    y0_totals[window["key"]][n] += float(prefix[n - 1] / n)

        for window in x_windows:
            for n in sample_counts:
                mean_d[window["key"]][n][i] = y0_totals[window["key"]][n] / ny

        lb_bar.set_postfix(
            {
                "lB": int(l_b),
                "sample_batch": sample_batch,
                "time": datetime.now().strftime("%H:%M:%S"),
            },
            refresh=False,
        )

    return mean_d


def chord_length(length, total_length):
    return (total_length / np.pi) * np.sin(np.pi * length / total_length)


def cross_ratio_chord(l_a, l_b, l_c, total_length):
    l_ab = l_a + l_b
    l_bc = l_b + l_c
    return (
        chord_length(l_a, total_length) * chord_length(l_c, total_length)
        / (chord_length(l_ab, total_length) * chord_length(l_bc, total_length))
    )


def build_contour_results(mean_entropy_by_window, ay_values, ny, x_windows, sample_counts):
    mid = ny / 2.0
    mask_lower = ay_values < mid
    mask_upper = ay_values > mid
    x_lower = np.log(np.sin(np.pi * ay_values[mask_lower] / ny))
    x_upper = np.log(np.sin(np.pi * ay_values[mask_upper] / ny))

    results = {int(n): {} for n in sample_counts}
    for window in x_windows:
        key = window["key"]
        for n in sample_counts:
            mean_entropy = mean_entropy_by_window[key][n]
            curves = {
                "Lower branch": mean_entropy[mask_lower],
                "Upper branch": mean_entropy[mask_upper],
            }
            fits = {
                "Lower branch": fit_line_with_error(x_lower, curves["Lower branch"]),
                "Upper branch": fit_line_with_error(x_upper, curves["Upper branch"]),
            }
            results[int(n)][key] = {
                "label": window["label"],
                "curves": curves,
                "fits": fits,
                "x_lower": x_lower,
                "x_upper": x_upper,
            }
    return results


def build_mi_results(mean_d_by_window, lb_values, ny, x_windows, sample_counts):
    x_list = np.array([cross_ratio_chord(L_A, int(l_b), L_C, ny) for l_b in lb_values], dtype=float)
    z_list = np.log(1.0 / (1.0 - x_list))

    results = {int(n): {} for n in sample_counts}
    for window in x_windows:
        key = window["key"]
        for n in sample_counts:
            d_avg = mean_d_by_window[key][n]
            results[int(n)][key] = {
                "label": window["label"],
                "x": x_list,
                "z": z_list,
                "lB": lb_values.copy(),
                "d_avg": d_avg,
                "fit": fit_line_with_error(z_list, d_avg),
            }
    return results


def save_pdf(pdf_path, contour_results, mi_results, x_windows, meta):
    ordered_counts = [int(n) for n in SAMPLE_COUNTS if int(n) in contour_results]
    window_colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(x_windows)))
    branch_styles = {
        "Lower branch": {"marker": "o", "linestyle": "-"},
        "Upper branch": {"marker": "s", "linestyle": "--"},
    }

    contour_title = (
        f"Contour fits | N={meta['Nx']}x{meta['Ny']} | C={meta['cycles']} | "
        f"sample avgs={ordered_counts}"
    )
    mi_title = (
        f"I2 - I3 fits | N={meta['Nx']}x{meta['Ny']} | C={meta['cycles']} | "
        f"sample avgs={ordered_counts}"
    )

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        axes = axes.ravel()
        for ax, n in zip(axes, ordered_counts):
            for color, window in zip(window_colors, x_windows):
                res = contour_results[n][window["key"]]
                for branch_name, branch_style in branch_styles.items():
                    fit = res["fits"][branch_name]
                    x_vals = res["x_lower"] if branch_name == "Lower branch" else res["x_upper"]
                    y_vals = res["curves"][branch_name]
                    ax.plot(
                        x_vals,
                        y_vals,
                        linestyle="None",
                        marker=branch_style["marker"],
                        ms=4,
                        color=color,
                        label=(
                            f"{res['label']}, {branch_name}: "
                            f"m={fit['m']:.4g} +/- {fit['m_err']:.2g}, R^2={fit['r2']:.3f}"
                        ),
                    )
                    if fit["x"].size > 1 and np.all(np.isfinite(fit["y_fit"])):
                        ax.plot(
                            fit["x"],
                            fit["y_fit"],
                            branch_style["linestyle"],
                            lw=1.4,
                            color=color,
                        )
            ax.set_title(f"samples={n}")
            ax.set_xlabel(r"$\log[\sin(\pi A_y/N_y)]$")
            ax.set_ylabel(r"$\sum_{x \in I_x}\sum_{y \in A(y_0,A_y)} s_R(x,y)$")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.suptitle(contour_title, fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        axes = axes.ravel()
        for ax, n in zip(axes, ordered_counts):
            for color, window in zip(window_colors, x_windows):
                res = mi_results[n][window["key"]]
                fit = res["fit"]
                ax.plot(
                    res["z"],
                    res["d_avg"],
                    linestyle="None",
                    marker="o",
                    ms=4,
                    color=color,
                    label=(
                        f"{res['label']}: m={fit['m']:.4g} +/- {fit['m_err']:.2g}, "
                        f"R^2={fit['r2']:.3f}"
                    ),
                )
                if fit["x"].size > 1 and np.all(np.isfinite(fit["y_fit"])):
                    ax.plot(fit["x"], fit["y_fit"], "-", lw=1.5, color=color)
            ax.set_title(f"samples={n}")
            ax.set_xlabel(r"$\log(1/(1-x))$")
            ax.set_ylabel(r"$I_2 - I_3$")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            ax.text(
                0.03,
                0.97,
                rf"$L_A={L_A},\ L_C={L_C}$" + "\n" + rf"$l_B={res['lB'].tolist()}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85),
            )
        fig.suptitle(mi_title, fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    t0 = time.time()
    meta = parse_run_metadata(DATA_PATH)
    print(f"[info] input cache   = {DATA_PATH}")
    print(f"[info] output dir    = {FIG_DIR}")
    print(f"[info] cpu cap       = {CPU_CAP}")
    print(f"[info] y0 batch      = {Y0_BATCH}")
    print(f"[info] sample batch  = {SAMPLE_BATCH}")
    print(f"[info] sample counts = {SAMPLE_COUNTS}")

    max_samples = max(SAMPLE_COUNTS)
    g_final = load_final_batch(DATA_PATH, max_samples=max_samples)
    n_samples, nlayer, nlayer2 = g_final.shape
    if nlayer != nlayer2:
        raise ValueError(f"final states must be square; got {g_final.shape}")

    model = classA_U1FGTN(meta["Nx"], meta["Ny"], nshell=None, DW=True, alpha_1=ALPHA_1, alpha_2=ALPHA_2)
    nx = model.Nx
    ny = model.Ny
    expected_n = 2 * nx * ny
    if nlayer != expected_n:
        raise ValueError(f"expected Nlayer={expected_n} from model, got {nlayer}")

    x_windows = normalize_x_sum_windows(nx, x_sum_windows, model.DW_loc)
    ay_values = np.arange(AY_MIN, (ny + 1) - AY_MIN, dtype=int)
    lb_values = np.arange(L_A + L_C, ny - 2 * (L_A + L_C) + 1, dtype=int)

    contour_cache = build_contour_subsystem_cache(nx, ny, ay_values)
    mi_cache = build_mi_subsystem_cache(nx, ny, lb_values)

    print(f"[info] available samples = {n_samples}")
    print(f"[info] contour Ay scan   = {int(ay_values[0])}..{int(ay_values[-1])}")
    print(f"[info] MI lB scan        = {lb_values.tolist()}")
    print(f"[info] x-sum windows     = {[window['label'] for window in x_windows]}")
    print("[info] contour averaging = sample-first within each y0, then y0-average")
    print("[info] MI averaging      = sample-first within each y0, then y0-average")

    contour_means = contour_entropy_sample_first_mean(
        g_final,
        ay_values,
        contour_cache,
        x_windows,
        nx,
        ny,
        Y0_BATCH,
        SAMPLE_COUNTS,
        SAMPLE_BATCH,
    )
    contour_results = build_contour_results(contour_means, ay_values, ny, x_windows, SAMPLE_COUNTS)

    mi_means = mi_i2_minus_i3_sample_first_mean(
        g_final,
        lb_values,
        mi_cache,
        x_windows,
        nx,
        ny,
        SAMPLE_COUNTS,
        SAMPLE_BATCH,
    )
    mi_results = build_mi_results(mi_means, lb_values, ny, x_windows, SAMPLE_COUNTS)

    pdf_path = FIG_DIR / (DATA_PATH.stem + "_xsum_window_integrated_contour_and_mi_fits_v2.pdf")
    save_pdf(pdf_path, contour_results, mi_results, x_windows, meta)

    print("[info] saved pdf:")
    print(f"  {pdf_path}")
    print("[info] contour fit summary")
    for n in [int(v) for v in SAMPLE_COUNTS if int(v) in contour_results]:
        print(f"  samples={n}")
        for window in x_windows:
            for branch_name, fit in contour_results[n][window['key']]["fits"].items():
                print(
                    f"    - {window['label']}, {branch_name}: slope={fit['m']:.8e}, "
                    f"slope_err={fit['m_err']:.8e}, intercept={fit['b']:.8e}, R^2={fit['r2']:.8f}"
                )
    print("[info] MI fit summary")
    for n in [int(v) for v in SAMPLE_COUNTS if int(v) in mi_results]:
        print(f"  samples={n}")
        for window in x_windows:
            fit = mi_results[n][window["key"]]["fit"]
            print(
                f"    - {window['label']}: slope={fit['m']:.8e}, slope_err={fit['m_err']:.8e}, "
                f"intercept={fit['b']:.8e}, R^2={fit['r2']:.8f}"
            )
    print(f"[info] elapsed = {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
