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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

from fgtn.classA_U1FGTN import classA_U1FGTN
from joblib import Parallel, delayed

# ---- CPU affinity / thread caps ----
# cpu_use controls process count and affinity; BLAS threads are pinned to 1.
cpu_use = 30
os.environ["MY_CPU_COUNT"] = str(cpu_use)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
try:
    os.sched_setaffinity(0, set(range(60, 60 + cpu_use)))
except Exception as exc:
    print(f"CPU affinity not set: {exc}")
try:
    print(os.sched_getaffinity(0))
except Exception:
    pass

# ---- Config ----
data_path = (
    "cache/G_history_samples/"
    "N12x31/"
    "N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_slope_drift_testing.npz"
)

Nx, Ny = 12, 31
alpha_1, alpha_2 = 30, 1

# time indices in G_hist (0 = initial if saved with init)
time_steps_to_fit = np.arange(5, 21)  # [5, 7, 9, 11, 13, 15, 17, 19]

# subsystem sizes |A_y| to consider (Ay in [4, Ny-4])
Ay_min = 4
Ay_max = Ny - 4

# optionally subsample samples for speed (None = use all trajectories)
max_samples = None

# parallelize contour calculation over samples
parallel_samples = True
n_jobs = cpu_use

# output
pdf_path = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_slope_drift_testing_plots_summary_only.pdf"
)

# if True, only write the final slope-vs-time page (skip per-time-step pages)
only_summary_page = True

# -----------------


def build_sub_indices_range(Nx, Ny, y_start, y_end):
    if y_start < 0 or y_end > Ny or y_start >= y_end:
        raise ValueError(f"invalid y range [{y_start}, {y_end}) for Ny={Ny}")
    yA = np.arange(y_start, y_end)
    sub_indices = []
    for y in yA:
        base = 2 * Nx * y
        for x in range(Nx):
            sub_indices.append(base + 2 * x)
            sub_indices.append(base + 2 * x + 1)
    return np.array(sub_indices, dtype=int)


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


def load_G_hist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as data:
        G_hist = data["G_hist"]
    return G_hist


def sample_indices(S, max_samples):
    if max_samples is None or max_samples >= S:
        return np.arange(S, dtype=int)
    # evenly spaced deterministic subsample
    return np.linspace(0, S - 1, max_samples, dtype=int)


def _entanglement_contour_single(G_sub, Nx, Ny_sub):
    G_sub = np.asarray(G_sub, dtype=np.complex128)
    I = np.eye(G_sub.shape[0], dtype=np.complex128)
    G2 = 0.5 * (I + G_sub)
    evals, vecs = np.linalg.eigh(G2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
    f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    diagF = np.einsum("ik,k,ik->i", vecs, f_eigs, vecs.conj(), optimize=True).real
    diagF = diagF.reshape(2, Nx, Ny_sub, order="F")
    return diagF.sum(axis=0)


def s_map_avg_for_range(G_hist, y_start, y_end, t_idx, sample_idx, Nx, Ny):
    sub_idx = build_sub_indices_range(Nx, Ny, y_start, y_end)
    Ny_sub = y_end - y_start

    def _one_sample(s):
        G_sub = G_hist[s, t_idx][np.ix_(sub_idx, sub_idx)]
        return _entanglement_contour_single(G_sub, Nx, Ny_sub)

    if parallel_samples and len(sample_idx) > 1:
        s_list = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_sample)(s) for s in sample_idx
        )
    else:
        s_list = [_one_sample(s) for s in sample_idx]

    s_batch = np.stack(s_list, axis=0)
    return s_batch.mean(axis=0)


def fit_line_log_chord(Ay_sizes, S_vals, Ny):
    x = np.log(np.sin(np.pi * Ay_sizes / Ny))
    mask = np.isfinite(x) & np.isfinite(S_vals)
    x = x[mask]
    y = S_vals[mask]
    if x.size < 2:
        return np.nan, np.nan, np.nan, np.nan, x, y
    coeffs = np.polyfit(x, y, 1)
    m, b = float(coeffs[0]), float(coeffs[1])
    y_hat = m * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot
    n = x.size
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    if n > 2 and sxx > 0:
        sigma2 = ss_res / (n - 2)
        m_err = float(np.sqrt(sigma2 / sxx))
    else:
        m_err = np.nan
    return m, b, r2, m_err, x, y


def plot_heatmap_and_cuts(ax_heat, ax_cuts, s_map, Nx, Ny_sub, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.2]})
    plt.close(fig)

    im = ax_heat.imshow(s_map, origin="lower", cmap="magma", aspect="auto")
    ax_heat.set_xlabel("y (subsystem)")
    ax_heat.set_ylabel("x")
    ax_heat.set_title(f"Entanglement contour {title_suffix}")

    y_vals = np.arange(Ny_sub)
    x_list = np.arange(Nx)
    colors_x = plt.cm.viridis(np.linspace(0, 1, len(x_list)))
    markers = ["o", "^"]
    for x_idx, color in zip(x_list, colors_x):
        marker = markers[0] if x_idx < Nx // 2 else markers[1]
        ax_cuts.plot(y_vals, s_map[x_idx, :], marker=marker, color=color, lw=1.2)
    ax_cuts.set_xlabel("y (subsystem)")
    ax_cuts.set_ylabel("s(x, y)")
    ax_cuts.set_title("Average cuts vs y for fixed x")
    ax_cuts.grid(alpha=0.3, linestyle="--", linewidth=0.8)

    return im


def main():
    t0 = time.time()
    G_hist = load_G_hist(data_path)
    S, T, N, N2 = G_hist.shape
    if N != N2:
        raise ValueError(f"G_hist last dims must be square; got {N}x{N2}")
    if 2 * Nx * Ny != N:
        raise ValueError(f"Nx,Ny mismatch: 2*Nx*Ny={2*Nx*Ny} != Nlayer={N}")

    model = classA_U1FGTN(Nx, Ny, nshell=None, DW=True, alpha_1=alpha_1, alpha_2=alpha_2)
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc not set; need DW=True with valid alpha profile")

    x0 = int(model.DW_loc[0]) % Nx
    x1 = int(model.DW_loc[1]) % Nx

    left_xs = [x0 - 1, x0, x0 + 1]
    right_xs = [x1 - 1, x1, x1 + 1]
    left_xs = [x % Nx for x in left_xs]
    right_xs = [x % Nx for x in right_xs]

    # for fits (pairs as requested)
    left_pair = [x0 % Nx, (x0 + 1) % Nx]
    right_pair = [(x1 - 1) % Nx, x1 % Nx]

    Ay_list = np.arange(Ay_min, Ay_max + 1, dtype=int)
    sample_idx = sample_indices(S, max_samples)

    # Two y-cut ranges (as in run_markov_p_sweep)
    y_cut_list_1 = np.arange(2, Ny // 2)
    y_cut_list_2 = np.arange(Ny // 2, Ny - 1)
    n = min(len(y_cut_list_1), len(y_cut_list_2))
    y_cut_list_1 = y_cut_list_1[:n]
    y_cut_list_2 = y_cut_list_2[:n]
    Ay_list_1 = Ny - y_cut_list_1
    Ay_list_2 = Ny - y_cut_list_2

    slopes = {
        "Left DW, y_cut_list_1": [],
        "Left DW, y_cut_list_2": [],
        "Right DW, y_cut_list_1": [],
        "Right DW, y_cut_list_2": [],
    }
    r2_vals = {
        "Left DW, y_cut_list_1": [],
        "Left DW, y_cut_list_2": [],
        "Right DW, y_cut_list_1": [],
        "Right DW, y_cut_list_2": [],
    }
    slope_errs = {
        "Left DW, y_cut_list_1": [],
        "Left DW, y_cut_list_2": [],
        "Right DW, y_cut_list_1": [],
        "Right DW, y_cut_list_2": [],
    }

    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        tbar = tqdm(time_steps_to_fit, desc="Time steps", unit="t")
        for t_idx in tbar:
            if not (0 <= t_idx < T):
                raise ValueError(f"time step {t_idx} out of range for T={T}")
            tbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)

            if not only_summary_page:
                # one page per time-step: heatmap/cuts + DW curves + DW fits
                fig = plt.figure(figsize=(14, 16))
                gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.3)
                ax_heat = fig.add_subplot(gs[0, 0])
                ax_cuts = fig.add_subplot(gs[0, 1])
                ax_left_1 = fig.add_subplot(gs[1, 0])
                ax_right_1 = fig.add_subplot(gs[1, 1])
                ax_left_2 = fig.add_subplot(gs[2, 0])
                ax_right_2 = fig.add_subplot(gs[2, 1])
                ax_left_fit_1 = fig.add_subplot(gs[3, 0])
                ax_right_fit_1 = fig.add_subplot(gs[3, 1])
                ax_left_fit_2 = fig.add_subplot(gs[4, 0])
                ax_right_fit_2 = fig.add_subplot(gs[4, 1])

                # heatmap + cuts for Ay = Ny//2 (subsystem y in [0, Ny//2))
                Ay_heat = Ny // 2
                s_map_heat = s_map_avg_for_range(G_hist, 0, Ay_heat, t_idx, sample_idx, Nx, Ny)
                im = plot_heatmap_and_cuts(ax_heat, ax_cuts, s_map_heat, Nx, Ay_heat, title_suffix=f"(t={t_idx})")
                fig.colorbar(im, ax=ax_heat, label="Avg entanglement contour s(r)")

            # cache s_map for each y_cut once per time-step
            all_y_cuts = np.concatenate([y_cut_list_1, y_cut_list_2])
            s_map_cache = {}
            ycut_bar = tqdm(all_y_cuts, desc=f"t={t_idx} | y_cut cache", unit="y_cut")
            for y_cut in ycut_bar:
                ycut_bar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
                if int(y_cut) in s_map_cache:
                    continue
                s_map_cache[int(y_cut)] = s_map_avg_for_range(
                    G_hist, int(y_cut), Ny, t_idx, sample_idx, Nx, Ny
                )

            def per_x_and_pair_for_ycuts(y_cut_list, x_set, pair_set, include_per_x=True):
                per_x = {x: [] for x in x_set} if include_per_x else None
                pair_sum = []
                for y_cut in y_cut_list:
                    s_map = s_map_cache[int(y_cut)]
                    if include_per_x:
                        for x in x_set:
                            per_x[x].append(float(np.sum(s_map[x, :])))
                    pair_sum.append(sum(float(np.sum(s_map[x, :])) for x in pair_set))
                return per_x, np.asarray(pair_sum, dtype=float)

            # Left DW curves (x0-1,x0,x0+1), fits use x0+x0+1
            left_per_x_1, left_pair_1 = per_x_and_pair_for_ycuts(
                y_cut_list_1, left_xs, left_pair, include_per_x=(not only_summary_page)
            )
            left_per_x_2, left_pair_2 = per_x_and_pair_for_ycuts(
                y_cut_list_2, left_xs, left_pair, include_per_x=(not only_summary_page)
            )

            if not only_summary_page:
                for x, vals in left_per_x_1.items():
                    ax_left_1.plot(Ay_list_1, vals, marker="o", lw=1.2, label=f"x={x}")
                ax_left_1.set_title(f"Left DW, y_cut_list_1 (t={t_idx})")
                ax_left_1.set_xlabel(r"$|A_y|$")
                ax_left_1.set_ylabel(r"$\sum_{y\in A} s_A(x,y)$")
                ax_left_1.grid(alpha=0.3)
                ax_left_1.legend(fontsize=8, ncol=3)

                for x, vals in left_per_x_2.items():
                    ax_left_2.plot(Ay_list_2, vals, marker="o", lw=1.2, label=f"x={x}")
                ax_left_2.set_title(f"Left DW, y_cut_list_2 (t={t_idx})")
                ax_left_2.set_xlabel(r"$|A_y|$")
                ax_left_2.set_ylabel(r"$\sum_{y\in A} s_A(x,y)$")
                ax_left_2.grid(alpha=0.3)
                ax_left_2.legend(fontsize=8, ncol=3)

            # Right DW curves (x1-1,x1,x1+1), fits use x1-1+x1
            right_per_x_1, right_pair_1 = per_x_and_pair_for_ycuts(
                y_cut_list_1, right_xs, right_pair, include_per_x=(not only_summary_page)
            )
            right_per_x_2, right_pair_2 = per_x_and_pair_for_ycuts(
                y_cut_list_2, right_xs, right_pair, include_per_x=(not only_summary_page)
            )

            if not only_summary_page:
                for x, vals in right_per_x_1.items():
                    ax_right_1.plot(Ay_list_1, vals, marker="o", lw=1.2, label=f"x={x}")
                ax_right_1.set_title(f"Right DW, y_cut_list_1 (t={t_idx})")
                ax_right_1.set_xlabel(r"$|A_y|$")
                ax_right_1.set_ylabel(r"$\sum_{y\in A} s_A(x,y)$")
                ax_right_1.grid(alpha=0.3)
                ax_right_1.legend(fontsize=8, ncol=3)

                for x, vals in right_per_x_2.items():
                    ax_right_2.plot(Ay_list_2, vals, marker="o", lw=1.2, label=f"x={x}")
                ax_right_2.set_title(f"Right DW, y_cut_list_2 (t={t_idx})")
                ax_right_2.set_xlabel(r"$|A_y|$")
                ax_right_2.set_ylabel(r"$\sum_{y\in A} s_A(x,y)$")
                ax_right_2.grid(alpha=0.3)
                ax_right_2.legend(fontsize=8, ncol=3)

            def _fit_panel(ax, title, Ay_list, pair_vals):
                Ay_arr = np.asarray(Ay_list, dtype=float)
                S_vals = np.asarray(pair_vals, dtype=float)
                m, b, r2, m_err, x_fit, y_fit = fit_line_log_chord(Ay_arr, S_vals, Ny)
                ax.plot(x_fit, y_fit, "o", ms=4, label="data")
                if np.isfinite(m):
                    if np.isfinite(m_err):
                        label = f"m={m:.4f}±{m_err:.4f}, R²={r2:.3f}"
                    else:
                        label = f"m={m:.4f}, R²={r2:.3f}"
                    ax.plot(x_fit, m * x_fit + b, "-", lw=1.6, label=label)
                ax.set_title(f"{title} (t={t_idx})")
                ax.set_xlabel(r"$\log(\sin(\pi |A_y| / N_y))$")
                ax.set_ylabel(r"$\sum_{x\in X}\sum_{y\in A} s_A(x,y)$")
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
                return m, m_err, r2

            if only_summary_page:
                m_l1, _, r2_l1, e_l1, _, _ = fit_line_log_chord(
                    np.asarray(Ay_list_1, dtype=float), np.asarray(left_pair_1, dtype=float), Ny
                )
                m_r1, _, r2_r1, e_r1, _, _ = fit_line_log_chord(
                    np.asarray(Ay_list_1, dtype=float), np.asarray(right_pair_1, dtype=float), Ny
                )
                m_l2, _, r2_l2, e_l2, _, _ = fit_line_log_chord(
                    np.asarray(Ay_list_2, dtype=float), np.asarray(left_pair_2, dtype=float), Ny
                )
                m_r2, _, r2_r2, e_r2, _, _ = fit_line_log_chord(
                    np.asarray(Ay_list_2, dtype=float), np.asarray(right_pair_2, dtype=float), Ny
                )
            else:
                m_l1, e_l1, r2_l1 = _fit_panel(ax_left_fit_1, "Left DW fit, y_cut_list_1", Ay_list_1, left_pair_1)
                m_r1, e_r1, r2_r1 = _fit_panel(ax_right_fit_1, "Right DW fit, y_cut_list_1", Ay_list_1, right_pair_1)
                m_l2, e_l2, r2_l2 = _fit_panel(ax_left_fit_2, "Left DW fit, y_cut_list_2", Ay_list_2, left_pair_2)
                m_r2, e_r2, r2_r2 = _fit_panel(ax_right_fit_2, "Right DW fit, y_cut_list_2", Ay_list_2, right_pair_2)

            slopes["Left DW, y_cut_list_1"].append(m_l1)
            slopes["Right DW, y_cut_list_1"].append(m_r1)
            slopes["Left DW, y_cut_list_2"].append(m_l2)
            slopes["Right DW, y_cut_list_2"].append(m_r2)
            r2_vals["Left DW, y_cut_list_1"].append(r2_l1)
            r2_vals["Right DW, y_cut_list_1"].append(r2_r1)
            r2_vals["Left DW, y_cut_list_2"].append(r2_l2)
            r2_vals["Right DW, y_cut_list_2"].append(r2_r2)
            slope_errs["Left DW, y_cut_list_1"].append(e_l1)
            slope_errs["Right DW, y_cut_list_1"].append(e_r1)
            slope_errs["Left DW, y_cut_list_2"].append(e_l2)
            slope_errs["Right DW, y_cut_list_2"].append(e_r2)

            if not only_summary_page:
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

        # final summary page(s)
        if only_summary_page:
            fig, (ax_slope, ax_r2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
            for title, m_list in slopes.items():
                err_list = slope_errs[title]
                ax_slope.errorbar(
                    time_steps_to_fit,
                    m_list,
                    yerr=err_list,
                    marker="o",
                    lw=1.6,
                    capsize=3,
                    label=title,
                )
                ax_r2.plot(time_steps_to_fit, r2_vals[title], marker="o", lw=1.6, label=title)
            ax_slope.set_ylabel("Fit slope m")
            ax_slope.set_title("Slope vs time-step (log-chord fits)")
            ax_slope.grid(alpha=0.3)
            ax_slope.legend(ncol=2, fontsize=9)
            ax_r2.set_xlabel("t")
            ax_r2.set_ylabel("Fit R^2")
            ax_r2.set_title("R^2 vs time-step (log-chord fits)")
            ax_r2.grid(alpha=0.3)
            ax_r2.legend(ncol=2, fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
            for title, m_list in slopes.items():
                err_list = slope_errs[title]
                ax.errorbar(
                    time_steps_to_fit,
                    m_list,
                    yerr=err_list,
                    marker="o",
                    lw=1.6,
                    capsize=3,
                    label=title,
                )
            ax.set_xlabel("t")
            ax.set_ylabel("Fit slope m")
            ax.set_title("Slope vs time-step (log-chord fits)")
            ax.grid(alpha=0.3)
            ax.legend(ncol=2, fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    elapsed = time.time() - t0
    print(f"Saved {pdf_path}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
