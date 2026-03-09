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

#cpu_cap = 50
#cpu_count = mp.cpu_count()
cpu_cap = 40
os.environ["MY_CPU_COUNT"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_MAX_THREADS"] = str(1)
try:
    os.sched_setaffinity(0, set(range(50, 50 + cpu_cap)))
except Exception as exc:
    print(f"CPU affinity not set: {exc}")
print(os.sched_getaffinity(0))

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm




import fgtn.classA_U1FGTN as classA_U1FGTN
import importlib

importlib.reload(classA_U1FGTN)
from fgtn.classA_U1FGTN import classA_U1FGTN
def product_state_one_per_cell_top_layer(Nx, Ny):
    """
    Full-layer covariance with one occupied top-layer mode per unit cell.
    Indexing matches i = mu + 2*x + 2*Nx*y for the top layer, and the bottom
    layer is offset by 2*Nx*Ny.
    """
    Nlayer = 2 * Nx * Ny
    G = -np.eye(Nlayer, dtype=np.complex128)  # start all empty (-1)

    # Occupy the mu=2 top-layer mode in each cell: set diag to +1
    for y in range(Ny):
        for x in range(Nx):
            idx_top_mu0 = 1 + 2 * x + 2 * Nx * y
            G[idx_top_mu0, idx_top_mu0] = 1.0

    return G


def build_sub_indices(Nx, Ny, y_cut):
    yA = np.arange(y_cut, Ny)
    sub_indices = []
    for y in yA:
        base = 2 * Nx * y
        for x in range(Nx):
            sub_indices.append(base + 2 * x)
            sub_indices.append(base + 2 * x + 1)
    return np.array(sub_indices, dtype=int), len(yA), yA


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
    return np.array(sub_indices, dtype=int), len(yA), yA


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


def s_map_avg_for_range(G_hist, Nx, Ny, y_start, y_end, t_idx):
    sub_indices, Ny_sub, _ = build_sub_indices_range(Nx, Ny, y_start, y_end)
    G_sub_batch = np.stack(
        [G_hist[s, t_idx][np.ix_(sub_indices, sub_indices)] for s in range(G_hist.shape[0])],
        axis=0
    )
    s_batch = entanglement_contour_batch(G_sub_batch, Nx, Ny_sub)
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


def sample_avg_entanglement_contour_batched(G_hist, Nx, Ny, y_cut=None):
    if y_cut is None:
        y_cut = Ny // 2
    sub_indices, Ny_sub, yA = build_sub_indices(Nx, Ny, y_cut)

    G_sub_batch = np.stack(
        [G_hist[s, -1][np.ix_(sub_indices, sub_indices)] for s in range(G_hist.shape[0])],
        axis=0
    )
    s_batch = entanglement_contour_batch(G_sub_batch, Nx, Ny_sub)
    s_map_avg = s_batch.mean(axis=0)
    return s_map_avg, yA


def local_chern_marker_flat_batch(G_batch, model, batch_size=8):
    G_batch = np.asarray(G_batch)
    S = G_batch.shape[0]
    out = np.zeros((S, model.Nx, model.Ny), dtype=float)
    for i in range(0, S, batch_size):
        j = min(i + batch_size, S)
        for k in range(i, j):
            out[k] = model.local_chern_marker_flat(G_batch[k])
    return out


def avg_contour_over_samples_batched(G_hist, y_cut_list, x_list, Nx, Ny):
    Sx_vs_y_cut_sum = np.zeros((len(x_list), len(y_cut_list)), dtype=float)
    sub_cache = [build_sub_indices(Nx, Ny, y_cut) for y_cut in y_cut_list]

    for i, (sub_idx, Ny_sub, _) in enumerate(tqdm(sub_cache, desc="Integrated contour cuts")):
        G_sub_batch = np.stack(
            [G_hist[s, -1][np.ix_(sub_idx, sub_idx)] for s in range(G_hist.shape[0])],
            axis=0
        )
        s_batch = entanglement_contour_batch(G_sub_batch, Nx, Ny_sub)
        Sx_batch = np.sum(s_batch, axis=2)
        Sx_avg = Sx_batch.mean(axis=0)
        Sx_vs_y_cut_sum[:, i] = Sx_avg[:len(x_list)]

    return {x: Sx_vs_y_cut_sum[j] for j, x in enumerate(x_list)}

def main():
    t0 = time.time()

    # Lattice + DW parameters and circuit depth
    Nx, Ny = 12, 31
    alpha_1, alpha_2 = 30, 1
    cycles = 20  # adjust as needed
    # time indices in G_hist (0 = initial if save_init=True)
    time_steps_to_fit = np.arange(5, 21)
    only_summary_page = True

    G_init = product_state_one_per_cell_top_layer(Nx, Ny)

    # Build model and run adaptive circuit; keep history in memory, avoid cache saves
    model = classA_U1FGTN(Nx, Ny, nshell=None, DW=True, alpha_1=alpha_1, alpha_2=alpha_2)
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc not set; need DW=True with valid alpha profile")
    x0 = int(model.DW_loc[0]) % Nx
    x1 = int(model.DW_loc[1]) % Nx
    left_xs = [(x0 - 1) % Nx, x0 % Nx, (x0 + 1) % Nx]
    right_xs = [(x1 - 1) % Nx, x1 % Nx, (x1 + 1) % Nx]
    left_pair = [x0 % Nx, (x0 + 1) % Nx]
    right_pair = [(x1 - 1) % Nx, x1 % Nx]
    p_meas = 1
    res = model.run_markov_circuit(
            G_history=True,
            cycles=cycles,
            progress=True,
            G_init=G_init,
            save=True,  
            samples=250,
            p_meas=p_meas,
            n_jobs=cpu_cap,
            parallelize_samples=True,
            sequence="random",
            top_triv_back_forth=True,
            max_in_flight=max(1, 9*cpu_cap//10),
            save_suffix='_slope_drift_testing'
        )
    print(f"Completed p_meas={p_meas}")

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

    figs_dir = os.path.join("figs")
    os.makedirs(figs_dir, exist_ok=True)
    save_path = res.get("save_path")
    if save_path:
        cache_key = os.path.splitext(os.path.basename(save_path))[0]
        pdf_name = f"{cache_key}_plots.pdf"
    else:
        pdf_name = f"markov_p_sweep_plots_pm{p_meas:.2f}.pdf"
    pdf_path = os.path.join(figs_dir, pdf_name)

    with PdfPages(pdf_path) as pdf:
        # Page 1: trajectory-averaged step size + local Chern marker
        diffs_sum = np.zeros(T - 1, dtype=float)
        diffs_sq_sum = np.zeros(T - 1, dtype=float)
        for s in tqdm(range(S), desc="Trajectory step sizes"):
            for t in range(1, T):
                diff = G_hist[s, t] - G_hist[s, t - 1]
                val = np.linalg.norm(diff, ord="fro")
                diffs_sum[t - 1] += val
                diffs_sq_sum[t - 1] += val * val
        diffs_avg = diffs_sum / S
        diffs_std = np.sqrt(np.maximum(diffs_sq_sum / S - diffs_avg**2, 0.0))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax0 = axes[0]
        t_vals = np.arange(1, T)
        ax0.plot(t_vals, diffs_avg, lw=1.6, label="avg")
        ax0.fill_between(t_vals, diffs_avg - diffs_std, diffs_avg + diffs_std, color="0.8", alpha=0.6, label="std")
        ax0.set_xlabel("t")
        ax0.set_ylabel(r"$|G(t)-G(t-1)|_F$")
        ax0.set_yscale("log")
        ax0.grid(alpha=0.3)
        ax0.legend()
        ax0.set_title("Trajectory-avg step size")

        # local Chern marker heatmap (same page)
        G_ss_batch = np.stack([G_hist[s, -1] for s in range(S)], axis=0)
        chmap_batch = local_chern_marker_flat_batch(G_ss_batch, model, batch_size=8)
        chmap_avg = chmap_batch.mean(axis=0)

        ax1 = axes[1]
        im = ax1.imshow(chmap_avg, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax1, label="Avg tanh(Chern marker)")
        ax1.set_xlabel("y")
        ax1.set_ylabel("x")
        ax1.set_title("Local Chern marker")

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # characterization to match plot_integrated_contour_by_Ay.py
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

        tbar = tqdm(time_steps_to_fit, desc="Time steps", unit="t")
        for t_idx in tbar:
            if not (0 <= t_idx < T):
                raise ValueError(f"time step {t_idx} out of range for T={T}")
            tbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)

            if not only_summary_page:
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

                Ay_heat = Ny // 2
                s_map_heat = s_map_avg_for_range(G_hist, Nx, Ny, 0, Ay_heat, t_idx)
                im = plot_heatmap_and_cuts(ax_heat, ax_cuts, s_map_heat, Nx, Ay_heat, title_suffix=f"(t={t_idx})")
                fig.colorbar(im, ax=ax_heat, label="Avg entanglement contour s(r)")

            # cache s_map for each y_cut once per time-step
            all_y_cuts = np.concatenate([y_cut_list_1, y_cut_list_2])
            s_map_cache = {}
            ycut_bar = tqdm(all_y_cuts, desc=f"t={t_idx} | y_cut cache", unit="y_cut")
            for y_cut in ycut_bar:
                ycut_bar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
                s_map_cache[int(y_cut)] = s_map_avg_for_range(G_hist, Nx, Ny, int(y_cut), Ny, t_idx)

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

        # removed separate Chern marker page (now on page 1)

    elapsed = time.time() - t0
    print(f"Saved plots to {pdf_path}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
