import os
import multiprocessing as mp

cpu_cap = 52
cpu_count = mp.cpu_count()
cpu_use = min(cpu_cap, cpu_count)
os.environ["MY_CPU_COUNT"] = str(cpu_use)
os.environ["OMP_NUM_THREADS"] = str(cpu_use)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_use)
os.environ["MKL_NUM_THREADS"] = str(cpu_use)
os.environ["NUMEXPR_MAX_THREADS"] = str(cpu_use)
try:
    os.sched_setaffinity(0, set(range(cpu_use)))
except Exception as exc:
    print(f"CPU affinity not set: {exc}")
print(os.sched_getaffinity(0))

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm



    
import classA_U1FGTN
import importlib

importlib.reload(classA_U1FGTN)
from classA_U1FGTN import classA_U1FGTN


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


def plot_current_maps_axes(fig, axes, J_x, J_y, vmin=None, vmax=None):
    Nx, Ny = J_x.shape
    extent = (-0.5, Ny - 0.5, -0.5, Nx - 0.5)
    cmap = "RdBu_r"
    quiver_cmap = "viridis"

    for ax, (title, data) in zip(axes[:2], [("J_x", J_x), ("J_y", J_y)]):
        im = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax, shrink=0.85)

    y_coords, x_coords = np.meshgrid(np.arange(Ny), np.arange(Nx))
    magnitude = np.hypot(J_x, J_y)
    quiv = axes[2].quiver(
        y_coords,
        x_coords,
        J_y,
        J_x,
        magnitude,
        cmap=quiver_cmap,
        angles="xy",
        scale_units="xy",
        scale=None,
        pivot="mid",
    )
    axes[2].set_xlim(extent[0], extent[1])
    axes[2].set_ylim(extent[2], extent[3])
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("x")
    axes[2].set_title("Current field (Jx, Jy)")
    axes[2].set_aspect("equal")
    fig.colorbar(quiv, ax=axes[2], shrink=0.85, label="|J|")


def main():

    # Lattice + DW parameters and circuit depth
    Nx, Ny = 12, 31
    alpha_1, alpha_2 = 30, 1
    cycles = 100  # adjust as needed

    G_init = product_state_one_per_cell_top_layer(Nx, Ny)

    # Build model and run adaptive circuit; keep history in memory, avoid cache saves
    model = classA_U1FGTN(Nx, Ny, nshell=None, DW=True, alpha_1=alpha_1, alpha_2=alpha_2)
    p_meas = 1
    res = model.run_markov_circuit(
            G_history=True,
            cycles=cycles,
            progress=True,
            G_init=G_init,
            save=True,  
            samples=500,
            p_meas=p_meas,
            n_jobs=cpu_cap,
            parallelize_samples=True,
            sequence="dw_symmetric_random",
            top_triv_back_forth=True,
            top_triv_block_cycles=1,
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
        # Page 1: sample-averaged entanglement contour heatmap + x cuts
        s_map_avg, yA = sample_avg_entanglement_contour_batched(G_hist, Nx, Ny, y_cut=Ny // 2)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.2]})
        im = axes[0].imshow(s_map_avg, origin="lower", cmap="magma", aspect="auto")
        fig.colorbar(im, ax=axes[0], label="Avg entanglement contour s(r)")
        axes[0].set_xlabel("y (subsystem)")
        axes[0].set_ylabel("x")

        y_vals = np.arange(len(yA))
        x_list = np.arange(Nx)
        colors_x = plt.cm.viridis(np.linspace(0, 1, len(x_list)))
        markers = ["o", "^"]
        for x_idx, color in zip(x_list, colors_x):
            marker = markers[0] if x_idx < Nx // 2 else markers[1]
            axes[1].plot(y_vals, s_map_avg[x_idx, :], marker=marker, color=color, lw=1.8)
        axes[1].set_xlabel("y (subsystem)")
        axes[1].set_ylabel("s(x, y)")
        axes[1].set_title("Average cuts vs y for fixed x")
        axes[1].grid(alpha=0.3, linestyle="--", linewidth=0.8)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=x_list.min(), vmax=x_list.max()))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[1], label="x")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: trajectory-averaged step size and integrated contour trends
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

        x_list = np.arange(Nx // 2)
        y_cut_list_1 = np.arange(2, Ny // 2)
        y_cut_list_2 = np.arange(Ny // 2, Ny - 1)
        n = min(len(y_cut_list_1), len(y_cut_list_2))
        y_cut_list_1 = y_cut_list_1[:n]
        y_cut_list_2 = y_cut_list_2[:n]
        A_y_list = Ny - y_cut_list_1

        Sx_vs_y_cut_avg_1 = avg_contour_over_samples_batched(G_hist, y_cut_list_1, x_list, Nx, Ny)
        Sx_vs_y_cut_avg_2 = avg_contour_over_samples_batched(G_hist, y_cut_list_2, x_list, Nx, Ny)

        fig = plt.figure(figsize=(12, 11))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        t_vals = np.arange(1, T)
        ax0.plot(t_vals, diffs_avg, lw=1.6, label="avg")
        ax0.fill_between(t_vals, diffs_avg - diffs_std, diffs_avg + diffs_std, color="0.8", alpha=0.6, label="std")
        ax0.set_xlabel("t")
        ax0.set_ylabel(r"$|G(t)-G(t-1)|_F$")
        ax0.set_yscale("log")
        ax0.grid(alpha=0.3)
        ax0.legend()
        ax0.set_title("Trajectory-avg step size")

        for x in x_list:
            ax1.plot(A_y_list, Sx_vs_y_cut_avg_1[x], marker="o", lw=1.4, label=f"x={x}")
        ax1.set_xlabel(r"Subsystem size $|A_y|$")
        ax1.set_ylabel(r"Avg integrated contour $\sum_{y\in A_y} s_{A_y}(x,y)$")
        ax1.set_title("Avg integrated contour vs subsystem size")
        ax1.grid(alpha=0.3)
        ax1.legend(ncol=3, fontsize=8)

        x0 = model.DW_loc[0]
        x_sum_range = np.arange(x0, x0 + 2)

        def sum_over_x_for_list(y_cut_list, Sx_vs_y_cut_avg):
            sums = []
            for i in range(len(y_cut_list)):
                total = 0.0
                for x in x_sum_range:
                    total += Sx_vs_y_cut_avg[x][i]
                sums.append(total)
            return np.array(sums, dtype=float)

        Ay1 = Ny - y_cut_list_1
        Ay2 = Ny - y_cut_list_2
        yvals1 = sum_over_x_for_list(y_cut_list_1, Sx_vs_y_cut_avg_1)
        yvals2 = sum_over_x_for_list(y_cut_list_2, Sx_vs_y_cut_avg_2)

        xvals1 = np.log(np.sin(np.pi * Ay1 / Ny))
        mask1 = np.isfinite(xvals1)
        x_fit1 = xvals1[mask1]
        y_fit1 = yvals1[mask1]
        m1, b1 = np.polyfit(x_fit1, y_fit1, 1)
        y_pred1 = m1 * x_fit1 + b1

        xvals2 = np.log(np.sin(np.pi * Ay2 / Ny))
        mask2 = np.isfinite(xvals2)
        x_fit2 = xvals2[mask2]
        y_fit2 = yvals2[mask2]
        m2, b2 = np.polyfit(x_fit2, y_fit2, 1)
        y_pred2 = m2 * x_fit2 + b2

        ax2.plot(Ay2, yvals2, marker="o", lw=1.6)
        ax2.set_xlabel(r"Subsystem size $|A_y|$")
        ax2.set_ylabel(r"$\sum_{x}\sum_{y\in A_y} s_{A_y}(x,y)$")
        ax2.set_title("Avg integrated contour vs |A_y| (Ay2)")
        ax2.grid(alpha=0.3)

        ax3.plot(x_fit2, y_fit2, marker="o", lw=1.4, label="data")
        ax3.plot(x_fit2, y_pred2, lw=2, label=f"fit: m={m2:.4f}, b={b2:.4f}")
        ax3.set_xlabel(r"$\log(\sin(\pi |A_y|/N_y))$")
        ax3.set_ylabel(r"$\sum_{x}\sum_{y\in A_y} s_{A_y}(x,y)$")
        ax3.set_title("Avg fit vs log(sin) (Ay2)")
        ax3.grid(alpha=0.3)
        ax3.legend()

        ax4.plot(Ay1, yvals1, marker="o", lw=1.6)
        ax4.set_xlabel(r"Subsystem size $|A_y|$")
        ax4.set_ylabel(r"$\sum_{x}\sum_{y\in A_y} s_{A_y}(x,y)$")
        ax4.set_title("Avg integrated contour vs |A_y| (Ay1)")
        ax4.grid(alpha=0.3)

        ax5.plot(x_fit1, y_fit1, marker="o", lw=1.4, label="data")
        ax5.plot(x_fit1, y_pred1, lw=2, label=f"fit: m={m1:.4f}, b={b1:.4f}")
        ax5.set_xlabel(r"$\log(\sin(\pi |A_y|/N_y))$")
        ax5.set_ylabel(r"$\sum_{x}\sum_{y\in A_y} s_{A_y}(x,y)$")
        ax5.set_title("Avg fit vs log(sin) (Ay1)")
        ax5.grid(alpha=0.3)
        ax5.legend()

        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: sample-averaged local Chern marker heatmap + x cuts
        G_ss_batch = np.stack([G_hist[s, -1] for s in range(S)], axis=0)
        chmap_batch = local_chern_marker_flat_batch(G_ss_batch, model, batch_size=8)
        chmap_avg = chmap_batch.mean(axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        im = axes[0].imshow(chmap_avg, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
        fig.colorbar(im, ax=axes[0], label="Avg tanh(Chern marker)")
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("x")

        for x in range(min(6, Nx)):
            axes[1].plot(chmap_avg[x, :], marker="o", label=f"x={x}")
        axes[1].legend()
        axes[1].set_xlabel("y")
        axes[1].set_ylabel("Avg tanh(Chern marker)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Page 4: sample-averaged current maps + y-integrated currents vs x
        Jx_sum = np.zeros((Nx, Ny), dtype=float)
        Jy_sum = np.zeros((Nx, Ny), dtype=float)
        for s in tqdm(range(S), desc="Current maps avg"):
            G_ss = G_hist[s, -1]
            Jx, Jy = model.current_maps_gauge_invariant(G_ss)
            Jx_sum += Jx
            Jy_sum += Jy
        Jx_avg = Jx_sum / S
        Jy_avg = Jy_sum / S

        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.8], hspace=0.35, wspace=0.25)
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, :])

        plot_current_maps_axes(fig, [ax_a, ax_b, ax_c], Jx_avg, Jy_avg, vmin=-0.005, vmax=0.005)

        Jx_int = np.sum(Jx_avg, axis=1)
        Jy_int = np.sum(Jy_avg, axis=1)
        x_vals = np.arange(Nx)
        ax_d.plot(x_vals, Jx_int, marker="o", lw=1.6, label=r"$\int dy\, J_x$")
        ax_d.plot(x_vals, Jy_int, marker="s", lw=1.6, label=r"$\int dy\, J_y$")
        ax_d.set_xlabel("x")
        ax_d.set_ylabel(r"$\int dy\, J_i(x,y)$")
        ax_d.set_title("Steady-state y-integrated currents (avg over samples)")
        ax_d.grid(alpha=0.3)
        ax_d.legend()

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Saved plots to {pdf_path}")


if __name__ == "__main__":
    main()
