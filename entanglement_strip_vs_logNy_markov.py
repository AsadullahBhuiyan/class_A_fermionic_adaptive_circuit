import os

os.environ["OMP_NUM_THREADS"] = "50"
os.environ["MKL_NUM_THREADS"] = "50"
os.environ["OPENBLAS_NUM_THREADS"] = "50"
os.environ["NUMEXPR_NUM_THREADS"] = "50"

import numpy as np
import matplotlib.pyplot as plt

import importlib
import classA_U1FGTN

importlib.reload(classA_U1FGTN)
from classA_U1FGTN import classA_U1FGTN


def product_state_one_per_cell_top_layer(Nx, Ny):
    """
    Full-layer covariance with one occupied top-layer mode per unit cell.
    Indexing matches i = mu + 2*x + 2*Nx*y for the top layer.
    """
    Nlayer = 2 * Nx * Ny
    G = -np.eye(Nlayer, dtype=np.complex128)

    # Occupy the mu=2 top-layer mode in each cell.
    for y in range(Ny):
        for x in range(Nx):
            idx_top_mu0 = 1 + 2 * x + 2 * Nx * y
            G[idx_top_mu0, idx_top_mu0] = 1.0

    return G


def entanglement_contour_map(model, G_top):
    Ny = model.Ny
    y_cut = Ny // 2
    yA = np.arange(y_cut, Ny)
    xA = np.arange(model.Nx)

    sub_indices = []
    for y in yA:
        for x in xA:
            sub_indices.append(0 + 2 * x + 2 * model.Nx * y)
            sub_indices.append(1 + 2 * x + 2 * model.Nx * y)
    sub_indices = np.array(sub_indices, dtype=int)

    G_sub = G_top[np.ix_(sub_indices, sub_indices)]
    Ny_sub = len(yA)
    contour_helper = classA_U1FGTN(model.Nx, Ny_sub, DW=False, alpha_1=1, alpha_2=1)
    s_map = contour_helper.entanglement_contour(G_sub)

    return s_map


def main():
    # Parameters from the provided setup
    Nx = 12
    Ny_list = [12, 16, 20, 24, 28, 32]
    alpha_1, alpha_2 = 30, 1
    cycles = 48
    nshell = 2

    samples = 100

    x_list = np.arange(Nx // 2)
    Sx_vs_Ny = {x: [] for x in x_list}

    for Ny in Ny_list:
        G_init = product_state_one_per_cell_top_layer(Nx, Ny)
        model = classA_U1FGTN(Nx, Ny, nshell=nshell, DW=True, alpha_1=alpha_1, alpha_2=alpha_2)
        res = model.run_markov_circuit(
            G_history=False,
            cycles=cycles,
            progress=True,
            G_init=G_init,
            save=True,
            samples=samples,
            p_meas=1,
            parallelize_samples=True,
            sequence="dw_symmetric",
            top_triv_back_forth=True,
            top_triv_block_cycles=12
        )

        G_finals = res["G_final"]
        s_map_avg = None
        for sample_idx in range(G_finals.shape[0]):
            s_map = entanglement_contour_map(model, G_finals[sample_idx])
            if s_map_avg is None:
                s_map_avg = s_map
            else:
                s_map_avg = s_map_avg + s_map
        s_map_avg = s_map_avg / G_finals.shape[0]
        Sx = np.sum(s_map_avg, axis=1)
        for x in x_list:
            Sx_vs_Ny[x].append(Sx[x])

    xvals = np.log(np.array(Ny_list))
    plt.figure(figsize=(7, 5))
    for x in x_list:
        curve = np.array(Sx_vs_Ny[x])
        plt.plot(xvals, curve, marker="o", lw=1.5)
        plt.annotate(
            f"x={x}",
            xy=(xvals[-1], curve[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=12,
            va="center",
        )

    x_fit = xvals
    y_fit = np.array(Sx_vs_Ny[4])
    m, b = np.polyfit(x_fit, y_fit, 1)
    plt.plot(x_fit, m * x_fit + b, "k--", lw=2, label=f"fit: slope={m:.4g}")

    plt.xlabel(r"$\log(N_y)$")
    plt.ylabel(r"Strip entanglement sum $\sum_y s(x,y)$")
    plt.title("Steady-state strip entanglement vs log Ny (Markov circuit)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dw_entanglement_vs_logNy_fit.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
