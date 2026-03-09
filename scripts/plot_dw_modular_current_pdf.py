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

# ---- CPU allocation / thread caps ----
cpu_cap = 30
affinity_start = 60
os.environ["MY_CPU_COUNT"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_MAX_THREADS"] = str(1)
try:
    os.sched_setaffinity(0, set(range(affinity_start, affinity_start + 1)))
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

from fgtn.classA_U1FGTN import classA_U1FGTN
CACHE_PATH = (
    "cache/G_history_samples/"
    "N12x31/"
    "N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit.npz"
)
PDF_PATH = "figs/dw_modular_current_comparison.pdf"


def compute_modular_hamiltonian(G, eps=1e-12):
    """
    Single-particle modular Hamiltonian from covariance:
        H_mod = ln((1-G)/(1+G)) = -2*arctanh(G)
    """
    G = np.asarray(G, dtype=np.complex128)
    if G.shape[0] != G.shape[1]:
        raise ValueError("G must be square.")

    evals, evecs = np.linalg.eigh(G)
    evals = np.real_if_close(evals, tol=1e-10)
    evals = np.clip(evals, -1.0 + eps, 1.0 - eps)
    h_eigs = -2.0 * np.arctanh(evals)
    H_mod = (evecs * h_eigs[None, :]) @ evecs.conj().T
    return H_mod


def compute_modular_current_density(H_mod, G, Nx, Ny_sub):
    """
    Current density from H_mod and G on a subsystem of size (Nx, Ny_sub).
    Returns Jx, Jy with shape (Nx, Ny_sub), representing +x and +y bonds.
    """
    Nx = int(Nx)
    Ny_sub = int(Ny_sub)
    Nlayer = 2 * Nx * Ny_sub

    H_mod = np.asarray(H_mod, dtype=np.complex128)
    G = np.asarray(G, dtype=np.complex128)
    if H_mod.shape != (Nlayer, Nlayer) or G.shape != (Nlayer, Nlayer):
        raise ValueError(f"Expected ({Nlayer},{Nlayer}) for H_mod and G.")

    C = 0.5 * (G + np.eye(Nlayer, dtype=np.complex128))

    H6 = H_mod.reshape(2, Nx, Ny_sub, 2, Nx, Ny_sub, order="F")
    H6 = np.transpose(H6, (1, 2, 0, 4, 5, 3))  # (Nx,Ny,mu,Nx,Ny,nu)
    C6 = C.reshape(2, Nx, Ny_sub, 2, Nx, Ny_sub, order="F")
    C6 = np.transpose(C6, (1, 2, 0, 4, 5, 3))  # (Nx,Ny,mu,Nx,Ny,nu)

    x_idx = np.arange(Nx)[:, None]
    y_idx = np.arange(Ny_sub)[None, :]
    x_next = (x_idx + 1) % Nx
    y_next = (y_idx + 1) % Ny_sub

    Hx = H6[x_idx, y_idx, :, x_next, y_idx, :]  # (Nx,Ny,2,2)
    Cx = C6[x_next, y_idx, :, x_idx, y_idx, :]  # C_{j,i}
    Jx = 2.0 * np.imag(np.einsum("xyab,xyba->xy", Hx, Cx, optimize=True))

    Hy = H6[x_idx, y_idx, :, x_idx, y_next, :]  # (Nx,Ny,2,2)
    Cy = C6[x_idx, y_next, :, x_idx, y_idx, :]  # C_{j,i}
    Jy = 2.0 * np.imag(np.einsum("xyab,xyba->xy", Hy, Cy, optimize=True))

    return Jx, Jy


def build_sub_indices_keep_y(Nx, Ny_full, y_keep):
    sub_indices = []
    for y in y_keep:
        base = 2 * Nx * int(y)
        for x in range(Nx):
            sub_indices.append(base + 2 * x + 0)
            sub_indices.append(base + 2 * x + 1)
    return np.asarray(sub_indices, dtype=int)


def modular_current_from_full_G(G_full, Nx, Ny_full):
    """
    Trace out rows with y > Ny_full//2, then compute modular-current maps.
    """
    y_keep = np.arange(0, Ny_full // 2 + 1, dtype=int)  # keep y <= Ny//2
    sub_idx = build_sub_indices_keep_y(Nx, Ny_full, y_keep)
    G_sub = np.asarray(G_full[np.ix_(sub_idx, sub_idx)], dtype=np.complex128)
    Ny_sub = len(y_keep)

    H_mod = compute_modular_hamiltonian(G_sub)
    Jx, Jy = compute_modular_current_density(H_mod, G_sub, Nx, Ny_sub)
    return Jx, Jy, Ny_sub


def plot_page(pdf, page_title, chern_map, Jx, Jy, subtitle):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax_c, ax_jx = axes[0, 0], axes[0, 1]
    ax_jy, ax_q = axes[1, 0], axes[1, 1]

    im_c = ax_c.imshow(chern_map, origin="lower", cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax_c.set_title("Local Chern marker")
    ax_c.set_xlabel("y")
    ax_c.set_ylabel("x")
    fig.colorbar(im_c, ax=ax_c, label="tanh(Chern marker)")

    im_jx = ax_jx.imshow(Jx, origin="lower", cmap="RdBu_r", aspect="auto")
    ax_jx.set_title("Modular current Jx")
    ax_jx.set_xlabel("y (subsystem)")
    ax_jx.set_ylabel("x")
    fig.colorbar(im_jx, ax=ax_jx, label="Jx")

    im_jy = ax_jy.imshow(Jy, origin="lower", cmap="RdBu_r", aspect="auto")
    ax_jy.set_title("Modular current Jy")
    ax_jy.set_xlabel("y (subsystem)")
    ax_jy.set_ylabel("x")
    fig.colorbar(im_jy, ax=ax_jy, label="Jy")

    Nx, Ny_sub = Jx.shape
    yy, xx = np.meshgrid(np.arange(Ny_sub), np.arange(Nx))
    mag = np.sqrt(Jx**2 + Jy**2)
    q = ax_q.quiver(yy, xx, Jy, Jx, mag, cmap="plasma", pivot="mid")
    ax_q.set_title("Modular current quiver")
    ax_q.set_xlabel("y (subsystem)")
    ax_q.set_ylabel("x")
    ax_q.set_aspect("equal")
    fig.colorbar(q, ax=ax_q, label="|J|")

    fig.suptitle(f"{page_title}\n{subtitle}", fontsize=14)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)
    stage_bar = tqdm(total=4, desc="Pipeline", unit="stage")

    # ---- Page 1: equilibrium ground state ----
    Nx_eq, Ny_eq = 12, 31
    model_eq = classA_U1FGTN(Nx_eq, Ny_eq, nshell=None, DW=True, alpha_1=30, alpha_2=1)
    G_eq = model_eq.G_CI_domain_wall()
    chern_eq = model_eq.local_chern_marker_flat(G_eq)
    Jx_eq, Jy_eq, Ny_sub_eq = modular_current_from_full_G(G_eq, Nx_eq, Ny_eq)
    stage_bar.update(1)
    stage_bar.set_postfix_str("equilibrium ready", refresh=False)

    # ---- Page 2: circuit-prepared final snapshot (sample-avg) ----
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(CACHE_PATH)
    with np.load(CACHE_PATH) as data:
        if "G_hist" not in data:
            raise KeyError(f"'G_hist' not found in {CACHE_PATH}")
        G_hist = data["G_hist"]
    stage_bar.update(1)
    stage_bar.set_postfix_str("cache loaded", refresh=False)

    S, T, N, N2 = G_hist.shape
    if N != N2:
        raise ValueError(f"G_hist last dims must be square; got {N}x{N2}")
    Nx_cp, Ny_cp = 12, 31
    if N != 2 * Nx_cp * Ny_cp:
        raise ValueError(f"Expected N={2*Nx_cp*Ny_cp} from Nx={Nx_cp}, Ny={Ny_cp}; got {N}")

    model_cp = classA_U1FGTN(Nx_cp, Ny_cp, nshell=None, DW=True, alpha_1=30, alpha_2=1)

    G_final_batch = G_hist[:, -1]  # final snapshot over samples

    chern_list = []
    jx_list = []
    jy_list = []
    for s in tqdm(range(S), desc="Sample averages for circuit-prepared page"):
        Gs = G_final_batch[s]
        chern_list.append(model_cp.local_chern_marker_flat(Gs))
        Jx_s, Jy_s, _ = modular_current_from_full_G(Gs, Nx_cp, Ny_cp)
        jx_list.append(Jx_s)
        jy_list.append(Jy_s)

    chern_cp_avg = np.mean(np.stack(chern_list, axis=0), axis=0)
    Jx_cp_avg = np.mean(np.stack(jx_list, axis=0), axis=0)
    Jy_cp_avg = np.mean(np.stack(jy_list, axis=0), axis=0)
    stage_bar.update(1)
    stage_bar.set_postfix_str("sample averages ready", refresh=False)

    with PdfPages(PDF_PATH) as pdf:
        page_items = [
            (
                "equilibrium",
                chern_eq,
                Jx_eq,
                Jy_eq,
                f"Exact DW ground state (Nx={Nx_eq}, Ny={Ny_eq}), keep y <= Ny//2 (Ny_sub={Ny_sub_eq})",
            ),
            (
                "circuit-prepared",
                chern_cp_avg,
                Jx_cp_avg,
                Jy_cp_avg,
                f"Final snapshot averaged over S={S} samples (Nx={Nx_cp}, Ny={Ny_cp})",
            ),
        ]
        for page_title, chern_map, Jx_map, Jy_map, subtitle in tqdm(
            page_items, desc="Writing PDF pages", unit="page"
        ):
            plot_page(
                pdf,
                page_title=page_title,
                chern_map=chern_map,
                Jx=Jx_map,
                Jy=Jy_map,
                subtitle=subtitle,
            )
    stage_bar.update(1)
    stage_bar.set_postfix_str("pdf written", refresh=False)
    stage_bar.close()

    print(f"Saved {PDF_PATH}")


if __name__ == "__main__":
    main()
