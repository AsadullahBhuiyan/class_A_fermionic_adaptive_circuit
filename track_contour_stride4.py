#!/usr/bin/env python3
"""
Track the entanglement contour along a single adaptive-circuit trajectory after
injecting maximally mixed blocks at the two domain walls (x = 4, 11) and a bulk
column (x = 7).  The script samples the contour every four measurement-feedback
steps and visualises both the contour slices and the peak motion.
"""

from __future__ import annotations

import os
import time
import importlib

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

import classA_U1FGTN as mod

importlib.reload(mod)
from classA_U1FGTN import classA_U1FGTN


# --------------------------------------------------------------------------- #
#                               Helper Routines                               #
# --------------------------------------------------------------------------- #

def unflatten_G(Gfull: np.ndarray, Nx: int, Ny: int, top: bool = True) -> np.ndarray:
    """Accept (..., Ntot, Ntot) and return (..., Nx, Ny, 2, Nx, Ny, 2)."""
    Gfull = np.asarray(Gfull)
    leading = Gfull.shape[:-2]
    Nlayer = 2 * Nx * Ny
    if top and Gfull.shape[-2:] != (Nlayer, Nlayer):
        raise ValueError(f"Expected (...,{Nlayer},{Nlayer}), got {Gfull.shape}")

    G6m = Gfull.reshape(leading + (2, Nx, Ny, 2, Nx, Ny), order="F")
    base = len(leading)
    axes = list(range(base)) + [
        base + 1,  # x
        base + 2,  # y
        base + 0,  # μ
        base + 4,  # x'
        base + 5,  # y'
        base + 3,  # ν
    ]
    return np.transpose(G6m, axes)


def flatten_G(G6: np.ndarray) -> np.ndarray:
    """
    Flatten G(x, y, μ; x', y', ν) -> G_flat with index i = μ + 2*x + 2*Nx*y,
    supporting any leading batch dimension.
    """
    G6 = np.asarray(G6, dtype=np.complex128)
    if G6.ndim < 6:
        raise ValueError(f"G must have at least 6 dims; got {G6.shape}")

    leading = G6.shape[:-6]
    Nx, Ny, s1, Nx2, Ny2, s2 = G6.shape[-6:]
    if (s1, s2) != (2, 2) or (Nx, Ny) != (Nx2, Ny2):
        raise ValueError(f"G must have shape (..., Nx, Ny, 2, Nx, Ny, 2); got {G6.shape}")

    base = len(leading)
    axes = list(range(base)) + [
        base + 2,  # μ
        base + 0,  # x
        base + 1,  # y
        base + 5,  # ν
        base + 3,  # x'
        base + 4,  # y'
    ]
    G6m = np.transpose(G6, axes)
    flat_shape = leading + (2 * Nx * Ny, 2 * Nx * Ny)
    return G6m.reshape(flat_shape, order="F")


def entanglement_contour_from_G6(G6: np.ndarray) -> np.ndarray:
    """Return s(x, y) for (..., Nx, Ny, 2, Nx, Ny, 2) covariances."""
    G6 = np.asarray(G6, dtype=np.complex128)
    if G6.ndim < 6:
        raise ValueError(f"G must have at least 6 dims; got {G6.shape}")

    Nx, Ny = G6.shape[-6], G6.shape[-5]
    leading = G6.shape[:-6]
    Gflat = flatten_G(G6)
    Gflat = 0.5 * (Gflat + np.swapaxes(Gflat.conj(), -1, -2))

    evals, vecs = np.linalg.eigh(Gflat)
    evals_2pt = np.clip(0.5 * (evals + 1), 1e-12, 1 - 1e-12)
    weights = -(evals_2pt * np.log(evals_2pt) + (1 - evals_2pt) * np.log(1 - evals_2pt))

    diagF = np.einsum("...ik,...k,...ik->...i", vecs, weights, vecs.conj(), optimize=True).real
    diagF = diagF.reshape(leading + (2, Nx, Ny), order="F")
    return diagF.sum(axis=-3)


def replace_covariance_blocks_with_maxmix(
    G: np.ndarray, x_coords, y_coords, Nx: int | None = None, Ny: int | None = None
) -> np.ndarray:
    """
    Zero out the lattice modes specified by (x_coords, y_coords) (0-based).
    Accepts top-layer tensors, flattened top layers, or the full two-layer covariance.
    """
    G = np.asarray(G, dtype=np.complex128)
    x_arr = np.asarray(list(x_coords), dtype=int)
    y_arr = np.asarray(list(y_coords), dtype=int)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x_coords and y_coords must have the same shape.")

    def collect_coords(x_in, y_in, Nx_, Ny_):
        coords, seen = [], set()
        for xi, yi in zip(x_in.flat, y_in.flat):
            if not (0 <= xi < Nx_) or not (0 <= yi < Ny_):
                raise ValueError(f"Coordinate {(int(xi), int(yi))} out of bounds for lattice ({Nx_}, {Ny_}).")
            key = (int(xi), int(yi))
            if key not in seen:
                seen.add(key)
                coords.append(key)
        return coords

    def selected_indices(coords, Nx_, Ny_, layer_offset=0, layer_span=0):
        shift = layer_offset * layer_span
        sel = []
        for x0, y0 in coords:
            base = 2 * (x0 + Nx_ * y0) + shift
            sel.extend((base, base + 1))
        return np.asarray(sel, dtype=int)

    def apply_zero_fill(mat, selected):
        if selected.size == 0:
            return np.array(mat, copy=True)
        total = mat.shape[0]
        mask = np.ones(total, dtype=bool)
        mask[selected] = False
        remaining = np.nonzero(mask)[0]
        out = np.zeros_like(mat)
        out[np.ix_(remaining, remaining)] = mat[np.ix_(remaining, remaining)]
        return out

    if G.ndim == 6:
        Nx6, Ny6, s1, Nx2, Ny2, s2 = G.shape
        if (s1, s2) != (2, 2) or (Nx6, Ny6) != (Nx2, Ny2):
            raise ValueError(f"Incompatible covariance shape {G.shape}; expected (Nx, Ny, 2, Nx, Ny, 2).")
        coords = collect_coords(x_arr, y_arr, Nx6, Ny6)
        if not coords:
            return np.array(G, copy=True)
        selected = selected_indices(coords, Nx6, Ny6)
        flat = flatten_G(G)
        return unflatten_G(apply_zero_fill(flat, selected), Nx6, Ny6, top=True)

    if G.ndim == 2:
        if G.shape[0] != G.shape[1]:
            raise ValueError("Covariance must be square.")
        if Nx is None or Ny is None:
            raise ValueError("Provide Nx and Ny for 2D covariance inputs.")
        coords = collect_coords(x_arr, y_arr, Nx, Ny)
        if not coords:
            return np.array(G, copy=True)
        total = G.shape[0]
        top_modes = 2 * Nx * Ny
        if total == top_modes or total == 2 * top_modes:
            selected = selected_indices(coords, Nx, Ny)
            return apply_zero_fill(G, selected)
        raise ValueError(f"Matrix dimension {total} incompatible with Nx={Nx}, Ny={Ny}.")

    raise ValueError("G must be either a 6D tensor or a 2D square matrix.")


# --------------------------------------------------------------------------- #
#                                Main Routine                                 #
# --------------------------------------------------------------------------- #

def main():
    start_time = time.time()

    Nx, Ny = 16, 16
    cycles = 20
    samples = 1  # single trajectory

    model = classA_U1FGTN(Nx, Ny, DW=True)
    DW_loc = tuple(int(v) for v in model.DW_loc)  # 0-based domain-wall columns

    data = np.load("cache/G_history_samples/N16x16_C20_S100_nshNone_DW1_init-default.npz", allow_pickle=True)
    G_hist_full = np.asarray(data[data.files[0]], dtype=np.complex128)
    G_init = G_hist_full[0, -1]  # steady-state snapshot of trajectory 0

    inj_sites = [
        (DW_loc[0], Ny // 2),
        (DW_loc[1], Ny // 2),
        (7, Ny // 2),
    ]
    x_coords, y_coords = zip(*inj_sites)
    G_init_mixed = replace_covariance_blocks_with_maxmix(G_init, x_coords=x_coords, y_coords=y_coords, Nx=Nx, Ny=Ny)

    rac_result = model.run_adaptive_circuit(
        G_history=True,
        cycles=cycles,
        samples=samples,
        progress=False,
        parallelize_samples=False,
        store="full",
        init_mode="default",
        G_init=G_init_mixed,
        save_suffix="_DW_mixed_injection_x=4-7-11",
    )
    G_hist_prime = rac_result["G_hist"][0]  # (T, 4*Nx*Ny, 4*Nx*Ny)

    Nlayer = 2 * Nx * Ny
    G_top = unflatten_G(G_hist_prime[:, :Nlayer, :Nlayer], Nx, Ny)  # (T, Nx, Ny, 2, Nx, Ny, 2)
    s_txy = entanglement_contour_from_G6(G_top)  # (T, Nx, Ny)

    step_stride = 4
    sampled_steps = np.arange(0, s_txy.shape[0], step_stride)
    tracked_x = np.array([DW_loc[0], 7, DW_loc[1]], dtype=int)
    tracked_labels = [f"x={x0 + 1}" for x0 in tracked_x]  # convert to 1-based for display
    y_axis = np.arange(1, Ny + 1)

    fig, axes = plt.subplots(len(tracked_x), 1, figsize=(6, 3.5), sharex=True)
    if len(tracked_x) == 1:
        axes = [axes]

    for ax, x0, lbl in zip(axes, tracked_x, tracked_labels):
        contours = s_txy[sampled_steps, x0, :]
        for idx, step in enumerate(sampled_steps):
            ax.plot(y_axis, contours[idx], marker="o", linewidth=1.2, label=f"t={step}")
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("y (1-based)")
    axes[0].set_title("Entanglement contour slices every 4 feedback steps")
    axes[0].legend(title="cycle", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    out_dir = "figs/contour_tracks"
    os.makedirs(out_dir, exist_ok=True)
    slice_path = os.path.join(out_dir, "contour_slices_stride4.png")
    plt.savefig(slice_path, dpi=300)
    print(f"Saved contour slice figure to {slice_path}")
    plt.show()

    peak_y = np.argmax(s_txy[:, tracked_x, :], axis=-1) + 1
    peak_s = np.take_along_axis(
        s_txy[:, tracked_x, :],
        np.argmax(s_txy[:, tracked_x, :], axis=-1)[..., None],
        axis=-1,
    ).squeeze(-1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    markers = ["o", "s", "^"]
    times = np.arange(s_txy.shape[0])
    mask = times % step_stride == 0
    log_min_global = np.log(np.maximum(peak_s[mask].min(), 1e-12))
    log_max_global = np.log(np.maximum(peak_s[mask].max(), 1e-12))
    norm = Normalize(log_min_global, log_max_global)

    for idx, lbl in enumerate(tracked_labels):
        log_vals = np.log(np.maximum(peak_s[mask, idx], 1e-12))
        sc = ax2.scatter(
            times[mask],
            peak_y[mask, idx],
            c=log_vals,
            cmap="viridis",
            norm=norm,
            marker=markers[idx % len(markers)],
            edgecolors="k",
            linewidths=0.3,
            label=lbl,
            zorder=3,
        )
        ax2.plot(times[mask], peak_y[mask, idx], color="0.4", linewidth=1.1, zorder=2)

    ax2.set_xlabel("cycle t")
    ax2.set_ylabel("y* (1-based)")
    ax2.set_title("Peak contour location every 4 steps")
    ax2.grid(alpha=0.3)
    ax2.legend(title="slice")
    cbar = fig2.colorbar(sc, ax=ax2, label="log max s(x, y, t)")
    cbar.ax.set_ylabel("log max s(x, y, t)")
    plt.tight_layout()

    peak_path = os.path.join(out_dir, "peak_tracks_stride4.png")
    plt.savefig(peak_path, dpi=300)
    print(f"Saved peak-track figure to {peak_path}")
    plt.show()

    print(f"Processed {s_txy.shape[0]} cycles; sampled every {step_stride}.")
    print(f"Total wall-clock: {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
