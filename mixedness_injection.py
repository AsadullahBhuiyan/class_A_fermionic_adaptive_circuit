import os, importlib, numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display
import pandas as pd
import time as time
import classA_U1FGTN as mod
importlib.reload(mod)
from classA_U1FGTN import classA_U1FGTN

start_time = time.time()

# parameters (DW = True)

Nx, Ny   = 16, 16
Ntot = 4*Nx*Ny
Nlayer = Ntot//2
cycles   = 20
samples  = 100
nshell   = None               # None for untruncated
DW       = True
model = classA_U1FGTN(Nx, Ny, DW=DW)
DW_loc = model.DW_loc


def unflatten_G(Gfull, Nx, Ny, top=True):
    """Accept (..., Ntot, Ntot) and return (..., Nx, Ny, 2, Nx, Ny, 2)."""
    Gfull = np.asarray(Gfull)
    leading = Gfull.shape[:-2]
    Nlayer = 2 * Nx * Ny
    if top and Gfull.shape[-2:] != (Nlayer, Nlayer):
        raise ValueError(f"Expected (...,{Nlayer},{Nlayer}), got {Gfull.shape}")

    # (..., μ, x, y, ν, x', y')
    G6m = Gfull.reshape(leading + (2, Nx, Ny, 2, Nx, Ny), order='F')

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


def flatten_G(G6):
    """
    Flatten G(x, y, μ; x', y', ν) -> G_flat with index
        i = μ + 2*x + 2*Nx*y   (Fortran order on (μ, x, y)),
    supporting any leading batch dimensions.

    Input shape: (..., Nx, Ny, 2, Nx, Ny, 2)
    Output shape: (..., 2*Nx*Ny, 2*Nx*Ny)
    """
    G6 = np.asarray(G6, dtype=np.complex128)
    if G6.ndim < 6:
        raise ValueError(f"G must have at least 6 dims; got {G6.shape}")

    leading = G6.shape[:-6]
    Nx, Ny, s1, Nx2, Ny2, s2 = G6.shape[-6:]
    if not ((s1, s2) == (2, 2) and (Nx, Ny) == (Nx2, Ny2)):
        raise ValueError(
            f"G must have shape (..., Nx, Ny, 2, Nx, Ny, 2); got {G6.shape}"
        )

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


def replace_covariance_blocks_with_maxmix(G, x_coords, y_coords, Nx=None, Ny=None):
    """
    Zero out the lattice modes (and all cross terms) specified by 0-based coordinates.
    Works with either:
      • G.shape == (Nx, Ny, 2, Nx, Ny, 2)            (top layer tensor)
      • G.shape == (2*Nx*Ny, 2*Nx*Ny)               (flattened top layer)
      • G.shape == (4*Nx*Ny, 4*Nx*Ny)               (full two-layer covariance)
    For 2D inputs, supply Nx and Ny so the mode indexing can be inferred.
    """
    G = np.asarray(G, dtype=np.complex128)
    x_arr = np.asarray(list(x_coords), dtype=int)
    y_arr = np.asarray(list(y_coords), dtype=int)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x_coords and y_coords must have the same shape.")

    def _collect_coords(x_in, y_in, Nx, Ny):
        coords, seen = [], set()
        for x0, y0 in zip(x_in.flat, y_in.flat):
            if not (0 <= x0 < Nx) or not (0 <= y0 < Ny):
                raise ValueError(f"Coordinate {(int(x0), int(y0))} out of bounds for lattice ({Nx}, {Ny}).")
            key = (int(x0), int(y0))
            if key not in seen:
                seen.add(key)
                coords.append(key)
        return coords

    def _selected_indices(coords, Nx, Ny, layer_offset=0, layer_span=0):
        sel = []
        shift = layer_offset * layer_span
        for x0, y0 in coords:
            base = 2 * (x0 + Nx * y0) + shift
            sel.extend((base, base + 1))
        return np.asarray(sel, dtype=int)

    def _apply_zero_fill(mat, selected):
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
        coords = _collect_coords(x_arr, y_arr, Nx6, Ny6)
        if not coords:
            return np.array(G, copy=True)
        selected = _selected_indices(coords, Nx6, Ny6)
        flat = flatten_G(G)
        flat_new = _apply_zero_fill(flat, selected)
        return unflatten_G(flat_new, Nx=Nx6, Ny=Ny6, top=True)

    if G.ndim == 2:
        if G.shape[0] != G.shape[1]:
            raise ValueError("Covariance must be square.")
        if Nx is None or Ny is None:
            raise ValueError("Provide Nx and Ny for 2D covariance inputs.")
        coords = _collect_coords(x_arr, y_arr, Nx, Ny)
        if not coords:
            return np.array(G, copy=True)
        total = G.shape[0]
        top_modes = 2 * Nx * Ny

        if total == top_modes:
            selected = _selected_indices(coords, Nx, Ny)
            return _apply_zero_fill(G, selected)

        if total == 2 * top_modes:
            selected = _selected_indices(coords, Nx, Ny)  # top layer occupies the first block
            return _apply_zero_fill(G, selected)

        raise ValueError(f"Matrix dimension {total} incompatible with Nx={Nx}, Ny={Ny} (expected 2*Nx*Ny or 4*Nx*Ny).")

    raise ValueError("G must be either a 6D tensor or a 2D square matrix.")


# Load cached history (full-layer covariances, shape (S, T, 4NxNy, 4NxNy))
data = np.load("cache/G_history_samples/N16x16_C20_S100_nshNone_DW1_init-default.npz", allow_pickle=True)
G_hist_full = np.asarray(data[data.files[0]], dtype=np.complex128)

# Final snapshot of the first trajectory -> 2D array (1024, 1024)
G_final_full = G_hist_full[0, -1]
Nx = model.Nx
Ny = model.Ny

# DW_loc is stored 0-based inside the model
x_targets = [int(model.DW_loc[0]), int(model.DW_loc[1]), 7]  # zero-based for the helper
y_targets = [Ny//2, Ny//2, Ny//2]   
                                # inject at y = 0 and y = Ny-1
x_targets_str= "-".join(map(str, x_targets))

# Zero the chosen top-layer sites (and all their cross terms) inside the full covariance
G_prime_full = replace_covariance_blocks_with_maxmix(
    G_final_full,
    x_coords=x_targets,
    y_coords=y_targets,
    Nx=Nx,
    Ny=Ny,
)

print(G_prime_full.shape)  # (1024, 1024)

# Run RAC from the modified state
rac_result = model.run_adaptive_circuit(
    G_history=True,
    cycles=20,
    samples=100,
    progress=False,
    parallelize_samples=False,
    store="full",
    init_mode="default",
    G_init=G_prime_full,
    save_suffix=f'_DW_mixed_injection_xtargets={x_targets_str}',
    n_jobs=56
)


G_hist_prime = rac_result["G_hist"]  # shape (1, T, 4NxNy, 4NxNy)
#data = np.load('cache/G_history_samples/N16x16_C20_S1_nshNone_DW1_init-default_store-full_DW_mixed_injection.npz', allow_pickle=True)
#G_hist_prime = data[data.files[0]]
x_pos = [int(model.DW_loc[0]), int(model.DW_loc[0]+1), 7,  int(model.DW_loc[1]-1), int(model.DW_loc[1])]
model.entanglement_contour_suite(G_hist_prime, save=True, custom_x_positions=x_pos, save_suffix=f'_DW_mixed_injection_xtargets={x_targets_str}')

print(f"Total Time Elapsed: {(time.time()-start_time):.3f} s")
