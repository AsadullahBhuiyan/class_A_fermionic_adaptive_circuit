import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from classA_U1FGTN import classA_U1FGTN

# Lattice + DW parameters
Nx, Ny = 12, 12
alpha_1, alpha_2 = 30, 1

# Build model and exact DW ground-state covariance (top layer only)
model = classA_U1FGTN(Nx, Ny, DW=True, alpha_1=alpha_1, alpha_2=alpha_2)
G_gs = model.G_CI_domain_wall(periodic=True)  # shape: (2*Nx*Ny, 2*Nx*Ny)

# Top-layer covariance
Nlayer = model.Ntot // 2
if G_gs.shape[0] == model.Ntot:
    G_top = G_gs[:Nlayer, :Nlayer]
elif G_gs.shape[0] == Nlayer:
    G_top = G_gs
else:
    raise ValueError(f"Unexpected G_gs shape: {G_gs.shape}")

# --- Correlation helper (matches plot_corr_y_profiles) ---
G2 = 0.5 * (G_top + np.eye(Nlayer, dtype=np.complex128))
G6 = G2.reshape(2, Nx, Ny, 2, Nx, Ny, order="F")
Gker = np.transpose(G6, (1, 2, 0, 4, 5, 3))  # (x,y,2,x',y',2)

ry_max = Ny // 2
ry_vals = np.arange(0, ry_max + 1, dtype=int)

# --- Plot y-profiles for all x with colorbar ---
fig, ax = plt.subplots(figsize=(7, 4))

cmap = cm.viridis
norm = colors.Normalize(vmin=0, vmax=Nx - 1)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)

for x0 in range(Nx):
    Gx = Gker[x0, :, :, x0, :, :]  # (Ny,2,Ny,2)
    Y = np.arange(Ny, dtype=np.intp)[:, None]
    Yp = (Y + ry_vals[None, :]) % Ny
    Gx_re = np.transpose(Gx, (0, 2, 1, 3)).reshape(Ny * Ny, 2, 2)
    flat_ix = (Y * Ny + Yp).reshape(-1)
    blocks = Gx_re[flat_ix].reshape(Ny, ry_vals.size, 2, 2)
    C_vec = np.sum(np.abs(blocks) ** 2, axis=(0, 2, 3)) / (2.0 * Ny)

    ax.plot(ry_vals, C_vec.real, lw=1.2, color=cmap(norm(x0)))

ax.set_xlabel(r"$r_y$")
ax.set_ylabel(r"$C_G(x_0, r_y)$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)

cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label(r"$x_0$")

plt.tight_layout()
plt.show()
