#!/usr/bin/env python3
"""Build the interactive equilibrium wall-crossing teaching notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "equilibrium_wall_crossing_explorer.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = [
    md(r"""
# What the wall crossing actually means

This is a single-state, equilibrium companion to the monitored-endpoint spectral-flow calculation.  It uses the canonical CPU overcomplete Wannier (OW) construction with $n_{\rm shell}=1$, but it does **not** run the monitored circuit.  The default $20\times24$ two-wall slab is translation invariant in $y$, so the Hamiltonian separates into small momentum blocks and the complete demonstration runs in roughly half a minute on a laptop-class CPU.

There are two things to inspect:

1. a magnified finite-size avoided crossing near $\phi=0$;
2. a full $2\pi$ flux cycle comparing instantaneous energy refilling with previous-overlap continuation.

The sliders do not approximate additional physics.  They only move through arrays computed by the preceding scan cells.
"""),
    code(r"""
# CPU allocation (edit this before running anything else).
import os

CPU_RANGE = "0-7"  # Examples: "0-7", "16-23", or "" to keep the current affinity.

def parse_cpu_range(spec):
    cpus = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(value) for value in item.split("-", 1))
            cpus.update(range(first, last + 1))
        else:
            cpus.add(int(item))
    return cpus

available = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set()
requested = parse_cpu_range(CPU_RANGE)
selected = (requested & available) if requested and available else (requested or available)
if selected and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, selected)
BLAS_THREADS = max(1, len(selected)) if selected else 1
for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = str(BLAS_THREADS)
print({
    "requested_cpu_range": CPU_RANGE,
    "active_cpus": sorted(selected) if selected else "unchanged",
    "blas_threads": BLAS_THREADS,
})
"""),
    md(r"""
## The object being threaded

The OW construction gives a single-particle slab Hamiltonian $h(\phi)$.  The topological region has $\alpha_1=1$, the exterior has $\alpha_2=30$, and periodic $y$ boundary conditions are twisted by the holonomy $e^{i\phi}$.  The two interfaces are therefore physical left- and right-localized Chern-wall channels.

At half filling, diagonalize

$$h(\phi)\lvert w_n(\phi)\rangle=\epsilon_n(\phi)\lvert w_n(\phi)\rangle.$$

The two modes straddling the occupation boundary are $\lvert w_{r-1}\rangle$ and $\lvert w_r\rangle$.  Their finite-size splitting is the internal wall-pair gap.  Far from the crossing, one is on the left wall and one is on the right.  At the avoided crossing, the energy eigenvectors hybridize.  Inside their two-dimensional span $E$, diagonalizing

$$B_E=E^\dagger(R_R-R_L)E$$

recovers explicitly left- and right-localized combinations.  This is what the production script means by identifying the wall pair separately from the spectator states.

There are then two different occupation rules:

- **Instantaneous refilling:** occupy the lowest $r$ energies independently at every $\phi$.  The occupied energy branch changes its wall identity at the crossing and the projector closes after $2\pi$.
- **Continued occupation:** match neighboring eigenvectors by maximum overlap and keep the occupation labels inherited from the initial point.  The same physical branch is followed through the crossing, even when it is no longer the lower-energy member.  This is spectral flow, not real-time Schrödinger evolution.
"""),
    code(r"""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import ipywidgets as widgets

relative = Path("00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot/notebooks/equilibrium_wall_crossing_explorer")
candidates = [Path.cwd(), Path.cwd() / relative]
MODULE_DIR = next((path.resolve() for path in candidates if (path / "wall_crossing_explorer.py").is_file()), None)
if MODULE_DIR is None:
    raise FileNotFoundError("Run this notebook from its own directory or from the repository root.")
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from wall_crossing_explorer import (
    ExplorerConfig,
    scan_flux_pump,
    scan_local_crossing,
    validate_results,
)

plt.rcParams.update({
    "font.family": "CMU Sans Serif",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
})

# Edit only this cell to change the demonstration.
CONFIG = ExplorerConfig(
    nx=20,
    ny=24,
    nshell=1,
    wall="soft",       # "soft" or "hard"
    direction="ccw",  # "ccw" or "cw"
    intervals=64,
    regulator=1.0e-7,
    local_half_width=0.08,
    local_points=49,
)
print(json.dumps(CONFIG.metadata(), indent=2))
"""),
    md(r"""
## 1. Build one exact equilibrium scan

Every flux point rebuilds the OW functions before diagonalization.  This matters: the code is changing the periodic boundary condition of the Hamiltonian, not merely multiplying a fixed state by a local phase.  The first progress bar magnifies the crossing; the second performs the complete flux cycle.
"""),
    code(r"""
local = scan_local_crossing(CONFIG, progress=True)
pump = scan_flux_pump(CONFIG, progress=True)
print("scan complete")
"""),
    md(r"""
## 2. See the avoided crossing

The left panel shows the two energy-ordered levels at half filling.  Their energies avoid one another at finite size.  The right panel shows the crucial information hidden by energy alone: the lower-energy eigenvector changes from one wall to the other as $\phi$ passes through zero.  The wall-localized basis obtained from $B_E$ remains polarized near $-1$ and $+1$.
"""),
    code(r"""
phi = np.asarray(local["phi"])
energy = np.asarray(local["edge_energy"])
energy_pol = np.asarray(local["edge_energy_polarization"])
wall_pol = np.asarray(local["wall_polarization"])

fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))
axes[0].plot(phi, energy[:, 0], "-", color="#D55E00", lw=1.4, label=r"lower energy $w_{r-1}$")
axes[0].plot(phi, energy[:, 1], "--", color="#0072B2", lw=1.4, label=r"upper energy $w_r$")
axes[0].axvline(0, color="0.45", ls=":", lw=0.8)
axes[0].axhline(0, color="0.45", ls=":", lw=0.8)
axes[0].set(xlabel=r"flux $\phi$", ylabel=r"edge energy $\epsilon$", title="Finite-size avoided crossing")
axes[0].legend(fontsize=7)

axes[1].plot(phi, energy_pol[:, 0], "-", color="#D55E00", lw=1.4, label="lower-energy mode")
axes[1].plot(phi, energy_pol[:, 1], "--", color="#0072B2", lw=1.4, label="upper-energy mode")
axes[1].plot(phi, wall_pol[:, 0], ":", color="0.25", lw=1.1, label="left wall basis")
axes[1].plot(phi, wall_pol[:, 1], "-.", color="0.25", lw=1.1, label="right wall basis")
axes[1].axvline(0, color="0.45", ls=":", lw=0.8)
axes[1].axhline(0, color="0.45", ls=":", lw=0.8)
axes[1].set(xlabel=r"flux $\phi$", ylabel=r"$\langle R_R-R_L\rangle$", title="The energy labels exchange walls")
axes[1].legend(fontsize=6.8, ncol=2)
for label, axis in zip("ab", axes):
    axis.text(-0.13, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
fig.tight_layout()
plt.show()
"""),
    md(r"""
### Slide through the crossing

Move the slider from negative to positive flux.  In the middle panel, the lower-energy mode migrates from the right wall to the left wall while the upper-energy mode does the reverse.  At the center they are symmetric and antisymmetric mixtures.  In the last panel, diagonalizing $B_E$ rotates those two mixtures back into a left/right basis.  Nothing has been added to the Hilbert space; only the basis inside the same two-dimensional cluster has changed.
"""),
    code(r"""
x = np.arange(CONFIG.nx)
crossing_slider = widgets.IntSlider(
    value=len(phi) // 2,
    min=0,
    max=len(phi) - 1,
    step=1,
    description="flux index",
    continuous_update=False,
    layout=widgets.Layout(width="75%"),
)

def draw_crossing(index):
    index = int(index)
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
    axes[0].plot(phi, energy[:, 0], color="#D55E00", lw=1.1)
    axes[0].plot(phi, energy[:, 1], color="#0072B2", lw=1.1, ls="--")
    axes[0].axvline(phi[index], color="k", lw=0.9)
    axes[0].axhline(0, color="0.5", lw=0.7, ls=":")
    axes[0].set(xlabel=r"$\phi$", ylabel=r"$\epsilon$", title="where we are")

    energy_density = np.asarray(local["energy_density_x"])[index]
    axes[1].plot(x, energy_density[0], color="#D55E00", marker="^", ms=3, lw=1.0, label="lower energy")
    axes[1].plot(x, energy_density[1], color="#0072B2", marker="o", ms=3, lw=1.0, ls="--", label="upper energy")
    axes[1].axvline(CONFIG.nx / 2 - 0.5, color="0.5", lw=0.7, ls=":")
    axes[1].set(xlabel=r"$x$", ylabel="mode density", title="energy eigenmodes")
    axes[1].legend(fontsize=6.5)

    wall_density = np.asarray(local["wall_density_x"])[index]
    axes[2].plot(x, wall_density[0], color="#009E73", marker="s", ms=3, lw=1.0, ls="--", label="left-localized")
    axes[2].plot(x, wall_density[1], color="#CC79A7", marker="o", ms=3, lw=1.0, label="right-localized")
    axes[2].axvline(CONFIG.nx / 2 - 0.5, color="0.5", lw=0.7, ls=":")
    axes[2].set(xlabel=r"$x$", ylabel="mode density", title=r"same cluster, diagonalize $B_E$")
    axes[2].legend(fontsize=6.5)
    fig.suptitle(
        rf"$\phi={phi[index]:+.5f}$; gap={local['edge_internal_gap'][index]:.3e}; "
        rf"energy polarizations=({energy_pol[index,0]:+.3f},{energy_pol[index,1]:+.3f})",
        fontsize=8,
    )
    fig.tight_layout()
    plt.show()

crossing_output = widgets.interactive_output(draw_crossing, {"index": crossing_slider})
display(crossing_slider, crossing_output)
"""),
    md(r"""
## 3. Turn the local crossing into a full flux pump

The overlap algorithm solves an assignment problem separately in each conserved momentum block.  If $V_{j-1}$ is the previous eigenbasis and $W_j$ is the newly diagonalized basis, it maximizes the total neighboring overlap $|V_{j-1}^\dagger W_j|^2$.  The initially occupied labels are then kept fixed.

This produces a continued frame $F_{\rm cont}(\phi)$.  By contrast, $F_{\rm inst}(\phi)$ is rebuilt from the lowest $r$ energies at every point.  Both give projectors $P=FF^\dagger$ and direct regional charges.  The plotted quantity is

$$q_x(\phi)=\frac{1}{2}\left[\Delta N_R(\phi)-\Delta N_L(\phi)\right].$$

No tangent mode, real-time ramp, monitored replay, or source subtraction enters this notebook.
"""),
    code(r"""
flux = np.asarray(pump["threaded_flux"])
q_cont = np.asarray(pump["continued_q_x"])
q_inst = np.asarray(pump["instantaneous_q_x"])

fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35))
axes[0].plot(flux, q_cont, color="#0072B2", lw=1.4, label=r"continued $P_{\rm cont}$")
axes[0].plot(flux, q_inst, color="#D55E00", lw=1.2, ls=":", label=r"instantaneous $P_{\rm inst}$")
axes[0].axhline(CONFIG.sigma, color="0.5", ls="--", lw=0.8)
axes[0].set_xticks([0, np.pi, 2*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
axes[0].set(xlabel="threaded flux", ylabel=r"$q_x$", title="Spectral-flow charge")
axes[0].legend(fontsize=6.5)

axes[1].plot(flux, pump["continued_delta_N_left"], color="#009E73", lw=1.2, ls="--", label=r"$\Delta N_L$")
axes[1].plot(flux, pump["continued_delta_N_right"], color="#CC79A7", lw=1.2, label=r"$\Delta N_R$")
axes[1].axhline(0, color="0.5", ls=":", lw=0.7)
axes[1].set_xticks([0, np.pi, 2*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
axes[1].set(xlabel="threaded flux", ylabel="regional charge change", title="Where the charge went")
axes[1].legend(fontsize=6.5)

weights = np.asarray(pump["edge_continued_occupation_weight"])
axes[2].plot(flux, weights[:, 0], color="#D55E00", lw=1.2, label="lower-energy occupation")
axes[2].plot(flux, weights[:, 1], color="#0072B2", lw=1.2, ls="--", label="upper-energy occupation")
axes[2].set_xticks([0, np.pi, 2*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
axes[2].set(xlabel="threaded flux", ylabel="continued occupation weight", title="Continuation can occupy the upper level")
axes[2].legend(fontsize=6.2)
for label, axis in zip("abc", axes):
    axis.text(-0.17, 1.04, f"({label})", transform=axis.transAxes, fontweight="bold")
fig.tight_layout()
plt.show()
"""),
    md(r"""
### Slide through the complete flux path

This is the closest literal view of what the code decides at one step.  The table reports the two current energy-ordered modes.  Instantaneous refilling always assigns occupations $(1,0)$ to the lower and upper members.  Previous-overlap continuation can instead assign the occupied weight to the upper member after the physical branch passes through the crossing.
"""),
    code(r"""
pump_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(flux) - 1,
    step=1,
    description="flux index",
    continuous_update=False,
    layout=widgets.Layout(width="75%"),
)

def draw_pump_step(index):
    index = int(index)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45))
    axes[0].plot(flux, q_cont, color="#0072B2", lw=1.3, label="continued")
    axes[0].plot(flux, q_inst, color="#D55E00", lw=1.1, ls=":", label="instantaneous refill")
    axes[0].scatter([flux[index]], [q_cont[index]], color="k", s=22, zorder=5)
    axes[0].axvline(flux[index], color="k", lw=0.7)
    axes[0].set_xticks([0, np.pi, 2*np.pi], [r"$0$", r"$\pi$", r"$2\pi$"])
    axes[0].set(xlabel="threaded flux", ylabel=r"$q_x$", title="charge accumulated so far")
    axes[0].legend(fontsize=7)

    delta_cont = np.asarray(pump["continued_density_x"])[index] - np.asarray(pump["continued_density_x"])[0]
    delta_inst = np.asarray(pump["instantaneous_density_x"])[index] - np.asarray(pump["instantaneous_density_x"])[0]
    axes[1].plot(x, delta_cont, color="#0072B2", marker="o", ms=3, lw=1.1, label="continued")
    axes[1].plot(x, delta_inst, color="#D55E00", marker="^", ms=3, lw=1.0, ls=":", label="instantaneous")
    axes[1].axhline(0, color="0.5", lw=0.7, ls=":")
    axes[1].axvline(CONFIG.nx / 2 - 0.5, color="0.5", lw=0.7, ls=":")
    axes[1].set(xlabel=r"$x$", ylabel=r"$\Delta n(x)$", title="same projectors, resolved in space")
    axes[1].legend(fontsize=7)
    fig.suptitle(rf"path point {index}/{CONFIG.intervals}, $|\phi-\phi_0|={flux[index]:.5f}$", fontsize=8)
    fig.tight_layout()
    plt.show()

    table = pd.DataFrame({
        "current mode": ["lower energy", "upper energy"],
        "energy": np.asarray(pump["edge_energy"])[index],
        "wall polarization": np.asarray(pump["edge_energy_polarization"])[index],
        "instantaneous occupation": np.asarray(pump["edge_instantaneous_occupation_weight"])[index],
        "continued occupation weight": np.asarray(pump["edge_continued_occupation_weight"])[index],
    })
    display(table.style.format({
        "energy": "{:+.6e}",
        "wall polarization": "{:+.4f}",
        "instantaneous occupation": "{:.1f}",
        "continued occupation weight": "{:.6f}",
    }))

pump_output = widgets.interactive_output(draw_pump_step, {"index": pump_slider})
display(pump_slider, pump_output)
"""),
    md(r"""
## 4. Mapping this demonstration onto the production wall-diabatized script

The production calculation uses a monitored-trajectory endpoint projector rather than this clean equilibrium OW Hamiltonian, but the logical steps are the same:

1. construct the flux-threaded single-particle parent;
2. diagonalize it at the next flux point;
3. continue the ordinary spectator subspace by previous-projector overlap;
4. remove the rank-crossing pair from that spectator assignment;
5. diagonalize $E^\dagger(R_R-R_L)E$ inside the pair;
6. preserve the entering left/right wall label through the avoided crossing;
7. recombine the selected wall mode with the occupied spectators;
8. form the projector and evaluate $N_L$, $N_R$, and raw $q_x$.

The extra wall step prevents a tiny finite-size avoided crossing from deciding the macroscopic branch by energy ordering.  It does **not** force an arbitrary result: the code first requires an isolated, oppositely localized wall pair.  If that structure is absent, the endpoint is marked unresolved.
"""),
    md(r"""
## 5. Raw numerical diagnostics

These checks ensure that the pictures above actually represent the intended calculation.  Translation invariance means that the off-diagonal momentum blocks vanish.  Regional charge must close, the instantaneous projector must return to zero transfer, and the continued path should reach the orientation-expected unit response for this demonstrator.
"""),
    code(r"""
diagnostics = validate_results(local, pump)
display(pd.Series(diagnostics, name="value").to_frame())
print(json.dumps(diagnostics, indent=2))
"""),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
