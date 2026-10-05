# Equilibrium flattened-parent spectral densities

Execute equilibrium_spectral_densities.ipynb locally on CPU. The first code cell exposes a CPU range. Uses the canonical CPU class to construct OW modes and the established equilibrium reference momentum-block algorithm. Nx=20, Ny=24,28,32,100, hard wall, alpha_1=1, alpha_2=30, nshell=1, zero twists, half filling.

Two separate log-axis plots: all Hamiltonian energies, normalized without rescaling energies; and half-system centered-covariance mixed modes, excluding abs(lambda)>=1-1e-8 and conditionally normalizing to unit area. One ground state per size, not 100 trajectories. Common 100-bin grids; empty bins are gaps.

Cached raw energies and centered spectra, source/configuration identities, numerical checks, histogram CSVs, PDF/PNG figures, and captions are saved beside the executed notebook. The local LaTeX compatibility shim and pdftoppm convert vector figures to 300-dpi PNG. Change bin counts or endpoint tolerance in the plotting cells without recomputing.

The Ny=100 extension uses 100 momentum blocks of size 40 and a 2000-mode half-system covariance. Earlier size caches remain valid when only the requested size list changes; their original identities and checksums are verified before reuse.
