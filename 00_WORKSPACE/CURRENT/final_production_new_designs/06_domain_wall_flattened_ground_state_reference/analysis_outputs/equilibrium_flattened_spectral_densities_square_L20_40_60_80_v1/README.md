# Square-system equilibrium spectral densities

Execute equilibrium_square_spectral_densities.ipynb. Nx=Ny=20,40,60,80; hard walls at L/4,3L/4; alpha_1=1, alpha_2=30, nshell=1, zero twists, half filling.

The first code cell sets CPU affinity (eight CPUs by default). Canonical CPU OW construction, L momentum blocks of dimension 2L, and a half-system comprising all x and the first L/2 rows. Exact hard-wall sector separation reduces the L^2-mode eigensolve to two independent blocks; compare with a full solve at L=20.

Both log-axis histograms use 100 common bins and separate unit-area normalization. Include all Hamiltonian energies without rescaling. Covariance mixed modes satisfy abs(lambda)<1-1e-8. Raw spectra, per-size checksum-bound caches, diagnostics, CSVs, vector PDFs, PNGs, captions, and the executed notebook are saved here.

When the Fermi gap falls below numerical resolution, the selected pure ground state follows the reference stable energy sort. Diagnostics record this limitation; energy densities do not depend on that occupation choice. No twist, finite-temperature state, or trajectory averaging is introduced.

The figures now display only 80x80. Edit PLOT_SIZES in the Hamiltonian plot cell to restore other cached sizes; no eigendecompositions are needed.
