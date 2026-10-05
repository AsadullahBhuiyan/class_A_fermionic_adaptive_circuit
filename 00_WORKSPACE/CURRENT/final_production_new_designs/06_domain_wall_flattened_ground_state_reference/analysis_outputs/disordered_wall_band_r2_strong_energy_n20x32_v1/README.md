# Wall-band iid Gaussian disorder (radius 2 unit cells)

300 static half-filled pure ground states of the matched hard-wall flattened parent:
100 each at disorder variance 0.64, 1, 4. Independent Gaussian diagonal potentials
have population mean zero; each x,y,orbital within x=3..7 or x=13..17 has an
independent potential, and all other entries are exactly zero.
The same underlying random draws as the full-diagonal strong-disorder scan
are used, with a spatial mask applied. This is a five-column band around each wall. No spatial
demeaning or post-disorder reflattening is applied. Nx=20, Ny=32, x walls=5,15.
The half-system cut has Ay=16, y0=0. Keep |lambda|<=0.99 and use 101 bins.

Run run_analysis.py to acquire/resume; run build_notebook.py for the executed
analysis notebook, raw histogram and clean/disordered normalized comparison.
The clean Hamiltonian, every random potential, spectrum, seed, configuration,
source hashes and realization checksums are saved. Ground-state eigenvectors
are computed transiently and can be reconstructed from the saved Hamiltonian
and potentials; they are not archived as dense frames.

No circuit dynamics or purification data enter this experiment.
