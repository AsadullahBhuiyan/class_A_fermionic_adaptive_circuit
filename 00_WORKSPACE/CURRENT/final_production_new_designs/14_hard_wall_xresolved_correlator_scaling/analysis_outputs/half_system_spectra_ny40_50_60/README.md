# Half-system occupation and modular spectra

This analysis pools the saved endpoint spectra from the completed hard-wall
x-resolved correlator campaign: Nx=20, Ny=40,50,60, 100 independent trajectories
per size, alpha_1=1, alpha_2=30, nshell=1, raster-y measurements, pure half-filled
initialization, perfect correction, and T=2Ny cycles. The subsystem is

$$
A=[0,N_x)\times[0,N_y/2),
$$

including both physical orbitals. Its single-particle dimension is Nx*Ny:
800, 1,000, or 1,200 modes per sample, respectively. There are 300,000 saved
occupation eigenvalues in total. The input is the immutable
`gpu_data/hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_endpoint_frame_halfcov_occupations_v2_30gib_batched/`
directory under the owning bundle. All 120 input files were checked against
the download manifest; each size covers sample indices 0--99 exactly once.

## Conversion and normalization

For each sample, the saved nu values are the eigenvalues of its restricted
occupation matrix C_A. The single-particle modular energies are

$$
\epsilon_a=\log\frac{1-\nu_a}{\nu_a}.
$$

They define the quadratic part of the many-body modular Hamiltonian through

$$
\hat\rho_A=Z_A^{-1}\exp\!\left[-\sum_a
\epsilon_a\hat d_a^\dagger\hat d_a\right].
$$

Here log denotes the natural logarithm. These are single-particle modular
energies; many-body levels would require the occupation-string sums and the
normalization constant. The conversion is made for each individual sample
before pooling, without averaging covariance matrices.

The occupation histogram includes every saved eigenvalue, including exact
0 and 1. With M eigenvalues and bin width Delta_nu, its density is

$$
p_j=\frac{n_j}{M\,\Delta\nu},\qquad
\sum_j p_j\Delta\nu=1.
$$

The modular plot uses the same rule with its finite eigenvalues and unit-width
energy bins. Its displayed finite-conditional density also integrates to one.
Every individual eigenvalue has equal weight in the pooled curve, so larger
subsystems contribute more levels. Each colored per-size curve is separately
normalized. The black curve combines all sizes. Vertical axes are logarithmic;
horizontal coordinates are linear. Empty bins receive no pseudocount.

## Saved endpoints and numerical precision

58,027 saved occupations equal 0 and 62,698 equal 1. The conversion retains
their positive and negative infinite energies in the output arrays; the
179,275 finite energies enter the modular histogram. Thus 40.2417% of the
saved levels are outside the finite-energy histogram. Finite energies range
from -36.7368 to 110.723. No new clipping or finite replacement for infinity
is performed.

The acquisition observer clipped tiny roundoff excursions into [0,1]. Its raw
maximum reached 1+1.23e-13. Consequently the extreme transformed tails, and
their apparent asymmetry, are sensitive to numerical precision near 0 and 1.
Gray shading marks |epsilon| beyond log[(1-1e-12)/1e-12], approximately 27.63,
as a visual precision guide only: it is not a filter or a rigorous error bound.
All finite saved values remain in the full histogram. The central-energy
figure is a view of the same histogram over [-12,12], without renormalization.

## Outputs

- `half_system_spectra_log_density.pdf` and `.png`: normalized occupation and
  finite modular-energy histograms, with all sizes pooled and per-size overlays.
- `half_system_modular_central_log_density.pdf` and `.png`: central modular view.
- `half_system_occupation_and_modular_spectra.npz`: all sample-resolved
  occupations and transformed energies, histogram edges, counts, and densities.
  Arrays `occupations_Ny*` and `modular_energies_by_occupation_Ny*` share sample
  and mode indices; occupations are ascending and their corresponding energies
  are descending. Infinities are retained. The `sample_indices_Ny*` arrays are
  the explicit sample coordinate.
- `histogram_*.csv`: bin bounds, raw counts, and probability density for each
  observable and size, including pooled histograms.
- `summary.json`: input identity, normalization checks, and endpoint counts.

Regenerate from the repository root with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python 00_WORKSPACE/CURRENT/final_production_new_designs/14_hard_wall_xresolved_correlator_scaling/analyze_half_system_spectra.py
```

Figure caption: Half-system occupation and single-particle modular spectra
for hard-wall trajectories at Nx=20, Ny=40,50,60, alpha_1=1, alpha_2=30,
nshell=1, and T=2Ny. Each size contains S=100 independent trajectories from
pure half-filled initialization with raster-y measurements and perfect
correction. The spectra are evaluated per trajectory and then pooled;
the black histogram gives the combined eigenvalue distribution. Histograms
show probability density on logarithmic vertical axes, with occupation-bin
width 0.0125 and modular-energy-bin width 1. Each full occupation histogram
has unit area. Each finite modular histogram is normalized conditionally on
finite levels; 40.2417% of all pooled saved levels are at the exact occupation
endpoints. Gray shading identifies precision-sensitive modular tails. There
are no fits or uncertainty bars in these empirical spectral histograms.
