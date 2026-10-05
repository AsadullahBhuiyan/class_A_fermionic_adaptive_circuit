# Sample-resolved purification modular gap

This analysis uses the completed maximally mixed purification ensembles to
extract the single-particle modular spectrum and its lowest Lyapunov gap. The
primary data are Nx=20, Ny=20,30,40, S=100, and T=4Ny, with hard and soft
walls kept as separate ensembles. Every uncertainty band or error bar is the
ordinary standard error over independent trajectories; no bootstrap is used.

For a conditioned Gaussian state,

    rho_xi(t) = Z_xi(t)^(-1) exp[-K_xi(t)],
    K_xi(t) = sum_a epsilon_a^xi(t) f_a^dagger f_a,
    epsilon_a^xi(t) = log[(1-nu_a^xi(t))/nu_a^xi(t)].

The lowest modular flip cost is

    d_1^xi(t) = min_a |epsilon_a^xi(t)|.

With the repository's squared-singular-value convention, the trajectory
single-particle Lyapunov gap is the late-time slope of d_1^xi(t). A critical
wall mode is expected to give Delta_sp ~ A/Ny, so the gap closes as the
circumference grows. Exact occupation caps are treated as infinite flip costs;
they are never converted into artificial finite gaps. A trajectory is marked
unresolved once every mode is capped at the locked 1e-9 numerical resolution.

## Figures

- purification_modular_occupation_spectrum: ordered one-particle occupations
  versus cycle number for deterministic representative hard- and soft-wall
  trajectories at Ny=40.
- purification_single_particle_gap_every_sample: one row per trajectory,
  showing the finite-time lowest modular gap for all 100 samples at every
  primary circumference. Gray pixels are precision-censored, not zeros.
- purification_single_particle_gap_summary: trajectory-mean dynamics with
  ordinary SEM, the hard-wall Delta_sp=A/Ny comparison, and the soft-wall
  resolution boundary. The completed eight-size T=2Ny hard-wall ensemble is
  shown as a separate depth-limited comparison and is never pooled with the
  T=4Ny data.

The hard-wall finite-size trend is visible, but the coefficient remains
provisional because the first-gap slope still shifts between the 2Ny..3Ny and
3Ny..4Ny windows. The soft-wall late spectrum is more severely purified: only
roughly half the trajectories retain a finite lowest gap near 4Ny in
complex128. A fit restricted to those survivors would be selection-biased and
is therefore not reported as an asymptotic soft-wall gap.

## Additional sizes in the repository

The repository also contains a completed hard-wall S=100, T=2Ny campaign at
Ny=20,22,24,26,28,30,36,40. It provides useful size resolution but is not deep
enough for a released asymptotic rate. Older Ny=30,40,50 purification data save
entropy and charge but not occupation spectra. A transverse-width campaign at
Nx=20,24,28, Ny=20 saves only occupation extrema. The attempted Nx=16,
hard/soft T=4Ny CPU extension completed only two trajectories before a
numerical-tolerance failure and is not production evidence.

Reproduce from the repository root with:

    python 00_WORKSPACE/CURRENT/experiment_review/purification_modular_gap_sample_resolved/analyze_modular_gap_sample_resolved.py
