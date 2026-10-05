# Ensemble-averaged purification modular gap

This folder contains the single paper-style plot requested for the purification
modular spectrum.  It replaces the multi-panel diagnostic presentation with a
direct gap-versus-size comparison.

For each independent trajectory at endpoint time `T`, the one-particle
occupation spectrum is converted to the modular spectrum

    epsilon_a^xi(T) = log[(1 - nu_a^xi(T)) / nu_a^xi(T)].

Because the conditioned particle number fluctuates between trajectories, the
signed spectra cannot be averaged at a fixed mode index without smearing the
modular Fermi level.  The analysis instead aligns each trajectory by excitation
order,

    d_j^xi(T) = sort_a |epsilon_a^xi(T)|,
    dbar_j(T) = (1/S) sum_xi d_j^xi(T),

and reads the gap from the first level of the averaged spectrum:

    Delta_sp(T) = dbar_1(T) / T.

Error bars are ordinary standard errors of `d_1^xi(T)/T` over the 100
independent trajectories.  No bootstrap is used.  Lines are descriptive
`A/Ny` fits through the origin.

The blue and red series use the completed hard- and soft-wall `T=4Ny`
campaigns at `Ny=20,30,40`.  The gray series is the separate hard-wall `T=2Ny`
campaign at `Ny=20,22,24,26,28,30,36,40`; it is shown to expose the denser size
grid but is never pooled with the deeper data.

The soft-wall endpoint is precision-censored: 45--48 of 100 trajectories have
all occupations at the locked `1e-9` cap.  The soft points therefore use the
finite cap and are lower bounds, rather than silently discarding those
trajectories.  Every hard-wall trajectory retains a resolved lowest mode.

Reproduce from the repository root with:

    python 00_WORKSPACE/CURRENT/experiment_review/purification_ensemble_averaged_modular_gap/make_figure.py
