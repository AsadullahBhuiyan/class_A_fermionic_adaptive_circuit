# Finite-time tangent gap: alpha sweep and size dependence

The requested two-row, one-column figure uses all 6,700 completed slot-17
trajectories (375 verified NPZ/completion pairs). It does not modify the
campaign, its previously saved gap convention, or any raw data.

## Estimator and figure caption

For each trajectory, use the saved signed pair rates

\[
\lambda_{ij,\xi}^{(T)}=
\frac{\log\sigma^o_{i,\xi}+\log\sigma^e_{j,\xi}}{T},\qquad
\gamma_\xi^{(T)}=\min_{ij}\left|\lambda_{ij,\xi}^{(T)}\right|.
\]

The saved five pairs were selected by increasing absolute rate over all
numerically finite occupied-empty pairs, so they include this minimum.
Numerical singular nulls were already excluded by the production estimator;
this analysis adds no cutoff, drops no trajectories, and does not replace a
genuine finite zero with a nonzero gap. The original saved effective gap was
`Delta = -2*lambda`; the present `gamma` is half its absolute minimum, **not**
the signed old gap. Compute the minimum absolute rate before taking any
trajectory average, not the absolute value of an averaged rate.

**Caption.** Finite-time endpoint tangent gaps for hard/support-truncated
domain walls, fixed transverse size \(N_x=20\),
\(n_{\rm shell}=1\), and \(\alpha_2=30\). Initial states are independent pure
half-filled random Slater states, followed by the campaign's hard-wall
exterior preparation. Raster-y dynamics use perfect correction, no
postselection, and complex128. Products include cycles 1 through
\(T=2N_y\), with no burn-in. (a) Mean gap versus \(\alpha_1\) for
\(N_y=24,28,32\), each on the 21-point sweep. (b) Mean gap versus \(N_y\)
at \(\alpha_1=1,3\), including \(N_y=36,40\). Each point is the arithmetic
mean of \(S=100\) trajectory-resolved \(\gamma_\xi^{(T)}\), with error bars
of one standard error of the mean (sample standard deviation divided by
\(\sqrt{100}\)); the independent sampling unit is a trajectory, not a batch
or a mode. Both vertical axes are logarithmic. Lines connect measurements;
there is no fit. Because the time window changes with size, this is a
finite-time size comparison, not a fixed-time comparison or an asymptotic
Lyapunov-gap extrapolation.

The 25 pinned v1 trajectories at Ny=40, alpha1=1, sample indices 0--24
are combined with the 75 remaining v2 samples for that case. All other
cases contain 100 v2 samples. Revisions are never pooled twice.

## Endpoint mode inventory

- The Ny=40 full-matrix singular vectors are now extracted separately by
  `extract_endpoint_singular_modes.py`. See [ENDPOINT_SINGULAR_MODES.md](ENDPOINT_SINGULAR_MODES.md)
  for array names, coordinate conventions, and the important distinction
  between whole-matrix singular vectors and occupied-empty tangent pair modes.
- Slot 17 does not save separate eigenvectors or singular vectors. It saves
  200 full normalized endpoint one-leg cocycles at Ny=40, alpha1=1,3,
  plus logarithmic scales and the canonical active-input indices. Their
  singular vectors can be computed from those matrices without rerunning
  the physical circuit. The other sizes in this campaign are gap-only.
- Earlier slot-09 **CPU replay v3** does save input/output occupied/empty
  singular vectors for the 16 slow tangent pairs, with spatial x-profiles,
  for both the full and late windows. The local verified subset has 100
  trajectories at hard Ny=24, alpha1=1; 100 at hard Ny=24, alpha1=3; and 79
  at hard Ny=28, alpha1=1. These are singular vectors, not eigenvectors of
  a generally nonnormal cocycle; they are not pooled into this figure.
- Slot-09 acquisition also saves endpoint occupied frames, which recover
  the endpoint covariance. An occupied frame is not a tangent-mode basis.
- The locally imported slot-18 purification outputs contain 100 samples
  at Ny=50 (50 each at alpha1=1.975 and 2). They contain a selected-eigenvector
  field, but all selections are empty: no mode satisfies that campaign's
  `abs(a)<=0.9` selection at the saved endpoint. This is not a live inventory
  of the entire Drive campaign.

`endpoint_mode_inventory.json` lists the individually checksummed older
mode-bearing files. `analysis_manifest.json` pins the plotted inputs and
outputs. The two CSV tables retain both case statistics and all individual
trajectory gaps, signed nearest rates, and null diagnostics.

## Reproduction

```bash
cd 00_WORKSPACE/CURRENT/experiment_review/hard_wall_tangent_gap_analysis
python -m unittest -v test_plot_endpoint_gaps.py
python plot_endpoint_gaps.py
```

Requires NumPy, Matplotlib, tqdm, LaTeX, CMU Sans Serif, and `pdftoppm`.
The 3.375-inch-wide vector PDF and 300-dpi PNG are saved under `figures/`.
