# Larger-system hard-wall correlator review

## What changed

The extra sizes resolve longer spatial separations and reveal a downward drift in the primary fitted exponent. They do not yet establish a window-independent exponent of exactly two. This is a static endpoint analysis of saved trajectories, not a new dynamics run.

The primary ensemble contains 600 trajectories: 100 each at Ny=24,28,32,40,50,60. All have Nx=20, alpha1=1, alpha2=30, n_shell=1, hard/support-terminated walls at x=5,15, raster-y order, pure half-filled initialization, perfect correction, complex128 and endpoint 2Ny. The independent separation range is 1..Ny/2, reaching 30; no longer transverse wall separation was created.

## Primary sample-wise exponents

Fit each trajectory first on 2 <= ry <= floor(Ny/4), using `log C = log A - beta log[(Ny/pi) sin(pi ry/Ny)]`, then take the sample mean. The critical squared-correlator guide is beta=2; it is not imposed.

| Ny | x average | left wall | right wall | two walls | two pairs |
|---:|---:|---:|---:|---:|---:|
| 24 | 2.2897 | 2.1766 | 2.2705 | 2.2157 | 2.2309 |
| 28 | 2.2925 | 2.1808 | 2.2776 | 2.2235 | 2.2354 |
| 32 | 2.2872 | 2.1735 | 2.2832 | 2.2229 | 2.2351 |
| 40 | 2.2431 | 2.1521 | 2.2417 | 2.1925 | 2.2024 |
| 50 | 2.2304 | 2.1597 | 2.2249 | 2.1887 | 2.1972 |
| 60 | 2.2108 | 2.1474 | 2.2156 | 2.1782 | 2.1843 |

The x-average exponent falls from 2.2872 at Ny32 to 2.2108 at Ny60. The two-wall value falls from 2.2229 to 2.1782. This extends the earlier observation of a wall-dominated algebraic sector, while making the finite-circumference drift visible in the wall-resolved data too.

## Wall averaging and left-right structure

At Ny60 the separate walls give 2.1474 and 2.2156; their difference is 0.0682, compared with 0.1098 at Ny32. It remains visible. This is a measured endpoint asymmetry; these data alone do not identify its cause.

Averaging the neighboring topological-side sites gives 2.1843, versus 2.1782 for the two exact wall columns. It therefore does not improve agreement with two in the primary window. The sample SDs at Ny60 are 0.0698 and 0.0698, respectively: essentially the same empirical spread. The averaging is performed on curves within a trajectory, not on already-fitted exponents.

## Does the longer tail settle the exponent?

No plateau at exactly two is demonstrated by this review. For Ny60 the x-average means are 2.2433 on 2..8, 2.2108 on 2..15, 2.2058 on 2..30, 2.1907 on 5..30, and 2.2555 on 15..30. The outer-tail window is more sensitive to sample variation and the compressed range of log chord near the antipode; extending a window is not automatically a cleaner asymptotic fit.

The C>1e-8 variant removes 0 points across the primary ensembles and seven observables in the outer-tail tests. Thus that amplitude cutoff does not explain the observed exponent drift. The separate short-distance bulk plots still mask values <=1e-8; the cutoff is not a measured floating-point error bound.

Nonlinear estimator order matters in the far tail. At Ny60, the x-average outer-tail exponent is 2.2555 when fitting each sample first, but 2.1569 when fitting the mean curve. Those are different questions, not interchangeable estimates.

## Static benchmark and interpretation

The hard-wall ground states were rebuilt with the canonical CPU OW constructor and the existing phi=1e-7 occupation-grid convention. They are static half-filled references; no flux-threading experiment is involved. Dense support remains a different Hamiltonian from n_shell=1.

Ny60 left wall, outer-tail fit: static n_shell=1 beta=2.0009, static dense beta=1.9999, dynamical mean-curve beta=2.1603.

Ny60 right wall, outer-tail fit: static n_shell=1 beta=2.0009, static dense beta=1.9999, dynamical mean-curve beta=2.1516.

The static reference approaches the exponent-two tail much more closely than the dynamical endpoint. The dynamical bulk also retains much larger small long-distance correlations than the ground-state bulk. The latter comparison becomes sensitive to very small denominators, so the ratio figure masks static values <=1e-14. This does not diagnose roundoff, incomplete relaxation, or a different stationary ensemble by itself. Finite-duration and finite-width explanations remain possibilities, not measured conclusions.

## Cohort cross-check and extrapolation

The old Ny32 representative selection and 400 numerical fits were reproduced before interpretation. Legacy streaming data are independent ensembles, not additional rows silently merged into the new datasets.

At Ny40, the new x-average primary exponent is 2.2431, versus 2.2525 in the legacy ensemble. The empirical trajectory IQRs overlap; this is descriptive agreement, not an SEM-based hypothesis test.

At Ny50, the new x-average primary exponent is 2.2304, versus 2.2312 in the legacy ensemble. The empirical trajectory IQRs overlap; this is descriptive agreement, not an SEM-based hypothesis test.

Descriptive extrapolated x-average intercepts are:

- all_six, linear in 1/Ny^1: 2.1602.

- all_six, linear in 1/Ny^2: 2.2086.

- Ny_ge_32, linear in 1/Ny^1: 2.1264.

- Ny_ge_32, linear in 1/Ny^2: 2.1834.

Their model/subset dependence is material. These are fixed-Nx=20 circumference extrapolations, not a two-dimensional thermodynamic limit. They do not justify replacing the measured exponents with two. The earlier qualitative claim of an algebraic wall contribution survives; the interpretation remains consistency in functional form with the critical prediction, not a precision confirmation of its exponent. The static/dynamical difference is explicitly a protocol/state comparison, not evidence of a plotting regression.

## What to put forward for review

Start with `xavg_scaling.pdf`, `wall_scaling.pdf`, `log_chord_Ny060.pdf`, and `window_sensitivity.pdf`. Use `dynamic_ground_state_Ny060.pdf` to expose the remaining benchmark differences. The paired-site plot is a useful stability check, not an improvement that should replace the wall estimator automatically.

`large_ny_correlator_review_atlas.pdf` contains 31 pages; `FIGURES.md` maps pages to files and captions. `size_summary.csv` gives primary means, SDs, quantiles and mean-curve fits. `trajectory_fits.csv` contains every sample/window/cutoff record, including invalid status and exclusions. `endpoint_curves.npz` preserves compact unmasked curves; no endpoint frame or covariance was loaded for these fits. Each figure has a plotted-series CSV; static arrays and diagnostics have their own files. All old figures, raw datasets and the manuscript are untouched.

All spread summaries describe trajectories themselves: SD, IQR and empirical percentiles. No bootstrap confidence intervals or SEM are used.
