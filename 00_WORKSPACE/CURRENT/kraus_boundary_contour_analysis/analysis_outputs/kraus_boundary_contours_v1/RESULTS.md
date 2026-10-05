# Kraus boundary-contour results

All four checks completed.  The inputs remain separate rather than being pooled.

- The complete legacy covariance history contains 10 trajectories, 41 time slices each.  The endpoint mean wall weight of the first four soft modes is 0.9842.
- The independent hard/soft endpoint ensembles contain 600 trajectories.  Across their six construction/size cells, the first-four-mode mean wall weight lies between 0.8919 and 0.9815.
- The bundle-13 reconstruction covers 700 trajectories and 27,900 spectrum checkpoints.  The endpoint first-gap wall fraction lies between 0.9889 and 0.9972.
- The deterministic event replay reproduces the covariance to 0.000e+00; the support-contour sum rule closes to 1.137e-13.
- Fitting late-window ordered gaps to $\Delta_i=A_i/N_y$ gives $A_i/A_1=1.000, 1.369, 1.689, 2.087$.

## Interpretation

The low-lying spectrum is genuinely boundary-localized; this is not an artifact of one system size, wall construction, or ten-sample legacy file.  Additive gap contours are therefore meaningful.  Relative ordered-gap coefficients can be extracted without the extensive leading level.

The present production files still do **not** determine a boundary central charge.  Their saved covariances determine the spectral term, but not the event-resolved spatial contour of the record log probability.  The new replay proves that the missing contour can be recorded exactly.  A production central-charge analysis would need that event contour plus a matched bulk/reference subtraction.  Sector labels are also needed before assigning the ordered gaps to named boundary operators.
