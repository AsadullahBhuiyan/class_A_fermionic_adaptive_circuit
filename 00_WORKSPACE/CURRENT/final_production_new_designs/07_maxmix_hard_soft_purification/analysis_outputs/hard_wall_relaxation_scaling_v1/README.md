# Hard-wall contour relaxation

Run `python plot_hard_wall_relaxation_scaling.py` from the owning bundle to
reproduce the PDF, 300-dpi PNG, tables, and provenance summary. This uses the
existing verified hard-wall v2 ensemble only.

## Figure caption

Spatially resolved purification for the hard-wall circuit at Nx=20,
Ny=20,30,40, with 100 independent Born trajectories per size and total depth
T=4Ny. The active slab is initialized maximally mixed; the exterior modes are
prepared in product states. Dynamics use support-truncated walls at x=5,15,
nshell=1, alpha1=1, alpha2=30, raster-y measurement order, and perfect correction.
(a) The entropy contour is computed within each trajectory, summed over y to
give s_x, then averaged over trajectories and divided by Ny. Curves show the
two wall columns x=5,15 and the interior column x=10 versus t/Ny. Shading is
one sample SEM, std(ddof=1)/sqrt(100); no covariance averaging precedes the
entropy estimator. Color and marker identify Ny; line style identifies x.
(b) The elapsed relaxation time tau_x is the first crossing of
mean[s_x(Ny+tau_x)] = mean[s_x(Ny)]/e, with linear interpolation between saved
integer cycles. Points show the crossing of the ensemble mean, not the mean
of trajectory-wise crossing times. Error bars are one standard error computed
by deleting one complete trajectory at a time (jackknife), retaining temporal
correlation and baseline uncertainty. Colored lines are descriptive fits
tau_x=A_x Ny^z by unweighted least squares in logarithmic coordinates across
the three sizes. The gray dashed z=1 reference is normalized to the average
of the two measured wall times at Ny=30. No exponential fit or bootstrap is
used for these points.

## Interpretation and estimator limitations

Both walls retain substantially more entropy than the interior and their
measured relaxation times increase with circumference. The wall times have
comparable orders of magnitude to Ny. Three sizes do not establish an
asymptotic exponent, and the mean contour amplitudes need not collapse
exactly after division by Ny. Also, the reference time t0=Ny explicitly varies
with size: a scale-free aging curve can give tau proportional to t0 without
establishing an intrinsic finite-size relaxation time. Thus this operational
threshold alone cannot prove z=1 or distinguish power-law aging from an
exponential finite-size tail. Baseline choices t0/Ny=0.5,1,1.5,2 are saved in
`baseline_sensitivity.csv` for inspection. No smoothing or monotonic fit is
applied to the curves before locating the first crossing.

Interior thresholds are included in `relaxation_times.csv` as diagnostics;
panel (b) displays only the two wall columns, where residual entropy is
larger. The two walls and different times from the same trajectory are not
counted as independent samples.
