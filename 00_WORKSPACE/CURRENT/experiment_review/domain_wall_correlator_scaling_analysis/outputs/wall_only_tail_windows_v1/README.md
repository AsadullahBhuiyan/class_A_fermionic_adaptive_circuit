# Wall-only tail fits (not in the technical report)

Run `../../analyze_wall_tail_windows.py` to regenerate. All six matching alpha1=1
hard-wall ensembles are independently verified by the shared endpoint loader.
Only the actual wall columns x=5 and x=15 are fit, separately, with no x averaging.
The fit is log C = log A - beta log d, with free amplitude and exponent.

The primary window is 8..Ny/2. Lower cutoffs 2,5,8,10,12,15 are scanned without
changing the upper cutoff Ny/2. All invalid fits are retained with point counts;
at least four finite positive observations are required. No exponent or fit-quality
selection, bootstrap, SEM, or numerical-floor cut is applied.

At Ny=60, fitting the ensemble-mean curves on 8..30 gives beta_L=2.153726 and
beta_R=2.163358, with R-squared 0.999977 and 0.999985. On 5..30 the exponents are
2.150896 and 2.171420; on 15..30 they are 2.160284 and 2.151599. The long windows
reduce the left-right difference, but do not establish beta=2.

`ny60_wall_tail_fits` shows the 8..30 region in gray with dashed fitted lines.
Gray crosses are excluded short-distance points. `ny60_wall_window_sensitivity`
shows free-fit slopes against the lower cutoff, with the unconstrained beta=2
reference dashed and chosen cutoff 8 dotted. These are arithmetic ensemble means
followed by logs, not averages of sample exponents.

`ensemble_mean_fits.csv` contains all mean-curve fits. `sample_fits.csv` retains
each trajectory's separate fit. `sample_exponent_distributions.csv` gives mean,
median, sample standard deviation, and empirical percentiles, not uncertainty
on the mean. For 8..30, sample-wise means are 2.1910 (left) and 2.2101 (right),
with standard deviations 0.4007 and 0.4150. These differ from the mean-curve fits.
The sample-wise spread grows markedly with a later lower cutoff (about 0.87/0.90
at 15..30): fewer points and compressed chord range reduce individual-fit
stability despite the excellent ensemble-mean straight lines.

`plotted_curves.csv` and `provenance.json` record the plotted values and verified
source identities. Old figures and raw data are preserved. None of these wall-fit
diagnostics is inserted into the technical report.
