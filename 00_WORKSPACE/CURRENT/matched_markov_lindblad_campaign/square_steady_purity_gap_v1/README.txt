Square hard-wall steady-state occupation-gap sweep
================================================

14 independent deterministic cases: alpha_1=1,3; Nx=Ny=L=20,30,40,50,60,70,80.
Walls are floor(L/4), floor(3L/4), inclusive, matching earlier square spectral
sweeps. For L=30,50,70, retain that integer rounding rather than silently
shifting the geometry to enforce another centering convention.

Retain the fixed-width campaign's canonical CPU run_markov_channel dynamics:
alpha_2=30, nshell=1, hard-wall support truncation, all slabs active, periodic,
zero twist, X trial orbitals, complex128, maxmix initialization, perfect
correction and measurement dephasing, raster-y Ap/Am/Bp/Bm, 61 cycles.
No new channel-gap diagonalization or trajectory sampling is performed.

Reuse steady_purity_gap_v1.run_purity.collect and occupation_spectra unchanged.
At cycles 0,1,5,10,20,40,60,61 save full and sector occupation spectra and the
ky-resolved spectrum of the y-translation-twirled correlation matrix. Save
both endpoint correlation matrices (cycles 60 and 61), every-cycle charge
and absolute Frobenius increments. No dense covariance history is retained.

Acceptance: last five increments <1e-10; full and twirled purity gaps change
by <2e-10 between cycles 60 and 61. Nonconverged data remain explicitly marked
not_converged, not represented as stationary. Purity gap=min|1-2n|; the
distance to half filling is half this number. The twirl is postprocessing,
not assumed to be a fixed point of the raster channel.

Launch: python run_square.py launch --cpus 9,10
Resume: python run_square.py launch --root /absolute/existing/result/root --cpus 9,10
Report: python run_square.py report --root /absolute/existing/result/root
Attach: tmux attach -t SESSION_FROM_LAUNCH
The queue has two single-core workers, one per alpha, in ascending L order;
one child process per case bounds memory retention. Each case has a cycle
tqdm bar in its log. The queue shows case progress. Existing jobs are untouched.

Resume validates NPZ/completion-JSON pairs, config, source hashes, byte count,
and SHA-256. Only completed cases are skipped. An interrupted case restarts
from its maxmix initialization; there is no intra-case checkpoint. Large-L
cases may be lengthy: full dense endpoint matrices and eigensolvers are kept
to reproduce the earlier estimator exactly. L=80 has 12800 orbitals and a
complex128 matrix alone is 2.44 GiB; the two-worker process peak is higher.
This is a local CPU campaign, not a Colab bundle. Old outputs are unchanged.
