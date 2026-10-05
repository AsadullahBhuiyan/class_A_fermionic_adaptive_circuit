# Raster-y channel comparison

Requested 2026-09-14: rerun both wall choices with the raster-y site order
used by the production trajectory scripts. Run alpha_1=1 and 3 for each wall.
All prior random-order results remain unchanged in their original folders.

The canonical CPU and GPU routines agree on the order: raster_y advances y
at fixed x, then advances x. Within each cell the order is Ap, Am, Bp, Bm
(deplete A, fill A, deplete B, fill B). The technical report's earlier
fill-first and x-fast prose was a description error, not an engine difference.

Only the channel site schedule changes in this comparison. Keep Nx=20,
Ny=64, nshell=1, alpha_2=30, inclusive walls x=5,...,15, all slabs evolving,
full-system maximally mixed initialization, perfect correction, number
dephasing, complex128, and 128 cycles. There is no frozen exterior.
The continuous Lindblad runs have no site order and are not rerun.
This matches the trajectory campaigns' ordering, not their different pure
initialization or optional active-slab restriction.

Launch after checking the selected CPU cores are free:

```sh
python launch_raster_y_channels.py --launch --cpus 8,9,10,11
```

Each job uses the canonical classA_U1FGTN.run_markov_channel. Every saved
cycle word is checked against the exact raster-y word. Independent tmux
sessions, progress logs, source hashes, full endpoint and late-average
covariances, momentum spectra, and checksum-bound completion files use the
existing endpoint runner. Fresh outputs live under
`results/raster_y_channel_endpoints_v1_TIMESTAMP`.

The four 8x4 smoke cases completed successfully with verified result hashes
and all eight saved cycle words matching raster_y. The new report plot must
not be substituted or labeled raster-y until the full-size results complete
and their saved configurations, words, and result hashes are verified.

## Completed production comparison

All four jobs in `results/raster_y_channel_endpoints_v1_20260915T025739Z`
completed 128 cycles. `plot_raster_y_channel_overlay.py` verifies their
completion checksums, source hashes, embedded configurations, and every
saved raster-y word, and independently rediagonalizes the saved twirled
covariance blocks. The largest normalized Frobenius increment in the last
ten cycles is below 1.1e-16 for each case. This establishes numerical
stationarity at cycle boundaries, not stationarity within an elementary sweep.

New hard- and soft-wall overlays, spectra, captions, and diagnostics are in
`analysis_outputs/raster_y_channel_alpha1_vs3_ky/`. The report uses the new
hard-wall overlay; earlier random-order products remain unchanged.
For hard-wall alpha_1=1, the two central occupations at ky=0 are
0.54659608 and 0.54659724. The offset from half occupation thus persists
with fixed raster-y order and is not a remaining convergence error at this
numerical precision. This is a correlation spectrum of the exactly
Born-averaged state, not a trajectory-averaged nonlinear spectral observable.
