# Boundary-only record-free-energy CPU pilot

This local CPU campaign tests whether the unstable full-slab Zabalo-style
finite-size coefficient is dominated by bulk measurement-record entropy.  It
starts from the pure, half-filled lowest-half ground state of the hard-wall OW
flattened parent, retains the full `Nx=16` physical Hilbert space, and restricts
only the canonical measurement-center schedule.

The primary arm measures exactly the two walls at `x=4,12` for
`Ny=16,20,24,28,32`, 100 independent Born trajectories per size, and `T=4*Ny`.
A separate 25-sample thin-strip arm measures `x=3,4,5,11,12,13`.  Both use
`nshell=1`, `alpha_1=1`, `alpha_2=30`, hard support truncation, raster-y order,
perfect correction, complex128, and `classA_U1FGTN.run_markov_circuit`.

This is a modified physical circuit, not an algebraic subtraction from the old
full-slab records.  Its primary observable is the cumulative Born log
probability.  It does not reconstruct a many-body operator spectrum.

## Run and resume

```bash
python run_campaign.py --workers 8
python run_campaign.py --report-only
python analyze_campaign.py
python analyze_combined_window.py
```

One trajectory is one immutable resume unit.  A valid NPZ/completion-JSON pair
is checksummed and skipped on restart.  Independent trajectories run in a
process pool; each process uses one BLAS thread.  Results live under
`outputs/boundary_only_flattened_ground_nx16_ny16-32_s100_4ny_v1/`.

The runner saves cycle-resolved record weights, all realized measurement-event
log probabilities, packed outcomes, schedules, and compact frame diagnostics.
It never saves covariance or occupied-frame histories.

The original analysis uses the last-quarter window
`W3=[3*Ny,4*Ny]`.  The combined-window analysis is a separate estimator that
fits one slope per trajectory over `W23=[2*Ny,4*Ny]` and then bootstraps whole
trajectories.  The additional cycle points improve each trajectory's slope
estimate; they are never counted as independent ensemble samples.  Its outputs
use the `*_combined_window` suffix and do not replace the original analysis.

## Persistent launch

`launch_tmux.sh` starts the resumable campaign with the measured eight-worker
default in a named tmux session and
writes a plain-text log next to the outputs.  Re-running the launcher refuses to
create a duplicate live session.
