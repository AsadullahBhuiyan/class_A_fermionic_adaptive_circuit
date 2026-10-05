# Square 2Ny-cycle hard-wall channel sweep

16 cases: alpha_1=1,3; Nx=Ny=L=20,24,28,32,36,40,44,50.
Each case runs exactly 2Ny cycles, observing cycle zero and every cycle.
Full-system maximally mixed initialization, alpha_2=30, nshell=1,
inclusive hard support-truncated slab, all slabs active, periodic boundaries,
zero twist, X trial orbitals, complex128, perfect correction and measurement
dephasing. Canonical CPU `classA_U1FGTN.run_markov_channel`, raster-y
(y advances at fixed x), Ap,Am,Bp,Bm within each cell.

After the user requested launching without another geometry clarification,
we adopted the stated recommended proportional-wall convention:
`[L//4, (3*L)//4]`, inclusive. At L=50 these walls are [12,37]; this rounding
is intentional. The walls are not kept fixed at [5,15] as Nx changes.

One detached tmux queue waits for the preceding 60-cycle sweep's two workers
and 16 checksum-verified outputs. It then starts two single-core workers on
CPUs 0 and 7, one alpha each, leaving existing jobs untouched. A failed
predecessor stops the queue. Each worker uses one numerical-library thread.

```sh
python run_square.py launch --cpus 0,7
python run_square.py report --root results/RUN_STAMP
```

Per-case logs show ordered-product construction, eigensolver boundaries and
canonical cycle tqdm progress; worker logs show case progress. `queue.log`
and `queue_status.json` identify waiting/running/completed/failed states.
Once both workers succeed, plots and CSV/JSON tables are generated automatically.
Never launch overlapping workers on the same root.

The old Nx=20 spectral gaps are not reused at larger square sizes. Each case
first saves a fresh full-spectrum ordered-product calculation in `spectral/`
using the existing validated spectral runner. Explicit dynamics uses the
60-cycle campaign's passive observer with a changed horizon and geometry.
Save compact all-cycle observables, the full endpoint physical correlation
matrix and occupation spectrum. Completion records bind source/configuration
identity, byte count, SHA-256 and the same-geometry spectral reference.
Verified spectra and dynamics are independently reused after interruption;
an unfinished dynamics run restarts that case's 2Ny cycles.

No clipping, artificial positive gaps, trajectory sampling, decay fits,
continuous Lindblad evolution, or many-body gap inference is introduced.
The diagnostic 1e-13 numerical floor does not stop dynamics. Existing code,
campaigns and their results are preserved.

Launched 2026-09-29 in tmux session `square2Ny_20260929T193748Z`.
Output root: `results/20260929T193748Z/`. Initially waiting for
`../explicit_60cycle_v1/results/20260929T193023Z/` to finish.

## Launcher correction

The original run `20260929T193748Z` failed before all 16 case calculations:
the command-line check incorrectly required two allowed CPUs even in a
child that inherited a worker's single-core affinity. Its logs remain
unchanged. This is an orchestration regression, not a scientific failure.
The corrected launcher validates two CPUs only for launch/queue modes,
one assigned CPU for worker/case modes, and no CPU allocation for report.
Real subprocess regression tests reproduce the original error and now
pass under single-core inherited affinity. The complete three-file test
suite passes 14 tests. Scientific parameters and canonical sources are
unchanged; the relaunch uses a fresh timestamped result directory.

Corrected relaunch: `results/20260929T213855Z/`, tmux session
`square2Ny_20260929T213855Z`, CPUs 0 and 7. The preceding 60-cycle sweep
is already complete, so this queue starts immediately after verification.
