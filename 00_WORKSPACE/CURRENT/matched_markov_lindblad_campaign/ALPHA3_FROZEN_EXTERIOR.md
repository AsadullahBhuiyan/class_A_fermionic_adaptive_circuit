# Four alpha_1=3 channel/Lindblad endpoints

User-approved 2026-09-14: use the **new frozen-exterior hard-wall protocol**, not
the original full-system support-truncated channel/Lindblad campaign.

Historical protocol: the user subsequently requested full-system dissipation;
see `ALPHA3_FULL_SYSTEM_DISSIPATION.md`. Frozen results remain unchanged.

Run `python launch_alpha3_endpoints.py --launch --hard-exterior frozen --cpus 8,9,10,11` from this
directory after checking CPU utilization. This launches four separate tmux
processes: soft/hard exact channel and soft/hard Lindblad with number dephasing.
Each uses one physical CPU and one BLAS thread at lower scheduling priority.
Results and logs go to a new `results/alpha3_frozen_exterior_v1_TIMESTAMP/`
directory; `launch.json` records session names and commands. Old data are untouched.

## Scientific contract

The previously inspected occupation spectra determine the matched settings:
Nx=20, Ny=64, walls x=5,15 inclusive, nshell=1, alpha_1=3, alpha_2=30,
maximally mixed initialization, complex128, perfect correction, 128 cycles/time
units. The channel uses random site permutations and root schedule seed 20260814;
the Lindblad uses unit gain/loss/number-dephasing rates and RK4 dt=0.05. One
outcome-averaged evolution is run per case, not an S100 trajectory ensemble.

Soft walls evolve all 2,560 modes with untruncated OW support. Hard walls evolve
the 1,408-mode slab only, with truncated OW support. The original Nx=20 OW arrays
are constructed first, then restricted in physical rows and active-center
columns by `classA_U1FGTN.restrict_ow_dynamics_to_slab`. They are **not**
reconstructed on a smaller periodic lattice. The canonical channel/Lindblad
entry points perform every dynamics update. Reduced dimensions in internal
engine metadata describe the computational block, not a new physical geometry.

For maxmix initialization the exterior preparation is exactly independent
Bernoulli(1/2) occupation sampling, in the canonical (x,y,orbital) preparation
order. Seed 2026091403 selects the same frozen exterior for both hard-wall jobs.
The active slab stays maximally mixed at cycle zero. Its subsequent mean
evolution is independent of those exterior outcomes because coupling is zero.
The saved full hard-wall state is conditional on that one exterior record;
it is not an average over exterior preparations. The hard and soft site words
are not paired permutations because their active site sets differ.

## Products and interpretation

Each run retains full physical `G_final` and `G_late_cycle_average` (cycles
65..128), the active endpoint, the frozen exterior occupations and physical
index mapping, every-cycle charge and state distance, sparse occupation
checkpoints, channel schedules, and explicitly translation-twirled endpoint
ky spectra with x weights. No response kick is run. The stored full covariance
permits subsequent observables without rerunning evolution.

A random frozen exterior is not y-translation invariant. Its twirled occupation
spectrum can show fractional occupations even though that exterior is a pure
product state. Use the saved active block or spatial mode weights to separate
these exterior artifacts from slab physics. Likewise Gaussian entropy/variance
functions of the outcome-averaged covariance are not averaged nonlinear
trajectory observables.

Completed results have a checksummed `completion.json`; a failure has an explicit
status and no completion. These approximately minutes-scale local jobs restart
from cycle zero if interrupted; no intra-run checkpoint is claimed. The original
pre-clarification `campaign_config.alpha3_endpoints.v1.json` is not used by this
launcher (it describes the older non-frozen protocol).

## Validation

The small-system tests compare one reduced channel cycle with the full-space
product of the original slab-centered modes, and compare the reduced Lindblad
action with the full-space sum over the original modes. They verify zero
exterior generator rows/columns and reject restriction of soft-wall modes.
All 54 selected canonical channel/Lindblad tests passed, and all four end-to-end
Nx=8, Ny=4 smoke runs completed with valid endpoint physicality and saved outputs.
See `tests/test_frozen_exterior_mean_dynamics.py` at repository root.
