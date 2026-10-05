# Max-mix many-body Lyapunov A100 pilot

This is a standalone, simple Colab campaign. It runs the canonical
`classA_U1FGTN_gpu.run_markov_circuit` engine from a local `/content` copy and
uses Google Drive only for finished task products.

## Locked campaign

- `Nx=20`; `Ny=20,22,24,26,28,30,36,40`
- 100 independent Born trajectories per circumference
- 20 deterministic tasks of five trajectories per circumference
- `T=2*Ny`
- hard/support-truncated wall with `DW_loc=[5,15]`
- `nshell=1`, `alpha_1=1`, `alpha_2=30`
- maximally mixed initialization, `raster_y`, active-slab measurements
- perfect correction, no postselection
- covariance representation in complex128 on an A100 40 GB

The active transfer sector contains `22*Ny` modes. The Born-conditioned
exterior product modes are excluded from the spectrum and from the record
normalization used for

```text
log Z = (22*Ny)*log(2) + omega .
```

The observer saves the record log probability every cycle. At
`{0,4,8,...,2*Ny} union {Ny,3*Ny/2,2*Ny}`, it saves the active occupation
spectrum, entropy, charge variance, leading 64 `log(sigma^2)` values, 16 soft
mode profiles and wall weights, and numerical residuals. Exact occupation caps
at zero or one remain infinite-gap caps; they are never replaced by a finite
clipping value. The v2 observer consistently treats CUDA-eigensolver boundary
roundoff up to `1e-9` as the corresponding exact zero/one cap and records the
unsnapped `occupation_bound_residual`; larger violations still abort.
Covariance histories and eigenvectors are not saved.

The v1 pilot was superseded before producing a completed task. Its observer
used a contradictory pair of tolerances: a spectrum could pass the `1e-9`
physicality guard and then fail the `1e-12` cap check. An A100 run encountered
exactly that condition at cycle 28. The corrected v2 campaign therefore writes
to a new output directory and does not mix contracts.

## Run and resume

Open `run_maxmix_manybody_lyapunov_pilot.ipynb` in an A100 Colab runtime and
run the cells from top to bottom. The editable configuration is all in one
cell. The defaults run or resume the full campaign; set `MAX_NEW_TASKS=1` for a
single worst-case `Ny=40` timing calibration, or set `REPORT_ONLY=True` to
inspect the checksum-verified inventory without computing.

The notebook shows an outer 160-task `tqdm` bar and the canonical engine's
inner cycle bar. It prints before projector construction and before every task,
so a long setup never looks like a silent hang.

Each task is computed under `/content`, then published as one compressed NPZ
and one completion JSON. The Drive temporary NPZ is reopened and checked for
byte count and SHA-256 before its stable rename; the completion JSON is written
last. A restart skips only pairs that reverify completely. An interruption can
therefore lose the current five-trajectory task, never an earlier verified
task. Run only one copy of this notebook at a time.

There is deliberately no Drive API, lease, archive, dashboard, migration,
remote-status, or intra-task checkpoint machinery.

## Analyze

The completed v2 data have now been analyzed locally. The analysis:

- reconstructs the saved many-body levels from the occupations and `log Z`;
- fits every trajectory before averaging;
- compares `[Ny,3*Ny/2]` with `[3*Ny/2,2*Ny]`;
- bootstraps whole trajectories 2,000 times;
- compares the same-late-window slopes `d omega/dt` and `lambda_0`, while
  retaining the endpoint ratio only as a finite-depth diagnostic;
- fits the `1/Ny^2` finite-size form, a `1/Ny^4` sensitivity, and an
  omit-`Ny=20` sensitivity;
- gates `alpha*c_eff`, `alpha*x_i`, and `x_i/c_eff` instead of forcing them;
- writes CSV/JSON plus a 7.05-inch PDF and 300-dpi PNG.

All 160 tasks and 800 trajectories pass historical checksum, source-identity,
level-reconstruction, exact-cap, cycle-mask, and numerical-residual checks.
Same-window record and leading-level slopes agree to better than 0.3% for every
size, and the first four soft modes carry about 95--97% of their weight near the
walls. Every size nevertheless fails at least one middle/late stability gate.
The fitted leading `1/Ny^2` coefficient has the wrong sign, includes zero, and
is strongly fit-window dependent. No `c_eff`, absolute operator dimension, or
`x_i/c_eff` claim is released from v2.

The complete CSV/JSON and double-column PDF/PNG products are under
`analysis_outputs/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/`.
The recommended next run is a fresh `T=4*Ny`, `S=100` depth study at
`Ny=20,30,40`; it is not launched here. Because v2 did not save final
covariance/RNG state, that study must rerun from `t=0`.

The completed v2 campaign is also preserved locally under
`gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/`. That directory
contains all 160 checksum-verified result/completion pairs, the locked v2
configuration, and its Drive-import provenance manifest.

## Runtime expectation

Archived A100 runs with the same hard-wall geometry took about 23 minutes for
five `Nx=20, Ny=40`, `T=80` trajectories before this campaign's repeated
spectral work. A reasonable planning range is 25--50 total A100-hours. The
runner starts from `Ny=40` and prints a measured projection after every task.
