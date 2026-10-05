# Matched perfect-correction channel / continuous-Lindblad campaign

This is the corrected, versioned campaign bundle for comparing the deterministic
trajectory-averaged perfect-correction Markov channel with its matched infinitesimal
continuous Lindblad construction. It does not overwrite or relabel the earlier
`mean_channel_lindblad_cpu_v2_hybrid_response` archive.

## Locked design

- $N_x=20$, $N_y=32,48,64$, walls fixed at $x=5,15$, periodic $y$.
- $(\alpha_{\rm run,in},\alpha_{\rm run,out})=(1,30)$ for the endpoint, plus the
  prior eight-point inner-alpha scan at $N_y=64$.
- $n_{\rm shell}=1,2$ for both dynamics; full-frame controls only for the continuous
  Lindblad arm.
- DW-support truncation on/off, maximally mixed initialization, and evolution through
  $2N_y$.
- The production Markov channel uses full measurement dephasing and perfect correction.
  The literal unit-step `decoh=False` covariance map is excluded because it is not
  positivity preserving. The continuous arm has dephasing off/on with unit
  perfect-correction gain, loss, and (when enabled) number-dephasing coefficients.
- Random schedules use shared root seed `20260814`. Main-grid matched cases use one
  schedule seed. A separate canonical $N_y=64,n_{\rm shell}=1$, DW-truncated channel
  product retains ten independent schedule samples.
- The late state is the temporal average over cycles $N_y+1,\ldots,2N_y$. The spatial
  sliding estimator is its exact $y$-translation twirl.
- Density-kick response is restricted to 22 declared endpoint cases: finite
  $n_{\rm shell}=1$ at $N_y=32,48,64$ with DW-support truncation on/off for all three
  dynamics arms, plus full-frame continuous cases at $N_y=64$ with truncation on/off.
  Alpha scans, $n_{\rm shell}=2$, and the grouped $S=10$ schedule control do not run
  response.

The expanded grid contains 105 case specifications: 27 full-dephasing channel cases,
39 continuous no-dephasing cases, and 39 continuous full-dephasing cases. The extra
channel case is the ten-schedule control.

## Engine wiring and launch gate

`dynamics_adapters.py` now connects the discrete arm to
`classA_U1FGTN.run_markov_channel` with explicit walls and seeded schedules, and the
continuous arm to the matched solver built from the same canonical OW arrays. The
runner records both adapter and canonical entry points. Do not launch production until
the small-system algebra, response, timestep, schema, and physicality smoke gates pass.

The reference file `/home/abhuiyan/Fermionic_Lindbladian/CI_Lindblad_DW.py` is not a
production engine: its wall geometry and mode construction differ. It is used only for
small-system algebra checks.

Validate and inspect the fully expanded grid without creating a run directory:

```bash
python run_campaign.py --validate-only
python run_campaign.py --list-cases
python validate_scaffold.py
```

After adapter integration, run the smoke matrix first. Production should use at most
eight workers pinned to physical cores `0-7` while the current host job remains active;
logical CPUs `64-111` are sibling threads of occupied cores and must not be treated as
free. The runner creates a new timestamped directory automatically and refuses to
overwrite any case:

```bash
taskset -c 0-7 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python run_campaign.py
```

Analyze only after the run-level `_SUCCESS` marker exists:

```bash
python analyze_campaign.py --run-root /absolute/path/to/the/completed/run
```

To create the archived, executed notebook copy from the same immutable run:

```bash
MATCHED_RUN_ROOT=/absolute/path/to/the/completed/run \
  jupyter nbconvert --to notebook --execute analyze_matched_campaign.ipynb \
  --output /absolute/path/to/the/completed/run/analysis/analyze_matched_campaign.executed.ipynb
```

The analyzer writes a new analysis directory, verifies every case checksum, treats the
sample axis as the independent unit, and produces PDF/PNG spectra, four-estimator alpha
scans, response heatmaps, velocity size checks, cycle-convergence panels,
schedule-control diagnostics, and direct channel--continuous difference tables. For
the ordered channel, displayed $k_y$ spectra are always those of the explicitly twirled
late-cycle state. The twirl deletes $k_y\ne k_y'$ coherences; it is neither a symmetry
of the random ordered map nor a trajectory/schedule average. The analyzer also plots
the raw late-state natural occupations in ascending order, without twirling, and reports
the untwirled translation residual. The raw rank spectrum has no exact $k_y$ label.

## Data contract

Every declared cheap scalar observable has logical shape `(sample, cycle)` at the exact
cycle coordinate `0,1,...,2*Ny`. Eigenspectrum-based estimators are evaluated only at
the explicit checkpoint cycles in the configuration, never by diagonalizing a
(2560\times2560) matrix after every cycle. Dense matrices are checkpoint-resolved only:
`G_final` and `G_late_cycle_average` retain the sample axis. Terminal products retain
global occupations, exact-twirl $k_y$ occupations and $x$-weights, wall branches,
global-mode momentum weights, response data for the declared 22-case subset, and
leading relaxation spectra.

Each case is committed atomically as a directory containing `observables.npz`,
`metadata.json`, and `_SUCCESS` or `_FAILED`. Run and case manifests contain SHA-256
hashes, source identity, the exact dynamics entry point, schedule seeds, wall locations,
and git state.

## Documents

The completed main note and supplement use the notation of Bhuiyan--Pan--Jian.  The
eight production figure families must be present under `docs/figures/`; the TeX sources
include them directly and therefore fail loudly if a figure is missing:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error matched_channel_lindblad_main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error matched_channel_lindblad_supplement.tex
```

The populated main note is six pages and remains below the nine-page ceiling; the
four-page supplement contains the cycle-convergence and ten-schedule diagnostics. The
previous gain/loss-only archive appears only as a provenance baseline:
its stationary no-dephasing result is reusable because its generator is exactly one-half
of the new no-dephasing perfect-correction generator, while its clock-dependent response
and its auxiliary Strang channel are not reused.
