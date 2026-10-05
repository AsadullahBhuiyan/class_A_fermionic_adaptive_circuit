# Repo Policy

## Where to Look First

Use this map before doing a repository-wide search. The directories under
`00_WORKSPACE/` are physical project directories, not aliases.

- `00_WORKSPACE/START_HERE/`: short orientation guide and recommended entry points.
- `00_WORKSPACE/CURRENT/`: current production bundles, experiment review, manuscript,
  validation campaigns, and recent diagnostics. Each active project remains separate.
- `00_WORKSPACE/COLAB/`: self-contained Colab packages. Keep each package's notebooks,
  scripts, source copies, metadata, and generated `gpu_data/`, `cpu_data/`, or
  `analysis_outputs/` together here; Colab outputs are active experiment data.
- `00_WORKSPACE/LARGE_RESULTS/`: standalone result-heavy campaigns that are not Colab
  packages.
- `00_WORKSPACE/LEGACY/`: older drafts, methods, analyses, and historical evidence that
  may still be needed for provenance or result reconciliation.
- `00_WORKSPACE/EXTERNAL/`: collaborator, reference, and distributed-computing code.
- `NOTES/`: generated documentation library and curated links to the documents most
  worth reading first. Canonical documents remain in their owning project directories.
- `PROJECT_ADMIN/`: this policy, layout/cleanup records, data catalog, archive manifests,
  and Git migration records.
- `src/`: canonical CPU/GPU dynamics implementations.
- `scripts/`: repository-wide runners, analysis utilities, and maintenance tools.
- `tests/`: maintained repository test suite.
- `notebooks/`: shared local notebooks not owned by a specific project package.
- `cache/` and `figs/`: shared historical covariance data and shared figures.

For an exact category tree and recency ordering, read
`PROJECT_ADMIN/WORKSPACE_LAYOUT.md` and `00_WORKSPACE/RECENCY_INDEX.md`.

## Dynamics Engine

Production runs of the adaptive Markov circuit must call the canonical dynamics
entry point:

- CPU path: `classA_U1FGTN.run_markov_circuit(...)`
- GPU path: `classA_U1FGTN_gpu.run_markov_circuit(...)`

Any GPU notebook or script that involves circuit simulation must use
`classA_U1FGTN_gpu.run_markov_circuit(...)`. Do not duplicate the Markov cycle
loop by calling private update helpers such as `_apply_grouped_site_updates(...)`
directly.

Any new GPU circuit-simulation method belongs in `classA_U1FGTN_gpu`, not as
private inline notebook/script logic. Notebooks and scripts should configure and
call the class, then compute observables from returned or saved data.

If a run should not save covariance histories, call the canonical engine with
`save=False` and consume the returned temporary history or final state in
memory, then discard it after computing the observable.

Every production output should record which canonical dynamics entry point was
used in its metadata.

## Colab Notebooks

All future final-production Colab redesigns and new standalone campaign contracts must live under the independent `00_WORKSPACE/CURRENT/final_production_new_designs/` parent. The preserved `00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/` tree is legacy-only and must not be a code or notebook dependency of redesigned campaigns. Register every redesigned bundle in the new parent's `bundle_layout.py` and `bundle_index.json`.

Colab notebooks are treated as separate runnable artifacts. When creating a new
GPU Colab notebook, upload the notebook and the required `src` file(s)
separately. The notebook should import/use `classA_U1FGTN_gpu` rather than
embedding private copies of GPU dynamics logic inline.

Whenever a Colab-ready notebook is requested, assume it will run on an A100 GPU
with 40 GB of GPU RAM. Size batches and memory usage accordingly.

Every new or substantially redesigned Colab notebook and companion Colab Python
runner must make long-running work visibly monitorable. Use clear `tqdm`
progress bars (prefer `tqdm.auto.tqdm` in notebooks) around the outer campaign,
trajectory, parameter-sweep, or batch loops that determine wall-clock progress.
Each bar must have an informative description and a meaningful total/unit when
known; nested bars should identify their level and avoid leaving unreadable
stacks. In addition, print concise, informative messages at major stages: the
resolved run configuration and workload, input/output locations, resume or
checkpoint status, important validation results or warnings, and an explicit
completion summary. Do not replace progress reporting with per-iteration print
spam, and do not add bars to trivial loops whose runtime is negligible. If the
main work is delegated to a helper, the notebook must either surface that
helper's progress bar or provide an equivalent bar at the notebook level.

### Default Colab execution and resume design

All future Colab campaigns must use the repository's legacy-proven execution
design by default. Keep the notebook and runner small, use deterministic units
of work, and resume at natural scientific boundaries. Do not turn Google Drive
into a transactional or distributed coordination system.

The standard bundle contains one notebook, one campaign runner, the canonical
GPU engine copy, and only the observer/helper modules required by that campaign.
Mount Google Drive once, copy the small executable bundle to local `/content`,
run code and transient scratch work locally, and use Drive only for durable
checkpoints and finished outputs. The notebook must expose the complete editable
campaign configuration in one place and print the resolved configuration,
workload, source paths, device, dtype, output directory, and resume inventory
before launching work.

Use a deterministic task table. A task should normally be one configuration,
one parameter point, or one immutable shard of at most five independent
trajectories. Task IDs and seeds must be stable across notebook restarts. The
outer notebook loop must use `tqdm` with a known task total and show completed,
skipped, pending, and failed counts. The expensive inner loop must expose an
engine, cycle-observer, trajectory, batch, or scan-point `tqdm` bar with the
correct `total` and, when resuming, the correct `initial` value. A console-only
heartbeat is permitted only when the canonical engine cannot expose meaningful
inner-loop progress; it must not maintain a long-lived Drive log handle.

Ordinary resumability is completion-based. Each task writes compact scientific
products atomically and writes a small completion JSON last. The completion JSON
must bind the task ID, configuration identity, seeds or global sample indices,
result filename, byte count, and SHA-256. On restart, skip a task only after the
result and completion JSON are both present and the declared filename, byte
count, checksum, and configuration identity match. Missing, partial, or
mismatched pairs are incomplete and must be rerun. Prefer one compact NPZ plus
one completion JSON per task; use a tar archive only when a task genuinely has
multiple products that must remain grouped.

Run each task in local `/content` scratch. Publish a checkpoint or result by
writing a temporary file, closing it, copying it to a temporary path in the
final Drive directory, reopening it through the mounted path to verify its byte
count and SHA-256, and then atomically replacing the stable filename. Publish
the completion JSON only after that readback succeeds. A failed Drive write or
readback must fail visibly and leave the task pending; it must never be converted
into completion by a warning or cached directory listing.

Configuration- or shard-level resume is sufficient when rerunning one task is
reasonably cheap. If one task can consume more than about one hour, or a material
fraction of a Colab session, it must additionally use one rolling scientific
checkpoint. The checkpoint must contain everything required for exact
continuation: completed cycle/point, native state (including occupied-frame
state and ranks when applicable), NumPy RNG state, Torch CPU and all CUDA RNG
states, partial observer accumulators, and the task/configuration identity.
Write the checkpoint atomically using a stable `checkpoint.npz` plus a small
`checkpoint.json`; do not use generation files or a pointer protocol. Restore
RNG state immediately before the next canonical dynamics call, initialize the
inner `tqdm` bar from the completed cycle/point, and remove the checkpoint only
after the final result and completion JSON have passed their readback checks.

The runner should follow this control flow unless the campaign documents a
scientific reason to differ:

```python
for task in tqdm(tasks, desc="campaign", unit="task"):
    if verified_complete(task):
        continue
    checkpoint = load_matching_checkpoint(task)
    result = run_or_resume(task, checkpoint=checkpoint)
    publish_result_atomically(result)
    write_completion_json_last(task, result)
```

Keep only safeguards that protect scientific correctness directly: canonical
engine use, explicit source path/version reporting, deterministic configuration
and seeds, CUDA/A100 and dtype checks, bounded local/Drive space checks, atomic
writes, final readback, checksums, and exact checkpoint restoration. By default,
do **not** add a second Drive API authorization path, server file-ID tracking,
remote-status subprocesses, parent/child verification layers, distributed
leases or lease elections, generation/pointer checkpoint protocols, migration
ledgers, orphan-repair state machines, persistent session dashboards, background
Drive logging, or multiple qualification/preflight workflows. Any exception
requires an explicit user request and a short design note identifying the
concrete failure mode that the simpler design cannot handle.

The last cell of every Colab-ready notebook must disconnect from the Colab
runtime after the run completes and then emit an explicit terminal completion
message.  Use this exact cell unless a notebook has a documented external
constraint:

```python
from google.colab import runtime
runtime.unassign()
print('done')
```

## Other:

### Prior-result failure reconciliation

- Before reporting that an observable, numerical experiment, or acceptance gate has
  failed, check whether the user or any repository artifact indicates that the same
  result was obtained previously. This check must search the entire repository and
  inspect the relevant scripts, notebooks, raw data, figures, metadata, and written
  claims; a filename search or figure-only comparison is not sufficient.
- When a prior success and a new failure conflict, compare at minimum the source version,
  Hamiltonian or dynamics entry point, geometry, state preparation and filling rule,
  injected perturbation, normalization, observable definition, fit window, and finite-size
  regime. Reproduce the protocol difference on a common input whenever that is
  computationally feasible.
- Classify the discrepancy explicitly as one of: code regression, protocol or estimator
  mismatch, finite-size or convergence effect, unsupported legacy claim, or genuine
  non-reproduction. Do not present the new failure as a physical conclusion until this
  classification is supported by saved diagnostics.
- Do not silently reinterpret a legacy result, change an acceptance threshold, or
  overwrite either artifact to make the conflict disappear. A corrected campaign must
  use a new versioned configuration and a new output directory, preserve the superseded
  run, and document the reason for the correction in its report and manifest.

- GPU notebooks/scripts that simulate circuits must use `classA_U1FGTN_gpu.run_markov_circuit(...)`.
- New GPU circuit-simulation methods must be added to `src/fgtn/classA_U1FGTN_gpu.py`, not written inline in notebooks/scripts.
- Keep every `classA_U1FGTN_gpu.py` outside `erroneous_gpu_stuff/` synchronized with `src/fgtn/classA_U1FGTN_gpu.py`.
- Colab GPU notebooks should be uploaded separately from the required `src` file(s).
- Colab-ready notebooks should assume an A100 GPU with 40 GB GPU RAM and must end with the repository-standard runtime-disconnect cell: `from google.colab import runtime`, `runtime.unassign()`, then `print('done')`.
- Unless otherwise specified, figures width should be 3.375, dpi=300, font should be "CMU Sans Serif", mathematical expressions should be typeset in latex using $$, tight layout, legible font size (preferably 8pt or larger.)
- Follow the plotting grammar used by
  `00_WORKSPACE/CURRENT/experiment_review/numerical_campaign_legacy_working.tex`
  for campaign and manuscript figures. Treat 3.375 inches as a single-column
  figure width and approximately 7.05 inches as a double-column compound-figure
  width; do not squeeze a multi-panel comparison into one column when labels or
  uncertainty marks become illegible. Save vector PDF and 300-dpi PNG versions.
  Use boxed axes, inward ticks, compact frameless legends, and panel letters just
  outside the upper-left axes corner. Use redundant color, marker, and line-style
  encodings (the established BPJ ordering is red triangles/dotted, green
  squares/dashed, and blue circles/solid), with black or gray dashed reference
  lines, so the comparison remains readable in grayscale. Keep axis labels,
  units, uncertainty definitions, and fit/reference targets explicit. A
  manuscript caption must state the trajectory count `S`, independent sampling
  unit, initialization, total trajectory or cycle window, estimator order,
  uncertainty type, and fit window whenever those items apply.
- In all LaTeX mathematics, typeset a many-body identity operator as
  `\hat{\mathbf{1}}` and a single-particle/matrix identity as `\mathbf{1}`
  (with optional dimension or space subscripts, e.g. `\mathbf{1}_N`). This
  combines the paper's convention that many-body operators carry hats with a
  form that renders reliably in the repository toolchain. Do not use
  `\mathbb{1}`, `\mathbb{I}`, `\mathbbm{1}`, or their unbraced variants.
- Acknowledge local changes manually made by a user. If you feel the need to question them or edit them, just ask before proceeding to edit the code
- When building notebook to characterize/analyze/plot repo data, use a consistent analysis-notebook structure by default:
  1. Start with a markdown title and short summary of what the notebook computes.
  2. State the basic theory/estimators used by the notebook before the first plot.
  3. Include a runtime/setup section. For local CPU notebooks, the top executable cell must have CPU allocation functionality such that I can specify a CPU range, unless the notebook is specifically designed for Google Colab.
  4. Include a data-loading/run-parameter section that reads metadata when available and renders the actual campaign/run parameters in the notebook output.
  5. Separate analysis stages with clear markdown headers and one or two sentences saying what the next plot/table is testing.
  6. Make sure that each output figure gets its own cell so that I can quickly edit plots on the fly. Each output cell should literally have the figure creation/plt commands in the immediate cell above the figure output, such that I can read and edit the code manually.
  7. End with a raw summary/diagnostics section that prints the fit dictionary, scalar rows, censor/null counts, or other sanity checks used by the plots.

### Mathematical write-ups and canonical paper notation

The canonical notation source for fermionic dynamics, topology, and adaptive
state preparation is Bhuiyan, Pan, and Jian, *Free-fermion dynamics with
measurements: Topological classification and adaptive preparation of
topological states*, Phys. Rev. Research **8**, 023147 (2026),
DOI `10.1103/nk38-ygyq`, including its appendixes. Whenever a new write-up uses
the same mathematical objects, or objects with the same mathematical role,
use the notation and terminology below rather than inventing replacements.
Apply the convention by semantic role, not merely by matching a symbol. If an
external format or an already established document creates a genuine symbol
collision, state the mapping explicitly at first use and then stay consistent.
Expand mEO, sTM, OW, and POVM at first use in every standalone document.

- **General typography and state spaces:** Put hats on many-body operators
  (`\hat H`, `\hat\rho`, `\hat V`, `\hat K`, `\hat c`, `\hat d`, `\hat\chi`,
  `\hat N`, `\hat\gamma`) and leave single-particle matrices, scalar functions,
  symmetry-class labels, and superoperators unhatted. Use double kets and bras
  for many-body wavefunctions, e.g.
  `\lvert\mathrm{TS}\rangle\!\rangle` and
  `\langle\!\langle\mathrm{TS}\rvert`, and ordinary kets and bras for
  single-particle states, e.g. `\lvert W_{\boldsymbol r,\mu,n}\rangle` and
  `\lvert\psi_n(\boldsymbol k)\rangle`. Use `\operatorname{Tr}` for a
  many-body/Fock-space trace and `\operatorname{tr}` for a single-particle
  matrix trace. Use `\dagger`, `*`, and `\mathsf T` for adjoint, complex
  conjugation, and transpose, respectively.
- **Dimensions, labels, and subsystems:** Write `d` for spatial dimension and
  `D=d+1` for spacetime dimension; use forms such as `2+1d` for spacetime and
  `2d` for space. Use `\boldsymbol r` for a unit-cell coordinate,
  `\boldsymbol k` for momentum, `\mu` for a physical orbital/mode label,
  `\nu` (often `A,B`) for the trial-orbital/OW-family label, and `n` or `\pm`
  for a band label, with `+` the upper band and `-` the lower band. Denote the
  physical, ancillary, and reference systems by `P`, `A`, and `R`; use
  subscripts `\mathrm{ph}`, `\mathrm a`, `\mathrm{tot}`, and `\mathrm{ss}`
  on their density matrices when needed. Use `N_{\mathrm{uc}}` for the number
  of unit cells and `n_{\mathrm{shell}}` for an OW truncation range.
- **Fermions and Gaussian states:** Use `\hat\psi_i` for a generic complex
  fermion, `\hat c_{\boldsymbol r,\mu}` for a physical lattice fermion,
  `\hat d_{\boldsymbol r,\mu}` for an ancillary fermion, and
  `\hat\gamma_i` for a Majorana fermion. State
  `\{\hat\psi_i^\dagger,\hat\psi_j\}=\delta_{ij}` and
  `\{\hat\gamma_i,\hat\gamma_j\}=2\delta_{ij}` when the algebra is first
  needed. Use *free-fermion* and *Gaussian* interchangeably as in the paper.
  A charge-conserving Gaussian state is written in the form
  `\hat\rho\propto\exp[-\lambda\sum_{ij}h_{ij}\hat\psi_i^\dagger\hat\psi_j]`,
  and its correlation matrix is
  `G_{ij}=\operatorname{Tr}(\hat\rho\,\hat\psi_i^\dagger\hat\psi_j)`.
- **Measurements and trajectories:** Denote outcomes by `m`, Kraus operators
  by `\hat K_m`, and their nonnegative weights by `w_m`. Write the POVM
  condition as
  `\sum_m w_m\hat K_m^\dagger\hat K_m=\hat{\mathbf{1}}` and the Born-rule
  probability as
  `p_m=w_m\langle\!\langle\psi\rvert\hat K_m^\dagger\hat K_m
  \lvert\psi\rangle\!\rangle`.
  Use `\boldsymbol m=(m_1,m_2,\ldots)` for a full trajectory record and
  `\hat V_{\boldsymbol m}` for its many-body evolution operator (mEO). Denote
  the mEO and single-particle transfer-matrix ensembles by
  `\mathcal M_{\mathrm{mEO}}=\{\hat V_{\boldsymbol m}\}_{\boldsymbol m}` and
  `\mathcal M_{\mathrm{sTM}}`, respectively. Do not describe an ensemble of
  operators proportional to unitaries as a measurement; call it random
  unitary evolution.
- **mEOs and sTMs:** For charge-conserving Gaussian dynamics, use
  `\hat V=\exp[-\sum_{ij}M_{ij}\hat\psi_i^\dagger\hat\psi_j]` and define its
  single-particle transfer matrix (sTM) `t=e^M` through
  `\hat V\hat\psi_i\hat V^{-1}=\sum_jt_{ij}\hat\psi_j`. Use lowercase `t`
  for an sTM, `\mathcal G` for its classical Lie group, `\mathfrak g` for the Lie
  algebra, and `M=\ln t\in\mathfrak g` for a generator. Reserve `G` with
  indices or arguments for a correlation matrix.
- **Symmetry classification:** Call the two Altland--Zirnbauer labels the mEO
  class `K_{\mathrm{mEO}}` and the sTM class `K_{\mathrm{sTM}}`; never call
  them simply "the symmetry class" when both are in play. Denote many-body
  time-reversal, particle-hole, and chiral symmetry actions by
  `\hat T`, `\hat C`, and `\hat S`, with first-quantized matrices
  `U_T`, `U_C`, and `U_S`. Use `\hat N_F` for total fermion number and
  `(-1)^{\hat N_F}` for fermion parity. Denote Pauli matrices by
  `\sigma^{x,y,z}` and the symplectic form by `\Omega\equiv i\sigma^y`.
  A symmetry of the dynamics must hold for every mEO/trajectory, not merely
  after trajectory averaging.
- **Class correspondence and POVM convention:** Use the paper's ordered map
  `K_{\mathrm{mEO}}\mapsto K_{\mathrm{sTM}}`:
  `A\mapsto AIII`, `AIII\mapsto A`, `AI\mapsto BDI`, `BDI\mapsto D`,
  `D\mapsto DIII`, `DIII\mapsto AII`, `AII\mapsto CII`, `CII\mapsto C`,
  `C\mapsto CI`, and `CI\mapsto AI`. State that postselection-free Gaussian
  POVMs are admissible exactly for
  `K_{\mathrm{mEO}}\in\{A,AI,BDI,D\}`; the other six mEO classes require
  postselection in the Gaussian limit, though symmetry-compatible interacting
  measurements may evade that restriction.
- **Continuum POVMs and Cartan-decomposition arguments:** When these appendix
  objects are needed, write a Gaussian Kraus operator as `\hat K(M)` with
  `M\in\mathfrak g`, its nonnegative weight density as `\omega(M)`, and the
  continuum POVM condition as
  `\int dM\,\omega(M)\hat K^\dagger(M)\hat K(M)=\hat{\mathbf{1}}`. Use
  `H` for the Hermitian single-particle generator of
  `\hat K^\dagger(M)\hat K(M)`, `\lambda_\alpha(M)` for its
  log-singular-value spectrum, `\mathcal U` for a maximal compact subgroup,
  `\mathfrak a` for a maximal Abelian subspace, and `D` for the real diagonal
  matrix in the Cartan form `H=U^\dagger A(D)U`. Use
  `\lvert\mathrm{vac}\rangle\!\rangle` for the many-body Fock vacuum. Do not
  reuse `H` for a many-body Hamiltonian in the same argument; the latter must
  remain `\hat H`.
- **Topological states and OW stabilizers:** Use
  `\lvert\mathrm{TS}\rangle\!\rangle` and `\hat H_{\mathrm{TS}}` for a generic
  target topological state and parent Hamiltonian, and
  `\lvert\mathrm{CI}\rangle\!\rangle` and `\hat H_{\mathrm{CI}}` for the Chern
  insulator example. Call the localized nonorthogonal construction the
  *overcomplete Wannier (OW) basis*. Denote an OW state and wavefunction by
  `\lvert W_{\boldsymbol r,\mu,n}\rangle` and
  `W_{\boldsymbol r,\mu,n}(\boldsymbol r',\mu')`, its fermion mode by
  `\hat\chi_{\boldsymbol r,\mu,n}`, and its number stabilizer by
  `\hat N_{\boldsymbol r,\mu,n}\equiv
  \hat\chi_{\boldsymbol r,\mu,n}^\dagger\hat\chi_{\boldsymbol r,\mu,n}`.
  Write occupied-band stabilizers as `\hat N` and empty-band stabilizers as
  `\hat{\mathbf{1}}-\hat N`. Explicitly note that distinct OW stabilizers need
  not commute and that the OW states are overcomplete rather than an
  orthonormal Wannier basis.
- **Chern-insulator construction:** Use `\alpha` for the parent-Hamiltonian
  tuning parameter, `\hat c(\boldsymbol k)` for the two-component momentum
  fermion, `\boldsymbol n(\boldsymbol k)\cdot\boldsymbol\sigma` for the
  two-band Bloch Hamiltonian, `P_\pm(\boldsymbol k)` for band projectors,
  `\lvert\psi_\pm(\boldsymbol k)\rangle` for Bloch states, and
  `\tau_\nu` (`\nu=A,B`) for trial orbitals. Use `f_{\nu,n}(\boldsymbol k)`
  for the OW form factors. Preserve the paper's distinction between a form
  factor zero for one OW family and the nonzero summed rate obtained from an
  overcomplete collection of families.
- **Adaptive protocol:** Denote the fermionic swap by
  `\operatorname{fSWAP}(\hat c,\hat d)` and use the paper's fill-lower-band,
  deplete-upper-band description. A cycle consists of OW-number measurements
  with conditional fSWAP feedforward, followed by incoherent particle
  redistribution and occupation measurement in the ancillary system. Reserve
  `t` for continuous effective time or cycle count only when it cannot be
  confused with the sTM; otherwise say "cycle number" explicitly. Use
  `T_{\mathrm{conv}}` for the convergence timescale and `O(1)` for
  system-size-independent depth or cycle count, always specifying which one.
- **Channels and Lindbladians:** Use calligraphic `\mathcal E` for quantum
  channels, a tilde (`\widetilde{\mathcal E}`) after tracing out the ancillary
  layer, `\mathrm{Id}` for the identity channel, and calligraphic `\mathcal L`
  for Lindbladian generators. Use the labels
  `\mathcal L_{\mathrm{cycle}}`, `\mathcal L_{\mathrm{gain}}`,
  `\mathcal L_{\mathrm{loss}}`, and `\mathcal L_{\mathrm{dephas.}}`, and write
  the standard dissipator as
  `\mathcal D[\hat L](\hat\rho)=\hat L\hat\rho\hat L^\dagger-
  \tfrac12\{\hat L^\dagger\hat L,\hat\rho\}`. Use `\bar n_{\mathrm a}` for
  the local ancillary occupation in the Markovian approximation. An ordered
  product of channels means composition, and this must be stated when first
  used. In momentum-space relaxation analyses, use `g_\pm(\boldsymbol k,t)`
  for band occupations, `\gamma_\pm(\boldsymbol k)=
  \sum_\nu|f_{\nu,\pm}(\boldsymbol k)|^2` for dissipation rates, and
  `\delta_\pm=\min_{\boldsymbol k}\gamma_\pm(\boldsymbol k)` for dissipation
  gaps.
- **Trajectory-resolved versus trajectory-averaged observables:** Use `G` for
  a single-trajectory correlation matrix, an overline for the Born-rule
  trajectory average, and `\overline G` for the trajectory-averaged
  correlation matrix. For a nonlinear functional, keep the order visible:
  `\overline{C_G}` is the averaged trajectory-resolved Chern number, whereas
  `C_{\overline G}` is computed from the averaged correlation matrix and is
  generally different. Use `C_G(\boldsymbol r)` for the squared two-point
  correlator, `I_{a,b}=S_a+S_b-S_{a\cup b}` for bipartite mutual information,
  `\Delta` for the correlation-matrix spectral gap around `1/2`,
  `C(\boldsymbol r)` for the local Chern marker, and `s(\boldsymbol r)` for
  the entanglement contour. Denote spectral flattening by `\widetilde G` and
  its Chern number by `C_{\widetilde G}`. Never silently substitute an
  averaged state into a nonlinear trajectory-resolved observable.

### Manuscript figure typography (mandatory for future edits)

For `00_WORKSPACE/CURRENT/manuscript_Overleaf/` and all figures prepared for
that manuscript, match the manuscript's Computer Modern text and mathematics.
This rule overrides the general CMU Sans Serif plotting default above for this
manuscript. Do not retrospectively alter unrelated campaigns or historical assets.

- Use `figures/new_figure/sources/manuscript_typography.py` in every active
  figure renderer: configure the style before creating artists, prepare the
  figure before layout, and record typography before saving. Both ordinary
  text and mathematical labels must use LaTeX. Missing dependencies must fail
  explicitly; do not silently substitute mathtext or system fonts.
- Sizes are measured at the final manuscript inclusion width, not merely the
  source canvas: 9 pt axis labels and normal-weight `(a)`, `(b)`, ... panel
  letters; 8 pt ticks, legends, and numerical annotations. Use 10--11 pt
  prominent schematic labels where space permits, with supporting labels at
  least 8 pt. Natural mathematical superscripts/subscripts are exempt.
- Keep schematic exceptions in the shared style. Compensate for inclusion
  scaling, including reduced-width schematics. Recheck the recorded RevTeX
  column/text widths when changing the manuscript class, preamble, or layout.
- Use black ordinary text; preserve meaningful colored labels and contrasting
  labels on dark backgrounds. Fix crowding through placement and spacing, not
  by shrinking text below the standard. Preserve scientific content and
  established panel arrangements and aspect ratios wherever possible.
- Deliver embedded Computer Modern/AMS fonts in vector PDFs and 300-dpi PNG
  previews. Run the bundle typography/data verifier and inspect every changed
  figure at manuscript size. Update notes, previews, and the manifest together.

### Standalone LaTeX document format

Write up LaTeX documents in RevTeX with high mathematical rigour/detail and the presentation and narrative style of a sophisticated theoretical physicist. Unless a requested target format requires otherwise, standalone theory notes must use the repository document shell:
  ```tex
  \documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}
  \usepackage{amsmath,amssymb,amsthm,mathtools,bm}
  \usepackage{booktabs}
  \usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
  \usepackage{microtype}
  ```
  This shell defines the default one-column RevTeX margins and default RevTeX type size/typeface; do not override them with `geometry`, custom margin settings, or font packages unless the document has a specific external requirement. Use a RevTeX title block with author and date, an abstract, `\maketitle`, and `\tableofcontents`, matching `00_WORKSPACE/COLAB/colab_regularized_choi_transfer_matrix/docs/regularized_choi_covariance_contractions.tex`. By default set `\setcounter{tocdepth}{2}` before the table of contents, and use `\texorpdfstring` for mathematical notation in displayed headings so linked contents and PDF bookmarks compile cleanly.
- For scripts/notebook that are to be excecuted locally, use the cpu-based classA_U1FGTN to call methods. Do not use the analogous gpu class.
- The top cell of every notebook must have CPU allocation functionality such that I can specifiy a range of cpus to allocate to the notebook at hand, unless the notebook is specifically designed for google colab
