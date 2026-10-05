# Combined manuscript figure notes

## Combined schematic and current manuscript numbering (October 5, 2026)

Figure 1 now combines the single-cell domain-wall geometry with the hard-wall support construction. The standalone hard-wall close-up is no longer included in the manuscript; both it and the soft/hard-wall alternative remain reproducible supplementary assets. The manuscript contains twelve figures: 1 in Section II, 2–10 in Section III, and A1–A2 in the appendices.

**Numbering convention for this provenance document:** existing section titles and numerical discussion below retain the stable bundle IDs used in filenames. They are not current manuscript figure numbers. The following mapping and the [ordered index](README.md) give the current numbering.

| Current manuscript figure | Stable asset stem | Contents |
|---|---|---|
| 1 | `Figure_01_schematic` | Domain-wall adaptive circuit |
| 2 | `Figure_03_bulk_topology` | Bulk topology |
| 3 | `Figure_04_purification` | Slow purification |
| 4 | `Figure_05_correlations` | Correlation functions |
| 5 | `Figure_06_entropy_charge` | Entropy and charge fluctuations |
| 6 | `Figure_07_entanglement_spectrum` | Occupation spectrum, energy spectrum, and mode count |
| 7 | `Figure_08_central_charge` | Central-charge convergence and size dependence |
| 8 | `Figure_09_wall_entropy` | Entropy carried by each wall |
| 9 | `Figure_10_modular_evolution` | Modular evolution |
| 10 | `Figure_11_mean_channel` | Trajectory-averaged dynamics and channel gap |
| A1 | `Figure_A01_ow_truncation` | Truncated OW modes |
| A2 | `Figure_A02_mutual_information` | Entropy contours and antipodal mutual information |
| Standalone (former 2) | `Figure_02_hard_wall` | Hard-wall support truncation |
| Alternative | `Figure_02_alt_soft_and_hard_walls` | Soft and hard walls |

The geometry is a conceptual 16-by-24 lattice drawing with equal site spacing, not a new simulated system. The central slab occupies half the horizontal cell. Four nominal 3-by-3 supports are shown at distinct heights: trivial bulk, slab-side interface, slab bulk, and exterior-side interface. The interface supports retain two columns by three rows; discarded support is not shaded. Salmon denotes exterior modes and darker blue denotes slab modes. Dots identify their centers, and labels show $\hat{\mathcal N}_{\boldsymbol r,\nu,\sigma}(\alpha_{\boldsymbol r})$. The vertical circumference is 1.5 times the horizontal dimension. Circuit-time layers and interface-coordinate labels have been removed. The flowchart retains the Born measurement, conditional fSWAP, target occupations, and fresh ancilla; only its layout and operator label have changed.

The schematic has no numerical samples, fit, or uncertainty. Periodicity and renormalization after clipping are specified in the caption; the drawing does not depict wavefunction amplitudes. The new caption text is blue. All scientific curves, data, fit windows, and original assets are preserved. Reproduce the combined vector PDF and 300-dpi preview with `python sources/plot_schematic.py`; reproduce the retained close-ups with `python sources/plot_wall_schematics.py`. The latter uses its historical 0.8-column width for typography even though it is no longer included in the manuscript.



## Typography standard (October 5, 2026)

All fourteen current figure versions match the manuscript’s Computer Modern text and mathematics, rendered with LaTeX. This section supersedes earlier typography descriptions below; those describe prior renderings. Numerical data, fits, uncertainties, color scales, panel order, and manuscript prose are unchanged by this pass.

Sizes are measured at the actual manuscript inclusion width: 9 pt axis labels and normal-weight panel letters, 8 pt ticks, legends, numerical annotations, and inset labels. Schematic descriptions use 10 pt, with 11 pt phase labels in Figure 2; the compact decision formula uses 9 pt and supporting yes/no/ancilla labels use 8 pt. Figure 2’s source fonts compensate for its 0.8-column inclusion. Natural math scripts use LaTeX’s script sizes. The measured RevTeX widths are 246 TeX pt per column and 510 TeX pt for the full text block.

The shared renderer configuration is [manuscript_typography.py](sources/manuscript_typography.py); the persistent repository rule is in PROJECT_ADMIN/REPO_POLICY.md. Every future renderer must configure the shared style, prepare its text before layout, and record typography before saving. Missing LaTeX dependencies cause an explicit error rather than font substitution. Required tools include LaTeX, AMS/bm packages, type1cm/type1ec (cm-super), dvipng, and Poppler.

Legends and annotations were repositioned or wrapped where larger text needed room. Figure A1 retains its canvas dimensions and 2×2 aspect ratio; its width legend shares a single `w` heading, its fit legend is above panel (c), and its retained-weight annotation is split across two lines. The wall-entropy fit annotations use separate lines; schematic box text is wrapped; the MI geometry inset is wider to accommodate 8 pt labels. No curves or fit values are changed.

Reproduce using the commands in [README](README.md), then rebuild the overview and run `python sources/verify_bundle.py --record` followed by `python sources/verify_bundle.py`. The verifier checks actual embedded PDF font families, recorded final-print sizes, panel labels, clipping, and preserved numerical input checksums. Per-figure typography receipts are under `data/typography/`.

This is the complete scientific and reproduction note for the ordered manuscript figure bundle. It consolidates the former individual Markdown notes without removing their protocol distinctions or source information. This working bundle is integrated into the revised [manuscript](../../manuscript.tex), with current numbering given in the mapping above. The standalone hard-wall and alternative soft/hard schematics are retained but are not included. Original figure assets, the earlier `restructured/` bundle, and campaign data are preserved; the manuscript and bibliography have been deliberately revised. No circuit simulations were run.

Figures 3 and 7 use **3×1** vertical layouts; Figure 8 uses **2×1** and Figure 11 uses **4×1**, all at 3.375-inch column width. Figure 7 compares normalized occupation and windowed-energy distributions for both $\alpha_1=1,3$ at the approved $N_y=32$, $A_y=16$, pooling all 32 cut origins. Its raw mean mode counts, fit window, and uncertainties are preserved. Figure 11 adds the fixed-width channel-gap scan and inverse-length fits below its original mean-state observables. Presentation-only edits harmonize outcomes and averaged quantities in Figures 1, 4, 6, 9, and 10. Figure A2 now combines the latest all-origin entropy contours with the existing mutual-information sweep in a 2×1 column-width layout. All retained numerical curves, fits, and uncertainties are preserved.

[Ordered figure index](README.md) · [Overview](overview.png) · [Verification](validation.json)

## Contents

- [Figure 1 — Domain-wall adaptive circuit](#figure-01-schematic)
- [Figure 2 — Hard-wall support truncation](#figure-02-hard-wall)
- [Figure 2 alt — Soft and hard walls](#figure-02-alt-soft-and-hard-walls)
- [Figure 3 — Bulk topology](#figure-03-bulk-topology)
- [Figure 4 — Slow purification](#figure-04-purification)
- [Figure 5 — Correlation functions](#figure-05-correlations)
- [Figure 6 — Entropy and charge fluctuations](#figure-06-entropy-charge)
- [Figure 7 — Occupation spectrum, energy spectrum, and mode count](#figure-07-entanglement-spectrum)
- [Separate comparison — Normalized energy within the spectral window](#normalized-energy-comparison)
- [What the area-law window modes represent](#area-law-window-modes)
- [Separate panel C — Fraction of subsystem modes](#normalized-mode-fraction)
- [Figure 8 — Central-charge convergence and size dependence](#figure-08-central-charge)
- [Figure 9 — Entropy carried by each wall](#figure-09-wall-entropy)
- [Figure 10 — Modular evolution](#figure-10-modular-evolution)
- [Figure 11 — Trajectory-averaged dynamics](#figure-11-mean-channel)
- [Channel-gap proof and sector distinction](#channel-gap-proof)
- [Figure A1 — Truncated OW modes](#figure-a01-ow-truncation)
- [Figure A2 — Antipodal mutual information](#figure-a02-mutual-information)
- [Figure 7 and Section V of Eisler–Peschel](#figure-7-section-v)

## Shared reading and reproduction conventions

Each numerical note identifies its own independent sampling unit, measurement region, initialization, evolution window, averaging order, normalization, exclusions, fit window, and uncertainty definition. The ensembles are not interchangeable. In particular: purification A/B/D use full-system measurements, purification C uses a separate slab-only sweep; wall entropy uses two-column windows; central-charge panels use separate ensembles and different error definitions; translated entanglement cuts are correlated within trajectories.

All plotting commands below run from this bundle directory unless stated otherwise. Modified-figure scripts use the compact saved inputs under `data/`; they do not invoke dynamics. All fourteen versions now use dedicated renderers. Original source assets remain checksum-verified and unchanged; restoring an old copy would discard the requested typography updates. The ordered index lists dependencies and commands for regenerating the overview and checking the bundle.


### Figure typography

Panel letters use plain, regular-weight CMU Sans Serif at 9 pt. Numerical plot labels use CMU Sans Serif with Computer Modern mathematics and black text; PDF fonts are embedded as TrueType instead of Type 3 glyphs. This removes the inconsistent Times-like labels in Figures 6 and 9 and the separate TeX text-font path in Figure 7. The correlation labels were already black; their PDF used a different Type 3 embedding, which has now been replaced to improve consistent rendering across viewers. Figure 7(c)'s fit annotation is placed in the open lower-left region beneath the fit line. These changes do not alter data, fits, or uncertainty definitions.

### Notation used in the integrated manuscript

The two-point matrix is $G_{ij}=\operatorname{Tr}(\hat\rho\,\hat c_i^\dagger\hat c_j)$ and the centered matrix is $G_c=2G-\mathbf1$. Occupations are $\nu_a\in[0,1]$, centered occupations are $\lambda_a=2\nu_a-1$, and finite-time purification rates are $\gamma_a(T)$. The squared two-point diagnostic remains $C_G$; it is distinct from the disk Chern number $\mathcal C_G$ and the local marker $C(\boldsymbol r)$. For the topology formulas, the occupied projector in the creation-wavefunction basis is $P=G^{\mathsf T}$. This is a notation mapping of the saved computation, with no change of Chern sign or data.

An overline means a Born trajectory average. For subsystem observables it additionally includes the average over translated $y_0$ **within each trajectory first**. Nonlinear observables are formed before either average. Relative row coordinates are $\delta y=(y-y_0)\bmod N_y$. SEMs use the independent trajectories, never the correlated translated cuts. Exceptions are specified below: e.g. the bulk disk uses ten random centers; the deterministic averaged channel sums outcomes analytically; the central-charge coefficient is fitted to averaged entropy. 

We use one chord-length convention throughout: $D(\ell)=(N_y/\pi)\sin(\pi\ell/N_y)$, with the circumference supplied by the curve's geometry, not by a subscript in the axis label. Figures 5(a,b) and 11(b) use $\log D(r_y)$; Figure 5(c) uses $\log[D(r_y)/D(N_y/2)]$; Figures 6 and 9 use $\log[D(A_y)/D(A_y^\star)]$, where $A_y^\star=\lfloor N_y/2\rfloor$; Figure 7(c) uses $\log D(A_y)$. These are label changes only: absolute and reference-normalized coordinates remain distinct, and every input value, fit, cutoff, and uncertainty is preserved. Historical source column names such as `delta_log_sine_chord` are retained for provenance.

Measurement outcomes are upright $\mathrm m$, with the record $\mathbf m$. In the text $S=S_1$, $s=s_1$, and $c=c_1$ denote the von Neumann quantities.

New and revised manuscript text is blue; sparse review notes are magenta `[GPT: ...]`. The baseline manuscript and original-asset checksums are preserved in `notes/manuscript_revision/baseline/`. Numerical provenance distinguishes the ensembles even when their decoupled hard-wall interior dynamics coincide.


---

<a id="figure-01-schematic"></a>

## Figure 1 — Domain-wall adaptive circuit

[Vector PDF](Figure_01_schematic.pdf) · [300-dpi preview](Figure_01_schematic.png)

[PDF](Figure_01_schematic.pdf) · [PNG](Figure_01_schematic.png)

**Contents.** Panel (a) shows one portrait lattice with a central topological slab, trivial exterior, two interfaces, and four local occupation measurements. Complete bulk supports contain 3×3 unit cells; the two interface supports are clipped to their centers’ regions. Panel (b) shows the Born-rule occupation measurement and conditional fermionic swap with a fresh ancilla: fill a lower-band overcomplete Wannier (OW) mode or empty an upper-band mode when its measured occupation disagrees with the target.

**Protocol.** The paper's geometry is periodic in both spatial directions; the circuit parameter is α₁ in the slab and α₂ outside. The ancilla target occupation is 1 for the lower band and 0 for the upper band. After the correction, the next OW mode is measured. The drawing is conceptual, not a literal system-size or depth specification.

**Data and analysis.** No trajectories, initialization, numerical averaging, fit, or uncertainty estimate enters this schematic. Its colors and arrows identify regions and operations; they are not quantitative data.

**Source and reproduction.** Regenerated from the original [drawing code](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/build_domain_wall_circuit_figure.py), with upright measurement outcome $\mathrm m$. Run `python sources/plot_schematic.py`. The active geometry was redesigned to combine the former Figures 1(a) and 2; the measurement protocol is unchanged. This native vector drawing has no numerical inputs. Source hashes are recorded in [notation_updates_provenance.json](data/notation_updates_provenance.json).


---

<a id="figure-02-hard-wall"></a>

## Figure 2 — Hard-wall support truncation

[Vector PDF](Figure_02_hard_wall.pdf) · [300-dpi preview](Figure_02_hard_wall.png)

[PDF](Figure_02_hard_wall.pdf) · [PNG](Figure_02_hard_wall.png)

**Contents.** Two overcomplete Wannier (OW) modes are centered on opposite sides of a domain wall. Gray and blue regions carry α₂ and α₁ respectively. Colored dots mark the mode centers, and the shaded rectangles illustrate their support after truncation at the interface.

**Protocol.** Construct each OW mode using the parent parameter at its center, remove support across the interface, and renormalize the surviving mode before measuring its occupation. Thus measurements centered on one side do not act on sites on the other side. The support rectangles illustrate the construction rather than the amplitude profile of a computed wavefunction.

**Data and analysis.** This is a conceptual schematic. No sampled data, initialization, cycle window, averaging, uncertainty, or fit applies.

**Source and reproduction.** Regenerated from an unchanged copy of the original [drawing code](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/build_ow_overlap_truncation_figure.py), changing typography only. Run `python sources/plot_wall_schematics.py` (requires `pypdf`). The original `v2` PDF is retained under `data/original_assets/`; verification found zero changed pixels outside the old/new text bounds, preserving all diagram geometry and colors. See [validation](data/wall_schematics/validation.json).


---

<a id="figure-02-alt-soft-and-hard-walls"></a>

## Figure 2 alternative — Soft and hard domain walls

[Vector PDF](Figure_02_alt_soft_and_hard_walls.pdf) · [300-dpi preview](Figure_02_alt_soft_and_hard_walls.png)

[PDF](Figure_02_alt_soft_and_hard_walls.pdf) · [PNG](Figure_02_alt_soft_and_hard_walls.png)

**Contents.** The left panel shows overlapping overcomplete Wannier (OW) support across a soft interface. The right panel shows support truncated at a hard interface. Dots identify the mode centers; the dashed line identifies the interface.

**Protocol.** Both constructions use different parent parameters on the two sides. The soft construction permits an OW mode to reach across the interface; the hard construction removes that support and renormalizes the mode. The drawing represents support, not a numerical amplitude or density.

**Data and analysis.** No trajectories, initialization, cycle count, estimator, uncertainty, or fit applies to this conceptual comparison.

**Source and reproduction.** The original attachment is retained in [data/original_assets](data/original_assets/ow_overlap_truncation_schematic.pdf). The delivered version replaces only its six text blocks with matching typography; all 317 non-text vector operations are preserved, and pixel checks found no change outside text bounds. Run `python sources/plot_wall_schematics.py` (requires `pypdf`). See [validation](data/wall_schematics/validation.json).


---

<a id="figure-03-bulk-topology"></a>

## Figure 3 — Bulk topology in the hard domain-wall geometry

[Vector PDF](Figure_03_bulk_topology.pdf) · [300-dpi preview](Figure_03_bulk_topology.png)

**Files:** `Figure_03_bulk_topology.pdf` (vector) and `.png` (300 dpi), 3.375 × 7.0 inches. Three vertically stacked panels (3×1) combine a new geometry drawing with the complete September 30 bulk-validation result. No dynamics was rerun. Panel (b) is taller to separate its inset from the curves; the marker map is enlarged, with a centered colorbar shorter than the map.

### Panels

- **(a) Geometry and estimator.** A 30 × 30 unit-cell lattice with trivial/topological/trivial regions labelled $\alpha_2,\alpha_1,\alpha_2$. The canonical integer interfaces are $x_L=8$ and $x_R=22$. A disk centered at $(15,15)$ with $R=6=0.2L$ is partitioned into three counterclockwise sectors: $A=[0,2\pi/3)$, $B=[2\pi/3,4\pi/3)$, $C=[4\pi/3,2\pi)$. One dot denotes one unit cell with two orbitals; both orbitals belong to the same sector. The drawn $y_0=15$ is illustrative: actual measurements use independently sampled transverse centers, with periodic minimum-image coordinates.
- **(b) Chern convergence.** $|\overline{\mathcal C_G}-1|$ against cycles 0–40 for $L=20,30,40$; inset shows the same $\overline{\mathcal C_G}$ on a linear scale. Curves, markers, line styles, and one-SEM shading retain the source data. The order is the absolute deviation of the ensemble mean, **not** the ensemble mean absolute deviation. The interval $[\overline{\mathcal C_G}-\mathrm{SEM},\overline{\mathcal C_G}+\mathrm{SEM}]$ is transformed through $|x-1|$; a zero lower endpoint is clipped to $10^{-6}$ only for logarithmic display. No fitting is applied.
- **(c) Local Chern marker.** The trajectory-averaged marker for $L=30$ at cycle 40, from all 100 endpoint projectors; orbitals are summed before averaging trajectories. Dashed lines mark the interfaces. The original `RdBu_r` scale is preserved with zero-centered separate linear ranges spanning the full data range, $[-13.8929122546,1.31778476234]$. There is no smoothing, clipping, or $\tanh$ transformation. Ordinary position coordinates on the periodic sample retain the large seam contribution. This is a position-commutator marker, not a spatial map of the finite-radius disk estimator in (b).

### Data and circuit protocol

There are **$S=100$ independent Born-rule trajectories at each of $L=20,30,40$**, with exactly 40 cycles for every size. Initial states are random pure states at half filling. The canonical GPU entry point is `classA_U1FGTN_gpu.run_markov_circuit`; it performs Born-conditioned exterior preparation before cycle zero. The geometry has periodic boundaries, hard support truncation of overcomplete Wannier (OW) modes, $n_{\mathrm{shell}}=1$, $\alpha_1=1$, $\alpha_2=30$, slab-only measurements in raster-$y$ order, and perfect correction with no postselection. The stored representation uses complex128 occupied frames. This is not the full-layer mixed-state purification protocol.

At each cycle, each trajectory draws ten distinct integer $y_0$ centers uniformly without replacement, using an observer RNG independent of the dynamics RNG. The other disk parameters are $x_0=L/2$, $R=0.2L$. Centers are averaged **within each trajectory first**, then the 100 trajectory values are averaged; SEM uses the sample standard deviation (`ddof=1`) divided by $\sqrt{100}$. The centers are not treated as independent trajectories. The sweep varies slab width, disk radius, and transverse size together. The interfaces for $L=20,30,40$ are $(5,15),(8,22),(10,30)$, respectively. All measurement disks remain inside the central slab.

### Estimators

With $P=G^{\mathsf T}$ the per-trajectory occupied projector in the creation-wavefunction basis, the disk estimator is

$$\mathcal C_G=\operatorname{Re}\!\left\{12\pi i\left[\operatorname{tr}(P_{CA}P_{AB}P_{BC})-\operatorname{tr}(P_{AC}P_{CB}P_{BA})\right]\right\}.$$

The native frame stores $G=VV^\dagger$; the topology observer contracts its transpose, $P=(VV^\dagger)^{\mathsf T}$. The local marker follows Bianco and Resta, [*Mapping topological order in coordinate space*, Phys. Rev. B **84**, 241106(R) (2011)](https://doi.org/10.1103/PhysRevB.84.241106), Eq. (9). Our overall sign is opposite to their convention, matching the occupied-band Chern convention in the manuscript (their footnote 10 explicitly discusses the sign difference from Kitaev). With unit-cell area one and both orbitals summed, it reads

$$C(\boldsymbol r)=\sum_{\mu=1}^{2}\operatorname{Re}\!\left[2\pi i\,(PXPYP-PYPXP)_{\boldsymbol r\mu,\boldsymbol r\mu}\right],$$

with ordinary finite-sample position matrices $X,Y$. The full finite marker sums to zero by the commutator trace identity; that identity does not make the central bulk Chern number zero. Neither nonlinear estimator is applied to an ensemble-averaged projector.

At cycle 40 the disk means and SEMs are:

| $L$ | $\overline{\mathcal C_G}$ | SEM |
| --- | ---: | ---: |
| 20 | 0.9975744098782162 | 0.0004702004718192011 |
| 30 | 0.9998569779041192 | 0.00007633294430813792 |
| 40 | 0.9999856796793450 | 0.0000036164261997793174 |

### Sources, reproduction, and checks

The owning campaign is `00_WORKSPACE/CURRENT/final_production_new_designs/27_square_hard_wall_random_center_chern`, revision `square_hard_wall_random_center_chern_l20-30-40_s100_t40_r0p2_v1`, root seed `2026092801`. The exact plot source is its `plot_convergence_and_marker.py`; the original products are under `analysis_outputs/convergence_and_endpoint_marker_S100/`. This complete dataset supersedes the older partial $L=40,S=20$ snapshot. It is distinct from the manuscript's old uniform no-domain-wall validation dataset.

The portable inputs in `data/bulk_topology/` are `plot_data.npz`, `cycles.csv`, `endpoint_marker.csv`, `source_summary.json`, and `source_caption.txt`. `provenance.json` records absolute and repository-relative original paths, SHA-256 hashes and byte counts of copied inputs and source scripts, the original scientific source identity, and receipts for all 25 raw production batches. Large endpoint frames are not duplicated in this figure bundle. No original input is modified.

From this figure bundle, run:

```bash
python sources/plot_bulk_topology.py
```

The renderer needs Python, NumPy, and Matplotlib; CMU Sans Serif is used with Computer Modern LaTeX math notation. It resolves data relative to itself and needs no source-campaign checkout. It verifies copied hashes; checks all 123 cycle mean/SEM pairs and 900 marker values against the original CSV files; reconstructs the marker mean and SEM from the 100 cached per-trajectory maps; and checks that displayed curves and image values equal the cache exactly. It writes `data/bulk_topology/validation.json` with source/output hashes and numerical results. The originating marker calculation agreed with the canonical CPU implementation to $4.31\times10^{-13}$, and its checked finite-disk contraction agreed with the saved observer to $1.45\times10^{-15}$.


---

<a id="figure-04-purification"></a>

## Figure 04 — Purification and wall-localized slow modes

[Vector PDF](Figure_04_purification.pdf) · [300-dpi preview](Figure_04_purification.png)

The four panels preserve the original `purification_ny30_full_measurement_4x1.pdf` values and fits. The regenerated vector figure overlines the trajectory means and uses a 3.375 × 6.8 inch layout.

**Protocol.** Hard/support-truncated domain walls at $x=5,15$, $N_x=20$,
overcomplete Wannier (OW) range $n_{\mathrm{shell}}=1$, $\alpha_2=30$,
maximally mixed initialization, perfect correction, and raster-y ordering.
Panels (a,b,d) use full-system measurements at $N_y=30$, $T=60=2N_y$,
with 100 independent Born trajectories per parameter: production campaign 21
for $\alpha_1=1$, and campaign 22 for $\alpha_1=3$.

**Panels and estimators.** (a) Mean total von Neumann entropy divided by
$N_y$, comparing $\alpha_1=1,3$. (b) At $\alpha_1=1$, the entropy
contour summed over y and divided by $N_y$, separately at wall columns
$x=5,15$ and trivial-slab columns $x=2,18$. Entropies and contours are
computed per trajectory before averaging. Shading is ordinary trajectory SEM;
cycle zero is excluded only from the logarithmic display. No entropy fit is
included. (d) At $T=60$, one unique minimum-absolute-rate mode is selected
and normalized per trajectory; its orbital-summed probability density is then
averaged. The plotted rates obey
$\gamma_j=\log[(1-\nu_j)/\nu_j]/(2T)$.

**The distinct panel (c) protocol.** This retained panel uses campaign 13's
slab-only measurements, $\alpha_1=1$, 100 trajectories per size, and
$N_y=20,24,30,36,44,56,60$, each evaluated at $T=2N_y$. It plots the
trajectory mean and SEM of
$\Delta=\min_j|\log[(1-\nu_j)/\nu_j]|/(2T)$.
The SEM-weighted log-space fit $A N_y^{-z}$ uses all seven sizes and gives
$z=1.0788\pm0.0540$. This is not a size sweep of the full-system protocol
in (a,b,d); the ensembles are not pooled.

**Numerical qualifications.** The $\alpha_1=3$ curve retains its original
observations through cycle 30, then uses spectral clipping at the handoff and
subsequent cycle ends. Entropy occupations are clipped to
$[10^{-12},1-10^{-12}]$; near-zero tails reflect this estimator floor.

**Source and reproduction.** Original producer:
[make_figure.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/purification_full_measurement_ny30/make_figure.py).
The source caption, analysis manifest, and gap summary are preserved under
`data/preserved_protocols/purification/`. Run `python sources/plot_purification.py`; the portable `data/purification/` inputs contain saved entropy/contour means and SEMs, the slow-mode density, and all seven gap points. `notation_validation.json` checks their reconstruction against the original source; no new dynamics or eigensolve is required. The full-system versus slab-only measurement distinction is retained here for provenance; the manuscript states the different initializations without exposing implementation flags.


---

<a id="figure-05-correlations"></a>

## Figure 5 — Algebraic correlations on the domain walls

[Vector PDF](Figure_05_correlations.pdf) · [300-dpi preview](Figure_05_correlations.png)

This revises manuscript Figure 3 by removing only the inward-neighbor columns
`x=6,14` from panel B. The manuscript's logarithmic chord coordinates, panel A/C
data, curve styles, plotting cutoff, and collapse fit are preserved.

### Panels and data

- **A:** Full-x average of the squared one-body correlation at `Ny=60`, comparing
  slab parameters `alpha1=1` and `3`.
- **B:** Column-resolved correlation at `alpha1=1`, `Ny=60`: walls `x=5,15` and
  slab center `x=10`. These retain their original blue-circle, orange-downward-
  triangle, and gray-diamond encodings respectively.
- **C:** Antipodally anchored full-x correlation for `Ny=24,28,32,40,50,60`,
  with the original common power-law fit.

Each ensemble contains **100 independent trajectories** on `Nx=20` unit cells,
with two orbitals per cell and periodic x/y boundaries. Data are endpoint
observations after `T=2Ny` cycles (48–120 cycles). The small sizes come from
campaign 08; `Ny=40,50,60` from campaign 14; the `alpha1=3`, `Ny=60` control
comes from campaign 15. The trajectories are not regenerated for this figure.

### Preparation and protocol

The hard-wall slab lies between `x=5` and `15`, with `alpha2=30` outside and
`alpha1=1` inside (or `3` for the control). The overcomplete Wannier (OW) support
has `nshell=1`, is cut at each wall, and is renormalized. Preparation uses the
campaign's random pure half-filled default initialization, followed by
Born-conditioned onsite preparation of the exterior before cycle zero.
Measurements then act on the slab (`meas_slab_only=True`), using `raster_y`
order and perfect occupation correction, without postselection. Historical
production uses `classA_U1FGTN_gpu.run_markov_circuit` in complex128 arithmetic.

### Estimator, fit, and uncertainty

For each trajectory the plotted column observable is

$$
C_G(x,r_y)=\frac{1}{2N_y}\sum_{y,\mu,\nu}
\left|G_{(x,y,\mu),(x,y+r_y,\nu)}\right|^2,
\qquad C_G^{\mathrm{av}}(r_y)=\frac1{N_x}\sum_x C_G(x,r_y).
$$

The periodic y-origin and orbital sums occur within each trajectory; then
trajectory observables are averaged arithmetically, **before taking logarithms**.
This is not a squared correlation computed from an averaged matrix. The chord is
`D(r)=(Ny/pi) sin(pi r/Ny)`, with natural logarithms throughout. Panels A/B
display `r=1,...,Ny/2` only where the mean exceeds `1e-8`; no plotted values are
replaced by that cutoff. Hidden values remain in the bundled data.

Panel C divides the ensemble mean at each separation by the ensemble mean at
`Ny/2`, then takes its logarithm. It is not the mean of trajectory-wise ratios.
The through-origin joint fit uses **8 <= r_y <= Ny/2**, giving each size equal
total weight, with `beta=2.1903330477862717` and `R0²=0.9997073485391841`.
Light/darker gray show the union/intersection of the size-specific fit windows.
No sampling error bars or confidence bands are displayed, and no uncertainty on
beta is claimed. The preserved fit-window sensitivity table documents systematic
window dependence; this figure does not add a new fit.

### Reproduction and provenance

Run `python sources/plot_correlations.py` from this bundle (or use the script's
absolute path from any directory); dependencies are NumPy and Matplotlib.
The renderer reads only local bundled data and writes the PDF, 300-dpi PNG,
plotted-data CSV, and validation JSON. `--output-dir` supports an isolated remake.

`data/correlations/compact_curves.npz` preserves every trajectory/column/separation.
`original_plotted_data.csv` and `original_summary.json` are unmodified exports of
`correlator_summary_3x1_v2_r8_half`; the latter records original data, receipts,
generation-source paths, and their SHA-256 hashes. `import_manifest.json` binds
these copied files and the original manuscript PDF to hashes. `validation.json`
checks exact preservation of A/C table rows, B columns `[5,10,15]`, reconstruction
of archived means, and independent recovery of the unchanged fitted exponent.


---

<a id="figure-06-entropy-charge"></a>

## Figure 6 — Entropy and charge-fluctuation scaling

[Vector PDF](Figure_06_entropy_charge.pdf) · [300-dpi preview](Figure_06_entropy_charge.png)

[PDF](Figure_06_entropy_charge.pdf) · [PNG](Figure_06_entropy_charge.png)

**Contents.** Former Figure 4(a,b), now a standalone figure: anchored mean von Neumann entropy and subsystem charge variance versus anchored log chord length. Display excludes Aᵧ=1; all fit inputs are unchanged.

**Data and protocol.** Campaign 05, hard-wall endpoint ensemble: Nₓ=20, Nᵧ=30,35,40,45,50,55,60; 100 independent random-pure trajectories per size; α₁=1, α₂=30, n_shell=1, perfect correction, raster-y order, periodic geometry, endpoint T=2Nᵧ. Circuit measurements act only on the slab (`meas_slab_only=True`); Born-conditioned onsite measurements prepare the exterior before cycle zero. The saved shards explicitly record `cycle_zero_semantics=after_born_conditioned_exterior_preparation`. The measured region differs from the observable's subsystem: each entropy/charge subsystem contains all x, both orbitals, and Aᵧ consecutive y rows. Periodic strip positions are averaged within each trajectory before averaging trajectories. The seven sizes comprise separate ensembles.

**Analysis.** For each observable X, subtract its value at Aᵧ*=floor(Nᵧ/2) and use x=log[D(Aᵧ)/D(Aᵧ*)]. Fit ΔX=m x through the fixed origin over 8≤Aᵧ≤floor(Nᵧ/2), assigning equal total weight to each size. Panel (a) reports c₁=3m; panel (b) reports k=π²m. Error bars are ordinary trajectory SEMs of anchored curves. Slope errors propagate the full within-trajectory covariance across widths; no bootstrap or residual-based replacement is used. Fits remain c₁=1.0429488183±0.0018259189 and k=1.0414242468±0.0018854260. Shading identifies the fitted long-distance region.

**Source and reproduction.** Saved numerical two-panel output, regenerated with overlined labels, from [entropy_charge_endpoint_sample_resolved](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/entropy_charge_endpoint_sample_resolved). The bundled [sample curves, original manifest, and input provenance](data/entropy_charge) preserve 700 sample identities and the original source hashes. Run `python sources/plot_entropy_charge.py` from this directory; `--output-dir /tmp/entropy-charge-preview` redirects the regenerated figure. No dynamics or original campaign writes occur.


---

<a id="figure-07-entanglement-spectrum"></a>

## Figure 07 — Occupation spectrum, entanglement energy, and central mode count

[Vector PDF](Figure_07_entanglement_spectrum.pdf) · [300-dpi preview](Figure_07_entanglement_spectrum.png)

Vector PDF and 300-dpi PNG; three vertically stacked panels (3×1), 3.375 × 5.8 inches. Panel (a) has a logarithmic density axis; panels (b,c) have linear axes. Both histograms are normalized. Panel (c) retains the raw mean count, without division by subsystem dimension.

### Data and protocol

Each of the two saved ensembles contains 100 independent Born-rule trajectories at $N_x=20$, $N_y=32$, endpoint cycle $64=2N_y$. They use hard, support-terminated domain walls, overcomplete Wannier (OW) range $n_{\mathrm{shell}}=1$, $\alpha_1=1$ or $3$, $\alpha_2=30$, interfaces $x=5,15$, pure initialization with a product-state exterior, slab-only measurements, perfect correction, no postselection, and `raster_y` ordering. Production used `classA_U1FGTN_gpu.run_markov_circuit`; this figure uses saved spectra and endpoint occupied frames only. No trajectory was rerun. The two parameter values are separate ensembles with matching geometry and protocol, not paired samples.

Each subsystem contains all x columns, both orbitals, and a periodic y interval of width $A_y$. The centered occupation is $\lambda=2\nu-1$, where $\nu$ is an eigenvalue of the restricted single-particle occupation matrix. **Every panel includes all 32 translated cut origins $y_0=0,\ldots,31$.** Panels (a,b) compare both parameter values at $A_y=16$; panel (c) retains the existing width sweep for $\alpha_1=1$ only. Origins within a trajectory are correlated; there are 100 independent samples per ensemble, not 3,200. At half width, complementary cuts have equal retained counts, verified for every trajectory. Spectra are computed per trajectory and cut before pooling; the figure does not diagonalize an ensemble-averaged correlation matrix.

### Panel (a): normalized full occupation spectrum

At $A_y=16$, each strip has $2N_xA_y=640$ modes. For a saved occupied frame $F$, the restricted occupation matrix is $G_A=F_AF_A^\dagger$; diagonalize $2G_A-\mathbf{1}_{640}$. Pool all eigenvalues over 100 trajectories and 32 origins separately for each $\alpha_1$, giving **2,048,000 observations per parameter value**, with no spectral-window exclusion or equilibrium data.

Use 100 equal-width bins over the full centered range $[-1,1]$, $\Delta\lambda=0.02$. For bin count $h_{\alpha,j}$, the displayed probability density is

$$
\rho_{\alpha,j}=\frac{h_{\alpha,j}}{2048000\,\Delta\lambda},
\qquad \sum_j\rho_{\alpha,j}\Delta\lambda=1.
$$

Normalize after pooling. The logarithmic ordinate resolves sparse interior weight alongside the endpoint peaks. Zero-count bins are absent on the log axis; no pseudocounts or smoothing are added. Numerical excursions beyond $[-1,1]$ are clipped only after checking a $10^{-8}$ tolerance; the largest observed excursion is below $5\times10^{-13}$. No observations, including endpoint modes, are dropped. Histogram curves have no error bars; modes and origins are not treated as independent trajectories.

### Panel (b): normalized entanglement-energy distribution within the window

Retain $W=\{|\lambda|\leq0.99\}$ separately for each trajectory and origin, and transform each retained eigenvalue before pooling:

$$
\varepsilon=\log\frac{1-\nu}{\nu}
=\log(1-\lambda)-\log(1+\lambda).
$$

The cutoff $|\lambda|\leq0.99$ gives $0.005\leq\nu\leq0.995$. Its endpoints map to $\varepsilon=\pm\log(0.995/0.005)=\pm\log199$, with $\log199=5.2933048247$. Thus 199 is the occupation ratio at the selected cutoff, not a fitted constant, a system size, or a physical gap scale.

These are signed single-particle entanglement energies, not many-body levels. The 101 equal-width bins span $[-\log199,\log199]$, with one centered on zero. For retained bin counts $g_{\alpha,j}$ and total retained count $M_\alpha$, display the conditional density

$$
p_{W,\alpha,j}=\frac{g_{\alpha,j}}{M_\alpha\,\Delta\varepsilon},
\qquad \sum_jp_{W,\alpha,j}\Delta\varepsilon=1.
$$

Each curve is normalized after pooling, not by averaging individually normalized cut or trajectory histograms. There is no smoothing, energy clipping, equilibrium reference, or histogram fit. The different total populations remain recorded below and in the CSV; unit-area curves compare shapes rather than mode numbers.

| Quantity, all 32 half-strip origins | $\alpha_1=1$ | $\alpha_1=3$ |
|---|---:|---:|
| Full occupation observations | 2,048,000 | 2,048,000 |
| Retained window observations $M_\alpha$ | 95,398 | 83,326 |
| Mean retained count per cut ± trajectory SEM | 29.811875 ± 0.0340418892 | 26.039375 ± 0.0090005173 |
| Pooled observations with $|\varepsilon|<0.1$ | 1,862 | 52 |
| Pooled observations with $|\varepsilon|<1$ | 18,836 | 204 |

The rows count correlated observations, not independent samples. Count SEMs first average all origins within a trajectory, then use the sample standard deviation of the 100 trajectory means divided by $\sqrt{100}$. The $|\varepsilon|<1$ row is a descriptive threshold, not a fitted gap or relaxation-rate criterion. Panel (b) shows finite density through zero for $\alpha_1=1$ and a pronounced central, gap-like depletion for $\alpha_1=3$. The latter is not a hard empty interval: its central histogram bin $[-0.05240896,0.05240896]$ contains 46 pooled levels, compared with 1,012 for $\alpha_1=1$. Levels with $|\varepsilon|<0.1$ occur in 24 of the 100 trivial trajectories, so the small central density should not be described as strictly zero. At this one size and time, the histogram does not establish a thermodynamic entanglement gap. Rare low entanglement energies for $\alpha_1=3$ do not by themselves identify slow dynamical modes. Its density is concentrated toward the outer parts of this window; an area law does not impose a universal reduced-state spectral distribution. The separate spatial check below concerns the earlier fixed $y_0=0$ diagnostic, not the full pooled ensemble.

### Panel (c): raw mean mode count against log chord length

For $\alpha_1=1$, count modes satisfying $|\lambda|\leq0.99$ separately for every trajectory, origin, and width. Average origins within each trajectory to obtain $n_s(A_y)$, then average trajectories. Error bars are ordinary trajectory SEM, $\mathrm{std}_s[n_s]/\sqrt{100}$, using sample standard deviation (`ddof=1`). **Counts are not divided by $2N_xA_y$.** No $\alpha_1=3$ width-scaling curve is inferred from its half-strip histogram.

All widths $A_y=1,\ldots,16$ are displayed. The shaded fit window is $5\leq A_y\leq16$. The dashed, unweighted two-parameter fit is extended across the displayed range:

$$
\overline N(A_y)=a+b\log D(A_y),\qquad
 D(A_y)=\frac{N_y}{\pi}\sin\frac{\pi A_y}{N_y}.
$$

Results: $a=27.0649991\pm0.0328796$, $b=1.1874139\pm0.0134173$, $R^2=0.9996965$. Coefficient uncertainties are SEMs of trajectory-level linear fits, equivalently propagating full cross-width trajectory covariance; they are not regression-residual errors. At half width, $\overline N(16)=29.811875\pm0.0340418892$, consistent with $95398/(100\times32)$. Every plotted count, fit coefficient, fit window, and uncertainty is preserved from the existing raw-count panel. This finite-size descriptive fit does not establish an asymptotic law.

### Provenance and reproduction

Sources are campaign 09's `centered_spectral_window_origin_averaged_n20x32_hard_alpha1_v2`, `mean_mode_count_window_scan_n20x32_v1`, and saved endpoint batches under `pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1/hard/Ny032/alpha1_3`. Paths, hashes, production receipts, and the original fit are preserved in [data/entanglement_spectrum](data/entanglement_spectrum).

- `pooled_half_strip_spectra.npz` contains the two 100×32×640 full spectra, parameter values, sample IDs, origins, geometry, and cycle. The $\alpha_1=1$ array reuses the saved all-origin cache. The $\alpha_1=3$ array is extracted from four verified endpoint batches, with orthonormality, Hermiticity, trace, and spectral-bound checks.
- `pooled_half_strip_provenance.json` binds this input and its extraction script to SHA-256, records the original source receipts, verifies agreement with the earlier fixed-origin extraction, and checks complementary-cut counts. Retained $\alpha_1=1$ values match the original all-origin window input exactly.
- The unchanged `inputs.npz` contains $\alpha_1=1$ retained occupations/energies with trajectory/origin/mode indices and raw counts for all 100 trajectories, 16 widths, and 32 origins. `mean_mode_counts.csv` and the archived fit remain unchanged.
- `occupation_histogram.csv` and `energy_histogram_comparison.csv` record both raw pooled counts and displayed densities for both parameter values. `half_strip_window_counts.npz` records each sample/origin count. The unchanged `energy_histogram.csv` retains the original $\alpha_1=1$ raw counts for auditing; its raw ordinate is no longer displayed.
- The earlier `occupation_inputs.npz` and `occupation_provenance.json` are retained for the separate fixed-origin diagnostic below; they are not the current Figure 7 histogram inputs.

Run `python sources/plot_entanglement_spectrum.py` from the bundle directory. Dependencies: NumPy, Matplotlib, and CMU Sans Serif. The current renderer uses native Computer Modern mathtext and does not require LaTeX. The renderer verifies pooled inputs and histogram normalization, reproduces the count fit/SEM, and writes PDF, PNG, numerical CSVs, and `data/entanglement_spectrum/validation.json`.

To repeat the optional extraction from original repository data, run `python sources/extract_pooled_half_strip_spectra.py --repo-root /path/to/repository --workers 8` first (additional dependencies: SciPy, threadpoolctl, tqdm). It performs endpoint eigendecompositions only and writes exclusively inside this figure bundle. Reusing the bundled compact spectra needs no endpoint eigendecomposition or circuit simulation.


---

<a id="normalized-energy-comparison"></a>

## Separate comparison — Normalized entanglement energy for $\alpha_1=1,3$

[Vector PDF](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.pdf) · [300-dpi preview](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.png)

This earlier fixed-origin diagnostic is retained separately. Final Figure 7 now includes both parameter values with all 32 origins; the fixed-origin counts below describe only this diagnostic.

**Data and protocol.** Use the earlier saved campaign-09 fixed-cut extraction from the same ensembles as Figure 7: $N_x=20$, $N_y=32$, $A_y=16$, cycle 64, 100 independent Born trajectories for each $\alpha_1=1,3$, $\alpha_2=30$, hard support-terminated interfaces at $x=5,15$, $n_{\mathrm{shell}}=1$, pure initialization with a product exterior, slab-only measurements, perfect correction, no postselection, and raster-y ordering. Each cut uses all x, both orbitals, and $y=0,\ldots,15$. Both ensembles use this one fixed origin, so the comparison does not mix fixed-cut and all-origin sampling. No dynamics or eigendecomposition was rerun for this diagnostic.

**Transformation and normalization.** Retain exactly $W=\{|\lambda|\leq0.99\}$, separately for each trajectory. Transform retained centered occupations using $\varepsilon=\log(1-\lambda)-\log(1+\lambda)$, then pool trajectories within each ensemble. Use the same 101 equal-width bins over $[-\log199,\log199]$ for both curves, with linear axes. For bin count $h_{\alpha,j}$ and total retained count $M_\alpha$, plot

$$
p_{W,\alpha,j}=\frac{h_{\alpha,j}}{M_\alpha\,\Delta\varepsilon},
\qquad \sum_j p_{W,\alpha,j}\Delta\varepsilon=1.
$$

This is the **conditional, unit-area distribution inside the window**. It is normalized after pooling, rather than averaging separately normalized trajectory histograms. No smoothing, pseudocounts, equilibrium reference, or fit is used. The histogram has no error bars; correlated modes are not counted as independent trajectories. The data table retains counts so that unit-area normalization does not conceal the different populations.

| Quantity, fixed half-strip cut | $\alpha_1=1$ | $\alpha_1=3$ |
|---|---:|---:|
| Retained modes $M_\alpha$ | 3,120 | 2,705 |
| Fraction of all 64,000 occupations | 4.875% | 4.2265625% |
| Mean count per trajectory ± trajectory SEM | 31.20 ± 0.07247 | 27.05 ± 0.02973 |
| Modes with $|\varepsilon|<1$ | 626 | 9 |
| Trajectories contributing such central modes | 100/100 | 9/100 |
| Conditional fraction with $|\varepsilon|<1$ | 20.0641% | 0.3327% |
| Smallest observed $|\varepsilon|$ | 0.0004014 | 0.0963714 |

Count SEMs use the sample standard deviation of the 100 trajectory counts divided by $\sqrt{100}$. The $|\varepsilon|<1$ threshold is an explicit descriptive diagnostic, not a fitted gap or a dynamical criterion. The fixed-origin mean 31.20 for $\alpha_1=1$ differs from Figure 7(c)'s 29.811875 because the latter averages all 32 origins within each trajectory. Those origins are correlated; they do not supply additional independent trajectories.

**What this says about “slow modes.”** The $\alpha_1=3$ ensemble has rare low-entanglement-energy observations, but its retained density is strongly concentrated toward the outer parts of this window. These levels are obtained from a reduced-state spectrum at one endpoint. Their small $|\varepsilon|$ indicates occupation near $1/2$; this calculation contains no temporal decay rate and therefore does **not establish slow dynamical relaxation**. It also does not establish a strict entanglement gap shared by every trajectory: the nine observed central levels are retained explicitly. A relaxation claim would need time-dependent data or an appropriate dynamical spectrum for the same protocol.

**Reproduction and verification.** Run `python sources/plot_energy_window_comparison.py`. The only numerical input is the existing portable `data/entanglement_spectrum/occupation_inputs.npz`, checked against its provenance hash. The script saves `data/energy_window_comparison/energy_histogram.csv`, `retained_modes.csv` (including trajectory and mode identities), `trajectory_counts.csv`, and `validation.json`. It checks selection, the energy inverse transform, count conservation, separate unit-area normalization, and figure bounds, and records that it leaves the numbered Figure 7 unchanged at the time it runs. The PDF is vector; its PNG is rendered at 300 dpi. Dependencies are the same as Figure 7's renderer. The numbered figures, original manuscript assets, and campaign data are preserved.

---

<a id="area-law-window-modes"></a>

## What the $\alpha_1=3$ window modes represent

**Direct spatial check.** The saved $\alpha_1=3$ endpoint frames were used to diagonalize the restricted matrix for the same $y_0=0$, $A_y=16$ cut in all 100 trajectories, retaining its eigenvectors for $|\lambda|\leq0.99$. Their eigenvalues reproduce the delivered spectrum to $5.33\times10^{-15}$. With $y\in\{0,\ldots,15\}$, define distance from the cut as $\min(y,15-y)$. Two boundary rows on each side means $y=0,1,14,15$; these are the horizontal entanglement cuts, not the domain walls in x.

| Mode group | Number of modes | Mean weight in first row at either cut | Mean weight in first two rows at either cut |
|---|---:|---:|---:|
| All retained modes | 2,705 | 98.4054% | 99.9133% |
| $|\varepsilon|\geq4$ | 2,686 | 98.6080% | 99.9403% |
| $|\varepsilon|<1$ | 9 | 56.2105% | 98.5181% |

These are descriptive averages over modes, with no independence assumption or SEM assigned. For each of the nine central modes, at least 93.04% of its probability lies within the first two rows at the cuts. The dominant modes are weakly mixed: $|\varepsilon|\geq4$ corresponds to occupation $\nu\leq0.0179862$ or $\nu\geq0.9820138$. This identifies cut-localized entanglement modes in the actual saved ensemble, rather than inferring their location from the histogram alone. It is consistent with ordinary short-range boundary entanglement; the spatial check is not a relaxation measurement or a system-size proof of an area law.

**Reduced-state spectrum.** Each trajectory is globally pure. Tracing out the complement produces a mixed $\hat\rho_A$ whenever that cut is entangled. In its Gaussian entanglement-mode basis,

$$
\hat\rho_A=\bigotimes_j\left[(1-\nu_j)|0_j\rangle\!\rangle\langle\!\langle0_j|
+\nu_j|1_j\rangle\!\rangle\langle\!\langle1_j|\right],
\qquad
p_{\boldsymbol n}=\prod_j\nu_j^{n_j}(1-\nu_j)^{1-n_j}.
$$

The plot shows the single-particle values $\varepsilon_j=\log[(1-\nu_j)/\nu_j]$, whereas $p_{\boldsymbol n}$ are the eigenvalues of the full reduced density matrix. Spectra are calculated within each trajectory before pooling. This is not diagonalization of the ensemble-averaged state. The Gaussian reduced-state construction and examples of eigenfunctions localized near entanglement cuts are discussed in [Peschel and Eisler (2009), Sections 3–5](https://arxiv.org/pdf/0906.1663).

**An area law does not specify a unique histogram.** The constraint is on $S_A=\sum_j h(\nu_j)$, where $h(p)=-p\log p-(1-p)\log(1-p)$. It does not require every $\nu_j$ to be almost 0 or 1, or forbid $\varepsilon_j=0$. For a concrete Gaussian example, put independent one-fermion pairs $\sqrt{p}\,|10\rangle\!\rangle+\sqrt{1-p}\,|01\rangle\!\rangle$ on bonds crossing the boundary. Each crossing contributes $h(p)$, so a number of bonds proportional to boundary length gives an area law for every choice of $p$. At $p=1/2$ every crossing contributes an exact zero entanglement energy and $\log2$ entropy; at $p$ near 0 or 1, its energy has large magnitude and its entropy is small. Thus a hard entanglement gap or a particular smooth spectral distribution cannot follow from the area law alone.

For the fixed window here, every counted Gaussian mode contributes at least $h(0.005)$, giving the elementary bound $N_{0.99}\leq S_A/h(0.005)$. The bound limits the abundance of appreciably mixed modes; it does not determine their distribution inside the window. In this strip geometry, increasing $A_y$ keeps the two cut lengths fixed. A saturating boundary contribution would therefore give a decreasing fraction $N_{0.99}/(40A_y)$, while a logarithmically increasing count can also give a decreasing fraction. This is why the normalized plot below and the original count plot answer different questions.

**Reproduction.** Run `python sources/analyze_trivial_window_modes.py` within the original repository checkout. It verifies the four original $\alpha_1=3$ batch hashes and uses saved endpoint frames only, with CPU eigendecompositions and four BLAS threads. It writes `data/energy_window_comparison/alpha1_3_mode_localization.csv` and `spatial_mode_checks.json`; no circuit simulations or changes to campaign files occur. Ordinary plot reproduction does not require this additional spatial check.

---

<a id="normalized-mode-fraction"></a>

## Separate panel C — Fraction of subsystem modes versus log chord length

[Vector PDF](diagnostics/normalized_mode_fraction_vs_log_chord.pdf) · [300-dpi preview](diagnostics/normalized_mode_fraction_vs_log_chord.png)

This is the requested standalone, normalized version of Figure 7(c), using its existing $\alpha_1=1$ data. It does not replace the manuscript panel or add an $\alpha_1=3$ curve.

**Data and protocol.** Campaign 09, $N_x=20$, $N_y=32$, 100 independent pure Born trajectories, endpoint cycle 64, $\alpha_1=1$, $\alpha_2=30$, hard support-terminated walls at $x=5,15$, OW range $n_{\mathrm{shell}}=1$, product-state exterior, slab-only measurements, perfect correction, no postselection, raster-y order. Each strip includes all x and both orbitals; all widths $A_y=1,\ldots,16$ and all 32 periodic origins $y_0$ are used.

**Normalization, averaging, and uncertainty.** Count eigenvalues with $|\lambda|\leq0.99$ in each cut and divide by the full subsystem dimension $D_A=2N_xA_y=40A_y$, including the exterior and modes outside the spectral window. Define

$$
f_s(A_y)=\frac{1}{32}\sum_{y_0=0}^{31}
\frac{N_s(A_y,y_0;|\lambda|\leq0.99)}{40A_y},
\qquad
\overline f(A_y)=\frac{1}{100}\sum_{s=1}^{100}f_s(A_y).
$$

The plotted error is $\mathrm{std}_s(f_s;\mathrm{ddof}=1)/\sqrt{100}$. Origins within a trajectory are correlated, and there are 100 independent samples. The denominator is the number of all subsystem modes, not the number of mixed modes or the number retained within the window. Because $D_A$ is fixed at each width, this equals Figure 7(c)'s origin-averaged count and SEM divided by $40A_y$.

**Axes and preserved fit.** The archived diagnostic labels its chord length $d(A_y)$, equivalent to the manuscript’s unified $D(A_y)$. The horizontal variable is $\log D(A_y)$, with $D(A_y)=(32/\pi)\sin(\pi A_y/32)$. The dashed curve is the existing raw-count fit $a+b\log D(A_y)$ divided by $40A_y$, using the unchanged $a=27.0649990871$, $b=1.1874139457$ and the original fit window $5\leq A_y\leq16$ (shaded). It is extended outside that window as before. No new fit is performed, and the fractions are not fitted to a straight line. The original coefficient SEMs remain $0.0328796392$ and $0.0134173240$.

At $A_y=16$, the result is $\overline f=0.0465810546875\pm0.0000531904519$, or $(4.6581055\pm0.0053190)\%$. Its decline with width reflects division by the growing subsystem dimension, even while the raw count rises. Thus this fraction plot measures the relative abundance of window modes; the original raw-count plot remains the direct display of their approximately logarithmic growth.

**Reproduction.** Run `python sources/plot_normalized_mode_fraction.py`. It reads the existing checksum-verified `data/entanglement_spectrum/inputs.npz` and writes the vector PDF, 300-dpi PNG, `data/normalized_mode_fraction/fraction_by_width.csv`, `rescaled_fit.csv`, compact trajectory-level fractions, and `validation.json`. It verifies the averaging order, normalization, SEM, original fit coefficients, and unchanged checksums of all 14 numbered figure versions. No circuit simulation or spectral calculation is rerun.

---

<a id="figure-08-central-charge"></a>

## Figure 8 — Central-charge convergence and size dependence

[Vector PDF](Figure_08_central_charge.pdf) · [300-dpi preview](Figure_08_central_charge.png)

[PDF](Figure_08_central_charge.pdf) · [PNG](Figure_08_central_charge.png)

**Contents.** Panel (a) shows c_eff versus t/Nᵧ for Nᵧ=30,40,50, with an inset of |c_eff−1| on a logarithmic vertical axis. Panel (b) shows endpoint c_eff versus Nᵧ=30,35,40,45,50,55,60. This is the recent two-panel replacement for manuscript Figure 8, now stacked vertically (2×1) at 3.375 × 5.6 inches. Only the layout changed; data, fit parameters, error bars, and inset content are preserved.

**Data and protocol.** Both panels use Nₓ=20, hard walls, α₁=1, α₂=30, n_shell=1, perfect correction, raster-y updates, periodic boundaries, and 100 independent random-pure trajectories per size, evolved through T=2Nᵧ. **The panels use separate ensembles:** (a) uses the legacy cycle-resolved S100 campaign, and (b) uses the campaign-05 endpoint ensemble. They are never pooled. Average over periodic strip positions within each trajectory and then over trajectories before fitting.

For **panel (b)**, circuit measurements act only on the slab (`meas_slab_only=True`), after Born-conditioned onsite preparation of the exterior before cycle zero; campaign-05 shards record this cycle-zero convention explicitly. For **panel (a)**, the saved legacy `config_json` does not record the measurement-region flag or exterior preparation. These details remain unspecified for that ensemble; the modern campaign's preparation is not assigned to it.

**Analysis.** At each time or endpoint, fit the mean full-strip entropy versus log D(Aᵧ), with a free intercept (the saved log-sine coordinate differs only by the constant log(Nᵧ/π), so the slope is identical), over 8≤Aᵧ≤floor(Nᵧ/2); c_eff=3m. Panel (a) displays multiples of five cycles from t=10 through 2Nᵧ. Its bars are three times the OLS slope standard error of the mean-curve fit, **not trajectory-sampling SEMs**. Panel (b) bars are trajectory-sampling SEMs propagated with the full covariance across widths. The horizontal reference is c_eff=1. Dashes connecting endpoint sizes are visual guides, not a size-extrapolation fit.

**Source and reproduction.** Replotted in a vertical layout from the same saved inputs as `ceff_cycle_and_endpoint_size_1x2` from [entropy_ceff_multisize](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/entropy_ceff_multisize). The two input tables and original provenance are in [data/central_charge](data/central_charge). Run `python sources/plot_central_charge.py`; `--output-dir /tmp/central-charge-preview` preserves the delivered copy while regenerating a preview.


---

<a id="figure-09-wall-entropy"></a>

## Figure 9 — Entropy carried by each wall

[Vector PDF](Figure_09_wall_entropy.pdf) · [300-dpi preview](Figure_09_wall_entropy.png)

[PDF](Figure_09_wall_entropy.pdf) · [PNG](Figure_09_wall_entropy.png)

**Contents.** Anchored entropy contour integrated over the left and right walls. The plotted windows are **x=5,6** and **x=14,15**, respectively. These are two-column windows; the revised manuscript caption now matches the plotted windows.

**Data and protocol.** Completed campaign-16 Lane-B contour ensemble, Nₓ=20, Nᵧ=30,35,40,45,55, 100 independent random-pure trajectories per size, endpoint T=2Nᵧ. Hard walls, α₁=1, α₂=30, n_shell=1, periodic boundary conditions, perfect correction and raster-y order. Circuit measurements act only on the slab (`meas_slab_only=True`); Born-conditioned onsite measurements prepare the exterior before cycle zero, as confirmed by the shards' `cycle_zero_semantics=after_born_conditioned_exterior_preparation`. The full-width strip contour is averaged over all periodic strip positions within each trajectory, in relative-y coordinates, then integrated over the indicated x columns and all Aᵧ subsystem rows. These ensembles are separate from the full-strip scalar curves used in Figure 6.

**Analysis.** Anchor each trajectory at Aᵧ*=floor(Nᵧ/2). Fit the resulting mean curves through the origin against $\log[D(A_y)/D(A_y^\star)]$ over 8≤Aᵧ≤floor(Nᵧ/2), with equal total weight per size. Errors use trajectory SEMs and the full covariance across widths. Left m=0.1728818152±0.0004422961; right m=0.1624067405±0.0005319065. No fit or uncertainty changed when Aᵧ=1 was removed from display. The inset's right-wall dashed line is drawn at the outer edge of cell x=15; this is a grid drawing convention, not a change in the integrated window.

**Source and reproduction.** Reused `lane_B_wall_two_cell_entropy_collapse_2x1` from [hard_wall_all_ay_half_entropy](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/hard_wall_all_ay_half_entropy). Bundled [curves, fits, and manifest](data/wall_entropy) specify exact source provenance. Run `python sources/plot_wall_entropy.py`; use `--output-dir /tmp/wall-entropy-preview` to render elsewhere.


---

<a id="figure-10-modular-evolution"></a>

## Figure 10 — Early-time modular propagation

[Vector PDF](Figure_10_modular_evolution.pdf) · [300-dpi preview](Figure_10_modular_evolution.png)

The plotted arrays and snapshots from `modular_packets_n20x32_y8_stacked.pdf` are preserved: density panels for $\alpha_1=3$ and 1, followed by the $\alpha_1=1$ displacement. The regenerated 3.375 × 6.2 inch figure overlines the mean displacement.

**Data and protocol.** Campaign 09 supplies 100 independent pure-state Born
trajectories per $\alpha_1=1,3$, at $N_x=20,N_y=32$, endpoint cycle 64.
The hard-wall protocol uses $\alpha_2=30$, overcomplete Wannier (OW)
range $n_{\mathrm{shell}}=1$, random pure initialization with
Born-conditioned exterior preparation, perfect correction, and raster-y order.
Each of 32 translated cuts contains all x and 16 y rows. Two separate packets
start at $(x,y-y_0)=(5,8),(15,8)$; each contains two incoherent local orbital
occupations, total charge 2.

**Analysis.** Each trajectory/cut's reduced centered covariance $G_{c,A}$
is diagonalized independently. The single-particle generator is
$h_A=-2\operatorname{atanh}G_{c,A}$, with eigenvalues clipped to
$[-1+10^{-10},1-10^{-10}]$. Spectral exponentiation generates unitary modular
evolution without division by circuit depth. No covariance or generator is
averaged before evolution.

(a,b) The snapshots are $t_{\mathrm{mod}}=0,0.1,0.2$. Evolved density is
averaged over cuts within each trajectory, then across trajectories. Marker
area scales as $\sqrt{\overline\rho/2}$, with the same normalization in
both panels. The $10^{-4}$ visibility threshold affects rendering only.

(c) Signed wall-window center-of-mass displacement is evaluated for
$t_{\mathrm{mod}}=0\ldots1$, spacing 0.01. The five-column window contains
periodic-x distances at most 2 from the injection column. Normalize each
realization by its instantaneous retained window charge before calculating
displacement; then average origins and trajectories. Shading is one SEM of
the 100 trajectory-level origin means, not 3,200 independent cuts.

There is no smoothing, unwrapping, wall-sign adjustment, or velocity fit.
Modular time is not monitored-circuit time. Motion magnitude and oscillation
frequency depend on the eigenvalue cutoff; this plot alone does not establish
regularization-independent speed or chirality.

**Source and reproduction.** Original producer:
[plot_early_time_stacked.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/endpoint_modular_packets/plot_early_time_stacked.py).
The exact caption, plotting metadata, and analysis summary are under
`data/preserved_protocols/modular_evolution/`. Run `python sources/plot_modular_evolution.py`. Portable arrays in `data/modular_evolution/` select the existing $10^{-10}$ regularization and preserve each snapshot and displacement value exactly; no spectrum is recomputed. With $G_{ij}=\langle\hat c_i^\dagger\hat c_j\rangle$, the Fock-space modular generator uses $[h_A]_{ji}$, while the annihilation-mode coefficient vector evolves with $e^{-i h_A t_{\rm mod}}$. This explicit transpose keeps the saved propagation convention unchanged.


---

<a id="figure-11-mean-channel"></a>

## Figure 11 — Mean-channel spectrum, correlations, and relaxation gap

[Vector PDF](Figure_11_mean_channel.pdf) · [300-dpi preview](Figure_11_mean_channel.png)

**Layout:** four vertically stacked panels, 3.375 × 7.15 inches, vector PDF
and 300-dpi PNG. The plotting panels are narrower and taller to match Figure 4’s panel proportions; fonts remain 8 pt. Panels (a,b) retain the original occupation and correlator
arrays, styles, axis limits, normalizations, and display cutoff. Panel (c)
shows the dimensionless multiplier gap $g_C=1-\rho(A)^2$ versus $\alpha_1$;
panel (d) shows its direct inverse-length fits. These replace the earlier
logarithmic-rate panels, $\Delta_C=-2\log\rho(A)$. The
original two-panel manuscript PDF remains unchanged in the parent folder.

### Identification of the replacement screenshots

Both attachments match the corresponding archived PNGs **pixel for pixel**.
Their vector versions are:

- Gap versus $\alpha_1$:
  [multiplier_gap_vs_alpha.pdf](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1/results/20261001T220546Z/multiplier_gap_alpha/20261002T021935Z/multiplier_gap_vs_alpha.pdf).
- Gap fit versus $1/N_y$:
  [multiplier_gap_fits.pdf](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1/results/20261001T220546Z/multiplier_gap_fits/20261002T020317Z/multiplier_gap_fits.pdf).

These are the **fixed-$N_x=20$** results, not the earlier $N_x=N_y$ scan.
The scan and fit renditions are dated `20261002T021935Z` and
`20261002T020317Z`, respectively. The earlier logarithmic-rate PDFs and
compact inputs remain preserved; their screenshot matches are also retained.
Screenshot identities, original file hashes, numerical audit results,
and producer paths are recorded in [provenance.json](data/mean_channel/provenance.json).

### Shared measurement protocol

All four panels use a periodic two-orbital lattice with $N_x=20$,
inclusive central slab $x=5,\ldots,15$, and active exterior sites
$x=0,\ldots,4,16,\ldots,19$. The exterior joins across the periodic seam.
The parent parameter is $\alpha_1$ inside and $\alpha_2=30$ outside.
Overcomplete Wannier (OW) modes use the $X$ trial-orbital basis and
$n_{\mathrm{shell}}=1$. Hard walls remove OW support across the slab
interfaces and renormalize each retained mode. Both slabs evolve; the
exterior is neither frozen nor discarded. Boundaries are periodic in
both directions with zero twist, and arithmetic is complex128.

Every cycle traverses increasing $y$ at fixed $x$, then increasing $x$,
with within-cell mode order `Ap, Am, Bp, Bm`. Each OW occupation is
measured and perfectly corrected to its band target: empty the upper
band, fill the lower band. Measurement outcomes are averaged analytically.
The same deterministic raster word repeats each cycle. The number
dephasing here belongs to the exact measurement-and-reset map; it is not
an extra dephasing-only channel appended after a reset. There is no
random schedule average, sampled trajectory count $S$, or trajectory SEM.

### Panels (a,b): cycle-128 mean state

**Data and protocol.** These are deterministic, analytically outcome-averaged
channel endpoints, not a Monte Carlo trajectory ensemble. Here $N_y=64$
and $\alpha_1=1,3$. The whole system evolves from maximal mixing,
$\overline G_0=\mathbf{1}/2$; no exterior preparation is applied. The
canonical CPU entry point is `classA_U1FGTN.run_markov_channel`.

The same raster-y schedule is repeated for 128 cycles: y increases at fixed x,
then x increases, with within-cell channel order `Ap, Am, Bp, Bm`. Both panels
use the cycle-128 endpoint; there is no temporal or schedule average. The
last ten normalized Frobenius increments are below $10^{-12}$.

**Panel (a).** Spatial y-translation averaging of the endpoint precedes
momentum block diagonalization. All 40 occupation eigenvalues at each of
64 momenta are retained, including exterior modes. The dashed line is
occupation $1/2$.

**Panel (b).** Write $\overline G$ for the occupation matrix of the
outcome-averaged endpoint. The plotted quantity is

$$
C_{\overline G}^{\mathrm{av}}(r)=\frac1{N_x}\sum_x\frac1{2N_y}
\sum_{y,\mu,\nu}
|\overline G_{(x,y,\mu),(x,y+r,\nu)}|^2.
$$

The ordinate is its logarithm; the abscissa is
$\log D(r_y)$ with $N_y=64$ in the common chord-length definition.
No additional spatial twirl precedes squaring in this panel. Only values
above $10^{-8}$ are displayed, leaving $r=1,2,3,4$ for both parameters.
This is the squared correlator of the mean state, not the trajectory mean
of a squared correlator. There are no sampling error bars, fits, or
finite-size collapse claims in panels (a,b).

### Panel (c): direct channel-spectrum scan

The completed `fixed_width_alpha_spectral_v1` run contains **105 cases**:
$N_y=20,40,60,80,100$ and $\alpha_1=1.0,1.1,\ldots,3.0$, always
$N_x=20$. These are spectral calculations of a single complete cycle,
with **no initial state, simulated time horizon, relaxation-time fit,
or Monte Carlo samples**. At $\alpha_1=2$ the canonical zero-Bloch-norm
prescription is used exactly (`nmag=1e-15` at a zero); the parameter is
not shifted off 2.

For each normalized creation-wavefunction column $W_j$, form $P_j=(W_jW_j^\dagger)^{\mathsf T}$ and
$Q_j=\mathbf{1}-P_j$. The transpose follows from $G_{ij}=\langle\hat c_i^\dagger\hat c_j\rangle$. In the actual raster order, step 1 first,

$$
A=Q_M\cdots Q_1,\qquad
\delta G' = A\delta G A^\dagger,\qquad
g_C=1-\rho(A)^2.
$$

Here $g_C$ is one minus the modulus of the slowest nonstationary
covariance-channel eigenvalue. It is dimensionless. Its relation to the
logarithmic decay rate used in the proof is
$g_C=1-\exp(-\Delta_C)$, with $\Delta_C=-2\log\rho(A)$ per complete cycle.

Hard support truncation makes the central slab and the entire periodic
exterior invariant blocks. The saved calculation constructs both products
from the full-lattice canonical OW modes, diagonalizes both, and takes
the largest eigenvalue modulus over their union. It does not reconstruct
the OW modes on smaller auxiliary lattices or keep only an interior gap.
The spectral radius, rather than the largest singular value, determines
the asymptotic rate of this generally nonnormal product.

All 105 cases have status `resolved_positive`; no points are excluded.
The largest independent dominant-eigenpair residual is
$8.59\times10^{-15}$ (rounded upward), and the largest difference between
the block action and an independently applied full-system projector word
is $3.04\times10^{-16}$. These are numerical checks, not statistical
uncertainties. Lines connect discrete parameter values to guide the eye;
no fit is applied in (c). There are no rate units on the plotted gap.

The plotted data are [multiplier_gaps.csv](data/mean_channel/multiplier_gaps.csv),
which retain the spectral radius and logarithmic rate alongside $g_C$.
The historical rate table is [gaps.csv](data/mean_channel/gaps.csv). Each original
case is stored under the scan root in
`Ny{Ny:03d}_a{alpha_index:02d}_{alpha_1:.1f}/spectrum.npz`, where
`alpha_index=10*(alpha_1-1)`, with a matching `completion.json`.
The NPZ contains the complete eigenvalue union, dominant left/right
eigenvectors, spectral radius, raw gap, block indices, configuration,
and diagnostics. All 105 source NPZ/receipt hashes were verified, and
every spectral radius and logarithmic rate was independently recovered
from its saved eigenvalues. Every displayed $g_C$ was checked against
$1-\rho(A)^2$ and $1-\exp(-\Delta_C)$.
Compact copies of all case receipts are in
[gap_case_receipts.json](data/mean_channel/gap_case_receipts.json).

### Panel (d): fixed-width extrapolation

Select $\alpha_1=1,3$ from (c), retaining all five sizes. Fit

$$
g_C(N_y)=g_\infty+\frac{a}{N_y}
$$

by ordinary **unweighted least squares on the five transformed $g_C$ values**.
The source refits $g_C$ directly; it does not exponentiate the previous
rate-fit intercept or curve. Circles and triangles are computed finite-length gaps; filled
squares at $1/N_y=0$ are fitted intercepts, not independently computed
infinite-size points. Solid lines extend the fitted model to the intercept.

| $\alpha_1$ | $g_\infty$ | $a$ | $R^2$ |
|---|---:|---:|---:|
| 1 | 0.831833310762 | 0.385187123840 | 0.999996949406 |
| 3 | 0.956729472459 | 0.090189643199 | 0.999843532475 |

Here $R^2=1-\sum_i(y_i-\widehat y_i)^2/\sum_i(y_i-\overline y)^2$.
The source fits have been recovered from the saved inputs and their
annotations checked. There are **no error bars or statistical confidence
intervals**: these are deterministic spectral data without a sampling
noise model. High $R^2$ measures agreement over the tested lengths; it
does not establish the extrapolation law at arbitrarily large $N_y$.
The fixed transverse width is 20 throughout, so this is not a simultaneous
two-dimensional thermodynamic limit.

The saved [multiplier_fits.json](data/mean_channel/multiplier_fits.json)
also records fit-window sensitivity and a quadratic inverse-length
alternative. These additional models and sensitivity shading are not drawn.
The plotted fit inputs are
[multiplier_fit_inputs.csv](data/mean_channel/multiplier_fit_inputs.csv).
The historical [fits.json](data/mean_channel/fits.json),
[fit_input_diagnostics.csv](data/mean_channel/fit_input_diagnostics.csv),
and [fit_annotations.json](data/mean_channel/fit_annotations.json) describe
the earlier logarithmic-rate fit and are not used for the current panels.

<a id="channel-gap-proof"></a>

### Why this is the charge-neutral many-body channel gap

The integrated proof is in the final appendix of [manuscript.tex](../../manuscript.tex), sourced from [channel_gap_appendix.tex](../../notes/manuscript_revision/channel_gap_appendix.tex). It states both the general upper bound from two-point closure and the stronger equality specific to exact ordered resets. The related purification derivation is in [purification_appendix.tex](../../notes/manuscript_revision/purification_appendix.tex); its finite-time gap is distinct from channel relaxation.

The earlier standalone two-page, two-column derivation is
[channel_gap_proof.pdf](../../notes/channel_gap_proof/channel_gap_proof.pdf)
with [RevTeX source](../../notes/channel_gap_proof/channel_gap_proof.tex).
It has been rewritten as a physical derivation following the organization
of BPJ Appendix D: local measurement/reset, two-point dynamics, the slow
density mode, and then higher-order correlations. The previous operator
hierarchy version is retained in
[the proof archive](../../notes/channel_gap_proof/archive/channel_gap_proof_operator_hierarchy_v1.pdf).
For the exact resets and repeated deterministic schedule used here, the
finite-system equality in the current figure's multiplier convention is

$$
\boxed{g_{q=0}=g_{\mathrm{even}}=g_C=1-\rho(A)^2.}
$$

The standalone proof uses the equivalent logarithmic-rate convention,
$\Delta=-\log(1-g)$:

$$
\boxed{\Delta_{q=0}=\Delta_{\mathrm{even}}=
\Delta_C=-2\log\rho(A).}
$$

The short argument has three steps:

1. **Exact local reset.** With
   $\hat N_\chi=\hat\chi^\dagger\hat\chi$, the empty and filled maps are
   $\widetilde{\mathcal E}_0(\hat\rho)=
   \hat\chi\hat\rho\hat\chi^\dagger+
   (\hat{\mathbf{1}}-\hat N_\chi)\hat\rho
   (\hat{\mathbf{1}}-\hat N_\chi)$ and
   $\widetilde{\mathcal E}_1(\hat\rho)=
   \hat\chi^\dagger\hat\rho\hat\chi+
   \hat N_\chi\hat\rho\hat N_\chi$.
   For an even operator, a reset kills a single measured-mode factor,
   replaces its occupation by the target $s=0$ or $1$, and leaves
   operators of the orthogonal modes unchanged. Thus normal-ordered
   operator degree can decrease but cannot increase.
2. **Triangular many-body hierarchy.** Charge-neutral operators have equal
   numbers $p$ of creation and annihilation operators. On the degree-$p$
   quotient, one cycle has block
   $(\wedge^p A)\otimes(\wedge^p A^*)$, where $\wedge^p$ is the
   antisymmetrized $p$-index action. Lower-degree source terms do not
   alter a triangular matrix's eigenvalues. If $a_i$ are the eigenvalues
   of $A$, the channel eigenvalues in this sector are all products
   $\prod_{i\in I}a_i\prod_{j\in J}a_j^*$ with $|I|=|J|=p$;
   the two index sets are independent and may overlap.
3. **The slowest multiplier.** Write $r=\rho(A)$, with $0<r<1$ as in
   these data. The $p=0$ block gives the stationary eigenvalue 1.
   Every nonconstant product obeys $|\Lambda|\le r^{2p}\le r^2$,
   and $p=1$ attains $r^2$ by using the same dominant eigenvalue in
   both factors. Therefore $-\log\max_{\Lambda\ne1}|\Lambda|
   =-2\log r$ in the neutral sector. This proves equality; quadratic
   covariance closure alone would establish only spectral inclusion
   and an upper bound on the many-body gap.

Charge neutral means $[\hat Q,\hat\rho]=0$, not fixed particle number.
Maximal mixing, fixed-number Slater states, and mixtures of number sectors
are neutral. Filling and emptying can change particle number while
preserving this operator-space sector. The proof also gives the same gap
on the entire parity-even operator space. For the specified Kraus
extension on unrestricted operator space, odd operators have slowest
multiplier $r$ instead, so $g_{\mathrm{full}}=1-r$ and
$\Delta_{\mathrm{full}}=\Delta_C/2$.
The proof PDF explains the parity similarity needed for that statement.

The plotted multiplier gap and its corresponding logarithmic rate
describe asymptotic **channel relaxation**. That rate need not equal
conditional-trajectory purification rates, and nonnormal/Jordan
prefactors can affect finite-time convergence. The finite-system proof
does not prove the numerical infinite-length extrapolation. Its exact
reset and fixed-schedule assumptions matter: imperfect feedback, extra
dephasing-only maps, or averaging different schedules require a new
higher-order analysis. The historical campaign captions conservatively
label the result a covariance-sector gap; those source files have been
preserved. The sector identification above is the additional result
established by the linked proof.

**Where the bound was established.** In this project it is Eq. (4) of the
September 29 note,
[A spectral bound on relaxation of the trajectory-averaged circuit](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/spectral_gap_v1/results/20260929T191448Z/analysis/gap_note.pdf).
The rewritten proof derives it as Eq. (5): with
$\Delta_{\rm sp}=-\log\rho(A)$,
$\Delta_{\rm MB}\le\Delta_C=2\Delta_{\rm sp}$ in a mixing sector
containing the two-point observables. Equivalently, monotonicity of
$1-e^{-\Delta}$ gives $g_{\rm MB}\le g_C$ in the multiplier convention.
It also constructs the corresponding
centered occupation eigenoperator, making the spectral inclusion explicit.

An explicit **continuous-time literature counterpart** is
[Barthel and Zhang, J. Stat. Mech. (2022) 113101](https://doi.org/10.1088/1742-5468/ac8e5c),
Sec. III.6, Proposition 7 and Eq. (63): the covariance generator is a
block in the many-body Liouvillian spectrum, implying the upper bound
by spectral inclusion. This is a deduction from their block identification,
not a quotation of a discrete-reset theorem. Their Sec. III.5, Proposition 6
and discussion after Eq. (57d), distinguish unrestricted and two-point
gaps for quasi-free Lindbladians. See pp. 14–16 of
[arXiv:2112.08344v5](https://arxiv.org/pdf/2112.08344).

[BPJ Appendix D](https://arxiv.org/pdf/2507.13437), particularly
Eqs. (D34)–(D40), instead bounds band-occupation relaxation using the
effective dissipation rates $\delta_\pm=\min_{\boldsymbol k}
\gamma_\pm(\boldsymbol k)$ for the untruncated bulk protocol. It motivates
the physical organization and notation of the rewritten proof; it does
not establish the present exact ordered-reset identity. The
single-particle rate $-\log\rho(A)$ here is distinct from those effective
band rates and from the occupation-spectrum gap around $1/2$.

Compile the proof from its directory with
`latexmk -pdf -bibtex- -outdir=build channel_gap_proof.tex`, then copy
`build/channel_gap_proof.pdf` to `channel_gap_proof.pdf`. Its original
two-page, two-column RevTeX format is retained. The standalone proof is preserved as a historical artifact. The updated manuscript now includes a tightened neutral/even-sector derivation in its channel-gap appendix, with the current $G,G_c$ convention and dimensionless gap.

### Sources and reproduction

Original producer for panels (a,b):
[plot_mean_channel_summary.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/plot_mean_channel_summary.py).
Endpoint configurations, hashes, and both observable summaries are copied to
`data/preserved_protocols/mean_channel/`. The exact plotted arrays are now
bundled as [spectra.npz](data/mean_channel/spectra.npz) and
[curves.npz](data/mean_channel/curves.npz), with both summaries. The endpoint
NPZ paths and hashes appear in those summaries; neither source endpoint
is changed. The displayed correlator values are also exported as
[panel_b_plotted_data.csv](data/mean_channel/panel_b_plotted_data.csv).

The current spectral, conversion, and fit producers are
[run_scan.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1/run_scan.py),
[plot_multiplier_gap_vs_alpha.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1/plot_multiplier_gap_vs_alpha.py),
and [plot_multiplier_gap.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign/fixed_width_alpha_spectral_v1/plot_multiplier_gap.py).
The spectral calculation uses canonical CPU
`classA_U1FGTN.construct_OW_projectors` to construct the represented
`classA_U1FGTN.run_markov_channel` map; it does not run covariance dynamics.

Rebuild the complete four-panel figure with
`python sources/plot_mean_channel.py`. This uses only the compact bundled
inputs; it does not rerun the campaign or eigensolver. An isolated render
can use `--output-dir /tmp/mean-channel-preview`. To recopy and audit
original inputs, use `python sources/extract_mean_channel.py` (requires
the source repository and the four historical/current attachment files). Numerical fit and
selection checks are in [validation.json](data/mean_channel/validation.json).
`restore_existing.py` skips Figure 11 so it cannot silently restore the
obsolete two-panel bundle version. Original campaign/manuscript assets
remain available at their recorded paths.


---

<a id="figure-a01-ow-truncation"></a>

## Figure A1 — Truncated overcomplete Wannier modes

[Vector PDF](Figure_A01_ow_truncation.pdf) · [300-dpi preview](Figure_A01_ow_truncation.png)

`Figure_A01_ow_truncation.pdf` and `.png` reproduce the four-panel `ow_truncation_summary` figure (old Figure 10) with consistent fonts and plain panel labels. The original PDF aspect ratio is restored (270.295:286.763, rendered at 3.375 inches wide); the panels are no longer vertically stretched. Scientific arrays, limits, and fit coefficients are preserved. This is a deterministic band-projector calculation: there are no sampled trajectories, initial states, circuit cycles, or sampling error bars.

### Panels and analysis

- **(a) Effective form factor.** The magnitude of the lower-band, $A$-family effective form factor at $\alpha=1$, along $k_y=0$ and $k_x/\pi\in[0.4,0.6]$, for square-envelope widths $w=1,2,4,6,8,\infty$. Each displayed scalar form-factor field is divided by its Brillouin-zone root-mean-square magnitude, evaluated on a uniform $241\times241$ grid. The cut contains 601 momenta. Filled symbols mark the separately refined zero positions; the vertical line is the untruncated zero $k_x/\pi=1/2$.
- **(b) Zero and critical-point drift.** Left axis: the zero position $k_x^{(0)}/\pi$ at $\alpha=1$, found by bounded scalar minimization. Right axis: the $\Gamma$-point gap-closing parameter $\alpha_c(w)$ of the normalized finite-window OW-frame Hamiltonian, found by a bracketed root search. Widths are $w=1,\ldots,12$. The blue curve fits $\alpha_c(w)=\alpha_\infty-A/(w+b)^p$ using **$w=3,\ldots,12$**. Stored parameters are $\alpha_\infty=2.00001402$, $A=0.40472731$, $b=0.41530477$, and $p=2.00068375$. Fit standard errors are least-squares covariance diagnostics, not trajectory uncertainties; none are drawn. This drift is not a shifted transition of the target band.
- **(c) Flattened-parent gap.** $|\Delta_0(w)-2|$ for the $\sigma^x$ trial basis, where $\Delta_0(w)=\min_{\boldsymbol k,n}|E_n[h_w(\boldsymbol k)]|$ is the half-filling gap; the direct band gap is $2\Delta_0$. The full Brillouin-zone search uses a $512\times512$ grid followed by continuous refinement. The dashed exponential $A_\Delta e^{-w/\xi}$ is fitted in log space for **$w=4,\ldots,12$**, with $A_\Delta=0.41042212$, $\xi=1.35834565$, and $R^2=0.99997515$. The untruncated limit is 2.
- **(d) Retained squared norm.** The fraction of the lower-band $A$-family OW mode's real-space norm inside the square support, relative to its full norm. At $w=1$ the retained fraction is about $99.2155\%$; the dashed line marks $99\%$.

### Construction and data

The Bloch Hamiltonian convention is $\boldsymbol n(\boldsymbol k)=(\sin k_x,\sin k_y,\alpha-\cos k_x-\cos k_y)$. Overcomplete Wannier (OW) modes are obtained from the band projectors acting on $\tau_A=(1,1)^\mathsf T/\sqrt2$ and $\tau_B=(1,-1)^\mathsf T/\sqrt2$. Fourier coefficients use a $1024\times1024$ grid; truncation retains integer cells with $|r_x|,|r_y|\leq w$. In the parent Hamiltonian, the four projected modes (two families, two bands) are individually normalized before their signed rank-one projectors are summed. This normalization is distinct from the scalar plotting normalization in (a). Panel (c) uses the $X$ rows of the Pauli-gap calculation, not its $Y$ or $Z$ rows.

Sources: [OW analysis and renderer](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/analyze_ow_truncation.py), [OW diagnostics](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/data/ow_truncation_diagnostics.csv), [OW fit metadata](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/data/ow_truncation_fit.json), [gap analysis](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/analyze_flattened_pauli_gaps.py), [gap table](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/data/flattened_pauli_gap_vs_w.csv), and [gap fit metadata](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/data/flattened_pauli_gap_analysis.json).

### Reproduction and verification

Run `python sources/plot_ow_truncation.py`. The portable renderer reads the exact native plot arrays captured in [data/ow_truncation](data/ow_truncation), including the original form-factor samples and saved fit curves. It does not recalculate spectra, optimize zero positions, refit curves, or simulate dynamics. The native reconstruction was checked against the original preview before typography changes; original PDFs, PNGs, source code, and campaign inputs remain unchanged. Source and plotted-array checks are recorded with the compact inputs.


---

<a id="figure-a02-mutual-information"></a>

## Figure A2 — Entropy contours and antipodal mutual information

[Vector PDF](Figure_A02_mutual_information.pdf) · [300-dpi preview](Figure_A02_mutual_information.png)

The figure is a **2×1 vertical composition**, 3.375 × 4.8 inches, suitable for one manuscript column. Panel (a) contains the two entropy maps side by side; panel (b) retains the original mutual-information curves, error bars, and geometry inset. The maps and MI sweep are separate ensembles.

### Panel (a): latest all-origin entropy contours

For each $\alpha_1=1,3$, use $N_x=20$, $N_y=32$, $A_y=16$, cycle $T=64$, $S=100$ independent random-pure half-filled Born trajectories, hard walls at $x=5,15$, exterior $\alpha_2=30$, and OW range $n_{\rm shell}=1$. The origin-averaged revision evaluates **all 32 cuts** $y_0=0,\ldots,31$ in every trajectory. Each cut is diagonalized independently to compute its Gaussian entropy contour; rows are aligned by $\delta y=y-y_0$, then averaged over origins within the trajectory and finally over trajectories. No contour is constructed from an averaged correlation matrix.

Both maps share `Blues` with `PowerNorm(gamma=0.5)`, lower limit zero and upper limit **0.4416719467194624**, and ticks 0, 0.10, 0.30, 0.44. The display has no interpolation or smoothing. The map sums are **9.324998607129578** and **1.9343572633145107**, agreeing with the corresponding mean half-strip entropies. Portable per-trajectory maps have shape `[100,16,20]`; means and SEMs have shape `[16,20]`. SEMs are retained in the data but not represented as map error bars. The 3,200 translated cuts are correlated within their 100 independent trajectories.

The source is [make_contour_comparison.py](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/experiment_review/entropy_contour_alpha_comparison/make_contour_comparison.py), run with `--average-origins`, and its `contours_y0avg.npz` and `analysis_manifest_y0avg.json`. The source-data SHA-256 is `aca27b2a586daee6b07bcc87ce3817aede7ef365434c506de92c861fa8a3e541`. Its output is `figures/entropy_contours_hard_n20x32_alpha1_1_3_y0avg.pdf`. This supersedes the fixed-origin map for the present manuscript; the earlier source remains untouched.

### Panel (b): mutual information

The curves show mean endpoint mutual information versus $\alpha_1$ for hard, support-truncated domain walls at $n_{\mathrm{shell}}=1$; the inset identifies the two opposite strips $a,b$ and interfaces. The dashed horizontal reference is $(\log 2)/3$. No fitting is performed. The strip width is labeled $\ell$ to distinguish it from the OW truncation range $w$.

### Data and protocol

- **Geometry:** $N_x=20$, periodic boundaries, interfaces at $x_L=5,x_R=15$, and $N_y=20,24,28$. Each full-$x$ strip has width $\ell=N_y/4=5,6,7$ and the two starts are separated by $N_y/2$. The OW truncation range is independently fixed to one.
- **Circuit:** hard OW-support truncation, $n_{\mathrm{shell}}=1$, exterior $\alpha_2=30$, $X$-basis trial orbitals, random pure half-filled initialization, perfect correction, no postselection, and slab-only raster-$y$ measurements. Production calls the canonical GPU `classA_U1FGTN_gpu.run_markov_circuit` using complex128 arithmetic.
- **Sampling:** $S=100$ independent trajectories per $(N_y,\alpha_1)$ case, observed at exactly $2N_y=40,48,56$ cycles. The 21 values are $\alpha_1=1,1.25,1.5,1.7,1.8,1.85,1.9,1.925,1.95,1.975,2,2.025,2.05,2.075,2.1,2.15,2.2,2.3,2.5,2.75,3$. The campaign evaluated this grid in descending order; the plot uses an increasing axis. Each parameter case has its own trajectories rather than a continuation along the sweep.

There are 63 plotted cases, 6,300 trajectories, and 210 result/completion pairs. The owning campaign is `domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25`. Its completed hard-wall lane contains 1,835 checksum-verified trajectory results imported from the earlier v1 campaign and 4,465 computed in v2; imported values are retained unchanged. No soft-wall curve or numerical ground-state dataset is plotted; the dashed line is the analytic equilibrium comparison.

### Estimator and uncertainty

For every trajectory and strip translation $y_0$, the Gaussian von Neumann entropies are evaluated from the restricted occupation eigenvalues $\nu_j$, in natural-log units:

$$S_A=-\sum_j[\nu_j\log\nu_j+(1-\nu_j)\log(1-\nu_j)],\qquad I_{a,b}=S_a+S_b-S_{a\cup b}.$$

Both orbitals of every cell are included. Periodic translations $y_0=0,\ldots,N_y/2-1$ suffice because the other half exchange $a$ and $b$. Mutual information is formed **for each translation of each trajectory**, then translation averaged within that trajectory, and finally averaged over the 100 trajectories. It is never calculated from an averaged covariance matrix. Error bars are $\pm1$ SEM across the 100 translation-averaged trajectory values (`ddof=1`), not across translations. The entropy routine clamps eigenvalues to $[10^{-12},1-10^{-12}]$ after physicality checks. There is no time averaging or fit window; all values are endpoints.

### Sources, reproduction, and checks

The [report wrapper](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/build_mutual_information_figure.py) imports the owning [plotting and verification engine](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/02_domain_wall_bipartite_mutual_information/make_hard_wall_figures.py). The [observer](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/02_domain_wall_bipartite_mutual_information/mutual_information_observer.py) defines the estimator, and the [runner](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/02_domain_wall_bipartite_mutual_information/run_campaign.py) fixes the protocol. Numerical means and SEMs are in the [63-row summary table](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/02_domain_wall_bipartite_mutual_information/analysis_outputs/hard_wall_nshell1/hard_wall_nshell1_sample_summary.csv). The [download manifest](/home/abhuiyan/class_A_fermionic_adaptive_circuit/00_WORKSPACE/CURRENT/final_production_new_designs/02_domain_wall_bipartite_mutual_information/gpu_data/domain_wall_bmi_nx20_ny20-28_alpha21_desc_c2ny_s100_v2_batched_50-25-25/DOWNLOAD_MANIFEST.json) binds the exact raw NPZ/completion files.

Run **`python sources/plot_mutual_information.py`** to reproduce both panels from compact inputs in `data/mutual_information/`; `sources/mi_geometry.py` reproduces the inset. `composition_validation.json` verifies origin/trajectory averaging, entropy-map sums, all 63 original MI means and SEMs, color normalization, and figure bounds. `notation_updates_provenance.json` binds the unchanged original source files. Only presentation and composition are new; all calculations use saved results.

The original MI-only [PDF](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/figures/hard_wall_nshell1_bmi_with_geometry.pdf) and [PNG](/home/abhuiyan/class_A_fermionic_adaptive_circuit/technical_report/figures/hard_wall_nshell1_bmi_with_geometry.png) remain unchanged. PDF SHA-256 `2a8853d4f1e75b081fed8a185ecf9cea8992e04dac40d41aeb15be35e43fec95`; PNG `4cb501bfdcc899f210e57b45737a4eb10b38c113e5b8543cbbb22d07d931f3a6`. The original campaign verification had checked all 420 result/completion files and all 6,300 trajectory entries, with maximum entropy-identity residual $7.61\times10^{-15}$ and Hermiticity error $5.00\times10^{-16}$.

The finite-size MI peak near $\alpha_1\simeq1.7$ is compared with the truncated-parent critical point $\alpha_c(1)\simeq1.8$ in A1. Their proximity is suggestive, not a determination of a common dynamical transition. Truncation shifts the parent critical point; the untruncated QWZ transition remains at two.

---

<a id="figure-7-section-v"></a>

## Figure 7 in light of Section V of Eisler and Peschel

**Reference.** V. Eisler and I. Peschel, *Solution of the fermionic entanglement problem with interface defects*, arXiv:1005.2144v3, Section V, printed pp. 9–11, especially Eqs. (21)–(23). [Paper, Section V](https://arxiv.org/pdf/1005.2144v3#page=9). The attached copy was read directly. This discussion interprets the saved figure; it does not change its plotted data or add a model fit.

### The connection: a spectral explanation of logarithmic entropy

In Section V, the authors convert a sum over entanglement modes into an integral with a prefactor proportional to $\log L$. To separate their notation from the signed energies in our figure, call their integration variable $u$. Their physical single-particle levels are $2\omega(u)$, and their result has the form

$$
\sum_k\longrightarrow \frac{\log L}{\pi^2}\int_0^\infty du,
\qquad
S=\frac{\log L}{\pi^2}\int_0^\infty s_{\rm f}(2\omega(u))\,du,
$$

where $s_{\rm f}$ is the entropy of one fermionic mode. The interface changes the relation $\cosh\omega=\cosh u/s$, hence the entropy coefficient. The paper treats critical Ising/XX ground states with equilibrium interface defects; it does not derive a formula for this monitored domain-wall ensemble. [Section V, Eqs. (21)–(26)](https://arxiv.org/pdf/1005.2144v3#page=9).

**Our inference:** Figure 7 probes the same spectral mechanism. For $\alpha_1=1$, the energy histogram displays an approximately flat pooled density across the selected signed energy window, including near zero; the $\alpha_1=3$ control has a different shape concentrated toward the window edges. The mode-count panel shows that the mean number of modes in that fixed window grows approximately linearly with log chord length over the chosen fit window. Thus it supplies a microscopic companion to Figure 6's logarithmic entropy and charge-variance curves: increasing strip width brings additional modes into a fixed range of entanglement energies. The evidence is finite-size and ensemble-averaged.

Panels (a,b) compare $\alpha_1=1$ and $\alpha_1=3$ at $N_y=32$, $A_y=16$, pooling all 32 translated cuts in each of 100 trajectories. Panel (a) has unit area over the full occupation range; panel (b) has unit area conditional on the selected energy window. This places partially occupied modes in the context of the full spectrum and resolves the different energy distributions of the two ensembles. Panel (c) preserves the raw number of window modes and its width dependence for $\alpha_1=1$. Geometry, protocol, evolution time, and origin sampling match across the two histogram ensembles; their full-spectrum and conditional density normalizations remain different.

### Exact Gaussian identities and the distinction between count and entropy

For the number-conserving Gaussian states used here, the reduced state can be written

$$
\hat\rho_A=Z_A^{-1}\exp(-\hat K_A),\qquad
\hat K_A=\sum_j\varepsilon_j\hat d_j^\dagger\hat d_j,
\qquad
\varepsilon_j=\log\frac{1-\nu_j}{\nu_j}.
$$

Define the mean spectral measure per cut by averaging origins within each trajectory and then averaging trajectories,

$$
\varrho_{A_y}(\varepsilon)=
\frac{1}{100}\sum_{s=1}^{100}\frac{1}{32}\sum_{y_0=0}^{31}
\sum_j\delta\!\left(\varepsilon-\varepsilon_{s,y_0,j}(A_y)\right).
$$

This is a mode density, not a probability density normalized to one. Panel (b) displays the conditional unit-area density $p_W$. Inside its window, recover the mean number density by $\varrho_{16}(\varepsilon)=\overline N_{0.99}(16)\,p_W(\varepsilon)$. The multiplying mean count is $95398/(100\times32)=29.811875$ for $\alpha_1=1$, or $83326/(100\times32)=26.039375$ for $\alpha_1=3$. Equivalently, divide the archived raw bin counts by $100\times32$ and the bin width. Histogram normalization therefore does not measure the number of entropy-carrying modes on its own; the raw count provides the missing scale.

The exact spectral weights for entropy and charge variance are

$$
\overline S(A_y)=\int_{-\infty}^{\infty}
\varrho_{A_y}(\varepsilon)\,s_{\rm f}(\varepsilon)\,d\varepsilon,
\qquad
s_{\rm f}(\varepsilon)=\log(1+e^{-\varepsilon})+
\frac{\varepsilon}{1+e^{\varepsilon}},
$$

$$
\overline{\operatorname{Var}(Q_A)}=
\int_{-\infty}^{\infty}\frac{\varrho_{A_y}(\varepsilon)}
{4\cosh^2(\varepsilon/2)}\,d\varepsilon.
$$

Both weights emphasize modes near $\varepsilon=0$, corresponding to occupation $\nu=1/2$. In contrast, the count panel applies a flat window weight:

$$
\overline N_{0.99}(A_y)=\int_{-E_c}^{E_c}\varrho_{A_y}(\varepsilon)\,d\varepsilon,
\qquad E_c=\log199.
$$

Occupation and energy histograms also have a useful change-of-variable relation. For identical pooled modes and identical normalization,

$$
\rho_\nu(\nu)=\frac{\rho_\varepsilon(\varepsilon(\nu))}{\nu(1-\nu)},
\qquad
\rho_\lambda(\lambda)=\frac{2\rho_\varepsilon(\varepsilon(\lambda))}{1-\lambda^2}.
$$

Thus a nearly flat energy density can produce pronounced peaks near occupation 0 and 1 purely through the Jacobian. The two representations are complementary: the full occupation histogram includes almost-empty/almost-filled modes, whereas the finite energy window resolves the partially occupied modes more evenly. For the displayed full-spectrum density $\rho_\lambda$ and conditional energy density $p_W$, the selection probability must also be included: $p_W(\varepsilon)=\rho_\lambda(\lambda(\varepsilon))(1-\lambda^2)/(2P(W))$, where $P(W)=M_\alpha/2048000$. This continuous-density relation explains the normalization difference; finite histogram bins also transform nonuniformly.

If the density has a scaling contribution

$$
\varrho_{A_y}(\varepsilon)=\varrho_0(\varepsilon)
+\varrho_1(\varepsilon)\log D(A_y)+\cdots,
$$

then the measured count slope $b$ and the entropy slope $m_S$ are different integrals:

$$
b=\int_{-E_c}^{E_c}\varrho_1(\varepsilon)\,d\varepsilon,
\qquad
m_S=\int_{-\infty}^{\infty}\varrho_1(\varepsilon)s_{\rm f}(\varepsilon)\,d\varepsilon.
$$

Consequently, **$b=1.1874\pm0.0134$ is not itself a central charge**. Its value depends on the spectral window. The normalized histogram at one width measures the shape $\varrho_{16}/\overline N_{0.99}(16)$ within the window; multiplying by the mean count recovers $\varrho_{16}$ there, but does not isolate $\varrho_1$. The count fit also has a large intercept, $a\simeq27.065$, so a width-independent spectral background cannot be neglected or identified with the logarithmic contribution.

As a conditional illustration, an energy-independent $\varrho_1=g$ over the entropy-relevant range gives $m_S=\pi^2g/3$, a charge-variance slope $m_Q=g$, and $b=2E_cg$ in the retained window. This would connect the entropy and charge coefficients in Figure 6. Flatness of the histogram at a single width does **not** establish that assumption for $\varrho_1$, so these relations are not used to extract a new coefficient from Figure 7.

### Scope of the evidence and a useful next comparison

The logarithmic count supports the spectral mechanism behind the entropy scaling; it does not independently establish conformal invariance, chirality, a thermodynamic entanglement gap closing, or the equilibrium defect law $c_{\rm eff}(t)$ from the cited paper. Pooling broadens the spectrum and can conceal gaps in individual realizations. The paper uses nonnegative Ising excitation energies $2\omega$; Figure 7 uses signed number-conserving single-particle energies. Numerical coefficients require matching those counting conventions and the subsystem geometry, rather than identifying the two energy axes directly.

A more quantitative connection would estimate the change in the spectral density with $\log D(A_y)$ from multiple saved widths, integrate that change against the entropy weight, and compare with an entropy fit on the **same trajectories and widths**. This also separates the fixed background. Figure 6 currently uses different endpoint ensembles and system sizes, and the finite energy cutoff omits contributions from nearly empty/full modes, so the current two figures do not implement that quantitative comparison. No new circuit simulation or scaling fit is introduced by this figure update.

**Suggested interpretation for the paper:** For $\alpha_1=1$, the nearly uniform pooled entanglement-energy distribution and the logarithmic growth of the number of modes in a fixed energy window support a spectral origin of the boundary entropy scaling. They complement the direct entropy and charge-fluctuation measurements; the count coefficient is a window-dependent diagnostic rather than an independent central-charge estimate.
