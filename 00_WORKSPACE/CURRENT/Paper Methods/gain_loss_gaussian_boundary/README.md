# Gain and loss as Gaussian boundary limits

A concise, independent consolidation for eventual manuscript use. The original
notes, manuscript, data, and dynamics implementations are not modified.

## Deliverables

- `gain_loss_gaussian_boundary.tex`: one-column RevTeX derivation.
- `build/gain_loss_gaussian_boundary.pdf`: compiled note.
- `references.bib`: foundational sources; algebraic extensions are derived here.
- `manuscript_insert.tex`: short, macro-independent proposed appendix paragraph;
  not inserted into the manuscript.
- `main_text_sentence.tex`: proposed one-sentence pointer to that appendix.

Build from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build gain_loss_gaussian_boundary.tex
```

Run the independent finite-dimensional algebra checks from the repository root:

```bash
python -m pytest -q tests/test_gain_loss_gaussian_boundary.py
```

No production simulation, numerical dataset, or existing plot is needed.
Single kets follow the user's requested presentation for the theory write-ups.
The exact endpoint section includes all four fill/deplete Kraus outcomes,
not just the particle-changing outcomes.

## Manuscript recommendation

Keep one short statement in the main text only if needed to connect odd feedback
to the transfer-matrix framework. Put `manuscript_insert.tex` in the existing
gain/loss appendix after its Kraus-operator and U(1) discussion. The supporting
derivation is Sections II–V of this note. Do not add the full chart taxonomy or
claim a new symmetry classification, a critical exponent, or a convergence
theorem from this construction.

## Consolidated sources and selection

1. Repository: `00_WORKSPACE/LEGACY/markov_transfer_operators/`
   `free_fermion_trajectory_lyapunov_cft.tex`, especially VI D–K.
2. User attachment `gain_regularization_vs_POVM_preview.pdf` (August 4, 2026):
   polar decomposition, weak measurement, scalar normalization.
3. User attachment `gain_loss_note_v2.pdf` (August 10, 2026):
   chronological occupied-projector reasoning and nonzero-branch limits.

The attached files remain in their original attachment location; this document
does not depend on them to build. The retained results have explicit elementary
derivations and independent small-Fock-space checks. This is not a certification
of every theorem or numerical-validation claim in the source documents.

Changes of scope/corrections made during consolidation:

- Boundary means a projective operator/normalized-state limit, not a finite
  singular matrix in the ordinary closure of the orthogonal group.
- The flip-both POVM is only one completion. Outcome-dependent feedback instead
  gives finite-strength instruments approaching the actual fill/deplete pairs.
- Odd Kraus branches preserve parity-even density operators; they can change
  their parity sector. They require parity exchange in a conserving dilation.
- Quadratic convergence applies to normal correlations under jump-only
  regularization and fixed-charge input; anomalous/full covariances generally
  have linear corrections. Regularized projectors need not have the quadratic
  improvement.
- Zero exact branches, arbitrary-depth uniformity, interchange of regulator and
  long-time limits, and universal subtraction of divergent Lyapunov rates are
  excluded.
- Broad odd-sector extensions of POVM no-go theorems, general singular-chart
  Schur-complement formulas, and the long note's overly strong Lemma XII.3 are
  not used. A same-mode GLGL word is a counterexample to that lemma as stated.
- BPJ's transpose convention is explicit; branch composition and sTM matrix
  multiplication have explicitly different orders in the row-index convention.

Existing local changes elsewhere in the repository were left untouched.

## Verification on October 7, 2026

The 40 algebra tests pass (one to four complex modes). They check the polar
identity, Majorana conjugation and product order, singular values, all three
instrument completions, parity, pure-state gain/loss and projector updates,
nonunitary Slater norms, nonzero and zero-branch limits, and the endpoint
many-body spectrum. Numerical agreement is a cross-check of the displayed
derivations, not a substitute for their assumptions.

The six-page PDF builds with resolved citations/references and no overfull or
underfull boxes. Every page was visually inspected; fonts are embedded. No
production simulations, commits, or pushes were performed.
