# U(1) Kac–Moody level and Rényi entanglement

A standalone, one-column pedagogical theory note following BPJ's appendix
notation and derivation-led exposition. It assumes graduate-level quantum
mechanics and linear algebra and introduces the needed conformal-field-theory
steps explicitly.

The note derives Gaussian reduced-state entropies, charge counting statistics,
and conditional versus mixture averages, then develops the continuum physics:

- Chiral and nonchiral free-fermion Hamiltonians, real-time and Euclidean
  actions, and the bosonization dictionary.
- Ordinary and chiral Luttinger liquids, their charge normalization, and why
  the ordinary Luttinger parameter, the edge K matrix, and the physical
  current level are different objects.
- Primary fields, conformal weights, Ward identities, replica twist dimensions,
  and finite-cylinder entropies.
- The full non-Abelian affine Kac–Moody algebra, central extension and grading,
  the U(1) specialization, representations, and Virasoro/Sugawara relations.
- Regulated current-mode sums, contact terms, charge zero modes, and the
  susceptibility/modular-energy kernel relations.
- The Eisler–Peschel interface-defect example, especially arXiv:1005.2144v3
  Eqs. (21)–(32), supplemented by their 2012 Rényi calculation. The derivation
  distinguishes a flat reference-energy measure from the gapped, nonflat
  physical modular spectrum, and an effective entropy coefficient from
  bulk central charge or current level.

It ends with one-wall/two-wall normalization, signed versus unsigned
central charges and levels, and the assumptions behind wall-integrated
entanglement contours. The presentation uses the amended conventions in the
user-supplied six-page bosonization cheat sheet (especially pp. 2–4):
Phi_± = (Phi ∓ varphi)/2, psi_± proportional to the normal-ordered
exp(±i sqrt(4 pi) Phi_±), and rho_± = derivative(Phi_±)/sqrt(pi).
The cross-chiral commutator supplies the relative fermion sign; no independent
Klein factors are added on top of it. All conventions are stated in the note,
so the handwritten PDF is not a build dependency.

Exact Gaussian identities are distinguished from infrared CFT assumptions.
The canonical source and PDF retain their historical filenames, but the note
is now theory only: it contains no campaign plans, numerical audits, fitted
results, or implementation instructions.
The continuum models are comparison theories, not an assumed Hamiltonian for
the adaptive protocol. The defect transmission is not identified with its
hard-wall support parameter.

## Files and build

- Source: kac_moody_renyi_validation.tex
- Bibliography: references.bib
- Canonical PDF: build/kac_moody_renyi_validation.pdf

Compile from this directory:

~~~bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build kac_moody_renyi_validation.tex
~~~

Only the source and bibliography are needed. The PDF does not load generated
figures, tables, simulation outputs, or the analysis manifest.

Small, data-independent algebra checks are in
tests/test_pedagogical_theory_notes.py and
tests/test_kac_moody_pedagogical_extensions.py at the repository root.
They cover Gaussian conventions, kernel limits, defect spectra and integrals,
and regulated current fluctuations. No production simulations are required.

## Existing numerical material

The historical build_validation_figures.py, analysis_manifest.json, figures/,
and tables/ remain unchanged as separate numerical artifacts. They are not
dependencies of the pedagogical note, and their historical conclusions should
not be interpreted as an assessment of a different or newer dataset.
The previous numerical write-up remains available in Git history.

The companion ../adaptive_entangling_capacity/adaptive_entangling_capacity.tex
derives the branchwise entangling-capacity and conditional preparation-depth
bounds. Each document can be read and compiled independently.
