# Truncated OW bulk-topology note

`ow_truncation_topology.pdf` is a two-page, one-column RevTeX working note;
`ow_truncation_topology.tex` is its self-contained source. The main technical
report, production sources, and numerical results are unchanged.

The derivation concerns the translation-invariant bulk reference projector,
not the monitored stationary ensemble. It gives the exact untruncated
tight-frame identity, an operator-norm bound from normalized real-space
truncation tails, a gapped Riesz-projector interpolation, and integer Chern
invariance. The sufficient truncation bound is uniform in volume. No claim
that n_shell=1 meets it is made.

The imaginary-time section establishes the algebraic connection between weak
occupation filters and desired-outcome projectors. It distinguishes the
lower-frame reference Hamiltonian from the two-band occupation-penalty
Hamiltonian, and states why a deformation to an arbitrary monitored record
does not follow: noncommuting strong filters, singular Kraus operators,
charge-changing feedback, and the need to control bulk locality and a gap.
The monitored-ensemble extension is explicitly a conjectural direction.

Build from this directory (repeat until references and TOC widths stabilize):

```sh
pdflatex -interaction=nonstopmode -halt-on-error ow_truncation_topology.tex
pdflatex -interaction=nonstopmode -halt-on-error ow_truncation_topology.tex
pdflatex -interaction=nonstopmode -halt-on-error ow_truncation_topology.tex
```

No BibTeX step is needed. References are BPJ (2026), Kitaev (2006, Appendix
C.3), and Goldstein (2019). Validation: exactly two pages, no unresolved
references or overfull boxes, and rendered-page inspection.
