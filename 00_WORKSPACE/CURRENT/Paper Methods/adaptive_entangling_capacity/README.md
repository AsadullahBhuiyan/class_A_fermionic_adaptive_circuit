# Entangling capacity of a local adaptive fermion circuit

A standalone, one-column pedagogical theory note following BPJ's appendix
notation and derivation-led exposition. It assumes graduate-level quantum
mechanics and linear algebra. There is no fixed page limit.

The note develops conditional trajectories and Schmidt rank, derives the
operator-Schmidt capacity bound, converts it into a physical-depth bound, and
specializes it to finite-range overcomplete-Wannier occupation measurements with
fresh-ancilla fermionic-swap correction. Its interface application is conditional
on product-state preparation and conformal entropy scaling; it does not prove
criticality, convergence, or constant-depth preparation.

## Files and build

- Source: adaptive_entangling_capacity.tex
- Bibliography: references.bib
- Canonical PDF: build/adaptive_entangling_capacity.pdf

Compile from this directory:

~~~bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build adaptive_entangling_capacity.tex
~~~

The cut-geometry schematic is drawn directly in LaTeX. No numerical datasets or
simulation runs are needed to build the note.

## References and companion note

The central rank argument follows Lu, Lessa, Kim, and Hsieh, PRX Quantum 3,
040337 (2022), Appendix D. The fermionic conventions and circuit motivation
follow Bhuiyan, Pan, and Jian, Physical Review Research 8, 023147 (2026).
The fresh-ancilla idealization is explicitly distinguished from BPJ's finite bath.

The companion ../kac_moody_renyi_validation/kac_moody_renyi_validation.tex
derives the entropy and current-algebra normalizations used in the interface
application. Each document can be read and compiled independently.
