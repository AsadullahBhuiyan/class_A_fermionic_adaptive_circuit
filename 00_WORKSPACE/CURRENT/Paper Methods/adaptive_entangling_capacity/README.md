# Entangling capacity of a local adaptive fermion circuit

This directory contains a standalone PRX Quantum-style internal technical note on the
branchwise entanglement-growth bound for local adaptive circuits and its application to
the class-A overcomplete-Wannier measurement-and-fSWAP controller.

The note proves the general operator-Schmidt-rank bound, derives the explicit ranks of
the four corrected fermionic occupation branches, and explains why a critical interface
embedded in a two-dimensional circuit obeys rather than contradicts the
Lu--Lessa--Kim--Hsieh bound.  The interface-CFT consequences are stated conditionally;
the note does not claim to prove wall criticality or a convergence rate.

## Canonical sources

- `00_WORKSPACE/CURRENT/prxq_draft/appendices.tex`: current branchwise and
  circuit-specific derivations.
- Lu, Lessa, Kim, and Hsieh, PRX Quantum **3**, 040337 (2022), Appendix D.
- Bhuiyan, Pan, and Jian, Physical Review Research **8**, 023147 (2026).

## Build

Compile from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build adaptive_entangling_capacity.tex
```

The canonical PDF is `build/adaptive_entangling_capacity.pdf`.  The main text is limited
to five two-column pages; references begin after an explicit page break and are excluded
from that limit.
