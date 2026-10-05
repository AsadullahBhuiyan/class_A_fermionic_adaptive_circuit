# Truncation and topology note

[Compiled note](truncation_topology.pdf) · [LaTeX](truncation_topology.tex) · [BibTeX](references.bib) · [Numerical receipt](truncation_check.json)

This standalone note collects the discussion of coherent band mixing, the Dubail–Read obstruction, particle-number changes, and the actual square-window OW truncation. It distinguishes analytic sufficient conditions, sampled numerical checks, and the missing control of long-time stochastic trajectories. The manuscript and its bibliography are not modified.

The Fourier derivation transforms the bra and ket separately, retains the normalization factors, and explains the cancellation that yields the auxiliary Hamiltonian. The finite-grid overlap check reproduces the numbers quoted in the note using the existing analytical renderer. No circuit simulations are run.

Section IV A treats the single-cell (`w=0`) limit explicitly: its auxiliary occupied projector is constant and trivial, its target overlap vanishes at Gamma, and the interpolation crosses a zero gap. The numerical receipt also checks the onsite normalization and crossing. Here a cell contains two orbitals; an individual onsite mode, the auxiliary band, and a conditioned trajectory remain distinct objects.

From this directory:

```bash
OPENBLAS_NUM_THREADS=1 python check_truncation.py
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build truncation_topology.tex
cp build/truncation_topology.pdf truncation_topology.pdf
```

The numerical script requires NumPy plus the existing renderer's imports (SciPy, pandas, and Matplotlib). The document requires RevTeX 4.2, standard AMS/LaTeX packages, and BibTeX. To compile elsewhere, copy the `.tex` and `.bib` files together; the numerical inputs are not required to compile the PDF.
