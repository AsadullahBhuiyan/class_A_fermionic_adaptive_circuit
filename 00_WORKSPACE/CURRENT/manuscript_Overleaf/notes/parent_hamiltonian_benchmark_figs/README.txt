Parent Hamiltonian Benchmark Figs

Open parent_hamiltonian_benchmark_figs.pdf for the standalone illustrated note.
Its LaTeX source, nine PDF/300-dpi PNG figure groups, Chern table, compact data,
fit summaries, and independent numerical checks are in this directory.

The parent is the clean signed sum of canonical normalized OW projectors:
nshell=1, alpha2=30, X trial orbitals, hard walls, periodic x and y, zero twist,
exact half filling. The dispersive parent spectrum is retained. No dynamics,
disorder, trajectory sampling, or additional spectral flattening is performed.

Run from this directory with the repository Python environment:

    python benchmark.py --threads 4
    python analyze.py --threads 4
    python validate.py --threads 4
    python render.py
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=build parent_hamiltonian_benchmark_figs.tex
    python verify.py --record

The final command copies the compiled PDF to this directory and records delivery
checksums. Run python verify.py without --record for a read-only final audit.
The computation reuses a case only if its configuration, source identities, byte
count, and SHA-256 match its JSON receipt. CPU threads are explicitly configurable.

The independent validation includes a directly assembled small parent, the
canonical local-marker and three-sector Chern implementations, the full-space
correlation definition, entropy/charge identities, and a small exact Fock-space
check of the finite-temperature transpose and factor of two. The complete record
is data/validation.json. Modular cutoff checks are data/modular_checks.json.

Important comparisons are intentionally retained in the note: the strictly
periodic lowest gap stays near 7.51e-10 at Nx=20; antipodal normalization gives a
poor chord collapse despite the earlier clean periodic cotangent correlator
being reproduced; and the sharp restricted-mode count plateaus at 30. See
data/prior_protocol_comparison.json and data/prior_periodic_comparison.json for
common-input comparisons with prior equilibrium results. None of the earlier
products, original manuscript files, or figure assets is modified.

The shared manuscript typography module is loaded as an isolated module instance
and its output/document paths point to this note. Its 510 pt text width is checked
against the actual RevTeX build; each figure is included at half that width.
All ordinary text and mathematics use LaTeX and embedded Computer Modern/AMS fonts.

Dependencies: Python, NumPy, SciPy, Matplotlib, Pillow, tqdm, threadpoolctl, the
canonical classA_U1FGTN dependencies, CPU Torch for the independent Chern observer
check, LaTeX/RevTeX, latexmk, dvipng, and Poppler (pdfinfo, pdftotext, pdffonts).
