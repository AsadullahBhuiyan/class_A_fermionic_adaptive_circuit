Overleaf manuscript copy — 2026-10-06

Upload the ZIP as a new Overleaf project. Set manuscript.tex as the main document and use pdfLaTeX. The project uses standard TeX Live packages, including REVTeX 4.2 and BibTeX with apsrev4-2.

Files:
- manuscript.tex: current manuscript, with the figure search path changed to figures/.
- references.bib: complete, unchanged bibliography.
- revtex_float_placement.tex: required local float-placement helper.
- figures/: all 16 current vector PDF versions; 13 are included in the manuscript.
- previews/: matching 300-dpi PNG previews, not used for typesetting.
- manuscript.pdf: independently compiled reference copy.

The three retained but non-included PDFs are Figure_02_adaptive_circuit.pdf, Figure_02_hard_wall.pdf, and Figure_02_alt_soft_and_hard_walls.pdf. Asset filenames intentionally differ from the automatically assigned manuscript figure numbers.

All revision colors, comments, equations, captions and figure inclusion sizes are preserved. Figure 1 uses 0.75 column width. Figure 2 includes the latest panel-(a) left-edge alignment.

Build locally: latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
