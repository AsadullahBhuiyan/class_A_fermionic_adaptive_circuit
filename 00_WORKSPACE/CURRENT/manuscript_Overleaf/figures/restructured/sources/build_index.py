#!/usr/bin/env python3
"""Build the ordered Markdown index and overview from delivered figure assets."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / 'data/figure_index.json').read_text())
    canvas = Image.new('RGB', (2400, 2620), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        font = ImageFont.truetype(font_path, 23)
        heading = ImageFont.truetype(font_path, 38)
    except OSError:
        font = ImageFont.load_default(size=23)
        heading = ImageFont.load_default(size=38)
    draw.text((35, 22), 'Restructured manuscript figures', font=heading, fill='#15232d')
    for i, row in enumerate(rows):
        col, line = i % 4, i // 4
        x, y = 20 + col * 595, 95 + line * 625
        draw.rounded_rectangle((x, y, x+580, y+605), radius=8, outline='#cbd4dc', width=2)
        label = f"Figure {row['number']}"
        draw.text((x+16, y+12), label, font=font, fill='#15232d')
        with Image.open(ROOT / f"{row['stem']}.png") as im:
            im = im.convert('RGB')
            im.thumbnail((550, 542), Image.Resampling.LANCZOS)
            canvas.paste(im, (x+(580-im.width)//2, y+53+(542-im.height)//2))
    canvas.save(ROOT / 'overview.png', dpi=(300, 300))

    lines = ['# Restructured manuscript figures', '',
             'Fourteen figure versions: eleven main-text figures, one Figure 2 alternative, and two appendix figures. '
             'Each has a vector PDF and 300-dpi PNG preview. All data, protocols, analysis, and reproduction details are collected in one combined note. '
             'The manuscript LaTeX, bibliography, original figures, and campaign data were preserved.', '',
             '[Combined figure notes](FIGURE_NOTES.md) · [Open the overview](overview.png) · [Verification](validation.json)', '',
             '## Ordered index', '',
             '| New | Figure | Previous figure | Change | Files |',
             '|---|---|---|---|---|']
    for row in rows:
        stem = row['stem']
        anchor = stem.lower().replace('_', '-')
        links = f'[PDF]({stem}.pdf) · [Preview]({stem}.png) · [Notes](FIGURE_NOTES.md#{anchor})'
        lines.append(f"| {row['number']} | {row['title']} | {row['old_figure']} | {row['change']} | {links} |")
    lines += ['', 'The old Figure 11 tri-junction is incorporated into new Figure 3. '
              'Old Figures 12–14 and the old Figure 4C occupation panel are excluded from this collection; Figure 7 contains a new matched-parameter occupation comparison. '
              'Their original manuscript assets remain in the parent directory.', '',
              '## Data and interpretation', '',
              '- Figure 3 uses a 3×1 vertical layout and the complete S=100 square-system Chern analysis. Its disk estimator and local-marker map are different observables.',
              '- Figure 4 intentionally combines full-system measurements in A/B/D with the separate slab-only size sweep in C.',
              '- Figure 7 uses a 3×1 vertical layout at Ny=32. Both histograms pool all 32 cut origins and 100 trajectories per parameter value: full-range occupation densities (2,048,000 observations each) and conditional energy densities (95,398 retained for alpha_1=1; 83,326 for alpha_1=3), each normalized to unit area. Panel C preserves the raw mean mode count for alpha_1=1, its fit, and trajectory SEM. The combined note discusses Section V of Eisler and Peschel (2010).',
              '- Figure 9 uses two-column wall windows. Its note records the mismatch with the preserved manuscript caption.',
              '- Figure 8 uses a 2×1 vertical layout and separate ensembles: cycle-panel bars are regression errors, while endpoint-panel bars are trajectory SEMs.',
              '- Figure 11 uses a 4×1 vertical layout: mean-state occupation spectrum, squared mean-state correlator, dimensionless channel gap g_C = 1 − ρ(A)² versus alpha_1 at fixed Nx=20, and direct inverse-Ny fits of g_C. The note identifies both replacement PDFs and relates this multiplier gap to the logarithmic rate used in the linked proof.', '',
              '## Separate diagnostics', '',
              '[Normalized entanglement-energy comparison, alpha_1=1 versus 3](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.pdf) · '
              '[Preview](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.png) · '
              '[Data, normalization, and interpretation](FIGURE_NOTES.md#normalized-energy-comparison). '
              'Both curves use Ny=32, Ay=16 and the same fixed-origin cut; each integrates to one inside the window |lambda| <= 0.99. '
              'This diagnostic is separate from Figure 7. Reproduce it with `python sources/plot_energy_window_comparison.py`.', '',
              '[Normalized mode fraction versus log chord length](diagnostics/normalized_mode_fraction_vs_log_chord.pdf) · '
              '[Preview](diagnostics/normalized_mode_fraction_vs_log_chord.png) · '
              '[Notes](FIGURE_NOTES.md#normalized-mode-fraction). '
              'Standalone Figure 7(c) variant for alpha_1=1: window count divided by all 2NxAy subsystem modes, '
              'averaged over 32 origins within each of 100 trajectories. The existing count fit is divided by the same denominator. '
              'Reproduce it with `python sources/plot_normalized_mode_fraction.py`.', '',
              '## Reproduction', '',
              'From this directory, run the corresponding renderer below. Modified figures use only the compact inputs in `data/`; no simulation is run. '
              'Unchanged/reused figures can also be restored from their checksum-verified source assets in this repository. '
              'The original soft/hard-wall alternative is retained inside the bundle for restoration.', '',
              '| Figure | Command |', '|---|---|']
    for row in rows:
        if row['renderer'] != 'restore_existing.py':
            lines.append(f"| {row['number']} | `python sources/{row['renderer']}` |")
    lines += ['', 'Restore one reused figure with `python sources/restore_existing.py --figure Figure_01_schematic`; '
              'omit `--figure` to restore all nine unchanged/reused assets. Figures 8 and 11 are regenerated by their plotting scripts to preserve the new layouts and panels. Use `--output-dir /tmp/figure-preview` to redirect restoration. '
              'Figures 5, 6, 8, 9, and 11 also support `--output-dir` for isolated plotting. Figures 3 and 7 write beside their bundled sources.', '',
              'Requirements: Python, NumPy, Matplotlib, and Pillow for the overview. '
              'Figure 7 additionally requires LaTeX and Poppler (`pdftoppm`); its TeX compatibility file is included. '
              'Font choices are recorded in the plotting scripts. Existing figures retain their selected appearance.', '',
              'Regenerate this index and overview with `python sources/build_index.py`. '
              'Check the bundle with `python sources/verify_bundle.py`. '
              'Data provenance and numerical checks are saved in the per-figure `data/` directories. '
              'The top-level `manifest.json` binds the delivered assets to checksums.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
