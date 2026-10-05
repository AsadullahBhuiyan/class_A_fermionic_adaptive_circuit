#!/usr/bin/env python3
"""Build the manuscript figure index and overview from the delivered assets."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / 'data/figure_index.json').read_text())
    canvas = Image.new('RGB', (2400, 2620), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = str(Path(matplotlib.get_data_path()) / 'fonts/ttf/cmr10.ttf')
    font = ImageFont.truetype(font_path, 23)
    heading = ImageFont.truetype(font_path, 38)
    draw.text((35, 22), 'Manuscript figure bundle', font=heading, fill='#15232d')
    for i, row in enumerate(rows):
        col, line = i % 4, i // 4
        x, y = 20 + col * 595, 95 + line * 625
        draw.rounded_rectangle((x, y, x+580, y+605), radius=8, outline='#cbd4dc', width=2)
        label = f"Figure {row['number']}" if row['included'] else row['number']
        draw.text((x+16, y+12), label, font=font, fill='#15232d')
        with Image.open(ROOT / f"{row['stem']}.png") as im:
            im = im.convert('RGB')
            im.thumbnail((550, 542), Image.Resampling.LANCZOS)
            canvas.paste(im, (x+(580-im.width)//2, y+53+(542-im.height)//2))
    canvas.save(ROOT / 'overview.png', dpi=(300, 300))

    lines = ['# Manuscript figure bundle', '',
        'Fifteen figure versions, each with a vector PDF and 300-dpi PNG: Figures 1–2 belong to Section II, Figures 3–11 to Section III, and A1–A2 are the only appendix figures. The standalone hard-wall close-up and soft/hard-wall alternative are retained but excluded from the manuscript. Filenames retain their stable bundle numbers; the index shows current manuscript numbers. Original figures, the earlier `restructured` bundle, and campaign data remain unchanged; this bundle accompanies the authorized manuscript revision.', '',
        '[Combined figure notes](FIGURE_NOTES.md) · [Ordered overview](overview.png) · [Verification](validation.json)', '',
        '## Ordered index', '',
        '| Figure | Section | Contents | Previous figure | Change | Files |',
        '|---|---|---|---|---|---|']
    for row in rows:
        stem = row['stem']
        anchor = stem.lower().replace('_', '-')
        links = f'[PDF]({stem}.pdf) · [Preview]({stem}.png) · [Notes](FIGURE_NOTES.md#{anchor})'
        lines.append(f"| {row['number']} | {row['section']} | {row['title']} | {row['old_figure']} | {row['change']} | {links} |")
    lines += ['', 'Old Figures 12–14 and the old Figure 4(c) are excluded. The old tri-junction schematic is incorporated into manuscript Figure 3 (asset Figure_03); manuscript Figure 7 (asset Figure_07) contains the replacement occupation and entanglement-energy comparisons.', '',
        '## Data and interpretation', '',
        'The figure numbers in the following provenance summary are stable bundle IDs (the numeric filename prefixes), not the renumbered manuscript labels. Use the index above for current manuscript numbers.', '',
        '- Figure 3 uses the complete S=100 square-system Chern analysis. The finite-disk estimator and local marker are distinct observables; the marker retains periodic-coordinate seam effects.',
        '- Figure 4 retains its original purification data and independent gap-size ensemble. Scientific protocol details remain in the combined note.',
        '- Figure 7 pools all 32 strip origins within each of 100 trajectories per parameter value at Ny=32. The full occupation histograms contain 2,048,000 observations each; the conditional energy histograms contain 95,398 and 83,326 retained observations for alpha1=1 and 3. Both histogram types integrate to one. Panel (c) retains the raw mean window count for alpha1=1 and its unchanged fit and trajectory SEM.',
        '- Figure 8 uses separate ensembles. Panel (a) bars are regression errors of mean-entropy fits; panel (b) bars propagate trajectory sampling fluctuations.',
        '- Figure 9 integrates two-column wall windows x=5,6 and x=14,15. Its displayed Ay=1 points are excluded without changing the fits.',
        '- Figure 10 retains snapshots at modular times 0, 0.1, and 0.2. Absolute direction depends on the explicitly stated correlation-matrix index convention; the numerical curves are preserved.',
        '- Figure 11 uses deterministic outcome-averaged dynamics and the dimensionless channel gap g_C=1−rho(A)^2. The inverse-length fits use fixed Nx=20; they are not a simultaneous two-dimensional thermodynamic extrapolation.',
        '- Figure A2(a) compares origin-averaged half-strip entropy contours for alpha1=1 and 3 at Nx=20, Ny=32, Ay=16, cycle 64: average all 32 origins within each trajectory, then average 100 trajectories. The maps share a square-root color scale. Panel (b) preserves the 63-point mutual-information scan and trajectory SEMs.', '',
        'All input data, fit windows, averaging order, uncertainties, exclusions, and notation conventions are documented in the combined note. No new circuit simulations were used.', '',
        '## Reproduction', '',
        'Run the matching renderer from this directory. Renderers use bundled compact data and write only inside this bundle. The original large trajectory datasets are not needed for plotting.', '',
        '| Figure | Command |', '|---|---|']
    for row in rows:
        command = f"python sources/{row['renderer']}"
        if row['stem'] == 'Figure_02_adaptive_circuit':
            command += ' --only circuit'
        elif row['stem'] == 'Figure_01_schematic':
            command += ' --only geometry'
        if row['renderer'] == 'restore_existing.py':
            command += f" --figure {row['stem']}"
        lines.append(f"| {row['number']} | `{command}` |")
    lines += ['',
        'All fifteen versions now use dedicated renderers to preserve their typography updates. `restore_existing.py` rejects restoring an outdated figure over these products; its `--check-only` mode verifies original source assets without writing files.', '',
        'Requirements: Python, NumPy, Matplotlib, Pillow, pypdf, Poppler, and a working LaTeX installation with AMS, bm, type1cm/type1ec (cm-super), and dvipng. All figure text and mathematics are rendered through LaTeX in Computer Modern; no font fallback is allowed. The shared sources/manuscript_typography.py enforces final-print sizes: axes and panel letters 9 pt, ticks/legends/annotations 8 pt, prominent schematic labels 10–11 pt. The non-included hard-wall close-up retains its historical 0.8-column print-size calibration. Per-figure records are in data/typography/. Figures 3, 7, and 11 retain their stacked layouts; A1 retains its original canvas aspect ratio.', '',
        'After intentional edits, rebuild this index and overview with `python sources/build_index.py`, then record the validated bundle with `python sources/verify_bundle.py --record`. Use `python sources/verify_bundle.py` for a read-only check. The verifier checks the original assets against `notes/manuscript_revision/baseline/original_figure_checksums.json`; it permits the authorized manuscript and bibliography revision. The top-level `manifest.json` binds the delivered bundle to SHA-256 checksums.', '',
        '## Separate diagnostics', '',
        '[Earlier fixed-origin energy comparison](diagnostics/normalized_entanglement_energy_alpha1_1_vs_3.pdf) and [normalized mode-fraction diagnostic](diagnostics/normalized_mode_fraction_vs_log_chord.pdf) are retained for reference. Neither is included in the manuscript; the spectrum figure (asset Figure_07, manuscript Figure 7) uses all origins and raw mean counts.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
