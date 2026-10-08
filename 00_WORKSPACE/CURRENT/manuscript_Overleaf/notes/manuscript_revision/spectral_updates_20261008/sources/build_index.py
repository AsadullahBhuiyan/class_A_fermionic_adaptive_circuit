#!/usr/bin/env python3
"""Build the manuscript figure index and overview from the delivered assets."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = json.loads((ROOT / 'data/figure_index.json').read_text())
    canvas = Image.new('RGB', (2400, 120 + 625 * ((len(rows) + 3) // 4)), 'white')
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

    included = [r for r in rows if r['included']]
    lines = ['# Manuscript figure bundle', '',
        f"{len(included)} manuscript figures and {len(rows)-len(included)} archived or standalone versions. Figures 1–10 are in the main text; A1–A3 are in the four appendices. Stable asset filenames are independent of manuscript numbering.", '',
        '[Combined figure notes](FIGURE_NOTES.md) · [Overview including archived assets](overview.png) · [Verification](validation.json)', '',
        '## Ordered index', '', '| Figure | Section | Contents | Change | Files |', '|---|---|---|---|---|']
    for row in rows:
        stem = row['stem']; anchor = stem.lower().replace('_','-')
        links = f'[PDF]({stem}.pdf) · [Preview]({stem}.png) · [Notes](FIGURE_NOTES.md#{anchor})'
        lines.append(f"| {row['number']} | {row['section']} | {row['title']} | {row['change']} | {links} |")
    lines += ['', '## Compression revision, 7 October 2026', '',
        'Figure 7 retains the occupation and conditional-energy histograms. Window counts and their original fit remain in compact data and the dated review bundle, but are omitted from the manuscript.',
        'Figure 8 adds the saved origin-averaged half-strip contour comparison above the unchanged wall fits. Maps use 100 trajectories and all 32 origins at cycle 64, with a shared square-root color scale. Fit points, slopes, covariance errors, shading and half-strip axis labels are unchanged.',
        'Figure A1 retains only the native parent-gap and retained-norm arrays. The form-factor and transition panels are preserved in the dated review bundle.',
        'Figure A2 displays saved slab-only Ny=30 histories through 4Ny, with 100 trajectories and one trajectory SEM. The full-measurement protocol is not pooled. Every mean and SEM is verified against saved sample rows, and g_mod=2t Delta is checked. No simulation, time fit or asymptotic extrapolation is used.',
        'Figure A3 retains the pure-state c_eff convergence plot unchanged; its initialization and regression errors differ from the purification diagnostic. The mutual-information and parameter-scan figures remain archived and are excluded from the manuscript.',
        'Other figures, correlation conventions, minor ticks, averaging order, scientific arrays and retained fit values are preserved. Parameter colors are defined in sources/manuscript_palette.py. The complete original manuscript, figures and source scripts are preserved in notes/manuscript_revision/compression_review_20261007 at the manuscript root.', '',
        '## Reproduction', '',
        'Run the matching sources/plot_*.py renderer with the compact bundled data. Renderers require NumPy, Matplotlib, Pillow, Poppler, LaTeX and dvipng. All final figure fonts are Computer Modern/AMS through LaTeX; receipts record printed font sizes and bounds.',
        'After an intentional revision, run python sources/build_index.py and python sources/verify_bundle.py --record. Run python sources/verify_bundle.py for a read-only audit. The manifest binds the delivered bundle to SHA-256 checksums; the verifier also checks pre-revision numerical products and the protected outlook.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))

if __name__ == '__main__': main()
