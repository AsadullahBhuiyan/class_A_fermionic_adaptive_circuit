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
        f"{len(included)} manuscript figures and {len(rows)-len(included)} archived or standalone versions. Figures 1–12 are in the main text; A1–A2 are in the four appendices. Stable asset filenames are independent of manuscript numbering.", '',
        '[Combined figure notes](FIGURE_NOTES.md) · [Overview including archived assets](overview.png) · [Verification](validation.json)', '',
        '## Ordered index', '', '| Figure | Section | Contents | Change | Files |', '|---|---|---|---|---|']
    for row in rows:
        stem = row['stem']; anchor = stem.lower().replace('_','-')
        links = f'[PDF]({stem}.pdf) · [Preview]({stem}.png) · [Notes](FIGURE_NOTES.md#{anchor})'
        lines.append(f"| {row['number']} | {row['section']} | {row['title']} | {row['change']} | {links} |")
    lines += ['', '## Current revision, 8 October 2026', '',
        'Figures 1 and 6 retain their full-width horizontal layouts. Figure 3 pairs raw total purification entropy with the alpha1=1 cycle60 full-system entropy contour; both axes now have the same width and height, and the alpha label remains centered. Figures 4 and 5 restore the original single-column vertical stacks. Figure 8 restores the original single-column convergence plot and inset, and Figure 9 restores the two vertically stacked wall fits. PDF placement checks enforce at most one vertical figure stack per page.',
        'Figure 7 retains the occupation and conditional-energy histograms and restores window counts for both phases. The original topological fit is unchanged; trivial-control counts come from saved endpoint frames.',
        'Figures 6(a,b) and 9(a,b) use the fresh endpoint campaign with Ny=24,28,32,40,50,60, 100 independent trajectories per size at 2Ny cycles. Half-strip labels use Ny/2 exactly. Joint fits and errors match the imported per-trajectory data, including covariance between widths. Figure 8 presents independent convergence histories. Figure 11 contains the ordered spectrum and entropy contour of the averaged Gaussian state. Figure 12 separately contains the correlator and channel-gap fits, relabeled (a,b). The contour is evaluated from the averaged state, not averaged trajectory contours. Figure 9 contains only the two vertically aligned wall fits; the contour comparison is excluded.',
        'Figure A1 retains only the native parent-gap and retained-norm arrays. The form-factor and transition panels are preserved in the dated review bundle.',
        'Figure A2 displays saved slab-only Ny=30 histories through 4Ny, with 100 trajectories and one trajectory SEM. The full-measurement protocol is not pooled. Every mean and SEM is verified against saved sample rows, and g_mod=2t Delta is checked. Panel (c) compares the seven-size half-secant growth rates over 3Ny–4Ny with the existing 2Ny rates, with paired trajectory SEMs and SEM-weighted power-law fits. No simulation or asymptotic extrapolation is used.',
        'Figure 8 contains the pure-state c_eff convergence curves and inset as its own figure. The c_eff(t) axis label is simplified; its extraction is defined in the caption. Its initialization and regression errors differ from the purification diagnostic. The mutual-information and parameter-scan figures remain archived and are excluded from the manuscript.',
        'Other figures, correlation conventions, minor ticks, averaging order, scientific arrays and retained fit values are preserved. Parameter colors are defined in sources/manuscript_palette.py. The complete original manuscript, figures and source scripts are preserved in notes/manuscript_revision/compression_review_20261007 at the manuscript root.', '',
        '## Separate previews and analysis, 8 October 2026', '',
        'The occupation-density previews and late-gap slope comparison are in deliverables/spectral_updates_20261008 at the manuscript root. The occupation-density previews remain excluded. The late-window comparison is now incorporated in Appendix C with a main-text pointer; its growth rate is distinguished from an asymptotic gap. The late endpoint and OLS slopes give finite-window size exponents 1.0682 +/- 0.0859 and 1.0751 +/- 0.0921, using paired trajectory SEMs and formal weighted-regression exponent errors, without bootstrapping.', '',
        '## Reproduction', '',
        'Run the matching sources/plot_*.py renderer with the compact bundled data. Renderers require NumPy, Matplotlib, Pillow, Poppler, LaTeX and dvipng. All final figure fonts are Computer Modern/AMS through LaTeX; receipts record printed font sizes and bounds.',
        'After an intentional revision, run python sources/build_index.py and python sources/verify_bundle.py --record. Run python sources/verify_bundle.py for a read-only audit. The manifest binds the delivered bundle to SHA-256 checksums; the verifier also checks pre-revision numerical products and the protected outlook.', '']
    (ROOT / 'README.md').write_text('\n'.join(lines))

if __name__ == '__main__': main()
