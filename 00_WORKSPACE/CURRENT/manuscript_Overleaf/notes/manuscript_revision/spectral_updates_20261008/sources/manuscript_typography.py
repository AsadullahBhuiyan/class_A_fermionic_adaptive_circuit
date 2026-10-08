"""Canonical manuscript typography, measured at the LaTeX inclusion width.

All production renderers must configure, prepare before layout, and record before
saving. LaTeX is mandatory: never fall back to mathtext or a system font.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil

import matplotlib as mpl
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT.parents[1] / 'manuscript.tex'
# Measured from the actual RevTeX preamble: 246 pt column, 510 pt text width.
TEX_POINTS_PER_INCH = 72.27
COLUMN_INCHES = 246 / TEX_POINTS_PER_INCH
TEXT_INCHES = 510 / TEX_POINTS_PER_INCH
PANEL = re.compile(r'^\([a-z]\)$')
FONT_PATTERN = re.compile(r'^(?:CM(?:R|MI|SY|EX|BX|B|TI|SL|TT|MIB|BSY)|MSAM|MSBM)\d+$', re.I)


def configure_style(extra=None):
    missing = [p for p in ('latex', 'dvipng', 'pdffonts') if not shutil.which(p)]
    if missing:
        raise RuntimeError('Manuscript typography requires: '+', '.join(missing)+'. Install these tools; font fallback is forbidden.')
    extra = extra or {}
    forbidden = [k for k in extra if k.startswith(('font.', 'mathtext.', 'text.latex')) or k == 'text.usetex']
    if forbidden:
        raise ValueError('Typography must be configured centrally: '+str(forbidden))
    mpl.rcParams.update(extra)
    mpl.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Computer Modern Roman'],
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath,amssymb,bm}',
        'font.size': 8, 'font.weight': 'normal', 'axes.labelsize': 9,
        'axes.labelweight': 'normal', 'axes.titlesize': 9, 'axes.titleweight': 'normal',
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'legend.title_fontsize': 8, 'text.color': 'black',
        'savefig.dpi': 300, 'savefig.bbox': None,
    })


def inclusion_width(stem):
    text = MANUSCRIPT.read_text()
    pattern = r'\\includegraphics\[width=([\d.]*)\\(columnwidth|textwidth)\]\{'+re.escape(stem)+r'\.pdf\}'
    match = re.search(pattern, text)
    if match:
        return float(match[1] or 1) * (COLUMN_INCHES if match[2] == 'columnwidth' else TEXT_INCHES)
    if stem == 'Figure_02_hard_wall':
        return 0.8 * COLUMN_INCHES  # Preserved standalone close-up; historical print width.
    if stem in ('Figure_02_alt_soft_and_hard_walls', 'Figure_02_adaptive_circuit', 'Figure_A02_mutual_information', 'Figure_A04_channel_gap_scan'):
        return COLUMN_INCHES  # Standalone alternative, not included in manuscript.
    raise ValueError('No supported manuscript inclusion width for '+stem)


def text_role(fig, artist, stem):
    text = artist.get_text()
    if PANEL.fullmatch(text):
        return 'panel', 10 if stem == 'Figure_07_entanglement_spectrum' else 9
    for ax in fig.findobj(mpl.axes.Axes):
        if artist is ax.xaxis.label or artist is ax.yaxis.label:
            return 'axis', 9
    # Geometry phase headers are prominent schematic labels (11 pt).
    if stem == 'Figure_01_schematic' and text.startswith(r'$\alpha_{\boldsymbol{r}}='):
        return 'schematic', 11
    if stem in ('Figure_01_schematic', 'Figure_02_adaptive_circuit'):
        if text == r'$\mathrm{m}=s_\sigma$?':
            return 'schematic', 9
        return 'schematic', 8 if text in ('yes', 'no', 'fresh\nancilla', '$s_-=1$ (fill)\n$s_+=0$ (empty)') else 10
    if stem == 'Figure_02_hard_wall':
        return 'schematic', 11 if r'\alpha' in text else 10
    if stem == 'Figure_02_alt_soft_and_hard_walls':
        return 'schematic', 10
    if stem == 'Figure_10_modular_evolution' and re.fullmatch(r'\$\\alpha_1=[13]\$', text):
        return 'parameter', 11
    if stem in ('Figure_04_purification', 'Figure_04_lyapunov') and re.fullmatch(r'\$\\alpha_1=[13]\$', text):
        # User-requested prominent parameter labels and entropy legend.
        return 'parameter', 10
    if stem == 'Figure_05_correlations':
        if re.fullmatch(r'\$\\alpha_1=[13]\$', text):
            return 'parameter', 9
        if text.startswith(r'$\beta='):
            return 'fit_annotation', 8
    if stem == 'Figure_07_entanglement_spectrum' and (
            re.fullmatch(r'\$\\alpha_1=[13]\$', text)
            or text.startswith((r'$W:', r'$R^2='))):
        return 'spectrum_annotation', 9
    return 'annotation_or_tick', 8


def prepare_figure(fig, stem):
    scale = inclusion_width(stem) / fig.get_figwidth()
    # Allocate ticks before applying size rules, including those in child insets.
    for ax in fig.findobj(mpl.axes.Axes):
        if ax.axison:
            ax.get_xticklabels(); ax.get_yticklabels()
    for artist in fig.findobj(Text):
        role, size = text_role(fig, artist, stem)
        artist.set_fontfamily('serif')
        artist.set_fontweight('normal')
        artist.set_usetex(True)
        artist.set_fontsize(size / scale)
        # Literal percentages need escaping when native mathtext is replaced by TeX.
        artist.set_text(re.sub(r'(?<!\\)%', r'\\%', artist.get_text()))
    for legend in fig.findobj(mpl.legend.Legend):
        # The requested larger legends need matching layout spacing as well as text.
        size = 9 if stem in ('Figure_05_correlations', 'Figure_07_entanglement_spectrum') and any(
            re.fullmatch(r'\$\\alpha_1=[13]\$', t.get_text()) for t in legend.get_texts()) else 8
        legend._fontsize = size / scale
        legend.prop.set_size(size / scale)
    fig._manuscript_typography_stem = stem


def record_typography(fig, stem):
    assert getattr(fig, '_manuscript_typography_stem', None) == stem
    fig.canvas.draw()
    scale = inclusion_width(stem) / fig.get_figwidth()
    renderer = fig.canvas.get_renderer()
    texts, clipped = [], []
    hidden = set()
    for ax in fig.findobj(mpl.axes.Axes):
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            if not ax.axison or not axis.get_visible():
                hidden.update(id(t) for t in axis.findobj(Text))
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not min(limits) <= tick.get_loc() <= max(limits):
                    hidden.update(id(t) for t in tick.findobj(Text))
    for artist in fig.findobj(Text):
        if id(artist) in hidden or not artist.get_visible() or not artist.get_text():
            continue
        role, target = text_role(fig, artist, stem)
        box = artist.get_window_extent(renderer)
        row = dict(text=artist.get_text(), role=role, source_pt=artist.get_fontsize(),
                   printed_pt=artist.get_fontsize()*scale, target_pt=target,
                   usetex=artist.get_usetex(), family=artist.get_fontfamily(),
                   weight=artist.get_fontweight())
        assert row['usetex'] and row['family'] == ['serif'] and row['weight'] == 'normal', row
        assert abs(row['printed_pt']-target) < .02, row
        texts.append(row)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1:
            clipped.append(artist.get_text())
    output = ROOT / 'data/typography'
    output.mkdir(exist_ok=True)
    record = dict(stem=stem, family='Computer Modern / AMS', renderer='LaTeX',
                  figure_inches=fig.get_size_inches().tolist(), inclusion_width_inches=inclusion_width(stem),
                  inclusion_scale=scale, text=texts, clipped_text=clipped,
                  style_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output / (stem+'.json')).write_text(json.dumps(record, indent=2)+'\n')
    if clipped:
        raise ValueError(f'{stem}: clipped text: {clipped}')


def verify_typography(stem):
    """Check recorded final-size artists and the actual delivered PDF fonts."""
    import subprocess
    record = json.loads((ROOT/'data/typography'/f'{stem}.json').read_text())
    assert record['style_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), (stem, 'Regenerate after changing the shared style')
    assert abs(record['inclusion_width_inches']-inclusion_width(stem)) < 1e-8, (stem, 'Inclusion width changed')
    assert not record['clipped_text'], stem
    assert record['text'], stem
    for item in record['text']:
        assert item['usetex'] and item['family'] == ['serif'] and item['weight'] == 'normal', (stem, item)
        assert abs(item['printed_pt']-item['target_pt']) < .02, (stem, item)
        assert item['printed_pt'] >= 7.99, (stem, item)
        if item['role'] == 'panel':
            size = 10 if stem == 'Figure_07_entanglement_spectrum' else 9
            assert PANEL.fullmatch(item['text']) and abs(item['printed_pt']-size) < .02
    fonts = subprocess.check_output(['pdffonts', str(ROOT/(stem+'.pdf'))], text=True)
    entries = fonts.splitlines()[2:]
    assert entries, (stem, 'No embedded fonts')
    names = []
    for line in entries:
        name = line.split()[0].split('+')[-1]
        assert FONT_PATTERN.fullmatch(name), (stem, 'Unexpected font', name)
        assert 'Type 3' not in line, (stem, 'Bitmap font')
        # The emb/sub/uni columns are the last three fields before the object ID.
        assert line.split()[-5] == 'yes', (stem, 'Font not embedded', line)
        names.append(name)
    return dict(fonts=sorted(set(names)), minimum_printed_pt=min(t['printed_pt'] for t in record['text']),
                panel_letters=[t['text'] for t in record['text'] if t['role'] == 'panel'],
                inclusion_scale=record['inclusion_scale'], clipped_text=[])
