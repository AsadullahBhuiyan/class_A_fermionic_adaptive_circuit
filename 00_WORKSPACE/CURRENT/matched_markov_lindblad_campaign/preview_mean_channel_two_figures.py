"""Preview two separate vertical figure pairs, preserving manuscript artifacts."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIGURES = REPO / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure'
CONTOUR = HERE / 'analysis_outputs/raster_y_endpoint_entropy_contour_v1'
OUT = HERE / 'analysis_outputs/mean_channel_two_figures_preview_v1'
STEMS = ('Spectrum_and_contour_AB', 'Correlator_and_gap_CD')
sys.path.insert(0, str(FIGURES / 'sources'))
import plot_mean_channel as original
import manuscript_typography as typography
import matplotlib.pyplot as plt


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def panel_label(ax, letter):
    position = ((.054-ax.get_position().x0)/ax.get_position().width, 1.04)
    for text in list(ax.texts):
        if typography.PANEL.fullmatch(text.get_text()):
            text.set_text(f'({letter})')
            text.set_position(position)
            return
    # Align panel labels in figure coordinates even with a colorbar beside B.
    ax.text(*position, f'({letter})', transform=ax.transAxes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in FIGURES.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected.append(FIGURES.parents[1] / 'manuscript.tex')
    before = {str(p): sha(p) for p in protected}
    manifest = json.loads((CONTOUR / 'manifest.json').read_text())
    assert sha(CONTOUR / 'contours.npz') == manifest['output_sha256']['contours.npz']
    with np.load(CONTOUR / 'contours.npz', allow_pickle=False) as saved:
        contour = saved['alpha1_contour']
    spectral = original.read_json('untwirled_spectra_provenance.json')
    assert manifest['cases']['1']['source_sha256'] == spectral['cases']['1']['source_sha256']
    typography.ROOT = OUT
    (OUT / 'data/typography').mkdir(parents=True, exist_ok=True)
    width = typography.inclusion_width
    typography.inclusion_width = lambda stem: typography.COLUMN_INCHES if stem in STEMS else width(stem)
    record = typography.record_typography
    font_checks = {}

    def save_pair(fig, stem):
        typography.prepare_figure(fig, stem)
        record(fig, stem)
        for ext in ('pdf', 'png'):
            fig.savefig(OUT / f'{stem}.{ext}', dpi=300)
        font_checks[stem] = typography.verify_typography(stem)
        plt.close(fig)

    def create_pairs_and_record(fig, stem):
        if stem == original.STEM:
            ab = copy.deepcopy(fig)
            ab.set_size_inches(3.375, 4.0)
            spectrum = ab.axes[0]
            for ax in list(ab.axes[1:]):
                ab.delaxes(ax)
            spectrum.set_position([.20, .60, .77, .34])
            panel_label(spectrum, 'a')
            heat = ab.add_axes([.20, .14, .66, .34])
            image = heat.imshow(contour.T, origin='lower', cmap='Blues',
                vmin=0, vmax=float(contour.max()), interpolation='nearest',
                extent=(-.5, 19.5, -.5, 63.5), aspect='auto')
            heat.set(xlabel=r'$x$', ylabel=r'$y$', xticks=[0, 5, 10, 15, 19],
                     yticks=[0, 16, 32, 48, 63])
            heat.tick_params(direction='in', top=True, right=True)
            for wall in (4.5, 15.5):
                heat.axvline(wall, color='.55', ls='--', lw=.4)
            heat.text(.5, .5, r'$\alpha_1=1$', transform=heat.transAxes,
                      ha='center', va='center')
            panel_label(heat, 'b')
            cax = ab.add_axes([.90, .14, .023, .34])
            cbar = ab.colorbar(image, cax=cax, ticks=[0, .2, .4])
            cbar.ax.set_title(r'$s(x,y)$', pad=3)
            cbar.ax.tick_params(direction='in', length=2, pad=1)
            save_pair(ab, STEMS[0])

            cd = copy.deepcopy(fig)
            cd.set_size_inches(3.375, 4.0)
            old_spectrum, correlator, gap = cd.axes[:3]
            cd.delaxes(old_spectrum)
            correlator.set_position([.20, .60, .77, .34])
            gap.set_position([.20, .14, .77, .34])
            panel_label(correlator, 'c')
            panel_label(gap, 'd')
            save_pair(cd, STEMS[1])
        record(fig, stem)

    original.record_typography = create_pairs_and_record
    sys.argv = [str(Path(__file__)), '--output-dir', str(OUT)]
    original.main()
    assert {str(p): sha(p) for p in protected} == before
    receipt = dict(status='preview_only', manuscript_unchanged=True,
        layout='two separate vertically stacked figures: A/B and C/D',
        figure_inches=[3.375, 4.0], protected_sha256=before,
        contour_manifest_sha256=sha(CONTOUR / 'manifest.json'),
        contour_alpha_1=1, colormap='Blues', units='nats per unit cell',
        contour_aspect='auto; full 20 by 64 grid, no spatial resampling',
        renderer_sha256=sha(Path(__file__)), original_renderer_sha256=sha(Path(original.__file__)),
        typography_checks=font_checks,
        outputs={f'{stem}.{ext}': sha(OUT / f'{stem}.{ext}') for stem in STEMS for ext in ('pdf', 'png')})
    (OUT / 'two_figures_provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('[two-figure preview saved; manuscript unchanged]', OUT)


if __name__ == '__main__':
    main()
