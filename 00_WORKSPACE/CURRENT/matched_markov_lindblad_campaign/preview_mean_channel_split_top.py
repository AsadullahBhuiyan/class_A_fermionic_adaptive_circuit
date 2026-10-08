"""Separate preview: spectrum and contour across the top; manuscript untouched."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIGURES = REPO / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure'
CONTOUR = HERE / 'analysis_outputs/raster_y_endpoint_entropy_contour_v1'
OUT = HERE / 'analysis_outputs/mean_channel_split_top_preview_v1'
sys.path.insert(0, str(FIGURES / 'sources'))
import plot_mean_channel as original
import manuscript_typography as typography


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


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
    record = typography.record_typography
    default_role = typography.text_role
    label_style = {}

    def role(fig, artist, stem):
        if artist is label_style.get('artist'):
            return 'user_requested_fitted_contour_label', label_style['printed_pt']
        return default_role(fig, artist, stem)

    typography.text_role = role

    def split_and_record(fig, stem):
        if stem == original.STEM:
            spectrum, correlator, gap = fig.axes[:3]
            pos = spectrum.get_position()
            spectrum.set_position([pos.x0, pos.y0, .40, pos.height])
            spectrum.legend(loc='upper left', frameon=False, handlelength=.7,
                            handletextpad=.25, borderpad=.2, borderaxespad=.4,
                            labelspacing=.25)
            for text in spectrum.texts:
                if text.get_text() == '(a)':
                    text.set_position((-.365, 1.04))
            for ax, old, new in ((correlator, '(b)', '(c)'), (gap, '(c)', '(d)')):
                for text in ax.texts:
                    if text.get_text() == old:
                        text.set_text(new)
            # Keep the two top plots aligned, without changing the lower rows.
            heat = fig.add_axes([.745, pos.y0, .14, pos.height])
            image = heat.imshow(contour.T, origin='lower', cmap='Blues',
                vmin=0, vmax=float(contour.max()), interpolation='nearest',
                extent=(-.5, 19.5, -.5, 63.5), aspect='auto')
            heat.set(xticks=[0, 19], yticks=[0, 32, 63], xlabel=r'$x$', ylabel=r'$y$')
            heat.yaxis.labelpad = 4
            heat.tick_params(direction='in', top=True, right=True, length=2, pad=1)
            for wall in (4.5, 15.5):
                heat.axvline(wall, color='.55', ls='--', lw=.35)
            heat.text(-.56, 1.04, '(b)', transform=heat.transAxes)
            label = heat.text(.5, .5, r'$\alpha_1=1$', transform=heat.transAxes,
                              ha='center', va='center', color='black')
            cax = fig.add_axes([.91, pos.y0, .018, pos.height])
            cbar = fig.colorbar(image, cax=cax, ticks=[0, .2, .4])
            cbar.ax.set_title(r'$s$', fontsize=8, pad=2)
            cbar.ax.tick_params(direction='in', length=2, pad=1)
            typography.prepare_figure(fig, stem)
            fig.canvas.draw()
            width = label.get_window_extent(fig.canvas.get_renderer()).width
            available = .38 * heat.get_window_extent().width
            scale = typography.inclusion_width(stem) / fig.get_figwidth()
            label_style.update(artist=label,
                printed_pt=label.get_fontsize()*scale*min(1., available/width))
            typography.prepare_figure(fig, stem)
        record(fig, stem)

    original.record_typography = split_and_record
    sys.argv = [str(Path(__file__)), '--output-dir', str(OUT)]
    original.main()
    assert {str(p): sha(p) for p in protected} == before
    # Correct the stock renderer's layout receipt for this preview only.
    validation_path = OUT / 'validation.json'
    validation = json.loads(validation_path.read_text())
    validation.update(layout='two horizontally aligned top panels; two full-width lower panels',
        displayed_panels=['occupation_spectrum', 'endpoint_entropy_contour_alpha1',
                          'mean_correlator', 'gap_size_fit'],
        original_to_main_panel_mapping={'a': 'a', 'b': 'c', 'd': 'd'},
        original_inset_preview_preserved=True)
    validation_path.write_text(json.dumps(validation, indent=2)+'\n')
    receipt = dict(status='preview_only', manuscript_unchanged=True,
        protected_sha256=before, contour_manifest_sha256=sha(CONTOUR/'manifest.json'),
        contour_alpha_1=1, colormap='Blues', units='nats per unit cell',
        center_label_fontsize_pt=label_style['printed_pt'],
        renderer_sha256=sha(Path(__file__)), original_renderer_sha256=sha(Path(original.__file__)),
        outputs={p.name: sha(p) for p in OUT.glob('Figure_11_mean_channel.*')})
    (OUT/'preview_provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('[split-top preview saved; manuscript unchanged]', OUT)


if __name__ == '__main__':
    main()
