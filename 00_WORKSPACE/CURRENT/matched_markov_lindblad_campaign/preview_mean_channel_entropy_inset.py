"""Standalone Figure 11 preview; never writes manuscript figures or metadata."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIGURES = REPO / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure'
CONTOUR = HERE / 'analysis_outputs/raster_y_endpoint_entropy_contour_v1'
OUT = HERE / 'analysis_outputs/mean_channel_entropy_inset_preview_v1'
sys.path.insert(0, str(FIGURES / 'sources'))
import plot_mean_channel as original
import manuscript_typography as typography


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Explicit output routing: the standard renderer accepts --output-dir,
    # but its typography recorder otherwise writes under the manuscript tree.
    protected = [p for p in FIGURES.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected.append(FIGURES.parents[1] / 'manuscript.tex')
    before = {str(p): sha(p) for p in protected}
    manifest = json.loads((CONTOUR / 'manifest.json').read_text())
    assert sha(CONTOUR / 'contours.npz') == manifest['output_sha256']['contours.npz']
    with np.load(CONTOUR / 'contours.npz', allow_pickle=False) as saved:
        contour = saved['alpha1_contour']
    assert contour.shape == (20, 64)
    spectral = original.read_json('untwirled_spectra_provenance.json')
    assert manifest['cases']['1']['source_sha256'] == spectral['cases']['1']['source_sha256']
    typography.ROOT = OUT
    (OUT / 'data/typography').mkdir(parents=True, exist_ok=True)
    record = typography.record_typography
    default_text_role = typography.text_role
    inset_label_style = {}

    def preview_text_role(fig, artist, stem):
        if artist is inset_label_style.get('artist'):
            return 'user_requested_inset_label', inset_label_style['printed_pt']
        return default_text_role(fig, artist, stem)

    typography.text_role = preview_text_role

    def add_inset_and_record(fig, stem):
        if stem == original.STEM:
            panel = fig.axes[0]
            inset = panel.inset_axes([.70, .20, .17, .62], zorder=5)
            image = inset.imshow(contour.T, origin='lower', cmap='Blues',
                vmin=0, vmax=float(contour.max()), interpolation='nearest',
                extent=(-.5, 19.5, -.5, 63.5), aspect='auto')
            inset.set(xticks=[0, 19], yticks=[0, 63], xlabel=r'$x$', ylabel=r'$y$')
            inset.xaxis.labelpad = -1
            inset.yaxis.labelpad = -3
            inset.tick_params(direction='in', length=2, pad=1)
            for wall in (4.5, 15.5):
                inset.axvline(wall, color='.55', ls='--', lw=.35)
            label = inset.text(.5, .5, r'$\alpha_1=1$', transform=inset.transAxes,
                               ha='center', va='center', color='black', zorder=8)
            cax = panel.inset_axes([.90, .20, .022, .62], zorder=5)
            cbar = fig.colorbar(image, cax=cax, ticks=[0, .4])
            cbar.ax.set_title(r'$s$', fontsize=8, pad=2)
            cbar.ax.tick_params(direction='in', length=2, pad=1)
            typography.prepare_figure(fig, stem)
            fig.canvas.draw()
            # Fit horizontally inside the pale region between the two walls.
            # The user explicitly requests a fitted inset-label font size.
            width = label.get_window_extent(fig.canvas.get_renderer()).width
            available = .38 * inset.get_window_extent().width
            scale = typography.inclusion_width(stem) / fig.get_figwidth()
            printed_pt = label.get_fontsize() * scale * min(1., available / width)
            inset_label_style.update(artist=label, printed_pt=printed_pt)
            typography.prepare_figure(fig, stem)
        record(fig, stem)

    original.record_typography = add_inset_and_record
    sys.argv = [str(Path(__file__)), '--output-dir', str(OUT)]
    original.main()
    after = {str(p): sha(p) for p in protected}
    assert after == before, 'Protected manuscript artifact changed'
    receipt = dict(status='preview_only', manuscript_unchanged=True,
        protected_sha256=before, contour_manifest_sha256=sha(CONTOUR / 'manifest.json'),
        contour_alpha_1=1, colormap='Blues', units='nats per unit cell',
        inset_label=dict(text='alpha_1=1', position='center',
                         printed_fontsize_pt=inset_label_style['printed_pt']),
        estimator='full-system untwirled endpoint entropy contour; not trajectory-averaged entropy',
        renderer_sha256=sha(Path(__file__)), original_renderer_sha256=sha(Path(original.__file__)),
        outputs={p.name: sha(p) for p in OUT.glob('Figure_11_mean_channel.*')})
    (OUT / 'preview_provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('[preview saved; manuscript unchanged]', OUT)


if __name__ == '__main__':
    main()
