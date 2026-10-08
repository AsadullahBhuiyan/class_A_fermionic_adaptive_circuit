"""Compact single-column four-row preview, isolated from the manuscript."""
import copy
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from preview_mean_channel_two_figures import FIGURES, CONTOUR, sha, panel_label
import plot_mean_channel as original
import manuscript_typography as typography
import matplotlib.pyplot as plt

OUT = HERE / 'analysis_outputs/mean_channel_four_rows_preview_v1'
STEM = 'Mean_channel_four_rows'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [p for p in FIGURES.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected.append(FIGURES.parents[1] / 'manuscript.tex')
    before = {str(p): sha(p) for p in protected}
    manifest = json.loads((CONTOUR / 'manifest.json').read_text())
    assert sha(CONTOUR/'contours.npz') == manifest['output_sha256']['contours.npz']
    with np.load(CONTOUR/'contours.npz', allow_pickle=False) as saved:
        contour = saved['alpha1_contour']
    assert manifest['cases']['1']['source_sha256'] == original.read_json(
        'untwirled_spectra_provenance.json')['cases']['1']['source_sha256']
    typography.ROOT = OUT
    (OUT/'data/typography').mkdir(parents=True, exist_ok=True)
    old_width = typography.inclusion_width
    typography.inclusion_width = lambda stem: typography.COLUMN_INCHES if stem == STEM else old_width(stem)
    record = typography.record_typography
    checks = {}

    def render_and_record(fig, stem):
        if stem == original.STEM:
            preview = copy.deepcopy(fig)
            preview.set_size_inches(3.375, 6.2)
            spectrum, correlator, gap = preview.axes[:3]
            height = 1.02/6.2
            bottom = [v/6.2 for v in (4.97, 3.48, 1.99, .50)]
            for ax, row, letter in ((spectrum, 0, 'a'), (correlator, 2, 'c'), (gap, 3, 'd')):
                ax.set_position([.20, bottom[row], .77, height])
                panel_label(ax, letter)
            heat = preview.add_axes([.20, bottom[1], .66, height])
            image = heat.imshow(contour.T, origin='lower', cmap='Blues',
                vmin=0, vmax=float(contour.max()), interpolation='nearest',
                extent=(-.5, 19.5, -.5, 63.5), aspect='auto')
            heat.set(xlabel=r'$x$', ylabel=r'$y$', xticks=[0, 5, 10, 15, 19], yticks=[0, 32, 63])
            heat.tick_params(direction='in', top=True, right=True)
            for wall in (4.5, 15.5):
                heat.axvline(wall, color='.55', ls='--', lw=.4)
            heat.text(.5, .5, r'$\alpha_1=1$', transform=heat.transAxes, ha='center', va='center')
            panel_label(heat, 'b')
            cax = preview.add_axes([.90, bottom[1], .023, height])
            cbar = preview.colorbar(image, cax=cax, ticks=[0, .2, .4])
            cbar.ax.set_title(r'$s(x,y)$', pad=3)
            cbar.ax.tick_params(direction='in', length=2, pad=1)
            typography.prepare_figure(preview, STEM)
            record(preview, STEM)
            renderer = preview.canvas.get_renderer()
            rows = [spectrum, heat, correlator, gap]
            boxes = [ax.get_tightbbox(renderer) for ax in rows]
            assert all(boxes[i].y0 > boxes[i+1].y1 for i in range(3)), 'Panel decorations overlap'
            for ext in ('pdf', 'png'):
                preview.savefig(OUT/f'{STEM}.{ext}', dpi=300)
            checks.update(typography.verify_typography(STEM))
            plt.close(preview)
        record(fig, stem)

    original.record_typography = render_and_record
    sys.argv = [str(Path(__file__)), '--output-dir', str(OUT)]
    original.main()
    assert {str(p): sha(p) for p in protected} == before
    receipt = dict(status='preview_only', manuscript_unchanged=True, layout=[4, 1],
        figure_inches=[3.375, 6.2], printed_height_inches=6.2*typography.COLUMN_INCHES/3.375,
        protected_sha256=before, renderer_sha256=sha(Path(__file__)),
        helper_sha256=sha(HERE/'preview_mean_channel_two_figures.py'),
        original_renderer_sha256=sha(Path(original.__file__)),
        contour_manifest_sha256=sha(CONTOUR/'manifest.json'), typography_checks=checks,
        outputs={f'{STEM}.{ext}': sha(OUT/f'{STEM}.{ext}') for ext in ('pdf', 'png')})
    (OUT/'preview_provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('[four-row preview saved; manuscript unchanged]', OUT)


if __name__ == '__main__':
    main()
