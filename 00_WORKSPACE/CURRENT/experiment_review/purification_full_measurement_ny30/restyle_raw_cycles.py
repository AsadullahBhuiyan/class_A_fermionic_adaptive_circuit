"""Display-only revision of the fixed-T40 figure; preserve previous assets."""
import csv
import json
from pathlib import Path

import numpy as np
import make_figure as figure

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'fixed_T40_gap_v1'
OUT = HERE / 'fixed_T40_gap_raw_cycles_v2'
STEM = 'purification_ny30_gap_T40_raw_cycles_4x1'


def main():
    manifest = json.loads((SOURCE/'analysis_manifest.json').read_text())
    for row in manifest['outputs']:
        assert figure.sha(Path(row['path'])) == row['sha256'], row['path']
    with (SOURCE/'gap_summary_T40.csv').open() as stream:
        gaps = list(csv.DictReader(stream))
    a1 = figure.load_data(figure.DATA1, 1)
    a3 = figure.load_data(figure.DATA3, 3)
    density, _ = figure.mean_sem(a1['densities'])
    with np.load(HERE/'Ny030_slowest_mode_density.npz') as data:
        np.testing.assert_array_equal(density, data['mean'])
    OUT.mkdir(exist_ok=True)
    entropy, spatial = figure.plot(a1['entropy'], a3['entropy'], a1['spatial'],
        density, gaps, manifest['fit'], gap_time_label=r'$T=40$',
        output_dir=OUT, stem=STEM, raw_cycles=True, gap_loglog=True,
        heatmap_time_annotation=False)
    for name, records in [('total_entropy_curves.csv',entropy), ('spatial_entropy_curves.csv',spatial)]:
        with (HERE/name).open() as stream:
            prior = list(csv.DictReader(stream))
        assert len(prior) == len(records)
        for old, new in zip(prior, records):
            for key, value in new.items():
                np.testing.assert_equal(float(old[key]), value)
    caption = (SOURCE/'caption.tex').read_text().replace(
        'purification_ny30_gap_T40_4x1.pdf', STEM+'.pdf').replace(
        'Panels (a,b) use logarithmic axes; cycle zero is omitted from display.',
        'Panels (a,b) show raw cycle number $t$ on logarithmic axes; cycle zero is omitted from display.').replace(
        '(c) The previously obtained', '(c) On log--log axes, the previously obtained')
    (OUT/'caption.tex').write_text(caption)
    figure.write_json(OUT/'analysis_manifest.json', dict(
        schema='purification_fixed_T40_raw_cycles_v2',
        display_only=True, panel_ab_cycles='1..60, raw t, log-log axes',
        panel_c='unchanged fixed T40 slab-only data and fit, log-log axes',
        panel_d='unchanged full-measurement T60 heatmap; time stated only in caption',
        gap_fit=manifest['fit'], previous_manifest_sha256=figure.sha(SOURCE/'analysis_manifest.json'),
        inputs=a1['records']+a3['records']+[
            dict(path=str(SOURCE/'gap_summary_T40.csv'),sha256=figure.sha(SOURCE/'gap_summary_T40.csv'))],
        source_hashes={str(p):figure.sha(p) for p in [Path(__file__), HERE/'make_figure.py']},
        outputs=[dict(path=str(p),sha256=figure.sha(p)) for p in sorted(OUT.iterdir())
                 if p.name != 'analysis_manifest.json']))
    print(OUT/STEM)


if __name__ == '__main__':
    main()
