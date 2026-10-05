#!/usr/bin/env python3
"""Copy verified, compact Figure 11 inputs from completed campaigns; no dynamics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mean_channel'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=ROOT.parents[4])
    args = parser.parse_args()
    campaign = args.repository / '00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign'
    scan = campaign / 'fixed_width_alpha_spectral_v1/results/20261001T220546Z'
    fits = scan / 'limiting_gap_fits/20261001T224848Z'
    combined = fits / 'combined_20261001T231741Z'
    multiplier_scan = scan / 'multiplier_gap_alpha/20261002T021935Z'
    multiplier_fits = scan / 'multiplier_gap_fits/20261002T020317Z'
    spectra = campaign / 'analysis_outputs/raster_y_channel_alpha1_vs3_ky'
    correlators = campaign / 'analysis_outputs/raster_y_mean_channel_squared_correlator_v1'
    DATA.mkdir(exist_ok=True)
    sources, copied = {}, []

    def verify(path, expected):
        actual = sha(path)
        assert actual == expected, f'Source checksum mismatch: {path}'
        sources[str(path)] = actual

    def copy(path, name, expected):
        verify(path, expected)
        shutil.copy2(path, DATA / name)
        copied.append(name)

    for folder, key, files in (
        (spectra, 'sha256', {'spectra.npz': 'spectra.npz', 'summary.json': 'spectrum_summary.json'}),
        (correlators, 'files_sha256', {'curves.npz': 'curves.npz', 'summary.json': 'correlator_summary.json'}),
        (scan / 'analysis', 'output_sha256', {'gaps.csv': 'gaps.csv', 'caption.txt': 'gap_source_caption.txt'}),
        (fits, 'output_sha256', {'fits.json': 'fits.json', 'input_diagnostics.csv': 'fit_input_diagnostics.csv'}),
        (combined, 'output_sha256', {'fit_annotations.json': 'fit_annotations.json', 'caption.txt': 'fit_source_caption.txt'}),
        (multiplier_scan, 'output_sha256', {'data.csv': 'multiplier_gaps.csv', 'caption.txt': 'multiplier_gap_source_caption.txt'}),
        (multiplier_fits, 'output_sha256', {'data.csv': 'multiplier_fit_inputs.csv', 'fits.json': 'multiplier_fits.json', 'caption.txt': 'multiplier_fit_source_caption.txt'}),
    ):
        manifest = json.loads((folder / 'manifest.json').read_text())
        for source, dest in files.items():
            copy(folder / source, dest, manifest[key][source])
        sources[str(folder / 'manifest.json')] = sha(folder / 'manifest.json')

    # Audit the full saved spectral input, including both hard-wall blocks.
    manifest = json.loads((scan / 'analysis/manifest.json').read_text())
    for filename, expected in manifest['input_sha256'].items():
        verify(Path(filename), expected)
    with (DATA / 'gaps.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 105
    receipts = []
    for row in rows:
        ny, alpha = int(row['Ny']), float(row['alpha_1'])
        index = round(10 * (alpha - 1))
        case = scan / f'Ny{ny:03d}_a{index:02d}_{alpha:.1f}'
        receipt = json.loads((case / 'completion.json').read_text())
        cfg, diagnostics = receipt['config'], receipt['diagnostics']
        assert cfg['Nx'] == 20 and cfg['Ny'] == ny and cfg['alpha_1'] == alpha
        assert cfg['walls'] == [5, 15] and cfg['all_slabs_active']
        assert cfg['perfect_correction'] and cfg['dephasing']
        assert cfg['sequence'] == 'raster_y' and cfg['channel_order'] == ['Ap', 'Am', 'Bp', 'Bm']
        assert row['status'] == diagnostics['gap_status'] == 'resolved_positive'
        result = case / receipt['result_filename']
        verify(result, receipt['result_sha256'])
        assert result.stat().st_size == receipt['result_bytes']
        with np.load(result, allow_pickle=False) as data:
            radius = float(np.max(np.abs(data['eigenvalues'])))
            assert len(data['eigenvalues']) == 2 * 20 * ny
            np.testing.assert_allclose(radius, float(row['rho']), rtol=0, atol=1e-14)
            np.testing.assert_allclose(-2 * np.log(radius), float(row['gap']), rtol=0, atol=1e-13)
        receipts.append({'path': str(case), **receipt})
    (DATA / 'gap_case_receipts.json').write_text(json.dumps(receipts, indent=2) + '\n')
    copied.append('gap_case_receipts.json')

    # The attachments are identified by exact decoded pixels, not filenames.
    matches = []
    for attachment, source in (
        ('/home/abhuiyan/.codex/attachments/68ef6608-33b7-4eb8-b21a-9da7404e0919/codex-clipboard-f19565b7-35a4-4a32-b09a-15970f01e16e.png', scan / 'analysis/channel_gap_vs_alpha.png'),
        ('/home/abhuiyan/.codex/attachments/05ffa477-3059-4ad4-9166-f8acd01b81ba/codex-clipboard-13d5aff6-adcd-4075-82a2-c2ccf17f8ec1.png', combined / 'combined_limiting_gap_fits.png'),
        ('/home/abhuiyan/.codex/attachments/32a41f94-538c-4bc6-bbfc-c16775968a6c/codex-clipboard-5b2af9da-704d-4aec-a6d7-cc6467a77aae.png', multiplier_scan / 'multiplier_gap_vs_alpha.png'),
        ('/home/abhuiyan/.codex/attachments/080e6b3e-6314-4d64-941c-a56dbd4675fd/codex-clipboard-e65bff07-1435-45d5-a018-b4fd5662dd3a.png', multiplier_fits / 'multiplier_gap_fits.png'),
    ):
        with Image.open(attachment) as a, Image.open(source) as b:
            assert a.size == b.size and ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox() is None
        source_manifest = json.loads((source.parent / 'manifest.json').read_text())['output_sha256']
        for path in (source, source.with_suffix('.pdf')):
            verify(path, source_manifest[path.name])
        matches.append({'attachment': attachment, 'source_png': str(source),
                        'source_pdf': str(source.with_suffix('.pdf')), 'exact_pixel_match': True})

    original = campaign / 'analysis_outputs/raster_y_mean_channel_summary_2x1/hard_wall_mean_channel_summary_2x1.pdf'
    sources[str(original)] = sha(original)
    provenance = {
        'scope': 'Read/copy existing results only; no simulations or eigensolver reruns.',
        'displayed_gap_definition': 'g_C=1-rho(A)^2; dimensionless multiplier gap',
        'historical_gap_definition': 'Delta_C=-2*log(rho(A)); logarithmic rate inputs retained separately',
        'source_files_sha256': sources,
        'screenshot_matches': matches,
        'original_panels_ab_pdf': str(original),
        'gap_case_count': len(receipts),
        'maximum_independent_dominant_residual': max(r['diagnostics']['independent_dominant_residual'] for r in receipts),
        'maximum_independent_product_action_error': max(r['diagnostics']['independent_product_action_error'] for r in receipts),
        'compact_inputs_sha256': {name: sha(DATA / name) for name in sorted(copied)},
        'extraction_source_sha256': sha(Path(__file__)),
    }
    provenance['source_producers'] = [str(campaign / 'plot_mean_channel_summary.py')] + [
        str(campaign / 'fixed_width_alpha_spectral_v1' / f)
        for f in ('run_scan.py', 'fit_limiting_gap.py', 'plot_combined_limiting_gaps.py',
                  'plot_multiplier_gap_vs_alpha.py', 'plot_multiplier_gap.py')]
    (DATA / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: provenance[k] for k in ('gap_case_count', 'maximum_independent_dominant_residual', 'maximum_independent_product_action_error')}, indent=2))


if __name__ == '__main__':
    main()
