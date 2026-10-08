#!/usr/bin/env python3
"""Audit delivered figures, scientific selections, and preserved originals.

Run normally to check the recorded manifest; --record writes the final audit
and checksum manifest after intentional bundle changes. No figures are edited.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

from manuscript_typography import verify_typography

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(name):
    return json.loads((ROOT / name).read_text())


def csv_rows(name):
    with (ROOT / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    rows = read_json('data/figure_index.json')
    stems = {row['stem'] for row in rows}
    assert len(rows) == len(stems) == 15
    # Included figure numbers are distinct from stable asset filename prefixes.
    manuscript = (ROOT.parents[1] / 'manuscript.tex').read_text()
    included = re.findall(r'\\includegraphics\[.*?\]\{(Figure_[^}]+)\.pdf\}', manuscript)
    assert included == [row['stem'] for row in rows if row['included']]
    assert len(included) == 13
    assert [row['number'] for row in rows if row['included']] == [str(i) for i in range(1, 12)] + ['A1', 'A2']
    assert r'fig:HardWall' not in manuscript
    assert r'Fig.~\ref{fig:Geometry}' in manuscript
    assert r'Fig.~\ref{fig:Circuit}(a)' not in manuscript
    figures = {}
    for ext in ('pdf', 'png'):
        assert {p.stem for p in ROOT.glob(f'Figure_*.{ext}')} == stems
    combined = (ROOT / 'FIGURE_NOTES.md').read_text()
    for stem in stems:
        assert f'id="{stem.lower().replace("_", "-")}"' in combined
    for stem in sorted(stems):
        pdf = ROOT / f'{stem}.pdf'
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        assert re.search(r'^Pages:\s+1$', info, re.M), stem
        # Text extraction distinguishes the scientific PDFs from flattened PNGs.
        text = subprocess.check_output(['pdftotext', str(pdf), '-'], text=True)
        assert len(text.strip()) > 10, stem
        with Image.open(ROOT / f'{stem}.png') as im:
            im.load()
            dpi = im.info.get('dpi', ())
            assert len(dpi) == 2 and all(abs(d - 300) < 0.1 for d in dpi), (stem, dpi)
            figures[stem] = {'pixels': list(im.size), 'dpi': list(dpi), 'pdf_pages': 1}

    baseline = ROOT.parents[1] / 'notes/manuscript_revision/baseline/original_figure_checksums.json'
    originals = json.loads(baseline.read_text())
    for name, expected in originals.items():
        assert digest(ROOT.parents[1] / name) == expected, f'Original changed: {name}'
    for row in read_json('data/existing_assets.json'):
        if not row.get('preserved', True):
            continue
        assert digest(ROOT / (row['stem'] + '.pdf')) == row['pdf_source']['sha256']
        if isinstance(row['png_source'], dict):
            assert digest(ROOT / (row['stem'] + '.png')) == row['png_source']['sha256']

    # Typography changes preserve the original diagrams and native A1 arrays.
    schematics = read_json('data/wall_schematics/validation.json')
    for row in schematics['figures'].values():
        assert row['nontext_pixels_changed'] == 0
        for name, expected in row['outputs'].items():
            assert digest(ROOT / name) == expected
    a1 = read_json('data/ow_truncation/typography_validation.json')
    assert a1['scientific_line_arrays_equal_to_native'] and a1['preserved_line_artists'] == 21
    assert not a1['clipped_text']
    for ext, expected in a1['outputs'].items():
        assert digest(ROOT / ('Figure_A01_ow_truncation.' + ext)) == expected['sha256']
    provenance = read_json('data/ow_truncation/provenance.json')
    for name, expected in provenance['bundle_inputs'].items():
        assert digest(ROOT / 'data/ow_truncation' / name) == expected['sha256']
    for name, expected in provenance['original_sources'].items():
        assert digest(ROOT.parents[4] / name) == expected['sha256']
    for stem in stems:
        fonts = subprocess.check_output(['pdffonts', str(ROOT / (stem + '.pdf'))], text=True)
        assert 'Type 3' not in fonts, stem

    typography = {stem: verify_typography(stem) for stem in sorted(stems)}
    for stem in ('Figure_01_schematic', 'Figure_02_adaptive_circuit'):
        assert typography[stem]['panel_letters'] == [], stem
    for name, expected in read_json('data/typography/preserved_inputs.json').items():
        assert digest(ROOT / name) == expected, ('Scientific input changed', name)

    # Cross-check audit receipts against the files actually delivered.
    for path in (ROOT / 'data').glob('*/validation.json'):
        receipt = json.loads(path.read_text())
        for name, expected in receipt.get('outputs', receipt.get('output_sha256', {})).items():
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            assert digest(ROOT / name) == expected, (path, name)

    old = csv_rows('data/correlations/original_plotted_data.csv')
    kept = [row for row in old if row['panel'] in ('a', 'c') or
            (row['panel'] == 'b' and row['series'] in ('$x=5$', '$x=10$', '$x=15$'))]
    assert kept == csv_rows('data/correlations/plotted_data.csv')
    corr = read_json('data/correlations/validation.json')
    assert corr['panel_b_columns'] == [5, 10, 15]
    assert abs(corr['primary_beta_archived'] - corr['primary_beta_independently_recovered']) < 1e-14
    for folder in ('entropy_charge', 'wall_entropy'):
        receipt = read_json(f'data/{folder}/validation.json')
        assert receipt['display_min_Ay'] == 2 and receipt['fits_unchanged']
        # Labels have intentionally changed. Validate scientific inputs and fits,
        # not obsolete pixel equality with the original labels.
        provenance = read_json(f'data/{folder}/input_provenance.json')
        for source in provenance['sources']:
            compact = ROOT / 'data' / folder / Path(source['path']).name
            if compact.is_file() and compact.suffix == '.csv':
                assert digest(compact) == source['sha256'], compact
    assert read_json('data/bulk_topology/validation.json')['layout'] == [3, 1]
    assert read_json('data/entanglement_spectrum/validation.json')['layout'] == [3, 1]
    assert read_json('data/central_charge/validation.json')['layout'] == [2, 1]

    channel = read_json('data/mean_channel/validation.json')
    assert channel['layout'] == [4, 1] and channel['panel_c_scan_points'] == 105
    assert channel['panel_b_display_cutoff'] == 1e-8
    assert channel['panel_b_displayed_separations'] == {'1': [1, 2, 3, 4], '3': [1, 2, 3, 4]}
    channel_source = read_json('data/mean_channel/provenance.json')
    for name, expected in channel_source['compact_inputs_sha256'].items():
        assert digest(ROOT / 'data/mean_channel' / name) == expected
    assert digest(ROOT / 'sources/extract_mean_channel.py') == channel_source['extraction_source_sha256']
    # Preservation of all identified original gap/fit/endpoint inputs and PDFs.
    for name, expected in channel_source['source_files_sha256'].items():
        assert digest(Path(name)) == expected, f'Channel source changed: {name}'
    gap_rows = csv_rows('data/mean_channel/gaps.csv')
    assert len(gap_rows) == 105
    assert all(int(r['Nx']) == 20 and r['status'] == 'resolved_positive' for r in gap_rows)
    np.testing.assert_allclose([float(r['gap']) for r in gap_rows],
                               -2*np.log([float(r['rho']) for r in gap_rows]), rtol=0, atol=1e-14)
    multiplier_rows = csv_rows('data/mean_channel/multiplier_gaps.csv')
    assert len(multiplier_rows) == 105
    assert channel['gap_definition'] == 'g_C=1-rho(A)^2' and channel['gap_units'] == 'dimensionless'
    for row in multiplier_rows:
        original = next(r for r in gap_rows if r['alpha_1'] == row['alpha_1'] and r['Ny'] == row['Ny'])
        assert int(row['Nx']) == 20 and row['status'] == 'resolved_positive'
        np.testing.assert_allclose(float(row['rho_A']), float(original['rho']), rtol=0, atol=1e-14)
        np.testing.assert_allclose(float(row['decay_rate']), float(original['gap']), rtol=0, atol=1e-14)
        np.testing.assert_allclose(float(row['g_C']), 1-float(original['rho'])**2, rtol=0, atol=1e-14)
    fit_rows = csv_rows('data/mean_channel/multiplier_fit_inputs.csv')
    saved_fits = read_json('data/mean_channel/multiplier_fits.json')['results']
    for alpha in (1, 3):
        selected = sorted((r for r in fit_rows if int(r['alpha_1']) == alpha), key=lambda r: int(r['Ny']))
        n = np.array([int(r['Ny']) for r in selected])
        y = np.array([float(r['g_C']) for r in selected])
        np.testing.assert_array_equal(n, [20, 40, 60, 80, 100])
        for row in selected:
            match = next(r for r in multiplier_rows if float(r['alpha_1']) == alpha and r['Ny'] == row['Ny'])
            np.testing.assert_allclose(float(row['g_C']), float(match['g_C']), rtol=0, atol=1e-14)
        fitted = np.polynomial.polynomial.polyfit(1/n, y, 1)
        saved = saved_fits[str(alpha)]
        np.testing.assert_allclose(fitted, [saved['g_infinity'], saved['a']], rtol=0, atol=1e-12)
        r_squared = 1-np.sum((y-np.polynomial.polynomial.polyval(1/n, fitted))**2)/np.sum((y-y.mean())**2)
        np.testing.assert_allclose(r_squared, saved['r_squared'], rtol=0, atol=1e-14)
    with np.load(ROOT / 'data/mean_channel/curves.npz') as curves:
        for row in csv_rows('data/mean_channel/panel_b_plotted_data.csv'):
            value = curves[f"alpha{row['alpha_1']}_xavg"][int(row['r_y'])]
            assert value == float(row['squared_mean_correlator'])
            assert np.log(value) == float(row['log_correlator'])
    proof = ROOT.parents[1] / 'notes/channel_gap_proof/channel_gap_proof.pdf'
    assert re.search(r'^Pages:\s+2$', subprocess.check_output(['pdfinfo', str(proof)], text=True), re.M)

    occupation_source = read_json('data/entanglement_spectrum/occupation_provenance.json')
    assert digest(ROOT / 'data/entanglement_spectrum/occupation_inputs.npz') == occupation_source['compact_input_sha256']
    assert digest(ROOT / 'sources/extract_occupation_comparison.py') == occupation_source['extraction_source_sha256']
    for alpha in (1, 3):
        records = [r for r in occupation_source['source_files'] if r['alpha_1'] == alpha]
        assert len(records) == 4
        assert sorted(i for r in records for i in r['completion']['case_sample_indices']) == list(range(100))
    pooled_source = read_json('data/entanglement_spectrum/pooled_half_strip_provenance.json')
    assert digest(ROOT / 'data/entanglement_spectrum/pooled_half_strip_spectra.npz') == pooled_source['compact_input_sha256']
    assert digest(ROOT / 'sources/extract_pooled_half_strip_spectra.py') == pooled_source['extraction_source_sha256']
    with np.load(ROOT / 'data/entanglement_spectrum/pooled_half_strip_spectra.npz') as data:
        np.testing.assert_array_equal(data['alpha_1'], [1, 3])
        np.testing.assert_array_equal(data['sample_ids'], np.arange(100))
        np.testing.assert_array_equal(data['origins'], np.arange(32))
        assert data['Nx'] == 20 and data['Ny'] == 32 and data['Ay'] == 16 and data['cycle'] == 64
        spectra = data['centered_occupations']
        assert spectra.shape == (2, 100, 32, 640) and np.isfinite(spectra).all()
        assert np.abs(spectra).max() < 1+1e-8
        occupation_edges = np.linspace(-1, 1, 101)
        occupation_counts = np.array([np.histogram(np.clip(v.ravel(), -1, 1), occupation_edges)[0] for v in spectra])
        np.testing.assert_array_equal(occupation_counts.sum(axis=1), [2048000, 2048000])
        occupation_density = occupation_counts / (2048000 * np.diff(occupation_edges))
        np.testing.assert_allclose((occupation_density * np.diff(occupation_edges)).sum(axis=1), 1, atol=1e-14)
        saved_occupation = np.genfromtxt(ROOT / 'data/entanglement_spectrum/occupation_histogram.csv', delimiter=',', names=True)
        for index, alpha in enumerate((1, 3)):
            np.testing.assert_array_equal(occupation_counts[index], saved_occupation[f'alpha1_{alpha}_count'])
            np.testing.assert_array_equal(occupation_density[index], saved_occupation[f'alpha1_{alpha}_density'])

    energy_edges = np.linspace(-np.log(199), np.log(199), 102)
    selected_energies = [np.log1p(-v[np.abs(v) <= .99]) - np.log1p(v[np.abs(v) <= .99]) for v in spectra]
    energy_counts = np.array([np.histogram(v, energy_edges)[0] for v in selected_energies])
    retained_totals = energy_counts.sum(axis=1)
    np.testing.assert_array_equal(retained_totals, [95398, 83326])
    energy_density = energy_counts / (retained_totals[:, None] * np.diff(energy_edges))
    np.testing.assert_allclose((energy_density * np.diff(energy_edges)).sum(axis=1), 1, atol=1e-14)
    saved_energy = np.genfromtxt(ROOT / 'data/entanglement_spectrum/energy_histogram_comparison.csv', delimiter=',', names=True)
    for index, alpha in enumerate((1, 3)):
        np.testing.assert_array_equal(energy_counts[index], saved_energy[f'alpha1_{alpha}_count'])
        np.testing.assert_array_equal(energy_density[index], saved_energy[f'alpha1_{alpha}_density'])
    window_counts = (np.abs(spectra) <= .99).sum(axis=-1)
    np.testing.assert_array_equal(window_counts[:, :, :16], window_counts[:, :, 16:])
    with np.load(ROOT / 'data/entanglement_spectrum/half_strip_window_counts.npz') as data:
        np.testing.assert_array_equal(window_counts, data['counts'])

    with np.load(ROOT / 'data/entanglement_spectrum/inputs.npz') as data:
        lam, energy = data['retained_centered_occupations'], data['retained_energies']
        assert len(lam) == 95398 and np.all(np.abs(lam) <= 0.99)
        np.testing.assert_allclose(energy, np.log1p(-lam) - np.log1p(lam), rtol=0, atol=1e-14)
        edges = np.linspace(-np.log(199), np.log(199), 102)
        counts, _ = np.histogram(energy, edges)
        saved = np.genfromtxt(ROOT / 'data/entanglement_spectrum/energy_histogram.csv', delimiter=',', names=True)
        np.testing.assert_array_equal(counts, saved['raw_count'])
        assert counts.sum() == 95398
        half_counts = data['counts_by_sample_width_origin'][:, -1, :]
        np.testing.assert_array_equal(half_counts, window_counts[0])
        np.testing.assert_array_equal(energy, selected_energies[0])
        assert half_counts.shape == (100, 32) and half_counts.sum() == 95398
        mean = half_counts.mean(axis=1).mean()
        assert abs(mean - 29.811875) < 1e-12
        sem = half_counts.mean(axis=1).std(ddof=1) / 10

    # The regenerated purification/modular/A2 figures are scientific-data
    # preserving renderers and must never be restored from obsolete PDFs.
    reusable = {row['stem'] for row in read_json('data/existing_assets.json')
                if row.get('preserved', True)}
    assert reusable == set(), 'Typography-normalized figures must use their current renderers'
    assert all((ROOT / 'sources' / row['renderer']).is_file() for row in rows)
    assert [row['number'] for row in rows if row['section'] == 'II'] == ['1', '2']
    assert [row['number'] for row in rows if row['section'] == 'III'] == [str(i) for i in range(3,12)]
    assert [row['number'] for row in rows if row['section'] == 'Appendix'] == ['A1', 'A2']

    purification = read_json('data/purification/notation_validation.json')
    assert purification['status'] == 'passed' and not purification['data_changed']
    assert purification['gap_points'] == 7
    expected_gap_fit = read_json('data/purification/analysis_manifest.json')['gap_fit']
    assert purification['fit'] == expected_gap_fit
    np.testing.assert_allclose(purification['density_sum'], 1, atol=1e-14, rtol=0)
    modular = read_json('data/modular_evolution/stacked_figure_metadata.json')
    assert not modular['data_changed'] and modular['snapshot_times'] == [0., .1, .2]
    assert modular['epsilon'] == 1e-10
    assert digest(ROOT / 'data/modular_evolution/averaged_observables.npz') == modular['data_sha256']
    assert digest(ROOT / 'sources/plot_modular_evolution.py') == modular['plot_script_sha256']

    a2 = read_json('data/mutual_information/composition_validation.json')
    assert a2['status'] == 'passed' and not a2['mi_data_changed']
    assert a2['contour_origins'] == 32 and a2['contour_trajectories_per_alpha'] == 100
    assert a2['contour_endpoint'] == 64 and a2['contour_shape'] == [16,20]
    assert a2['mi_points'] == 63 and a2['gamma'] == .5 and a2['clipped_text'] == []
    contour_source = read_json('data/mutual_information/analysis_manifest_y0avg.json')
    assert contour_source['origin_average'] and contour_source['origin_count'] == 32
    assert digest(ROOT / 'data/mutual_information/contours_y0avg.npz') == contour_source['data_sha256']
    with np.load(ROOT / 'data/mutual_information/contours_y0avg.npz', allow_pickle=False) as contours:
        np.testing.assert_array_equal(contours['y0_values'], np.arange(32))
        np.testing.assert_array_equal(contours['sample_ids'], np.arange(100))
        peaks = []
        for alpha in (1,3):
            samples = contours[f'alpha1_{alpha}_per_sample']
            assert samples.shape == (100,16,20)
            np.testing.assert_allclose(samples.mean(0), contours[f'alpha1_{alpha}_mean'], atol=1e-15, rtol=1e-14)
            np.testing.assert_allclose(samples.sum((1,2)), contours[f'alpha1_{alpha}_entropy'], atol=1e-12)
            np.testing.assert_allclose(samples.mean(0).sum(), a2['contour_sum'][str(alpha)], atol=1e-12)
            peaks.append(float(samples.mean(0).max()))
        np.testing.assert_allclose(max(peaks), a2['vmax'], atol=1e-14, rtol=0)
    mi = csv_rows('data/mutual_information/hard_wall_nshell1_sample_summary.csv')
    assert len(mi) == 63 and all(int(r['samples']) == 100 for r in mi)
    assert {int(r['Ny']) for r in mi} == {20,24,28}
    assert all(int(r['cycles']) == 2*int(r['Ny']) and int(r['width']) == int(r['Ny'])//4 for r in mi)

    # All relative links in the delivered scientific notes/index must resolve.
    for path in (ROOT / 'FIGURE_NOTES.md', ROOT / 'README.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if not re.match(r'https?://', target):
                if args.record and path.name == 'README.md' and target == 'validation.json':
                    continue  # This audit writes that linked file below.
                filename, _, fragment = target.partition('#')
                dest = path.parent / filename if filename else path
                assert dest.exists(), (path.name, target)
                if fragment and dest.name == 'FIGURE_NOTES.md':
                    assert f'id="{fragment}"' in dest.read_text(), target

    audit = {
        'all_checks_passed': True,
        'figure_versions': 15,
        'manuscript_figures': 13,
        'numbering_convention': 'figure_N audit keys use stable asset IDs; index numbers match manuscript',
        'manuscript_figure_mapping': {row['stem']: row['number'] for row in rows if row['included']},
        'typography': typography,
        'combined_figure_note': 'FIGURE_NOTES.md',
        'figure_3_layout': [3, 1],
        'figure_7_layout': [3, 1],
        'figure_8_layout': [2, 1],
        'figure_11_layout': [4, 1],
        'figure_A2_layout': [2, 1],
        'figure_A2_contour_origins': 32,
        'figure_A2_contour_trajectories_per_alpha': 100,
        'figure_A2_mutual_information_points': len(mi),
        'restorable_figures': sorted(reusable),
        'figure_11_gap_scan_points': len(gap_rows),
        'figure_11_gap_definition': channel['gap_definition'],
        'figure_11_fits': channel['panel_d_saved_fits_recovered'],
        'figure_11_original_sources_unchanged': len(channel_source['source_files_sha256']),
        'figure_11_proof': str(proof),
        'figure_11_proof_sha256': digest(proof),
        'figure_11_proof_tex_sha256': digest(proof.with_suffix('.tex')),
        'original_figure_assets_and_bundle_files_unchanged': len(originals),
        'preservation_baseline': str(baseline),
        'authorized_manuscript_and_bibliography_edits': True,
        'existing_PDFs_exact_source_copies': sum(row.get('preserved', True) for row in read_json('data/existing_assets.json')),
        'correlation_rows_exact_source_subset': len(kept),
        'correlation_panel_B_x': [5, 10, 15],
        'correlation_fit_unchanged': True,
        'entropy_charge_and_wall_entropy_min_display_Ay': 2,
        'retained_spectrum_observations': int(counts.sum()),
        'occupation_comparison': {'Ny': 32, 'Ay': 16, 'alpha_1': [1, 3],
                                  'counts_per_ensemble': occupation_counts.sum(axis=1).tolist(),
                                  'normalization': 'unit-area density; full range [-1,1]', 'cut_origins': list(range(32))},
        'energy_comparison_retained_observations': retained_totals.tolist(),
        'histogram_bins': len(counts),
        'histogram_normalization': 'occupation: unit area over [-1,1]; energy: unit area within window',
        'count_panel': 'alpha_1=1; raw count, origin average within trajectory, then ensemble mean and trajectory SEM',
        'half_strip_mean_count': float(mean),
        'half_strip_trajectory_SEM': float(sem),
        'figures': figures,
        'visual_review': read_json('data/visual_review.json'),
        'scope': 'Existing results only. Original files checked against pre-work hashes; no circuit simulation was run.'
    }
    if args.record:
        (ROOT / 'validation.json').write_text(json.dumps(audit, indent=2) + '\n')
        files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
                 for p in sorted(ROOT.rglob('*')) if p.is_file()
                 and p.name != 'manifest.json' and '__pycache__' not in p.parts
                 and p.suffix != '.pyc'}
        # Nested source manifests are included too; only this receipt excludes itself.
        for p in sorted((ROOT / 'data').rglob('manifest.json')):
            files[str(p.relative_to(ROOT))] = {'bytes': p.stat().st_size, 'sha256': digest(p)}
        (ROOT / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
    else:
        for name, entry in read_json('manifest.json')['files'].items():
            assert digest(ROOT / name) == entry['sha256'], f'Bundle changed: {name}'
    print(json.dumps({k: v for k, v in audit.items() if k not in ('figures', 'visual_review')}, indent=2))


if __name__ == '__main__':
    main()
