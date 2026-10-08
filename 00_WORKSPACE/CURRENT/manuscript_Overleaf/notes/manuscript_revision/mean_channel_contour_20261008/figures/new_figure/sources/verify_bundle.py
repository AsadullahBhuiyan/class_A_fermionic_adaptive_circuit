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
import sys

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
    assert len(rows) == len(stems) == 18
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
    assert a1['scientific_line_arrays_equal_to_native'] and a1['preserved_line_artists'] == 4 and a1['native_axis_indices'] == [2,3]
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
    assert typography['Figure_01_schematic']['panel_letters'] == ['(a)', '(b)']
    assert typography['Figure_02_adaptive_circuit']['panel_letters'] == []
    for name, expected in read_json('data/typography/preserved_inputs.json').items():
        if name in ('data/correlations/plotted_data.csv', 'data/mean_channel/panel_b_plotted_data.csv'):
            # Derived display flags changed by request; every numerical field is
            # checked below against the preserved original, not a new baseline.
            continue
        if name == 'data/entanglement_spectrum/occupation_histogram.csv':
            # Authorized coordinate conversion: preserve the old receipt and
            # verify the exact transformation instead of resetting its baseline.
            old = ROOT.parents[1] / 'notes/manuscript_revision/before_nu_spectrum_20261006/figures/new_figure' / name
            assert digest(old) == expected
            before = np.loadtxt(old, delimiter=',', skiprows=1)
            after = np.loadtxt(ROOT / name, delimiter=',', skiprows=1)
            np.testing.assert_array_equal(after[:, :2], (before[:, :2]+1)/2)
            np.testing.assert_array_equal(after[:, 2:4], before[:, 2:4])
            np.testing.assert_allclose(after[:, 4:], 2*before[:, 4:], rtol=1e-13, atol=1e-14)
            continue
        assert digest(ROOT / name) == expected, ('Scientific input changed', name)

    # Cross-check audit receipts against the files actually delivered.
    for path in (ROOT / 'data').glob('*/validation.json'):
        receipt = json.loads(path.read_text())
        for name, expected in receipt.get('outputs', receipt.get('output_sha256', {})).items():
            expected = expected['sha256'] if isinstance(expected, dict) else expected
            assert digest(ROOT / name) == expected, (path, name)

    old = csv_rows('data/correlations/original_plotted_data.csv')
    endpoints = {int(row['Ny']):float(row['unmasked_correlator']) for row in old
                 if row['panel'] == 'c' and int(row['ry']) == int(row['Ny'])//2}
    kept = []
    for old_row in old:
        if old_row['panel'] not in ('a','c') or int(old_row['ry']) < 2:
            continue
        row = dict(old_row)
        ny, r, value = int(row['Ny']), int(row['ry']), float(row['unmasked_correlator'])
        row['source_log_x'], row['source_log_y'] = row['x'], row['y']
        row['x'] = str(float(r) if row['panel'] == 'a' else float(np.sin(np.pi*r/ny)))
        row['y'] = str(value if row['panel'] == 'a' else value/endpoints[ny])
        row['displayed'] = str(row['panel'] == 'c' or value > 1e-20)
        kept.append(row)
    assert kept == csv_rows('data/correlations/plotted_data.csv')
    corr = read_json('data/correlations/validation.json')
    assert corr['display_min_ry'] == 2 and corr['panel_a_cutoff'] == 1e-20
    assert corr['display_panel_source_mapping'] == {'a': 'a', 'b': 'c'}
    assert corr['requested_presentation_values_matched']
    assert typography['Figure_05_correlations']['panel_letters'] == ['(a)', '(b)']
    assert abs(corr['primary_beta_archived'] - corr['primary_beta_independently_recovered']) < 1e-14
    for folder in ('entropy_charge', 'wall_entropy'):
        receipt = read_json(f'data/{folder}/validation.json')
        assert receipt['display_min_Ay'] == 2 and receipt['fits_match_imported_campaign']
        assert receipt['sizes'] == [24,28,32,40,50,60]
        assert receipt['fit_window'] == '8 <= Ay <= Ny/2'
        assert digest(ROOT/'data/endpoint_even_20261008/sample_curves.npz') == receipt['input_sha256']
        # Labels have intentionally changed. Validate scientific inputs and fits,
        # not obsolete pixel equality with the original labels.
        provenance = read_json(f'data/{folder}/input_provenance.json')
        for source in provenance['sources']:
            compact = ROOT / 'data' / folder / Path(source['path']).name
            if compact.is_file() and compact.suffix == '.csv':
                assert digest(compact) == source['sha256'], compact
    bulk = read_json('data/bulk_topology/validation.json')
    assert bulk['layout'] == [2, 1] and bulk['top_row'] == ['geometry', 'marker']
    assert bulk['marker_color_norm'] == 'tanh of trajectory mean, linear color scale [-1,1]'
    assert bulk['convergence_legend']=='lower left' and bulk['phase_labels_raised']
    assert bulk['periodic_edge_marks']=={'top_bottom':'centered single slash','left_right':'centered double slash'}
    assert read_json('data/central_charge/validation.json')['y_label']=='c_eff(t); extraction defined in caption'
    assert read_json('data/ow_truncation/typography_validation.json')['retained_weight_annotation_removed']
    assert all(label in manuscript for label in ('app:PurificationSpectroscopy','app:FiniteTimeGap','app:TrajectoryResponse','app:PureTrajectoryResponse'))
    assert r'fig:WallEntropy}(b,c)' not in manuscript
    assert r'Figure~\ref{fig:WallEntropy}(a) compares' not in manuscript
    assert typography['Figure_04_purification']['panel_letters'] == ['(a)', '(b)']
    purification = read_json('data/purification/notation_validation.json')
    assert purification['layout'] == [2, 1]
    assert purification['displayed_panels'] == ['total_entropy', 'raw_entropy_contour_alpha1_1']
    assert purification['spatial_panel_displayed'] and purification['fit_unchanged']
    assert not purification['slowest_mode_panel_displayed'] and not purification['contour_normalized']
    assert typography['Figure_04_lyapunov']['panel_letters'] == ['(a)', '(b)', '(c)']
    assert purification['lyapunov_layout'] == [1,3]
    assert purification['lyapunov_panels'] == ['occupation_alpha1_3','occupation_alpha1_1','lyapunov_gap']
    assert purification['entropy_geometry_inches']['heatmap_height'] == 1.5
    assert purification['entropy_geometry_inches']['alpha_annotation_axes_coordinates'] == [.5,.5]
    assert purification['entropy_geometry_inches']['alpha_font_matches_entropy_legend']
    assert purification['entropy_scaling_factor'] == 30
    for alpha in (1,3):
        selected = sorted((r for r in csv_rows('data/purification/total_entropy_curves.csv')
                           if int(r['alpha_1']) == alpha),key=lambda r:int(r['cycle']))
        for key,outkey in [('mean','mean'),('sem','trajectory_SEM')]:
            np.testing.assert_array_equal(purification['entropy_plotted_arrays'][str(alpha)][outkey],
                                          30*np.array([float(r[key]) for r in selected]))
    contour = read_json('data/purification/entropy_contour_provenance.json')
    assert contour['status'] == 'passed' and not contour['new_simulations']
    assert not contour['normalization_in_manuscript'] and contour['samples'] == 100
    assert digest(ROOT/'data/purification/Ny030_entropy_contour_cycle60.npz') == contour['output_sha256']
    with np.load(ROOT/'data/purification/Ny030_entropy_contour_cycle60.npz') as z:
        np.testing.assert_array_equal(z['sample_ids'],np.arange(100))
        np.testing.assert_array_equal(z['mean_raw_contour'],z['raw_contours'].mean(0))
        np.testing.assert_allclose(z['raw_contours'].sum((1,2)),z['entropy_per_trajectory'],atol=1e-14,rtol=1e-13)
        np.testing.assert_allclose(z['normalized_contours'].sum((1,2)),1,atol=1e-12)
        np.testing.assert_array_equal(z['mean_normalized_contour'],z['normalized_contours'].mean(0))
        np.testing.assert_allclose(z['mean_raw_contour'].sum(),purification['contour_mean_sum'],atol=1e-14)
    assert r'\includegraphics[width=\textwidth]{Figure_01_schematic.pdf}' in manuscript
    assert manuscript.count(r'\StartStackedFigurePage') == 0
    assert r'\label{eq:SlowestModeDensity}' not in manuscript
    assert r'\label{eq:PurificationEntropyContour}' in manuscript
    pagination = json.loads(subprocess.check_output([
        sys.executable, str(ROOT/'sources/verify_stack_pages.py'),
        str(ROOT.parents[1]/'manuscript.pdf')], text=True))
    assert pagination['status'] == 'passed' and pagination['maximum_vertical_stacks_per_page'] == 1
    assert purification['occupation_cycles'] == [1,5,60]
    occ = read_json('data/purification/occupation_validation.json')
    assert occ['status'] == 'passed' and occ['rank_mean_and_sem_match_raw']
    assert digest(ROOT/'data/purification/ranked_occupation_means.csv') == occ['csv_sha256']
    with (ROOT/'data/purification/ranked_occupation_means.csv').open() as stream:
        occupations = list(csv.DictReader(stream))
    assert len(occupations) == 2*3*1200
    for alpha in (1,3):
        for cycle in (1,5,60):
            selected = sorted((r for r in occupations if int(r['alpha_1'])==alpha and int(r['cycle'])==cycle), key=lambda r:int(r['rank']))
            np.testing.assert_array_equal([int(r['rank']) for r in selected],np.arange(1,1201))
            assert all(int(r['samples'])==100 for r in selected)
            values = np.array([float(r['mean_occupation']) for r in selected])
            assert np.isfinite(values).all() and np.all(np.diff(values)>=0)

    assert read_json('data/entanglement_spectrum/validation.json')['layout'] == [3, 1]
    for group in ('entropy_charge', 'wall_entropy'):
        legend_receipt = read_json(f'data/{group}/validation.json')
        assert legend_receipt['size_legend'] == {
            'title':'N_y','columns':3,'entries':[24,28,32,40,50,60],
            'marker_only':True,'errorbar_glyphs':False}
        assert legend_receipt['plotted_errorbars_retained']
    spectrum_receipt = read_json('data/entanglement_spectrum/validation.json')
    assert spectrum_receipt['count_legend'] == {'marker_only':True,'errorbar_glyphs':False}
    assert spectrum_receipt['plotted_errorbars_retained']
    assert read_json('data/central_charge/validation.json')['layout'] == [1, 1]
    assert typography['Figure_08_central_charge']['panel_letters'] == []

    channel = read_json('data/mean_channel/validation.json')
    assert channel['layout'] == [3, 1] and channel['appendix_scan_points'] == 105
    assert channel['appendix_scan_layout'] == [1, 1]
    assert typography['Figure_11_mean_channel']['panel_letters'] == ['(a)', '(b)', '(c)']
    assert typography['Figure_A04_channel_gap_scan']['panel_letters'] == []
    assert channel['panel_b_display_cutoff'] == 1e-20 and channel['panel_b_min_ry'] == 2
    assert channel['panel_b_axes'] == {'x': 'r_y', 'y': 'C_Gbar(r_y)', 'xscale': 'log', 'yscale': 'log'}
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
        for alpha in (1, 3):
            values = curves[f'alpha{alpha}_xavg'][1:]
            selected = np.flatnonzero(np.isfinite(values) & (values >= 1e-20) & (np.arange(1, len(values)+1) >= 2)) + 1
            assert channel['panel_b_displayed_separations'][str(alpha)] == selected.tolist()
            exported = [int(row['r_y']) for row in csv_rows('data/mean_channel/panel_b_plotted_data.csv') if int(row['alpha_1']) == alpha]
            assert exported == selected.tolist()
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
        occupation_edges = (occupation_edges + 1) / 2
        occupation_density = occupation_counts / (2048000 * np.diff(occupation_edges))
        np.testing.assert_allclose((occupation_density * np.diff(occupation_edges)).sum(axis=1), 1, atol=1e-14)
        saved_occupation = np.genfromtxt(ROOT / 'data/entanglement_spectrum/occupation_histogram.csv', delimiter=',', names=True)
        np.testing.assert_array_equal(saved_occupation['nu_left'], occupation_edges[:-1])
        np.testing.assert_array_equal(saved_occupation['nu_right'], occupation_edges[1:])
        assert not np.any(np.abs(spectra) == .99)
        # Main-text occupations use nu; Appendix B's filter strength uses lambda.
        main_text = manuscript.split(r'\appendix', 1)[0]
        assert r'\lambda' not in main_text and 'λ' not in main_text
        for index, alpha in enumerate((1, 3)):
            np.testing.assert_array_equal(occupation_counts[index], saved_occupation[f'alpha1_{alpha}_count'])
            np.testing.assert_array_equal(occupation_density[index], saved_occupation[f'alpha1_{alpha}_density'])

    energy_edges = np.linspace(-np.log(199), np.log(199), 102)
    selected_energies = [np.log1p(-v[np.abs(v) < .99]) - np.log1p(v[np.abs(v) < .99]) for v in spectra]
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

    comparison_source = read_json('data/entanglement_spectrum/window_count_comparison_provenance.json')
    assert digest(ROOT/'data/entanglement_spectrum/window_count_comparison.npz') == comparison_source['compact_sha256']
    with np.load(ROOT/'data/entanglement_spectrum/window_count_comparison.npz') as comparison:
        counts_by_width = comparison['counts']
        assert counts_by_width.shape == (2,100,16,32)
        with np.load(ROOT/'data/entanglement_spectrum/inputs.npz') as original:
            np.testing.assert_array_equal(counts_by_width[0], original['counts_by_sample_width_origin'])
        np.testing.assert_array_equal(counts_by_width[:,:,-1,:],window_counts)
        averaged = counts_by_width.mean(-1)
        receipt = read_json('data/entanglement_spectrum/validation.json')['count_comparison']
        np.testing.assert_allclose(averaged.mean(1),receipt['means'],atol=1e-13)
        np.testing.assert_allclose(averaged.std(1,ddof=1)/10,receipt['SEMs'],atol=1e-13)
    assert typography['Figure_06_entropy_charge']['panel_letters'] == ['(a)','(b)']

    # The regenerated purification/modular/A2 figures are scientific-data
    # preserving renderers and must never be restored from obsolete PDFs.
    reusable = {row['stem'] for row in read_json('data/existing_assets.json')
                if row.get('preserved', True)}
    assert reusable == set(), 'Typography-normalized figures must use their current renderers'
    assert all((ROOT / 'sources' / row['renderer']).is_file() for row in rows)
    assert [row['number'] for row in rows if row['section'] == 'II'] == ['1']
    assert [row['number'] for row in rows if row['section'] == 'III'] == [str(i) for i in range(2,12)]
    assert [row['number'] for row in rows if row['section'] == 'Appendix'] == ['A1', 'A2']

    purification = read_json('data/purification/notation_validation.json')
    assert purification['status'] == 'passed' and not purification['data_changed']
    assert purification['gap_points'] == 7
    expected_gap_fit = read_json('data/purification/analysis_manifest.json')['gap_fit']
    assert purification['fit'] == expected_gap_fit
    with np.load(ROOT/'data/purification/Ny030_slowest_mode_density.npz') as archived_density:
        np.testing.assert_allclose(archived_density['mean'].sum(), 1, atol=1e-14, rtol=0)
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

    # Compression preserves the protected outlook and all pre-existing numerical products.
    review = ROOT.parents[1] / 'notes/manuscript_revision/compression_review_20261007'
    start = manuscript.index(r'\FloatBarrier'+'\n'+r'{\color{blue}\section{Conclusions and Outlook}')
    end = manuscript.index(r'{\color{blue}'+'\n'+r'\begin{acknowledgments}', start)
    assert manuscript[start:end] == (review/'outlook_exact.tex').read_text()
    hashes = json.loads((review/'scientific_hashes.json').read_text())
    for name, expected in hashes.items():
        assert digest(ROOT.parents[1]/name) == expected, ('Compression changed scientific data',name)
    wall = read_json('data/wall_entropy/validation.json')
    assert wall['fits_match_imported_campaign']
    assert wall['layout']==[1,2] and wall['contour_panel_removed']
    from endpoint_even_support import load_fits
    endpoint_fits, endpoint_contour, endpoint_source = load_fits()
    assert digest(ROOT/'sources/extract_even_endpoint.py') == endpoint_source['extraction_source_sha256']
    assert endpoint_contour.shape == (100,16,20)
    assert endpoint_source['config']['Ny_values'] == [24,28,32,40,50,60]
    for row in endpoint_source['inputs']:
        path=Path(endpoint_source['source_root'])/row['path']
        assert digest(path)==row['sha256']
        assert digest(path.with_suffix('.json'))==row['receipt_sha256']
    for key,folder,label in [('entropy','entropy_charge','c1'),('variance','entropy_charge','k'),('left','wall_entropy','left'),('right','wall_entropy','right')]:
        groups,fit=endpoint_fits[key]
        receipt=read_json(f'data/{folder}/validation.json')
        if folder=='entropy_charge':
            for field in ['slope','slope_covariance_SEM','converted_coefficient','converted_covariance_SEM','R0_squared']:
                np.testing.assert_allclose(fit[field],receipt['fits'][label][field],atol=1e-14,rtol=1e-12)
        else:
            np.testing.assert_allclose(fit['slope'],receipt['slopes_independently_recovered'][label],atol=1e-14,rtol=1e-12)
    assert typography['Figure_09_wall_entropy']['panel_letters'] == ['(a)', '(b)']
    assert read_json('data/entropy_charge/validation.json')['layout'] == [1,2]
    assert read_json('data/entropy_charge/validation.json')['convergence_separate']
    for stem in ('Figure_06_entropy_charge', 'Figure_09_wall_entropy'):
        text = subprocess.check_output(['pdftotext', str(ROOT/(stem+'.pdf')), '-'], text=True)
        assert 'sin' in text and 'D(A' not in text
    for w in ('left','right'):
        np.testing.assert_allclose(wall['displayed_wall_coefficients'][w]['value'], 6*wall['slopes_independently_recovered'][w])
    assert typography['Figure_07_entanglement_spectrum']['panel_letters'] == ['(a)', '(b)', '(c)']
    gap = read_json('data/gap_convergence/validation.json')
    assert gap['sample_means_and_SEMs_verified'] and gap['raw_gap_equals_2t_rate']
    assert gap['cycle_range'][1] == 120 and gap['samples']==100 and gap['protocol']=='slab_only'
    assert gap['time_fit_or_extrapolation'] and not gap['asymptotic_extrapolation'] and not gap['simulation']
    assert gap['layout']==[3,1] and gap['paired_trajectory_SEMs_verified']
    assert gap['slope_estimate_is_not_assumed_asymptotic_gap']
    for name,expected in gap['late_window_inputs'].items():
        assert digest(ROOT/'data/gap_convergence'/name)==expected
    assert typography['Figure_A05_gap_convergence']['panel_letters']==['(a)','(b)','(c)']
    for name, expected in gap['inputs'].items():
        assert digest(ROOT/'data/gap_convergence'/name) == expected

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
        'figure_versions': len(rows),
        'manuscript_figures': len(included),
        'numbering_convention': 'figure_N audit keys use stable asset IDs; index numbers match manuscript',
        'manuscript_figure_mapping': {row['stem']: row['number'] for row in rows if row['included']},
        'typography': typography,
        'combined_figure_note': 'FIGURE_NOTES.md',
        'figure_3_layout': [2, 1],
        'figure_4_layout': [1, 3],
        'figure_1_layout': 'full-width horizontal geometry and circuit',
        'stack_page_boundaries': 0,
        'pagination': pagination,
        'figure_7_layout': [3, 1],
        'figure_9_layout': 'two full-width horizontal wall fits; contour comparison removed',
        'figure_11_layout': [3, 1],
        'figure_A2_layout': [3, 1],
        'figure_A2_contour_origins': 32,
        'figure_A2_contour_trajectories_per_alpha': 100,
        'archived_mutual_information_points': len(mi),
        'restorable_figures': sorted(reusable),
        'figure_11_gap_scan_points': len(gap_rows),
        'figure_11_gap_definition': channel['gap_definition'],
        'figure_11_fits': channel['panel_c_saved_fits_recovered'],
        'figure_11_original_sources_unchanged': len(channel_source['source_files_sha256']),
        'figure_11_proof': str(proof),
        'figure_11_proof_sha256': digest(proof),
        'figure_11_proof_tex_sha256': digest(proof.with_suffix('.tex')),
        'original_figure_assets_and_bundle_files_unchanged': len(originals),
        'preservation_baseline': str(baseline),
        'authorized_manuscript_and_bibliography_edits': True,
        'existing_PDFs_exact_source_copies': sum(row.get('preserved', True) for row in read_json('data/existing_assets.json')),
        'correlation_rows_source_subset_with_display_mask': len(kept),
        'correlation_excluded_ry': [0, 1],
        'correlation_panel_A_cutoff': 1e-20,
        'correlation_fit_unchanged': True,
        'correlation_display_panel_source_mapping': corr['display_panel_source_mapping'],
        'entropy_charge_and_wall_entropy_min_display_Ay': 2,
        'entropy_charge_and_wall_entropy_sizes': [24,28,32,40,50,60],
        'entropy_charge_and_wall_entropy_fits_match_imported_campaign': True,
        'marker_only_legends_with_plotted_errorbars_retained': ['6(a)', '7(c)', '9(a)'],
        'wall_contour_comparison_removed_from_manuscript': True,
        'retained_spectrum_observations': int(counts.sum()),
        'occupation_comparison': {'Ny': 32, 'Ay': 16, 'alpha_1': [1, 3],
                                  'counts_per_ensemble': occupation_counts.sum(axis=1).tolist(),
                                  'normalization': 'unit-area density in nu; full range [0,1]', 'cut_origins': list(range(32))},
        'energy_comparison_retained_observations': retained_totals.tolist(),
        'histogram_bins': len(counts),
        'histogram_normalization': 'occupation: unit area over [0,1]; energy: unit area within window',
        'count_panel': 'restored for both controls; historical alpha_1=1 fit unchanged',
        'half_strip_mean_count': float(mean),
        'half_strip_trajectory_SEM': float(sem),
        'figures': figures,
        'visual_review': read_json('data/visual_review.json'),
        'outlook_exactly_preserved': True,
        'compression_scientific_products_unchanged': len(hashes),
        'gap_convergence': gap,
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
