"""Check receipts, preserve originals, and inventory the final local deliverables."""
import csv
import io
import json
from pathlib import Path
import re
import subprocess
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from reproduce import HERE, ROOT, digest
from test_derivations import DerivationTests


def main():
    baseline = json.loads((HERE/'data/protected_sources.json').read_text())['files']
    changed = [path for path, sha in baseline.items() if digest(ROOT/path) != sha]
    concurrent = json.loads((HERE/'data/concurrent_source_change.json').read_text())
    assert concurrent['before_sha256'] == baseline[concurrent['path']]
    assert changed == [concurrent['path']], f'Unexpected protected-source changes: {changed}'
    assert digest(ROOT/concurrent['path']) == concurrent['observed_sha256'], 'Additional external edit detected'
    receipt = json.loads((HERE/'data/numerical_checks.json').read_text())
    assert receipt['script_sha256'] == digest(HERE/'reproduce.py')
    assert receipt['native_sha256'] == digest(ROOT/receipt['native_source'])
    assert receipt['fourier_grids'] == [1024, 2048]
    assert receipt['momentum_grids'] == [201, 401]
    assert not receipt['new_circuit_simulations'] and not receipt['continuum_bound_certified']
    assert len(receipt['checks']) == 12 and len(receipt['winding_checks']) == 72
    assert receipt['coefficient_convergence_max_abs'] < 1e-12
    assert receipt['onsite']['crossing_residual'] < 1e-13
    legacy_path = ROOT/'00_WORKSPACE/CURRENT/manuscript_Overleaf/notes/truncation_topology/truncation_check.json'
    legacy = json.loads(legacy_path.read_text())
    comparisons = []
    key_pairs = [('sampled_max_operator_error', 'sampled_operator_error'),
                 ('sampled_min_target_band_overlap', 'sampled_min_overlap'),
                 ('sampled_min_absolute_frame_energy', 'sampled_min_abs_frame_energy')]
    for old in legacy['results']:
        new = next(r for r in receipt['checks'] if r['w']==1 and r['fourier_grid']==1024
                   and r['momentum_grid']==old['momentum_grid'])
        comparisons.extend(abs(old[k]-new[j]) for k,j in key_pairs)
    assert max(comparisons) < 1e-12
    for scale in receipt['scales']:
        assert abs(scale['unnormalized_weight']-.5) < 1e-13
        np.testing.assert_allclose(scale['centered_rms']**2,
            scale['rms_squared']-np.dot(scale['centroid'], scale['centroid']), atol=1e-14)
    for row in receipt['checks']:
        assert max(row[k] for k in ('frame_identity_residual', 'hermiticity_residual',
                                   'idempotency_residual')) < 1e-12
    a = np.load(HERE/'data/figure_data.npz')
    for name in a.files:
        assert np.isfinite(a[name]).all(), name
    assert a['cuts'].shape == (4, 2001)
    assert a['min_abs_interpolation_energy'].shape == (3, 252)
    crossing_index = np.argmin(abs(a['interpolation']-receipt['onsite']['s_crossing']))
    assert a['min_abs_interpolation_energy'][0, crossing_index] < 1e-13
    assert a['min_abs_interpolation_energy'][1:].min() > .89
    assert (np.diff(a['discarded']) < 0).all()
    assert a['mismatch'].min() > -1e-13 and a['mismatch'].max() < .01
    for stem in ('ow_mode_truncation', 'auxiliary_band_stability'):
        with Image.open(HERE/f'figures/{stem}.png') as im:
            assert im.width == 1950
            assert abs(im.info['dpi'][0]-300) < .01
        fonts = subprocess.check_output(['pdffonts', str(HERE/f'figures/{stem}.pdf')], text=True)
        assert 'Type 3' not in fonts
        assert (HERE/f'figures/data/typography/{stem}.json').is_file()
    log = (HERE/'build/ow_truncation_consolidated.log').read_text()
    assert not re.search(r'undefined|Overfull|Underfull|LaTeX Error|Emergency stop', log)
    pdf = HERE/'ow_truncation_consolidated.pdf'
    assert digest(pdf) == digest(HERE/'build/ow_truncation_consolidated.pdf')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M)[1])
    bbox_text = subprocess.check_output(['pdftotext', '-bbox', str(pdf), '-'], text=True)
    # Some TeX glyph extractions contain XML-forbidden control characters;
    # discard these from text only, leaving all position attributes unchanged.
    bbox_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', bbox_text)
    bounds = ET.fromstring(bbox_text)
    for page in bounds.findall('.//{*}page'):
        width, height = float(page.attrib['width']), float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            assert 0 <= float(word.attrib['xMin']) <= float(word.attrib['xMax']) <= width
            assert 0 <= float(word.attrib['yMin']) <= float(word.attrib['yMax']) <= height
    test_output = io.StringIO()
    result = unittest.TextTestRunner(stream=test_output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(DerivationTests))
    assert result.wasSuccessful(), test_output.getvalue()
    study = ROOT/'00_WORKSPACE/CURRENT/experiment_review/slab_topology_charge_study/results/v1/summary.json'
    previous = json.loads(study.read_text())
    examples = [dict(L=c['nx'], cycle=c['endpoint_cycle'], samples=c['samples'],
                    saved_center_chern_mean=c['saved_endpoint_chern_mean'],
                    abs_ensemble_mean_minus_one=abs(c['saved_endpoint_chern_mean']-1))
                for c in previous['cases'] if c['cohort']=='square' and c['nx'] in (20,40)]
    assert len(examples) == 2 and all(c['cycle']==40 and c['samples']==100 for c in examples)
    audit = list(csv.DictReader((HERE/'source_audit.csv').open()))
    reference_audit = list(csv.DictReader((HERE/'reference_audit.csv').open()))
    final = dict(status='passed_with_documented_concurrent_manuscript_change',
        protected_files_unchanged=len(baseline)-len(changed), pages=pages,
        concurrent_source_change=concurrent,
        numerical_grid_cases=len(receipt['checks']), local_winding_checks=len(receipt['winding_checks']),
        reproduced_legacy_numerics_max_abs_error=max(comparisons),
        algebra_tests=result.testsRun, test_output=test_output.getvalue(),
        source_audit_rows=len(audit), primary_references=len(reference_audit),
        no_unresolved_citations_or_references=True, no_overfull_or_underfull_boxes=True,
        pdf_text_within_page_bounds=True, figure_png_dpi=300,
        trajectory_source=str(study.relative_to(ROOT)), trajectory_source_sha256=digest(study),
        trajectory_examples=examples, manuscript_modified_by_this_task=False,
        original_notes_modified=False,
        continuum_extrema_certified=False, new_circuit_simulations=False)
    (HERE/'data/validation.json').write_text(json.dumps(final, indent=2)+'\n')
    paths = [p for p in HERE.rglob('*') if p.is_file() and
             not {'build', '__pycache__'}.intersection(p.relative_to(HERE).parts)
             and p.name != 'output_manifest.json']
    manifest = {str(p.relative_to(HERE)):dict(bytes=p.stat().st_size, sha256=digest(p))
                for p in sorted(paths)}
    (HERE/'data/output_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in final.items() if k!='test_output'}, indent=2))


if __name__ == '__main__':
    main()
