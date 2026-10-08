"""Verify saved campaigns, regenerate tables, check proofs, and compile the note.

Run from any directory:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python /path/to/build_note.py
Requires NumPy, SciPy, pypdf, TeX Live (RevTeX 4.2), and Poppler tools.
Only this note directory is written. No dynamics, fits, or remote calls are run.
The TeX source and generated table files can also be compiled independently.
"""
from pathlib import Path
import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import numpy as np
from pypdf import PdfReader
from math_checks import check_math, check_additions, check_pedagogy

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').is_file())
CAMPAIGN = REPO/'00_WORKSPACE/CURRENT/matched_markov_lindblad_campaign'
STEM = 'channel_gap_consolidated'
DATASETS = {
    'fixed_width': ('steady_purity_gap_v1/results/20261005T184136Z', [20, 40, 60, 80, 100]),
    'square': ('square_steady_purity_gap_v1/results/20261005T225446Z', [20, 30, 40, 50, 60, 70, 80]),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_pair(folder):
    receipt = folder/'completion.json'
    rec = json.loads(receipt.read_text())
    assert rec['status'] == 'complete', receipt
    assert Path(rec['result_filename']).name == rec['result_filename'], receipt
    result = folder/rec['result_filename']
    assert result.stat().st_size == rec['result_bytes'], result
    assert sha(result) == rec['result_sha256'], result
    provenance = dict(folder=str(folder.relative_to(REPO)),
                      receipt_sha256=sha(receipt), result_filename=result.name,
                      result_bytes=rec['result_bytes'], result_sha256=rec['result_sha256'],
                      config=rec['config'], saved_source_hashes=rec['sources'])
    return rec, result, provenance


def verify_data():
    records, channel_records, tables = [], [], {}
    common = dict(alpha_2=30, nshell=1, DW=True, dw_truncation=True,
                  all_slabs_active=True, trial_orbitals='X', twist_y=0,
                  boundary_conditions='periodic_x_y', dtype='complex128',
                  perfect_correction=True, dephasing=True, sequence='raster_y',
                  channel_order=['Ap', 'Am', 'Bp', 'Bm'], explicit_dynamics=True,
                  init_mode='maxmix', n_a=0.5, cycles=61,
                  canonical_source='classA_U1FGTN.run_markov_channel',
                  observation_cycles=[0, 1, 5, 10, 20, 40, 60, 61])
    core_hashes = {}
    for geometry, (root, sizes) in DATASETS.items():
        data = CAMPAIGN/root
        csv_path = data/'analysis/gaps.csv'
        with csv_path.open() as stream:
            csv_rows = list(csv.DictReader(stream))
        assert len(csv_rows) == 2*len(sizes)
        csv_index = {(int(float(r['alpha_1'])), int(r['Ny'])): r for r in csv_rows}
        assert set(csv_index) == {(a, ny) for a in [1, 3] for ny in sizes}
        indexed = {}
        for ny in sizes:
            for alpha in [1, 3]:
                nx = 20 if geometry == 'fixed_width' else ny
                prefix = 'Ny' if geometry == 'fixed_width' else 'L'
                folder = data/f'alpha{alpha}_{prefix}{ny:03d}'
                rec, result, provenance = verify_pair(folder)
                cfg = rec['config']
                for key, value in dict(common, Nx=nx, Ny=ny, alpha_1=alpha,
                                       walls=[nx//4, 3*nx//4]).items():
                    assert cfg[key] == value, (folder, key, cfg[key], value)
                assert rec['diagnostics']['converged'], folder
                for key in ['src/fgtn/classA_U1FGTN.py', 'src/fgtn/occupied_frame.py']:
                    core_hashes.setdefault(key, rec['sources'][key])
                    assert rec['sources'][key] == core_hashes[key], (folder, key)
                with np.load(result, allow_pickle=False) as z:
                    assert z['observation_cycles'].tolist() == common['observation_cycles']
                    assert z['occupations_full'].shape == (8, 2*nx*ny)
                    assert z['occupations_ky_twirl'].shape == (8, ny, 2*nx)
                    full = z['occupations_full']
                    twirl = z['occupations_ky_twirl']
                    assert np.isfinite(full).all() and np.isfinite(twirl).all()
                    assert min(full.min(), twirl.min()) >= -1e-10
                    assert max(full.max(), twirl.max()) <= 1+1e-10
                    np.testing.assert_allclose(full[0], 0.5, atol=1e-13, rtol=0)
                    np.testing.assert_allclose(twirl[0], 0.5, atol=1e-13, rtol=0)
                    full_gap = np.min(abs(1-2*full), axis=1)
                    twirl_gap = np.min(abs(1-2*twirl), axis=(1, 2))
                    for suffix, derived in [('full', full_gap), ('twirl', twirl_gap)]:
                        np.testing.assert_allclose(z['purity_gap_'+suffix], derived, atol=1e-13, rtol=0)
                        np.testing.assert_allclose(z['half_filling_distance_'+suffix], derived/2,
                                                   atol=1e-13, rtol=0)
                        np.testing.assert_allclose(rec['diagnostics']['purity_gap_'+suffix], derived[-1],
                                                   atol=1e-13, rtol=0)
                        np.testing.assert_allclose(float(csv_index[alpha, ny]['purity_gap_'+suffix]),
                                                   derived[-1], atol=1e-13, rtol=0)
                        assert abs(derived[-1]-derived[-2]) < 2e-10, folder
                    last_five = float(np.max(z['successive_frobenius_change'][-5:]))
                    assert last_five < 1e-10, folder
                    provenance.update(geometry=geometry, alpha_1=alpha, Nx=nx, Ny=ny,
                                      purity_gap_full=float(full_gap[-1]),
                                      purity_gap_twirl=float(twirl_gap[-1]),
                                      full_gap_cycle60_to61=float(abs(full_gap[-1]-full_gap[-2])),
                                      twirl_gap_cycle60_to61=float(abs(twirl_gap[-1]-twirl_gap[-2])),
                                      last_five_max_frobenius_change=last_five,
                                      final_frobenius_change=float(z['successive_frobenius_change'][-1]),
                                      final_translation_residual=float(z['translation_residual'][-1]))
                if geometry == 'fixed_width':
                    # Resolve by repository-relative suffix, allowing relocation of the repository.
                    ref = rec['spectral_reference']
                    rel = ref['folder'].split('/00_WORKSPACE/', 1)[1]
                    spectral_folder = REPO/'00_WORKSPACE'/rel
                    sr, sp, spectral_provenance = verify_pair(spectral_folder)
                    assert spectral_provenance['receipt_sha256'] == ref['receipt_sha256']
                    assert spectral_provenance['result_sha256'] == ref['spectrum_sha256']
                    for key, value in sr['sources'].items():
                        assert rec['sources'][key] == value, (folder, key)
                    for key in ['Nx', 'Ny', 'alpha_1', 'alpha_2', 'nshell', 'walls', 'sequence',
                                'channel_order', 'dw_truncation', 'all_slabs_active', 'dtype',
                                'dephasing', 'perfect_correction', 'trial_orbitals', 'twist_y',
                                'boundary_conditions']:
                        assert sr['config'][key] == cfg[key], (folder, key)
                    with np.load(sp, allow_pickle=False) as z:
                        radius = float(np.max(abs(z['eigenvalues'])))
                        np.testing.assert_allclose(radius, sr['diagnostics']['spectral_radius'],
                                                   atol=1e-13, rtol=0)
                    gap = 1-radius**2
                    np.testing.assert_allclose(gap, ref['channel_gap'], atol=1e-13, rtol=0)
                    np.testing.assert_allclose(gap, float(csv_index[alpha, ny]['channel_gap']),
                                               atol=1e-13, rtol=0)
                    provenance['channel_gap'] = gap
                    spectral_provenance.update(spectral_radius=radius, channel_gap=gap)
                    channel_records.append(spectral_provenance)
                indexed[alpha, ny] = provenance
                records.append(provenance)
                print(f'Verified {geometry}: alpha={alpha}, Ny={ny}', flush=True)
        lines = ['% Generated from verified cycle-61 occupation arrays by build_note.py.']
        for ny in sizes:
            values = [indexed[alpha, ny][key] for alpha in [1, 3]
                      for key in ['purity_gap_full', 'purity_gap_twirl']]
            lines.append(str(ny)+' & '+' & '.join(f'{v:.6f}' for v in values)+r' \\')
        filename = geometry+'_table.tex'
        (HERE/filename).write_text('\n'.join(lines)+'\n')
        tables[filename] = sha(HERE/filename)
    assert len(records) == 24 and len(channel_records) == 10
    return dict(verified_purity_cases=records, verified_spectral_cases=channel_records,
                table_sha256=tables, shared_core_source_hashes=core_hashes,
                current_core_sources_match_saved={p: sha(REPO/p) == digest
                                                  for p, digest in core_hashes.items()})


def verify_preservation():
    baseline = json.loads((HERE/'preservation_baseline.json').read_text())
    external = json.loads((HERE/'external_change_observation.json').read_text())
    effective = dict(baseline)
    for name, record in external['files'].items():
        assert baseline[name] == record['original_sha256']
        effective[name] = record['observed_sha256']
    changed = [name for name, digest in effective.items() if not (REPO/name).is_file()
               or sha(REPO/name) != digest]
    assert not changed, f'Original files changed: {changed}'
    return dict(checked_files=len(baseline), original_notes_unchanged=True,
                manuscript_preserved_at_observed_external_revision=True,
                external_changes=external,
                baseline_sha256=sha(HERE/'preservation_baseline.json'))


def compile_note():
    build = HERE/'build'
    build.mkdir(exist_ok=True)
    # Pin TeX timestamps for reproducible PDF generation with the same toolchain.
    env = dict(os.environ, SOURCE_DATE_EPOCH='1791244800', FORCE_SOURCE_DATE='1', TZ='UTC')
    for i in range(4):
        result = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                                 '-output-directory=build', STEM+'.tex'], cwd=HERE,
                                env=env, capture_output=True, text=True)
        (build/f'compile-pass-{i+1}.txt').write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'LaTeX failed; see {build}/compile-pass-{i+1}.txt')
    log = (build/(STEM+'.log')).read_text()
    problems = [line for line in log.splitlines() if any(s in line for s in
                ['Overfull', 'undefined references', 'undefined citations', 'multiply defined',
                 'Token not allowed in a PDF string', 'Rerun to get', 'There were undefined',
                 'A float is stuck'])]
    assert not problems, problems
    pdf = HERE/(STEM+'.pdf')
    shutil.copy2(build/pdf.name, pdf)
    doc = PdfReader(pdf)
    assert len(doc.pages) >= 2
    annotations = [a.get_object() for page in doc.pages for a in page.get('/Annots', [])]
    uris = sorted(set(str(a['/A']['/URI']) for a in annotations
                      if '/A' in a and '/URI' in a['/A']))
    assert len(uris) == 25 and all(u.startswith('https://') for u in uris), uris
    font_report = subprocess.check_output(['pdffonts', str(pdf)], text=True)
    (build/'fonts.txt').write_text(font_report)
    for line in font_report.splitlines()[2:]:
        fields = line.split()
        assert fields[-5] == 'yes', line  # embedded
        assert 'Type 3' not in line, line
    source = (HERE/(STEM+'.tex')).read_text()
    assert re.search(r'\\documentclass\[[^\]]*\bonecolumn\b', source)
    assert not re.search(r'\\documentclass\[[^\]]*\btwocolumn\b', source)
    bibs = re.findall(r'\\bibitem\{([^}]+)\}', source)
    assert len(bibs) == len(set(bibs)) == 13
    cited = {key for group in re.findall(r'\\cite\{([^}]+)\}', source)
             for key in group.split(',')}
    assert cited == set(bibs), (cited, bibs)
    subprocess.run(['pdftotext', '-layout', str(pdf), str(build/(STEM+'.txt'))], check=True)
    subprocess.run(['pdftotext', '-bbox', str(pdf), str(build/(STEM+'.xml'))], check=True)
    # Poppler can emit XML-illegal C0 control glyphs for CM math delimiters.
    # Remove those text glyphs only for parsing; retain the raw XML and all bounds.
    bbox_raw = (build/(STEM+'.xml')).read_text()
    bbox_clean, controls = re.subn(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', bbox_raw)
    tree = ET.ElementTree(ET.fromstring(bbox_clean))
    ns = {'h': 'http://www.w3.org/1999/xhtml'}
    for page in tree.findall('.//h:page', ns):
        w, h = float(page.attrib['width']), float(page.attrib['height'])
        for word in page.findall('h:word', ns):
            assert 0 <= float(word.attrib['xMin']) <= float(word.attrib['xMax']) <= w
            assert 0 <= float(word.attrib['yMin']) <= float(word.attrib['yMax']) <= h
    preview = HERE/'preview'
    preview.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '110', '-png', str(pdf), str(preview/'page')],
                   check=True, capture_output=True)
    padding = len(str(len(doc.pages)))
    current_previews = [preview/f'page-{i:0{padding}d}.png' for i in range(1, len(doc.pages)+1)]
    assert all(p.is_file() for p in current_previews)
    # Remove only obsolete generated page images, after the new render succeeds.
    for old in preview.iterdir():
        if re.fullmatch(r'page-[0-9]+\.png', old.name) and old not in current_previews:
            old.unlink()
    return dict(pages=len(doc.pages), columns=1, exposition='pedagogical',
                bibliography_entries=len(bibs), hyperlinks=uris,
                rendered_pages=[dict(page=i, path=str(p.relative_to(HERE)), sha256=sha(p))
                                for i, p in enumerate(current_previews, 1)],
                latex_layout_errors=problems, embedded_fonts_verified=True,
                page_bounds_verified=True, bbox_xml_control_glyphs_filtered=controls,
                page_text_nonempty=all(p.extract_text().strip() for p in doc.pages),
                tex_engine=subprocess.check_output(['pdflatex', '--version'], text=True).splitlines()[0],
                source_date_epoch=env['SOURCE_DATE_EPOCH'])


def build():
    preservation_before = verify_preservation()
    # Confirm the existing 29-test function was reused verbatim, not silently weakened.
    old_source = (HERE.parent/'channel_gap_discussion/build_note.py').read_text()
    new_source = (HERE/'math_checks.py').read_text()
    def check_function(source):
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'check_math')
        return ast.get_source_segment(source, node)
    assert check_function(old_source) == check_function(new_source)
    data = verify_data()
    maths = dict(reused=check_math(), added=check_additions(), pedagogical_examples=check_pedagogy())
    assert maths['reused']['test_count'] == 29
    layout = compile_note()
    preservation_after = verify_preservation()
    source_paths = ['channel_gap_proof/channel_gap_proof.tex',
                    'channel_gap_discussion/channel_gap_discussion.tex',
                    'channel_gap_discussion/build_note.py']
    sources = {p: sha(HERE.parent/p) for p in source_paths}
    artifacts = [STEM+'.tex', STEM+'.pdf', 'fixed_width_table.tex', 'square_table.tex',
                 'build_note.py', 'math_checks.py', 'preservation_baseline.json',
                 'external_change_observation.json']
    record = dict(data=data, mathematical_checks=maths, existing_29_checks_reused_verbatim=True,
                  document=layout,
                  source_note_hashes=sources, preservation_before=preservation_before,
                  preservation_after=preservation_after,
                  build_command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python build_note.py',
                  artifact_sha256={p: sha(HERE/p) for p in artifacts})
    (HERE/'validation.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(pages=layout['pages'], verified_purity_cases=24,
                          verified_spectral_cases=10,
                          mathematical_checks=sum(x['test_count'] for x in maths.values()),
                          maximum_math_error=max(x['maximum_error'] for x in maths.values()),
                          original_notes_unchanged=True,
                          external_manuscript_change_preserved=True), indent=2), flush=True)


if __name__ == '__main__':
    build()
