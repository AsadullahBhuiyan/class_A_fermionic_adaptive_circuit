"""Check generated products and preserve pre-existing sources; no dynamics."""
import csv
import json
import re
import subprocess
import numpy as np
import reproduce as r


def read_csv(name):
    with (r.DATA/name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    checks = json.loads((r.DATA/'numerical_checks.json').read_text())
    for path, expected in checks['source_sha256'].items():
        assert r.digest(r.ROOT/path) == expected, ('Regenerate after source change', path)
    protected = json.loads((r.DATA/'protected_sources.json').read_text())
    changed = [p for p, sha in protected.items() if r.digest(r.ROOT/p) != sha]
    concurrent = json.loads((r.DATA/'concurrent_source_changes.json').read_text())
    acknowledged = {x['path']: x for x in concurrent['files']}
    assert set(changed) == set(acknowledged), ('Unreviewed source change', changed)
    for path in changed:
        assert acknowledged[path]['before_sha256'] == protected[path]
        assert acknowledged[path]['after_sha256'] == r.digest(r.ROOT/path)
    rows = read_csv('band_checks.csv')
    assert len(rows) == 20
    maxima = {}
    for support in r.supports():
        group = [x for x in rows if x['support'] == support]
        for row in group:
            assert abs(float(row['retained_weight'])-2*float(row['Z'])) < 1e-13
            assert float(row['frame_identity_residual']) < 1e-12
            assert abs(float(row['chern'])-(0 if support=='onsite' else 1)) < 1e-10
        # Compare Fourier grids holding the momentum grid fixed.
        for size in ('201', '401'):
            pair = [x for x in group if x['momentum_grid'] == size]
            assert len(pair) == 2
            for key in ('Z', 'retained_weight', 'opposite_band_weight',
                        'sampled_uniform_error', 'sampled_min_target_overlap', 'sampled_half_gap'):
                assert abs(float(pair[0][key])-float(pair[1][key])) < 1e-11, (support,key)
        delta = max(abs(float(group[i][key])-float(group[j][key]))
                    for i in range(4) for j in range(4)
                    for key in ('sampled_uniform_error','sampled_min_target_overlap','sampled_half_gap'))
        assert delta < 1e-3
        maxima[support] = delta
    plus = next(x for x in rows if x['support']=='plus' and x['momentum_grid']=='401')
    assert abs(float(plus['sampled_half_gap'])-checks['plus_analytic_half_gap']) < 1e-12
    assert abs(checks['plus_full_gap']-2*checks['plus_analytic_half_gap']) < 1e-12
    roots = read_csv('critical_points.csv')
    assert len(roots) == 8
    for support in ('plus','square1','square2','square4'):
        pair = [float(x['alpha_c']) for x in roots if x['support']==support]
        assert abs(pair[0]-pair[1]) < 1e-9
    windings = read_csv('winding_checks.csv')
    assert len(windings) == 80
    assert all(abs(float(x['winding'])+1) < 1e-10 for x in windings)
    assert all(float(x['minimum_boundary_amplitude']) > 1e-7 for x in windings)
    ranks = read_csv('finite_lattice_rank.csv')
    assert len(ranks) == 8
    assert all(int(x['rank']) == (64 if x['support']=='dense' else 128) for x in ranks)
    a = np.load(r.DATA/'plot_data.npz')
    assert len(a['lambda_corners']) == 101 and a['half_gap'].min() > .9
    source = (r.HERE/'ow_plus_truncation.tex').read_text()
    assert not any(ord(c)<32 and c not in '\n\t' for c in source)
    assert source.count(r'\includegraphics[width=6.5in]') == 2
    assert r'\Gamma_{\boldsymbol m}=G_{\boldsymbol m}^{\mathsf T}' in source
    log = (r.HERE/'build/ow_plus_truncation.log').read_text()
    assert not re.search(r'Overfull|undefined references|undefined citations|Citation .*undefined|Reference .*undefined', log)
    pdf = r.HERE/'ow_plus_truncation.pdf'
    assert r.digest(pdf) == r.digest(r.HERE/'build'/pdf.name)
    pages = int(re.search(r'Pages:\s+(\d+)', subprocess.check_output(['pdfinfo',str(pdf)],text=True))[1])
    assert pages == 7
    helper = r.module_at('prior_validation_helper', r.PREVIOUS)
    style = r.module_at('plus_validation_style', helper.STYLE)
    style.ROOT = r.FIG
    style.inclusion_width = lambda stem: 6.5
    typography = {stem: style.verify_typography(stem)
                  for stem in ('support_windows','mixing_and_corner_removal')}
    output = dict(status='passed', pdf_pages=pages, protected_files_unchanged=len(protected)-len(changed),
                  reviewed_concurrent_source_changes=changed,
                  band_grid_cases=len(rows), winding_cases=len(windings), transition_roots=len(roots),
                  maximum_momentum_grid_differences=maxima, typography=typography,
                  interval_certified=False, circuit_simulations_run=0)
    (r.DATA/'validation.json').write_text(json.dumps(output,indent=2)+'\n')
    manifest = {}
    for path in r.HERE.rglob('*'):
        if (path.is_file() and not any(x in ('build','__pycache__') for x in path.relative_to(r.HERE).parts)
                and path.name != 'output_manifest.json'):
            manifest[str(path.relative_to(r.HERE))] = r.digest(path)
    (r.DATA/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
