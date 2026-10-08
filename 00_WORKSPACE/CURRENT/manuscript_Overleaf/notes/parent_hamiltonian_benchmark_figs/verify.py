"""Audit scientific data, typography, document build, and delivery checksums."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image
import benchmark as b
import render

STEMS=('Parent_02_topology','Parent_03_occupations','Parent_04_gap','Parent_05_correlations',
       'Parent_06_entropy_charge','Parent_07_spectrum','Parent_08_wall_entropy',
       'Parent_09_modular','Parent_A2_contour_mi')


def audit(record):
    sources=b.source_identity()
    for nx,ny,a in b.task_table():
        p=b.HERE/'data/cases'/f'{b.case_id(nx,ny,a)}.npz'
        assert b.verified_case(p,b.configuration(nx,ny,a),sources),p
    validation=json.loads((b.HERE/'data/validation.json').read_text())
    assert validation['scientific_sources']==sources
    assert validation['analysis_script_sha256']==b.sha(b.HERE/'analyze.py')
    assert validation['validation_script_sha256']==b.sha(b.HERE/'validate.py')
    assert validation['saved_products']['verified_cases']==79
    figure_checks={}
    for stem in STEMS:
        info=subprocess.check_output(['pdfinfo',str(render.FIG/f'{stem}.pdf')],text=True)
        assert re.search(r'^Pages:\s+1$',info,re.M)
        figure_checks[stem]=render.style.verify_typography(stem)
        with Image.open(render.FIG/f'{stem}.png') as im:
            assert all(abs(d-300)<.1 for d in im.info['dpi'])
            figure_checks[stem]['png_pixels']=list(im.size)
        figure_checks[stem]['manuscript_counterpart']={
            'Parent_02_topology':'2(b), with table replacing 2(c)',
            'Parent_03_occupations':'3(b,c)','Parent_04_gap':'4(a,b)',
            'Parent_05_correlations':'5(a,b)','Parent_06_entropy_charge':'6(a,b)',
            'Parent_07_spectrum':'7(a,b,c)','Parent_08_wall_entropy':'8(a,b)',
            'Parent_09_modular':'9(a,b,c)','Parent_A2_contour_mi':'A2(a,b)'}[stem]
    log=(b.HERE/'build/parent_hamiltonian_benchmark_figs.log').read_text()
    assert 'BENCHMARK-TEXTWIDTH=510.0pt' in log
    assert 'Overfull' not in log and 'undefined references' not in log and 'undefined citations' not in log
    built=b.HERE/'build/parent_hamiltonian_benchmark_figs.pdf'
    text=subprocess.check_output(['pdftotext','-layout',str(built),'-'],text=True)
    captions=re.findall(r'FIG\.\s+(\d+)\.',text)
    assert captions==list(map(str,range(1,10))),captions
    assert 'TABLE I.' in text
    root_pdf=b.HERE/'parent_hamiltonian_benchmark_figs.pdf'
    if record:shutil.copyfile(built,root_pdf)
    assert root_pdf.exists() and b.sha(root_pdf)==b.sha(built)
    run=json.loads((b.HERE/'data/run.json').read_text())
    # Primary manuscript and figures are inputs; this workflow never writes them.
    assert b.sha(b.MANUSCRIPT/'manuscript.tex')==run['manuscript_sha256']
    assert b.sha(b.MANUSCRIPT/'manuscript.pdf')==run['manuscript_pdf_sha256']
    files=[]
    for p in b.HERE.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(b.HERE)
        if rel.parts[0] in ('build','__pycache__') or p.suffix in ('.log','.pyc') or p.name=='manifest.json':continue
        files.append(p)
    result=dict(title='Parent Hamiltonian Benchmark Figs',configurations=79,figure_groups=9,
                data_uncertainty='deterministic; no SEM',figures=figure_checks,
                manuscript_sha256=run['manuscript_sha256'],manuscript_pdf_sha256=run['manuscript_pdf_sha256'],
                sources=sources,products={str(p.relative_to(b.HERE)):dict(bytes=p.stat().st_size,sha256=b.sha(p)) for p in sorted(files)})
    if record:
        b.json_write(b.HERE/'manifest.json',result)
    else:
        existing=json.loads((b.HERE/'manifest.json').read_text())
        assert result==existing,'Delivery differs from the recorded manifest'
    print(f'PASS: {len(files)} files, 79 configurations, 9 figure groups, all fonts embedded; clean 12-page PDF build.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record',action='store_true')
    audit(parser.parse_args().record)
