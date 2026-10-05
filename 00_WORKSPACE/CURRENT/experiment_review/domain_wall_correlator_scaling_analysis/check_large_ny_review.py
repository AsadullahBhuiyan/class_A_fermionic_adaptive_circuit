"""Check completed review artifacts and render six-page visual contact sheets."""
from pathlib import Path
import json

import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from build_large_ny_correlator_review import OUTPUT


def main():
    output=OUTPUT
    pages=json.loads((output/'figure_index.json').read_text())
    def page_count(path):
        info=subprocess.check_output(['pdfinfo',str(path)],text=True)
        return int(next(line.split(':',1)[1] for line in info.splitlines() if line.startswith('Pages:')))
    assert page_count(output/'large_ny_correlator_review_atlas.pdf')==len(pages)==31
    for entry in pages:
        for suffix in ('png','pdf'):
            assert (output/f"{entry['stem']}.{suffix}").stat().st_size>1000
        assert page_count(output/f"{entry['stem']}.pdf")==1
        text=subprocess.check_output(['pdftotext',str(output/f"{entry['stem']}.pdf"),'-'],text=True)
        assert text.strip(),entry['stem']
        assert (output/f"{entry['stem']}_plotted.csv").stat().st_size>100
        if entry['stem']=='fit_window_grid':
            assert '4' in text and '3' in text and '2' in text
    diagnostics=json.loads((output/'ground_state_diagnostics.json').read_text())
    assert len(diagnostics['cases'])==6
    for d in diagnostics['cases']:
        assert d['half_filling_rank']==20*d['Ny']
        assert d['projector_idempotency_max_abs']<1e-11
        assert d['projector_block_hermiticity_max_abs']<1e-11
    summary=json.loads((output/'fit_summary.json').read_text())
    for ny in (24,28,32,40,50,60):
        for observable,windows in summary[f'primary_{ny}'].items():
            assert windows['primary']['trajectory_beta']['n']==100
            assert windows['primary']['invalid_fits']==0
    qa=output/'visual_qa';qa.mkdir(exist_ok=True)
    for first in range(0,len(pages),6):
        fig,axes=plt.subplots(2,3,figsize=(15,10),layout='constrained')
        for ax in axes.ravel():ax.axis('off')
        for ax,entry in zip(axes.ravel(),pages[first:first+6]):
            pixels=plt.imread(output/f"{entry['stem']}.png")
            assert np.std(pixels)>.01
            ax.imshow(pixels)
            ax.set_title(f"{entry['page']}. {entry['stem']}",fontsize=9)
        fig.savefig(qa/f'pages_{first+1:02}_{min(first+6,len(pages)):02}.png',dpi=140)
        plt.close(fig)
    (qa/'artifact_checks.json').write_text(json.dumps(dict(
        atlas_pages=31,individual_figures=31,static_references=6,
        primary_trajectory_count=600,all_primary_fits_valid=True,
        note='Contact sheets are visual-review aids; automated checks do not claim human inspection.'
    ),indent=2)+'\n')
    print('31 PDF/PNG/CSV figure sets, atlas, primary fits and six ground-state references checked.')


if __name__=='__main__':main()
