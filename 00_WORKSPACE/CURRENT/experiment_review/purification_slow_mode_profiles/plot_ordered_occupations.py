"""Ascending sample-resolved occupation spectra of slot07 hard endpoints."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from analyze_profiles import sha256

HERE=Path(__file__).resolve().parent
OUT=HERE/'ordered_hard_occupations'


def main():
    OUT.mkdir(exist_ok=True)
    manifest=json.loads((HERE/'analysis_manifest.json').read_text())
    hashes={r['result_file']:r['result_sha256'] for r in manifest['outputs']}
    os.environ['TEXINPUTS']=str(HERE.parent/'hard_wall_tangent_gap_analysis/latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,
        'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}',
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    fig,axes=plt.subplots(2,3,figsize=(7.05,5.1),layout='constrained')
    summary=[];inputs=[]
    for col,(ny,color,marker) in enumerate(zip([20,30,40],['#d62728','#2ca02c','#1f77b4'],['^','s','o'])):
        paths=sorted((HERE/'endpoint_modes/hard'/f'Ny{ny:03d}').glob('sample_*.npz'))
        if len(paths)!=100:raise ValueError('sample coverage')
        values=[];ids=[];counts=[]
        for p in paths:
            digest=sha256(p)
            if digest!=hashes[str(p.relative_to(HERE))]:raise ValueError('source checksum')
            with np.load(p) as z:
                nu=z['occupations'];keep=z['finite_mask']
                assert str(z['construction'])=='hard' and int(z['cycles'])==4*ny
                assert len(nu)==22*ny and np.all(np.diff(nu)>=0)
                np.testing.assert_array_equal(keep,(nu>1e-9)&(nu<1-1e-9))
                values.append(nu);ids.append(int(z['sample_index']));counts.append(int(keep.sum()))
                axes[0,col].plot(np.arange(1,len(nu)+1),nu,color=color,alpha=.12,lw=.5,rasterized=True)
                finite=nu[keep]
                axes[1,col].plot(np.arange(1,len(finite)+1),finite,color=color,alpha=.18,lw=.4,
                    marker=marker,markersize=2,rasterized=True)
            inputs.append(dict(file=str(p.relative_to(HERE)),sha256=digest))
        np.testing.assert_array_equal(ids,np.arange(100))
        mean=float(np.mean(counts));unique,freq=np.unique(counts,return_counts=True)
        summary.append(dict(Ny=ny,samples=100,active_dimension=22*ny,cycles=4*ny,
            unsaturated_mean=mean,counts={str(n):int(f) for n,f in zip(unique,freq)}))
        np.savez_compressed(OUT/f'Ny{ny:03d}_ordered_spectra.npz',occupations=values,
            sample_indices=ids,unsaturated_counts=counts,Ny=ny,cycles=4*ny,tolerance=1e-9)
        axes[0,col].set(title=rf'$N_y={ny}$, $S=100$'+ '\n'+rf'$2$--$4$ unsaturated; mean ${mean:.2f}$',
            xlabel='Ascending eigenvalue index',ylabel=r'Occupation $\nu_i$' if col==0 else '',
            ylim=(-.03,1.03),xlim=(1,22*ny),yticks=[0,.5,1])
        axes[1,col].set_yscale('logit')
        axes[1,col].set(xlabel='Ascending unsaturated index',ylabel=r'Occupation $\nu_j$ (logit scale)' if col==0 else '',
            title='Unsaturated modes only',xlim=(.8,4.2),xticks=[1,2,3,4],ylim=(7e-10,1-7e-10))
        ticks=[1e-9,1e-6,1e-3,.5,1-1e-3,1-1e-6,1-1e-9]
        axes[1,col].set_yticks(ticks,[r'$10^{-9}$',r'$10^{-6}$',r'$10^{-3}$',r'$1/2$',r'$1-10^{-3}$',r'$1-10^{-6}$',r'$1-10^{-9}$'])
        axes[1,col].minorticks_off()
        for t in [1e-9,.5,1-1e-9]:axes[1,col].axhline(t,color='.6',ls='--',lw=.5,zorder=0)
        for row in [0,1]:axes[row,col].text(-.08,1.045,f'({chr(97+row*3+col)})',transform=axes[row,col].transAxes,ha='right')
    pdf=OUT/'slot07_hard_endpoint_occupations_ascending.pdf'
    fig.savefig(pdf);plt.close(fig)
    subprocess.run(['pdftoppm','-png','-r','300','-singlefile',str(pdf),str(pdf.with_suffix(''))],check=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(script_sha256=sha256(Path(__file__)),
        cap=1e-9,summary=summary,inputs=inputs),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
