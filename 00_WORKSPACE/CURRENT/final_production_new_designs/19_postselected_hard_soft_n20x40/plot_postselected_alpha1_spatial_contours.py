"""Plot checksum-verified alpha1=1 postselected column entropy histories."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'gpu_data/postselected_maxmix_hard_soft_alpha1_3_nx20_ny40_s1_4ny_contours_gpu_v2'
OUT = ROOT/'analysis_outputs/postselected_alpha1_spatial_contours_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = DATA/'DOWNLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    file_rows = {r['path']:r for r in manifest['files']}
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],
        'mathtext.fontset':'cm','font.size':8,'axes.labelsize':8,'legend.fontsize':8,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'pdf.fonttype':42})
    fig, axes = plt.subplots(2,1,figsize=(3.375,4.5),sharex=True,sharey=True,layout='constrained')
    rows, sources = [], []
    for ax, wall, letter in zip(axes,['hard','soft'],['(a)','(b)']):
        prefix=f'alpha1_1/results/{wall}/'
        for name in ['completion.json','postselected_trajectory.npz']:
            row=file_rows[prefix+name]; path=DATA/row['path']
            assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
        path=DATA/prefix/'postselected_trajectory.npz'
        receipt=json.loads(path.with_name('completion.json').read_text())
        run=next(r for r in manifest['runs'] if r['alpha_1']==1 and r['construction']==wall)
        assert sha(path)==receipt['result_sha256']
        assert receipt['source_hashes']==run['executed_source_hashes']
        with np.load(path,allow_pickle=False) as z:
            assert str(z['construction'])==wall and float(z['alpha_1'])==1
            assert str(z['configuration_hash'])==receipt['configuration_hash']==run['configuration_hash']
            assert (int(z['Nx']),int(z['Ny']))==(20,40)
            t=z['cycles']; s=z['entropy_contour'].sum(axis=2)
            np.testing.assert_array_equal(t,np.arange(161))
            np.testing.assert_allclose(s,z['entropy_contour_x'],atol=1e-12)
            np.testing.assert_allclose(s.sum(axis=1),z['total_entropy_nats'],atol=1e-10)
        assert np.isfinite(s).all() and (s[:,[5,10,15]]>0).all()
        endpoints={}
        for x,color,marker,style in [(5,'#d62728','^',':'),(10,'#2ca02c','s','--'),(15,'#1f77b4','o','-')]:
            ax.plot(t,s[:,x],color=color,marker=marker,ls=style,lw=1.2,ms=3,
                    mfc='white',mew=.8,markevery=16,label=rf'$x={x}$')
            endpoints[str(x)]=float(s[-1,x])
            rows.extend(dict(construction=wall,alpha_1=1,cycle=int(cycle),x=x,entropy_x_nats=float(value))
                        for cycle,value in zip(t,s[:,x]))
        ax.set(yscale='log',xlim=(0,160),ylabel=r'$s_x(t)$ (nats)',title=f'{wall.capitalize()} wall')
        ax.set_xticks([0,40,80,120,160])
        ax.legend(frameon=False,loc='center right')
        ax.text(-.17,1.045,letter,transform=ax.transAxes,fontsize=9)
        sources.append(dict(construction=wall,path=str(path),sha256=sha(path),endpoint_entropy_x_nats=endpoints))
    axes[-1].set_xlabel(r'cycle $t$')
    OUT.mkdir(parents=True,exist_ok=True)
    stem=OUT/'postselected_alpha1_entropy_x5_x10_x15'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'),dpi=300)
    plt.close(fig)
    with (OUT/'curves.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(alpha_1=1,Nx=20,Ny=40,cycles=160,samples_per_wall=1,
        estimator='s_x(t)=sum_y entropy_contour[t,x,y]; orbitals already summed',
        normalization='None; entropy in nats and raw cycle count',uncertainty='None: single postselected trajectory',
        scales=dict(x='linear',y='log'),manifest_sha256=sha(manifest_path),sources=sources)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
