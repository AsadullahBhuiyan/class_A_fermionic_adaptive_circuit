#!/usr/bin/env python3
"""All resolved half-system modes; preserve the earlier nearest-four analysis."""
import csv
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from analyze import HERE, REPO, RAW, WALL_X, sha

OUT = HERE / 'full_spectrum_scan'
CAP = 1e-9


def resolve(frame, ny):
    gram = float(np.max(abs(frame.conj().T @ frame - np.eye(frame.shape[1]))))
    if gram > 1e-8: raise ValueError('frame Gram error')
    fa = frame[:20*ny]
    g = fa @ fa.conj().T
    nu, u = eigh(g, driver='evd')
    if nu.min() < -1e-8 or nu.max() > 1+1e-8: raise ValueError('nonphysical occupation')
    ids = np.flatnonzero((nu > CAP) & (nu < 1-CAP))
    eps = np.log1p(-nu[ids])-np.log(nu[ids])
    v = u[:,ids].copy()
    phases = v[np.argmax(abs(v),axis=0),np.arange(len(ids))]
    v *= (phases.conj()/abs(phases))[None,:]
    residual = np.linalg.norm(g@v-v*nu[ids],axis=0)
    if residual.max(initial=0)>1e-9: raise ValueError('eigen residual')
    xy = (abs(v)**2).reshape(ny//2,20,2,-1).sum(axis=2).transpose(2,0,1)
    np.testing.assert_allclose(xy.sum(axis=(1,2)),1,atol=1e-9)
    cut = np.zeros(ny//2,bool);cut[[0,1,ny//2-2,ny//2-1]]=True
    wall = np.zeros(20,bool);wall[WALL_X]=True
    w = xy[:,:,wall].sum(axis=(1,2)); c=xy[:,cut,:].sum(axis=(1,2))
    joint = xy[:,~cut,:][:,:,wall].sum(axis=(1,2))
    gaps = np.minimum(np.r_[np.inf,np.diff(nu)],np.r_[np.diff(nu),np.inf])[ids]
    clusters=np.r_[0,np.cumsum(np.diff(nu)>1e-10)]
    sizes=np.bincount(clusters)
    # Near-cap modes are saved but not represented as uniquely resolved vectors.
    separated=sizes[clusters[ids]]==1
    return dict(occupations=nu, finite_indices=ids, modular_energies=eps,
        eigenvectors=v, xy=xy, wall_weight=w, cut_weight=c, wall_away_from_cut_weight=joint,
        neighbor_occupation_gap=gaps, cluster_ids=clusters[ids], cluster_sizes=sizes[clusters[ids]],
        individually_separated=separated, residual=residual, frame_gram_error=np.array(gram),
        numerical_candidate=(w>.8)&(c<.5),
        robust_candidate=(w>.8)&(c<.5)&separated,
        cap_survival_1e8=(nu[ids]>1e-8)&(nu[ids]<1-1e-8))


def batch(path):
    d=json.loads(path.with_suffix('.complete.json').read_text())
    if (d['status']!='complete' or d['result_filename']!=path.name or
        d['result_bytes']!=path.stat().st_size or sha(path)!=d['result_sha256']):
        raise ValueError(f'invalid input {path}')
    rows=[]; artifacts=[]
    with np.load(path,allow_pickle=False) as z, threadpool_limits(limits=2):
        for key in ['Ny','Nx','alpha_1','alpha_2','nshell','construction','configuration_sha256','task_id']:
            if z[key].item()!=d[key]: raise ValueError(f'identity {key}')
        np.testing.assert_array_equal(z['case_sample_indices'], d['case_sample_indices'])
        ny=int(z['Ny']);wall=str(z['construction']);a=float(z['alpha_1'])
        frames=z['final_frame'];ranks=z['final_ranks']
        for row,sample in enumerate(z['case_sample_indices']):
            r=resolve(frames[row,:,:int(ranks[row])],ny)
            r.update(Ny=np.array(ny), alpha1=np.array(a),wall=np.array(wall),sample=np.array(sample),
                source_file=np.array(str(path.relative_to(REPO))),source_sha256=np.array(d['result_sha256']),
                source_receipt_json=np.array(json.dumps(d,sort_keys=True)))
            out=OUT/'modes'/wall/f'Ny{ny:03d}'/f'alpha1_{a:g}'/f'sample_{sample:03d}.npz'
            out.parent.mkdir(parents=True,exist_ok=True)
            tmp=out.with_suffix('.partial.npz');np.savez_compressed(tmp,**r);os.replace(tmp,out)
            artifacts.append(dict(file=str(out.relative_to(OUT)),sha256=sha(out),
                source_sha256=d['result_sha256'],wall=wall,Ny=ny,alpha1=a,sample=int(sample),
                max_residual=float(r['residual'].max(initial=0)),frame_gram_error=float(r['frame_gram_error'])))
            for k,i in enumerate(r['finite_indices']):
                rows.append(dict(wall=wall,Ny=ny,alpha1=a,sample=int(sample),index=int(i),
                    epsilon=float(r['modular_energies'][k]),occupation=float(r['occupations'][i]),
                    wall_weight=float(r['wall_weight'][k]),cut_weight=float(r['cut_weight'][k]),
                    wall_away_weight=float(r['wall_away_from_cut_weight'][k]),
                    separated=bool(r['individually_separated'][k]),neighbor_gap=float(r['neighbor_occupation_gap'][k]),
                    candidate=bool(r['numerical_candidate'][k]),robust_candidate=bool(r['robust_candidate'][k]),
                    cap_survival_1e8=bool(r['cap_survival_1e8'][k]),file=str(out.relative_to(OUT)),row=k))
    return rows,artifacts


def summaries(rows):
    summary=[]
    for key in sorted({(r['wall'],r['Ny'],r['alpha1']) for r in rows}):
        group=[r for r in rows if (r['wall'],r['Ny'],r['alpha1'])==key]
        s=dict(zip(['wall','Ny','alpha1'],key));s['resolved_modes']=len(group)
        for name,pred in [('wall_over_80',lambda r:r['wall_weight']>.8),
                          ('candidates',lambda r:r['candidate']),
                          ('separated_candidates',lambda r:r['robust_candidate']),
                          ('separated_candidates_cap1e8',lambda r:r['robust_candidate'] and r['cap_survival_1e8'])]:
            sub=[r for r in group if pred(r)];s[name]=len(sub);s[name+'_samples']=len({r['sample'] for r in sub})
            s[name+'_min_abs_epsilon']=min((abs(r['epsilon']) for r in sub),default=None)
        s['energy_windows']=[]
        for edge in [2,4,6,8,12,16,21]:
            sub=[r for r in group if abs(r['epsilon'])<edge]
            selected=[r for r in sub if r['robust_candidate']]
            s['energy_windows'].append(dict(abs_epsilon_lt=edge,modes=len(sub),
                separated_wall_away_modes=len(selected),samples=len({r['sample'] for r in selected})))
        # Threshold sensitivity: fixed geometric windows; report rather than tune.
        s['threshold_sensitivity']=[dict(wall_min=w,cut_max=c,modes=sum(r['separated'] and r['wall_weight']>w and r['cut_weight']<c for r in group)) for w in [.7,.8,.9] for c in [.25,.5]]
        summary.append(s)
    return summary


def plot(rows):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import subprocess
    os.environ['TEXINPUTS']=str(HERE.parent/'hard_wall_tangent_gap_analysis/latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,
        'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}',
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'legend.frameon':False})
    fig,axes=plt.subplots(2,2,figsize=(7.05,5.1),layout='constrained')
    maps,mx=plt.subplots(2,2,figsize=(7.05,5.3),layout='constrained')
    examples=[]
    for row,a in enumerate([1.,3.]):
        for col,wall in enumerate(['hard','soft']):
            sub=[r for r in rows if r['wall']==wall and r['alpha1']==a and r['Ny']==32]
            ax=axes[row,col]
            p=ax.scatter([r['epsilon'] for r in sub],[r['wall_weight'] for r in sub],
                c=[r['cut_weight'] for r in sub],s=2,cmap='viridis',vmin=0,vmax=1,rasterized=True)
            candidates=[r for r in sub if r['robust_candidate']]
            ax.scatter([r['epsilon'] for r in candidates],[r['wall_weight'] for r in candidates],
                s=24,facecolors='none',edgecolors='red',linewidths=.7,label='Separated wall/away candidates')
            ax.axhline(.8,color='.4',ls='--',lw=.7)
            ax.set(title=rf'{wall}, $\alpha_1={a:g}$, $N_y=32$, $S=100$',xlabel=r'$\epsilon$',ylabel='Physical-wall weight',xlim=(-21,21),ylim=(0,1.03))
            ax.text(-.07,1.04,f'({chr(97+2*row+col)})',transform=ax.transAxes)
            if row==0 and col==0:ax.legend(fontsize=7,loc='lower left')
            fig.colorbar(p,ax=ax,label='Entanglement-cut weight')
            # Lowest |epsilon| qualifying mode, not maximum wall localization.
            example=min(candidates,key=lambda r:(abs(r['epsilon']),r['sample'],r['index']))
            examples.append(example)
            with np.load(OUT/example['file']) as z:
                im=z['xy'][example['row']]
                artist=mx[row,col].imshow(im,origin='lower',aspect='auto',cmap='magma',extent=(-.5,19.5,-.5,15.5))
            title=rf"{wall}, $\alpha_1={a:g}$, sample {example['sample']}"+'\n'+rf"$\epsilon={example['epsilon']:.2f}$, $w_{{\rm wall}}={example['wall_weight']:.2f}$, $w_{{\rm cut}}={example['cut_weight']:.2f}$"
            mx[row,col].set(title=title,xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19])
            mx[row,col].text(-.07,1.04,f'({chr(97+2*row+col)})',transform=mx[row,col].transAxes)
            for x in [5,15]:mx[row,col].axvline(x,color='cyan',ls='--',lw=.6)
            maps.colorbar(artist,ax=mx[row,col],label=r'$\sum_\mu |u(x,y,\mu)|^2$')
    for f,name in [(fig,'all_resolved_modes_Ny32'),(maps,'wall_modes_away_from_cut_Ny32')]:
        pdf=OUT/f'{name}.pdf';f.savefig(pdf);plt.close(f)
        subprocess.run(['pdftoppm','-png','-r','300','-singlefile',str(pdf),str(pdf.with_suffix(''))],check=True)
    (OUT/'examples.json').write_text(json.dumps(examples,indent=2)+'\n')


def main():
    paths=sorted(RAW.rglob('*.npz'))
    if len(paths)!=32: raise ValueError('input coverage')
    rows=[];artifacts=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        fs=[pool.submit(batch,p) for p in paths]
        for f in tqdm(as_completed(fs),total=len(fs),desc='All resolved modular eigenvectors',unit='batch'):
            r,a=f.result();rows.extend(r);artifacts.extend(a)
    if len(artifacts)!=1200:raise ValueError('sample count')
    for wall in ['hard','soft']:
        for ny in [24,28,32]:
            for a in [1,3]:
                assert sorted(r['sample'] for r in artifacts if (r['wall'],r['Ny'],r['alpha1'])==(wall,ny,a))==list(range(100))
    rows.sort(key=lambda r:(r['wall'],r['Ny'],r['alpha1'],r['sample'],r['index']))
    with (OUT/'all_resolved_modes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=summaries(rows)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(dict(script_sha256=sha(__file__),samples=1200,
        resolved_modes=len(rows),cap=CAP,occupation_separation=1e-10,subsystem='all x; first half y',
        candidate_rule='wall weight > 0.8 and cut weight < 0.5; individually separated for robust_candidate',
        in_gap_caveat='Scan covers every numerically resolved mode; no physical bulk modular gap boundary assumed.',
        outputs=artifacts),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    plot(rows)


if __name__=='__main__':main()
