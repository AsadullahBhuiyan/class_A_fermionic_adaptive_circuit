#!/usr/bin/env python3
"""Sample-first probability densities of full one-leg cocycle singular modes."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from tqdm import tqdm
from extract_endpoint_singular_modes import sha256

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'endpoint_singular_modes_ny40_v1'
OUT = HERE / 'cocycle_mean_heatmaps'


def mean_density(vectors, indices):
    if not vectors.shape[1]: raise ValueError('Empty mode selection')
    density = np.zeros(1600)
    density[indices] = np.mean(abs(vectors)**2, axis=1)
    density = density.reshape(40, 20, 2).sum(axis=2)
    np.testing.assert_allclose(density.sum(), 1, atol=1e-10)
    return density


def near_zero(rates, mask):
    ids = np.flatnonzero(mask)
    if len(ids) < 4: raise ValueError('Fewer than four resolved modes')
    limit = np.sort(abs(rates[ids]))[3]
    return ids[abs(rates[ids]) <= limit + 1e-10]


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    if not manifest['complete'] or manifest['samples_completed'] != 200:
        raise ValueError('Incomplete source extraction')
    OUT.mkdir(exist_ok=True)
    arrays = {(a, side, sel): [] for a in [1,3] for side in ['initial','endpoint'] for sel in ['all','nearest4']}
    counts = {(a,sel): [] for a in [1,3] for sel in ['all','nearest4']}
    inputs, sample_rows = [], []
    seen=set()
    for d in tqdm(manifest['results'], desc='Verify and average cocycle vectors', unit='sample'):
        a, sample = int(d['task']['alpha_1']), int(d['task']['sample_index'])
        if (a,sample) in seen: raise ValueError('Duplicate sample')
        seen.add((a,sample))
        p=SOURCE/f'alpha1_{a}'/d['result_filename']
        if p.stat().st_size!=d['result_bytes'] or sha256(p)!=d['result_sha256']:
            raise ValueError(f'Input checksum: {p}')
        with np.load(p,allow_pickle=False) as z:
            if (int(z['Nx']),int(z['Ny']),int(z['cycles']),int(z['alpha_1']),int(z['sample_index']))!=(20,40,80,a,sample):
                raise ValueError('Scientific identity mismatch')
            u=z['left_vectors_initial_active']; vh=z['right_vectors_endpoint_dagger']
            active=z['active_input_indices']; mask=z['resolved_mask'];rates=z['resolved_one_leg_rates_per_cycle']
            s=z['singular_values_normalized']; threshold=float(z['numerical_rank_threshold_normalized'])
            np.testing.assert_array_equal(mask,s>threshold)
            selections={'all':np.flatnonzero(mask),'nearest4':near_zero(rates,mask)}
            for sel,ids in selections.items():
                counts[(a,sel)].append(len(ids))
                for side,v,indices,profile_name in [
                    ('initial',u[:,ids],active,'initial_x_profiles'),
                    ('endpoint',vh[ids].conj().T,np.arange(1600),'endpoint_x_profiles')]:
                    density=mean_density(v,indices)
                    np.testing.assert_allclose(density.sum(axis=0),z[profile_name][ids].mean(axis=0),atol=1e-10)
                    arrays[(a,side,sel)].append(density)
                    sample_rows.append(dict(alpha1=a,sample=sample,side=side,selection=sel,modes=len(ids),
                        wall_weight=float(density[:,[4,5,6,14,15,16]].sum()),
                        exterior_weight=float(density[:,:5].sum()+density[:,16:].sum()),
                        min_selected_rate=float(rates[ids].min()),max_selected_rate=float(rates[ids].max())))
        inputs.append(dict(file=str(p.relative_to(HERE)),sha256=d['result_sha256']))
    if seen!={(a,s) for a in [1,3] for s in range(100)}:raise ValueError('Coverage mismatch')
    means={};summary=[]
    for (a,side,sel),values in arrays.items():
        values=np.stack(values);mean=values.mean(axis=0);sem=values.std(axis=0,ddof=1)/10
        means[(a,side,sel)]=mean
        np.savez_compressed(OUT/f'alpha1_{a}_{side}_{sel}.npz',sample_densities=values,
            mean_density=mean,sem_density=sem,mode_counts=counts[(a,sel)],
            sample_indices=[d['sample'] for d in sample_rows if (d['alpha1'],d['side'],d['selection'])==(a,side,sel)],
            Nx=20,Ny=40,cycles=80,alpha1=a,side=side,selection=sel)
        group=[d for d in sample_rows if (d['alpha1'],d['side'],d['selection'])==(a,side,sel)]
        weights=np.array([r['wall_weight'] for r in group])
        summary.append(dict(alpha1=a,side=side,selection=sel,samples=100,
            mode_count_min=min(counts[(a,sel)]),mode_count_max=max(counts[(a,sel)]),
            mode_count_mean=float(np.mean(counts[(a,sel)])),
            mean_wall_weight=float(weights.mean()),sem_wall_weight=float(weights.std(ddof=1)/10),
            mean_exterior_weight=float(mean[:,:5].sum()+mean[:,16:].sum())))
    plot(means)
    (OUT/'manifest.json').write_text(json.dumps(dict(script_sha256=sha256(Path(__file__)),inputs=inputs,
        estimator='Normalize within each selected singular subspace; average densities over 100 trajectories, never average vectors or cocycles',
        convention='K=exp(ell) U diag(s) Vdagger; HT=Kdagger H0 K; initial KKdagger, endpoint KdaggerK',
        limitation='Unrestricted one-leg singular modes, NOT occupied-empty physical tangent pair modes',
        summary=summary,samples=sample_rows,
        outputs=[dict(file=p.name,sha256=sha256(p)) for p in sorted(OUT.glob('*.npz'))]),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def plot(means):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    os.environ['TEXINPUTS']=str(HERE/'latex_support')+os.pathsep+os.environ.get('TEXINPUTS','')
    mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif'],'font.size':8,
        'text.usetex':True,'text.latex.preamble':r'\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}',
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True})
    for sel in ['all','nearest4']:
        fig,axes=plt.subplots(2,2,figsize=(7.05,5.5),layout='constrained')
        vmax=max(float(means[(a,side,sel)].max()) for a in [1,3] for side in ['initial','endpoint'])
        for row,side in enumerate(['initial','endpoint']):
            for col,a in enumerate([1,3]):
                ax=axes[row,col]
                im=ax.imshow(means[(a,side,sel)],origin='lower',interpolation='nearest',aspect='auto',
                    extent=(-.5,19.5,-.5,39.5),cmap='magma',vmin=0,vmax=vmax)
                operator=r'KK^\dagger' if side=='initial' else r'K^\dagger K'
                ax.set(title=rf'${operator}$ ({side}), $\alpha_1={a}$',xlabel='$x$',ylabel='$y$',
                    xticks=[0,5,10,15,19],yticks=[0,10,20,30,39])
                ax.text(-.06,1.04,f'({chr(97+row*2+col)})',transform=ax.transAxes,ha='right')
        title='All numerically resolved modes' if sel=='all' else 'Four one-leg rates closest to zero (including ties)'
        fig.suptitle(title+r'; hard walls, $N_x=20$, $N_y=40$, $S=100$, $T=80$')
        fig.colorbar(im,ax=axes.ravel().tolist(),label=r'Sample-averaged probability $\overline{p(x,y)}$',shrink=.9)
        pdf=OUT/f'cocycle_mean_density_{sel}.pdf';fig.savefig(pdf);plt.close(fig)
        subprocess.run(['pdftoppm','-png','-r','300','-singlefile',str(pdf),str(pdf.with_suffix(''))],check=True)


if __name__=='__main__':main()
