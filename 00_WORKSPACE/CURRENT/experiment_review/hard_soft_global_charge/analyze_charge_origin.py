"""Resolve global-charge variability and test same-trajectory boundary associations."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from analyze_pure_endpoints import REPO, HERE, PUMP, write_csv


def verify(receipt):
    m = json.loads(receipt.read_text())
    p = receipt.parent / m['result_filename']
    assert p.stat().st_size == m['result_bytes']
    with p.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == m['result_sha256']
    return p, m


def main():
    out = HERE / 'outputs/charge_origin'
    out.mkdir(parents=True, exist_ok=True)
    sources, spatial, groups = [], [], defaultdict(list)
    for receipt in tqdm(sorted((PUMP/'endpoints').rglob('*.completion.json')), desc='Spatial charge split'):
        p, m = verify(receipt)
        sources.append(dict(path=str(p), sha256=m['result_sha256']))
        with np.load(p) as z:
            nx, ny = int(z['Nx']), int(z['Ny'])
            left, right = z['wall_locations']
            d, q = z['density_x'], z['ranks']
            slab = d[:,left:right+1].sum(axis=1)
            exterior = d.sum(axis=1)-slab
            assert np.max(abs(slab+exterior-q)) < 1e-8
            for sid, a, b, c in zip(z['sample_ids'],slab,exterior,q):
                row=dict(shell=str(z['nshell_label']), Nx=nx, Ny=ny, wall=str(z['wall']),
                         sample_id=int(sid), slab_mean_charge=float(a), exterior_mean_charge=float(b), Q=int(c))
                spatial.append(row)
                groups[(row['shell'], nx, row['wall'])].append(row)
    decomposition=[]
    for (shell,nx,wall),rows in sorted(groups.items()):
        assert sorted(r['sample_id'] for r in rows)==list(range(100))
        a=np.array([[r['slab_mean_charge'],r['exterior_mean_charge'],r['Q']] for r in rows])
        cov=np.cov(a.T,ddof=1)
        assert abs(cov[0,0]+cov[1,1]+2*cov[0,1]-cov[2,2])<1e-8
        decomposition.append(dict(shell=shell,Nx=nx,Ny=24,wall=wall,samples=100,
            slab_variance=cov[0,0], exterior_variance=cov[1,1], twice_covariance=2*cov[0,1],
            total_variance=cov[2,2],exterior_variance_over_total=cov[1,1]/cov[2,2]))
    write_csv(out/'spatial_trajectory_charges.csv',spatial)
    write_csv(out/'variance_decomposition.csv',decomposition)

    boundary=[]
    for wall,bundle in [('hard','05_hard_wall_entropy_charge_batched_v2'),('soft','10_soft_wall_entropy_charge_batched_v2')]:
        root=REPO/'00_WORKSPACE/CURRENT/final_production_new_designs'/bundle/'gpu_data'
        for receipt in sorted(root.rglob('*.complete.json')):
            m=json.loads(receipt.read_text())
            if int(m['Ny']) not in (40,60): continue
            p,m=verify(receipt)
            sources.append(dict(path=str(p),sha256=m['result_sha256']))
            with np.load(p) as z:
                ny=int(z['Ny']); ay=z['ay_values']; keep=(ay>=8)&(ay<=ny//2)
                x=np.log(np.sin(np.pi*ay[keep]/ny)); xc=x-x.mean(); proj=xc/np.dot(xc,xc)
                ent=z['endpoint__entropy_von_neumann'][:,keep]
                fv=z['endpoint__charge_variance'][:,keep]
                ms=ent@proj; mf=fv@proj
                residual=np.sqrt(np.mean((ent-ent.mean(axis=1)[:,None]-ms[:,None]*xc)**2,axis=1))
                for i,sid in enumerate(z['sample_ids']):
                    delta=int(z['global_charge'][i,-1])-20*ny
                    boundary.append(dict(wall=wall,Ny=ny,sample_id=int(sid),Q_offset=delta,
                        abs_Q_offset=abs(delta),c_eff=float(3*ms[i]),k_eff=float(np.pi**2*mf[i]),
                        abs_c_minus1=float(abs(3*ms[i]-1)),abs_k_minus1=float(abs(np.pi**2*mf[i]-1)),
                        entropy_shape_rms=float(residual[i])))
    write_csv(out/'boundary_same_trajectory.csv',boundary)
    correlations=[];means=[];rng=np.random.default_rng(2026091402)
    for wall in ('hard','soft'):
        for ny in (40,60):
            rows=sorted([r for r in boundary if r['wall']==wall and r['Ny']==ny],key=lambda r:r['sample_id'])
            assert [r['sample_id'] for r in rows]==list(range(100))
            x=np.array([r['abs_Q_offset'] for r in rows],float)
            means.append(dict(wall=wall,Ny=ny,samples=100,
                total_Q_variance=float(np.var([r['Q_offset'] for r in rows],ddof=1)),
                c_eff_mean=float(np.mean([r['c_eff'] for r in rows])),
                c_eff_sem=float(np.std([r['c_eff'] for r in rows],ddof=1)/10),
                k_eff_mean=float(np.mean([r['k_eff'] for r in rows])),
                k_eff_sem=float(np.std([r['k_eff'] for r in rows],ddof=1)/10)))
            draws=rng.integers(0,100,(5000,100));xx=x[draws];xx-=xx.mean(axis=1)[:,None]
            for metric in ('c_eff','k_eff','abs_c_minus1','abs_k_minus1','entropy_shape_rms'):
                y=np.array([r[metric] for r in rows]);r,p=pearsonr(x,y)
                yy=y[draws];yy-=yy.mean(axis=1)[:,None]
                boot=(xx*yy).sum(axis=1)/np.sqrt((xx*xx).sum(axis=1)*(yy*yy).sum(axis=1))
                lo,hi=np.quantile(boot,[.025,.975])
                correlations.append(dict(wall=wall,Ny=ny,metric=metric,pearson_r=float(r),
                    bootstrap95_low=float(lo),bootstrap95_high=float(hi),p_unadjusted=float(p)))
    # Exploratory family: adjust all 20 correlations together (Benjamini-Hochberg).
    order=np.argsort([r['p_unadjusted'] for r in correlations]);last=1.
    for rank in range(len(order),0,-1):
        row=correlations[order[rank-1]];last=min(last,row['p_unadjusted']*len(order)/rank)
        row['p_BH']=last
    write_csv(out/'boundary_associations.csv',correlations)
    write_csv(out/'boundary_group_means.csv',means)

    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(2,2,figsize=(7.05,5.5))
    for col,wall in enumerate(('hard','soft')):
        ds=[r for r in decomposition if r['wall']==wall and r['shell']=='dense'];nx=[r['Nx'] for r in ds]
        for key,label,color,marker,ls in [('total_variance','Total','black','o','-'),
                ('exterior_variance','Exterior','#c0392b','^',':'),('slab_variance','Slab','#29804a','s','--'),
                ('twice_covariance',r'$2\,\mathrm{Cov}$','0.55','x','-.')]:
            axes[0,col].plot(nx,[r[key] for r in ds],label=label,c=color,marker=marker,ls=ls,ms=4)
        axes[0,col].set(title=f'{wall.capitalize()} wall: charge decomposition',xlabel=r'$N_x$')
        axes[0,col].set_xticks(nx);axes[0,col].legend(frameon=False,fontsize=8,ncol=2)
        for ny,color,marker in [(40,'#2468ad','o'),(60,'#a0522d','^')]:
            rows=[r for r in boundary if r['wall']==wall and r['Ny']==ny]
            axes[1,col].scatter([r['abs_Q_offset'] for r in rows],[r['c_eff'] for r in rows],
                s=12,alpha=.65,c=color,marker=marker,label=f'$N_y={ny}$')
        axes[1,col].axhline(1,c='0.5',ls='--',lw=.8)
        axes[1,col].set(xlabel=r'$|Q-N_xN_y|$',ylabel=r'$c_{\rm eff}$')
        axes[1,col].legend(frameon=False,fontsize=8)
    axes[0,0].set_ylabel('Across-trajectory variance / covariance')
    for i,ax in enumerate(axes.flat):
        ax.tick_params(top=True,right=True);ax.text(-.15,1.05,f'({chr(97+i)})',transform=ax.transAxes)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(out/f'charge_origin_and_boundary.{ext}',dpi=300,bbox_inches='tight')
    (out/'source_manifest.json').write_text(json.dumps(dict(sources=sources,bootstrap_seed=2026091402,
        fit_window='Ay=8..Ny/2 inclusive, origin-averaged full-x strip curves',
        interpretation='Spatial decomposition is covariance across conditional regional means; not intrinsic regional quantum variance. '
        'Different campaigns are analyzed separately. Endpoint associations are exploratory, not causal. '
        'Slab includes both wall columns. Same-trajectory fits use c_eff=3*mS and k_eff=pi^2*mF. '
        '20 correlation p-values adjusted together with BH; bootstrap intervals are pointwise.'),indent=2)+'\n')
    print('DECOMPOSITION',json.dumps(decomposition,indent=2))
    print('BOUNDARY MEANS',json.dumps(means,indent=2))
    print('ASSOCIATIONS',json.dumps(correlations,indent=2))


if __name__=='__main__':main()
