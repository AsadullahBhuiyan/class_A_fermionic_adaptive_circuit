"""Trajectory-first statistics, figures, and bounded conclusions for the study."""
import csv
import json
from pathlib import Path
import string
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_core import geometry,region_columns,REGIONS
from run_study import digest,write_json,ROOT

COLORS = ('#d94738','#249b57','#1675bd')
MARKERS = ('^','s','o')
STYLES = (':','--','-')


def stats(values):
    values = np.asarray(values)
    assert len(values) == 100
    mean = values.mean(axis=0)
    sem = values.std(axis=0,ddof=1)/10
    np.testing.assert_allclose(sem,np.sqrt(((values-mean)**2).sum(axis=0)/(100*99)),atol=1e-14,rtol=1e-10)
    return mean,sem


def variance_budget(slab,exterior):
    exterior = np.broadcast_to(exterior,slab.shape)
    total = slab+exterior
    vs,ve,vt = [v.var(axis=0,ddof=1) for v in (slab,exterior,total)]
    cov = ((slab-slab.mean(axis=0))*(exterior-exterior.mean(axis=0))).sum(axis=0)/99
    np.testing.assert_allclose(vt,vs+ve+2*cov,rtol=1e-10,atol=1e-9)
    return vt,vs,ve,2*cov


def save_csv(path,rows):
    if not rows: return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)


def plot_line(ax,x,values,color,label,marker=None,ls='-',log=False,display_floor=0):
    mean,sem = stats(values)
    shown = np.where(mean>display_floor,mean,np.nan) if log else mean
    ax.plot(x,shown,color=color,label=label,lw=1,marker=marker,ms=3,mfc='white',ls=ls)
    low = mean-sem
    if log: low=np.where(low>display_floor,low,np.nan)
    ax.fill_between(x,low,mean+sem,color=color,alpha=.13,lw=0)
    if log: ax.set_yscale('log')


def finish(fig,axes,path,caption):
    for label,ax in zip(string.ascii_lowercase,np.asarray(axes).ravel()):
        ax.tick_params(top=True,right=True)
        ax.text(-.12,1.03,f'({label})',transform=ax.transAxes,fontsize=9)
    for ext in ('pdf','png'):fig.savefig(path.with_suffix('.'+ext),dpi=300)
    path.with_suffix('.txt').write_text(caption+'\n')
    plt.close(fig)


def load_groups(out):
    inventory = json.loads((out/'input_inventory.json').read_text())
    identity = json.loads((out/'study_identity.json').read_text())
    grouped,products = {},[]
    for row in inventory:
        cohort,task = row['cohort'],row['task']
        nx,ny = task.get('nx',20),task['ny']
        path = out/'batches'/cohort/(task['id']+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert receipt['file'] == digest(path)
        assert receipt['identity']['study'] == identity and receipt['identity']['input'] == row
        assert receipt['identity']['reference'] == digest(out/'references'/f'nx{nx}_ny{ny}.npz')
        with np.load(path,allow_pickle=False) as z: payload={k:z[k] for k in z.files}
        grouped.setdefault((cohort,nx,ny),[]).append(payload)
        products.append(dict(path=str(path),**receipt['file']))
    assert len(products)==38 and len(grouped)==6
    result = {}
    common = {'cycles','radii','reference_names'}
    for key,parts in grouped.items():
        joined = {}
        for k in parts[0]:
            if k in common:
                for p in parts[1:]:np.testing.assert_array_equal(p[k],parts[0][k])
                joined[k]=parts[0][k]
            else:joined[k]=np.concatenate([p[k] for p in parts])
        order=np.argsort(joined['sample_ids'])
        for k in joined:
            if k not in common:joined[k]=joined[k][order]
        np.testing.assert_array_equal(joined['sample_ids'],np.arange(100))
        result[key]=joined
    return result,products


def report(out):
    groups,inputs=load_groups(out)
    tables=out/'tables';figures=out/'figures'
    tables.mkdir(exist_ok=True);figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,'mathtext.fontset':'cm',
        'legend.frameon':False,'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42})
    charge_rows,size_rows,budget_rows,chern_rows,radius_rows,corr_rows,mismatch_rows,sample_rows=[],[],[],[],[],[],[],[]
    case_summaries=[]
    association_rows=[]
    references={}
    for (cohort,nx,ny),a in groups.items():
        left,right,active,outside,radii,distance=geometry(nx,ny)
        base=dict(cohort=cohort,nx=nx,ny=ny,endpoint_cycle=int(a['cycles'][-1]),samples=100)
        q=a['global_charge'];qs=a['q_slab'];qe=np.broadcast_to(a['q_exterior'][:,None],q.shape)
        np.testing.assert_array_equal(q,qs+qe)
        budget=variance_budget(qs,qe)
        for i,t in enumerate(a['cycles']):
            budget_rows.append(dict(**base,cycle=int(t),var_total=budget[0][i],var_slab=budget[1][i],var_exterior=budget[2][i],twice_covariance=budget[3][i]))
        a['charge_regions']={}
        for name,values,modes in [('total',q,2*nx*ny),('slab',qs,len(active)),('exterior',qe,len(outside))]:
            delta=values-modes/2
            a['charge_regions'][name]=(delta,modes)
            fm,fs=stats(delta/modes);am,ase=stats(abs(delta)/modes)
            for i,t in enumerate(a['cycles']):
                charge_rows.append(dict(**base,region=name,orbitals=modes,cycle=int(t),
                    signed_filling_mean=fm[i],signed_filling_sem=fs[i],absolute_filling_mean=am[i],absolute_filling_sem=ase[i],
                    filling_variance=(delta[:,i]/modes).var(ddof=1),charge_variance=delta[:,i].var(ddof=1)))
            for window,sel in [('cycles21_40',slice(21,41)),('cycle40',slice(40,41)),('endpoint',slice(-1,None))]:
                signed=delta[:,sel].mean(axis=1);absolute=abs(delta[:,sel]).mean(axis=1)
                sm,ss=stats(signed);mm,ms=stats(absolute)
                size_rows.append(dict(**base,region=name,window=window,orbitals=modes,signed_charge_mean=sm,signed_charge_sem=ss,
                    absolute_charge_mean=mm,absolute_charge_sem=ms,signed_filling_mean=sm/modes,signed_filling_sem=ss/modes,
                    absolute_filling_mean=mm/modes,absolute_filling_sem=ms/modes,
                    mean_cyclewise_charge_variance=delta[:,sel].var(axis=0,ddof=1).mean(),
                    mean_cyclewise_filling_variance=(delta[:,sel]/modes).var(axis=0,ddof=1).mean()))
        c=a['center_average'];cm,cs=stats(c);ma,mas=stats(abs(c-1))
        for i,t in enumerate(a['cycles']):
            qu=np.quantile(c[:,i],[.05,.25,.5,.75,.95])
            chern_rows.append(dict(**base,cycle=int(t),mean=cm[i],sem=cs[i],mean_absolute_deviation=ma[i],sem_absolute_deviation=mas[i],
                absolute_mean_deviation=abs(cm[i]-1),q05=qu[0],q25=qu[1],median=qu[2],q75=qu[3],q95=qu[4],
                center_variance_mean=a['real_space_chern'][:,i].var(axis=1,ddof=1).mean()))
        radius_mean=a['chern_radius'].mean(axis=2)
        for ir,r in enumerate(radii):
            v=radius_mean[:,ir];m,s=stats(v);am,ase=stats(abs(v-1))
            spatial=a['chern_radius'][:,ir].var(axis=1,ddof=1)
            radius_rows.append(dict(**base,radius=int(r),rho=r/distance,contained=True,mean=m,sem=s,
                absolute_mean_deviation=abs(m-1),mean_absolute_deviation=am,sem_absolute_deviation=ase,
                center_variance_mean=spatial.mean(),center_variance_sem=spatial.std(ddof=1)/10))
        for axis,values,regions in [('y',a['corr_y'],REGIONS),('x',a['corr_x'],('core1','core2','core3'))]:
            for k,region in enumerate(regions):
                for j in range(values.shape[-1]):
                    if not np.isfinite(values[:,k,j]).all():continue
                    m,s=stats(values[:,k,j]);corr_rows.append(dict(**base,source='trajectory',reference='',axis=axis,region=region,separation=j+1,mean=m,sem=s))
        refpath=out/'references'/f'nx{nx}_ny{ny}.npz'
        with np.load(refpath,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json']))
            references[cohort,nx,ny]={m['key']:{k:z[m['key']+'__'+k] for k in ('chern','corr_y','corr_x')} for m in meta}
        columns=region_columns(nx)
        edgecols=np.r_[np.arange(left,left+3),np.arange(right-2,right+1)]
        for ir,item in enumerate(meta):
            ref=item['key'];maps=a['mismatch_maps'][:,ir]
            for region,cols in [('slab',np.arange(left,right+1)),('interfaces',edgecols)]+[(n,columns[n]) for n in ('core1','core2','core3')]:
                density=maps[:,cols,:].mean(axis=(1,2))/2
                m,s=stats(density)
                mismatch_rows.append(dict(**base,reference=ref,unique_half_filling=item['unique_half_filling'],reference_rank=item['rank'],
                    half_filling_gap=item['gap'],region=region,density_mean=m,density_sem=s,
                    particles_mean=a['mismatch'][:,ir,0].mean(),holes_mean=a['mismatch'][:,ir,1].mean()))
            rv=references[cohort,nx,ny][ref]
            for axis,values,regions in [('y',rv['corr_y'],REGIONS),('x',rv['corr_x'],('core1','core2','core3'))]:
                for k,region in enumerate(regions):
                    for j,v in enumerate(values[k]):
                        if np.isfinite(v):corr_rows.append(dict(**base,source='reference',reference=ref,axis=axis,region=region,separation=j+1,mean=v,sem=0))
            for s in range(100):
                vals=a['mismatch'][s,ir]
                sample_rows.append(dict(**base,sample_id=s,reference=ref,particles=vals[0],holes=vals[1],mismatch_sum=vals[2],
                    mismatch_density=vals[2]/len(active),charge_mismatch_to_reference=vals[3],
                    slab_charge_imbalance=qs[s,-1]-len(active)/2,exterior_charge_imbalance=qe[s,-1]-len(outside)/2,
                    saved_endpoint_chern=c[s,-1],all_y_R4_chern=radius_mean[s,list(radii).index(4)],
                    cycle40_chern=c[s,40],late21_40_chern=c[s,21:41].mean(),late21_40_abs_total_charge=np.abs(q[s,21:41]-nx*ny).mean()))
            error=abs(radius_mean[:,list(radii).index(4)]-1)
            for name,x in [('absolute_slab_charge',abs(qs[:,-1]-len(active)/2)),
                           ('slab_reference_mismatch',a['mismatch'][:,ir,2]/len(active)),
                           ('core_reference_mismatch',maps[:,columns['core2'],:].mean(axis=(1,2))/2),
                           ('core_correlation_longest_distance',a['corr_y'][:,REGIONS.index('core2'),-1])]:
                rho=float(spearmanr(x,error).statistic) if np.ptp(x)>0 and np.ptp(error)>0 else None
                association_rows.append(dict(**base,reference=ref,predictor=name,response='absolute_R4_chern_error',spearman_rho=rho))
        late_budget=[float(v[21:41].mean()) for v in budget]
        case_summaries.append(dict(**base,interfaces=[left,right],late_variance_budget=late_budget,
            saved_endpoint_chern_mean=float(cm[-1]),saved_endpoint_chern_sem=float(cs[-1]),
            endpoint_R4_chern_mean=float(radius_mean[:,list(radii).index(4)].mean()),
            endpoint_R4_mean_absolute_error=float(abs(radius_mean[:,list(radii).index(4)]-1).mean()),
            endpoint_R4_quantiles=np.quantile(radius_mean[:,list(radii).index(4)],[.05,.5,.95]).tolist(),
            radius_means=radius_mean.mean(axis=0).tolist(),radii=radii.tolist(),
            reference_mismatch=[dict(reference=item['key'],particles_mean=float(a['mismatch'][:,i,0].mean()),
                holes_mean=float(a['mismatch'][:,i,1].mean()),
                core_density=float(a['mismatch_maps'][:,i,columns['core2'],:].mean()/2),
                interface_density=float(a['mismatch_maps'][:,i,edgecols,:].mean()/2)) for i,item in enumerate(meta)],
            correlations_y_mean={name:a['corr_y'][:,i].mean(axis=0).tolist() for i,name in enumerate(REGIONS)},
            max_endpoint_validation_residual=float(a['diagnostics'].max()),
            references=meta))
    for name,rows in [('charge_cycles',charge_rows),('charge_sizes',size_rows),('variance_budget',budget_rows),
                     ('chern_cycles',chern_rows),('chern_radii',radius_rows),('correlations',corr_rows),
                     ('mismatch_regions',mismatch_rows),('trajectory_comparisons',sample_rows),
                     ('trajectory_associations',association_rows)]:save_csv(tables/(name+'.csv'),rows)
    # Preserve full per-trajectory/all-y arrays in the resumable batch products;
    # the following figures are views of those tables, not replacement datasets.
    common=('S=100 independent trajectories per case; direct sample SEM where shaded or barred. '
        'Centers/time windows are averaged within each trajectory first. Hard-wall, nshell=1, '
        'alpha1=1, alpha2=30, perfect correction, raster-y slab-only updates, periodic boundaries, '
        'pure half-filled random initialization followed by Born-conditioned exterior preparation. '
        'Square endpoints are cycle40; rectangular endpoints are cycles40,60,80. '
        'No fit or quantization acceptance gate is imposed. ')
    for cohort in ('square','rectangle'):
        cases=sorted([(nx,ny,a) for (co,nx,ny),a in groups.items() if co==cohort])
        fig,axes=plt.subplots(3,3,figsize=(7.05,7.2),layout='constrained',sharex='col')
        for col,region in enumerate(('total','slab','exterior')):
            for (nx,ny,a),color,mark,ls in zip(cases,COLORS,MARKERS,STYLES):
                delta,modes=a['charge_regions'][region];f=delta/modes;t=a['cycles'];label=f'{nx}×{ny}'
                plot_line(axes[0,col],t,f,color,label,ls=ls)
                plot_line(axes[1,col],t,abs(f),color,label,ls=ls)
                axes[2,col].plot(t,f.var(axis=0,ddof=1),color=color,ls=ls,label=label)
            axes[0,col].set_title(region);axes[2,col].set_xlabel('cycle')
        for row,label in enumerate(('mean signed filling deviation','mean absolute filling deviation','sample filling variance')):axes[row,0].set_ylabel(label)
        axes[0,0].legend(fontsize=7)
        finish(fig,axes,figures/f'{cohort}_01_charge_cycles',common+'Regional half filling uses half the region orbital count. Variance has no SEM shading.')

        fig,axes=plt.subplots(2,3,figsize=(7.05,4.6),layout='constrained')
        sizes=[nx if cohort=='square' else ny for nx,ny,_ in cases]
        for r,normalized in enumerate((False,True)):
            fields=(('signed_filling_mean','signed_filling_sem'),('absolute_filling_mean','absolute_filling_sem'),('mean_cyclewise_filling_variance',None)) if normalized else (
                ('signed_charge_mean','signed_charge_sem'),('absolute_charge_mean','absolute_charge_sem'),('mean_cyclewise_charge_variance',None))
            for c,(field,error) in enumerate(fields):
                for region,color,mark,ls in zip(('total','slab','exterior'),COLORS,MARKERS,STYLES):
                    rows=[next(row for row in size_rows if row['cohort']==cohort and row['nx']==nx and row['ny']==ny and row['region']==region and row['window']=='cycles21_40') for nx,ny,_ in cases]
                    axes[r,c].errorbar(sizes,[v[field] for v in rows],yerr=[v[error] for v in rows] if error else None,
                        color=color,marker=mark,ls=ls,mfc='white',ms=3,capsize=2,label=region)
                axes[r,c].set_xlabel('L' if cohort=='square' else r'$N_y$')
                axes[r,c].set_title(('signed mean','mean absolute','mean cyclewise variance')[c])
            axes[r,0].set_ylabel('filling deviation' if normalized else 'charge deviation')
        axes[0,0].legend(fontsize=7)
        finish(fig,axes,figures/f'{cohort}_02_charge_sizes',common+'Cycles21–40 within each trajectory; variance is instead computed across trajectories at each cycle then averaged over that window. Lines guide the eye.')

        fig,axes=plt.subplots(1,3,figsize=(7.05,2.55),layout='constrained')
        for ax,(nx,ny,a) in zip(axes,cases):
            for vals,label,color,ls in zip(variance_budget(a['q_slab'],a['q_exterior'][:,None]),
                    ('total','slab','exterior','2 covariance'),('k',*COLORS),('-','--',':','-.')):
                ax.plot(a['cycles'],vals,label=label,color=color,ls=ls,lw=1)
            ax.set_title(f'{nx}×{ny}');ax.set_xlabel('cycle');ax.axhline(0,color='.6',lw=.5)
        axes[0].set_ylabel('charge variance contribution');axes[-1].legend(fontsize=7)
        finish(fig,axes,figures/f'{cohort}_03_variance_budget',common+'Var(total)=Var(slab)+Var(exterior)+2Cov(slab,exterior), verified numerically. Signed covariance contribution is retained.')

        fig,axes=plt.subplots(2,3,figsize=(7.05,4.8),layout='constrained')
        for col,((nx,ny,a),color) in enumerate(zip(cases,COLORS)):
            t=a['cycles'];v=a['center_average'];axes[0,col].plot(t,v.T,color=color,alpha=.05,lw=.5)
            plot_line(axes[0,col],t,v,color,'mean ± SEM');axes[0,col].axhline(1,color='.4',ls='--',lw=.7)
            plot_line(axes[1,col],t,abs(v-1),color,r'$\langle|C-1|\rangle$',log=True)
            axes[1,col].plot(t,np.maximum(abs(v.mean(axis=0)-1),np.finfo(float).tiny),color='.2',ls='--',label=r'$|\langle C\rangle-1|$')
            axes[0,col].set_title(f'{nx}×{ny}');axes[1,col].set_xlabel('cycle');axes[1,col].legend(fontsize=7)
        axes[0,0].set_ylabel('trajectory Chern number');axes[1,0].set_ylabel('deviation from unity')
        finish(fig,axes,figures/f'{cohort}_04_chern_trajectories',common+'Each faint curve is one trajectory after averaging its ten saved centers. Lower panels distinguish mean absolute error from cancellation in the mean.')

        fig,axes=plt.subplots(1,3,figsize=(7.05,2.5),layout='constrained')
        for ax,(nx,ny,a),color in zip(axes,cases,COLORS):
            v=a['chern_radius'][:,list(a['radii']).index(4)].mean(axis=1)
            ax.hist(v,bins=15,color=color,alpha=.7,edgecolor='white');ax.axvline(1,color='k',ls='--',lw=.8)
            ax.set_title(f'{nx}×{ny}, cycle {a["cycles"][-1]}');ax.set_xlabel(r'endpoint $C_G$, $R=4$')
        axes[0].set_ylabel('trajectory count')
        finish(fig,axes,figures/f'{cohort}_05_chern_distributions',common+'R=4, every transverse center averaged within each trajectory; histogram contains100 entries per panel. Endpoint times differ in the rectangular cohort.')

        fig,axes=plt.subplots(2,2,figsize=(7.05,5),layout='constrained')
        for (nx,ny,a),color,marker in zip(cases,COLORS,MARKERS):
            radii=a['radii'];rho=radii/geometry(nx,ny)[5];v=a['chern_radius'].mean(axis=2);label=f'{nx}×{ny}'
            plot_line(axes[0,0],rho,v,color,label,marker)
            mean,sem=stats(v);delta=abs(mean-1)
            lower=np.where(delta<=sem,np.nan,delta-sem)
            axes[0,1].plot(rho,delta,color=color,marker=marker,label=label,ms=3,mfc='white',lw=1)
            axes[0,1].fill_between(rho,lower,delta+sem,color=color,alpha=.13,lw=0)
            plot_line(axes[1,0],rho,abs(v-1),color,label,marker,log=True)
            plot_line(axes[1,1],rho,a['chern_radius'].var(axis=2,ddof=1),color,label,marker,log=True)
            for ref,refdata in references[cohort,nx,ny].items():
                ls='--' if ref.startswith('nsh1') else ':'
                axes[0,0].plot(rho,refdata['chern'].mean(axis=1),color=color,ls=ls,lw=.7,alpha=.8)
            ir=list(radii).index(int(.2*nx))
            axes[0,0].plot(rho[ir],v.mean(axis=0)[ir],marker='*',ms=8,color=color)
        axes[0,0].axhline(1,color='.5',ls='--');axes[0,1].set_yscale('log')
        for ax,label in zip(axes.ravel(),('mean Chern','absolute mean deviation','mean absolute deviation','center-to-center variance')):
            ax.set_xlabel(r'$R/d_{\rm wall}$');ax.set_ylabel(label);ax.axvline(1,color='.5',ls='--',lw=.7)
        axes[0,0].legend(fontsize=7)
        finish(fig,axes,figures/f'{cohort}_06_radius_stability',common+'Solid curves use trajectory endpoints; dashed/dotted reference curves in panel(a) are nsh1/dense. Stars mark original radii. All tested disks are strictly contained. Raw R is retained in tables.')

        fig,axes=plt.subplots(2,3,figsize=(7.05,4.8),layout='constrained')
        for col,(nx,ny,a) in enumerate(cases):
            r=np.arange(1,ny//2+1)
            for row in range(2):
                for name,color,ls in zip(('core2','left_wall','right_wall','exterior'),('k',*COLORS),('-','--',':','-.')):
                    label='exterior (numerical zero)' if name=='exterior' else name
                    plot_line(axes[row,col],r,a['corr_y'][:,REGIONS.index(name)],color,label,ls=ls,log=True,display_floor=1e-28)
                for ref,v in references[cohort,nx,ny].items():
                    curve=v['corr_y'][REGIONS.index('core2')]
                    axes[row,col].plot(r,np.where(curve>1e-28,curve,np.nan),color='.5',lw=.7,ls='--' if ref.startswith('nsh1') else ':')
                if row:axes[row,col].set_xscale('log')
                axes[row,col].set_xlabel(r'$r_y$')
            axes[0,col].set_title(f'{nx}×{ny}, cycle {a["cycles"][-1]}')
        axes[0,0].set_ylabel('squared correlation');axes[1,0].set_ylabel('squared correlation');axes[0,0].legend(fontsize=6)
        finish(fig,axes,figures/f'{cohort}_07_regional_correlations',common+'Orbital-summed squared projector entries, spatial averages formed within trajectories. Gray dashed/dotted curves are nsh1/dense core references. Squared correlations <=1e-28 are omitted only for logarithmic display, not replaced by a physical floor. This hides the ~1e-31 floating-point residual of the verified exterior product state; all raw values remain in tables and NPZs.')

        fig,axes=plt.subplots(2,3,figsize=(7.05,4.8),layout='constrained')
        for col,(nx,ny,a) in enumerate(cases):
            for b,color in zip((1,2,3),COLORS):
                plot_line(axes[0,col],np.arange(1,ny//2+1),a['corr_y'][:,REGIONS.index(f'core{b}')],color,f'buffer {b}',log=True,display_floor=1e-28)
                values=a['corr_x'][:,b-1];mask=np.isfinite(values).all(axis=0)
                plot_line(axes[1,col],np.arange(1,nx)[mask],values[:,mask],color,f'buffer {b}',log=True,display_floor=1e-28)
            axes[0,col].set_title(f'{nx}×{ny}');axes[0,col].set_xlabel(r'$r_y$');axes[1,col].set_xlabel(r'$r_x$')
        axes[0,0].legend(fontsize=7);axes[0,0].set_ylabel('core squared correlation');axes[1,0].set_ylabel('core squared correlation')
        finish(fig,axes,figures/f'{cohort}_08_core_sensitivity',common+'Core excludes columns within one, two, or three spacings of either interface. Transverse pairs remain inside that core; no artificial periodicity across the slab is introduced. Squared correlations <=1e-28 are omitted only on log displays, with raw values preserved.')

        all_refs=sorted(set(ref for nx,ny,a in cases for ref in a['reference_names']))
        fig,axes=plt.subplots(len(all_refs),3,figsize=(7.05,max(2.8,2.45*len(all_refs))),layout='constrained',squeeze=False)
        for col,(nx,ny,a) in enumerate(cases):
            for row,ref in enumerate(all_refs):
                ax=axes[row,col]
                if ref not in a['reference_names']:ax.set_axis_off();continue
                ir=list(a['reference_names']).index(ref)
                m=a['mismatch_maps'][:,ir].mean(axis=0)/2
                im=ax.imshow(m.T,origin='lower',cmap='magma',vmin=0,interpolation='nearest',aspect='equal')
                for wall in geometry(nx,ny)[:2]:ax.axvline(wall,color='cyan',ls='--',lw=.6)
                ax.set_title(f'{nx}×{ny}, {ref}');ax.set_xlabel('x');ax.set_ylabel('y');fig.colorbar(im,ax=ax,shrink=.8)
        finish(fig,axes,figures/f'{cohort}_09_mismatch_maps',common+'Per-orbital reference mismatch: diagonal of (Gamma_slab-P0)^2, summed over orbitals then divided by two. Maps are computed per trajectory then averaged; exterior is excluded, not assigned zero mismatch. Color scales are displayed per panel.')

        fig,axes=plt.subplots(len(all_refs),2,figsize=(7.05,2.35*len(all_refs)),layout='constrained',squeeze=False)
        for ir,ref in enumerate(all_refs):
            for region,color,marker,ls in zip(('core2','interfaces','slab'),COLORS,MARKERS,STYLES):
                selected=[next((row for row in mismatch_rows if row['cohort']==cohort and row['nx']==nx and row['ny']==ny and row['reference']==ref and row['region']==region),None) for nx,ny,_ in cases]
                present=[(size,row) for size,row in zip(sizes,selected) if row is not None]
                if present:axes[ir,0].errorbar([v[0] for v in present],[v[1]['density_mean'] for v in present],yerr=[v[1]['density_sem'] for v in present],
                    color=color,ls=ls,marker=marker,mfc='white',ms=3,label=region)
            for (nx,ny,a),color,marker in zip(cases,COLORS,MARKERS):
                if ref not in a['reference_names']:continue
                k=list(a['reference_names']).index(ref)
                v=a['chern_radius'][:,list(a['radii']).index(4)].mean(axis=1)
                axes[ir,1].scatter(a['mismatch'][:,k,2]/len(geometry(nx,ny)[2]),abs(v-1),s=9,alpha=.5,color=color,marker=marker,label=f'{nx}×{ny}')
            axes[ir,0].set(xlabel='L' if cohort=='square' else r'$N_y$',ylabel='mismatch per orbital',title=ref)
            axes[ir,1].set(xlabel='slab mismatch per orbital',ylabel=r'$|C_G(R=4)-1|$',yscale='log',title=ref)
            axes[ir,0].legend(fontsize=7);axes[ir,1].legend(fontsize=7)
        finish(fig,axes,figures/f'{cohort}_10_mismatch_comparison',common+'Reference mismatch is not equated to topological failure. Scatter points are individual trajectories at their saved endpoint times. All reference selections are explicitly labeled.')
        fig,axes=plt.subplots(1,3,figsize=(7.05,2.5),layout='constrained')
        for ax,(nx,ny,a),color in zip(axes,cases,COLORS):
            v=a['chern_radius'][:,list(a['radii']).index(4)].mean(axis=1)
            charge=abs(a['q_slab'][:,-1]-len(geometry(nx,ny)[2])/2)
            ax.scatter(charge,abs(v-1),color=color,alpha=.5,s=9)
            ax.set(xlabel=r'$|Q_{\rm slab}-N_{\rm uc,slab}|$',ylabel=r'$|C_G(R=4)-1|$',yscale='log',title=f'{nx}×{ny}')
        finish(fig,axes,figures/f'{cohort}_11_charge_chern_comparison',common+'One point per trajectory; slab-only charge imbalance versus endpoint Chern deviation at fixed R4. No state averaging or charge selection is performed.')
    findings=['SLAB TOPOLOGY AND CHARGE: COMPLETE OFFLINE STUDY','',
        'Both independent cohorts are analyzed separately: 600 trajectories,38 input batches. No dynamics were rerun.',
        'Charge histories and common-window comparisons use cycles21–40. Square endpoint frames are cycle40; rectangular frames are cycles40,60,80.',
        'The frozen exterior is verified as a decoupled occupation product state before deriving slab charge histories.',
        'The following charge variance budget uses unnormalized charges and averages the per-cycle sample variance over cycles21–40. Covariance is not omitted.','']
    for s in case_summaries:
        vt,vs,ve,cov=s['late_variance_budget']
        findings.append(f"{s['cohort']} {s['nx']}x{s['ny']}: Var(total)={vt:.6g}; Var(slab)={vs:.6g}; Var(exterior)={ve:.6g}; 2Cov={cov:.6g}. Endpoint saved-center Chern={s['saved_endpoint_chern_mean']:.9f} +/- {s['saved_endpoint_chern_sem']:.3g} SEM; all-y R4={s['endpoint_R4_chern_mean']:.9f}.")
        findings.append(f"  R4 trajectory 5/50/95 percentiles: {s['endpoint_R4_quantiles']}; mean absolute error={s['endpoint_R4_mean_absolute_error']:.6g}. Radius means (R={s['radii']}): {s['radius_means']}.")
        for ref in s['reference_mismatch']:
            findings.append(f"  {ref['reference']}: expected particles={ref['particles_mean']:.6g}, holes={ref['holes_mean']:.6g}; per-orbital mismatch core={ref['core_density']:.6g}, interface strips={ref['interface_density']:.6g}.")
    findings.extend(['','INTERPRETATION AND LIMITS',
        'Inspect the regional budget rather than interpreting total-charge variance as bulk excitation density. The exterior is frozen random preparation, not a ground-state target.',
        'A mean close to unity does not establish concentration: trajectory distributions, mean absolute deviations, and radius curves are supplied separately.',
        'The equilibrium references use active-slab restrictions of OW modes built on the original geometry. They are not assumed to be the circuit steady state.',
        'Nonzero occupation mismatch can reflect a topologically harmless deformation. Degenerate cutoff cases are reported as below/above sensitivity projectors, not a unique half-filled state.',
        'Core correlations and buffer sensitivity diagnose locality over the available finite distances. Neither exponential localization nor a mobility gap is proven by these finite-size data.',
        'No asymptotic scaling exponent is claimed from three sizes. The two 20x20 ensembles remain independent. Rectangular endpoint comparisons are not time-matched across Ny.',
        'The study supplies evidence concerning sampled trajectories, not a theorem for arbitrary monitored records.'])
    (out/'findings.txt').write_text('\n'.join(findings)+'\n')
    products={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.csv','.pdf','.png','.txt')}
    write_json(out/'summary.json',dict(status='complete',trajectories=600,verified_batches=38,cases=case_summaries,
        analysis_inputs=inputs,products=products,report_source=digest(Path(__file__)),statistics='trajectory-first; sample SEM ddof=1; no bootstrap'))
    print('[report] complete: '+str(out/'findings.txt'),flush=True)
