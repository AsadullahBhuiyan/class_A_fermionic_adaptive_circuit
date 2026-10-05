"""Fixed-size, cycle-dependent gap diagnostics. No simulation or bootstrap."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

HERE=Path(__file__).resolve().parent
BUNDLES=HERE.parents[1]/'final_production_new_designs'
FULL=BUNDLES/'21_hard_wall_full_measurement_purification/gpu_data/hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1'
SLAB=BUNDLES/'13_maxmix_manybody_lyapunov_4ny'
CAP=1e-9


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def gaps(nu, cycles, caps=None):
    nu=np.asarray(nu,dtype=float);cycles=np.asarray(cycles)
    if nu.ndim!=3 or nu.shape[1]!=len(cycles) or np.any(cycles<=0):
        raise ValueError('Need sample/time/mode spectra at positive saved cycles')
    if not np.isfinite(nu).all() or nu.min() < -CAP or nu.max() > 1+CAP:
        raise ValueError('Invalid occupations')
    expected=(nu<=CAP)|(nu>=1-CAP)
    if caps is not None:
        np.testing.assert_array_equal(caps,expected)
    cost=np.full(nu.shape,np.inf)
    good=~expected
    cost[good]=abs(np.log1p(-nu[good])-np.log(nu[good]))
    raw=cost.min(-1)
    if not np.isfinite(raw).all():
        raise ValueError('A trajectory has no resolved mixed modes; do not silently censor it')
    return raw,raw/(2*cycles[None,:])


def mean_sem(values):
    a=np.asarray(values,dtype=float)
    return a.mean(0),a.std(0,ddof=1)/np.sqrt(len(a))


def paired_change(values,cycles,start,stop):
    """Mean change and sampling SEM retain the within-trajectory time pairing."""
    i=int(np.flatnonzero(cycles==start)[0]);j=int(np.flatnonzero(cycles==stop)[0])
    x,y=values[:,i],values[:,j]
    delta,sem=mean_sem(y-x)
    xm,ym=x.mean(),y.mean()
    # Delta-method SEM of ratio of ensemble means, not mean of per-sample ratios.
    influence=(y-ym)/xm - ym*(x-xm)/xm**2
    return dict(start_cycle=int(start),stop_cycle=int(stop),start_mean=float(xm),
                stop_mean=float(ym),change=float(delta),change_sem=float(sem),
                percent_change=float(100*(ym/xm-1)),
                percent_change_sem=float(100*influence.std(ddof=1)/np.sqrt(len(x))))


def verify_pair(path):
    receipt_path=path.with_suffix('.complete.json')
    receipt=json.loads(receipt_path.read_text())
    digest=sha(path)
    assert receipt['result_filename']==path.name
    assert receipt['result_bytes']==path.stat().st_size
    assert receipt['result_sha256']==digest
    records=[dict(path=str(p),bytes=p.stat().st_size,sha256=h)
             for p,h in ((path,digest),(receipt_path,sha(receipt_path)))]
    return receipt,records


def load_full():
    paths=sorted((FULL/'results/hard/alpha1_1/Ny030').glob('*.npz'))
    assert len(paths)==20
    ids=[];raws=[];rates=[];inputs=[]
    source_sets=[]
    for path in tqdm(paths,desc='Verify full-measurement Ny30',unit='shard'):
        receipt,records=verify_pair(path);inputs.extend(records)
        with np.load(path,allow_pickle=False) as z:
            config=json.loads(str(z['configuration_json']))
            assert hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()==receipt['configuration_hash']==str(z['configuration_hash'])
            source=json.loads(str(z['source_hashes_json']));assert source==receipt['source_hashes']
            if source not in source_sets:source_sets.append(source)
            assert int(z['Nx'])==20 and int(z['Ny'])==30 and float(z['alpha_1'])==1
            assert not bool(z['meas_slab_only']) and bool(z['dw_truncation'])
            assert config['nshell']==1 and config['alpha_2']==30
            assert config['init_mode']=='maxmix' and config['perfect_correction']
            assert config['sequence']=='raster_y' and config['dtype']=='complex128'
            assert str(z['canonical_dynamics_entry_point'])=='classA_U1FGTN_gpu.run_markov_circuit'
            assert not config.get('covariance_spectral_clip',False)
            np.testing.assert_array_equal(z['sample_indices'],receipt['sample_indices'])
            np.testing.assert_array_equal(z['cycles'],np.arange(61))
            cycle=z['cycles'][1:]
            nu=z['occupation_spectrum'][:,1:]
            assert nu.shape==(5,60,1200)
            raw,rate=gaps(nu,cycle)
            np.testing.assert_allclose(rate[:,-1],z['slow_mode_abs_rate'],atol=1e-10)
            ids.extend(z['sample_indices'].tolist());raws.append(raw);rates.append(rate)
    np.testing.assert_array_equal(ids,np.arange(100))
    return {30:dict(cycles=cycle,raw=np.concatenate(raws),rate=np.concatenate(rates))},inputs,source_sets


def load_slab():
    spec=importlib.util.spec_from_file_location('slab13_identity',SLAB/'analyze_campaign.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    validator.verify_manifest()
    paths=sorted(validator.DATA_ROOT.rglob('*.npz'));assert len(paths)==140
    groups={};inputs=[]
    for path in tqdm(paths,desc='Verify slab-only spectra',unit='shard'):
        receipt,records=verify_pair(path);inputs.extend(records)
        validator.validate_completion(receipt,path)
        with np.load(path,allow_pickle=False) as z:
            ny=int(z['Ny'])
            assert int(z['Nx'])==20 and str(z['configuration_hash'])==validator.CONFIGURATION_HASH
            np.testing.assert_array_equal(z['sample_indices'],receipt['sample_indices'])
            np.testing.assert_array_equal(z['spectrum_cycles'],validator.expected_spectrum_cycles(ny))
            assert z['spectrum_seen'].all()
            cycle=z['spectrum_cycles'][1:]
            raw,rate=gaps(z['occupations'][:,1:],cycle,z['cap_mask'][:,1:])
            np.testing.assert_allclose(raw,z['soft_mode_flip_costs'][:,1:].min(-1),atol=5e-12)
            row=groups.setdefault(ny,dict(cycles=cycle,ids=[],raw=[],rate=[]))
            np.testing.assert_array_equal(cycle,row['cycles'])
            row['ids'].extend(z['sample_indices'].tolist())
            row['raw'].append(raw);row['rate'].append(rate)
    assert sorted(groups)==list(validator.NY_VALUES)
    for row in groups.values():
        np.testing.assert_array_equal(row.pop('ids'),np.arange(100))
        row['raw']=np.concatenate(row['raw']);row['rate']=np.concatenate(row['rate'])
    return groups,inputs,validator.SOURCE_HASHES


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif','DejaVu Sans'],
        'mathtext.fontset':'cm','font.size':8,'axes.labelsize':9,'legend.fontsize':7,
        'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,
        'axes.linewidth':.8,'pdf.fonttype':42})


def plot(groups,stem,protocol):
    style()
    fig,axes=plt.subplots(2,1,figsize=(3.375,4.5),layout='constrained',sharex=True)
    colors=['#d62728','#2ca02c','#1f77b4','#ff7f0e','#9467bd','#222222','#17becf']
    markers=['^','s','o','v','D','P','>']
    for index,(ny,row) in enumerate(sorted(groups.items())):
        color=colors[index] if len(groups)>1 else '#1f77b4'
        marker=markers[index] if len(groups)>1 else 'o'
        linestyle=[':', '--', '-'][index%3] if len(groups)>1 else '-'
        for ax,key in zip(axes,('rate','raw')):
            mean,sem=mean_sem(row[key])
            ax.fill_between(row['cycles'],mean-sem,mean+sem,color=color,alpha=.12,lw=0)
            ax.plot(row['cycles'],mean,color=color,marker=marker,ms=3,mfc='white',
                    markevery=max(1,len(mean)//9),lw=1,ls=linestyle,label=rf'$N_y={ny}$')
    for index,ax in enumerate(axes):
        ax.text(-.15,1.02,'('+chr(97+index)+')',transform=ax.transAxes,va='bottom')
        ax.axvline(40,color='.6',ls='--',lw=.8,zorder=0)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel(r'$\langle\Delta(t)\rangle$')
    axes[1].set_ylabel(r'$\langle g_{\rm mod}(t)\rangle$')
    axes[1].set_xlabel(r'physical cycle $t$')
    axes[0].legend(frameon=False,ncol=2 if len(groups)>1 else 1,
                   loc='upper right' if len(groups)>1 else 'upper left')
    axes[0].text(.98,.04,protocol+'\n'+r'$N_x=20$',ha='right',va='bottom',transform=axes[0].transAxes)
    for ext in ('pdf','png'):fig.savefig(HERE/(stem+'.'+ext),dpi=300)
    plt.close(fig)


def write_csv(name,rows):
    with (HERE/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    full,full_inputs,full_sources=load_full()
    slab,slab_inputs,slab_sources=load_slab()
    means=[];samples=[];changes=[];arrays={}
    for protocol,groups in [('full_measurement',full),('slab_only',slab)]:
        for ny,row in sorted(groups.items()):
            t=row['cycles'];raw=row['raw'];rate=row['rate']
            m,e=mean_sem(rate);gm,ge=mean_sem(raw)
            for i,cycle in enumerate(t):
                means.append(dict(protocol=protocol,Nx=20,Ny=ny,cycle=int(cycle),samples=100,
                                  mean_gap=float(m[i]),gap_sem=float(e[i]),
                                  mean_modular_gap=float(gm[i]),modular_gap_sem=float(ge[i])))
                for sample in range(100):
                    samples.append(dict(protocol=protocol,Ny=ny,cycle=int(cycle),sample=sample,
                                        gap=float(rate[sample,i]),modular_gap=float(raw[sample,i])))
            intervals=[(20,40),(40,60)] if protocol=='full_measurement' else [(40,4*ny),(2*ny,4*ny),(3*ny,4*ny)]
            for start,stop in dict.fromkeys(intervals):
                changes.append(dict(protocol=protocol,Ny=ny,**paired_change(rate,t,start,stop)))
            for key,values in row.items():arrays[f'{protocol}_Ny{ny}_{key}']=values
    write_csv('cycle_gap_summary.csv',means)
    write_csv('sample_cycle_gaps.csv',samples)
    write_csv('paired_time_changes.csv',changes)
    np.savez_compressed(HERE/'sample_cycle_gaps.npz',**arrays)
    plot(full,'full_measurement_ny30_gap_convergence','full measurement')
    plot(slab,'slab_only_fixed_size_gap_convergence','slab-only measurement')
    manifest=dict(schema='purification_fixed_size_cycle_gap_convergence_v1',
        gap_definition='minimum absolute modular energy per sample, divided by 2t, then mean',
        uncertainty='ordinary trajectory SEM; paired time differences, no bootstrap',
        cap_tolerance=CAP,full_sources=full_sources,slab_sources=slab_sources,
        full_measurement_sizes=[30],slab_only_sizes=sorted(slab),
        campaign28_data_used=False,fit_performed=False,
        inputs=full_inputs+slab_inputs,
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(SLAB/'analyze_campaign.py'):sha(SLAB/'analyze_campaign.py')},
        changes=changes,
        outputs={p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.csv','.npz','.pdf','.png')})
    (HERE/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(changes,indent=2))


if __name__=='__main__':main()
