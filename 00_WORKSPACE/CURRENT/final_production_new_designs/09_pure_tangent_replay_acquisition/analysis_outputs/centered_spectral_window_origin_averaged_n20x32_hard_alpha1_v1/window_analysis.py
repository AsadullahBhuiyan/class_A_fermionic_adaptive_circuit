"""CPU analysis of saved pure-state frames; no circuit simulation."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent
CAMPAIGN=OUT.parent.parent
DATA=CAMPAIGN/'gpu_data/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1/hard/Ny032/alpha1_1'
EXPECTED_CONFIG='d1a7c212d8f2c06b0f774af14272a2c71b9d0482240f0c4c43be75a9cd494be4'
WIDTHS=np.arange(1,17)
ORIGINS=np.arange(32)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def one_sample(args):
    sid,frame,rank=args
    rank=int(rank);assert 0<=rank<=frame.shape[1]
    frame=frame[:,:rank];assert np.isfinite(frame).all()
    orth=float(np.max(abs(frame.conj().T@frame-np.eye(rank))))
    assert orth<1e-8
    q=2*(frame@frame.conj().T)-np.eye(1280)
    assert np.isfinite(q).all()
    herm=float(np.max(abs(q-q.conj().T)));assert herm<1e-10
    sample={int(a):np.empty((32,40*a)) for a in WIDTHS}
    max_trace=0.
    for origin in ORIGINS:
        idx=(((origin+np.arange(16))%32)[:,None]*40+np.arange(40)).ravel()
        half=q[np.ix_(idx,idx)]
        for a in WIDTHS:
            n=40*a;qa=half[:n,:n]
            ev=eigvalsh((qa+qa.conj().T)/2,check_finite=False,driver='evr')
            assert ev.shape==(n,) and np.isfinite(ev).all()
            err=float(abs(ev.sum()-np.trace(qa).real));assert err<1e-8
            assert ev.min()>=-1-1e-8 and ev.max()<=1+1e-8
            max_trace=max(max_trace,err);sample[int(a)][origin]=ev
    cross=0.
    if sid==0:
        fa=frame[:640]
        independent=2*eigvalsh(fa@fa.conj().T)-1
        cross=float(np.max(abs(independent-sample[16][0])))
        assert cross<1e-10
    return int(sid),sample,orth,herm,max_trace,cross

def compute_spectra():
    receipts=sorted(DATA.glob('*.complete.json'))
    assert len(receipts)==4
    provenance=[];ids=[]
    for p in tqdm(receipts,desc='Validate endpoint batches',unit='batch'):
        r=json.loads(p.read_text());f=p.with_name(r['result_filename'])
        for k,v in dict(status='complete',Nx=20,Ny=32,cycles=64,construction='hard',alpha_1=1.,alpha_2=30.,nshell=1,configuration_sha256=EXPECTED_CONFIG).items():assert r[k]==v,(k,r[k])
        assert f.stat().st_size==r['result_bytes']
        assert sha(f)==r['result_sha256']
        ids.extend(r['case_sample_indices']);provenance.append(dict(path=str(f),receipt=r))
    assert sorted(ids)==list(range(100))
    identity=dict(Nx=20,Ny=32,widths=WIDTHS.tolist(),origins_y=ORIGINS.tolist(),cycles=64,samples=100,configuration_sha256=EXPECTED_CONFIG,estimator='eigvalsh(2 F_A F_A^dagger - identity)',inputs=[p['receipt']['result_sha256'] for p in provenance])
    cache=OUT/'subsystem_spectra.npz';diagnostic_file=OUT/'spectra_diagnostics.json'
    if cache.exists() and diagnostic_file.exists():
        diag=json.loads(diagnostic_file.read_text())
        assert diag['identity']==identity and sha(cache)==diag['cache_sha256']
        with np.load(cache) as z: spectra={int(a):z[f'eigenvalues_Ay{a:02d}'] for a in WIDTHS}
        print('Loaded verified spectrum cache; all four inputs revalidated.')
        return spectra,diag
    spectra={int(a):np.empty((100,32,40*a)) for a in WIDTHS}
    diag=dict(identity=identity,inputs=provenance,max_hermiticity_error=0.,max_trace_error=0.,max_frame_orthonormality_error=0.,raw_min=1.,raw_max=-1.)
    for item in provenance:
        with np.load(item['path']) as z:
            r=item['receipt']
            for k,v in dict(Nx=20,Ny=32,cycles_total=64,construction='hard',alpha_1=1.,alpha_2=30.,nshell=1,configuration_sha256=EXPECTED_CONFIG,canonical_entry_point=r['canonical_entry_point']).items():assert z[k].item()==v,k
            assert np.array_equal(z['case_sample_indices'],r['case_sample_indices'])
            assert np.array_equal(z['global_sample_indices'],r['global_sample_indices'])
            assert json.loads(z['source_hashes_json'].item())==r['source_hashes']
            assert int(z['batch_seed'])==r['seed']
            frames=z['final_frame'];ranks=z['final_ranks'];sample_ids=z['case_sample_indices']
        # Independent CPU eigensolves; cap each BLAS pool to one thread.
        workers=min(8,len(__import__('os').sched_getaffinity(0)))
        with threadpool_limits(limits=1),ProcessPoolExecutor(max_workers=workers) as pool:
            for sid,sample,orth,herm,err,cross in tqdm(pool.map(one_sample,((sid,frames[j],ranks[j]) for j,sid in enumerate(sample_ids))),total=len(sample_ids),desc=f"Batch {r['batch_index']}: 32 origins x 16 widths",unit='sample'):
                for a in WIDTHS:spectra[int(a)][sid]=sample[int(a)]
                diag['max_frame_orthonormality_error']=max(diag['max_frame_orthonormality_error'],orth)
                diag['max_hermiticity_error']=max(diag['max_hermiticity_error'],herm)
                diag['max_trace_error']=max(diag['max_trace_error'],err)
                diag['raw_min']=min(diag['raw_min'],min(float(v.min()) for v in sample.values()))
                diag['raw_max']=max(diag['raw_max'],max(float(v.max()) for v in sample.values()))
                if sid==0:diag['sample0_alternate_eigenvalue_error']=cross
        del frames
    previous=OUT.parent/'half_system_centered_spectrum_n20_sizes_hard_alpha1_v1/centered_spectra_Ny032.npz'
    with np.load(previous) as z:
        assert np.array_equal(z['sample_ids'],np.arange(100))
        diag['half_system_previous_cache_max_error']=float(np.max(abs(spectra[16][:,0]-z['eigenvalues'])))
        assert diag['half_system_previous_cache_max_error']<1e-10
    diag['previous_cache']=dict(path=str(previous),sha256=sha(previous))
    np.savez_compressed(cache,sample_ids=np.arange(100),widths=WIDTHS,origins=ORIGINS,**{f'eigenvalues_Ay{a:02d}':s for a,s in spectra.items()})
    diag['cache_sha256']=sha(cache)
    diagnostic_file.write_text(json.dumps(diag,indent=2)+'\n')
    return spectra,diag

def analyze(spectra,L=.5,mixed_tol=1e-8,fit_min=4):
    assert 0<L<1-mixed_tol
    a=np.array(sorted(spectra))
    # Arrays below: trajectory, width, origin. Normalize pooled density at
    # each origin exactly as the previous figure, then average origins equally.
    n=np.stack([(abs(spectra[v])<=L).sum(-1) for v in a],axis=1)
    m=np.stack([(abs(spectra[v])<1-mixed_tol).sum(-1) for v in a],axis=1)
    assert (m>0).all() and (n<=m).all()
    n_avg=n.mean(-1);m_avg=m.mean(-1);mean=n_avg.mean(0)
    fraction_by_origin=n.sum(0)/m.sum(0)
    fraction=fraction_by_origin.mean(-1)
    influences={'mode_count':n_avg-mean,
                'mixed_normalized_integral':((n-fraction_by_origin*m)/m.mean(0)).mean(-1),
                'all_modes_integral':(n_avg-mean)/(40*a)}
    values={'mode_count':mean,'mixed_normalized_integral':fraction,'all_modes_integral':mean/(40*a)}
    chord=32/np.pi*np.sin(np.pi*a/32);x=np.log(chord)
    table=pd.DataFrame(dict(Ay=a,modes_in_subsystem=40*a,chord=chord,log_chord=x,
                           central_count_total=n.sum((0,2)),mixed_count_total=m.sum((0,2)),
                           mean_mixed_count=m_avg.mean(0),
                           globally_pooled_integral=n.sum((0,2))/m.sum((0,2)),
                           mean_individually_normalized_integral=(n/m).mean((0,2))))
    fits={};covariances={}
    for name,y in values.items():
        cov=np.cov(influences[name],rowvar=False,ddof=1)/100
        covariances[name]=cov;table[name]=y;table[name+'_sem']=np.sqrt(np.diag(cov))
        fits[name]={}
        for lo in sorted(set([2,4,8,int(fit_min)])):
            mask=a>=lo;design=np.column_stack([np.ones(mask.sum()),x[mask]])
            operator=np.linalg.pinv(design);beta=operator@y[mask]
            bcov=operator@cov[np.ix_(mask,mask)]@operator.T
            residual=y[mask]-design@beta
            fits[name][str(lo)]=dict(fit_widths=a[mask].tolist(),intercept=float(beta[0]),slope=float(beta[1]),slope_sem=float(np.sqrt(bcov[1,1])),R_squared=float(1-np.sum(residual**2)/np.sum((y[mask]-y[mask].mean())**2)),residual_rms=float(np.sqrt(np.mean(residual**2))),max_abs_residual=float(np.max(abs(residual))))
    table.to_csv(OUT/'window_integrals.csv',index=False)
    pd.DataFrame([dict(sample_id=sid,Ay=int(v),origin_y=int(origin),central_count=int(n[sid,j,origin]),mixed_count=int(m[sid,j,origin])) for sid in range(100) for j,v in enumerate(a) for origin in ORIGINS]).to_csv(OUT/'sample_window_counts.csv',index=False)
    np.savez_compressed(OUT/'window_statistics.npz',widths=a,origins=ORIGINS,sample_ids=np.arange(100),central_counts=n,mixed_counts=m,**{k+'_covariance':v for k,v in covariances.items()})
    results=dict(L=L,mixed_tolerance=mixed_tol,default_fit_min_width=fit_min,fit_model='intercept + slope * log[(32/pi) sin(pi Ay/32)]',estimator='at each origin normalize pooled trajectory counts; average the 32 normalized densities equally. Raw counts: average origins within each trajectory, then trajectories',uncertainty='trajectory SEM; delta method for ratio of pooled counts; fits propagate full cross-width trajectory covariance',fits=fits,half_system=table.iloc[-1].to_dict(),normalization_checks=dict(max_complement_probability_error=float(np.max(abs(n.sum(0)/m.sum(0)+(m-n).sum(0)/m.sum(0)-1))),spectra_per_width=3200,independent_trajectories=100,origins=32))
    (OUT/'window_diagnostics.json').write_text(json.dumps(results,indent=2)+'\n')
    return table,results
