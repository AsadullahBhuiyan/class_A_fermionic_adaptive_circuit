"""Matched spectra, cut-origin controls, and saved postselection diagnostics; CPU only."""
from pathlib import Path
import os
available=sorted(os.sched_getaffinity(0))
CPU_RANGE=(available[0],available[min(7,len(available)-1)])
os.sched_setaffinity(0,range(CPU_RANGE[0],CPU_RANGE[1]+1))
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]=str(CPU_RANGE[1]-CPU_RANGE[0]+1)
import sys,json,hashlib,contextlib,io,csv,time
import numpy as np
from scipy.linalg import eigh,eigvalsh
from scipy.special import entr
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
threadpool_limits(CPU_RANGE[1]-CPU_RANGE[0]+1)
OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
BASE=ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs'
SAVED=BASE/'09_pure_tangent_replay_acquisition/analysis_outputs/half_system_centered_spectrum_n20_sizes_hard_alpha1_v1'
EQSAVED=BASE/'06_domain_wall_flattened_ground_state_reference/analysis_outputs/equilibrium_flattened_spectral_densities_n20_sizes_hard_alpha1_v1'
sys.path.insert(0,str(ROOT/'src/fgtn'))
from classA_U1FGTN import classA_U1FGTN

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)
def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,default=native,indent=2,allow_nan=False)+'\n')
def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def check(a,msg):
    if not a:raise ValueError(msg)
def hermitian(a):
    check(np.isfinite(a).all(),'nonfinite matrix')
    err=float(np.max(abs(a-a.conj().T)))
    check(err<1e-10,f'non-Hermitian input {err}')
    return (a+a.conj().T)/2

def metrics(lam):
    raw=np.asarray(lam)
    check(np.isfinite(raw).all() and abs(raw).max()<1+1e-8,'spectrum bounds')
    l=np.clip(raw,-1,1);n=(1+l)/2;v=n*(1-n)
    return dict(S1=(entr(n)+entr(1-n)).sum(axis=-1),V=v.sum(axis=-1),
                S2=-np.log(n*n+(1-n)**2).sum(axis=-1),K4=(v*(1-6*v)).sum(axis=-1),
                charge=n.sum(axis=-1),central=(abs(l)<.5).sum(axis=-1),
                interior=(abs(l)<.9).sum(axis=-1),mixed=(abs(l)<1-1e-8).sum(axis=-1))
def ms(a):
    a=np.atleast_1d(a);return dict(mean=float(a.mean()),sem=float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None)

def equilibrium(nx,ny,walls=None):
    with contextlib.redirect_stdout(io.StringIO()):
        model=classA_U1FGTN(Nx=nx,Ny=ny,DW=True,nshell=1,alpha_1=1,alpha_2=30,
                            trial_orbitals='X',dw_truncation=True,dw_interval=walls)
        model.construct_OW_projectors(nshell=1,DW=True,trial_orbitals='X',dw_truncation=True)
    q=2*nx;hk=np.zeros((ny,q,q),complex)
    phase=np.exp(-2j*np.pi*np.arange(ny)[:,None]*np.arange(ny)[None,:]/ny)
    terr=0.
    for name,sign in [('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)]:
        w=np.asarray(getattr(model,name)).reshape(ny,q,nx,ny)
        wk=np.fft.fft(w,axis=0)/np.sqrt(ny)
        terr=max(terr,float(abs(wk-wk[:,:,:,:1]*phase[:,None,None,:]).max()))
        v=wk[:,:,:,0];hk+=sign*ny*np.einsum('kar,kbr->kab',v,v.conj())
    check(terr<2e-10,'OW translation check')
    es=[];us=[]
    for h in hk:
        e,u=eigh(hermitian(h),driver='evd');es.append(e);us.append(u)
    es=np.array(es);us=np.array(us);occ=np.zeros(es.size,bool)
    order=np.argsort(es.ravel(),kind='stable');occ[order[:nx*ny]]=True;occ=occ.reshape(es.shape)
    pk=np.array([u[:,o]@u[:,o].conj().T for u,o in zip(us,occ)])
    d=np.fft.ifft(pk,axis=0);hd=np.fft.ifft(hk,axis=0)
    def dense(delta):
        ids=(np.arange(ny)[:,None]-np.arange(ny)[None,:])%ny
        return delta[ids].transpose(0,2,1,3).reshape(ny*q,ny*q)
    p=hermitian(dense(d));h=hermitian(dense(hd))
    check(np.max(abs(pk@pk-pk))<1e-10,'projector idempotence')
    diag={'walls':model.DW_loc,'translation_error':terr,'gap':np.diff(np.sort(es.ravel())[nx*ny-1:nx*ny+1])[0]}
    return p,h,es,us,diag

def reduced(p,nx,ny,y0=0,vectors=False):
    ids=(((np.arange(ny//2)+y0)%ny)[:,None]*2*nx+np.arange(2*nx)[None,:]).ravel()
    c=hermitian(p[np.ix_(ids,ids)])
    g=2*c-np.eye(len(ids))
    if vectors:
        l,u=eigh(g,driver='evd')
    else:l=eigvalsh(g,driver='evr')
    check(abs(l.sum()-np.trace(g))<1e-8,'reduced trace')
    metrics(l)
    if not vectors:return l
    n=np.clip((l+1)/2,0,1);w=abs(u)**2
    ent=(w@(entr(n)+entr(1-n))).reshape(ny//2,nx,2).sum(axis=(0,2))
    var=(w@(n*(1-n))).reshape(ny//2,nx,2).sum(axis=(0,2))
    central=w[:,abs(l)<.5].sum(axis=1).reshape(ny//2,nx,2).sum(axis=(0,2))
    check(abs(ent.sum()-metrics(l)['S1'])<1e-9,'contour closure')
    return l,ent,var,central

provenance=[]
def record(p,**extra):
    provenance.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p),**extra))


def cached_comparison():
    overlay=json.loads((SAVED/'overlay_diagnostics.json').read_text())
    rows=[];sens=[];raw={};summ={}
    for ny in [24,28,32]:
        sf=SAVED/f'centered_spectra_Ny{ny:03}.npz';ef=EQSAVED/f'equilibrium_spectra_Ny{ny:03}.npz'
        md=json.loads((EQSAVED/f'equilibrium_spectra_Ny{ny:03}.json').read_text())
        check(sha(sf)==overlay['sizes'][str(ny)]['spectra_sha256'],'stochastic cache checksum')
        check(sha(ef)==md['cache_sha256'],'equilibrium cache checksum')
        with np.load(sf) as z:
            st=z['eigenvalues'];check(np.array_equal(z['sample_ids'],np.arange(100)),'sample IDs')
            check(np.array_equal(z['subsystem_indices'],np.arange(20*ny)),'cut indices')
        eq=np.load(ef)['centered_eigenvalues'][None,:]
        record(sf);record(ef)
        meta=json.loads((SAVED/f'spectra_diagnostics_Ny{ny:03}.json').read_text())
        for r in meta['contract']['inputs']:
            f=ROOT/r['result'];receipt=ROOT/r['receipt'];d=json.loads(receipt.read_text())
            check(f.stat().st_size==r['bytes']==d['result_bytes'],'raw byte count')
            check(sha(receipt)==r['receipt_sha256'],'receipt hash')
            check(sha(f)==r['sha256']==d['result_sha256'],'raw checksum')
            check(d['status']=='complete' and d['configuration_sha256']==meta['contract']['configuration_sha256'],'completion identity')
            record(receipt);provenance.append(dict(path=r['result'],sha256=r['sha256'],bytes=r['bytes']))
        for label,a in [('stochastic',st),('equilibrium',eq)]:
            raw[f'{label}_{ny}']=a;m=metrics(a)
            for i in range(len(a)):rows.append(dict(Ny=ny,protocol=label,sample_id=i,**{k:float(v[i]) for k,v in m.items()}))
            summ[f'{label}_{ny}']={k:ms(v) for k,v in m.items()}
            for tol in [1e-4,1e-6,1e-8,1e-10]:
                for bins in [20,50,100,200]:
                    vals=a[abs(a)<1-tol];counts,edges=np.histogram(vals,np.linspace(-1,1,bins+1))
                    density=counts/(len(vals)*np.diff(edges))
                    check(counts.sum()==len(vals) and abs(np.dot(density,np.diff(edges))-1)<1e-12,'histogram normalization')
                    sens.append(dict(Ny=ny,protocol=label,tolerance=tol,bins=bins,retained=len(vals),area=float(np.dot(density,np.diff(edges))),empty_bins=int((counts==0).sum()),central_fraction=float((abs(vals)<.5).mean()),interior_fraction=float((abs(vals)<.9).mean())))
        print('Matched',ny,'S',summ[f'stochastic_{ny}']['S1'],summ[f'equilibrium_{ny}']['S1'],flush=True)
    write_csv('sample_spectral_metrics.csv',rows);write_csv('histogram_sensitivity.csv',sens)
    np.savez_compressed(OUT/'matched_spectra.npz',**raw);dump('matched_summary.json',summ)
    return raw


def spatial_control(raw):
    target=OUT/'spatial_and_origin_controls_all100.npz'
    if target.exists():
        print('Reusing spatial controls',flush=True);record(target);return
    nx=20;ny=32;n=nx*ny
    p,h,es,us,eqdiag=equilibrium(nx,ny)
    eql,eqent,eqvar,eqcentral=reduced(p,nx,ny,vectors=True)
    diff=float(abs(eql-raw['equilibrium_32'][0]).max());check(diff<1e-6,'current source equilibrium mismatch')
    eqdiag['cached_spectrum_max_error']=diff
    shape=(100,n);l0=np.empty(shape);l8=np.empty(shape)
    ent=np.empty((100,nx));var=ent.copy();central=ent.copy();ranks=np.empty(100,int)
    rank_list=[];scalar=[];meanp=np.zeros_like(p)
    origin_ids=np.arange(100)
    origin_spec=np.empty((len(origin_ids),16,n))
    energyocc=np.empty((100,ny,2*nx));crosschecks=[]
    ent8=np.empty_like(ent);var8=np.empty_like(var);central8=np.empty_like(central)
    topidx=np.where((np.arange(2*n)//2%nx>=5)&(np.arange(2*n)//2%nx<=15))[0]
    htop=h[np.ix_(topidx,topidx)];ptop=p[np.ix_(topidx,topidx)]
    meta=json.loads((SAVED/'spectra_diagnostics_Ny032.json').read_text())
    shift=((np.arange(2*n)//(2*nx)+1)%ny)*(2*nx)+np.arange(2*n)%(2*nx)
    sorted_energy=np.sort(es.ravel());eq_energy=sorted_energy[:n].sum()
    for batch in tqdm(meta['contract']['inputs'],desc='Raw endpoint batches'):
        with np.load(ROOT/batch['result']) as z:
            ids=z['case_sample_indices'];frames=z['final_frame'];rank=z['final_ranks']
            check(str(z['sequence'])=='raster_y','sequence');check(int(z['cycles_total'])==64,'cycle')
        for i,f,r in tqdm(zip(ids,frames,rank),total=len(ids),desc='Spectra and contours',leave=False):
            i=int(i);r=int(r);f=f[:,:r];ranks[i]=r
            check(np.isfinite(f).all(),'finite frame')
            ferr=float(abs(f.conj().T@f-np.eye(r)).max());check(ferr<1e-9,'occupied frame orthogonality')
            ps=hermitian(f@f.conj().T);meanp+=ps/100
            l0[i],ent[i],var[i],central[i]=reduced(ps,nx,ny,vectors=True)
            check(abs(l0[i]-raw['stochastic_32'][i]).max()<1e-10,'recomputed endpoint spectrum')
            l8[i],ent8[i],var8[i],central8[i]=reduced(ps,nx,ny,8,vectors=True)
            pst=ps[np.ix_(topidx,topidx)]
            fk=np.fft.fft(f.reshape(ny,2*nx,r),axis=0)/np.sqrt(ny)
            amplitudes=np.einsum('kji,kjr->kir',us.conj(),fk,optimize=True)
            energyocc[i]=(abs(amplitudes)**2).sum(axis=2)
            energy=float(np.sum(es*energyocc[i]))
            scalar.append(dict(sample_id=i,rank=r,frame_orthogonality=ferr,
                topological_charge=float(np.trace(pst).real),
                topological_excess_energy=float(np.einsum('ij,ji->',htop,pst-ptop).real),
                topological_projector_distance_squared=float(np.sum(abs(pst-ptop)**2)),
                excess_energy_half_filling=energy-eq_energy,
                excess_energy_same_rank=energy-sorted_energy[:r].sum(),
                projector_distance_squared=float(np.sum(abs(ps-p)**2)),
                translation_breaking_squared=float(np.sum(abs(ps-ps[np.ix_(shift,shift)])**2)),
                trace_error=float(abs(np.trace(ps)-r))))
            if i in origin_ids:
                j=int(np.where(origin_ids==i)[0][0])
                for origin in range(16):
                    origin_spec[j,origin]=l0[i] if origin==0 else l8[i] if origin==8 else reduced(ps,nx,ny,origin)
                # Complementary half has identical entropy; excess unit eigenvalues can differ when rank != n.
                lc=reduced(ps,nx,ny,16)
                err=abs(metrics(lc)['S1']-metrics(l0[i])['S1']);check(err<1e-8,'complement entropy')
                crosschecks.append(float(err))
    mean_l=reduced(meanp,nx,ny)
    mean_e=eigvalsh(hermitian(meanp),driver='evr')
    np.savez_compressed(target,equilibrium_eigenvalues=eql,equilibrium_entropy_x=eqent,
        equilibrium_variance_x=eqvar,equilibrium_central_weight_x=eqcentral,
        y0_spectra=l0,y8_spectra=l8,entropy_x=ent,variance_x=var,central_weight_x=central,ranks=ranks,
        entropy_x_y8=ent8,variance_x_y8=var8,central_weight_x_y8=central8,
        origin_sample_ids=origin_ids,origin_spectra=origin_spec,origin_values=np.arange(16),
        averaged_covariance_spectrum=mean_l,averaged_full_occupations=mean_e,
        energies_by_ky=es,energy_occupations=energyocc,mean_projector=meanp,equilibrium_projector=p)
    write_csv('endpoint_state_diagnostics.csv',scalar)
    dump('spatial_diagnostics.json',dict(equilibrium=eqdiag,complement_entropy_max_error=max(crosschecks),
        origin_sampling='all 100 independent trajectories; all 16 inequivalent half cuts',
        source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py']}))
    print('Spatial controls saved',flush=True)


def postselection_controls():
    root=ROOT/'00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data/covariance_protocol_histories/campaigns/multi_geometry_C10_dwtrunc1_init-default_nsh1_S10'
    rows=[];arrays={};diag=[]
    for summary in tqdm(sorted(root.rglob('run_summary.json')),desc='Saved protocol histories'):
        cfg=json.loads(summary.read_text());nx,ny=cfg['Nx'],cfg['Ny'];protocol=cfg['protocol']
        if ny==25:continue
        f=summary.parent/'batch_00000_history.npy';a=np.load(f,mmap_mode='r')
        check(list(a.shape)==cfg['shard_shape'] and f.stat().st_size==cfg['shard_file_size_bytes'],'legacy shard shape/bytes')
        check(cfg['dw_loc']==[5,11] and cfg['alpha_1']==1 and cfg['alpha_2']==30,'legacy protocol')
        record(f,quality='legacy: shape/byte validation and new checksum; no acquisition SHA-256 receipt')
        record(summary)
        for cycle in [1,5,10]:
            spectra=[]
            for i in range(len(a)):
                g=hermitian(np.asarray(a[i,cycle-1]));p=(g+np.eye(g.shape[0]))/2
                purity=float(np.sum(abs(p@p-p)**2));check(purity<1e-12,'legacy full purity')
                l=reduced(p,nx,ny);spectra.append(l)
                rows.append(dict(Nx=nx,Ny=ny,protocol=protocol,cycle=cycle,sample_id=i,**{k:float(v) for k,v in metrics(l).items()}))
            arrays[f'{protocol}_{ny}_cycle{cycle}']=np.array(spectra)
        diag.append(dict(Ny=ny,protocol=protocol,samples=len(a),initialization=cfg['init_mode'],cycles=cfg['cycles'],walls=cfg['dw_loc']))
    write_csv('legacy_protocol_metrics.csv',rows);np.savez_compressed(OUT/'legacy_protocol_spectra.npz',**arrays);dump('legacy_protocol_diagnostics.json',diag)
    root=ROOT/'00_WORKSPACE/COLAB/colab_partial_post-select/gpu_data/streaming_covariance_observables/campaigns/N20x40_nsh1_dwtrunc1_alpha2-30_init-default_S10_C40_partial-postselect-psweep'
    rows=[];curves={}
    for run in sorted((root/'runs').iterdir()):
        cfg=json.loads((run/'run_summary.json').read_text());f=run/'entropy_y0avg_vs_ay.npz'
        with np.load(f) as z:
            a=z['entropy_y0avg_vs_ay'];cycles=z['cycle_labels'];ay=z['ay_values'];config=json.loads(str(z['config_json']))
        check(a.shape[1:]==(40,21) and np.isfinite(a).all(),'postselection entropy shape')
        prob=cfg.get('postselect_probability',config.get('postselect_probability'))
        check(prob is not None,'postselect probability')
        curves[f'p_{prob}']=a
        for i in range(len(a)):
            for j,c in enumerate(cycles):
                rows.append(dict(p=prob,sample_id=i,cycle=int(c),S_half=float(a[i,j,-1])))
        record(f,quality='legacy streaming observables; no acquisition SHA-256 receipt');record(run/'run_summary.json')
    write_csv('partial_postselection_entropy.csv',rows);np.savez_compressed(OUT/'partial_postselection_entropy.npz',**curves)

if __name__=='__main__':
    start=time.time();print('Output',OUT,'CPUs',CPU_RANGE,flush=True)
    raw=cached_comparison();spatial_control(raw);postselection_controls()
    record(Path(__file__))
    dump('input_provenance.json',provenance)
    dump('run_status.json',dict(status='complete',elapsed_seconds=time.time()-start,cpus=CPU_RANGE,
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.npz','.csv','.json'] and p.name!='run_status.json'}))
    print('Complete',time.time()-start,flush=True)
