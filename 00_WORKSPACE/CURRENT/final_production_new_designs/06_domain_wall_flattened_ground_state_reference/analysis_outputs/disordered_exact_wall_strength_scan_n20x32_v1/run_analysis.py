"""Static disordered ground states; canonical CPU clean parent; no circuit dynamics."""
from pathlib import Path
import os
CPU_RANGE=(8,15)
os.sched_setaffinity(0,range(CPU_RANGE[0],CPU_RANGE[1]+1))
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='8'
import sys,json,hashlib,time,contextlib,io
import numpy as np
from scipy.linalg import eigh,eigvalsh
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
threadpool_limits(8)
OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
BASE=ROOT/'00_WORKSPACE/CURRENT/final_production_new_designs'
sys.path.insert(0,str(ROOT/'src/fgtn'))
from classA_U1FGTN import classA_U1FGTN
NX,NY=20,32
VARIANCES=[1.,2.,3.,4.,6.,9.]
SAMPLES=100
SEED=2026092703
L=.99
DISORDER_RADIUS=0
DISORDER_X=[5,15]
CONFIG={'Nx':NX,'Ny':NY,'walls':[5,15],'alpha_1':1,'alpha_2':30,'nshell':1,
 'variances':VARIANCES,'samples_per_variance':SAMPLES,'seed':SEED,'L':L,
 'disorder':'iid Gaussian diagonal potentials only on x=5,15; independent for the two orbitals; zero elsewhere',
 'disorder_radius':DISORDER_RADIUS,'disorder_x_columns':DISORDER_X,
 'seed_matching':'independent draws across strengths and realizations',
 'mean':0.,'sample_mean_subtracted':False,'hamiltonian':'h_flat-diag(v)',
 'filling':'lowest Nx*Ny levels globally','subsystem':'all x, y=0..15, both orbitals',
 'origins':[0],'CPU_RANGE':CPU_RANGE,'dynamics':'none; static ground-state diagonalization'}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
SOURCES={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py',Path(__file__)]}
IDENTITY=hashlib.sha256(json.dumps({'configuration':CONFIG,'sources':SOURCES},sort_keys=True).encode()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def herm(a):
 assert np.isfinite(a).all()
 err=float(np.max(np.abs(a-a.conj().T)));assert err<1e-10,err
 return (a+a.conj().T)/2
def solve(h):
 e,u=eigh(h,driver='evd')
 residual=float(np.linalg.norm(h@u-u*e)/max(1,np.linalg.norm(h)))
 orth=float(np.max(np.abs(u.conj().T@u-np.eye(len(e)))))
 assert residual<1e-10 and orth<1e-10
 return e,u,residual,orth
def reduced(u,selected,rows):
 f=u[np.ix_(rows,np.flatnonzero(selected))]
 c=f@f.conj().T;g=2*c-np.eye(len(rows))
 he=float(np.max(np.abs(g-g.conj().T)));assert he<1e-10
 lam=eigvalsh(herm(g),driver='evr')
 assert np.isfinite(lam).all() and np.max(np.abs(lam))<=1+1e-8
 assert abs(lam.sum()-np.trace(g).real)<1e-8
 return lam
print('Configuration:',json.dumps(CONFIG),flush=True)
with contextlib.redirect_stdout(io.StringIO()):
 model=classA_U1FGTN(Nx=NX,Ny=NY,DW=True,nshell=1,alpha_1=1,alpha_2=30,
                    trial_orbitals='X',dw_truncation=True)
 model.construct_OW_projectors(nshell=1,DW=True,trial_orbitals='X',dw_truncation=True)
assert list(model.DW_loc)==[5,15]
# Reproduce the existing momentum-space clean parent, then Fourier-transform to real space.
q=2*NX;hk=np.zeros((NY,q,q),complex)
for name,sign in [('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)]:
 w=np.asarray(getattr(model,name)).reshape(NY,q,NX,NY)
 wk=np.fft.fft(w,axis=0)/np.sqrt(NY)
 v=wk[:,:,:,0];hk+=sign*NY*np.einsum('kar,kbr->kab',v,v.conj())
hd=np.fft.ifft(hk,axis=0)
diff=(np.arange(NY)[:,None]-np.arange(NY)[None,:])%NY
h0=herm(hd[diff].transpose(0,2,1,3).reshape(1280,1280))
del model,w,wk
index=np.arange(1280);x=index//2%NX;y=index//(2*NX)
disorder_mask=np.minimum(abs(x-5),abs(x-15))<=DISORDER_RADIUS
assert np.array_equal(np.unique(x[disorder_mask]),DISORDER_X)
assert disorder_mask.sum()==128
active=np.flatnonzero((x>=5)&(x<=15));exterior=np.setdiff1d(index,active)
cross=float(np.max(np.abs(h0[np.ix_(active,exterior)])))
assert cross<1e-12,cross
ha=herm(h0[np.ix_(active,active)]);he=herm(h0[np.ix_(exterior,exterior)])
arows=np.flatnonzero(y[active]<16);erows=np.flatnonzero(y[exterior]<16)
assert len(arows)+len(erows)==640
def state_spectrum(potential):
 h=ha.copy()
 assert potential.shape==(1280,) and np.isfinite(potential).all()
 assert np.all(potential[~disorder_mask]==0)
 h[np.diag_indices_from(h)]-=potential[active]
 exterior_h=he.copy()
 exterior_h[np.diag_indices_from(exterior_h)]-=potential[exterior]
 ea,ua,res,orth=solve(herm(h))
 ee,ue,er,eo=solve(herm(exterior_h))
 res=max(res,er);orth=max(orth,eo)
 all_e=np.concatenate([ea,ee]);order=np.argsort(all_e,kind='stable')
 occ=np.zeros(1280,bool);occ[order[:640]]=True
 oa,oe=occ[:len(ea)],occ[len(ea):]
 la=reduced(ua,oa,arows);le=reduced(ue,oe,erows)
 lam=np.sort(np.r_[la,le])
 es=np.sort(all_e)
 diag={'eigen_residual':res,'orthogonality':orth,'active_rank':int(oa.sum()),
       'exterior_rank':int(oe.sum()),'half_filling_gap':float(es[640]-es[639]),
       'fermi_midpoint':float((es[640]+es[639])/2),
       'raw_lambda_min':float(lam.min()),'raw_lambda_max':float(lam.max())}
 assert oa.sum()+oe.sum()==640 and diag['half_filling_gap']>1e-12
 return lam,es,diag,h,ua,oa,oe
clean,clean_e,clean_diag,_,_,_,_=state_spectrum(np.zeros(1280))
reference=BASE/'06_domain_wall_flattened_ground_state_reference/analysis_outputs/equilibrium_flattened_spectral_densities_n20_sizes_hard_alpha1_v1/equilibrium_spectra_Ny032.npz'
refmeta=json.loads(reference.with_suffix('.json').read_text())
assert sha(reference)==refmeta['cache_sha256']
with np.load(reference) as z:
 clean_error=float(np.max(np.abs(clean-z['centered_eigenvalues'])))
 energy_error=float(np.max(np.abs(clean_e-np.sort(z['energies']))))
assert clean_error<1e-5 and energy_error<1e-10,(clean_error,energy_error)
np.savez_compressed(OUT/'clean_reference.npz',centered_eigenvalues=clean,energies=clean_e,
 hamiltonian=h0,active_indices=active,exterior_indices=exterior,disorder_mask=disorder_mask)
dump(OUT/'input_provenance.json',{'configuration':CONFIG,'identity':IDENTITY,'sources':SOURCES,
 'clean_reference_path':str(reference),'clean_reference_sha256':sha(reference),
 'clean_spectral_max_error':clean_error,'clean_energy_max_error':energy_error,
 'active_exterior_coupling_max':cross,'clean_diagnostics':clean_diag})
print(f'Clean benchmark matched: max spectral error {clean_error:.3g}; exact block separation {cross:.3g}',flush=True)
shards=OUT/'realizations';shards.mkdir(exist_ok=True)
start=time.perf_counter()
for vi,variance in enumerate(VARIANCES):
 for s in tqdm(range(SAMPLES),desc=f'Disorder variance {variance:g}',unit='state'):
  f=shards/f'variance_{vi:02}_sample_{s:03}.npz';receipt=f.with_suffix('.json')
  if f.exists() and receipt.exists():
   d=json.loads(receipt.read_text())
   if d['identity']==IDENTITY and d['filename']==f.name and d['sample_id']==s and d['variance']==variance and d['bytes']==f.stat().st_size and d['sha256']==sha(f):continue
  seed=np.random.SeedSequence([SEED,vi,s])
  potential=np.random.default_rng(seed).normal(0,np.sqrt(variance),1280)*disorder_mask
  assert np.count_nonzero(potential)==128
  lam,es,diag,h,ua,oa,oe=state_spectrum(potential)
  # First realization at each strength: cross-check the exact block reduction against
  # a full-space diagonalization and direct 640-mode restricted covariance.
  if s==0:
   full=h0.copy()
   full[np.diag_indices_from(full)]-=potential
   ef,uf=eigh(full,driver='evd')
   full_lam=reduced(uf,np.arange(1280)<640,np.arange(640))
   diag['full_space_spectrum_crosscheck']=float(np.max(np.abs(full_lam-lam)))
   diag['full_space_energy_crosscheck']=float(np.max(np.abs(ef-es)))
   assert diag['full_space_spectrum_crosscheck']<1e-7
   assert diag['full_space_energy_crosscheck']<1e-10
  temp=f.with_suffix('.tmp.npz')
  np.savez_compressed(temp,centered_eigenvalues=lam,hamiltonian_energies=es,
      diagonal_potential=potential,sample_id=s,variance=variance,seed_components=[SEED,vi,s])
  temp.replace(f)
  dump(receipt,{'identity':IDENTITY,'variance':variance,'sample_id':s,
       'seed_components':[SEED,vi,s],'filename':f.name,'bytes':f.stat().st_size,
       'sha256':sha(f),'diagnostics':diag})
 print(f'Completed variance {variance:g}: 100 states; elapsed {time.perf_counter()-start:.1f}s',flush=True)
dump(OUT/'acquisition_complete.json',{'status':'complete','identity':IDENTITY,
 'configuration':CONFIG,'realizations':len(VARIANCES)*SAMPLES,'elapsed_seconds':time.perf_counter()-start})
print('DONE',flush=True)
