"""CPU equilibrium size sweep; resumable independent ground-state realizations."""
from pathlib import Path
import os
CPU_RANGE=(8,39)
os.sched_setaffinity(0,range(CPU_RANGE[0],CPU_RANGE[1]+1))
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,contextlib,io,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.linalg import eigh,eigvalsh
from scipy.special import xlogy
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
threadpool_limits(1)
OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
sys.path.insert(0,str(ROOT/'src/fgtn'))
SIZES=[80,100,120,160];VARIANCES=[1,2,3,4,6,9,12,16,25];SEED=2026092705
CONFIG=dict(Nx=20,sizes=SIZES,variances=VARIANCES,samples=100,walls=[5,15],
 disorder_x_columns=[5,6,14,15],alpha_1=1,alpha_2=30,nshell=1,
 seed=SEED,seed_components=['root','Ny','variance_index','sample_id'],
 filling='globally lowest 20*Ny levels',disorder='iid Gaussian diagonal, independent y and orbital; mean zero; variance W^2',
 entropy='full entropy in nats; all widths 1..Ny/2; all Ny periodic origins',
 estimator='average origins within each realization then realizations',
 dynamics='none; canonical CPU OW parent and static ground-state diagonalization')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
SOURCES={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py',Path(__file__)]}
IDENTITY=hashlib.sha256(json.dumps({'configuration':CONFIG,'sources':SOURCES},sort_keys=True).encode()).hexdigest()
def herm(a):
 assert np.isfinite(a).all()
 assert np.max(np.abs(a-a.conj().T))<1e-10
 return (a+a.conj().T)/2
def entropy(ev):
 assert np.isfinite(ev).all() and ev.min()>=-1e-8 and ev.max()<=1+1e-8
 p=np.clip(ev,0,1)
 return float(np.sum(-xlogy(p,p)-xlogy(1-p,1-p)))
def profiles(c,q,ny,checkpoint=None,translation_invariant=False):
 vals=np.empty((ny//2,ny));lo=1.;hi=0.;traceerr=0.
 start_width=1
 if checkpoint is not None and checkpoint.exists() and checkpoint.with_suffix('.json').exists():
  r=json.loads(checkpoint.with_suffix('.json').read_text())
  if r['identity']==IDENTITY and r['sha256']==sha(checkpoint):
   with np.load(checkpoint) as z:
    n=int(z['completed_width']);vals[:n]=z['values'];start_width=n+1
    lo=float(z['lo']);hi=float(z['hi']);traceerr=float(z['trace_error'])
 if translation_invariant:
  shifted=np.roll(np.roll(c.reshape(ny,q,ny,q),1,axis=0),1,axis=2).reshape(c.shape)
  assert np.max(np.abs(c-shifted))<1e-10
 for a in range(start_width,ny//2+1):
  for y0 in range(1 if translation_invariant else ny):
   rows=(((np.arange(a)+y0)%ny)[:,None]*q+np.arange(q)).ravel()
   sub=herm(c[np.ix_(rows,rows)])
   ev=eigvalsh(sub,driver='evr',check_finite=False)
   lo=min(lo,float(ev.min()));hi=max(hi,float(ev.max()))
   traceerr=max(traceerr,abs(float(ev.sum()-np.trace(sub).real)))
   vals[a-1,y0]=entropy(ev)
  if translation_invariant:vals[a-1,:]=vals[a-1,0]
  if checkpoint is not None:
   tmp=checkpoint.with_suffix('.tmp.npz')
   np.savez_compressed(tmp,values=vals[:a],completed_width=a,lo=lo,hi=hi,trace_error=traceerr)
   tmp.replace(checkpoint)
   dump(checkpoint.with_suffix('.json'),dict(identity=IDENTITY,sha256=sha(checkpoint),completed_width=a))
 complement=float(np.max(np.abs(vals[-1,:ny//2]-vals[-1,ny//2:])))
 assert traceerr<1e-8 and complement<1e-7
 return vals,dict(occupation_min=lo,occupation_max=hi,trace_error=traceerr,complement_error=complement)
def prepare(ny):
 from classA_U1FGTN import classA_U1FGTN
 folder=OUT/f'Ny{ny:03d}';folder.mkdir(exist_ok=True);(folder/'realizations').mkdir(exist_ok=True)
 f=folder/'parent.npz';receipt=folder/'parent.json'
 if f.exists() and receipt.exists():
  r=json.loads(receipt.read_text())
  if r['identity']==IDENTITY and r['sha256']==sha(f) and r['bytes']==f.stat().st_size:return
 print('Constructing canonical CPU parent for Ny',ny,flush=True)
 with contextlib.redirect_stdout(io.StringIO()):
  model=classA_U1FGTN(Nx=20,Ny=ny,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
  model.construct_OW_projectors(nshell=1,DW=True,trial_orbitals='X',dw_truncation=True)
 assert list(model.DW_loc)==[5,15]
 hk=np.zeros((ny,40,40),complex)
 rows=np.array([0,17,201,40*ny-1]);direct=np.zeros((len(rows),40*ny),complex)
 for name,sign in [('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)]:
  wf=np.asarray(getattr(model,name)).reshape(40*ny,-1);direct+=sign*wf[rows]@wf.conj().T
  w=wf.reshape(ny,40,20,ny);wk=np.fft.fft(w,axis=0)/np.sqrt(ny);v=wk[:,:,:,0]
  hk+=sign*ny*np.einsum('kar,kbr->kab',v,v.conj())
 hd=np.fft.ifft(hk,axis=0);diff=(np.arange(ny)[:,None]-np.arange(ny)[None,:])%ny
 h0=herm(hd[diff].transpose(0,2,1,3).reshape(40*ny,40*ny))
 direct_error=float(np.max(np.abs(h0[rows]-direct)));assert direct_error<1e-10
 xx=np.arange(40*ny)//2%20;active=np.flatnonzero((xx>=5)&(xx<=15));ext=np.flatnonzero((xx<5)|(xx>15))
 mask=np.isin(xx,[5,6,14,15]);assert mask.sum()==8*ny
 assert np.max(np.abs(h0[np.ix_(active,ext)]))<1e-12
 ee,ue=eigh(herm(h0[np.ix_(ext,ext)]),driver='evd')
 ce=ue[:,:9*ny]@ue[:,:9*ny].conj().T
 ep,ed=profiles(ce,18,ny,translation_invariant=True);assert np.max(np.ptp(ep,axis=1))<1e-7
 le=2*eigvalsh(herm(ce[:9*ny,:9*ny]))-1
 tmp=f.with_suffix('.tmp.npz')
 np.savez_compressed(tmp,hamiltonian=h0,active=active,exterior=ext,disorder_mask=mask,
                     exterior_energies=ee,exterior_entropy=ep,exterior_half_spectrum=le)
 tmp.replace(f)
 dump(receipt,dict(identity=IDENTITY,Ny=ny,bytes=f.stat().st_size,sha256=sha(f),
                   canonical_gram_row_check=direct_error,exterior_diagnostics=ed))
def init(ny):
 global NY,FOLDER,H0,HA,ACTIVE,EXT,EE,EP,LE,MASK
 NY=ny;FOLDER=OUT/f'Ny{ny:03d}'
 with np.load(FOLDER/'parent.npz') as z:
  H0=z['hamiltonian'];ACTIVE=z['active'];EXT=z['exterior'];EE=z['exterior_energies']
  EP=z['exterior_entropy'];LE=z['exterior_half_spectrum'];MASK=z['disorder_mask']
 HA=herm(H0[np.ix_(ACTIVE,ACTIVE)])
def task(t):
 variance,sid=t;ny=NY
 seed=[SEED,ny,VARIANCES.index(variance) if variance else 99,sid]
 f=FOLDER/'realizations'/f'W2_{variance:02d}_sample_{sid:03d}.npz';receipt=f.with_suffix('.json')
 if f.exists() and receipt.exists():
  r=json.loads(receipt.read_text())
  if r['identity']==IDENTITY and r['Ny']==ny and r['variance']==variance and r['sample_id']==sid and r['seed_components']==seed and r['bytes']==f.stat().st_size and r['sha256']==sha(f):return str(f)
 potential=np.random.default_rng(np.random.SeedSequence(seed)).normal(0,np.sqrt(variance),40*ny)*MASK
 h=HA.copy();h[np.diag_indices_from(h)]-=potential[ACTIVE]
 ea,u=eigh(herm(h),driver='evd',check_finite=False)
 es=np.r_[ea,EE];order=np.argsort(es,kind='stable');occ=np.zeros(40*ny,bool);occ[order[:20*ny]]=True
 oa=occ[:22*ny];oe=occ[22*ny:]
 assert np.array_equal(np.flatnonzero(oe),np.arange(9*ny)),'Exterior occupation differs; do not reuse clean exterior'
 gap=float(np.sort(es)[20*ny]-np.sort(es)[20*ny-1]);assert gap>1e-12
 fa=u[:,oa];c=fa@fa.conj().T
 purity=float(np.max(np.abs(c@c-c)));assert purity<1e-10
 residual=float(np.linalg.norm(h@fa-fa*ea[oa])/max(1,np.linalg.norm(h)));assert residual<1e-10
 partial=f.with_suffix('.partial.npz')
 vals,diag=profiles(c,22,ny,checkpoint=partial);vals+=EP
 half=np.sort(np.r_[2*eigvalsh(herm(c[:11*ny,:11*ny]))-1,LE])
 assert abs(entropy((half+1)/2)-vals[-1,0])<1e-7
 diag.update(purity_error=purity,eigen_residual=residual,half_filling_gap=gap,
             active_rank=int(oa.sum()),exterior_rank=int(oe.sum()))
 if variance==0:
  diag['clean_origin_spread']=float(np.max(np.ptp(vals,axis=1)));assert diag['clean_origin_spread']<1e-7
 if sid==0 and variance in [0,25]:
  ef,uf=eigh(herm(H0-np.diag(potential)),driver='evd')
  ff=uf[:,:20*ny];cf=ff@ff.conj().T;errors=[]
  for a,y0 in [(5,ny-3),(ny//2,7)]:
   rows=(((np.arange(a)+y0)%ny)[:,None]*40+np.arange(40)).ravel()
   errors.append(abs(entropy(eigvalsh(herm(cf[np.ix_(rows,rows)])))-vals[a-1,y0]))
  diag['full_space_entropy_error']=max(errors);assert max(errors)<1e-7
  diag['full_space_energy_error']=float(np.max(np.abs(ef-np.sort(es))));assert diag['full_space_energy_error']<1e-10
 tmp=f.with_suffix('.tmp.npz')
 np.savez_compressed(tmp,entropy_by_origin=vals,diagonal_potential=potential,hamiltonian_energies=np.sort(es),
                     half_centered_eigenvalues=half,Ny=ny,variance=variance,sample_id=sid,seed_components=seed,
                     widths=np.arange(1,ny//2+1),origins=np.arange(ny))
 tmp.replace(f)
 dump(receipt,dict(identity=IDENTITY,Ny=ny,variance=variance,sample_id=sid,seed_components=seed,
                   filename=f.name,bytes=f.stat().st_size,sha256=sha(f),diagnostics=diag))
 for checkpoint_file in [partial,partial.with_suffix('.json')]:
  checkpoint_file.unlink(missing_ok=True)
 return str(f)
def run(ny,workers,pilot):
 prepare(ny);init(ny);cleanpath=task((0,0))
 tasks=[(v,s) for v in VARIANCES for s in range(100)]
 if pilot:
  task((25,0));print('Pilot passed Ny',ny,flush=True);return
 start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(ny,)) as pool:
  pending=[pool.submit(task,t) for t in tasks]
  files=[f.result() for f in tqdm(as_completed(pending),total=len(tasks),desc=f'Ny={ny} entropy profiles',unit='state')]
 data=np.empty((9,100,ny//2,ny));diagnostics=[]
 for f in files:
  r=json.loads(Path(f).with_suffix('.json').read_text());assert r['sha256']==sha(Path(f))
  with np.load(f) as z:data[VARIANCES.index(int(z['variance'])),int(z['sample_id'])]=z['entropy_by_origin']
  diagnostics.append(r['diagnostics'])
 with np.load(cleanpath) as z:clean=z['entropy_by_origin']
 f=FOLDER/'entropy_profiles.npz'
 np.savez_compressed(f,entropy_by_origin=data,clean_entropy_by_origin=clean,variances=VARIANCES,
                     widths=np.arange(1,ny//2+1),origins=np.arange(ny),sample_ids=np.arange(100),Ny=ny)
 checks={k:max(d[k] for d in diagnostics if k in d) for k in diagnostics[0]}
 dump(FOLDER/'acquisition_complete.json',dict(status='complete',identity=IDENTITY,Ny=ny,realizations=900,
      clean_states=1,elapsed_seconds=time.perf_counter()-start,cache_sha256=sha(f),bytes=f.stat().st_size,
      diagnostic_maxima=checks,full_space_checks=[d['full_space_entropy_error'] for d in diagnostics if 'full_space_entropy_error' in d]))
 print('Completed Ny',ny,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--sizes',nargs='+',type=int,default=SIZES)
 p.add_argument('--workers',type=int,default=32);p.add_argument('--pilot',action='store_true');args=p.parse_args()
 assert set(args.sizes)<=set(SIZES)
 dump(OUT/'input_provenance.json',dict(configuration=CONFIG,identity=IDENTITY,sources=SOURCES,CPU_RANGE=CPU_RANGE))
 print('Configuration:',json.dumps(CONFIG),flush=True);print('CPU range',CPU_RANGE,'workers',args.workers,flush=True)
 for ny in args.sizes:run(ny,args.workers,args.pilot)
 print('ACQUISITION COMPLETE',flush=True)
if __name__=='__main__':main()
