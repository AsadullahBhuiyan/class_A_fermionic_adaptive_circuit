"""Square-system Hamiltonian spectra with exact-wall disorder; no circuit dynamics."""
from pathlib import Path
import os
CPU_RANGE=(40,55)
os.sched_setaffinity(0,range(CPU_RANGE[0],CPU_RANGE[1]+1))
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,time,contextlib,io,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.linalg import eigvalsh
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
threadpool_limits(1)
OUT=Path(__file__).resolve().parent
ROOT=next(p for p in OUT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
sys.path.insert(0,str(ROOT/'src/fgtn'))
SIZES=[12,16,20,24,32,40];SAMPLES=100;W2=9;SEED=2026092802
CONFIG=dict(sizes=SIZES,Nx_equals_Ny=True,variance=W2,samples=100,
 walls='N/4,3N/4; inclusive topological slab',disorder='iid zero-mean Gaussian diagonal on wall columns only; independent y and orbital',
 alpha_1=1,alpha_2=30,nshell=1,trial_orbitals='X',dw_truncation=True,
 seed=SEED,seed_components=['root','N','sample_id'],energy_zero='original Hamiltonian zero; no centering or rescaling',
 estimator='mean over realizations of min(abs(E)); no reflattening',
 size40_reference='reuse existing validated Nx40 Ny40 W2=9 ensemble, original seeds retained',
 dynamics='none; canonical CPU classA_U1FGTN OW parent, eigenvalues only')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
SOURCES={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py',Path(__file__)]}
IDENTITY=hashlib.sha256(json.dumps(dict(configuration=CONFIG,sources=SOURCES),sort_keys=True).encode()).hexdigest()
def herm(a):
 assert np.isfinite(a).all()
 assert np.max(np.abs(a-a.conj().T))<1e-10
 return (a+a.conj().T)/2
def prepare(n):
 from classA_U1FGTN import classA_U1FGTN
 folder=OUT/f'N{n:03d}';folder.mkdir(exist_ok=True);(folder/'realizations').mkdir(exist_ok=True)
 f=folder/'parent.npz';rp=f.with_suffix('.json')
 if f.exists() and rp.exists():
  r=json.loads(rp.read_text())
  if r['identity']==IDENTITY and r['bytes']==f.stat().st_size and r['sha256']==sha(f):return
 walls=[n//4,3*n//4];d=2*n
 print('Constructing canonical CPU square parent:',n,'walls',walls,flush=True)
 with contextlib.redirect_stdout(io.StringIO()):
  model=classA_U1FGTN(Nx=n,Ny=n,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
  model.construct_OW_projectors(nshell=1,DW=True,trial_orbitals='X',dw_truncation=True)
 assert list(model.DW_loc)==walls
 hk=np.zeros((n,d,d),complex)
 rows=np.unique([0,2*walls[0],2*walls[1]+1,d*(n//2)+n,d*(n-1)+2*walls[1]])
 direct=np.zeros((len(rows),d*n),complex)
 for name,sign in [('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)]:
  w=np.asarray(getattr(model,name)).reshape(d*n,-1);direct+=sign*w[rows]@w.conj().T
  wk=np.fft.fft(w.reshape(n,d,n,n),axis=0)/np.sqrt(n);v=wk[:,:,:,0]
  hk+=sign*n*np.einsum('kar,kbr->kab',v,v.conj())
 hd=np.fft.ifft(hk,axis=0);diff=(np.arange(n)[:,None]-np.arange(n)[None,:])%n
 h=herm(hd[diff].transpose(0,2,1,3).reshape(d*n,d*n))
 error=float(np.max(np.abs(h[rows]-direct)));assert error<1e-10
 xx=np.arange(d*n)//2%n
 active=np.flatnonzero((xx>=walls[0])&(xx<=walls[1]));ext=np.flatnonzero((xx<walls[0])|(xx>walls[1]))
 mask=np.isin(xx,walls);assert mask.sum()==4*n
 cross=float(np.max(np.abs(h[np.ix_(active,ext)])));assert cross<1e-12
 ee=eigvalsh(herm(h[np.ix_(ext,ext)]),driver='evd',check_finite=False)
 tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,hamiltonian=h,active=active,exterior=ext,disorder_mask=mask,exterior_energies=ee,walls=walls)
 tmp.replace(f);dump(rp,dict(identity=IDENTITY,N=n,bytes=f.stat().st_size,sha256=sha(f),canonical_gram_row_error=error,checked_rows=rows.tolist(),cross_block_max=cross))
def init(n):
 global N,FOLDER,H0,HA,ACTIVE,EE,MASK
 N=n;FOLDER=OUT/f'N{n:03d}'
 with np.load(FOLDER/'parent.npz') as z:
  H0=z['hamiltonian'];ACTIVE=z['active'];EE=z['exterior_energies'];MASK=z['disorder_mask']
 HA=herm(H0[np.ix_(ACTIVE,ACTIVE)])
def task(sid):
 n=N;seed=[SEED,n,sid];f=FOLDER/'realizations'/f'sample_{sid:03d}.npz';rp=f.with_suffix('.json')
 if f.exists() and rp.exists():
  r=json.loads(rp.read_text())
  if r['identity']==IDENTITY and r['sample_id']==sid and r['seed_components']==seed and r['bytes']==f.stat().st_size and r['sha256']==sha(f):return str(f)
 pot=np.random.default_rng(np.random.SeedSequence(seed)).normal(0,3,2*n*n)*MASK
 h=HA.copy();h[np.diag_indices_from(h)]-=pot[ACTIVE]
 ea=eigvalsh(herm(h),driver='evd',check_finite=False);e=np.sort(np.r_[ea,EE])
 assert len(e)==2*n*n and np.isfinite(e).all() and np.all(np.diff(e)>=0)
 trace_error=abs(float(e.sum()-(np.trace(H0).real-pot.sum())));assert trace_error<1e-8
 diag=dict(trace_error=trace_error,min_abs_eigenvalue=float(np.min(np.abs(e))),half_filling_gap=float(e[n*n]-e[n*n-1]))
 if sid==0:
  full=eigvalsh(herm(H0-np.diag(pot)),driver='evd',check_finite=False)
  diag['full_space_energy_error']=float(np.max(np.abs(e-full)));assert diag['full_space_energy_error']<1e-10
 tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,hamiltonian_energies=e,diagonal_potential=pot,Nx=n,Ny=n,variance=9,sample_id=sid,seed_components=seed)
 tmp.replace(f);dump(rp,dict(identity=IDENTITY,Nx=n,Ny=n,variance=9,sample_id=sid,seed_components=seed,filename=f.name,bytes=f.stat().st_size,sha256=sha(f),diagnostics=diag))
 return str(f)
def reuse40():
 source=OUT.parent/'disordered_exact_wall_Nx_scan_Ny040_W2_9_v1'
 provenance=json.loads((source/'input_provenance.json').read_text())
 assert provenance['configuration']['variance']==9 and provenance['configuration']['Ny']==40
 for rel,digest in SOURCES.items():
  if rel.startswith('src/'):assert provenance['sources'][rel]==digest
 fdir=OUT/'N040';fdir.mkdir(exist_ok=True);energies=[];inputs=[];diags=[]
 for sid in tqdm(range(100),desc='Verify saved 40x40 spectra',unit='state'):
  f=source/f'Nx040/realizations/sample_{sid:03d}.npz';r=json.loads(f.with_suffix('.json').read_text())
  assert r['identity']==provenance['identity'] and r['sample_id']==sid and r['Nx']==40 and r['Ny']==40 and r['variance']==9
  assert r['bytes']==f.stat().st_size and r['sha256']==sha(f)
  with np.load(f) as z:
   e=z['hamiltonian_energies'];pot=z['diagonal_potential']
   assert e.shape==(3200,) and np.isfinite(e).all() and np.all(np.diff(e)>=0)
   assert np.all(pot[~np.isin(np.arange(3200)//2%40,[10,30])]==0)
   energies.append(e)
  inputs.append(dict(path=str(f),sha256=r['sha256'],bytes=r['bytes'],sample_id=sid,seed_components=r['seed_components']))
  diags.append(r['diagnostics'])
 f=fdir/'hamiltonian_spectra.npz';np.savez_compressed(f,hamiltonian_energies=energies,sample_ids=np.arange(100),Nx=40,Ny=40,variance=9)
 dump(fdir/'acquisition_complete.json',dict(status='complete',identity=IDENTITY,N=40,samples=100,bytes=f.stat().st_size,sha256=sha(f),reused_inputs=inputs,source_provenance=provenance))
def run(n,workers):
 if n==40:reuse40();return
 prepare(n);init(n);start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(n,)) as pool:
  fs=[pool.submit(task,sid) for sid in range(100)]
  files=[f.result() for f in tqdm(as_completed(fs),total=100,desc=f'{n}x{n} spectra',unit='state')]
 es=np.empty((100,2*n*n));ids=[];diags=[]
 for path in files:
  f=Path(path);r=json.loads(f.with_suffix('.json').read_text());assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes']
  with np.load(f) as z:es[int(z['sample_id'])]=z['hamiltonian_energies'];ids.append(int(z['sample_id']))
  diags.append(r['diagnostics'])
 assert sorted(ids)==list(range(100))
 f=FOLDER/'hamiltonian_spectra.npz';np.savez_compressed(f,hamiltonian_energies=es,sample_ids=np.arange(100),Nx=n,Ny=n,variance=9)
 dump(FOLDER/'acquisition_complete.json',dict(status='complete',identity=IDENTITY,N=n,samples=100,bytes=f.stat().st_size,sha256=sha(f),
  elapsed_seconds=time.perf_counter()-start,max_trace_error=max(d['trace_error'] for d in diags),
  full_space_energy_errors=[d['full_space_energy_error'] for d in diags if 'full_space_energy_error' in d]))
 print('COMPLETED square size',n,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=16);a=p.parse_args()
 dump(OUT/'input_provenance.json',dict(configuration=CONFIG,identity=IDENTITY,sources=SOURCES,CPU_RANGE=CPU_RANGE))
 print('Configuration:',json.dumps(CONFIG),flush=True)
 reuse40()
 for n in SIZES[:-1]:run(n,a.workers)
 print('ACQUISITION COMPLETE',flush=True)
if __name__=='__main__':main()
