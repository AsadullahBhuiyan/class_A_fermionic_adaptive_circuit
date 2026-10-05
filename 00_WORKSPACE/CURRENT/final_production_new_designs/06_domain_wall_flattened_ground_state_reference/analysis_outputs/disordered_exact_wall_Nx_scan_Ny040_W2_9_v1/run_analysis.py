"""Canonical CPU ground-state entropy: transverse size scan, exact-wall disorder."""
from pathlib import Path
import os
CPU_RANGE=(40,55)
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
SIZES=[12,16,20,24,32,40];NY=40;VARIANCE=9;SAMPLES=100;SEED=2026092710
CONFIG=dict(Nx_values=SIZES,Ny=NY,variance=VARIANCE,samples=SAMPLES,
 walls='Nx/4,3*Nx/4 (Nx divisible by four); inclusive topological slab',
 disorder='iid zero-mean Gaussian diagonal; variance9; independent y and orbital; wall columns only',
 alpha_1=1,alpha_2=30,nshell=1,seed=SEED,seed_components=['root','Nx','Ny','sample_id'],
 filling='globally lowest Nx*Ny eigenvalues; no reflattening',
 entropy='full entropy in nats; all widths1..20; all40 periodic origins; all x and both orbitals',
 estimator='average origins within each realization, then disorder; OLS fit to mean entropy versus log chord',
 fits=['5<=Ay<=20','8<=Ay<=20'],c_effective='3*slope; total strip, not per-wall',
 Nx20_reference='reused completed exact-wall campaign, original seeds and receipts retained',
 dynamics='none; canonical CPU classA_U1FGTN OW parent and static ground-state diagonalization')
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

def prepare(nx):
 from classA_U1FGTN import classA_U1FGTN
 ny=NY;d=2*nx;walls=[nx//4,3*nx//4]
 folder=OUT/f'Nx{nx:03d}';folder.mkdir(exist_ok=True);(folder/'realizations').mkdir(exist_ok=True)
 f=folder/'parent.npz';rpath=f.with_suffix('.json')
 if f.exists() and rpath.exists():
  r=json.loads(rpath.read_text())
  if r['identity']==IDENTITY and r['bytes']==f.stat().st_size and r['sha256']==sha(f):return
 print('Constructing canonical CPU parent for Nx',nx,'walls',walls,flush=True)
 with contextlib.redirect_stdout(io.StringIO()):
  model=classA_U1FGTN(Nx=nx,Ny=ny,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
  model.construct_OW_projectors(nshell=1,DW=True,trial_orbitals='X',dw_truncation=True)
 assert list(model.DW_loc)==walls
 hk=np.zeros((ny,d,d),complex);rows=np.unique([0,d-1,d*ny//2,d*ny-1]);direct=np.zeros((len(rows),d*ny),complex)
 for name,sign in [('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)]:
  wf=np.asarray(getattr(model,name)).reshape(d*ny,-1);direct+=sign*wf[rows]@wf.conj().T
  wk=np.fft.fft(wf.reshape(ny,d,nx,ny),axis=0)/np.sqrt(ny);v=wk[:,:,:,0]
  hk+=sign*ny*np.einsum('kar,kbr->kab',v,v.conj())
 hd=np.fft.ifft(hk,axis=0);diff=(np.arange(ny)[:,None]-np.arange(ny)[None,:])%ny
 h0=herm(hd[diff].transpose(0,2,1,3).reshape(d*ny,d*ny))
 error=float(np.max(np.abs(h0[rows]-direct)));assert error<1e-10
 xx=np.arange(d*ny)//2%nx;active=np.flatnonzero((xx>=walls[0])&(xx<=walls[1]));ext=np.flatnonzero((xx<walls[0])|(xx>walls[1]))
 qa=len(active)//ny;qe=len(ext)//ny;mask=np.isin(xx,walls);assert mask.sum()==4*ny
 cross=float(np.max(np.abs(h0[np.ix_(active,ext)])));assert cross<1e-12
 ee,ue=eigh(herm(h0[np.ix_(ext,ext)]),driver='evd')
 ce=ue[:,:len(ext)//2]@ue[:,:len(ext)//2].conj().T
 ep,ed=profiles(ce,qe,ny,translation_invariant=True)
 tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,hamiltonian=h0,active=active,exterior=ext,disorder_mask=mask,
  exterior_energies=ee,exterior_entropy=ep,qa=qa,qe=qe,walls=walls)
 tmp.replace(f);dump(rpath,dict(identity=IDENTITY,Nx=nx,Ny=ny,bytes=f.stat().st_size,sha256=sha(f),
  canonical_gram_row_error=error,active_exterior_coupling_max=cross,exterior_diagnostics=ed))
def init(nx):
 global NX,FOLDER,H0,ACTIVE,EXT,EE,EP,MASK,HA,QA
 NX=nx;FOLDER=OUT/f'Nx{nx:03d}'
 with np.load(FOLDER/'parent.npz') as z:
  H0=z['hamiltonian'];ACTIVE=z['active'];EXT=z['exterior'];EE=z['exterior_energies'];EP=z['exterior_entropy'];MASK=z['disorder_mask'];QA=int(z['qa'])
 HA=herm(H0[np.ix_(ACTIVE,ACTIVE)])
def task(sid):
 nx=NX;ny=NY;seed=[SEED,nx,ny,sid]
 f=FOLDER/'realizations'/f'sample_{sid:03d}.npz';rp=f.with_suffix('.json')
 if f.exists() and rp.exists():
  r=json.loads(rp.read_text())
  if r['identity']==IDENTITY and r['sample_id']==sid and r['seed_components']==seed and r['bytes']==f.stat().st_size and r['sha256']==sha(f):return str(f)
 potential=np.random.default_rng(np.random.SeedSequence(seed)).normal(0,np.sqrt(VARIANCE),2*nx*ny)*MASK
 h=HA.copy();h[np.diag_indices_from(h)]-=potential[ACTIVE]
 ea,u=eigh(herm(h),driver='evd',check_finite=False)
 es=np.r_[ea,EE];order=np.argsort(es,kind='stable');occupied=np.zeros(len(es),bool);occupied[order[:nx*ny]]=True
 oa=occupied[:len(ACTIVE)];oe=occupied[len(ACTIVE):]
 assert np.array_equal(np.flatnonzero(oe),np.arange(len(EXT)//2)),'Exterior occupation changed; block cache not valid'
 sorted_es=np.sort(es);gap=float(sorted_es[nx*ny]-sorted_es[nx*ny-1]);assert gap>1e-12
 fa=u[:,oa];c=fa@fa.conj().T
 purity=float(np.max(np.abs(c@c-c)));residual=float(np.linalg.norm(h@fa-fa*ea[oa])/max(1,np.linalg.norm(h)))
 assert purity<1e-10 and residual<1e-10
 partial=f.with_suffix('.partial.npz');vals,diag=profiles(c,QA,ny,checkpoint=partial);vals+=EP
 diag.update(purity_error=purity,eigen_residual=residual,half_filling_gap=gap,active_rank=int(oa.sum()),exterior_rank=int(oe.sum()))
 if sid==0:
  ef,uf=eigh(herm(H0-np.diag(potential)),driver='evd');ff=uf[:,:nx*ny];cf=ff@ff.conj().T;errors=[]
  for a,y0 in [(5,ny-3),(ny//2,7)]:
   rows=(((np.arange(a)+y0)%ny)[:,None]*(2*nx)+np.arange(2*nx)).ravel()
   errors.append(abs(entropy(eigvalsh(herm(cf[np.ix_(rows,rows)])))-vals[a-1,y0]))
  diag['full_space_entropy_error']=max(errors);assert max(errors)<1e-7
  diag['full_space_energy_error']=float(np.max(np.abs(ef-sorted_es)));assert diag['full_space_energy_error']<1e-10
 tmp=f.with_suffix('.tmp.npz');np.savez_compressed(tmp,entropy_by_origin=vals,diagonal_potential=potential,
  hamiltonian_energies=sorted_es,Nx=nx,Ny=ny,variance=VARIANCE,sample_id=sid,seed_components=seed,widths=np.arange(1,21),origins=np.arange(ny))
 tmp.replace(f);dump(rp,dict(identity=IDENTITY,Nx=nx,Ny=ny,variance=VARIANCE,sample_id=sid,seed_components=seed,
  filename=f.name,bytes=f.stat().st_size,sha256=sha(f),diagnostics=diag))
 for p in [partial,partial.with_suffix('.json')]:p.unlink(missing_ok=True)
 return str(f)
def reuse20():
 prior=OUT.parent/'disordered_exact_wall_central_charge_through_ny100_v1'
 folder=OUT/'Nx020';folder.mkdir(exist_ok=True)
 provenance=json.loads((prior/'input_provenance.json').read_text());cfg=provenance['configuration']
 assert cfg['Nx']==20 and cfg['disorder_x_columns']==[5,15] and cfg['samples']==100
 for key in ['alpha_1','alpha_2','nshell']:assert cfg[key]==CONFIG[key]
 for rel,digest in SOURCES.items():
  if rel.startswith('src/'):assert provenance['sources'][rel]==digest
 inputs=[];data=[];diags=[]
 for sid in tqdm(range(100),desc='Verify existing Nx20 states',unit='state'):
  f=prior/'Ny040/realizations'/f'W2_09_sample_{sid:03d}.npz';r=json.loads(f.with_suffix('.json').read_text())
  assert r['identity']==provenance['identity'] and r['variance']==9 and r['Ny']==40 and r['sample_id']==sid
  assert r['bytes']==f.stat().st_size and r['sha256']==sha(f)
  with np.load(f) as z:
   assert z['entropy_by_origin'].shape==(20,40)
   assert np.all(z['diagonal_potential'][~np.isin(np.arange(1600)//2%20,[5,15])]==0)
   data.append(z['entropy_by_origin'])
  diags.append(r['diagnostics']);inputs.append(dict(path=str(f),bytes=r['bytes'],sha256=r['sha256'],seed_components=r['seed_components']))
 f=folder/'entropy_profiles.npz';np.savez_compressed(f,entropy_by_origin=data,widths=np.arange(1,21),origins=np.arange(40),sample_ids=np.arange(100),Nx=20,Ny=40,variance=9)
 dump(folder/'acquisition_complete.json',dict(status='complete',identity=IDENTITY,Nx=20,Ny=40,variance=9,samples=100,
  bytes=f.stat().st_size,cache_sha256=sha(f),reused_inputs=inputs,diagnostic_maxima={k:max(d[k] for d in diags if k in d) for k in diags[0]},
  full_space_checks='canonical block reduction already validated in source campaign; new Nx sizes receive direct full-space checks'))
def run(nx,workers,pilot=False):
 if nx==20:reuse20();return
 prepare(nx);init(nx)
 if pilot:task(0);print('PILOT PASSED Nx',nx,flush=True);return
 start=time.perf_counter()
 with ProcessPoolExecutor(max_workers=workers,initializer=init,initargs=(nx,)) as pool:
  futures=[pool.submit(task,s) for s in range(SAMPLES)]
  files=[f.result() for f in tqdm(as_completed(futures),total=SAMPLES,desc=f'Nx={nx} full entropy',unit='state')]
 data=np.empty((100,20,40));ids=[];diags=[]
 for path in files:
  f=Path(path);r=json.loads(f.with_suffix('.json').read_text());assert r['sha256']==sha(f)
  with np.load(f) as z:
   sid=int(z['sample_id']);data[sid]=z['entropy_by_origin'];ids.append(sid)
  diags.append(r['diagnostics'])
 assert sorted(ids)==list(range(100)) and np.isfinite(data).all()
 f=FOLDER/'entropy_profiles.npz';np.savez_compressed(f,entropy_by_origin=data,widths=np.arange(1,21),origins=np.arange(40),sample_ids=np.arange(100),Nx=nx,Ny=40,variance=9)
 dump(FOLDER/'acquisition_complete.json',dict(status='complete',identity=IDENTITY,Nx=nx,Ny=40,variance=9,samples=100,
  bytes=f.stat().st_size,cache_sha256=sha(f),elapsed_seconds=time.perf_counter()-start,
  diagnostic_maxima={k:max(d[k] for d in diags if k in d) for k in diags[0]},
  full_space_checks=[d['full_space_entropy_error'] for d in diags if 'full_space_entropy_error' in d]))
 print('COMPLETED Nx',nx,flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--sizes',nargs='+',type=int,default=SIZES)
 parser.add_argument('--workers',type=int,default=16);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
 assert set(args.sizes)<=set(SIZES)
 dump(OUT/'input_provenance.json',dict(configuration=CONFIG,identity=IDENTITY,sources=SOURCES,CPU_RANGE=CPU_RANGE))
 print('Configuration:',json.dumps(CONFIG),flush=True)
 for nx in args.sizes:run(nx,args.workers,args.pilot)
if __name__=='__main__':main()
