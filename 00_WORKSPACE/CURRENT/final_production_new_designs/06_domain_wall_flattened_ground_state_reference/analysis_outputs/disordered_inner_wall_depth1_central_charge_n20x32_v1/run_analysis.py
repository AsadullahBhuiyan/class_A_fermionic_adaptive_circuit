"""Full-entropy profiles of the exact saved disorder realizations. CPU only."""
from pathlib import Path
import os
CPU_RANGE=(8,23)
os.sched_setaffinity(0,range(CPU_RANGE[0],CPU_RANGE[1]+1))
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json,hashlib,time,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.linalg import eigh,eigvalsh
from scipy.special import xlogy
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
threadpool_limits(1)
OUT=Path(__file__).resolve().parent
SOURCES=[OUT.parent/'disordered_inner_wall_depth1_strength_scan_n20x32_v1',
         OUT.parent/'disordered_inner_wall_depth1_W2_12_16_25_n20x32_v1']
CONFIG={'Nx':20,'Ny':32,'widths':list(range(1,17)),'origins':list(range(32)),
 'variances':[1,2,3,4,6,9,12,16,25],'samples_per_variance':100,
 'disorder_x_columns':[5,6,14,15],'filling':'globally lowest 640 energies',
 'entropy':'full binary entropy in nats; no spectral window',
 'averaging':'all 32 origins within each realization, then independent realizations',
 'fit':'S=s0+(c_fit/3)*log[(Ny/pi)sin(pi Ay/Ny)]',
 'fit_range':[5,16],'sensitivity_fit_ranges':[[2,16],[8,16]],
 'dynamics':'none; reconstruct static ground states from saved canonical Hamiltonian and potentials'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
IDENTITY=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()+Path(__file__).read_bytes()).hexdigest()
def herm(a):
 assert np.isfinite(a).all()
 err=float(np.max(np.abs(a-a.conj().T)));assert err<1e-10,err
 return (a+a.conj().T)/2
def binary(p):
 assert np.isfinite(p).all() and p.min()>=-1e-8 and p.max()<=1+1e-8
 p=np.clip(p,0,1)
 return float(np.sum(-xlogy(p,p)-xlogy(1-p,1-p)))
def profiles(c,q):
 values=np.empty((16,32));lo=1.;hi=0.;trace_err=0.
 for j,a in enumerate(range(1,17)):
  for y0 in range(32):
   rows=(((np.arange(a)+y0)%32)[:,None]*q+np.arange(q)).ravel()
   sub=c[np.ix_(rows,rows)]
   ev=eigvalsh(herm(sub),driver='evr',check_finite=False)
   lo=min(lo,float(ev.min()));hi=max(hi,float(ev.max()))
   trace_err=max(trace_err,abs(float(ev.sum()-np.trace(sub).real)))
   values[j,y0]=binary(ev)
 assert trace_err<1e-8
 complement=float(np.max(np.abs(values[-1,:16]-values[-1,16:])))
 assert complement<1e-7,complement
 return values,{'occupation_min':lo,'occupation_max':hi,'trace_max_error':trace_err,'half_complement_entropy_max_error':complement}
def init():
 global H0,HA,EE,UE,ACTIVE,EXT,EXT_PROFILE,EXT_LAM
 with np.load(SOURCES[0]/'clean_reference.npz') as z:
  H0=z['hamiltonian'];ACTIVE=z['active_indices'];EXT=z['exterior_indices']
 HA=herm(H0[np.ix_(ACTIVE,ACTIVE)])
 assert np.max(np.abs(H0[np.ix_(ACTIVE,EXT)]))<1e-12
 EE,UE=eigh(herm(H0[np.ix_(EXT,EXT)]),driver='evd',check_finite=False)
 with np.load(OUT/'exterior_reference.npz') as z:
  EXT_PROFILE=z['entropy'];EXT_LAM=z['half_centered_eigenvalues']
def task(t):
 variance,sid,path=t
 f=OUT/'realizations'/f'W2_{variance:02d}_sample_{sid:03d}.npz'
 receipt=f.with_suffix('.json')
 input_digest=sha(Path(path)) if path else sha(SOURCES[0]/'clean_reference.npz')
 if f.exists() and receipt.exists():
  r=json.loads(receipt.read_text())
  if r['identity']==IDENTITY and r['input_sha256']==input_digest and r['sha256']==sha(f) and r['bytes']==f.stat().st_size:return str(f)
 if path:
  with np.load(path) as z:potential=z['diagonal_potential'];ref=z['centered_eigenvalues'];eref=z['hamiltonian_energies']
 else:
  potential=np.zeros(1280)
  with np.load(SOURCES[0]/'clean_reference.npz') as z:ref=z['centered_eigenvalues'];eref=z['energies']
 h=HA.copy();h[np.diag_indices_from(h)]-=potential[ACTIVE]
 ea,u=eigh(herm(h),driver='evd',check_finite=False)
 all_e=np.r_[ea,EE];order=np.argsort(all_e,kind='stable');occupied=np.zeros(1280,bool);occupied[order[:640]]=True
 oa=occupied[:704];oe=occupied[704:]
 assert np.array_equal(np.flatnonzero(oe),np.arange(288))
 energy_error=float(np.max(np.abs(np.sort(all_e)-eref)));assert energy_error<1e-10
 fa=u[:,oa];c=fa@fa.conj().T
 purity=float(np.max(np.abs(c@c-c)));assert purity<1e-10
 entropy,diag=profiles(c,22);entropy+=EXT_PROFILE
 half=np.sort(np.r_[2*eigvalsh(herm(c[:352,:352]),driver='evr')-1,EXT_LAM])
 spectral_error=float(np.max(np.abs(half-ref)));assert spectral_error<(1e-6 if variance==0 else 1e-8)
 entropy_error=abs(float(entropy[-1,0])-binary((ref+1)/2));assert entropy_error<1e-7
 diag.update(purity_max_error=purity,saved_spectral_max_error=spectral_error,
             saved_energy_max_error=energy_error,saved_half_entropy_error=entropy_error)
 if sid==0 and variance in [0,25]:
  # Direct full-space covariance checks, including a cut wrapping across y=31.
  ef,uf=eigh(herm(H0-np.diag(potential)),driver='evd')
  ff=uf[:,:640];full=ff@ff.conj().T;errs=[]
  for a,y0 in [(5,29),(16,7)]:
   rows=(((np.arange(a)+y0)%32)[:,None]*40+np.arange(40)).ravel()
   val=binary(eigvalsh(herm(full[np.ix_(rows,rows)]),driver='evr'))
   errs.append(abs(val-entropy[a-1,y0]))
  assert max(errs)<1e-7
  diag['full_space_entropy_crosscheck']=max(errs)
 tmp=f.with_suffix('.tmp.npz')
 np.savez_compressed(tmp,entropy_by_origin=entropy,variance=variance,sample_id=sid,
                     widths=np.arange(1,17),origins=np.arange(32))
 tmp.replace(f)
 dump(receipt,{'identity':IDENTITY,'variance':variance,'sample_id':sid,'input_path':path,
              'input_sha256':input_digest,'filename':f.name,'bytes':f.stat().st_size,'sha256':sha(f),'diagnostics':diag})
 return str(f)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
 (OUT/'realizations').mkdir(exist_ok=True)
 inputs=[];tasks=[]
 for source in SOURCES:
  meta=json.loads((source/'completion_manifest.json').read_text())
  for name in ['clean_reference.npz','input_provenance.json']:
   f=source/name;assert sha(f)==meta['files'][name]['sha256']
  for f in sorted((source/'realizations').glob('*.npz')):
   r=json.loads(f.with_suffix('.json').read_text())
   assert f.stat().st_size==r['bytes'] and sha(f)==r['sha256']==meta['files']['realizations/'+f.name]['sha256']
   assert r['identity']==json.loads((source/'input_provenance.json').read_text())['identity']
   tasks.append((int(r['variance']),int(r['sample_id']),str(f)))
   inputs.append({'path':str(f),'bytes':r['bytes'],'sha256':r['sha256'],'variance':r['variance'],'sample_id':r['sample_id']})
 for v in CONFIG['variances']:assert sorted(s for w,s,p in tasks if w==v)==list(range(100))
 with np.load(SOURCES[0]/'clean_reference.npz') as z:
  h0=z['hamiltonian'];ext=z['exterior_indices']
 e,u=eigh(herm(h0[np.ix_(ext,ext)]),driver='evd')
 c=u[:,:288]@u[:,:288].conj().T
 if not (OUT/'exterior_reference.npz').exists():
  ep,ed=profiles(c,18)
  assert np.max(np.ptp(ep,axis=1))<1e-7
  np.savez_compressed(OUT/'exterior_reference.npz',entropy=ep,half_centered_eigenvalues=2*eigvalsh(herm(c[:288,:288]))-1)
  dump(OUT/'exterior_diagnostics.json',ed)
 dump(OUT/'input_provenance.json',{'configuration':CONFIG,'identity':IDENTITY,'inputs':inputs,
       'hamiltonian_path':str(SOURCES[0]/'clean_reference.npz'),'hamiltonian_file_sha256':sha(SOURCES[0]/'clean_reference.npz'),
       'canonical_parent_provenance':json.loads((SOURCES[0]/'input_provenance.json').read_text()),
       'method':'exact block reduction; same 900 saved disorder potentials; reconstruct occupied projector; no new draws'})
 init()
 start=time.perf_counter()
 clean=task((0,0,''))
 if args.pilot:
  task(tasks[0]);print('Pilot elapsed:',time.perf_counter()-start,flush=True);return
 print('900 realizations; 16 widths; 32 origins; CPU range',CPU_RANGE,flush=True)
 with ProcessPoolExecutor(max_workers=16,initializer=init) as pool:
  files=list(tqdm(pool.map(task,tasks,chunksize=1),total=len(tasks),desc='Entropy profiles',unit='state'))
 data=np.empty((9,100,16,32))
 diagnostics=[]
 for f in files:
  with np.load(f) as z:
   i=CONFIG['variances'].index(int(z['variance']));data[i,int(z['sample_id'])]=z['entropy_by_origin']
  diagnostics.append(json.loads(Path(f).with_suffix('.json').read_text())['diagnostics'])
 with np.load(clean) as z:clean_entropy=z['entropy_by_origin']
 np.savez_compressed(OUT/'entropy_profiles.npz',entropy_by_origin=data,clean_entropy_by_origin=clean_entropy,
                     variances=CONFIG['variances'],widths=np.arange(1,17),origins=np.arange(32),sample_ids=np.arange(100))
 dump(OUT/'acquisition_complete.json',{'status':'complete','identity':IDENTITY,'realizations':900,'clean_states':1,
       'elapsed_seconds':time.perf_counter()-start,'cache_sha256':sha(OUT/'entropy_profiles.npz'),
       'diagnostic_maxima':{k:max(d[k] for d in diagnostics if k in d) for k in diagnostics[0]},
       'full_space_checks':[d['full_space_entropy_crosscheck'] for d in diagnostics if 'full_space_entropy_crosscheck' in d]})
 print('Completed full entropy profiles.',flush=True)
if __name__=='__main__':main()
