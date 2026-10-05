from pathlib import Path
import os
os.sched_setaffinity(0,range(40,56))
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,contextlib,io
import numpy as np
root=Path('/home/abhuiyan/class_A_fermionic_adaptive_circuit')
sys.path.insert(0,str(root/'src/fgtn'))
from classA_U1FGTN import classA_U1FGTN
out=root/'00_WORKSPACE/CURRENT/final_production_new_designs/06_domain_wall_flattened_ground_state_reference/analysis_outputs/disordered_exact_wall_Nx_scan_Ny040_W2_9_v1'
results=[]
for nx in [12,16,20,24,32,40]:
    walls=[nx//4,3*nx//4];ny=40
    p=out/f'Nx{nx:03d}/parent.npz' if nx!=20 else out.parent/'disordered_exact_wall_central_charge_through_ny100_v1/Ny040/parent.npz'
    with np.load(p) as z:h=z['hamiltonian']
    with contextlib.redirect_stdout(io.StringIO()):
        model=classA_U1FGTN(Nx=nx,Ny=ny,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
        model.construct_OW_projectors(nshell=1,DW=True,trial_orbitals='X',dw_truncation=True)
    rows=np.array([2*walls[0],2*walls[1]+1,2*nx*7+2*(walls[0]+1),2*nx*39+2*walls[1],2*nx*13+nx])
    direct=np.zeros((len(rows),2*nx*ny),complex)
    for name,sign in [('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)]:
        w=np.asarray(getattr(model,name)).reshape(2*nx*ny,-1)
        direct+=sign*w[rows]@w.conj().T
    error=float(np.max(np.abs(direct-h[rows])))
    assert error<1e-10
    results.append(dict(Nx=nx,Ny=ny,walls=walls,rows=rows.tolist(),max_absolute_error=error,parent_path=str(p),parent_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    print('Nx',nx,'active-row canonical check:',error,flush=True)
    del model,h,w,direct
(out/'active_parent_rows_validation.json').write_text(json.dumps({'checks':results,'tolerance':1e-10,'source_sha256':hashlib.sha256((root/'src/fgtn/classA_U1FGTN.py').read_bytes()).hexdigest()},indent=2)+'\n')
