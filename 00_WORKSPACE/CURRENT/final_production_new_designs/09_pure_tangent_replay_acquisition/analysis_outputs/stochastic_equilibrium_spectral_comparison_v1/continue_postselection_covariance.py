import os
cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[16:20] if len(cpus)>20 else cpus[:4])
from analyze_comparison import *
if __name__=='__main__':
    z=np.load(OUT/'factorized_postselection_progress.npz');f=z['frame_32'];p=f@f.conj().T
    top=np.where((np.arange(1280)//2%20>=5)&(np.arange(1280)//2%20<=15))[0];outside=np.setdiff1d(np.arange(1280),top)
    original_exterior=p[np.ix_(outside,outside)].copy()
    p[outside,:]=0;p[:,outside]=0;p[outside,outside]=1
    g=2*p-np.eye(1280)
    model=classA_U1FGTN(Nx=20,Ny=32,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
    spectra={};records=[];start=time.time()
    def observe(cycle,G,**kw):
        if cycle not in [0,1,8,16,32]:return
        pp=hermitian((G+np.eye(1280))/2);pt=pp[np.ix_(top,top)]
        gram=float(abs(pt@pt-pt).max());leak=float(abs(pp[np.ix_(top,outside)]).max());rank=float(np.trace(pt).real)
        print('checks',cycle,gram,leak,rank,flush=True)
        check(gram<1e-8 and leak<1e-12 and abs(rank-362)<1e-6,'covariance continuation invariant')
        pp[np.ix_(outside,outside)]=original_exterior
        arr=np.array([reduced(pp,20,32,o) for o in range(16)]);spectra[f'spectra_{cycle+32}']=arr
        np.savez_compressed(OUT/'covariance_postselection_progress.npz',**spectra,final_projector=pp)
        records.append(dict(cycle=cycle+32,purity_error=gram,cross_sector_error=leak,active_charge=rank,
            S_y0=float(metrics(arr)['S1'][0]),S_origin_average=float(metrics(arr)['S1'].mean()),elapsed_seconds=time.time()-start))
        dump('covariance_postselection_progress.json',records)
    model.run_markov_circuit(cycles=32,postselect=True,perfect_correction=False,samples=1,G_init=g,
        meas_slab_only=True,sequence='raster_y',G_history=False,save=False,progress=True,
        state_representation='covariance',cycle_observer=observe,random_seed=20260923)
    np.savez_compressed(OUT/'covariance_postselection.npz',**spectra)
    dump('covariance_postselection.json',dict(status='complete',canonical_entry_point='classA_U1FGTN.run_markov_circuit',
        initial_cycle=32,final_cycle=64,representation='covariance',diagnostics=records,
        method='Continue the accepted factorized frame snapshot at cycle 32 using canonical covariance updates. Replace inert exterior with filled product during evolution and restore its original product projector for observables; initial cross-sector roundoff is set to exact zero.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py']},
        saved_sha256=sha(OUT/'covariance_postselection.npz')))
