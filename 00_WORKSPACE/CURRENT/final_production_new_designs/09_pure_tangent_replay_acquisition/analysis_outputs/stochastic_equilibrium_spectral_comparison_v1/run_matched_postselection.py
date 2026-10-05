from analyze_comparison import *
from extend_sizes import frame_spectrum
if __name__=='__main__':
    meta=json.loads((SAVED/'spectra_diagnostics_Ny032.json').read_text());f=ROOT/meta['contract']['inputs'][0]['result']
    with np.load(f) as z:initial=z['initial_frame'][0,:,:int(z['initial_ranks'][0])].copy()
    model=classA_U1FGTN(Nx=20,Ny=32,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
    frames={};spectra={};records=[];start=time.time()
    def observe(cycle,state,**kw):
        if cycle not in [0,8,16,32,64,128]:return
        ff=state.snapshot(copy=True)['frame'];frames[f'frame_{cycle}']=ff
        a=np.array([frame_spectrum(ff,20,32,o) for o in range(16)])
        spectra[f'spectra_{cycle}']=a;m=metrics(a)
        record=dict(cycle=cycle,rank=ff.shape[1],S_y0=float(m['S1'][0]),S_origin_average=float(m['S1'].mean()),V_origin_average=float(m['V'].mean()),elapsed_seconds=time.time()-start)
        if records:
            prev=frames[f"frame_{records[-1]['cycle']}"]
            record['projector_change_frobenius']=float(np.sqrt(max(0,ff.shape[1]+prev.shape[1]-2*np.sum(abs(ff.conj().T@prev)**2))))
        records.append(record);print(record,flush=True)
        np.savez_compressed(OUT/'matched_postselection_progress.npz',**frames,**spectra)
        dump('matched_postselection_progress.json',records)
    result=model.run_markov_circuit(cycles=128,postselect=True,perfect_correction=False,samples=1,
        frame_init=initial,frame_init_prepared=True,meas_slab_only=True,sequence='raster_y',
        G_history=False,save=False,progress=True,return_native_state=True,state_representation='physical_frame',
        native_cycle_observer=observe,require_no_covariance_materialization=True,random_seed=20260923)
    np.savez_compressed(OUT/'matched_postselection.npz',**frames,**spectra)
    dump('matched_postselection.json',dict(status='complete',entry_point='classA_U1FGTN.run_markov_circuit',
        cycles=128,initial_sample_id=0,initial_source=str(f.relative_to(ROOT)),initial_source_sha256=sha(f),
        frame_init_prepared=True,meas_slab_only=True,sequence='raster_y',postselect=True,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py']},
        saved_sha256=sha(OUT/'matched_postselection.npz'),diagnostics=records))
    print('Matched postselection complete',flush=True)
