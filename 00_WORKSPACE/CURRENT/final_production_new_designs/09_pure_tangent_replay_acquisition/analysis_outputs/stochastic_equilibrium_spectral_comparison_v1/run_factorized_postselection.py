from analyze_comparison import *
from extend_sizes import frame_spectrum
if __name__=='__main__':
    old=np.load(OUT/'matched_postselection_progress.npz');original=old['frame_0'];ip=original@original.conj().T
    nx=20;ny=32;dim=1280
    top=np.where((np.arange(dim)//2%nx>=5)&(np.arange(dim)//2%nx<=15))[0]
    outside=np.setdiff1d(np.arange(dim),top);rank=int(round(np.trace(ip[np.ix_(top,top)]).real))
    vals,vec=eigh(hermitian(ip[np.ix_(top,top)]),driver='evd');check(abs(vals-np.rint(vals)).max()<1e-10,'initial active purity')
    active=np.zeros((dim,rank),complex);active[top]=vec[:,-rank:]
    occupied_outside=outside[np.diag(ip)[outside].real>.5]
    spectators=np.eye(dim,dtype=complex)[:,occupied_outside]
    reconstructed=np.column_stack([active,spectators]);initial_error=float(abs(reconstructed@reconstructed.conj().T-ip).max())
    check(initial_error<1e-12,'factorized initial-state equivalence')
    model=classA_U1FGTN(Nx=20,Ny=32,DW=True,nshell=1,alpha_1=1,alpha_2=30,trial_orbitals='X',dw_truncation=True)
    p,h,es,us,ed=equilibrium(20,32);ht=h[np.ix_(top,top)];et=eigvalsh(hermitian(ht));eg=et[:rank].sum()
    frames={};spectra={};records=[];start=time.time()
    def observe(cycle,state,**kw):
        if cycle not in [0,8,16,32,64]:return
        f=state.snapshot(copy=True)['frame'];check(f.shape[1]==rank,'active charge conservation')
        leakage=float(abs(f[outside]).max());gram=float(abs(f.conj().T@f-np.eye(rank)).max())
        check(leakage<1e-12 and gram<1e-9,'factorized-state integrity')
        ff=np.column_stack([f,spectators]);frames[f'frame_{cycle}']=ff
        a=np.array([frame_spectrum(ff,20,32,o) for o in range(16)])
        spectra[f'spectra_{cycle}']=a;m=metrics(a);pt=f[top]@f[top].conj().T
        row=dict(cycle=cycle,rank=ff.shape[1],active_rank=rank,exterior_row_maximum=leakage,orthogonality_maximum=gram,
            S_y0=float(m['S1'][0]),S_origin_average=float(m['S1'].mean()),V_origin_average=float(m['V'].mean()),
            active_energy_excess=float(np.einsum('ij,ji->',ht,pt).real-eg),commutator_frobenius=float(np.linalg.norm(ht@pt-pt@ht)),elapsed_seconds=time.time()-start)
        if f'frame_{cycle}' in old:
            fo=old[f'frame_{cycle}'];row['unfactorized_projector_difference_frobenius']=float(np.linalg.norm(ff@ff.conj().T-fo@fo.conj().T))
        if records:
            prev=frames[f"frame_{records[-1]['cycle']}"];row['projector_change_frobenius']=float(np.linalg.norm(ff@ff.conj().T-prev@prev.conj().T))
        records.append(row);print(row,flush=True)
        np.savez_compressed(OUT/'factorized_postselection_progress.npz',**frames,**spectra)
        dump('factorized_postselection_progress.json',records)
    model.run_markov_circuit(cycles=64,postselect=True,perfect_correction=False,samples=1,
        frame_init=active,frame_init_prepared=True,meas_slab_only=True,sequence='raster_y',
        G_history=False,save=False,progress=True,return_native_state=True,state_representation='physical_frame',
        native_cycle_observer=observe,require_no_covariance_materialization=True,random_seed=20260923)
    np.savez_compressed(OUT/'factorized_postselection.npz',**frames,**spectra)
    dump('factorized_postselection.json',dict(status='complete',canonical_entry_point='classA_U1FGTN.run_markov_circuit',
        cycles=64,initial_sample_id=0,initial_projector_maximum_error=initial_error,active_rank=rank,restored_spectator_rank=len(occupied_outside),
        preparation='Exact factorization into active slab and inert product spectator. Only the active frame evolves under the canonical full-geometry engine; spectators are restored for all observables.',
        prior_run_issue='Unfactorized initial frame has order-1e-14 cross-sector coherence. Deterministic rare-outcome conditioning amplified it by cycle 64. That run is retained as a numerical diagnostic, not an accepted late-time reference.',
        classification='Numerical conditioning of an approximately factorized initial state; controlled input-factorization comparison. No canonical engine code was changed.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/fgtn/classA_U1FGTN.py',ROOT/'src/fgtn/occupied_frame.py']},
        saved_sha256=sha(OUT/'factorized_postselection.npz'),diagnostics=records))
    print('Factorized matched postselection complete',flush=True)
