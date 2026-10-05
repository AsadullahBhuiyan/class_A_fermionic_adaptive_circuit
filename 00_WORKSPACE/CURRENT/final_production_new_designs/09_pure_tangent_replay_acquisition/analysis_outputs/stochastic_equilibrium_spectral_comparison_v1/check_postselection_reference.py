from analyze_comparison import *
if __name__=='__main__':
    z=np.load(OUT/'matched_postselection_progress.npz');initial=z['frame_0'];p,h,e,u,diag=equilibrium(20,32)
    ix=np.where((np.arange(1280)//2%20>=5)&(np.arange(1280)//2%20<=15))[0]
    ht=hermitian(h[np.ix_(ix,ix)]);et,ut=eigh(ht,driver='evd');rank=int(round(np.sum(abs(initial[ix])**2)))
    mu=(et[rank]+et[rank-1])/2;lo=et<mu-1e-10;deg=abs(et-mu)<=1e-10;k=rank-int(lo.sum());d=int(deg.sum())
    print('rank',rank,'Fermi degeneracy',d,'occupied in degenerate space',k,flush=True)
    rng=np.random.default_rng(20260923);rows=[]
    outside=np.setdiff1d(np.arange(1280),ix);initialp=initial@initial.conj().T
    for draw in tqdm(range(32),desc='Degenerate ground-state choices'):
        q,_=np.linalg.qr(rng.normal(size=(d,k))+1j*rng.normal(size=(d,k)))
        f=np.column_stack([ut[:,lo],ut[:,deg]@q]);ps=np.zeros_like(p)
        ps[np.ix_(ix,ix)]=f@f.conj().T;ps[np.ix_(outside,outside)]=initialp[np.ix_(outside,outside)]
        vals=np.array([metrics(reduced(ps,20,32,o))['S1'] for o in range(16)])
        rows.append(dict(draw=draw,S_y0=vals[0],S_origin_mean=vals.mean(),S_origin_min=vals.min(),S_origin_max=vals.max()))
    write_csv('sector_ground_state_degeneracy.csv',rows)
    obs=[]
    for cycle in [8,16,32,64,128]:
        key=f'frame_{cycle}'
        if key not in z:continue
        f=z[key][ix];ps=f@f.conj().T
        energy=float(np.einsum('ij,ji->',ht,ps).real-et[:rank].sum())
        obs.append(dict(cycle=cycle,active_rank=float(np.trace(ps).real),energy_above_same_sector_ground_state=energy,
            commutator_frobenius=float(np.linalg.norm(ht@ps-ps@ht))))
    dump('postselection_reference_checks.json',dict(active_rank=rank,degeneracy=d,occupied_in_degenerate_space=k,
        ground_state_entropy_draw_min=min(r['S_origin_mean'] for r in rows),ground_state_entropy_draw_max=max(r['S_origin_mean'] for r in rows),
        warning='Sampled pure-state choices within the numerically degenerate Fermi subspace; this is not a proven extremal bound.',observations=obs))
    print(obs,flush=True)
