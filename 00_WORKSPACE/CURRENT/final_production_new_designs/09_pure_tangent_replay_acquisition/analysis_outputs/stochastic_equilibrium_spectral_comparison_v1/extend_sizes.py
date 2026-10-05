from analyze_comparison import *

def frame_spectrum(f,nx,ny,origin):
    ids=(((np.arange(ny//2)+origin)%ny)[:,None]*2*nx+np.arange(2*nx)[None,:]).ravel()
    fa=f[ids];g=hermitian(2*(fa@fa.conj().T)-np.eye(len(ids)))
    l=eigvalsh(g,driver='evr');metrics(l)
    check(abs(l.sum()-np.trace(g))<1e-8,'trace consistency')
    return l

if __name__=='__main__':
    root=next((BASE/'14_hard_wall_xresolved_correlator_scaling/gpu_data').glob('*v2*'))
    arrays={};rows=[];diagnostics={}
    for ny in tqdm([40,50,60],desc='Larger production sizes'):
        n=20*ny;st=np.empty((100,n));quarter=st.copy();ranks=np.empty(100,int);idsall=[]
        # First sample in every immutable five-sample shard: 20 independent trajectories.
        origin_ids=np.arange(0,100,5);origins=np.arange(ny//2)
        origin_spec=np.empty((20,ny//2,n))
        max_cross=0.;rawlo=1.;rawhi=0.
        for receipt in tqdm(sorted((root/f'results/Ny{ny:03}').glob('*.complete.json')),desc=f'Ny={ny} shards',leave=False):
            d=json.loads(receipt.read_text());f=receipt.parent/d['result_filename']
            check(d['status']=='complete' and d['Ny']==ny and d['cycles']==2*ny,'completion metadata')
            check(d['configuration_sha256']=='49998b3e6fd0f44e9a55d9d2b2f2b5a85707a175b6029c259f2a667ddf00efd4','config identity')
            check(f.stat().st_size==d['result_bytes'] and sha(f)==d['result_sha256'],'original checksum')
            record(receipt);provenance.append(dict(path=str(f.relative_to(ROOT)),sha256=d['result_sha256'],bytes=d['result_bytes']))
            with np.load(f) as z:
                ids=z['global_sample_indices'];check(np.array_equal(ids,d['global_sample_indices']),'sample identities')
                check(bool(z['meas_slab_only']) and bool(z['dw_truncation']) and str(z['sequence'])=='raster_y','protocol')
                check(int(z['Nx'])==20 and float(z['alpha_1'])==1 and float(z['alpha_2'])==30 and int(z['nshell'])==1,'Hamiltonian identity')
                nu=z['half_system_occupation_spectrum'][:,0];fr=z['occupied_frame'];rk=z['occupied_ranks'].reshape(-1)
                rawlo=min(rawlo,float(np.min(z['half_system_raw_occupation_minimum'])))
                rawhi=max(rawhi,float(np.max(z['half_system_raw_occupation_maximum'])))
                check(float(np.max(z['maximum_half_system_hermiticity_residual']))<1e-9,'saved Hermiticity')
            for j,i in enumerate(ids):
                i=int(i);ff=fr[j].reshape(2*n,-1)[:,:int(rk[j])];ranks[i]=rk[j]
                st[i]=2*nu[j]-1;quarter[i]=frame_spectrum(ff,20,ny,ny//4)
                if i in origin_ids:
                    oi=int(np.where(origin_ids==i)[0][0])
                    for o in origins:
                        l=quarter[i] if o==ny//4 else frame_spectrum(ff,20,ny,int(o))
                        origin_spec[oi,o]=l
                    max_cross=max(max_cross,float(abs(origin_spec[oi,0]-st[i]).max()))
            idsall.extend(ids.tolist())
        check(sorted(idsall)==list(range(100)),'exactly 100 distinct samples')
        check(max_cross<1e-9 and rawlo>-1e-8 and rawhi<1+1e-8,'saved occupations roundoff/crosscheck')
        p,h,e,u,diag=equilibrium(20,ny)
        eq=reduced(p,20,ny);eqtop=p.copy()
        # Exact block separation: replace the inert exterior with a product projector.
        top=np.where((np.arange(2*n)//2%20>=5)&(np.arange(2*n)//2%20<=15))[0]
        outside=np.setdiff1d(np.arange(2*n),top)
        check(abs(p[np.ix_(top,outside)]).max()<1e-10,'hard-wall cross-sector')
        eqtop[outside,:]=0;eqtop[:,outside]=0;eqtop[outside,outside]=(outside%2==0).astype(float)
        eq_active=reduced(eqtop,20,ny)
        for label,arr in [('stochastic',st),('equilibrium',eq[None,:]),('equilibrium_active_only',eq_active[None,:]),('quarter_cut',quarter)]:
            arrays[f'{label}_{ny}']=arr
            m=metrics(arr)
            for i in range(len(arr)):rows.append(dict(Ny=ny,protocol=label,sample_id=i,**{k:float(v[i]) for k,v in m.items()}))
        arrays[f'origins_{ny}']=origin_spec;arrays[f'origin_ids_{ny}']=origin_ids;arrays[f'ranks_{ny}']=ranks
        diagnostics[str(ny)]=dict(equilibrium=diag,occupation_crosscheck=max_cross,raw_occupation_minimum=rawlo,raw_occupation_maximum=rawhi,origin_samples=20,origin_cut_count=ny//2,
            S_y0=ms(metrics(st)['S1']),S_quarter=ms(metrics(quarter)['S1']),S_origin_average=ms(metrics(origin_spec)['S1'].mean(1)),S_equilibrium=float(metrics(eq)['S1']))
        print(ny,diagnostics[str(ny)],flush=True)
    np.savez_compressed(OUT/'extended_size_controls.npz',**arrays)
    write_csv('extended_size_metrics.csv',rows);dump('extended_size_diagnostics.json',diagnostics);dump('extended_input_provenance.json',provenance)
    print('Extended comparison complete',flush=True)
