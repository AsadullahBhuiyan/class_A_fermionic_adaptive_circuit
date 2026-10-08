"""Independent numerical checks for the clean-parent research note."""
import argparse
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
from scipy.linalg import eigh, expm
from scipy.special import expit
from threadpoolctl import threadpool_limits

import benchmark as b
import analyze as a


def assert_close(actual,expected,tol=2e-11):
    np.testing.assert_allclose(actual,expected,rtol=0,atol=tol)
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))


def small_dense_reference():
    nx,ny=8,8
    h,checks=b.build_parent(nx,ny,1.,dense_check=True)
    data,diag=b.diagonalize(h)
    dense=b.dense_from_delta(np.fft.ifft(h,axis=0),ny)
    de,dv=eigh(dense)
    checks['full_vs_block_spectrum']=assert_close(de,np.sort(data['raw_energies'].ravel()))
    p=b.dense_from_delta(data['projector_delta'],ny)
    checks['dense_ground_state_projector']=assert_close(p,dv[:,:nx*ny]@dv[:,:nx*ny].conj().T)
    checks['full_projector_idempotency']=assert_close(p@p,p)
    checks['rank']=assert_close(np.trace(p),nx*ny)
    perm=np.roll(np.arange(2*nx*ny).reshape(ny,2*nx),1,axis=0).ravel()
    checks['translation']=assert_close(p[np.ix_(perm,perm)],p)
    left,right=b.interfaces(nx)
    region=((np.arange(2*nx*ny)//2)%nx>=left)&((np.arange(2*nx*ny)//2)%nx<=right)
    checks['parent_region_separation']=assert_close(dense[np.ix_(region,~region)],0.)
    g=p.T
    independent=[]
    for r in range(ny//2+1):
        value=0.
        for y in range(ny):
            for x in range(nx):
                row=2*(x+nx*y)+np.arange(2)
                col=2*(x+nx*((y+r)%ny))+np.arange(2)
                value+=np.sum(abs(g[np.ix_(row,col)])**2)/(2*nx*ny)
        independent.append(value)
    checks['correlation_definition']=assert_close(independent,b.correlations(data['projector_delta'])[0])
    width=3
    nu,s=b.restricted(data['projector_delta'],width,True)
    A=np.arange(width*2*nx);Ac=np.arange(width*2*nx,2*nx*ny)
    pa=p[np.ix_(A,A)];cross=p[np.ix_(A,Ac)]
    checks['contour_sum']=assert_close(s.sum(),b.entropy_values(nu).sum())
    checks['variance_matrix_identity']=assert_close(np.sum(nu*(1-nu)),np.trace(pa-pa@pa).real)
    checks['variance_cross_cut_identity']=assert_close(np.sum(nu*(1-nu)),np.sum(abs(cross)**2))
    checks['complement_entropy']=assert_close(b.entropy_values(nu).sum(),b.entropy_values(eigh(p[np.ix_(Ac,Ac)],eigvals_only=True)).sum())
    checks['translated_strip']=assert_close(b.dense_from_delta(data['projector_delta'],width,5),pa)
    frame=a.occupied_frame(data)
    marker=a.local_marker(frame,nx,ny)
    model=SimpleNamespace(Nx=nx,Ny=ny,Ntot=2*nx*ny)
    canonical=b.classA_U1FGTN.local_chern_marker_flat(model,2*p-np.eye(len(p)),apply_tanh=False)
    checks['canonical_local_marker']=assert_close(marker,canonical.T,1e-9)
    checks['local_marker_sum']=assert_close(marker.sum(),0.,1e-9)
    # Compare the independently written disk trace to the maintained production observer.
    observer=b.REPO/'00_WORKSPACE/CURRENT/final_production_new_designs/27_square_hard_wall_random_center_chern/random_center_observer.py'
    spec=importlib.util.spec_from_file_location('parent_validation_chern',observer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(2)
    tables=module.sector_table(nx,ny,2.)
    centers=np.arange(ny)[None,:]
    reference=module.batched_chern(torch.from_numpy(frame)[None],np.array([nx*ny]),centers,tables).numpy()[0]
    values=np.array([a.disk_chern(g,nx,ny,2.,cy) for cy in range(ny)])
    checks['canonical_disk_chern']=assert_close(values,reference)
    assert values.mean()>0
    checks['observer_source_sha256']=b.sha(observer)
    # Unit tests of the explicitly deterministic unresolved-Fermi-level policy.
    synthetic=np.tile(np.diag([-1.,0.,0.,1.]),(2,1,1)).astype(complex)
    test,diagnostic=b.diagonalize(synthetic)
    assert diagnostic['fermi_degenerate_count']==4 and diagnostic['rank']==4
    checks['degenerate_primary_rank']=diagnostic['rank']
    alt=b.dense_from_delta(test['alternative_projector_delta'],2)
    assert_close(alt@alt,alt);assert_close(np.trace(alt),4)
    checks['degenerate_alternative_energy']=assert_close(np.trace(b.dense_from_delta(np.fft.ifft(synthetic,axis=0),2)@alt),-2)
    return checks


def exact_fock_check():
    """A complex hopping matrix makes an erroneous transpose observable."""
    h=np.array([[.4,.2+.3j,.1j],[.2-.3j,-.6,.15],[ -.1j,.15,.8]],complex)
    n=3;dimension=2**n;operators=[]
    for j in range(n):
        c=np.zeros((dimension,dimension),complex)
        for state in range(dimension):
            if state&(1<<j):
                sign=(-1)**((state&((1<<j)-1)).bit_count())
                c[state^(1<<j),state]=sign
        operators.append(c)
    many=sum(h[i,j]*(operators[i].conj().T@operators[j]) for i in range(n) for j in range(n))
    energies,v=eigh(h);out={}
    for t in (0.,.3,1.,60.):
        rho=expm(-2*t*(many-np.linalg.eigvalsh(many).min()*np.eye(dimension)))
        rho/=np.trace(rho)
        direct=np.array([[np.trace(rho@operators[i].conj().T@operators[j]) for j in range(n)] for i in range(n)])
        gaussian=((v*expit(-2*t*energies))@v.conj().T).T
        out[f'filter_t{t:g}']=assert_close(direct,gaussian)
    ground=(v[:,energies<0]@v[:,energies<0].conj().T).T
    out['large_t_filter_limit']=assert_close(((v*expit(-2000*energies))@v.conj().T).T,ground)
    # Independently compare the modular generator and its transposed action.
    p=(v*np.array([.15,.45,.82]))@v.conj().T
    g=p.T;nu,u=eigh(g)
    h_g=(u*np.log((1-nu)/nu))@u.conj().T
    nup,up=eigh(p)
    action=(up*np.exp(-.37j*np.log((1-nup)/nup)))@up.conj().T
    out['modular_transpose_action']=assert_close(action,expm(-.37j*h_g.T))
    out['modular_unitarity']=assert_close(action.conj().T@action,np.eye(n))
    return out


def saved_products():
    sources=b.source_identity();count=0;maximum=0.
    for nx,ny,alpha in b.task_table():
        path=b.HERE/'data/cases'/f'{b.case_id(nx,ny,alpha)}.npz'
        assert b.verified_case(path,b.configuration(nx,ny,alpha),sources),path
        data,diag=b.load_case(nx,ny,alpha)
        assert all(np.isfinite(value).all() for value in data.values())
        assert diag['rank']==nx*ny and diag['ow_translation_residual']<2e-12
        if 'half_contour' in data:assert_close(data['half_contour'].sum(),data['entropy'][-1],2e-10)
        if 'mutual_information' in diag:assert diag['mutual_information']>=-1e-10
        maximum=max(maximum,diag['projector_idempotency_residual']);count+=1
    hist=np.load(b.HERE/'data/histograms.npz')
    for alpha in (1,3):
        assert_close(np.dot(hist[f'occupation_density_a{alpha}'],np.diff(hist['occupation_edges'])),1.)
        assert_close(np.dot(hist[f'energy_density_a{alpha}'],np.diff(hist['energy_edges'])),1.)
    modular=np.load(b.HERE/'data/modular.npz')
    for alpha in (1,3):
        for cutoff in (8,10,12):
            key=f'a{alpha}_eps{cutoff}'
            assert_close(modular[f'{key}_density'].sum((2,3)),2.)
            assert_close(modular[f'{key}_displacement'][:,0],0.)
            assert (modular[f'{key}_retained_weight']>0).all()
            if alpha==1:
                assert (modular[f'{key}_displacement'][0]>=-1e-12).all()
                assert (modular[f'{key}_displacement'][1]<=1e-12).all()
    fits=json.loads((b.HERE/'data/fits.json').read_text())
    assert fits['mode_count']['weighted_r_squared'] is None
    return dict(verified_cases=count,max_projector_idempotency_residual=maximum,
                histogram_normalization=True,modular_cutoff_handedness=True,
                constant_fit_r_squared='undefined, explicitly recorded as null')


def main(threads):
    with threadpool_limits(limits=threads):
        result=dict(dense_and_estimators=small_dense_reference(),
                    exact_fock=exact_fock_check(),saved_products=saved_products(),
                    validation_script_sha256=b.sha(__file__),
                    analysis_script_sha256=b.sha(b.HERE/'analyze.py'),
                    scientific_sources=b.source_identity())
    b.json_write(b.HERE/'data/validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=4)
    main(parser.parse_args().threads)
