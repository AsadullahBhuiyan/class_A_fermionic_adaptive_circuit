"""Small, fast tests; real endpoint validation is performed by the study runner."""
import json
import numpy as np
import pytest
import study_core as C
import run_study as R
from report_study import stats,variance_budget


def test_geometry_and_periodic_sectors():
    assert C.geometry(30,30)[:2] == (8,22)
    assert list(C.geometry(30,30)[4]) == [2,3,4,5,6]
    tables=C.sectors(20,30,4)
    for y in range(30):
        indices=np.concatenate([t[y] for t in tables])
        assert len(indices)==len(set(indices))
        assert np.any(indices//(2*20)==(y-1)%30)
        assert len(indices)%2==0
    assert np.all(C.region_columns(20)['core2']==np.arange(8,13))


def test_frame_and_explicit_chern():
    rng=np.random.default_rng(71)
    frame=np.linalg.qr(rng.normal(size=(200,55))+1j*rng.normal(size=(200,55)))[0]
    p=(frame@frame.conj().T).T
    values=C.disk_chern(p,10,10,[2])
    for y in (0,4,9):
        np.testing.assert_allclose(values[0,y],C.frame_disk_check(frame,10,10,2,y),atol=1e-12)
        a,b,c=[t[y] for t in C.sectors(10,10,2)]
        forward=np.trace(p[np.ix_(c,a)]@p[np.ix_(a,b)]@p[np.ix_(b,c)])
        reverse=np.trace(p[np.ix_(a,c)]@p[np.ix_(c,b)]@p[np.ix_(b,a)])
        np.testing.assert_allclose(values[0,y],(12j*np.pi*(forward-reverse)).real,atol=1e-12)


def test_mismatch_counts_and_self():
    p0=np.diag([1.,1.,0.,0.]).astype(complex)
    for p,counts in [(p0,[0,0]),(np.diag([1.,0.,0.,0.]),[0,1]),(np.diag([1.,1.,1.,0.]),[1,0]),(np.diag([1.,0.,1.,0.]),[1,1])]:
        values,density=C.mismatch(p,p0)
        np.testing.assert_allclose(values[:2],counts,atol=1e-12)
        np.testing.assert_allclose(density.sum(),sum(counts),atol=1e-12)


def test_cutoff_degeneracy_not_silently_half_filled():
    selection,gap=C.spectral_selections(np.array([-2.,-.1,.1,2.]),2,1e-10)
    assert selection[0][0]=='half' and gap==.2
    selection,gap=C.spectral_selections(np.array([-2.,0.,0.,2.]),2,1e-10)
    assert [(n,len(i)) for n,i in selection]==[('below',1),('above',3)]


def test_variance_covariance_and_trajectory_sem():
    rng=np.random.default_rng(8)
    slab=rng.integers(0,15,(100,41));exterior=rng.integers(0,15,(100,1))
    total,vs,ve,cov=variance_budget(slab,exterior)
    np.testing.assert_allclose(total,(slab+exterior).var(axis=0,ddof=1))
    late=np.abs(slab-7)[:,21:41].mean(axis=1)
    mean,sem=stats(late)
    np.testing.assert_allclose(sem,late.std(ddof=1)/10)
    assert mean==late.mean()


def test_completion_skip_partial_corrupt_and_identity(tmp_path):
    path=tmp_path/'result.npz';identity={'a':1}
    assert not R.valid(path,identity)
    R.publish(path,{'x':np.arange(10)},identity)
    assert R.valid(path,identity) and not R.valid(path,{'a':2})
    path.with_suffix('.json').unlink()
    assert not R.valid(path,identity)
    R.publish(path,{'x':np.arange(10)},identity)
    with path.open('ab') as stream:stream.write(b'corrupted')
    assert not R.valid(path,identity)


def test_exterior_inference_rejects_entangled_input():
    nx=ny=20
    _,_,active,outside,_,_=C.geometry(nx,ny)
    frame=np.zeros((800,1),complex)
    frame[active[0],0]=frame[outside[0],0]=1/np.sqrt(2)
    with pytest.raises(ValueError,match='exterior not frozen'):
        C.endpoint(frame,nx,ny,{},[])


def test_product_endpoint_and_correlation():
    nx=ny=20;_,_,active,_,radii,_=C.geometry(nx,ny)
    frame=np.eye(800,dtype=complex)[:,::2]
    p=(frame@frame.conj().T).T
    ref=p[np.ix_(active,active)]
    meta=[{'key':'product'}];refs={'product':{'projector':ref}}
    result,_=C.endpoint(frame,nx,ny,refs,meta)
    np.testing.assert_allclose(result['chern_radius'],0,atol=1e-12)
    np.testing.assert_allclose(result['corr_y'],0,atol=1e-12)
    np.testing.assert_allclose(result['mismatch'],0,atol=1e-12)
    assert result['q_exterior']==180


def test_mismatch_not_evaluated_on_average_state():
    p0=np.diag([1.,0.]);p1=np.diag([0.,1.])
    actual=(np.sum(abs(p0-p0)**2)+np.sum(abs(p1-p0)**2))/2
    wrong=np.sum(abs((p0+p1)/2-p0)**2)
    assert actual==1 and wrong==.5


def test_correlation_pair_count_and_y_seam():
    u=np.zeros(800,complex)
    u[2*(10+20*19)]=u[2*10]=1/np.sqrt(2)
    cy,cx=C.correlations(np.outer(u,u.conj()),20,20)
    np.testing.assert_allclose(cy[C.REGIONS.index('core2'),0],.25/(5*20),atol=1e-15)
    u[:]=0;u[2*8]=u[2*9]=1/np.sqrt(2)
    cy,cx=C.correlations(np.outer(u,u.conj()),20,20)
    np.testing.assert_allclose(cx[1,0],.25/(4*20),atol=1e-15)
