from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis'))
import large_ny_correlator_data as d
import build_large_ny_correlator_review as review


def payload(ny=24, count=25, every=True):
    cycles=np.arange(2*ny+1) if every else np.array([2*ny])
    rng=np.random.default_rng(18)
    x=rng.random((count,len(cycles),20,ny//2+1))
    return dict(Nx=20,Ny=ny,alpha_1=1.,alpha_2=30.,nshell=1,dtype='complex128',
                sequence='raster_y',perfect_correction=True,dw_truncation=True,meas_slab_only=True,
                canonical_entry_point='classA_U1FGTN_gpu.run_markov_circuit',init_mode='default',
                construction='hard',cycles=cycles,ry_values=np.arange(ny//2+1),x_values=np.arange(20),
                dw_location=np.array([5,15]),wall_locations=np.array([5,15]),
                global_sample_indices=np.arange(count),x_resolved_square_correlator=x,
                xavg_square_correlator_vs_ry=x.mean(axis=2))


@pytest.mark.parametrize('ny,count,every',[(24,25,True),(40,5,False),(60,5,False)])
def test_extract_compact_endpoint(ny,count,every):
    class Guard(dict):
        def __getitem__(self,key):
            assert key not in ('occupied_frame','half_system_covariance','half_system_occupation_spectrum')
            return super().__getitem__(key)
    source=Guard(payload(ny,count,every))
    ids,avg,x=d.extract_endpoint(source,ny,count,every)
    np.testing.assert_array_equal(x,source['x_resolved_square_correlator'][:,-1])
    np.testing.assert_array_equal(avg,x.mean(axis=1))
    assert ids.size==count


@pytest.mark.parametrize('key,value',[('Ny',50),('perfect_correction',False),('dtype','complex64'),('sequence','random')])
def test_incompatible_contract_rejected(key,value):
    source=payload();source[key]=value
    with pytest.raises(ValueError):d.extract_endpoint(source,24,25,True)


def test_missing_duplicate_samples_and_bad_x_average():
    p=d.extract_endpoint(payload(),24,25,True)
    with pytest.raises(ValueError):d.assemble(24,'test',[p],[])
    with pytest.raises(ValueError):d.assemble(24,'test',[p]*4,[])
    source=payload();source['xavg_square_correlator_vs_ry']+=1e-4
    with pytest.raises(AssertionError):d.extract_endpoint(source,24,25,True)
    source=payload();source['cycles'][-1]-=1
    with pytest.raises(AssertionError):d.extract_endpoint(source,24,25,True)


def test_result_pair_checks(tmp_path):
    p=tmp_path/'result.npz';np.savez(p,x=np.arange(3))
    receipt=dict(status='complete',result_filename=p.name,result_bytes=p.stat().st_size,result_sha256=d.sha256(p))
    d.verify_pair(p,receipt)
    for field,value in [('status','pending'),('result_filename','other.npz'),('result_bytes',0),('result_sha256','bad')]:
        with pytest.raises(ValueError):d.verify_pair(p,dict(receipt,**{field:value}))


def test_trajectory_averages_are_before_fitting():
    source=payload();ids,avg,x=d.extract_endpoint(source,24,25,True)
    obs=d.observables(d.Endpoint(24,'test',ids,avg,x,[]))
    np.testing.assert_array_equal(obs['pair_left'],(x[:,5]+x[:,6])/2)
    np.testing.assert_array_equal(obs['pair_right'],(x[:,14]+x[:,15])/2)
    np.testing.assert_array_equal(obs['walls'],(x[:,5]+x[:,15])/2)
    np.testing.assert_allclose(obs['pairs'],(obs['pair_left']+obs['pair_right'])/2)


@pytest.mark.parametrize('beta',[1.2,2.,2.35])
def test_known_chord_exponent_and_independent_window(beta):
    ny=60;r=np.arange(31);curve=np.ones(31)
    curve[1:]=.12*d.prior.chord(ny,r[1:])**-beta
    for lo,hi in d.windows(ny).values():
        f=d.fit(curve,ny,lo,hi)
        assert f['valid']
        assert abs(f['beta']-beta)<2e-13
    f=d.fit(np.r_[curve,np.ones(30)*999],ny,2,60)
    assert abs(f['beta']-beta)<2e-13 and f['n_points']==29


def test_invalid_points_cutoff_and_spread():
    curve=np.array([1,0,-1,np.nan,1e-9,.2,.1,.05,.02])
    a=d.fit(curve,16,1,8);b=d.fit(curve,16,1,8,cutoff=1e-8)
    assert a['n_points']==5 and a['excluded_points']==3
    assert b['n_points']==4 and b['excluded_points']==4
    assert not d.fit(curve,16,1,6,cutoff=1e-8)['valid']
    assert d.distribution([None,float('nan')])['n']==0
    assert d.distribution([1,2,3])['sd']==1


@pytest.mark.parametrize('log_coordinates',[False,True])
def test_displayed_dashed_fit_is_unconstrained_mean_curve_fit(tmp_path,log_coordinates):
    ny=60
    chord=d.prior.chord(ny,np.arange(1,ny//2+1))
    curves=np.ones((2,ny//2+1))
    curves[0,1:]=.12*chord**-1.4
    curves[1,1:]=.4*chord**-3.1
    mean=curves.mean(axis=0)
    expected=d.fit(mean,ny,2,ny//4)
    plot=review.Review(tmp_path)
    fig,(ax,)=review.axes_grid()
    try:
        model=review.mean_curve_fit_overlay(plot,ax,mean,ny,'test',log_coordinates=log_coordinates)
        assert model==expected
        assert abs(model['beta']-2)>.1
        assert abs(model['beta']-np.mean([d.fit(c,ny,2,ny//4)['beta'] for c in curves]))>.1
        line=ax.lines[0]
        assert line.get_linestyle()=='--' and line.get_color()=='black'
        x,y=line.get_data()
        if not log_coordinates:x,y=np.log(x),np.log(y)
        np.testing.assert_allclose(y,expected['log_amplitude']-expected['beta']*x)
        assert {r['role'] for r in plot.current}=={'mean_curve_fit'}
        assert {r['fit_min'] for r in plot.current}=={2}
        assert {r['fit_max'] for r in plot.current}=={15}
        assert all(r['beta']==expected['beta'] for r in plot.current)
    finally:
        plot.pdf.savefig(fig);review.plt.close(fig);plot.pdf.close()


@pytest.mark.parametrize('log_coordinates',[False,True])
def test_gray_band_is_primary_window_envelope(log_coordinates):
    fig,(ax,)=review.axes_grid()
    try:
        actual=review.primary_fit_band(ax,d.SIZES,log_coordinates=log_coordinates)
        bounds=np.array([d.prior.chord(n,np.array([2,n//4])) for n in d.SIZES])
        if log_coordinates:bounds=np.log(bounds)
        np.testing.assert_allclose(actual,[bounds[:,0].min(),bounds[:,1].max()])
        assert len(ax.patches)==1
    finally:review.plt.close(fig)


def test_actual_preparation_and_old_regression():
    identity=d.preparation_identity()
    assert identity['config']['protocol']['filling_frac']==.5
    assert identity['config']['protocol']['init_mode']=='default'
    endpoint=d.load_endpoint(32)
    result=review.old_regression(endpoint,{})
    assert result['saved_wall_fits_reproduced']==300
    assert result['prior_xavg_fits_reproduced']==100


@pytest.mark.parametrize('alpha',[1.,3.])
def test_matched_alpha_comparison_loader(alpha):
    avg,x,ids,records=d.prior.load_new_size(28,alpha_1=alpha)
    assert x.shape==(100,20,15)
    np.testing.assert_array_equal(ids,np.arange(100))
    np.testing.assert_allclose(avg,x.mean(axis=1),rtol=2e-14,atol=1e-15)
    assert len(records)==8
    assert all(f'alpha1_{alpha:g}/nshell_1/' in r['path'] for r in records)


def test_static_correlator_matches_dense_covariance_estimator():
    ny,nx=8,4
    rng=np.random.default_rng(43)
    blocks=[]
    for _ in range(ny):
        q,_=np.linalg.qr(rng.normal(size=(2*nx,nx))+1j*rng.normal(size=(2*nx,nx)))
        blocks.append(q@q.conj().T)
    delta=np.fft.ifft(np.asarray(blocks),axis=0)
    hybrid=review.correlator_from_delta(delta)
    covariance=np.empty((ny*2*nx,ny*2*nx),dtype=complex)
    for y in range(ny):
        for yy in range(ny):
            covariance[y*2*nx:(y+1)*2*nx,yy*2*nx:(yy+1)*2*nx]=delta[(y-yy)%ny]
    dense=np.zeros_like(hybrid)
    for x in range(nx):
        for r in range(ny//2+1):
            for y in range(ny):
                rows=y*2*nx+2*x+np.arange(2)
                cols=((y+r)%ny)*2*nx+2*x+np.arange(2)
                dense[x,r]+=np.square(np.abs(covariance[np.ix_(rows,cols)])).sum()/(2*ny)
    np.testing.assert_allclose(hybrid,dense,rtol=2e-14,atol=1e-15)
    np.testing.assert_allclose(hybrid.mean(axis=0),dense.mean(axis=0),rtol=2e-14,atol=1e-15)
