"""Estimator-order, spatial closure, and unconstrained collapse regressions."""
from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1] / '00_WORKSPACE/CURRENT/experiment_review/domain_wall_correlator_scaling_analysis'
sys.path.insert(0, str(PROJECT))
import build_correlator_summary_3x1 as summary


def test_raw_separation_curve_preserves_values_and_cutoff():
    import matplotlib.pyplot as plt
    mean=np.array([1.,.1,.01,1e-9])
    fig,ax=plt.subplots()
    rows=[]
    summary.logcurve(ax,mean,6,'test','blue','o',rows=rows,raw_separation=True)
    line=ax.lines[0]
    np.testing.assert_array_equal(line.get_xdata(),[1,2,3])
    np.testing.assert_allclose(line.get_ydata()[:2],mean[1:3])
    assert np.isnan(line.get_ydata()[2])
    assert rows[-1]['unmasked_correlator']==1e-9
    assert not rows[-1]['displayed']
    plt.close(fig)


@pytest.mark.parametrize('beta', [1.3, 2., 2.7])
@pytest.mark.parametrize('upper', ['quarter', 'third', 'half'])
def test_chord_collapse_recovers_free_exponent(beta, upper):
    means = {}
    for n in summary.SIZES:
        c = np.ones(n//2+1)
        c[1:] = (n/17)**.31 * summary.data.prior.chord(n, np.arange(1,n//2+1))**(-beta)
        means[n] = c
    fit, curves = summary.anchored_fit(means, upper=upper)
    assert fit['beta'] == pytest.approx(beta, abs=1e-12)
    for r,x,y,keep in curves.values():
        assert x[-1] == y[-1] == 0
        assert keep.sum() >= 4


def test_exact_spatial_partition_and_averaging_order():
    x = np.random.default_rng(81).random((100,20,31))
    b,u,total = summary.partition_curves(x)
    np.testing.assert_allclose(b+u,total,rtol=1e-14)
    np.testing.assert_allclose((b+u).mean(0),x.mean(1).mean(0),rtol=1e-14)
    np.testing.assert_allclose(b, x[:,summary.BOUNDARY,:].mean(1)*4/20,rtol=1e-14)


def test_bad_anchor_and_too_short_window_rejected():
    c = np.ones(13); c[-1] = 0
    with pytest.raises(ValueError): summary.anchored_fit({24:c})
    with pytest.raises(ValueError): summary.anchored_fit({24:np.ones(13)},lower=5)


def test_measured_fit_not_forced_to_two():
    import json
    report = json.loads((summary.OUT/'summary.json').read_text())
    assert report['primary_fit']['beta'] == pytest.approx(2.1903330477862717,abs=1e-12)
    assert report['primary_fit']['lower'] == 8
    assert report['primary_fit']['upper'] == 'half'
    assert report['boundary_fraction']['30'] > .999


def test_window_scan_keeps_size_cohort_and_rejects_unknown_upper():
    means={n:np.ones(n//2+1) for n in summary.SIZES}
    scan=summary.window_scan(means)
    assert len(scan)==21
    for row in scan:
        assert set(row['points_per_size'])==set(summary.SIZES)
        assert row['valid']==(min(row['points_per_size'].values())>=4)
    invalid=next(row for row in scan if row['lower']==4 and row['upper']=='quarter')
    assert invalid['beta'] is None
    with pytest.raises(ValueError):summary.anchored_fit(means,upper='typo')


def test_shading_is_fit_union_and_common_interval():
    import matplotlib.pyplot as plt
    means={n:np.ones(n//2+1) for n in summary.SIZES}
    _,curves=summary.anchored_fit(means)
    fig,ax=plt.subplots()
    spans=summary.shade_fit_window(ax,curves)
    assert len(ax.patches)==2
    for r,x,y,k in curves.values():
        assert x[k].min()>=spans['union'][0]
        assert x[k].max()<=spans['union'][1]
        assert x[k].min()<=spans['intersection'][0]
        assert x[k].max()>=spans['intersection'][1]
    assert spans['union'][1]<0  # antipode not part of quarter regression
    plt.close(fig)


def test_density_wick_identity_including_contact_and_cell_sum():
    from itertools import combinations
    rng=np.random.default_rng(43)
    frame=np.linalg.qr(rng.normal(size=(4,2))+1j*rng.normal(size=(4,2)))[0]
    configs=list(combinations(range(4),2))
    weights=np.array([abs(np.linalg.det(frame[list(c),:]))**2 for c in configs])
    np.testing.assert_allclose(weights.sum(),1,atol=1e-14)
    occupations=np.array([[int(i in c) for i in range(4)] for c in configs])
    mean=weights@occupations
    connected=occupations.T@(weights[:,None]*occupations)-np.outer(mean,mean)
    C=frame@frame.conj().T
    np.testing.assert_allclose(connected,np.diag(mean)-np.abs(C)**2,atol=1e-14)
    # Two different cells, each with two orbitals; the observer has a factor 1/2.
    cell0=occupations[:,:2].sum(1);cell1=occupations[:,2:].sum(1)
    cell_connected=weights@(cell0*cell1)-(weights@cell0)*(weights@cell1)
    squared_per_orbital=np.sum(np.abs(C[:2,2:])**2)/2
    assert squared_per_orbital == pytest.approx(-cell_connected/2,abs=1e-14)


def test_trajectory_mixture_is_not_trajectory_connected_average():
    # Mixture of |10> and |01>: within-trajectory covariances vanish,
    # but covariance of their fluctuating mean occupations is -1/4.
    n=np.array([[1.,0.],[0.,1.]])
    mixture_connected=np.mean(n[:,0]*n[:,1])-n[:,0].mean()*n[:,1].mean()
    assert mixture_connected == -.25
    assert np.mean([0.,0.]) != mixture_connected
