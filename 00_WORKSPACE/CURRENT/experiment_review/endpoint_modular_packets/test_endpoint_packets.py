import importlib.util
from pathlib import Path

import numpy as np
from scipy.linalg import expm

spec = importlib.util.spec_from_file_location("packets", Path(__file__).with_name("analyze_endpoint_packets.py"))
packets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packets)


def test_exact_evolution_density_and_com():
    rng = np.random.default_rng(123)
    nx, length, start, walls = 6, 4, 2, (0, 3)
    n = 2 * nx * length
    v, _ = np.linalg.qr(rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)))
    energies = np.linspace(-3,3,n)
    h = (v*energies)@v.conj().T
    g = (v*np.tanh(-energies/2))@v.conj().T
    times = np.array([0., .2, .5, 1.])
    result = packets.evolve(g,nx=nx,length=length,walls=walls,y_start=start,
                            times=times,snapshot_times=times)
    for ti,t in enumerate(times):
        u=expm(-1j*h*t)
        for p,x in enumerate(walls):
            cols=[2*nx*start+2*x+o for o in (0,1)]
            density=(abs(u[:,cols])**2).sum(axis=1).reshape(length,nx,2).sum(axis=2)
            np.testing.assert_allclose(result['density'][p,ti],density,atol=1e-12)
            ycom=(density*np.arange(length)[:,None]).sum()/density.sum()
            np.testing.assert_allclose(result['dy_full'][p,ti],ycom-start,atol=1e-12)
            distance=abs(np.arange(nx)-x)
            local=density[:,np.minimum(distance,nx-distance)<=2]
            local_com=(local*np.arange(length)[:,None]).sum()/local.sum()
            np.testing.assert_allclose(result['dy_window'][p,ti],local_com-start,atol=1e-12)
    np.testing.assert_allclose(result['density'].sum(axis=(-2,-1)),2,atol=1e-12)
    np.testing.assert_allclose(result['dy_window'][:,0],0,atol=1e-12)
    assert result['charge_error']<1e-12


def test_capped_eigenvalues_and_stationary_packet():
    g=np.diag(np.tile([-1.,1.],16)).astype(complex)
    result=packets.evolve(g,nx=4,length=4,walls=(0,2),y_start=2,
                          times=np.array([0.,1.]),snapshot_times=np.array([0.,1.]))
    np.testing.assert_allclose(result['dy_window'],0,atol=1e-12)
    np.testing.assert_allclose(result['density'][:,0],result['density'][:,1],atol=1e-12)
    assert result['clipped_modes']==32


def test_observable_average_is_not_mean_generator_evolution():
    # H and -H average to zero, but both spread a localized packet.
    h=np.array([[0.,1.],[1.,0.]])
    rho=np.diag([1.,0.])
    result=sum(expm(-1j*s*h)@rho@expm(1j*s*h) for s in (-1,1))/2
    assert result[1,1].real>.7
    assert not np.allclose(result,rho)


def test_reused_spectrum_and_cutoff_against_direct_exponential():
    rng = np.random.default_rng(777)
    nx, length, n = 4, 4, 32
    vectors, _ = np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    vals = np.linspace(-1,1,n)
    times = np.array([0., .1, .2, 1.])
    source = [2*nx*2, 2*nx*2+1]
    densities = []
    for eps in (1e-8, 1e-10, 1e-12):
        result = packets.evolve_spectrum(vals,vectors,nx=nx,length=length,walls=(0,2),
                         y_start=2,times=times,snapshot_times=times,eps=eps)
        energies = -2*np.arctanh(np.clip(vals,-1+eps,1-eps))
        h = (vectors*energies)@vectors.conj().T
        for ti,t in enumerate(times):
            u = expm(-1j*h*t)
            density = (abs(u[:,source])**2).sum(axis=1).reshape(length,nx,2).sum(axis=2)
            np.testing.assert_allclose(result['density'][0,ti],density,rtol=0,atol=1e-12)
        densities.append(result['density'])
    assert np.max(abs(densities[0]-densities[-1])) > .01
