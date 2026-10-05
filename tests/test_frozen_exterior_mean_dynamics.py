import numpy as np
import pytest
from src.fgtn.classA_U1FGTN import classA_U1FGTN
from src.fgtn.diagnostics.mean_lindblad import PerfectCorrectionLindblad


@pytest.fixture
def models():
    parent = classA_U1FGTN(8,4,DW=True,nshell=1,alpha_1=3,
                           alpha_2=30,dw_truncation=True,dw_interval=(2,5))
    child, active = parent.restrict_ow_dynamics_to_slab()
    return parent, child, active


def test_restricted_channel_matches_full_slab_only_product(models):
    parent, child, active = models
    g = np.eye(parent.Nlayer,dtype=complex)*.5
    # Independent full-space reference: apply only original slab-centered modes.
    for x in range(2,6):
        for y in range(parent.Ny):
            for name,eta in [('WF_Ap',0),('WF_Am',1),('WF_Bp',0),('WF_Bm',1)]:
                v = getattr(parent,name)[:,x,y]
                p = np.outer(v,v.conj()); q=np.eye(parent.Nlayer)-p
                g = q@g@q + eta*p
    result=child.run_markov_channel(cycles=1,init_mode='maxmix',sequence='raster_y',
        perfect_correction=True,decoh=True,G_history=False,save=False,progress=False)
    actual=(result['G_final']+np.eye(child.Nlayer))/2
    np.testing.assert_allclose(actual,g[np.ix_(active,active)],atol=3e-13)


def test_restricted_lindblad_matches_full_slab_generator(models):
    parent,child,active=models
    gen=PerfectCorrectionLindblad.from_canonical_model(child)
    rng=np.random.default_rng(71)
    a=rng.normal(size=(child.Nlayer,child.Nlayer))+1j*rng.normal(size=(child.Nlayer,child.Nlayer))
    g=(a+a.conj().T)/50+np.eye(child.Nlayer)*.5
    full=np.zeros((parent.Nlayer,parent.Nlayer),complex)
    full[np.ix_(active,active)]=g
    rhs=np.zeros_like(full)
    for name,eta in [('WF_Ap',0),('WF_Am',1),('WF_Bp',0),('WF_Bm',1)]:
        for x in range(2,6):
            for y in range(parent.Ny):
                v=getattr(parent,name)[:,x,y]; p=np.outer(v,v.conj())
                rhs += eta*p-(p@full+full@p)+p@full@p
    reduced=gen.dense_rhs(g,include_number_dephasing=True)
    np.testing.assert_allclose(reduced,rhs[np.ix_(active,active)],atol=3e-13)
    outside=np.setdiff1d(np.arange(parent.Nlayer),active)
    np.testing.assert_allclose(rhs[outside],0,atol=3e-13)
    np.testing.assert_allclose(rhs[:,outside],0,atol=3e-13)


def test_restriction_rejects_soft_walls():
    parent=classA_U1FGTN(8,4,DW=True,nshell=1,dw_truncation=False)
    with pytest.raises(ValueError,match='requires'):
        parent.restrict_ow_dynamics_to_slab()
