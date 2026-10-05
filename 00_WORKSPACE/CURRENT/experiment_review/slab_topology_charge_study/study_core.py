"""CPU-only endpoint observables. No dynamics, averaging of states, or smoothing."""
from functools import lru_cache
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').exists())
sys.path.insert(0, str(REPO/'src/fgtn'))
from classA_U1FGTN import classA_U1FGTN

REGIONS = ('core1', 'core2', 'core3', 'left_wall', 'right_wall', 'exterior')


def geometry(nx, ny):
    left, right = nx//2-nx//4, nx//2+nx//4
    x = np.tile(np.arange(nx), ny)
    active = np.flatnonzero(np.repeat((x >= left) & (x <= right), 2))
    outside = np.setdiff1d(np.arange(2*nx*ny), active)
    distance = min(nx/2-left, right-nx/2)
    radii = np.arange(2, int(np.ceil(distance)))
    return left, right, active, outside, radii, distance


def region_columns(nx):
    left, right, *_ = geometry(nx, nx)
    result = {f'core{b}': np.arange(left+b+1, right-b) for b in (1,2,3)}
    result.update(left_wall=np.array([left]), right_wall=np.array([right]),
                  exterior=np.r_[np.arange(left), np.arange(right+1,nx)])
    assert all(len(v) for v in result.values())
    return result


@lru_cache(maxsize=None)
def sectors(nx, ny, radius):
    x, y = np.tile(np.arange(nx),ny), np.repeat(np.arange(ny),nx)
    tables = [[],[],[]]
    for y0 in range(ny):
        dx = (x-nx/2+nx/2) % nx-nx/2
        dy = (y-y0+ny/2) % ny-ny/2
        angle = np.mod(np.arctan2(dy,dx),2*np.pi)
        disk = dx*dx+dy*dy <= radius**2
        for k in range(3):
            cells = np.flatnonzero(disk & (angle >= k*2*np.pi/3) & (angle < (k+1)*2*np.pi/3))
            tables[k].append((2*cells[:,None]+[0,1]).ravel())
    return tuple(np.stack(t) for t in tables)


def disk_chern(p, nx, ny, radii):
    result = []
    for radius in radii:
        a,b,c = sectors(nx,ny,float(radius))
        ca = p[c[:,:,None],a[:,None,:]]
        ab = p[a[:,:,None],b[:,None,:]]
        bc = p[b[:,:,None],c[:,None,:]]
        result.append(-24*np.pi*np.einsum('yij,yji->y', ca @ ab, bc).imag)
    return np.stack(result)


def frame_disk_check(frame, nx, ny, radius, y0):
    a,b,c = [t[y0] for t in sectors(nx,ny,float(radius))]
    va,vb,vc = [frame[ix].conj() for ix in (a,b,c)]
    ca,ab,bc = vc@va.conj().T, va@vb.conj().T, vb@vc.conj().T
    return float(-24*np.pi*np.trace(ca@ab@bc).imag)


def pair_values(p, nx, x, y, xx, yy):
    a = 2*(np.asarray(x)+nx*np.asarray(y))
    b = 2*(np.asarray(xx)+nx*np.asarray(yy))
    return sum(np.abs(p[a+i,b+j])**2 for i in (0,1) for j in (0,1))


def correlations(p, nx, ny):
    columns = region_columns(nx)
    along = np.empty((len(REGIONS),ny//2))
    across = np.full((3,nx-1),np.nan)
    for k, name in enumerate(REGIONS):
        x,y = np.meshgrid(columns[name],np.arange(ny),indexing='ij')
        for r in range(1,ny//2+1):
            along[k,r-1] = pair_values(p,nx,x,y,x,(y+r)%ny).mean()
    for b in (1,2,3):
        cols = columns[f'core{b}']
        for r in range(1,nx):
            x0 = cols[np.isin(cols+r,cols)]
            if not len(x0): continue
            x,y = np.meshgrid(x0,np.arange(ny),indexing='ij')
            across[b-1,r-1] = pair_values(p,nx,x,y,x+r,y).mean()
    return along,across


def spectral_selections(energies, half_rank, tolerance):
    gap = float(energies[half_rank]-energies[half_rank-1])
    if gap > tolerance:
        return [('half',np.arange(half_rank))],gap
    cutoff = (energies[half_rank]+energies[half_rank-1])/2
    below = np.flatnonzero(energies < cutoff-tolerance)
    through = np.flatnonzero(energies <= cutoff+tolerance)
    return [('below',below),('above',through)],gap


def mismatch(p, ref):
    overlap = float(np.einsum('ij,ji->',p,ref).real)
    particles = float(np.trace(p).real-overlap)
    holes = float(np.trace(ref).real-overlap)
    density = np.sum(np.abs(p-ref)**2,axis=1)
    np.testing.assert_allclose(particles+holes,density.sum(),atol=2e-7,rtol=1e-9)
    np.testing.assert_allclose(particles-holes,np.trace(p-ref).real,atol=2e-9,rtol=0)
    assert min(particles,holes) > -2e-7
    return np.array([particles,holes,particles+holes,particles-holes]),density


def build_references(nx, ny, config):
    """Canonical CPU OW arrays, restricted AFTER construction on original torus."""
    left,right,active,outside,radii,distance = geometry(nx,ny)
    refs, metadata = {}, []
    for shell in config['reference_shells']:
        tag = 'dense' if shell is None else f'nsh{shell}'
        model = classA_U1FGTN(nx,ny,DW=True,nshell=shell,alpha_1=1,alpha_2=30,
                            trial_orbitals='X',dw_truncation=True)
        model.construct_OW_projectors(shell,True,'X',True)
        assert tuple(model.DW_loc) == (left,right)
        h = np.zeros((len(active),len(active)),np.complex128)
        max_leak = 0.
        for name,sign in (('WF_Ap',1),('WF_Bp',1),('WF_Am',-1),('WF_Bm',-1)):
            w = np.asarray(getattr(model,name))[:,left:right+1,:].reshape(2*nx*ny,-1)
            max_leak = max(max_leak,float(np.max(abs(w[outside]))))
            w = w[active]
            np.testing.assert_allclose(np.sum(abs(w)**2,axis=0),1,atol=1e-10,rtol=0)
            h += sign*(w@w.conj().T)
        assert max_leak < 1e-12
        energies,u = eigh((h+h.conj().T)/2,driver='evd')
        tol = config['reference_cutoff_relative_tolerance']*max(1,float(abs(energies).max()))
        selections,gap = spectral_selections(energies,len(active)//2,tol)
        for selection,indices in selections:
            key = f'{tag}_{selection}'
            frame = u[:,indices]
            p = (frame@frame.conj().T).T.copy()
            np.testing.assert_allclose(p@p,p,atol=1e-10,rtol=0)
            full = np.zeros((2*nx*ny,2*nx*ny),np.complex128)
            full[np.ix_(active,active)] = p
            cy,cx = correlations(full,nx,ny)
            refs[key] = dict(projector=p,chern=disk_chern(full,nx,ny,radii),corr_y=cy,corr_x=cx)
            metadata.append(dict(key=key,shell=shell,selection=selection,rank=len(indices),
                                 gap=gap,tolerance=tol,unique_half_filling=selection=='half',
                                 ow_exterior_leak=max_leak))
        refs[f'{tag}_spectrum'] = energies
        del model,h,u
    return refs,metadata


def endpoint(frame, nx, ny, refs, metadata, tolerance=1e-8):
    left,right,active,outside,radii,distance = geometry(nx,ny)
    rank = frame.shape[1]
    orth = float(np.max(abs(frame.conj().T@frame-np.eye(rank))))
    assert orth < tolerance
    p = (frame@frame.conj().T).T.copy()
    ext = p[np.ix_(outside,outside)]
    ext_diag = np.diag(ext).real
    product_error = float(np.max(abs(ext-np.diag(np.rint(ext_diag)))))
    cross_error = float(np.max(abs(p[np.ix_(active,outside)])))
    # Never infer a historical slab charge if the endpoint factorization fails.
    if max(product_error,cross_error) > tolerance:
        raise ValueError(f'exterior not frozen product/decoupled: {product_error}, {cross_error}')
    qext = int(np.rint(ext_diag).sum())
    ps = p[np.ix_(active,active)]
    qslab = float(np.trace(ps).real)
    np.testing.assert_allclose(qslab,rank-qext,atol=1e-7,rtol=0)
    cy,cx = correlations(p,nx,ny)
    chern = disk_chern(p,nx,ny,radii)
    measures,maps = [],[]
    for item in metadata:
        vals,density = mismatch(ps,refs[item['key']]['projector'])
        measures.append(vals)
        full_density = np.full(2*nx*ny,np.nan)
        full_density[active] = density
        maps.append(full_density.reshape(ny,nx,2).sum(axis=2).T)
    return dict(chern_radius=chern,corr_y=cy,corr_x=cx,mismatch=np.stack(measures),
                mismatch_maps=np.stack(maps),q_exterior=np.array(qext),
                diagnostics=np.array([orth,product_error,cross_error,abs(qslab-(rank-qext))])),p


def reference_controls(refs, metadata, nx, ny):
    """Finite-radius responses recorded, not incorrectly required to be integer."""
    _,_,active,_,radii,_ = geometry(nx,ny)
    item = metadata[0]
    p0 = refs[item['key']]['projector']
    vals,density = mismatch(p0,p0)
    np.testing.assert_allclose(vals,0,atol=2e-7,rtol=0)
    results = {'reference_self_mismatch':vals.tolist()}
    center = int(np.argmin(abs(active-(2*(nx//2+nx*(ny//2))))))
    for label,space,sign in [('hole',p0,-1),('particle',np.eye(len(p0))-p0,1)]:
        u = space[:,center]
        u = u/np.linalg.norm(u)
        modified = p0+sign*np.outer(u,u.conj())
        np.testing.assert_allclose(modified@modified,modified,atol=2e-9,rtol=0)
        v,d = mismatch(modified,p0)
        np.testing.assert_allclose(v[:2],[0,1] if sign<0 else [1,0],atol=2e-7,rtol=0)
        full = np.zeros((2*nx*ny,2*nx*ny),np.complex128)
        full[np.ix_(active,active)] = modified
        results[label] = dict(mismatch=v.tolist(),chern=disk_chern(full,nx,ny,radii).tolist())
    product = np.diag(np.tile([1.,0.],nx*ny)).astype(np.complex128)
    product_chern = disk_chern(product,nx,ny,radii)
    np.testing.assert_allclose(product_chern,0,atol=1e-13,rtol=0)
    results['product_chern_max'] = float(abs(product_chern).max())
    return results
