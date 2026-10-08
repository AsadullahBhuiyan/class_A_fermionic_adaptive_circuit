"""Deterministic checks and two figures for the consolidated OW note.

No dynamics, no writes outside this directory. Numerical extrema are sampled,
not interval-certified. Run with OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8.
"""
from pathlib import Path
import ast
import csv
import hashlib
import importlib.util
import json
import os
import types

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'PROJECT_ADMIN/REPO_POLICY.md').is_file())
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'build/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from tqdm import tqdm

NATIVE = ROOT / 'technical_report/analyze_ow_truncation.py'
STYLE = ROOT / '00_WORKSPACE/CURRENT/manuscript_Overleaf/figures/new_figure/sources/manuscript_typography.py'
FIG = HERE / 'figures'
DATA = HERE / 'data'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native():
    # The existing renderer has two top-level output-directory mkdir calls.
    # Suppress ONLY those calls; retain its functions without copying/retyping.
    tree = ast.parse(NATIVE.read_text(), filename=str(NATIVE))
    removed = []
    nodes = []
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'mkdir'):
            removed.append(node.lineno)
        else:
            nodes.append(node)
    assert len(removed) == 2, 'Review native import side effects after source changes'
    tree.body = nodes
    module = types.ModuleType('ow_native_readonly')
    module.__file__ = str(NATIVE)
    exec(compile(tree, str(NATIVE), 'exec'), module.__dict__)
    return module, removed


def load_style():
    spec = importlib.util.spec_from_file_location('ow_note_typography', STYLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Configure this module instance for the standalone note, not the manuscript.
    # The shared source is unchanged; figures are included at exactly 6.5 inches.
    module.ROOT = FIG
    module.inclusion_width = lambda stem: 6.5
    return module


def adj(x):
    return x.conj().swapaxes(-1, -2)


def matrix(native, coeff, kx, ky, width):
    return np.stack([native.truncated_spinor(kx, ky, coeff, native.IDENTITY[:, j], width)
                     for j in range(2)], axis=-1)


def projector_from_f(f):
    values, vectors = np.linalg.eigh(f)
    v = vectors[..., :, 1]
    return v[..., :, None] * v.conj()[..., None, :], values


def winding_check(native, coeff, width, radius, points, gauge_index):
    def scalar(kx):
        u = native.local_frame(kx, 0., 1., native.IDENTITY[:, 0])
        v = native.truncated_spinor(kx, 0., coeff, native.TAU_A, width, normalize=True)
        value = np.vdot(u, v)
        assert abs(value.imag) < 1e-12
        return float(value.real)
    zero = brentq(scalar, .05*np.pi, .99*np.pi, xtol=2e-14)
    theta = np.linspace(0, 2*np.pi, points+1)
    kx, ky = zero + radius*np.cos(theta), radius*np.sin(theta)
    u = native.local_frame(kx, ky, 1., native.IDENTITY[:, gauge_index])
    v = native.truncated_spinor(kx, ky, coeff, native.TAU_A, width, normalize=True)
    a = np.einsum('...i,...i->...', u.conj(), v)
    increments = np.angle(a[1:]*a[:-1].conj())
    assert np.min(abs(a)) > 1e-7 and np.max(abs(increments)) < .1
    winding = float(increments.sum()/(2*np.pi))
    assert abs(winding+1) < 1e-10
    return dict(w=width, radius=radius, points=points, gauge=gauge_index,
                zero_kx_over_pi=zero/np.pi, zero_residual=abs(scalar(zero)),
                minimum_boundary_amplitude=float(abs(a).min()), winding=winding)


def fhs(v):
    ux = np.einsum('...i,...i->...', v.conj(), np.roll(v, -1, axis=0))
    uy = np.einsum('...i,...i->...', v.conj(), np.roll(v, -1, axis=1))
    assert min(abs(ux).min(), abs(uy).min()) > 1e-6
    ux /= abs(ux); uy /= abs(uy)
    flux = np.angle(ux*np.roll(uy, -1, axis=0)*np.roll(ux.conj(), -1, axis=1)*uy.conj())
    return float(flux.sum()/(2*np.pi))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG/'data').mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    native, removed = load_native()
    rows, scales, windings, fingerprints = [], [], [], []
    for grid in tqdm((1024, 2048), desc='Fourier convergence', unit='grid'):
        native.FOURIER_GRID = grid
        k = 2*np.pi*np.arange(grid)/grid
        native.FOURIER_KX, native.FOURIER_KY = np.meshgrid(k, k, indexing='ij')
        coeff, plus = native.projector_fourier_coefficients(1.)
        norm = native.total_weight(coeff, native.TAU_A)
        np.testing.assert_allclose(norm, .5, atol=1e-13, rtol=0)
        coordinate = np.fft.fftfreq(grid)*grid
        density = np.sum(abs(coeff @ native.TAU_A)**2, axis=-1)/norm
        rms2 = float(np.sum(density*(coordinate[:, None]**2+coordinate[None, :]**2)))
        centroid = np.array([np.sum(density*coordinate[:, None]),
                             np.sum(density*coordinate[None, :])])
        centered_rms = float(np.sqrt(rms2-np.dot(centroid, centroid)))
        weights = np.array([native.support_weight(coeff, native.TAU_A, w)/norm for w in range(9)])
        scales.append(dict(fourier_grid=grid, unnormalized_weight=norm,
                           rms_squared=rms2, rms=np.sqrt(rms2), centroid=centroid.tolist(),
                           centered_rms=centered_rms, retained_weights=weights.tolist()))
        fingerprint = np.array([coeff[x % grid, y % grid] for x in range(-8, 9) for y in range(-8, 9)])
        fingerprints.append(fingerprint)
        for size in tqdm((201, 401), desc=f'Bloch checks, Fourier {grid}', leave=False):
            k = 2*np.pi*np.arange(size)/size
            kx, ky = np.meshgrid(k, k, indexing='ij')
            p = native.band_projector(kx, ky, 1., -1)
            for w in (0, 1, 2):
                f = matrix(native, coeff, kx, ky, w)
                r, ev = projector_from_f(f)
                z = [native.support_weight(c, t, w) for c in (coeff, plus) for t in native.TRIAL_SPINORS]
                np.testing.assert_allclose(z, z[0], atol=1e-13, rtol=0)
                h_direct = np.zeros_like(p)
                for sign, c in ((-1, coeff), (1, plus)):
                    for trial in native.TRIAL_SPINORS:
                        v = native.truncated_spinor(kx, ky, c, trial, w, normalize=True)
                        h_direct += sign*v[..., :, None]*v.conj()[..., None, :]
                h = (native.IDENTITY-2*f)/z[0]
                identity_error = float(np.max(abs(h_direct-h)))
                herm_error = float(np.max(abs(f-adj(f))))
                idem_error = float(np.max(abs(r@r-r)))
                assert max(identity_error, herm_error, idem_error) < 1e-12
                delta = float(np.max(abs(np.linalg.eigvalsh(f-p))))
                overlap = np.einsum('...ij,...ji->...', p, r).real
                _, rv = np.linalg.eigh(h)
                chern = fhs(rv[..., :, 0])
                np.testing.assert_allclose(chern, 0 if w == 0 else 1, atol=1e-10)
                # Single compact mode and the auxiliary occupied band are different.
                mode = native.truncated_spinor(kx, ky, coeff, native.TAU_A, w)
                min_mode_norm = float(np.linalg.norm(mode, axis=-1).min())
                mode /= np.linalg.norm(mode, axis=-1)[..., None]
                compact_chern = fhs(mode)
                np.testing.assert_allclose(compact_chern, 0, atol=1e-10)
                assert w == 0 or (delta < .5 and ev[..., 0].max() < .5 < ev[..., 1].min())
                rows.append(dict(fourier_grid=grid, momentum_grid=size, w=w,
                    retained_weight=weights[w], Z=z[0], sampled_operator_error=delta,
                    sampled_min_overlap=float(overlap.min()),
                    sampled_min_abs_frame_energy=float(abs(np.linalg.eigvalsh(h)).min()),
                    auxiliary_fhs_chern=chern, compact_spinor_fhs_chern=compact_chern,
                    sampled_min_compact_spinor_norm=min_mode_norm,
                    frame_identity_residual=identity_error, hermiticity_residual=herm_error,
                    idempotency_residual=idem_error))
        for w in (0, 1, 2):
            for radius in (.01, .02, .04):
                for points in (360, 720):
                    for gauge in (0, 1):
                        windings.append(dict(fourier_grid=grid,
                            **winding_check(native, coeff, w, radius, points, gauge)))
        del plus, density

    coeff_difference = float(np.max(abs(fingerprints[0]-fingerprints[1])))
    assert coeff_difference < 1e-12
    for size in (201, 401):
        for w in (0, 1, 2):
            a, b = [r for r in rows if r['momentum_grid']==size and r['w']==w]
            for key in ('sampled_operator_error', 'sampled_min_overlap', 'sampled_min_abs_frame_energy'):
                assert abs(a[key]-b[key]) < 1e-10
    # Exact onsite algebra and the interpolating zero at Gamma.
    f0 = coeff[0, 0]
    d0 = float((f0[1, 1]-f0[0, 0]).real)
    sstar = 1/(1+d0)
    pgamma = native.band_projector(0., 0., 1., -1)
    crossing = native.IDENTITY-2*((1-sstar)*pgamma+sstar*f0)
    np.testing.assert_allclose(crossing, 0, atol=1e-13)
    # Check nonzero-overlap interpolation on deterministic test projectors.
    ptest = native.band_projector(.43, 1.02, 1., -1)
    rtest, _ = projector_from_f(matrix(native, coeff, .43, 1.02, 1))
    for s in (0., .2, .8, 1.):
        b = ptest+(1-s)*(native.IDENTITY-ptest)
        rpath = b@rtest@b
        rpath /= np.trace(rpath)
        np.testing.assert_allclose(rpath@rpath, rpath, atol=1e-13)
        if s == 1:
            np.testing.assert_allclose(rpath, ptest, atol=1e-13)

    with (DATA/'band_checks.csv').open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Both figures use the highest Fourier resolution, with all samples saved.
    cut_k = np.linspace(-np.pi, np.pi, 2001)
    pcut = native.band_projector(cut_k, 0., 1., -1)
    cuts=[]
    for w in (0, 1, 2, -1):
        v = (pcut@native.TAU_A)/np.sqrt(.5) if w == -1 else native.truncated_spinor(
            cut_k, 0., coeff, native.TAU_A, w, normalize=True)
        cuts.append(np.sqrt(np.maximum(np.einsum('...i,...ij,...j->...', v.conj(), pcut, v).real, 0)))
    interpolation=np.sort(np.unique(np.r_[np.linspace(0, 1, 251), sstar]))
    ik=2*np.pi*np.arange(401)/401
    ix,iy=np.meshgrid(ik,ik,indexing='ij')
    p=native.band_projector(ix,iy,1.,-1)
    gaps=[]
    for w in tqdm((0,1,2),desc='Auxiliary interpolation',unit='window'):
        f=matrix(native,coeff,ix,iy,w)
        # Traceless 2x2 Hermitian H: minimum |E|=sqrt(H00^2+|H01|^2).
        h0=native.IDENTITY-2*p;h1=native.IDENTITY-2*f
        vals=[]
        for s in interpolation:
            h=(1-s)*h0+s*h1
            vals.append(float(np.sqrt(h[...,0,0].real**2+abs(h[...,0,1])**2).min()))
        gaps.append(vals)
    map_k=np.linspace(-np.pi,np.pi,241)
    mx,my=np.meshgrid(map_k,map_k,indexing='ij')
    p=native.band_projector(mx,my,1.,-1)
    r,_=projector_from_f(matrix(native,coeff,mx,my,1))
    mismatch=1-np.einsum('...ij,...ji->...',p,r).real
    assert mismatch.min()>-1e-13
    np.savez_compressed(DATA/'figure_data.npz',cut_k=cut_k,cuts=np.array(cuts),widths=np.arange(9),
        discarded=1-np.array(scales[-1]['retained_weights']),interpolation=interpolation,
        min_abs_interpolation_energy=np.array(gaps),map_k=map_k,mismatch=mismatch)
    receipt=dict(alpha=1.,fourier_grids=[1024,2048],momentum_grids=[201,401],scales=scales,
        checks=rows,winding_checks=windings,coefficient_convergence_max_abs=coeff_difference,
        onsite=dict(d0=d0,s_crossing=sstar,crossing_residual=float(abs(crossing).max())),
        normalization='Each OW mode has unit real-space norm; no separate overlap normalization',
        new_circuit_simulations=False,continuum_bound_certified=False,
        native_source=str(NATIVE.relative_to(ROOT)),native_sha256=digest(NATIVE),
        suppressed_import_mkdir_lines=removed,script_sha256=digest(__file__),numpy_version=np.__version__)
    (DATA/'numerical_checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    # Produce a table directly from the verified rows rather than hand transcription.
    table=[]
    for row in rows:
        if row['fourier_grid']==2048 and row['momentum_grid']==401:
            w=row['w'];zero=next(r['zero_kx_over_pi'] for r in windings if r['w']==w and r['fourier_grid']==2048)
            table.append(f"{w} & {row['retained_weight']:.6f} & {zero:.6f} & "
                f"{row['sampled_operator_error']:.6f} & {row['sampled_min_overlap']:.6f} & "
                f"{row['sampled_min_abs_frame_energy']:.6f} \\\\")
    (DATA/'table_rows.tex').write_text('\n'.join(table)+'\n')
    render()
    print('Completed deterministic checks and two figures.',flush=True)


def render():
    style=load_style()
    style.configure_style({'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in',
                           'xtick.top':True,'ytick.right':True,'legend.frameon':False})
    a=np.load(DATA/'figure_data.npz')
    colors=['#D55E4A','#3A9D5D','#2878B5'];marks=['^','s','o'];lines=[':', '--', '-']
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.65))
    for i,w in enumerate((0,1,2)):
        axs[0].plot(a['cut_k']/np.pi,a['cuts'][i],color=colors[i],ls=lines[i],
                    marker=marks[i],markevery=160,ms=3,mfc='white',lw=1.1,label=rf'$w={w}$')
    axs[0].plot(a['cut_k']/np.pi,a['cuts'][3],color='.25',ls='-.',lw=1,label=r'$w=\infty$')
    axs[0].set(xlabel=r'$k_x/\pi$',ylabel=r'$|f^{(w)}_{A,-}(k_x,0)|$',xlim=(-1,1))
    axs[0].legend(ncol=1,handlelength=2.5,loc='upper right')
    axs[1].semilogy(a['widths'],a['discarded'],'o-',color=colors[2],ms=4,mfc='white',lw=1)
    axs[1].set(xlabel=r'$w=n_{\rm shell}$',ylabel=r'Discarded probability $1-p_w$',xticks=[0,2,4,6,8])
    save(fig,axs,'ow_mode_truncation',style)
    fig,axs=plt.subplots(1,2,figsize=(6.5,2.65))
    for i,w in enumerate((0,1,2)):
        axs[0].plot(a['interpolation'],a['min_abs_interpolation_energy'][i],color=colors[i],ls=lines[i],
                    marker=marks[i],markevery=25,ms=3,mfc='white',lw=1.1,label=rf'$w={w}$')
    axs[0].set(xlabel=r'Interpolation $s$',ylabel=r'$\min_{\boldsymbol k,n}|E_n[H_{w,s}]|$',xlim=(0,1),ylim=(-.03,1.05))
    axs[0].legend(loc='lower left',handlelength=2.8)
    im=axs[1].pcolormesh(a['map_k']/np.pi,a['map_k']/np.pi,np.maximum(a['mismatch'],0).T,
                          cmap='magma',shading='gouraud',rasterized=False,vmin=0,
                          edgecolors='none',antialiased=False)
    axs[1].set(xlabel=r'$k_x/\pi$',ylabel=r'$k_y/\pi$',xticks=[-1,0,1],yticks=[-1,0,1],aspect='equal')
    cb=fig.colorbar(im,ax=axs[1],pad=.025)
    # Gouraud mesh shading keeps the heatmap vector-native without tile seams.
    # Same-color edges remove the analogous seams in the colorbar.
    cb.solids.set_edgecolor('face')
    cb.solids.set_rasterized(False)
    cb.set_label(r'$1-\operatorname{tr}(P_-R_1)$')
    save(fig,axs,'auxiliary_band_stability',style)


def save(fig,axs,stem,style):
    for i,ax in enumerate(axs):
        ax.text(-.16,1.045,f'({chr(97+i)})',transform=ax.transAxes,ha='left',va='bottom')
    style.prepare_figure(fig,stem)
    fig.tight_layout(pad=.9,w_pad=2.)
    style.record_typography(fig,stem)
    fig.savefig(FIG/f'{stem}.pdf',dpi=300)
    fig.savefig(FIG/f'{stem}.png',dpi=300)
    style.verify_typography(stem)
    plt.close(fig)


if __name__=='__main__':
    main()
