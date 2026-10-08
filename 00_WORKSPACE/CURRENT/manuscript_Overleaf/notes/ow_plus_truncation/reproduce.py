"""Deterministic OW square/plus comparison; no circuit evolution.

Run with python -B and OPENBLAS_NUM_THREADS=8. All outputs stay in this note.
Fourier integrals and sampled extrema are numerical, not interval certificates.
"""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'PROJECT_ADMIN/REPO_POLICY.md').is_file())
os.environ.setdefault('MPLCONFIGDIR', str(HERE / 'build/mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import brentq
from tqdm import tqdm

DATA = HERE / 'data'
FIG = HERE / 'figures'
PREVIOUS = HERE.parent / 'ow_truncation_consolidated/reproduce.py'


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supports():
    return {'onsite': [(0, 0)], 'plus': [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
            **{f'square{w}': [(x, y) for x in range(-w, w+1) for y in range(-w, w+1)]
               for w in (1, 2, 4)}}


def fourier(native, n, alpha=1.):
    native.FOURIER_GRID = n
    k = 2*np.pi*np.arange(n)/n
    native.FOURIER_KX, native.FOURIER_KY = np.meshgrid(k, k, indexing='ij')
    return native.projector_fourier_coefficients(alpha)


def window(coeff, points, kx, ky):
    result = np.zeros(np.broadcast_shapes(np.shape(kx), np.shape(ky))+(2, 2), complex)
    n = len(coeff)
    for x, y in points:
        result += np.exp(-1j*(kx*x+ky*y))[..., None, None]*coeff[x % n, y % n]
    return result


def weight(coeff, points, tau):
    n = len(coeff)
    return float(sum(np.vdot(coeff[x % n, y % n]@tau, coeff[x % n, y % n]@tau).real
                     for x, y in points))


def write_csv(name, rows):
    with (DATA/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def critical_points(n):
    """Gamma-mass zeros of the actual windowed projector, not bare QWZ mass."""
    k = 2*np.pi*np.arange(n)/n
    cx, cy = np.cos(k)[:, None], np.cos(k)[None, :]
    ss = np.sin(k)[:, None]**2 + np.sin(k)[None, :]**2
    def kernel(w):
        d = 1+2*sum(np.cos(j*k) for j in range(1, w+1))
        return d[:, None]*d[None, :]
    kernels = {'plus': 1+2*cx+2*cy, **{f'square{w}': kernel(w) for w in (1, 2, 4)}}
    rows = []
    for name, ker in kernels.items():
        def mass(alpha):
            z = alpha-cx-cy
            return float(np.mean(z/np.sqrt(ss+z*z)*ker))
        root = brentq(mass, 1.01, 1.999, xtol=2e-13)
        rows.append(dict(fourier_grid=n, support=name, alpha_c=root,
                         gamma_mass_residual=abs(mass(root))))
    return rows


def plus_coefficients(native, coeff):
    sigma = [native.SIGMA_X, native.SIGMA_Y, native.SIGMA_Z]
    d0 = np.array([-np.trace(coeff[0, 0]@s) for s in sigma])
    dx = np.array([-np.trace(coeff[1, 0]@s) for s in sigma])
    dxy = np.array([-np.trace(coeff[1, 1]@s) for s in sigma])
    return dict(a=float(2*dx[0].imag), b=float(d0[2].real),
                c=float(-2*dx[2].real), d=float(4*dxy[0].imag),
                e=float(4*dxy[2].real))


def harmonics(native, pars, kx, ky, lam=0.):
    a, b, c, d, e = (pars[k] for k in ('a', 'b', 'c', 'd', 'e'))
    return ((a+lam*d*np.cos(ky))*np.sin(kx))[..., None, None]*native.SIGMA_X + \
           ((a+lam*d*np.cos(kx))*np.sin(ky))[..., None, None]*native.SIGMA_Y + \
           (b-c*(np.cos(kx)+np.cos(ky))+lam*e*np.cos(kx)*np.cos(ky))[..., None, None]*native.SIGMA_Z


def winding(native, coeff, points, z):
    def overlap(kx, ky, gauge):
        u = native.local_frame(kx, ky, 1., native.IDENTITY[:, gauge])
        phi = (window(coeff, points, kx, ky)@native.TAU_A)/np.sqrt(z)
        return np.sum(u.conj()*phi, axis=-1)
    def scalar(kx):
        value = overlap(kx, 0., 0)
        assert abs(value.imag) < 1e-12
        return float(value.real)
    root = brentq(scalar, .05*np.pi, .99*np.pi, xtol=2e-14)
    rows = []
    for radius in (.01, .03):
        for size in (360, 720):
            for gauge in (0, 1):
                theta = np.linspace(0, 2*np.pi, size+1)
                ov = overlap(root+radius*np.cos(theta), radius*np.sin(theta), gauge)
                inc = np.angle(ov[1:]*ov[:-1].conj())
                wind = float(inc.sum()/(2*np.pi))
                assert abs(wind+1) < 1e-10 and abs(ov).min() > 1e-7
                assert abs(inc).max() < .1
                rows.append(dict(radius=radius, points=size, local_gauge=gauge,
                                 zero_kx_over_pi=root/np.pi, winding=wind,
                                 minimum_boundary_amplitude=float(abs(ov).min())))
    return rows


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    (FIG/'data').mkdir(parents=True, exist_ok=True)
    helper = module_at('previous_ow_readonly', PREVIOUS)
    native, removed = helper.load_native()
    masks = supports()
    rows, roots, winds, fingerprints, pars_rows = [], [], [], [], []
    for n in tqdm((1024, 2048), desc='Fourier convergence'):
        coeff, upper = fourier(native, n)
        pars = plus_coefficients(native, coeff)
        pars_rows.append(dict(fourier_grid=n, **pars))
        fingerprints.append(np.array([coeff[x % n, y % n] for x in range(-4, 5) for y in range(-4, 5)]))
        for size in (201, 401):
            k = 2*np.pi*np.arange(size)/size
            kx, ky = np.meshgrid(k, k, indexing='ij')
            p = native.band_projector(kx, ky, 1., -1)
            for name, points in masks.items():
                f = window(coeff, points, kx, ky)
                zs = [weight(c, points, t) for c in (coeff, upper) for t in native.TRIAL_SPINORS]
                np.testing.assert_allclose(zs, zs[0], atol=1e-13, rtol=0)
                z = zs[0]
                h = (native.IDENTITY-2*f)/z
                direct = np.zeros_like(h)
                leaks = []
                for sign, cc in ((-1, coeff), (1, upper)):
                    fw = window(cc, points, kx, ky)
                    for tau in native.TRIAL_SPINORS:
                        phi = (fw@tau)/np.sqrt(z)
                        outer = phi[..., :, None]*phi.conj()[..., None, :]
                        direct += sign*outer
                        opposite = native.IDENTITY-p if sign == -1 else p
                        leaks.append(float(np.mean(np.einsum('...i,...ij,...j->...', phi.conj(), opposite, phi).real)))
                np.testing.assert_allclose(leaks, leaks[0], atol=1e-13, rtol=0)
                np.testing.assert_allclose(direct, h, atol=1e-12, rtol=0)
                np.testing.assert_allclose(f, helper.adj(f), atol=1e-13, rtol=0)
                energies, vectors = np.linalg.eigh(h)
                v = vectors[..., :, 0]
                r = v[..., :, None]*v.conj()[..., None, :]
                np.testing.assert_allclose(r@r, r, atol=1e-13, rtol=0)
                chern = helper.fhs(v)
                np.testing.assert_allclose(chern, 0 if name == 'onsite' else 1, atol=1e-10)
                if name == 'plus':
                    np.testing.assert_allclose(native.IDENTITY-2*f, harmonics(native, pars, kx, ky), atol=1e-13, rtol=0)
                rows.append(dict(fourier_grid=n, momentum_grid=size, support=name, cells=len(points),
                                 Z=z, retained_weight=2*z, opposite_band_weight=leaks[0],
                                 sampled_uniform_error=float(abs(np.linalg.eigvalsh(f-p)).max()),
                                 sampled_min_target_overlap=float(np.einsum('...ij,...ji->...', p, r).real.min()),
                                 sampled_half_gap=float(abs(energies).min()),
                                 chern=chern, frame_identity_residual=float(abs(direct-h).max())))
        for name in ('onsite', 'plus', 'square1', 'square2', 'square4'):
            winds += [dict(fourier_grid=n, support=name, **row) for row in
                      winding(native, coeff, masks[name], weight(coeff, masks[name], native.TAU_A))]
        roots += critical_points(n)
    fourier_error = float(abs(fingerprints[1]-fingerprints[0]).max())
    assert fourier_error < 1e-13
    for name in ('plus', 'square1', 'square2', 'square4'):
        pair = [r['alpha_c'] for r in roots if r['support'] == name]
        assert abs(pair[0]-pair[1]) < 1e-9, (name, pair)
    a, b, c, d, e = (pars[key] for key in ('a', 'b', 'c', 'd', 'e'))
    zp = weight(coeff, masks['plus'], native.TAU_A)
    zsq = weight(coeff, masks['square1'], native.TAU_A)
    np.testing.assert_allclose(zp, (1+a*a+b*b+c*c)/4, atol=1e-13)
    assert a*a > c*c and 0 < b < 2*c
    analytic_gap = min(abs(b-2*c), abs(b), abs(b+2*c))/zp
    assert a > abs(d) and e < 0 and b+2*c+e > 0
    # Rotation covariance of the parent (not a claim about individual OW modes).
    u = np.diag(np.exp(-1j*np.pi*np.array([1, -1])/4))
    np.testing.assert_allclose(harmonics(native, pars, -.71, .37),
                               u@harmonics(native, pars, .37, .71)@u.conj().T, atol=1e-13)
    k = 2*np.pi*np.arange(401)/401
    kx, ky = np.meshgrid(k, k, indexing='ij')
    fplus = window(coeff, masks['plus'], kx, ky)
    fsq = window(coeff, masks['square1'], kx, ky)
    lam = np.linspace(0, 1, 101)
    gaps = []
    for t in tqdm(lam, desc='Corner-removal path'):
        z = zp+t*t*(zsq-zp)
        f = fplus+t*(fsq-fplus)
        h = (native.IDENTITY-2*f)/z
        np.testing.assert_allclose(z*h, harmonics(native, pars, kx, ky, t), atol=1e-13, rtol=0)
        gaps.append(float(abs(np.linalg.eigvalsh(h)).min()))
    np.testing.assert_allclose(gaps[0], analytic_gap, atol=1e-12)
    assert min(gaps) > .9
    # Removing the origin makes the two band families identical up to a sign.
    arms = [r for r in masks['plus'] if r != (0, 0)]
    np.testing.assert_allclose(window(coeff, arms, kx, ky), -window(upper, arms, kx, ky), atol=1e-13)
    # Exact 8x8 target and translated-frame singular values, via Bloch blocks.
    cc8, uu8 = fourier(native, 8)
    kk = 2*np.pi*np.arange(8)/8
    xx, yy = np.meshgrid(kk, kk, indexing='ij')
    rank_rows = []
    for name in ('plus', 'square1', 'square2', 'dense'):
        for band, cc in ((-1, cc8), (1, uu8)):
            if name == 'dense':
                f = native.band_projector(xx, yy, 1., band)/np.sqrt(.5)
            else:
                f = window(cc, masks[name], xx, yy)/np.sqrt(weight(cc, masks[name], native.TAU_A))
            sv = np.linalg.svd(f, compute_uv=False)
            rank_rows.append(dict(size=8, support=name, band=band,
                                  rank=int((sv > 1e-10).sum()), minimum_singular_value=float(sv.min())))
            assert int((sv > 1e-10).sum()) == (64 if name == 'dense' else 128)
    # Check actual topology on both sides of each reported Gamma closing.
    transition_checks = []
    for row in [r for r in roots if r['fourier_grid'] == 2048]:
        for offset in (-.02, .02):
            alpha = row['alpha_c']+offset
            cc, _ = fourier(native, 512, alpha)
            f = window(cc, masks[row['support']], kx, ky)
            ev, vv = np.linalg.eigh(native.IDENTITY-2*f)
            ch = helper.fhs(vv[..., :, 0])
            np.testing.assert_allclose(ch, 1 if offset < 0 else 0, atol=1e-10)
            transition_checks.append(dict(support=row['support'], alpha=alpha, chern=ch,
                                          sampled_min_abs_unnormalized_energy=float(abs(ev).min())))
    write_csv('band_checks.csv', rows)
    write_csv('critical_points.csv', roots)
    write_csv('winding_checks.csv', winds)
    write_csv('finite_lattice_rank.csv', rank_rows)
    write_csv('transition_checks.csv', transition_checks)
    write_csv('corner_path.csv', [dict(lambda_corners=t, sampled_half_gap=g) for t, g in zip(lam, gaps)])
    selected = [r for r in rows if r['fourier_grid'] == 2048 and r['momentum_grid'] == 401]
    labels = dict(onsite='Onsite', plus='Plus', square1=r'Square $w=1$', square2=r'Square $w=2$', square4=r'Square $w=4$')
    table = []
    for row in selected:
        zero = next(r['zero_kx_over_pi'] for r in winds if r['fourier_grid'] == 2048 and r['support'] == row['support'])
        table.append(f"{labels[row['support']]} & {row['cells']} & {row['retained_weight']:.6f} & "
                     f"{row['opposite_band_weight']:.3g} & {zero:.6f} & {row['sampled_half_gap']:.6f} & {round(row['chern'])} " + r'\\')
    (DATA/'table_rows.tex').write_text('\n'.join(table)+'\n')
    # Gauge-invariant absolute overlap cuts; physical real-space mode normalization.
    cutk = np.linspace(0, np.pi, 1601)
    pp = native.band_projector(cutk, cutk*0, 1., -1)
    cutdata = {}
    for name in ('onsite', 'plus', 'square1', 'square2'):
        phi = (window(coeff, masks[name], cutk, cutk*0)@native.TAU_A)/np.sqrt(weight(coeff, masks[name], native.TAU_A))
        cutdata[name] = np.sqrt(np.maximum(0, np.einsum('...i,...ij,...j->...', phi.conj(), pp, phi).real))
    ideal = (pp@native.TAU_A)/np.sqrt(.5)
    cutdata['dense'] = np.linalg.norm(ideal, axis=-1)
    np.savez_compressed(DATA/'plot_data.npz', kx_over_pi=cutk/np.pi, **cutdata,
                        lambda_corners=lam, half_gap=np.array(gaps))
    checks = dict(alpha=1., fourier_grids=[1024, 2048], momentum_grids=[201, 401],
                  suppressed_native_mkdir_lines=removed, fourier_coefficient_convergence=fourier_error,
                  plus_coefficients=pars_rows, plus_analytic_half_gap=analytic_gap,
                  plus_full_gap=2*analytic_gap, corner_path_min_half_gap=min(gaps),
                  geometric_minimality='C4-invariant submasks of the five-cell plus only',
                  units='Each real-space OW mode has norm one; Delta_0=min|E|, full gap=2 Delta_0',
                  source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (Path(__file__), PREVIOUS, helper.NATIVE, helper.STYLE)},
                  extrema_status='Finite-grid extrema are numerical, not interval-certified')
    (DATA/'numerical_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    render(helper, masks, cutk/np.pi, cutdata, lam, gaps)
    print(json.dumps(dict(plus=selected[1], critical_points=[r for r in roots if r['fourier_grid']==2048],
                          analytic_half_gap=analytic_gap), indent=2))


def savefig(style, fig, stem):
    style.prepare_figure(fig, stem)
    fig.tight_layout(pad=.8, w_pad=1.5,
                     rect=(0, 0, 1, .89) if stem == 'mixing_and_corner_removal' else (0, 0, 1, 1))
    style.record_typography(fig, stem)
    fig.savefig(FIG/f'{stem}.pdf', dpi=300)
    fig.savefig(FIG/f'{stem}.png', dpi=300)
    style.verify_typography(stem)
    plt.close(fig)


def render(helper, masks, k, cuts, lam, gaps):
    style = module_at('plus_typography', helper.STYLE)
    style.ROOT = FIG
    style.inclusion_width = lambda stem: 6.5
    style.configure_style({'axes.linewidth': .8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                           'xtick.top': True, 'ytick.right': True})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.2))
    titles = ('Nine-cell square', 'Five-cell plus', 'Onsite')
    for i, (ax, name, title) in enumerate(zip(axes, ('square1', 'plus', 'onsite'), titles)):
        for x in range(-1, 2):
            for y in range(-1, 2):
                active = (x, y) in masks[name]
                ax.add_patch(Rectangle((x-.48, y-.48), .96, .96,
                                      facecolor='#a8cee5' if active else '#f3f3f3',
                                      edgecolor='#254d65' if active else '#d8d8d8',
                                      linewidth=1 if active else .5))
        ax.plot(0, 0, 'o', color='#333333', ms=3)
        ax.set(aspect='equal', xlim=(-1.6, 1.6), ylim=(-1.7, 1.9))
        ax.set_axis_off()
        ax.text(.02, .98, f'({chr(97+i)})', transform=ax.transAxes, va='top')
        ax.text(0, -1.7, title, ha='center', va='top')
    savefig(style, fig, 'support_windows')
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.15))
    colors = ('#777777', '#D55E4A', '#2878B5', '#3A9D5D', '#202020')
    names = ('onsite', 'plus', 'square1', 'square2', 'dense')
    labels = ('Onsite', 'Plus', r'Square $w=1$', r'Square $w=2$', 'Untruncated')
    for color, name, label, line in zip(colors, names, labels, (':', '-', '--', '-.', (0, (5, 2)))):
        axes[0].plot(k, cuts[name], color=color, label=label, lw=1.2, ls=line)
    axes[0].set(xlabel=r'$k_x/\pi\quad(k_y=0)$', ylabel=r'$|\langle u_-|\phi_{A,-}\rangle|$', xlim=(0, 1))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc='upper center', ncol=5,
               handlelength=2.5, columnspacing=1, bbox_to_anchor=(.5, 1))
    axes[1].plot(lam, gaps, color='#2878B5', lw=1.6)
    axes[1].set(xlabel=r'Corner amplitude $\lambda$', ylabel=r'$\Delta_0=\min_{\boldsymbol{k},n}|E_n|$', xlim=(0, 1), ylim=(0, 2.05))
    axes[1].axhline(0, color='#777777', ls=':', lw=.7)
    axes[1].text(.03, .12, 'Plus', transform=axes[1].transAxes)
    axes[1].text(.97, .12, 'Square', transform=axes[1].transAxes, ha='right')
    for i, ax in enumerate(axes):
        ax.text(-.13, 1.035, f'({chr(97+i)})', transform=ax.transAxes, va='bottom')
    savefig(style, fig, 'mixing_and_corner_removal')


if __name__ == '__main__':
    main()
