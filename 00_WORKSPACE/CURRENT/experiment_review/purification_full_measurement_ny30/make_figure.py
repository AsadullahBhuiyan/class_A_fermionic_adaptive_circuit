"""New Ny30 full-measurement panels A/B/D; preserve the published old gap panel C."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator, FixedLocator, FuncFormatter
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
BUNDLES = HERE.parents[1] / 'final_production_new_designs'
DATA1 = BUNDLES / '21_hard_wall_full_measurement_purification/gpu_data/hard_wall_full_measurement_nx20_ny30_alpha1-3_s100_2ny_v1'
DATA3 = BUNDLES / '22_hard_wall_full_measurement_clipped/gpu_data/hard_wall_full_measurement_nx20_ny30_alpha3_s100_2ny_clipped_v2'
OLD = BUNDLES / '13_maxmix_manybody_lyapunov_4ny/analysis_outputs/purification_2ny_v1'
OUT = HERE / 'figures'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_sem(values):
    assert len(values) == 100
    return values.mean(0), values.std(0, ddof=1)/10


def entropy_power_fit(samples, start_cycle=1, end_cycle=60):
    """OLS on log(mean entropy), with full trajectory-covariance delta SEM."""
    cycles = np.arange(start_cycle, end_cycle + 1)
    values = np.asarray(samples, dtype=float)[:, cycles] / 30
    assert len(values) > 1 and np.isfinite(values).all() and (values > 0).all()
    mean = values.mean(axis=0)
    x = np.log(cycles / 30)
    design = np.column_stack((np.ones(len(x)), x))
    linear = np.linalg.pinv(design)
    beta = linear @ np.log(mean)
    # The same trajectory supplies every time point: keep temporal covariance.
    jacobian = linear / mean[None, :]
    influence = (values - mean) @ jacobian.T
    covariance = influence.T @ influence / (len(values) * (len(values) - 1))
    residual = np.log(mean) - design @ beta
    r2 = 1 - residual @ residual / np.sum((np.log(mean) - np.log(mean).mean())**2)
    return dict(start_cycle=int(start_cycle), end_cycle=int(end_cycle),
                normalized_start=start_cycle/30, normalized_end=end_cycle/30,
                point_count=len(cycles), amplitude=float(np.exp(beta[0])),
                amplitude_sem=float(np.exp(beta[0]) * np.sqrt(covariance[0, 0])),
                exponent=float(-beta[1]), exponent_sem=float(np.sqrt(covariance[1, 1])),
                log_space_r2=float(r2),
                max_relative_residual=float(np.max(abs(np.expm1(-residual)))))


def load_data(root, alpha):
    paths = sorted((root / f'results/hard/alpha1_{alpha}/Ny030').glob('*.npz'))
    assert len(paths) == 20
    ids, entropy, spatial, densities, records, mode_rows = [], [], [], [], [], []
    executed_sources = []
    for path in tqdm(paths, desc=f'Verify alpha1={alpha}', unit='shard'):
        receipt_path = path.with_suffix('.complete.json')
        receipt = json.loads(receipt_path.read_text())
        digest = sha(path)
        assert receipt['result_filename'] == path.name
        assert receipt['result_bytes'] == path.stat().st_size and receipt['result_sha256'] == digest
        for p, h in ((path, digest), (receipt_path, sha(receipt_path))):
            records.append(dict(path=str(p), sha256=h, bytes=p.stat().st_size))
        with np.load(path, allow_pickle=False) as z:
            config = json.loads(str(z['configuration_json']))
            canonical = json.dumps(config, sort_keys=True, separators=(',', ':'))
            assert hashlib.sha256(canonical.encode()).hexdigest() == str(z['configuration_hash']) == receipt['configuration_hash']
            source = json.loads(str(z['source_hashes_json']))
            assert source == receipt['source_hashes']
            if source not in executed_sources:
                executed_sources.append(source)
            assert int(z['Ny']) == 30 and int(z['Nx']) == 20 and float(z['alpha_1']) == alpha
            assert bool(z['dw_truncation']) and not bool(z['meas_slab_only'])
            assert config['init_mode'] == 'maxmix' and config['perfect_correction']
            assert config['sequence'] == 'raster_y' and config['dtype'] == 'complex128'
            assert config['nshell'] == 1 and config['alpha_2'] == 30
            np.testing.assert_array_equal(z['cycles'], np.arange(61))
            np.testing.assert_array_equal(z['sample_indices'], receipt['sample_indices'])
            ids.extend(z['sample_indices'].tolist())
            total = z['total_entropy']
            assert total.shape == (5, 61) and np.isfinite(total).all() and (total >= 0).all()
            nu = z['occupation_spectrum']
            assert nu.shape == (5, 61, 1200) and np.isfinite(nu).all()
            assert nu.min() >= -1e-9 and nu.max() <= 1+1e-9
            safe = np.clip(nu, 1e-12, 1-1e-12)
            recomputed = -(safe*np.log(safe)+(1-safe)*np.log1p(-safe)).sum(-1)
            np.testing.assert_allclose(recomputed, total, atol=1e-10, rtol=1e-10)
            entropy.append(total)
            if alpha == 1:
                contour = z['entropy_contour']
                assert contour.shape == (5, 61, 20, 30) and np.isfinite(contour).all()
                np.testing.assert_allclose(contour.sum((2,3)), total, atol=1e-10, rtol=1e-10)
                spatial.append(contour.sum(-1))
                density, vector = z['slow_mode_density'], z['slow_mode_vector']
                assert z['slow_mode_resolved'].all()
                assert np.all(z['slow_mode_min_abs_multiplicity'] == 1)
                np.testing.assert_allclose(density.sum((1,2)), 1, atol=1e-10)
                rebuilt = np.abs(vector.reshape(5,30,20,2))**2
                np.testing.assert_allclose(density, rebuilt.sum(-1).transpose(0,2,1), atol=1e-12)
                endpoint = nu[:, -1]
                cost = np.full(endpoint.shape, np.inf)
                valid = (endpoint > 1e-9) & (endpoint < 1-1e-9)
                cost[valid] = abs(np.log1p(-endpoint[valid])-np.log(endpoint[valid]))/120
                np.testing.assert_allclose(z['slow_mode_abs_rate'], cost.min(-1), atol=1e-10, rtol=1e-9)
                assert np.max(z['slow_mode_residual']) < 1e-9
                densities.append(density)
                for i, sample in enumerate(z['sample_indices']):
                    mode_rows.append(dict(sample_index=int(sample), signed_rate=float(z['slow_mode_signed_rate'][i]),
                                          occupation=float(z['slow_mode_occupation'][i]), residual=float(z['slow_mode_residual'][i])))
            else:
                assert config['covariance_spectral_clip'] and int(z['unclipped_prefix_cycles']) == 30
                provenance = json.loads(str(z['fork_provenance_json']))
                assert provenance == json.loads((root/'fork_provenance.json').read_text())
    np.testing.assert_array_equal(ids, np.arange(100))
    return dict(entropy=np.concatenate(entropy), spatial=np.concatenate(spatial) if spatial else None,
                densities=np.concatenate(densities) if densities else None, records=records,
                sources=executed_sources, modes=mode_rows)


def configure_style():
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['CMU Sans Serif', 'DejaVu Sans'],
        'mathtext.fontset':'cm', 'font.size':8, 'axes.labelsize':8, 'legend.fontsize':8,
        'xtick.labelsize':8, 'ytick.labelsize':8, 'axes.linewidth':.8,
        'xtick.direction':'in', 'ytick.direction':'in', 'xtick.top':True, 'ytick.right':True,
        'pdf.fonttype':42, 'ps.fonttype':42})


def plot(e1, e3, sx, density, gaps, fit, *, gap_time_label=r'$T=2N_y$',
         output_dir=None, stem='purification_ny30_full_measurement_4x1',
         raw_cycles=False, gap_loglog=False, heatmap_time_annotation=True):
    configure_style()
    fig, axes = plt.subplots(4,1,figsize=(3.375,6.8),layout='constrained')
    fig.get_layout_engine().set(h_pad=.025,w_pad=.025,hspace=.02)
    a,b,c,d = axes
    times = np.arange(61) if raw_cycles else np.arange(61)/30
    cycle_limits = (1, 60) if raw_cycles else (1/30, 2)
    cycle_label = r'cycle $t$' if raw_cycles else r'cycle $t/N_y$'
    positive = times > 0
    marker_indices = np.unique(np.rint(np.geomspace(1,60,10)).astype(int)-1).tolist()
    entropy_rows, spatial_rows = [], []
    for alpha, samples, color, marker, style in [(1,e1,'#d62728','^',':'),(3,e3,'#1f77b4','o','-')]:
        m,e = mean_sem(samples/30)
        a.fill_between(times[positive],np.maximum(m-e,1e-13)[positive],(m+e)[positive],color=color,alpha=.1,lw=0)
        a.plot(times[positive],m[positive],color=color,marker=marker,ls=style,lw=1.3,ms=3.5,mfc='white',mew=.8,
               markevery=marker_indices,label=rf'$\alpha_1={alpha}$')
        entropy_rows.extend(dict(alpha_1=alpha,Ny=30,cycle=t,normalized_cycle=t/30,mean=float(m[t]),sem=float(e[t])) for t in range(61))
    a.set(xscale='log',yscale='log',xlim=cycle_limits,xlabel=cycle_label,ylabel=r'$\langle S(t)\rangle/N_y$')
    a.legend(loc='lower left',bbox_to_anchor=(.02,.07),frameon=False)
    a.text(.98,.51,r'$N_y=30$',transform=a.transAxes,ha='right')
    region_labels = {5: 'left edge', 15: 'right edge', 2: 'left trivial bulk', 18: 'right trivial bulk'}
    for x,color,marker,style in [(5,'#1f77b4','o','-'),(15,'#1f77b4','s','--'),
                                (2,'#d62728','^',':'),(18,'#2ca02c','v','-.')]:
        m,e = mean_sem(sx[:,:,x]/30)
        b.fill_between(times[positive],np.maximum(m-e,1e-13)[positive],(m+e)[positive],color=color,alpha=.1,lw=0)
        b.plot(times[positive],m[positive],color=color,marker=marker,ls=style,lw=1.2,ms=3.1,mfc='white',mew=.8,
               markevery=marker_indices if x!=2 else [i for i in range(60) if i+1 in (2,4,7,12,20,34,50)],label=region_labels[x])
        spatial_rows.extend(dict(Ny=30,x=x,cycle=t,normalized_cycle=t/30,mean=float(m[t]),sem=float(e[t])) for t in range(61))
    b.set(xscale='log',yscale='log',xlim=cycle_limits,xlabel=cycle_label,ylabel=r'$\langle s_x(t)\rangle/N_y$')
    b.legend(loc='center right',bbox_to_anchor=(1,.45),frameon=False,handlelength=1.8)
    b.text(.22,.73,r'$\alpha_1=1$',transform=b.transAxes,ha='left',va='top')
    for ax in (a,b):
        ax.xaxis.set_major_locator(FixedLocator([1,3,10,30,60] if raw_cycles else [.05,.1,.3,1,2]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4))
        ax.yaxis.set_minor_formatter(NullFormatter())
    sizes=np.array([int(r['Ny']) for r in gaps])
    c.errorbar(sizes,[float(r['mean_gap']) for r in gaps],yerr=[float(r['sample_sem']) for r in gaps],
               color='#1f77b4',marker='o',ls='none',ms=4,mfc='white',mew=1,capsize=2,
               label=gap_time_label+r': mean $\pm$ SEM')
    dense=np.linspace(sizes.min(),sizes.max(),300)
    c.plot(dense,fit['amplitude']*dense**(-fit['exponent']),color='.25',ls='--',lw=1,
           label=rf"$A N_y^{{-z}},\ z={fit['exponent']:.2f}\pm{fit['exponent_sem']:.2f}$")
    c.set(xlabel=r'circumference $N_y$',ylabel=r'$\Delta$',ylim=(0,.065),xticks=sizes)
    if gap_loglog:
        values = np.array([float(r['mean_gap']) for r in gaps])
        errors = np.array([float(r['sample_sem']) for r in gaps])
        assert np.all(values-errors > 0)
        c.set(ylim=(float((values-errors).min())*.75, float((values+errors).max())*2),
              xscale='log', yscale='log', xlim=(sizes.min()*.95,sizes.max()*1.05))
        # Standard, sparse log-axis ticks, independent of sampled sizes.
        c.xaxis.set_major_locator(FixedLocator([20, 30, 40, 60]))
        c.xaxis.set_minor_locator(FixedLocator([25, 35, 45, 50, 55]))
        c.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
        c.xaxis.set_minor_formatter(NullFormatter())
        c.yaxis.set_major_locator(FixedLocator([.01, .02, .04, .08]))
        c.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:rf'${value*100:g}\!\times\!10^{{-2}}$'))
        c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1,10), numticks=100))
        c.yaxis.set_minor_formatter(NullFormatter())
        c.tick_params(which='major', length=4)
        c.tick_params(which='minor', length=2.5)
    c.legend(loc='upper right',frameon=False)
    image=d.imshow(density.T,origin='lower',interpolation='nearest',aspect='auto',
                   extent=(-.5,19.5,-.5,29.5),cmap='magma',vmin=0,vmax=float(density.max()))
    d.set(xlabel='$x$',ylabel='$y$',xticks=[0,5,10,15,19],yticks=[0,10,20,29])
    d.text(.5,.97,r'$N_y=30,\ T=2N_y$' if heatmap_time_annotation else r'$N_y=30$',
           color='white',ha='center',va='top',transform=d.transAxes)
    cb=fig.colorbar(image,cax=d.inset_axes([1.025,0,.035,1]))
    cb.locator=MaxNLocator(nbins=3); cb.update_ticks()
    cb.set_label(r'$\overline{p_{\min}(x,y)}$',labelpad=2)
    cb.ax.tick_params(labelsize=8,pad=2)
    for ax,label in zip(axes,'abcd'):
        ax.text(-.18,1.035,f'({label})',transform=ax.transAxes,fontsize=9)
    destination = OUT if output_dir is None else Path(output_dir)
    destination.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(destination/f'{stem}.{ext}',dpi=300)
    plt.close(fig)
    return entropy_rows,spatial_rows


def plot_entropy_fit(samples, fit):
    """Separate single-column fit diagnostic; no fit in the four-panel asset."""
    configure_style()
    fig, ax = plt.subplots(figsize=(3.375, 2.5), layout='constrained')
    times = np.arange(1, 61)/30
    mean, sem = mean_sem(samples[:, 1:]/30)
    markers = np.unique(np.rint(np.geomspace(1, 60, 13)).astype(int)-1).tolist()
    ax.fill_between(times, mean-sem, mean+sem, color='#d62728', alpha=.15, lw=0)
    ax.plot(times, mean, color='#d62728', marker='^', ls=':', lw=1.2,
            ms=3.5, mfc='white', mew=.8, markevery=markers,
            label=r'$\alpha_1=1,\ N_y=30$')
    dense = np.geomspace(fit['normalized_start'], fit['normalized_end'], 200)
    ax.plot(dense, fit['amplitude']*dense**(-fit['exponent']), color='.25',
            ls='--', lw=1, label=r'$A(t/N_y)^{-p}$')
    ax.set(xscale='log', yscale='log', xlim=(1/30, 2),
           xlabel=r'cycle $t/N_y$', ylabel=r'$\langle S(t)\rangle/N_y$')
    ax.xaxis.set_major_locator(FixedLocator([.05, .1, .3, 1, 2]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(loc='upper right', frameon=False)
    ax.text(.05, .08,
            rf"$p={fit['exponent']:.3f}\pm{fit['exponent_sem']:.3f}$"+'\n'+
            rf"$R^2_{{\log}}={fit['log_space_r2']:.4f}$; cycles {fit['start_cycle']}–{fit['end_cycle']}",
            transform=ax.transAxes, va='bottom')
    for ext in ('pdf', 'png'):
        fig.savefig(OUT/f'purification_ny30_alpha1_entropy_power_law.{ext}', dpi=300)
    plt.close(fig)


def update_entropy_fit_only():
    """Update the standalone diagnostic without rewriting the four-panel files."""
    protected = [OUT/f'purification_ny30_full_measurement_4x1.{ext}' for ext in ('pdf', 'png')]
    hashes_before = [sha(p) for p in protected]
    data = load_data(DATA1, 1)
    fits = [entropy_power_fit(data['entropy'], start) for start in (1,3,5,10,15,20,30)]
    selected = next(row for row in fits if row['start_cycle'] == 15)
    plot_entropy_fit(data['entropy'], selected)
    write_csv(HERE/'entropy_power_law_fits.csv', fits)
    manifest = json.loads((HERE/'analysis_manifest.json').read_text())
    manifest['entropy_power_fit'] = selected
    manifest['entropy_power_fit_sensitivity'] = fits
    manifest['entropy_fit_method'] = manifest['entropy_fit_method'].replace('cycles 1..60', 'cycles 15..60')
    manifest['script_sha256'] = sha(Path(__file__))
    for row in manifest['outputs']:
        row['sha256'] = sha(Path(row['path']))
    write_json(HERE/'analysis_manifest.json', manifest)
    assert [sha(p) for p in protected] == hashes_before
    print(json.dumps(selected, indent=2))
    print('Four-panel PDF and PNG unchanged (SHA-256 verified).')


def main():
    a1,a3 = load_data(DATA1,1),load_data(DATA3,3)
    with (OLD/'gap_summary.csv').open() as stream:
        gaps=list(csv.DictReader(stream))
    fit=json.loads((OLD/'analysis_summary.json').read_text())['fit']
    entropy_fits = [entropy_power_fit(a1['entropy'], start) for start in (1,3,5,10,15,20,30)]
    entropy_fit = next(row for row in entropy_fits if row['start_cycle'] == 15)
    write_csv(HERE/'entropy_power_law_fits.csv', entropy_fits)
    density,error=mean_sem(a1['densities'])
    np.testing.assert_allclose(density.sum(),1,atol=1e-10)
    er,sr=plot(a1['entropy'],a3['entropy'],a1['spatial'],density,gaps,fit)
    plot_entropy_fit(a1['entropy'], entropy_fit)
    write_csv(HERE/'total_entropy_curves.csv',er)
    write_csv(HERE/'spatial_entropy_curves.csv',sr)
    write_csv(HERE/'selected_minimum_modes.csv',a1['modes'])
    write_csv(HERE/'retained_gap_summary.csv',gaps)
    np.savez_compressed(HERE/'Ny030_slowest_mode_density.npz',mean=density,sem=error,
                        sample_densities=a1['densities'],sample_indices=np.arange(100))
    inputs=a1['records']+a3['records']+[dict(path=str(p),sha256=sha(p)) for p in [OLD/'gap_summary.csv',OLD/'analysis_summary.json']]
    outputs=[p for p in OUT.iterdir()]+[HERE/n for n in ('total_entropy_curves.csv','spatial_entropy_curves.csv','selected_minimum_modes.csv','retained_gap_summary.csv','Ny030_slowest_mode_density.npz','caption.tex','entropy_power_law_fits.csv','entropy_power_law_caption.tex')]
    write_json(HERE/'analysis_manifest.json',dict(schema='purification_full_measurement_ny30_v3',
        panels_abd='New campaigns 21 (alpha1=1) and 22 (alpha1=3 clipped continuation), Ny30, T60, full-system measurements',
        panel_c='Unchanged campaign-13 slab-only gap data and fit at T=2Ny; different protocol, not pooled',
        source_hashes_alpha1=a1['sources'],source_hashes_alpha3=a3['sources'],
        normalization='Total entropy/Ny; fixed-x contour summed over all y then divided by Ny',
        controls_x=[2,18],walls_x=[5,15],sample_count=100,uncertainty='Ordinary trajectory SEM; no bootstrap',
        heatmap='Mean of one unique minimum-|lambda| normalized mode density per trajectory at T60',
        entropy_floor='Stored entropy uses occupations clipped to [1e-12,1-1e-12]; near-zero tails are numerical estimator floors',
        alpha3_clipping='Unclipped prefix cycles 0..30; state clipped at handoff and at each subsequent cycle end',
        figure_size_inches=[3.375,6.8],panel_ab_axes='log-log; cycle zero omitted from display only',gap_fit=fit,inputs=inputs,
        entropy_power_fit=entropy_fit,entropy_power_fit_sensitivity=entropy_fits,
        entropy_fit_figure='Separate purification_ny30_alpha1_entropy_power_law.pdf/png, 3.375 x 2.5 inches; four-panel figure has no entropy-fit overlay',
        entropy_fit_method='Unweighted OLS of log ensemble-mean entropy/Ny versus log(t/Ny), cycles 15..60; delta-method sampling SEM using full within-trajectory temporal covariance, no residual-error SEM or bootstrap. Finite-window descriptive fit, not an asymptotic-exponent determination.',
        outputs=[dict(path=str(p),sha256=sha(p)) for p in outputs],script_sha256=sha(Path(__file__))))
    print(OUT)
    print('Endpoint entropy/Ny:',a1['entropy'][:,-1].mean()/30,a3['entropy'][:,-1].mean()/30)
    print('Alpha1=1 entropy power fit:', json.dumps(entropy_fit))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--entropy-fit-only', action='store_true')
    args = parser.parse_args()
    if args.entropy_fit_only:
        update_entropy_fit_only()
    else:
        main()
