"""Offline sample-wise mean/SEM; saturation sentinels never enter statistics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run_campaign import default_config, tasks, identity, result_paths, pair_verified, sha
from endpoint_spectrum import spectral_products, validate_products


def validate_offline_products(products, cycles):
    """Allow libm roundoff, not changes in cap, tie, or eigenmode identities."""
    expected = spectral_products(products['centered_spectrum_raw'], cycles)
    checked = dict(products)
    for key, value in expected.items():
        if np.issubdtype(value.dtype, np.floating):
            np.testing.assert_allclose(products[key], value, rtol=8*np.finfo(float).eps,
                                       atol=0, equal_nan=False, err_msg=key)
        else:
            np.testing.assert_array_equal(products[key], value, err_msg=key)
        checked[key] = value
    for key, source in [('gap_mode_rate', 'lyapunov_rates'),
                        ('gap_mode_occupation', 'occupation_spectrum_raw')]:
        value = np.zeros(len(expected['gap_raw']))
        for i, valid in enumerate(expected['gap_mode_valid']):
            if valid:
                value[i] = expected[source][i, expected['gap_mode_index'][i]]
        np.testing.assert_allclose(products[key], value, rtol=8*np.finfo(float).eps,
                                   atol=0, equal_nan=False, err_msg=key)
        checked[key] = value
    # Run original shape, phase, normalization and residual checks on recomputed
    # products only after comparing every saved product. Never rewrite raw files.
    validate_products(checked, cycles)


def verify_offline_pair(output, task, start, ident):
    path, receipt = result_paths(output, task, start)
    ids = task.sample_ids(start, min(start+5, task.samples)).tolist()
    if not pair_verified(path, receipt, dict(ident, kind='result', sample_indices=ids)):
        raise ValueError(f'Checksum or scientific/source identity mismatch: {path}')
    with np.load(path, allow_pickle=False) as z:
        validate_offline_products(z, task.cycles)
        active = np.array([mu+2*x+2*task.Nx*y for y in range(task.Ny)
                           for x in range(task.Nx//4, 3*task.Nx//4+1) for mu in range(2)])
        np.testing.assert_array_equal(z['active_indices'], active)
        coords = np.column_stack(((active//2)%task.Nx, active//(2*task.Nx), active%2))
        np.testing.assert_array_equal(z['active_coordinates_x_y_orbital'], coords)
        np.testing.assert_array_equal(z['sample_indices'], ids)
        for key, value in [('T',task.cycles), ('Nx',task.Nx), ('Ny',task.Ny), ('alpha_1',task.alpha_1)]:
            np.testing.assert_equal(z[key], value)
        if z['occupation_spectrum_raw'].shape != (len(ids), task.active_modes):
            raise ValueError(f'Unexpected spectrum shape: {path}')


def summarize(alpha, raw, infinite):
    raw, infinite = np.asarray(raw), np.asarray(infinite, dtype=bool)
    if np.isnan(raw).any() or not np.array_equal(np.isposinf(raw), infinite):
        raise ValueError('Raw gaps and infinite flags disagree')
    finite = raw[~infinite]
    return dict(alpha_1=alpha, samples=len(raw), finite_samples=len(finite),
                infinite_samples=int(infinite.sum()), infinite_fraction=float(infinite.mean()),
                mean=float(finite.mean()) if len(finite) else None,
                sem=float(finite.std(ddof=1)/np.sqrt(len(finite))) if len(finite)>1 else None,
                statistic='full_ensemble' if not infinite.any() else 'finite_subset' if len(finite) else 'all_capped')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def analyze(output, destination, config=None):
    output, destination = Path(output), Path(destination)
    config = default_config() if config is None else config
    pair_count = sum((task.samples+4)//5 for task in tasks(config))
    download_manifest = output/'DOWNLOAD_MANIFEST.json'
    if download_manifest.exists():
        downloaded = json.loads(download_manifest.read_text())
        if len(downloaded['files']) != 2*pair_count or len({r['relative'] for r in downloaded['files']}) != 2*pair_count:
            raise ValueError(f'Expected {2*pair_count} unique downloaded files')
        for row in downloaded['files']:
            path = output/row['relative']
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError(f'Download manifest mismatch: {path}')
    sample_rows, summary, inputs = [], [], []
    for task in sorted(tasks(config), key=lambda t:t.alpha_1):
        ident = identity(task,config)
        raw, flags = [], []
        for start in range(0,task.samples,5):
            try:
                verify_offline_pair(output,task,start,ident)
            except (OSError, ValueError, KeyError, AssertionError) as exc:
                raise RuntimeError(f'Analysis requires all {pair_count} verified pairs; missing/invalid '
                                   +task.name+f' shard {start}: {exc}') from exc
            path, receipt = result_paths(output,task,start)
            inputs.extend(dict(path=str(p),sha256=sha(p)) for p in (path,receipt))
            with np.load(path,allow_pickle=False) as z:
                for i,s in enumerate(z['sample_indices']):
                    r, flag = float(z['gap_raw'][i]), bool(z['gap_is_infinite'][i])
                    raw.append(r); flags.append(flag)
                    sample_rows.append(dict(alpha_1=task.alpha_1,sample_index=int(s),seed=task.seed,
                                            gap_raw=r,gap_value=float(z['gap_value'][i]),gap_is_infinite=flag,
                                            gap_mode_valid=bool(z['gap_mode_valid'][i])))
        summary.append(summarize(task.alpha_1,raw,flags))
    destination.mkdir(parents=True,exist_ok=True)
    write_csv(destination/'sample_gaps.csv',sample_rows)
    write_csv(destination/'gap_summary.csv',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':8,'mathtext.fontset':'cm'})
    fig,ax=plt.subplots(figsize=(3.375,3.1))
    for category,marker,color,label in (
        ('full_ensemble','o','C0',f"Mean $\\pm$ SEM ({config['samples_per_case']} samples)"),
        ('finite_subset','s','C1','Finite subset only: mean $\\pm$ SEM')):
        rows=[r for r in summary if r['statistic']==category]
        if rows:
            ax.errorbar([r['alpha_1'] for r in rows],[r['mean'] for r in rows],
                        yerr=[r['sem'] or 0 for r in rows],fmt=marker,color=color,
                        mfc='white',capsize=2,ms=4,label=label)
    capped=[r for r in summary if r['infinite_samples']]
    if capped:
        ax.scatter([r['alpha_1'] for r in capped],[100]*len(capped),marker='^',facecolors='none',
                   edgecolors='C3',s=24,label='Capped: 100 placeholder')
    ax.set_yscale('symlog',linthresh=1e-4)
    ax.set(xlabel=r'$\alpha_1$',ylabel=r'$\Delta$',xlim=(.95,3.05))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both',direction='in',top=True,right=True)
    ax.legend(frameon=False,fontsize=6)
    fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(destination/f'gap_vs_alpha.{ext}',dpi=300)
    plt.close(fig)
    manifest=dict(configuration=config,summary=summary,inputs=inputs,
                  offline_reconstruction_rtol=8*np.finfo(float).eps,
                  offline_reconstruction_atol=0,
                  exact_checks=['file SHA256 and bytes','configuration and sample identities',
                                'approved source tuples','cap masks','tie masks','selected mode indices'],
                  uncertainty='sample SD / sqrt(finite sample count); no bootstrap',
                  gap_placeholder=100,placeholder_is_not_a_measurement=True,
                  caption='Hard-wall Nx=20, Ny=30, T=60, maximally mixed with canonical exterior preparation; '
                          f"{config['samples_per_case']} independent Born trajectories per alpha, perfect correction, slab-only raster-y. "
                          'The minimum absolute finite-time rate is computed per sample before averaging. '
                          'Error bars are ordinary SEM. Triangles at 100 indicate numerical saturation, not finite gaps. '
                          'Squares summarize only finite subsets. Symlog y axis retains exact zero gaps; no fit.')
    (destination/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--config',type=Path,help='Use campaign_config.json for add90; omitted means historical S10')
    args=parser.parse_args()
    analyze(args.output_root,args.destination or args.output_root/'analysis',
            None if args.config is None else json.loads(args.config.read_text()))
