"""Verify 600 endpoint trajectories and reproduce the two manuscript figures."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from endpoint import CONTOUR, ENTROPY, VARIANCE, KEYS, validate_endpoint
from run_campaign import HERE, read_config, read_schedule, shard_identity, shard_paths, source_identity
from scheduling import digest
from storage import atomic_json, sha, verified_pair

STYLES = {24: ('#D92725', '^'), 28: ('#F08050', '<'), 32: ('#8FC1E3', 'v'),
          40: ('#2CA02C', 's'), 50: ('#6B6B6B', 'D'), 60: ('#1F77B4', 'o')}


def load_cases(root, config, schedule):
    rows = {ny: [] for ny in config['Ny_values']}
    inputs = []
    allowed = set(KEYS) | {'sample_ids', 'ay_values', 'valid_dy_count', 'endpoint_cycle',
                         'origin_average_count', 'contour_coordinate', 'Nx', 'Ny'}
    for worker in schedule['workers']:
        for task in worker:
            for first, path in zip(range(task['first'], task['stop'], config['result_shard_size']),
                                   shard_paths(root, task, config['result_shard_size'])):
                if not verified_pair(path, shard_identity(config, task, first)):
                    raise ValueError(f'Unverified or missing result: {path}')
                with np.load(path, allow_pickle=False) as archive:
                    payload = {key: archive[key].copy() for key in archive.files}
                if set(payload) != allowed:
                    raise ValueError('Result contains missing/unrequested observables')
                np.testing.assert_array_equal(payload['sample_ids'], np.arange(first, first+config['result_shard_size']))
                np.testing.assert_array_equal(payload['ay_values'], np.arange(task['ny']//2+1))
                np.testing.assert_array_equal(payload['valid_dy_count'], payload['ay_values'])
                if (int(payload['Nx']) != config['Nx'] or int(payload['Ny']) != task['ny']
                        or int(payload['endpoint_cycle']) != task['cycles']
                        or int(payload['origin_average_count']) != task['ny']
                        or str(payload['contour_coordinate']) != 'relative_dy=(y-y0)_mod_Ny'):
                    raise ValueError('Result geometry/endpoint metadata differs')
                validate_endpoint({key: payload[key] for key in KEYS}, nx=config['Nx'], ny=task['ny'],
                    samples=config['result_shard_size'], closure_tolerance=config['closure_tolerance'])
                rows[task['ny']].append(payload)
                inputs.append(dict(filename=str(path.relative_to(root)), sha256=sha(path)))
    cases = {}
    for ny, shards in rows.items():
        ids = np.concatenate([r['sample_ids'] for r in shards])
        order = np.argsort(ids)
        np.testing.assert_array_equal(ids[order], np.arange(config['samples']))
        cases[ny] = {key: np.concatenate([r[key] for r in shards])[order] for key in KEYS}
    if len(inputs) != len(config['Ny_values'])*config['samples']//config['result_shard_size']:
        raise ValueError('Incorrect number of result shards')
    return cases, inputs


def joint_fit(curves, fit_min=8, factor=1.):
    numerator = denominator = total = 0.
    groups = {}
    for ny, values in curves.items():
        ay = np.arange(ny//2+1)
        delta = values-values[:, [-1]]
        selected = ay >= fit_min
        x = np.log(np.sin(np.pi*ay[1:]/ny))
        x = x-x[-1]
        fit_x = x[selected[1:]]
        samples = delta[:, selected]
        mean = samples.mean(0)
        weight = 1/fit_x.size
        numerator += weight*float(fit_x@mean)
        denominator += weight*float(fit_x@fit_x)
        total += weight*float(mean@mean)
        groups[ny] = dict(x=x, fit_x=fit_x, samples=samples, fit_mean=mean, weight=weight,
            ay=ay[1:], mean=delta[:, 1:].mean(0), sem=delta[:, 1:].std(0, ddof=1)/np.sqrt(len(values)),
            raw_mean=values.mean(0), raw_sem=values.std(0, ddof=1)/np.sqrt(len(values)))
    slope = numerator/denominator
    variance = residual = 0.
    for group in groups.values():
        x, samples = group['fit_x'], group['samples']
        projection = group['weight']*x/denominator
        # This linear functional includes all covariance between widths and
        # the shared half-strip anchor without treating origins as samples.
        contributions = samples@projection
        variance += contributions.var(ddof=1)/len(samples)
        residual += group['weight']*float(np.sum((group['fit_mean']-slope*x)**2))
    summary = dict(slope=slope, slope_covariance_SEM=float(np.sqrt(variance)),
        coefficient=factor*slope, coefficient_SEM=factor*float(np.sqrt(variance)),
        R0_squared=1-residual/total, prefactor=factor, fit_min_Ay=fit_min,
        fit_max_Ay='Ny/2', weighting='equal_total_weight_per_size',
        uncertainty='trajectory_SEM_with_full_within_trajectory_width_covariance')
    return groups, summary


def typography(output):
    spec = importlib.util.spec_from_file_location('endpoint_manuscript_typography', HERE/'manuscript_typography.py')
    style = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(style)
    # Use the exact shared typography implementation with a standalone output
    # destination and the original figures' recorded manuscript column width.
    style.ROOT = output
    style.inclusion_width = lambda stem: style.COLUMN_INCHES
    (output/'data/typography').mkdir(parents=True, exist_ok=True)
    style.configure_style({'axes.spines.top': True, 'axes.spines.right': True,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'legend.frameon': False})
    return style


def wall_inset(ax, config, component):
    inset = ax.inset_axes([.07, .46, .17, .28])
    nx = config['Nx']
    for x in range(nx):
        inset.add_patch(Rectangle((x, 0), 1, 8, facecolor='.92', edgecolor='.7', linewidth=.15))
    for x in config['wall_windows'][component]:
        inset.add_patch(Rectangle((x, 0), 1, 8, facecolor='#8FC1E3', alpha=.65, edgecolor='none'))
    for x, label in ((5, r'$x_{\rm L}$'), (16, r'$x_{\rm R}$')):
        inset.plot([x, x], [0, 8], '--', color='#D92725', linewidth=.5)
        inset.text(x, 8.2, label, color='#D92725', ha='center', va='bottom', fontsize=8)
    inset.set(xlim=(0, nx), ylim=(0, 9.7))
    inset.axis('off')


def render(output, config, fits):
    style = typography(output)
    products, font_reports = [], {}
    definitions = [
        ('Figure_06_entropy_charge', ('entropy', 'variance'), (r'$\Delta\overline{S}_1$', r'$\Delta\overline{F}_A$'), ('c_1', 'k')),
        ('Figure_09_wall_entropy', ('left', 'right'), (r'$\Delta\overline{S}_{\rm L}^{\rm wall}$', r'$\Delta\overline{S}_{\rm R}^{\rm wall}$'), (r'm_{\rm L}', r'm_{\rm R}')),
    ]
    for stem, components, labels, symbols in definitions:
        fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.6), sharex=True)
        for index, (ax, component, label, symbol) in enumerate(zip(axes, components, labels, symbols)):
            groups, fit = fits[component]
            ax.axvspan(min(g['fit_x'].min() for g in groups.values()), 0, color='.5', alpha=.15, linewidth=0)
            for ny in config['Ny_values']:
                group = groups[ny]
                color, marker = STYLES[ny]
                keep = group['ay'] >= 2
                ax.errorbar(group['x'][keep], group['mean'][keep], yerr=group['sem'][keep],
                    color=color, marker=marker, linestyle='none', markerfacecolor='white',
                    markeredgewidth=.8, markersize=3.4, elinewidth=.45, capsize=0, label=rf'$N_y={ny}$')
            minimum = min(g['x'][g['ay'] >= 2].min() for g in groups.values())
            xx = np.linspace(minimum, 0, 300)
            ax.plot(xx, fit['slope']*xx, 'k--', linewidth=.9)
            ax.set(xlim=(minimum-.06, .05), ylabel=label)
            ax.text(-.18, 1.055, f'({chr(97+index)})', transform=ax.transAxes, fontsize=9)
            if component in ('left', 'right'):
                name = 'left' if component == 'left' else 'right'
                cells = ','.join(map(str, config['wall_windows'][component]))
                annotation = rf'{name} wall ($x={cells}$)'+'\n'+rf'${symbol}={fit["slope"]:.5f}\pm{fit["slope_covariance_SEM"]:.5f}$'
                wall_inset(ax, config, component)
            else:
                annotation = rf'${symbol}={fit["coefficient"]:.4f}\pm{fit["coefficient_SEM"]:.4f}$'
            annotation += '\n'+rf'$R_0^2={fit["R0_squared"]:.6f}$'
            ax.text(.035, .96, annotation, transform=ax.transAxes, fontsize=8, va='top', linespacing=1.35)
            if component in ('entropy', 'variance'):
                coefficient = r'\frac{c_1}{3}' if component == 'entropy' else r'\frac{k}{\pi^2}'
                ax.plot([.055, .10], [.685, .685], 'k--', transform=ax.transAxes, linewidth=.9)
                ax.text(.115, .685, r'$'+coefficient+r'\log\frac{D(A_y)}{D(N_y/2)}$',
                        transform=ax.transAxes, va='center', fontsize=8)
            else:
                ax.plot([.60, .65], [.425, .425], 'k--', transform=ax.transAxes, linewidth=.9)
                ax.text(.67, .425, r'$'+symbol+r'\log\frac{D(A_y)}{D(N_y/2)}$',
                        transform=ax.transAxes, va='center', fontsize=8)
        axes[0].legend(ncol=2, loc='lower right', fontsize=8, columnspacing=.7, handletextpad=.3)
        axes[1].set_xlabel(r'$\log[D(A_y)/D(N_y/2)]$')
        style.prepare_figure(fig, stem)
        fig.subplots_adjust(left=.19, right=.975, bottom=.105, top=.935, hspace=.26)
        style.record_typography(fig, stem)
        for suffix in ('.pdf', '.png'):
            path = output/(stem+suffix)
            fig.savefig(path, dpi=300)
            products.append(dict(filename=path.name, sha256=sha(path)))
        font_reports[stem] = style.verify_typography(stem)
        plt.close(fig)
    return products, font_reports


def analyze(root, output=None, plots=True):
    root = Path(root)
    output = Path(output) if output else root/'analysis'
    output.mkdir(parents=True, exist_ok=True)
    config = read_config(root)
    schedule = read_schedule(root, config)
    cases, inputs = load_cases(root, config, schedule)
    curves = {'entropy': {ny: v[ENTROPY] for ny, v in cases.items()},
              'variance': {ny: v[VARIANCE] for ny, v in cases.items()}}
    for wall, columns in config['wall_windows'].items():
        curves[wall] = {ny: v[CONTOUR][:, :, columns, :].sum((2, 3)) for ny, v in cases.items()}
    fits = {key: joint_fit(value, config['fit_min_Ay'], 3 if key == 'entropy' else np.pi**2 if key == 'variance' else 1)
            for key, value in curves.items()}
    summaries = {key: fit for key, (_, fit) in fits.items()}
    atomic_json(output/'fits.json', summaries)
    rows = []
    for component, (groups, fit) in fits.items():
        for ny, group in groups.items():
            for index, ay in enumerate(group['ay']):
                rows.append(dict(observable=component, Ny=ny, Ay=int(ay), samples=config['samples'],
                    x=float(group['x'][index]), anchored_mean=float(group['mean'][index]),
                    trajectory_SEM=float(group['sem'][index]), raw_mean=float(group['raw_mean'][ay]),
                    raw_SEM=float(group['raw_sem'][ay]), in_fit_window=bool(ay >= config['fit_min_Ay'])))
    with (output/'curves.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    products, fonts = render(output, config, fits) if plots else ([], {})
    report = dict(verified_shards=len(inputs), trajectories=len(cases)*config['samples'],
        config_sha256=digest(config), source_sha256=source_identity(), schedule_sha256=schedule['schedule_sha256'],
        Ny_values=config['Ny_values'], endpoint_cycles={str(ny): 2*ny for ny in config['Ny_values']},
        wall_windows=config['wall_windows'], inputs=inputs, fits=summaries, products=products, typography=fonts,
        estimator_order='per_strip_observables_then_origin_average_then_trajectory_anchor_then_ensemble_fit')
    atomic_json(output/'analysis_complete.json' if plots else output/'data_validation.json', report)
    print('[ANALYSIS COMPLETE]', json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    analyze(args.output_root, args.output_dir, plots=not args.data_only)
