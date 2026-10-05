"""Title-free deterministic density-correlation figures; no fitted exponent."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def configure_style():
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['CMU Sans Serif', 'DejaVu Sans'], 'mathtext.fontset': 'cm',
        'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True, 'pdf.fonttype': 42})


def draw_curves(ax, data, kind='local', log_time=True):
    tau = data['tau']
    for x, label, color, marker, style in [
            (5, 'left wall', '#1f77b4', 'o', '-'),
            (15, 'right wall', '#1f77b4', 's', '--'),
            (2, 'left trivial bulk', '#d62728', '^', ':'),
            (18, 'right trivial bulk', '#2ca02c', 'v', '-.')]:
        values = data[kind][:, x]
        good = np.isfinite(values) & (values > 0)
        if log_time:
            good &= tau > 0
        ax.plot(tau[good], values[good], color=color, marker=marker, ls=style,
                mfc='white', ms=3, lw=1, markevery=18, label=label)
    ax.set(xscale='log' if log_time else 'linear', yscale='log',
           ylim=(1e-12, 1),
           xlabel=r'imaginary time $\tau$',
           ylabel=r'$C_x^{\mathrm{local}}(\tau)$' if kind == 'local' else r'$C_x^{\mathrm{column}}(\tau)$')
    ax.legend(frameon=False, loc='best')


def plot_all(result, output):
    matplotlib.use('Agg')  # CLI rendering only; importing from Colab preserves its backend.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    with np.load(result, allow_pickle=False) as z:
        data = {key: z[key] for key in ('tau', 'x', 'local', 'column', 'local_normalized', 'column_normalized')}
    for stem, kind, log_time in [('local_loglog', 'local', True),
                                 ('local_semilog', 'local', False),
                                 ('column_loglog', 'column', True)]:
        fig, ax = plt.subplots(figsize=(3.375, 2.6), layout='constrained')
        draw_curves(ax, data, kind, log_time)
        for extension in ('pdf', 'png'):
            fig.savefig(output/f'{stem}.{extension}', dpi=300)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.375, 2.6), layout='constrained')
    # Colors below 1e-12 saturate, but the stored raw data are never floored.
    values = np.ma.masked_invalid(data['local_normalized'][1:])
    values = np.ma.masked_less_equal(values, 0)
    mesh = ax.pcolormesh(np.arange(20), data['tau'][1:], values,
                         shading='nearest', cmap='Blues', norm=LogNorm(1e-12, 1))
    ax.set(yscale='log', xlabel='$x$', ylabel=r'imaginary time $\tau$', xticks=[0,5,10,15,19])
    fig.colorbar(mesh, ax=ax, label=r'$C_x(\tau)/C_x(0)$', extend='min')
    for extension in ('pdf', 'png'):
        fig.savefig(output/f'local_x_tau.{extension}', dpi=300)
    plt.close(fig)
    with (output/'curves.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['tau', 'x', 'local', 'column_per_Ny', 'local_normalized', 'column_normalized'])
        for i, tau in enumerate(data['tau']):
            for x in data['x']:
                writer.writerow([tau, x]+[data[key][i, x] for key in
                                ('local', 'column', 'local_normalized', 'column_normalized')])
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    outputs = [dict(filename=p.name, bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted(output.iterdir()) if p.suffix in ('.pdf', '.png', '.csv')]
    (output/'figure_manifest.json').write_text(json.dumps(dict(
        result_filename=Path(result).name, result_sha256=digest(result),
        dimensions_inches=[3.375, 2.6], dpi=300, sampling_uncertainty=None,
        fits=None, normalized_color_range=[1e-12, 1], curve_display_y_range=[1e-12, 1],
        zero_handling='Undefined normalized curves are NaN; zero raw values omitted on log axes',
        outputs=outputs), indent=2)+'\n')
