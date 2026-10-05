"""Mean-subtracted square-system spectral gaps, retaining signs on a symlog axis."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    data_path = snapshot / 'gaps.csv'
    source_manifest = json.loads((snapshot / 'manifest.json').read_text())
    assert sha(data_path) == source_manifest['output_sha256']['gaps.csv']
    with data_path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['geometry'] == 'square']
    sizes = [20, 24, 28, 32, 36, 40, 44, 50]
    out = snapshot / ('mean_centered_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 9,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), sharey=True)
    centered_rows = []
    means = {}
    for ax, alpha, color, marker, style, panel in zip(
            axes, (3, 1), ('#c0392b', '#2468ad'), ('^', 'o'), (':', '-'), ('a', 'b')):
        selected = sorted((r for r in rows if int(r['alpha_1']) == alpha), key=lambda r: int(r['Ny']))
        assert [int(r['Ny']) for r in selected] == sizes
        assert all(r['status'] == 'resolved_positive' and r['Nx'] == r['Ny'] for r in selected)
        gaps = np.array([float(r['gap']) for r in selected])
        mean = float(gaps.mean())
        centered = gaps - mean
        assert abs(centered.sum()) < 1e-13
        means[str(alpha)] = mean
        for size, gap, delta in zip(sizes, gaps, centered):
            centered_rows.append(dict(alpha_1=alpha, Nx=size, Ny=size, gap=gap,
                                      mean_gap=mean, centered_gap=delta))
        ax.axhline(0, color='.5', ls='--', lw=.8)
        ax.plot(sizes, centered, color=color, marker=marker, ls=style,
                mfc='white', ms=4.5, lw=1)
        ax.set_yscale('symlog', linthresh=1e-3, linscale=1, base=10)
        ax.set(xlabel=r'$N_y=N_x$', xlim=(18.5, 51.5), ylim=(-.06, .06))
        ax.set_xticks(sizes)
        ax.set_yticks([-.01, -.001, 0, .001, .01])
        ax.tick_params(top=True, right=True)
        ax.text(-.12, 1.03, f'({panel})', transform=ax.transAxes)
        ax.text(.96, .95, rf'$\alpha_1={alpha}$', ha='right', va='top', transform=ax.transAxes)
    axes[0].set_ylabel(r'$\Delta_C-\langle\Delta_C\rangle_{N_y}$ (cycle$^{-1}$)')
    fig.tight_layout(pad=.8)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'centered_square_gaps.{ext}', dpi=300)
    plt.close(fig)
    with (out / 'centered_gaps.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(centered_rows[0]))
        writer.writeheader()
        writer.writerows(centered_rows)
    (out / 'caption.txt').write_text(
        'Square-system data only: exact covariance-channel spectral gaps minus the unweighted '
        'arithmetic mean across Ny=20,24,28,32,36,40,44,50, separately for each alpha. '
        'Left alpha_1=3; right alpha_1=1. Signed base-10 logarithmic (symlog) vertical scale, '
        'linear within +/-0.001 cycle^-1. No absolute value, fit, or asymptotic-gap subtraction. '
        'Lines connect measured sizes only; the zero crossing follows from mean subtraction. '
        'Underlying scientific contract: ' + (snapshot / 'caption.txt').read_text())
    manifest = dict(input_csv=str(data_path), input_sha256=sha(data_path),
                    source_sha256=sha(__file__), means=means,
                    scale='symlog', linthresh=1e-3,
                    output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), means=means), indent=2))


if __name__ == '__main__':
    main()
