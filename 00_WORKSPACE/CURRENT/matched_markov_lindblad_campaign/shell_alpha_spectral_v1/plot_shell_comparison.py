"""Replot verified saved data without changing the pinned simulation runner."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'analysis'
    provenance_path = out / 'provenance.json'
    provenance = json.loads(provenance_path.read_text())
    for relative, expected in provenance['input_sha256'].items():
        if sha(root / relative) != expected:
            raise ValueError(f'Saved scientific input changed: {relative}')
    if sha(out / 'gaps.csv') != provenance['output_sha256']['gaps.csv']:
        raise ValueError('Saved results table changed')
    with (out / 'gaps.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'text.usetex': True, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.6), sharex=True)
    styles = list(zip((20, 40, 60, 80, 100),
                      ('#c0392b', '#23934c', '#2468ad', '#8e44ad', '#d17b0f'),
                      ('^', 's', 'o', 'D', 'v'),
                      (':', '--', '-', '-.', (0, (3, 1, 1, 1)))))
    for panel, (ax, shell) in enumerate(zip(axes, ('1', '2', 'inf'))):
        for ny, color, marker, line in styles:
            selected = sorted((r for r in rows if r['nshell'] == shell and int(r['Ny']) == ny),
                              key=lambda r: float(r['alpha_1']))
            if len(selected) != 21:
                raise ValueError(f'Expected 21 points for shell={shell}, Ny={ny}')
            ax.plot([float(r['alpha_1']) for r in selected], [float(r['g_C']) for r in selected],
                    color=color, marker=marker, ls=line, mfc='white', ms=2.5, lw=.8, label=str(ny))
        ax.axvline(2, color='gray', ls='--', lw=.6)
        ax.tick_params(top=True, right=True)
        ax.set_ylabel(r'$g_C=1-\rho(A)^2$')
        label = r'\infty' if shell == 'inf' else shell
        ax.text(.03, .96, rf'$n_{{\mathrm{{shell}}}}={label}$', transform=ax.transAxes,
                fontsize=11, ha='left', va='top')
        ax.text(-.15, 1.02, f'({chr(97+panel)})', transform=ax.transAxes)
    axes[0].legend(title=r'$N_y$ ($N_x=20$)', ncol=2, frameon=False, fontsize=8,
                   loc='upper left', bbox_to_anchor=(.02, .83), borderaxespad=0)
    axes[-1].set_xlabel(r'$\alpha_1$')
    fig.tight_layout(pad=.8)
    for ext in ('pdf', 'png'):
        path = out / f'channel_gap_shell_comparison.{ext}'
        fig.savefig(path, dpi=300)
        provenance['output_sha256'][path.name] = sha(path)
    plt.close(fig)
    provenance['plot_revision'] = dict(renderer=str(Path(__file__).resolve()),
        renderer_sha256=sha(Path(__file__)), shell_label_position='upper left',
        shell_label_fontsize_pt=11, scientific_inputs_unchanged=True)
    temporary = provenance_path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(provenance, indent=2) + '\n')
    temporary.replace(provenance_path)
    print(out / 'channel_gap_shell_comparison.pdf')


if __name__ == '__main__':
    main()
