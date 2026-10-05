"""Replot the complete fixed-width spectrum sweep as g_C=1-rho(A)^2."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    source = root / 'analysis/gaps.csv'
    original_manifest = source.parent / 'manifest.json'
    metadata = json.loads(original_manifest.read_text())
    assert sha(source) == metadata['output_sha256']['gaps.csv']
    with source.open() as handle:
        rows = list(csv.DictReader(handle))
    sizes = [20, 40, 60, 80, 100]
    assert len(rows) == 105
    converted = []
    for row in rows:
        radius, rate = float(row['rho']), float(row['gap'])
        gap = 1 - radius**2
        assert int(row['Nx']) == 20
        assert row['status'] == 'resolved_positive', row
        assert np.isfinite(gap) and 0 <= gap <= 1
        np.testing.assert_allclose(gap, -np.expm1(-rate), atol=1e-14, rtol=1e-14)
        converted.append(dict(alpha_1=float(row['alpha_1']), Nx=20, Ny=int(row['Ny']),
                              rho_A=radius, g_C=gap, decay_rate=rate, status=row['status']))
    out = root / 'multiplier_gap_alpha' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'CMU Sans Serif', 'font.size': 8,
                         'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, ax = plt.subplots(figsize=(3.375, 2.8))
    colors = ['#c0392b', '#23934c', '#2468ad', '#8e44ad', '#d17b0f']
    markers = ['^', 's', 'o', 'D', 'v']
    styles = [':', '--', '-', '-.', (0, (3, 1, 1, 1))]
    for size, color, marker, style in zip(sizes, colors, markers, styles):
        selected = sorted((r for r in converted if r['Ny'] == size), key=lambda r: r['alpha_1'])
        alpha = [r['alpha_1'] for r in selected]
        np.testing.assert_allclose(alpha, np.linspace(1, 3, 21), atol=1e-14, rtol=0)
        ax.plot(alpha, [r['g_C'] for r in selected], color=color, marker=marker,
                ls=style, mfc='white', ms=3, lw=.8, label=str(size))
    ax.set(xlabel=r'$\alpha_1$', ylabel=r'$g_C=1-\rho(A)^2$')
    ax.tick_params(top=True, right=True)
    ax.legend(title=r'$N_y$ ($N_x=20$)', frameon=False, loc='lower right')
    fig.tight_layout(pad=.7)
    for extension in ('png', 'pdf'):
        fig.savefig(out / f'multiplier_gap_vs_alpha.{extension}', dpi=300)
    plt.close(fig)
    with (out / 'data.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    (out / 'caption.txt').write_text(
        'Dimensionless channel multiplier gap g_C=1-rho(A)^2 versus alpha_1, '
        'at fixed Nx=20 and Ny=20,40,60,80,100. All 105 saved spectral points are '
        'included (21 alphas from 1 to 3 in steps of 0.1); lines guide the eye, '
        'not fits. This equals the charge-neutral and parity-even many-body channel '
        'gap for this exact perfect-reset, fixed-schedule protocol; see '
        'manuscript_Overleaf/notes/channel_gap_proof/channel_gap_proof.tex. '
        'It is not the unrestricted odd-sector gap or the logarithmic decay rate. '
        'Hard-wall support truncation, all slabs active, inclusive walls x=5,15, '
        'alpha_2=30, nshell=1, X trial orbitals, periodic boundaries, zero twist, '
        'raster-y Ap/Am/Bp/Bm, complex128, perfect correction and measurement '
        'dephasing. Both invariant hard-wall blocks included. Alpha_1=2 uses '
        'the canonical zero-Bloch-norm prescription without an offset. '
        'Direct eigenspectra: no trajectory sampling, initialization, cycle horizon, '
        'fits, or statistical uncertainty bars. Original decay-rate products preserved.\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        source_sha256=sha(__file__), definition='g_C=1-rho(A)^2', points=len(converted),
        input_sha256={str(p): sha(p) for p in (source, original_manifest)},
        figure_inches=[3.375, 2.8],
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
    print(json.dumps(dict(output=str(out), points=len(converted),
                          gap_range=[min(r['g_C'] for r in converted),
                                     max(r['g_C'] for r in converted)]), indent=2))


if __name__ == '__main__':
    main()
