"""Local deterministic ground-state counterpart to bundle 23's R=4 estimator.

One half-filled ground state per geometry, ten distinct periodic y centers.
No circuit simulation, trajectory averaging, or sampling-error interpretation.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent
REPO = next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').is_file())
OBSERVER = ROOT.parent/'23_hard_wall_random_center_chern/random_center_observer.py'
sys.path.insert(0, str(REPO/'src/fgtn'))
sys.path.insert(0, str(OBSERVER.parent))
from classA_U1FGTN import classA_U1FGTN
from random_center_observer import center_choices, sector_table, batched_chern


def analyze(ny):
    started = time.perf_counter()
    nx, rank = 20, 20*ny
    model = classA_U1FGTN(Nx=nx, Ny=ny, DW=True, nshell=1,
                         filling_frac=.5, alpha_1=1., alpha_2=30.,
                         trial_orbitals='X', dw_truncation=True)
    model.construct_OW_projectors(nshell=1, DW=True, trial_orbitals='X', dw_truncation=True)
    assert tuple(model.DW_loc) == (5, 15)
    h = np.zeros((2*rank, 2*rank), dtype=np.complex128)
    for name, sign in [('WF_Ap', 1), ('WF_Bp', 1), ('WF_Am', -1), ('WF_Bm', -1)]:
        w = np.asarray(getattr(model, name), dtype=np.complex128).reshape(2*rank, -1)
        h += sign * (w @ w.conj().T)
    h = (h + h.conj().T) / 2
    energies, eigenvectors = eigh(h, driver='evd', check_finite=True)
    v = eigenvectors[:, :rank].copy()
    # Same occupied-frame ordering as the existing equilibrium covariance and
    # the campaign observer. Convert to spectral-projector ordering by transpose.
    gamma = (v @ v.conj().T).T
    ys = center_choices(2026092701, nx, ny, [0], 0, 10)
    tables = sector_table(nx, ny, 4.)
    values = batched_chern(torch.from_numpy(v)[None], [rank], ys, tables).numpy()[0]
    explicit = []
    for y0 in ys[0]:
        a, b, c = (table[y0] for table in tables)
        first = np.trace(gamma[np.ix_(c,a)] @ gamma[np.ix_(a,b)] @ gamma[np.ix_(b,c)])
        reverse = np.trace(gamma[np.ix_(a,c)] @ gamma[np.ix_(c,b)] @ gamma[np.ix_(b,a)])
        explicit.append((12j*np.pi*(first-reverse)).real)
    explicit = np.asarray(explicit)
    error = float(np.max(np.abs(values-explicit)))
    assert error < 1e-11
    gram = v.conj().T @ v
    orthogonality = float(np.max(np.abs(gram-np.eye(rank))))
    assert orthogonality < 1e-11
    # Translate BOTH matrix indices by one y cell. A nonzero residual would
    # flag a translation-breaking half-filled choice at a degenerate Fermi level.
    permutation = np.roll(np.arange(2*rank).reshape(ny,2*nx),1,axis=0).ravel()
    translation = float(np.max(np.abs(gamma[np.ix_(permutation,permutation)]-gamma)))
    summary = dict(nx=nx, ny=ny, nshell=1, radius=4., center_x=10.,
                   centers_y=ys[0].tolist(), chern_values=values.tolist(),
                   center_mean=float(values.mean()), center_min=float(values.min()),
                   center_max=float(values.max()), center_spread=float(np.ptp(values)),
                   absolute_deviation_from_one=float(abs(values.mean()-1)),
                   total_charge=rank, half_filling_gap=float(energies[rank]-energies[rank-1]),
                   occupied_max_energy=float(energies[rank-1]),
                   empty_min_energy=float(energies[rank]),
                   translation_residual=translation, orthogonality_residual=orthogonality,
                   frame_vs_explicit_max_error=error, elapsed_seconds=time.perf_counter()-started)
    data = dict(centers_y=ys[0], real_space_chern=values, explicit_chern=explicit,
                eigenvalues=energies, occupied_frame=v, rank=np.asarray(rank))
    return summary, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT/'results/random_center_chern_hard_nx20_ny20-30-40_nsh1_r4_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise RuntimeError('output directory is nonempty; use a new output directory')
    summaries, rows = [], []
    torch.set_num_threads(args.threads)
    np.random.seed(2026092701)
    with threadpool_limits(limits=args.threads):
        for ny in tqdm([20,30,40], desc='half-filled ground states', unit='geometry'):
            summary, arrays = analyze(ny)
            summaries.append(summary)
            np.savez_compressed(args.output/f'nx20_ny{ny}.npz', **arrays,
                                summary_json=np.asarray(json.dumps(summary)))
            for center, value in zip(summary['centers_y'], summary['chern_values']):
                rows.append(dict(nx=20, ny=ny, nshell=1, x0=10, y0=center, radius=4., chern=value))
            print(json.dumps(summary), flush=True)
    with (args.output/'centers.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [Path(__file__), OBSERVER, REPO/'src/fgtn/classA_U1FGTN.py',
               REPO/'src/fgtn/occupied_frame.py', ROOT/'run_flattened_ground_state_large_ny.py']
    provenance = dict(protocol='hard-wall OW flattened parent; lowest half of spectrum occupied',
                      alpha_1=1., alpha_2=30., nshell=1, trial_orbitals='X',
                      interfaces=[5,15], boundary_conditions='periodic', dtype='complex128',
                      root_seed=2026092701, center_selection='bundle23 sample=0 cycle=0; ten distinct y centers',
                      projector_convention='Gamma=(V V^dagger)^T',
                      uncertainty='none; ten centers of one deterministic ground state, not independent samples',
                      comparison_note='equilibrium reference, not Born-conditioned exterior preparation or circuit evolution',
                      source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      results=summaries,
                      products={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                                for p in sorted(args.output.iterdir())})
    (args.output/'summary.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
