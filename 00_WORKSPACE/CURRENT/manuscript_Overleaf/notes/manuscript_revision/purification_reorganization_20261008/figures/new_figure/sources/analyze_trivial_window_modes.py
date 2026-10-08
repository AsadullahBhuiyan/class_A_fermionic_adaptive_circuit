#!/usr/bin/env python3
"""Locate alpha_1=3 modes from saved frames at the fixed half-strip cut."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/entanglement_spectrum'
OUTPUT = ROOT/'data/energy_window_comparison'


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    repo = next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').is_file())
    provenance = json.loads((SOURCE/'occupation_provenance.json').read_text())
    with np.load(SOURCE/'occupation_inputs.npz') as data:
        reference = data['centered_occupations'][1]
    records = [r for r in provenance['source_files'] if r['alpha_1']==3]
    rows, distances, seen = [], np.minimum(np.arange(16), 15-np.arange(16)), []
    max_spectrum_error = 0.
    with threadpool_limits(limits=4), tqdm(total=100, desc='Locate saved alpha_1=3 modes', unit='trajectory') as bar:
        for record in records:
            path = repo/record['result']
            assert digest(path)==record['completion']['result_sha256']
            with np.load(path,allow_pickle=False) as data:
                assert data['alpha_1'].item()==3 and data['Ny'].item()==32 and data['cycles_total'].item()==64
                frames, ranks, sample_ids = data['final_frame'], data['final_ranks'], data['case_sample_indices']
            for frame, rank, sid in zip(frames,ranks,sample_ids):
                seen.append(int(sid))
                restricted = frame[:640,:int(rank)]
                q = 2*(restricted@restricted.conj().T)-np.eye(640)
                values,vectors = eigh((q+q.conj().T)/2,subset_by_value=(-.990000000001,.990000000001),driver='evr')
                select = abs(values)<=.99
                values,vectors = values[select],vectors[:,select]
                target = reference[sid,abs(reference[sid])<=.99]
                np.testing.assert_allclose(values,target,atol=1e-10,rtol=0)
                max_spectrum_error = max(max_spectrum_error,float(np.max(abs(values-target))))
                energies = np.log1p(-values)-np.log1p(values)
                weights = (abs(vectors)**2).reshape(16,20,2,-1).sum(axis=2)
                np.testing.assert_allclose(weights.sum(axis=(0,1)),1,atol=1e-12,rtol=0)
                by_y = weights.sum(axis=1)
                for j,(lam,energy) in enumerate(zip(values,energies)):
                    rows.append([int(sid),float(lam),float(energy),
                                 *[float(by_y[distances<n,j].sum()) for n in (1,2,3)],
                                 float((by_y[:,j]*distances).sum())])
                bar.update()
    assert sorted(seen)==list(range(100))
    rows = np.array(rows)
    assert rows.shape==(2705,7)
    summaries = {}
    for name,mask in [('all_retained',np.ones(len(rows),dtype=bool)),
                      ('abs_energy_at_least_4',abs(rows[:,2])>=4),
                      ('abs_energy_below_1',abs(rows[:,2])<1)]:
        selected = rows[mask]
        summaries[name] = dict(modes=len(selected), contributing_trajectories=len(np.unique(selected[:,0])),
                               mode_weighted_mean_probability_in_cut_rows_1_2_3=selected[:,3:6].mean(axis=0).tolist(),
                               minimum_probability_in_two_cut_rows=float(selected[:,4].min()),
                               mode_weighted_mean_distance_from_cut=float(selected[:,6].mean()))
    np.savetxt(OUTPUT/'alpha1_3_mode_localization.csv',rows,delimiter=',',
               header='sample_id,lambda,energy,weight_cut_layer_1,weight_cut_layer_2,weight_cut_layer_3,mean_y_distance_to_cut',comments='')
    checks = dict(source_sha256=digest(Path(__file__)), no_dynamics_rerun=True,
                  source_provenance_sha256=digest(SOURCE/'occupation_provenance.json'),
                  spectrum_max_error=max_spectrum_error,
                  geometry='Ny32, Ay16, fixed y0=0; cut rows y=0,15; distance=min(y,15-y)',
                  averaging='descriptive pooled average over retained modes, no independence or SEM assigned',
                  summaries=summaries)
    (OUTPUT/'spatial_mode_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
