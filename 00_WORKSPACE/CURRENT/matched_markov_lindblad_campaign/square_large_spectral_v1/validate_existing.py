"""Regression against saved full-matrix L20 spectra for both alphas."""
import argparse
from pathlib import Path
import json
import numpy as np
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment
from run_large import PROJECT, config, spectral, construct_blocks, solve_blocks, sha, atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    for alpha in (1,3):
        old = PROJECT/f'square_2Ny_v1/results/20260929T213855Z/alpha{alpha}_L020/spectral'
        receipt = json.loads((old/'completion.json').read_text())
        assert sha(old/'spectrum.npz') == receipt['result_sha256']
        for path, checksum in spectral.sources().items():
            assert receipt['sources'][path] == checksum
        model = spectral.make_model(config(alpha,20))
        idx, blocks, checks = construct_blocks(model)
        full, _ = spectral.construct_product(model, progress=False)
        reconstructed = np.zeros_like(full)
        backward_errors = []
        for indices, block in zip(idx, blocks):
            reconstructed[np.ix_(indices,indices)] = block
            ev, vectors = eig(block, right=True, left=False)
            backward_errors.append(float(np.max(np.linalg.norm(
                block @ vectors - vectors * ev[None,:], axis=0))))
        matrix_error = float(np.max(abs(full-reconstructed)))
        actual, dominant, diagnostics = solve_blocks(model,idx,blocks)
        with np.load(old/'spectrum.npz',allow_pickle=False) as data:
            expected = data['eigenvalues'].copy()
        costs = abs(actual[:,None]-expected[None,:])
        i,j = linear_sum_assignment(costs)
        error = float(costs[i,j].max())
        gap = float(-2*np.log(dominant['radius']))
        gap_error = abs(gap-receipt['diagnostics']['covariance_gap_raw'])
        worst = int(np.argmax(costs[i,j]))
        print('[comparison]', json.dumps(dict(alpha=alpha, spectrum_error=error, gap_error=gap_error,
            worst_pair=[str(actual[i[worst]]), str(expected[j[worst]])])), flush=True)
        # Forward matching of ALL eigenvalues is not a reliable gate for this
        # highly nonnormal product. At L20, near-zero eigenvalues shift ~1e-4
        # even though the matrices agree to roundoff and eigenpair residuals
        # are ~1e-15. Preserve that discrepancy explicitly. Validate the
        # operator, every block eigenpair, and the requested dominant gap.
        assert matrix_error < 1e-13 and max(backward_errors) < 1e-11 and gap_error < 1e-11
        rows.append(dict(alpha=alpha,size=20,spectrum_matching_error=error,gap_error=gap_error,
                         full_matrix_max_error=matrix_error,block_all_eigenpair_max_residuals=backward_errors,
                         full_spectrum_forward_match_1e9_passed=error < 1e-9,
                         reference_sha256=sha(old/'spectrum.npz'),checks=checks,blocks=diagnostics))
    atomic_json(args.output,dict(status='passed_operator_and_gap_checks',cases=rows,source_sha256=sha(__file__),
        caveat='Near-zero eigenvalues are forward-sensitive; full-spectrum forward matching is not claimed.'))
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
