"""GPU/NumPy endpoint parity and actual interrupted native-state continuation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from endpoint import CONTOUR, ENTROPY, VARIANCE, KEYS, averaged_width
from run_campaign import (gpu_contract, identity, read_config, run_task, shard_identity,
                          source_identity)
from storage import atomic_json, load_checkpoint
from scheduling import digest


def numpy_reference(frame, nx, ny, ay):
    """Independent dense CPU calculation, including the periodic wrap."""
    s = frame.shape[0]
    result = {CONTOUR: np.zeros((s, nx, ay)), ENTROPY: np.zeros(s), VARIANCE: np.zeros(s)}
    for sample in range(s):
        correlation = frame[sample] @ frame[sample].conj().T
        for origin in range(ny):
            indices = [2*((origin+dy)%ny*nx+x)+mu for dy in range(ay) for x in range(nx) for mu in range(2)]
            restricted = correlation[np.ix_(indices, indices)]
            nu, vectors = np.linalg.eigh((restricted+restricted.conj().T)*.5)
            nu = np.clip(nu, 0, 1)
            h = np.zeros_like(nu)
            mask = (nu > 0) & (nu < 1)
            h[mask] = -nu[mask]*np.log(nu[mask])-(1-nu[mask])*np.log1p(-nu[mask])
            result[ENTROPY][sample] += h.sum()/ny
            result[VARIANCE][sample] += (nu*(1-nu)).sum()/ny
            contour = (abs(vectors)**2 @ h).reshape(ay, nx, 2).sum(-1).T
            result[CONTOUR][sample] += contour/ny
    return result


def endpoint_parity(device):
    rng = np.random.default_rng(1933)
    raw = rng.normal(size=(3, 48, 21))+1j*rng.normal(size=(3, 48, 21))
    frame = np.linalg.qr(raw)[0].astype(np.complex128)
    maximum = 0.
    for ay in (1, 2, 3):
        reference = numpy_reference(frame, 4, 6, ay)
        for batch in (2, 64):
            measured, _ = averaged_width(torch.as_tensor(frame, device=device), nx=4, ny=6, ay=ay, matrix_batch=batch)
            for key in KEYS:
                error = float(abs(reference[key]-measured[key]).max(initial=0))
                maximum = max(maximum, error)
                np.testing.assert_allclose(measured[key], reference[key], rtol=0, atol=2e-8)
    return dict(maximum_absolute_error=maximum, widths=[1, 2, 3], matrix_batches=[2, 64])


def resume_pilot(root, config):
    pilot = dict(config, Nx=4, Ny_values=[6], samples=5)
    task = dict(task_id='resume_pilot', ny=6, samples=5, first=0, stop=5, cycles=6,
                seed=1933, matrix_batch_by_ay={str(ay): 8 for ay in range(4)})
    reference, resumed = Path(root)/'validation/reference', Path(root)/'validation/resumed'
    if not run_task(reference, pilot, task, segment_cycles=6, retain_checkpoints=True):
        raise RuntimeError('Uninterrupted pilot did not complete')
    # A repeated validation run reuses already verified products. Interrupted
    # phases are exercised only while their final shards are absent.
    result_path = resumed/'results/Ny006/samples_000-004.npz'
    if not result_path.exists():
        assert not run_task(resumed, pilot, task, max_segments=1, retain_checkpoints=True)
        assert not run_task(resumed, pilot, task, max_widths=2, retain_checkpoints=True)
        partial = load_checkpoint(resumed/'checkpoints/resume_pilot/endpoint_progress.npz', dict(identity(pilot, task),
            final_checkpoint_sha256=json.loads((resumed/'checkpoints/resume_pilot/checkpoint.json').read_text())['sha256']))
        np.testing.assert_array_equal(partial['seen_widths'], [True, True, False, False])
    assert run_task(resumed, pilot, task, retain_checkpoints=True)
    expected = identity(pilot, task)
    left = load_checkpoint(reference/'checkpoints/resume_pilot/checkpoint.npz', expected)
    right = load_checkpoint(resumed/'checkpoints/resume_pilot/checkpoint.npz', expected)
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=f'Exact dynamics resume: {key}')
    with np.load(reference/'results/Ny006/samples_000-004.npz') as left_values, np.load(result_path) as right_values:
        for key in KEYS:
            np.testing.assert_allclose(left_values[key], right_values[key], rtol=0, atol=2e-8)
    # Pilot states are temporary resume products, never production artifacts.
    for directory in (reference/'checkpoints/resume_pilot', resumed/'checkpoints/resume_pilot'):
        for name in ('checkpoint.npz', 'checkpoint.json', 'endpoint_progress.npz', 'endpoint_progress.json'):
            (directory/name).unlink(missing_ok=True)
    return dict(exact_native_frame_ranks_and_rng=True, endpoint_resume=True, resumed_after_cycle=5,
                resumed_after_width=1, endpoint_tolerance=2e-8)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.output_root)
    gpu = gpu_contract(config)
    path = args.output_root/'validation/gpu_complete.json'
    if path.exists():
        row = json.loads(path.read_text())
        if row.get('passed') and row['source_sha256'] == source_identity() and row['config_sha256'] == digest(config):
            print('[GPU VALIDATION verified]', flush=True)
            return
        raise ValueError('Preserving mismatched validation receipt')
    parity = endpoint_parity('cuda')
    resume = resume_pilot(args.output_root, config)
    atomic_json(path, dict(passed=True, gpu=gpu, endpoint_parity=parity, resume=resume,
                          source_sha256=source_identity(), config_sha256=digest(config)))
    print('[GPU VALIDATION PASSED]', parity, resume, flush=True)


if __name__ == '__main__':
    main()
