"""Atomic products with identity-bound receipts and complete RNG snapshots."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def publish_npz(path, payload, identity, scratch):
    path, scratch = Path(path), Path(scratch)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch / path.name
    np.savez(local, **payload)
    count, checksum = local.stat().st_size, sha(local)
    temp = path.with_name(path.name + '.tmp')
    shutil.copyfile(local, temp)
    if temp.stat().st_size != count or sha(temp) != checksum:
        raise OSError(f'Publication readback failed: {temp}')
    os.replace(temp, path)
    if path.stat().st_size != count or sha(path) != checksum:
        raise OSError(f'Final publication readback failed: {path}')
    atomic_json(path.with_suffix('.json'), dict(identity, filename=path.name, bytes=count, sha256=checksum))
    local.unlink()


def verified_pair(path, expected):
    path = Path(path)
    try:
        row = json.loads(path.with_suffix('.json').read_text())
        return (all(row.get(k) == v for k, v in expected.items()) and row['filename'] == path.name
                and row['bytes'] == path.stat().st_size and row['sha256'] == sha(path))
    except (OSError, ValueError, KeyError):
        return False


def load_checkpoint(path, expected):
    path = Path(path)
    if not path.exists() and not path.with_suffix('.json').exists():
        return None
    if not verified_pair(path, expected):
        raise ValueError(f'Incomplete, corrupted, or mismatched checkpoint: {path}')
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def capture_rng():
    state = np.random.get_state()
    result = dict(rng_np_algorithm=np.array(state[0]), rng_np_state=state[1],
        rng_np_position=np.array(state[2]), rng_np_has_gauss=np.array(state[3]),
        rng_np_gauss=np.array(state[4]), rng_torch=torch.get_rng_state().numpy(),
        rng_cuda_count=np.array(torch.cuda.device_count() if torch.cuda.is_available() else 0))
    if torch.cuda.is_available():
        for index, value in enumerate(torch.cuda.get_rng_state_all()):
            result[f'rng_cuda_{index}'] = value.cpu().numpy()
    return result


def restore_rng(payload):
    np.random.set_state((str(payload['rng_np_algorithm']), payload['rng_np_state'],
        int(payload['rng_np_position']), int(payload['rng_np_has_gauss']), float(payload['rng_np_gauss'])))
    torch.set_rng_state(torch.as_tensor(payload['rng_torch'], dtype=torch.uint8).cpu())
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count != int(payload['rng_cuda_count']):
        raise ValueError('Checkpoint CUDA RNG count differs')
    if count:
        torch.cuda.set_rng_state_all([torch.as_tensor(payload[f'rng_cuda_{index}'], dtype=torch.uint8).cpu()
                                     for index in range(count)])
