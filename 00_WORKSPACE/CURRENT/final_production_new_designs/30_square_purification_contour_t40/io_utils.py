"""Small DriveFS publication and exact RNG helpers (same contract as Campaign 25)."""
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
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pair_verified(path, receipt, expected):
    try:
        row = json.loads(receipt.read_text())
        return (all(row.get(key) == value for key,value in expected.items()) and
                row['filename'] == path.name and row['bytes'] == path.stat().st_size and
                row['sha256'] == sha(path))
    except (OSError, ValueError, KeyError):
        return False


def publish_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.name+'.tmp')
    shutil.copyfile(source, temp)
    if temp.stat().st_size != source.stat().st_size or sha(temp) != sha(source):
        raise OSError(f'DriveFS temporary readback failed: {temp}')
    os.replace(temp, destination)
    if destination.stat().st_size != source.stat().st_size or sha(destination) != sha(source):
        raise OSError(f'DriveFS final readback failed: {destination}')


def publish_pair(payload, path, receipt, expected, scratch, compressed):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    local = scratch/path.name
    saver = np.savez_compressed if compressed else np.savez
    saver(local, **payload)
    publish_file(local, path)
    record = dict(expected, filename=path.name, bytes=path.stat().st_size, sha256=sha(path))
    local_json = scratch/receipt.name
    local_json.write_text(json.dumps(record, indent=2)+'\n')
    publish_file(local_json, receipt)
    if not pair_verified(path, receipt, expected):
        raise OSError('Published pair failed final verification')


def capture_rng():
    state = np.random.get_state()
    payload = dict(rng_np_algorithm=np.array(state[0]), rng_np_state=state[1],
                   rng_np_position=np.array(state[2]), rng_np_has_gauss=np.array(state[3]),
                   rng_np_gauss=np.array(state[4]), rng_torch=torch.get_rng_state().numpy(),
                   rng_cuda_count=np.array(torch.cuda.device_count() if torch.cuda.is_available() else 0))
    if torch.cuda.is_available():
        for i,state in enumerate(torch.cuda.get_rng_state_all()):
            payload[f'rng_cuda_{i}'] = state.cpu().numpy()
    return payload


def restore_rng(payload):
    np.random.set_state((str(payload['rng_np_algorithm']), payload['rng_np_state'],
                        int(payload['rng_np_position']), int(payload['rng_np_has_gauss']),
                        float(payload['rng_np_gauss'])))
    torch.set_rng_state(torch.as_tensor(payload['rng_torch'], dtype=torch.uint8).cpu())
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count != int(payload['rng_cuda_count']):
        raise ValueError('CUDA RNG device count differs from checkpoint')
    if count:
        torch.cuda.set_rng_state_all([torch.as_tensor(payload[f'rng_cuda_{i}'],dtype=torch.uint8).cpu()
                                     for i in range(count)])
