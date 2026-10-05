#!/usr/bin/env python3
"""Generate the two drop-in A100 lane notebooks and deployment manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def build(lane: str) -> Path:
    output = HERE / f"run_hard_wall_tangent_lane_{lane.lower()}.ipynb"
    cells = [
        markdown(
            f"""
# Hard-wall tangent gaps and endpoint cocycles v2 — lane {lane}

This A100 notebook runs lane **{lane}** of the S100 hard-wall pure-state tangent
campaign. It accumulates the chronological cocycle from cycle 1 through
`2*Ny`, saves five trajectory-resolved finite-time gaps everywhere, and saves
the complete scale-separated cocycle only at `Ny=40`.

The two lane notebooks have disjoint immutable task tables and may run at the
same time. The saved 25-row v1 qualification is verified and reused without
rerunning. Remaining v2 batches are smaller after that 25-row batch took
1.70 hours. Restarting verifies and skips completed NPZ/JSON pairs.
"""
        ),
        code(
            """
from google.colab import drive
drive.mount('/content/drive')
"""
        ),
        markdown(
            """
## 1. Editable configuration

All editable paths and execution controls are here. Scientific parameters and
batch boundaries are locked in `campaign_config.json`.
"""
        ),
        code(
            f"""
from pathlib import Path

LANE = '{lane}'
BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/17_hard_wall_tangent_gap_cocycle')
REUSED_ACQUISITION_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1')
SAVED_V1_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v1')
OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v2')
LOCAL_BUNDLE_DIR = Path('/content/17_hard_wall_tangent_gap_cocycle')
SCRATCH_ROOT = Path(f'/content/hard_wall_tangent_gap_cocycle_lane_{{LANE}}')

REPORT_ONLY = False
MAX_NEW_TASKS = None       # Production resume; set an integer only to cap new tasks.
RUN_ANALYSIS = False       # Enable only after both lanes report complete.
"""
        ),
        markdown(
            """
## 2. Verify the GPU and stage the executable bundle

The notebook copies only executable files to `/content`. Existing acquisition
records remain on Drive and are staged one verified batch at a time.
"""
        ),
        code(
            """
import shutil
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA is unavailable; select an A100 GPU runtime')
props = torch.cuda.get_device_properties(0)
if 'A100' not in props.name.upper() or props.total_memory < 38 * 1024**3:
    raise RuntimeError(f'Need an A100 with 40-GB-class memory; found {props.name}, {props.total_memory / 1024**3:.2f} GiB')
print(f'[device] {props.name}; total GPU RAM={props.total_memory / 1024**3:.2f} GiB; allocator cutoff=38 GiB; dtype=complex128', flush=True)

required = [
    'run_campaign.py',
    'analyze_campaign.py',
    'replay_record_observer.py',
    'campaign_config.json',
    'src/classA_U1FGTN_gpu.py',
    'src/occupied_frame_gpu.py',
]
if not BUNDLE_DRIVE_DIR.is_dir():
    raise FileNotFoundError(f'Missing Drive bundle: {BUNDLE_DRIVE_DIR}')
if LOCAL_BUNDLE_DIR.exists():
    shutil.rmtree(LOCAL_BUNDLE_DIR)
for relative in required:
    source = BUNDLE_DRIVE_DIR / relative
    destination = LOCAL_BUNDLE_DIR / relative
    if not source.is_file():
        raise FileNotFoundError(f'Missing bundle file: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
print(f'[stage] copied {len(required)} executable files to {LOCAL_BUNDLE_DIR}', flush=True)
print(f'[reused input] {REUSED_ACQUISITION_ROOT}', flush=True)
print(f'[saved v1 import] {SAVED_V1_ROOT}', flush=True)
print(f'[output] {OUTPUT_ROOT}', flush=True)
print(f'[scratch] {SCRATCH_ROOT}', flush=True)
"""
        ),
        markdown(
            """
## 3. Run or resume this lane

The outer `tqdm` bar tracks durable batches, including the verified v1 import.
New parameter points show a
physical-acquisition cycle bar followed by a tangent-replay cycle bar. Reused
slot-09 records show only the replay bar. A result is complete only after its
NPZ and completion JSON pass DriveFS readback checks. The notebook relays both
stdout and stderr from the local runner live, including text-mode `tqdm` bars.
"""
        ),
        code(
            """
import codecs
import os
import subprocess
import sys

def stream_child(command, environment):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        bufsize=0,
    )
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    try:
        while True:
            block = os.read(process.stdout.fileno(), 4096)
            if not block:
                break
            sys.stdout.write(decoder.decode(block))
            sys.stdout.flush()
        tail = decoder.decode(b'', final=True)
        if tail:
            sys.stdout.write(tail)
            sys.stdout.flush()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

command = [
    sys.executable,
    '-u',
    str(LOCAL_BUNDLE_DIR / 'run_campaign.py'),
    '--config', str(LOCAL_BUNDLE_DIR / 'campaign_config.json'),
    '--lane', LANE,
    '--reused-root', str(REUSED_ACQUISITION_ROOT),
    '--v1-root', str(SAVED_V1_ROOT),
    '--output-root', str(OUTPUT_ROOT),
    '--scratch-root', str(SCRATCH_ROOT),
]
if REPORT_ONLY:
    command.append('--report-only')
if MAX_NEW_TASKS is not None:
    command.extend(['--max-new-tasks', str(int(MAX_NEW_TASKS))])
print('[launch] ' + ' '.join(command), flush=True)
environment = dict(os.environ)
environment['PYTHONUNBUFFERED'] = '1'
environment['TQDM_MININTERVAL'] = '300'
stream_child(command, environment)
print(f'[notebook] lane {LANE} runner exited successfully', flush=True)
"""
        ),
        markdown(
            """
## 4. Quenched ensemble analysis

After both lanes reach completion, set `RUN_ANALYSIS=True` in either notebook.
The analysis extracts each trajectory's gaps first and only then averages over
the 100 trajectories, with SEM and a 10,000-draw whole-trajectory bootstrap.
"""
        ),
        code(
            """
if RUN_ANALYSIS:
    analysis_command = [
        sys.executable,
        '-u',
        str(LOCAL_BUNDLE_DIR / 'analyze_campaign.py'),
        '--output-root', str(OUTPUT_ROOT),
        '--v1-root', str(SAVED_V1_ROOT),
        '--scratch-root', str(SCRATCH_ROOT / 'analysis'),
    ]
    print('[analysis launch] ' + ' '.join(analysis_command), flush=True)
    stream_child(analysis_command, environment)
else:
    print('[analysis] skipped; enable only after both lanes complete', flush=True)
"""
        ),
        markdown(
            """
## 5. Release the runtime

Run this only after the campaign and optional analysis cells exit.
"""
        ),
        code(
            """
from google.colab import runtime
runtime.unassign()
print('done')
"""
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": output.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    outputs = [build(lane) for lane in ("A", "B")]
    deployment_files = (
        "run_hard_wall_tangent_lane_a.ipynb",
        "run_hard_wall_tangent_lane_b.ipynb",
        "run_campaign.py",
        "analyze_campaign.py",
        "replay_record_observer.py",
        "campaign_config.json",
        "build_notebooks.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    )
    manifest = {
        "schema": "hard_wall_tangent_gap_cocycle_deployment_manifest_v1",
        "sampling_revision": "hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_v2",
        "files": {
            relative: {
                "bytes": int((HERE / relative).stat().st_size),
                "sha256": sha256_file(HERE / relative),
            }
            for relative in deployment_files
        },
    }
    manifest_path = HERE / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for output in outputs:
        print(output)
    print(manifest_path)
