#!/usr/bin/env python3
"""Regenerate the two canonical Colab notebooks for this bundle."""

from __future__ import annotations

import importlib.util
import json
import pprint
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_soft_wall_v2_notebook_runner", ROOT / "run_campaign.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load runner")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def build(lane: str, *, analysis: bool) -> dict:
    lane_sizes = list(runner.LANE_NY_VALUES[lane])
    bundle = "10_soft_wall_entropy_charge_batched_v2"
    revision = runner.EXPECTED_REVISION
    config_source = pprint.pformat(runner.expected_config(), sort_dicts=True, width=100)
    cells = [
        markdown(
            f"# Batched endpoint soft-wall entropy/charge v2 — lane {lane}\n\n"
            f"This disjoint lane owns $N_y={lane_sizes}$. Dynamics records charge only; all "
            "entropy, Rényi, and intrinsic charge-variance eigensolves run after the final "
            "occupied frame is durable.\n"
        ),
        code("from google.colab import drive\ndrive.mount('/content/drive')\n"),
        markdown("## 1. Locked configuration and runtime controls\n"),
        code(
            "from pathlib import Path\n\n"
            f"LANE = {lane!r}\n"
            f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{bundle}')\n"
            f"OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs/{revision}')\n"
            f"LOCAL_BUNDLE_DIR = Path('/content/{bundle}_lane_' + LANE)\n"
            "SCRATCH_ROOT = Path('/content/soft_wall_endpoint_v2_lane_' + LANE + '_scratch')\n"
            "REPORT_ONLY = False\n"
            "BENCHMARK_ONLY = False  # True runs/reuses the Ny=60 gate without production.\n"
            "MAX_NEW_EXECUTION_BATCHES = None\n"
            + ("RUN_ANALYSIS = False  # Enable only after both reports total 140/140.\n" if analysis else "")
            + f"\nCONFIG = {config_source}\n"
            "print(f'Lane {LANE}; Ny={CONFIG[\"lane_Ny_values\"][LANE]}')\n"
            "print(f'Bundle: {BUNDLE_DRIVE_DIR}')\n"
            "print(f'Output: {OUTPUT_ROOT}')\n"
            "print('Endpoint products: S1, S2, S3, kappa1, Fq; fixed y0=0 half-strip contours')\n"
        ),
        markdown("## 2. A100 and complex128 check\n"),
        code(
            "import torch\n"
            "if not torch.cuda.is_available():\n    raise RuntimeError('CUDA is required')\n"
            "name = torch.cuda.get_device_name(0)\n"
            "gib = torch.cuda.get_device_properties(0).total_memory / 1024**3\n"
            "if 'A100' not in name.upper() or gib < 38:\n"
            "    raise RuntimeError(f'A 40-GB-class A100 is required; found {name}, {gib:.1f} GiB')\n"
            "probe = torch.zeros(1, dtype=torch.complex128, device='cuda:0')\n"
            "del probe\nprint(f'Device: {name}; memory={gib:.1f} GiB; dtype=complex128')\n"
        ),
        markdown(
            "## 3. Stage locally, benchmark, and run/resume\n\n"
            "The runner streams outer shard, physical-cycle, and endpoint-$A_y$ progress. "
            "Production cannot start unless the saved or fresh Ny=60 benchmark projects "
            "20 trajectories in at most four hours.\n"
        ),
        code(
            "import codecs, json, os, shutil, subprocess, sys\n\n"
            "def run_streaming_child(command):\n"
            "    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)\n"
            "    assert process.stdout is not None\n"
            "    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')\n"
            "    try:\n"
            "        while True:\n"
            "            raw = os.read(process.stdout.fileno(), 4096)\n"
            "            if not raw: break\n"
            "            sys.stdout.write(decoder.decode(raw)); sys.stdout.flush()\n"
            "        tail = decoder.decode(b'', final=True)\n"
            "        if tail: sys.stdout.write(tail); sys.stdout.flush()\n"
            "    except BaseException:\n"
            "        process.terminate()\n"
            "        try: process.wait(timeout=5)\n"
            "        except subprocess.TimeoutExpired: process.kill(); process.wait()\n"
            "        raise\n"
            "    finally: process.stdout.close()\n"
            "    returncode = process.wait()\n"
            "    if returncode: raise subprocess.CalledProcessError(returncode, command)\n\n"
            "if not BUNDLE_DRIVE_DIR.is_dir(): raise FileNotFoundError(BUNDLE_DRIVE_DIR)\n"
            "if LOCAL_BUNDLE_DIR.exists(): shutil.rmtree(LOCAL_BUNDLE_DIR)\n"
            "shutil.copytree(BUNDLE_DRIVE_DIR, LOCAL_BUNDLE_DIR)\n"
            "config_path = Path('/content/soft_wall_endpoint_v2_lane_' + LANE + '_config.json')\n"
            "config_path.write_text(json.dumps(CONFIG, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
            "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR/'run_campaign.py'), '--config', str(config_path), '--lane', LANE, '--output-root', str(OUTPUT_ROOT), '--scratch-root', str(SCRATCH_ROOT)]\n"
            "if REPORT_ONLY: command.append('--report-only')\n"
            "if BENCHMARK_ONLY: command.append('--benchmark-only')\n"
            "if MAX_NEW_EXECUTION_BATCHES is not None: command += ['--max-new-execution-batches', str(int(MAX_NEW_EXECUTION_BATCHES))]\n"
            "print('[launch]', ' '.join(command), flush=True)\n"
            "run_streaming_child(command)\nprint('[notebook] runner exited successfully', flush=True)\n"
        ),
    ]
    if analysis:
        cells.extend(
            [
                markdown("## 4. Optional combined endpoint analysis\n"),
                code(
                    "if RUN_ANALYSIS:\n"
                    "    run_streaming_child([sys.executable, '-u', str(LOCAL_BUNDLE_DIR/'analyze_campaign.py'), '--output-root', str(OUTPUT_ROOT), '--analysis-root', str(OUTPUT_ROOT/'analysis_outputs')])\n"
                    "else:\n    print('Analysis skipped (RUN_ANALYSIS=False)')\n"
                ),
            ]
        )
    cells.extend(
        [
            markdown(f"## {5 if analysis else 4}. Release the runtime\n"),
            code("from google.colab import runtime\nruntime.unassign()\nprint('done')\n"),
        ]
    )
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for lane, filename, analysis in (
    ("A", "run_lane_A_Ny40_Ny60.ipynb", True),
    ("B", "run_lane_B_endpoint_Ny30_35_45_50_55.ipynb", False),
):
    (ROOT / filename).write_text(json.dumps(build(lane, analysis=analysis), indent=1) + "\n", encoding="utf-8")
