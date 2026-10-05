#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bundle_layout import bundle_group, bundle_path  # noqa: E402

PRODUCTION_COLLECTION = (
    "classA_final_production_outputs/production_10sample_v4_occupied_frame_cycle_resolved"
)
PILOT_COLLECTION = (
    "classA_pilot_outputs/production_10sample_v4_occupied_frame_cycle_resolved"
)
BUNDLES = [
    "00_validation",
    "01_bulk_width_gate",
    "02_pure_wall_master",
    "03_chirality_replay",
    "04_maxmix_master",
    "05_scans_and_controls",
    "06_b1_controller_frame",
]

DISCONNECT_SOURCE = (
    "from google.colab import runtime\n"
    "runtime.unassign()\n"
    "print('done')\n"
)

DESCRIPTIONS = {
    "00_validation": "V0 engine, random-record, tangent, resume, and shard validation",
    "01_p1_chern_dynamics": "lean P1 cycle-resolved periodic-trijunction Chern dynamics",
    "01_bulk_width_gate": "non-gating fixed-Nx=20 wall/control baseline",
    "02_wall_cft_windows": "raw hard/soft programmable-wall window spectra, contours, and squared correlators",
    "02_pure_wall_master": "S1/T1/B2 master trajectories, live-state H1/H2 descendants, exact G5 fixed-word spectra, and supplemental R1 record statistics",
    "03_h1_modular_response": "trajectory-resolved signed retarded modular response and supplemental packet drift",
    "03_chirality_replay": "H3 frozen-record twist circles and exact replay recovery",
    "04_maxmix_operator_cft": "redesigned G4 max-mix purification, entropy/charge contours, direct operator spectrum, and secondary operator CFT",
    "04_maxmix_master": "P2 max-mix purification and same-record physical tangent scaling",
    "05_pure_tangent_stability": "redesigned G5 independent pure Born trajectories with online analytic occupied-empty tangent cocycle",
    "05_scans_and_controls": "Two-construction pure S2 entanglement-transition scan, max-mix S2 arm, and M2–M3 controls",
    "06_b1_controller_frame": "required B1 controller-frame surrogate-attraction diagnostic",
}

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



def bundle_notebook(bundle: str) -> dict:
    trajectory_text = (
        "25 independent trajectories in five immutable five-trajectory shards"
        if bundle in ("01_p1_chern_dynamics", "02_wall_cft_windows", "03_h1_modular_response", "04_maxmix_operator_cft", "05_pure_tangent_stability")
        else "10 independent trajectories in two immutable five-trajectory shards"
    )
    collection_source = (
        f"production_collection = bundle_config.get('production_output_collection', {PRODUCTION_COLLECTION!r})\n"
        f"pilot_collection = bundle_config.get('pilot_output_collection', {PILOT_COLLECTION!r})\n"
        if bundle in ("01_p1_chern_dynamics", "02_wall_cft_windows", "03_h1_modular_response", "04_maxmix_operator_cft", "05_pure_tangent_stability")
        else
        f"production_collection = {PRODUCTION_COLLECTION!r}\n"
        f"pilot_collection = {PILOT_COLLECTION!r}\n"
    )
    cells = [
        markdown(
            f"# {bundle}: independent resumable A100 bundle\n\n"
            f"{DESCRIPTIONS[bundle]}. This notebook launches only `{bundle}`; it never "
            f"continues into another experiment. Every ordinary stochastic case uses {trajectory_text}. Run the "
            "default report-only pass first, inspect the exact verified/pending inventory, "
            "then set `RESUME_REPORT_ONLY=False` to compute only missing shards."
        ),
        markdown("## 1. Mount Drive and locate the campaign package\n"),
        code(
            "from pathlib import Path\n"
            "import hashlib, json, subprocess, sys\n\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n\n"
            "MYDRIVE = Path('/content/drive/MyDrive')\n"
            "CAMPAIGN_ROOT = MYDRIVE / 'final_production_ready_figure_scripts'\n"
            "runner = CAMPAIGN_ROOT / 'colab_bundle_runner.py'\n"
            f"bundle_root = CAMPAIGN_ROOT / {bundle_group(bundle)!r} / {bundle!r}\n"
            "if not runner.is_file() or not (bundle_root / 'run_bundle.py').is_file():\n"
            "    raise FileNotFoundError('Upload the complete updated final_production_ready_figure_scripts folder')\n"
            "bundle_config = json.loads((bundle_root / 'production_config.json').read_text())\n"
            "OUTPUT_BUNDLE = bundle_config.get('output_bundle', bundle_root.name)\n"
            "print(f'[campaign] {CAMPAIGN_ROOT}')\n"
            "print(f'[bundle] {bundle_root}')\n"
        ),
        markdown("## 2. Select this bundle's queue\n"),
        code(
            "RUN_PROFILE = 'production'\n"
            f"BUNDLE = {bundle!r}\n"
            "CASE_PREFIXES = ()  # e.g. ('S2_',); empty runs every currently eligible case\n"
            "CASE_IDS = ()       # exact IDs; empty uses prefixes/full bundle\n"
            "MAX_SESSION_GPU_HOURS_OVERRIDE = None\n"
            "PREFLIGHT_ONLY = False\n"
            "RESUME_REPORT_ONLY = True  # inspect first; change to False to launch missing work\n"
            "HEARTBEAT_SECONDS = 60\n"
            "RUN_QUEUE = True\n"
            "if HEARTBEAT_SECONDS <= 0:\n"
            "    raise ValueError('HEARTBEAT_SECONDS must be positive')\n"
            "if CASE_PREFIXES and CASE_IDS:\n"
            "    raise ValueError('use CASE_PREFIXES or CASE_IDS, not both')\n"
            "print(json.dumps({\n"
            "    'bundle': BUNDLE, 'profile': RUN_PROFILE, 'case_prefixes': CASE_PREFIXES,\n"
            "    'case_ids': CASE_IDS, 'resume_report_only': RESUME_REPORT_ONLY,\n"
            "    'heartbeat_seconds': HEARTBEAT_SECONDS,\n"
            "}, indent=2))\n"
        ),
        markdown(
            f"## {4 if bundle in ('01_p1_chern_dynamics', '03_h1_modular_response') else 3}. "
            "Verify, resume, and stop at the end of this bundle\n"
        ),
        code(
            "if not RUN_QUEUE:\n"
            "    raise RuntimeError('RUN_QUEUE is False')\n"
            "command = [\n"
            "    sys.executable, '-u', str(runner), '--drive-root', str(MYDRIVE),\n"
            "    '--bundle', BUNDLE, '--profile', RUN_PROFILE,\n"
            "    '--heartbeat-seconds', str(HEARTBEAT_SECONDS),\n"
            "]\n"
            "for prefix in CASE_PREFIXES:\n"
            "    command += ['--case-prefix', str(prefix)]\n"
            "for case_id in CASE_IDS:\n"
            "    command += ['--case-id', str(case_id)]\n"
            "if MAX_SESSION_GPU_HOURS_OVERRIDE is not None:\n"
            "    command += ['--max-session-hours', str(MAX_SESSION_GPU_HOURS_OVERRIDE)]\n"
            "if PREFLIGHT_ONLY:\n"
            "    command.append('--preflight-only')\n"
            "if RESUME_REPORT_ONLY:\n"
            "    command.append('--resume-report-only')\n"
            f"{collection_source}"
            "collection = production_collection if RUN_PROFILE == 'production' else pilot_collection\n"
            "output_root = MYDRIVE / collection\n"
            "session_dir = output_root / '_bundle_sessions'\n"
            "build_id = hashlib.sha256(runner.read_bytes()).hexdigest()[:16]\n"
            "print('[bundle dashboard]')\n"
            "print(json.dumps({\n"
            "    'bundle': BUNDLE, 'profile': RUN_PROFILE, 'output_root': str(output_root),\n"
            "    'runner_build_id': build_id, 'heartbeat_seconds': HEARTBEAT_SECONDS,\n"
            "    'session_dir': str(session_dir), 'case_prefixes': CASE_PREFIXES,\n"
            "    'case_ids': CASE_IDS, 'resume_report_only': RESUME_REPORT_ONLY,\n"
            "}, indent=2))\n"
            "print(f'[launch bundle] {\" \".join(command)}', flush=True)\n"
            "result = subprocess.run(command, check=False)\n"
            "if result.returncode:\n"
            "    sessions = sorted(session_dir.glob(f'*_{BUNDLE}_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)\n"
            "    if sessions:\n"
            "        latest = json.loads(sessions[0].read_text())\n"
            "        print('[saved bundle failure]')\n"
            "        print(json.dumps({'session': str(sessions[0]), 'failure': latest.get('failure'), 'current': latest.get('current'), 'log': latest.get('session_log')}, indent=2))\n"
            "    raise subprocess.CalledProcessError(result.returncode, command)\n"
        ),
        markdown("## Final. Disconnect this completed Colab runtime\n"),
        code(DISCONNECT_SOURCE),
    ]
    if bundle == "01_p1_chern_dynamics":
        cells[5:5] = [
            markdown(
                "## 3. Required A100 cost and memory preflight\n\n"
                "Run this once before production. It executes one complete five-trajectory "
                "`L=64`, `n_shell=1` shard without archiving it, saves a version-matched "
                "safety receipt, and reports the measured cost projection. The production "
                "runner refuses to launch without a current safe receipt.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'run_bundle.py'), '--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight'], check=True)\n"
            ),
        ]
        cells[-2:-2] = [
            markdown("## 5. Merge the completed matrix and create the reference-style figure\n"),
            code(
                "RUN_P1_ANALYSIS = False\n"
                "if RUN_P1_ANALYSIS:\n"
                "    archive_root = output_root / OUTPUT_BUNDLE\n"
                "    analysis_root = archive_root / 'P1_chern_analysis'\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'src' / 'p1_chern_analysis.py'), '--archive-root', str(archive_root), '--output-root', str(analysis_root)], check=True)\n"
                "    print((analysis_root / 'p1_chern_analysis_summary.json').read_text())\n"
            ),
        ]
    elif bundle == "02_wall_cft_windows":
        cells[5:5] = [
            markdown(
                "## 3. Required Ny=60 hard/soft A100 preflight\n\n"
                "Run this once before production. It evaluates a complete five-trajectory "
                "shard for each wall construction at the largest geometry, including all "
                "Ay=30 origins, and writes a version-matched safety receipt. No scientific "
                "parameter is reduced if the preflight is unsafe.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'run_bundle.py'), '--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight'], check=True)\n"
            ),
        ]
    elif bundle == "03_h1_modular_response":
        cells[5:5] = [
            markdown(
                "## 3. Required complete-shard A100 preflight\n\n"
                "Run this once before production. It executes all 80 cycles and all six "
                "H1 observations for one complete five-trajectory soft-wall shard, then "
                "writes a version-matched memory/runtime/storage receipt. Production "
                "remains locked if the receipt is missing, stale, or unsafe; no scientific "
                "parameter is reduced automatically.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'run_bundle.py'), '--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight'], check=True)\n"
            ),
        ]
        cells[-2:-2] = [
            markdown("## 5. Merge completed H1 shards, analyze, and create figures\n"),
            code(
                "RUN_H1_ANALYSIS = False\n"
                "if RUN_H1_ANALYSIS:\n"
                "    archive_root = output_root / OUTPUT_BUNDLE\n"
                "    analysis_root = archive_root / 'analysis_outputs'\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'src' / 'h1_modular_analysis.py'), '--archive-root', str(archive_root), '--output-root', str(analysis_root), '--bundle-root', str(bundle_root)], check=True)\n"
                "    print((analysis_root / 'h1_analysis_summary.json').read_text())\n"
            ),
        ]
    elif bundle == "00_validation":
        cells[-2:-2] = [
            markdown("## 4. Display the newest saved GPU-kernel benchmark\n"),
            code(
                "import tarfile\n"
                "archives = sorted((output_root / OUTPUT_BUNDLE).glob('*.tar.gz'), key=lambda p: p.stat().st_mtime_ns)\n"
                "if not archives:\n"
                "    print('[validation summary] no saved validation archive yet')\n"
                "else:\n"
                "    with tarfile.open(archives[-1], 'r:gz') as archive:\n"
                "        member = next(m for m in archive.getmembers() if m.name.lstrip('./') == 'manifest.json')\n"
                "        handle = archive.extractfile(member)\n"
                "        if handle is None:\n"
                "            raise RuntimeError(f'unreadable manifest in {archives[-1]}')\n"
                "        manifest = json.load(handle)\n"
                "    benchmark = manifest['validation']['diagnostics']['sitewise_batch_benchmark']\n"
                "    print(f'[validation archive] {archives[-1]}')\n"
                "    print(json.dumps(benchmark, indent=2, sort_keys=True))\n"
                "    print(f\"[SITEWISE GPU SPEEDUP] {benchmark['speedup_over_grouped_reference']:.3f}x\")\n"
            ),
        ]
    elif bundle == "02_pure_wall_master":
        cells[-2:-2] = [
            markdown("## 4. Optional supplemental record large-deviation summary\n"),
            code(
                "RUN_R1_ANALYSIS = False\n"
                "if RUN_R1_ANALYSIS:\n"
                "    archive_root = output_root / OUTPUT_BUNDLE\n"
                "    analysis_root = archive_root / 'R1_analysis'\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'src' / 'record_spectrum_analysis.py'), '--archive-root', str(archive_root), '--output-root', str(analysis_root)], check=True)\n"
                "    print((analysis_root / 'R1_record_spectrum_summary.json').read_text())\n"
            ),
            markdown(
                "## 5. Exact G5 fixed-word many-body singular spectrum\n\n"
                "This deterministic descendant reuses completed primary explicit-interface "
                "and matched-trivial master words. It initializes the one-particle "
                "correlation matrix at $\\mathbf{1}/2$ only as an algebraic probe of "
                "$\\widehat K\\widehat K^\\dagger/Z_K$. It never propagates a Choi state, "
                "never changes a parent archive, and resumes from checksum-verified G5 receipts.\n\n"
                "### Formula dictionary\n\n"
                "For $N_{\\rm orb}=2N_xN_y$, the normalized positive operator and its "
                "identity-input normalization are\n"
                "$$\\rho_R^K=\\frac{\\widehat K\\widehat K^\\dagger}{Z_K},\\qquad "
                "p_K(\\mathbf{1}/2)=2^{-N_{\\rm orb}}Z_K,\\qquad "
                "\\log Z_K=N_{\\rm orb}\\log2+\\log p_K.$$\n"
                "If $\\nu_i$ are the natural occupations of $\\rho_R^K$, then\n"
                "$$j_i=2\\sqrt{\\nu_i(1-\\nu_i)},\\qquad "
                "\\mu_i^\\pm=\\frac{1\\pm\\sqrt{1-j_i^2}}2,$$\n"
                "and the complete normalized many-body eigenvalues and absolute singular "
                "values are\n"
                "$$\\lambda_{\\boldsymbol n}=\\prod_i\\nu_i^{n_i}(1-\\nu_i)^{1-n_i},"
                "\\qquad \\sigma_{\\boldsymbol n}=\\sqrt{Z_K\\lambda_{\\boldsymbol n}}.$$\n"
                "Relative to the dominant occupation $n_i^{(0)}=\\mathbf{1}_{\\nu_i\\geq1/2}$, "
                "a flip carries $q_i=1-2n_i^{(0)}$ and singular-amplitude gap\n"
                "$$\\delta_i=\\frac12\\left|\\log\\frac{\\nu_i}{1-\\nu_i}\\right|"
                "=\\operatorname{arcosh}(1/j_i).$$\n"
                "The final finite-size tests use\n"
                "$$\\Delta_{q,n}(L)=\\frac{2\\pi v}{L}"
                "\\left(\\frac{q^2}{2k}+n\\right)+O(L^{-2}),\\qquad "
                "f_K(L)=f_\\infty-\\frac{\\pi c_{\\rm eff}v}{12L^2}+O(L^{-4}),$$\n"
                "where $f_K$ is formed after matched-trivial subtraction and division by "
                "the two walls. Population gaps $|\\log[\\nu_i/(1-\\nu_i)]|$ are twice "
                "the coherent singular-amplitude gaps $\\delta_i$.\n"
            ),
            code(
                "RUN_G5_MANYBODY_ANALYSIS = False\n"
                "if RUN_G5_MANYBODY_ANALYSIS:\n"
                "    archive_roots = [\n"
                "        output_root / OUTPUT_BUNDLE,\n"
                "        MYDRIVE / 'classA_final_production_outputs' / 'production_10sample_v3_fixed_nx20_ny20_30_40_50_60' / OUTPUT_BUNDLE,\n"
                "    ]\n"
                "    analysis_root = output_root / OUTPUT_BUNDLE / 'G5_manybody_spectrum'\n"
                "    command = [sys.executable, '-u', str(bundle_root / 'src' / 'g5_manybody_spectrum_analysis.py'), '--bundle-root', str(bundle_root), '--output-root', str(analysis_root), '--bootstrap-replicates', '2000', '--mode', RUN_PROFILE]\n"
                "    for archive_root in archive_roots:\n"
                "        command += ['--archive-root', str(archive_root)]\n"
                "    subprocess.run(command, check=True)\n"
                "    print((analysis_root / 'G5_manybody_spectrum_summary.json').read_text())\n"
            ),
        ]
    elif bundle == "05_scans_and_controls":
        cells[-2:-2] = [
            markdown("## 4. Analyze the completed 10-trajectory pure S2 matrix\n"),
            code(
                "RUN_S2_ENTANGLEMENT_ANALYSIS = False\n"
                "if RUN_S2_ENTANGLEMENT_ANALYSIS:\n"
                "    analysis_root = output_root / OUTPUT_BUNDLE / 'S2_entanglement_analysis'\n"
                "    archive_roots = [\n"
                "        output_root / OUTPUT_BUNDLE,\n"
                "        MYDRIVE / 'classA_final_production_outputs' / BUNDLE,\n"
                "    ]\n"
                "    command = [sys.executable, '-u', str(bundle_root / 'src' / 's2_entanglement_analysis.py'), '--bundle-root', str(bundle_root), '--output-root', str(analysis_root)]\n"
                "    for archive_root in archive_roots:\n"
                "        command += ['--archive-root', str(archive_root)]\n"
                "    subprocess.run(command, check=True)\n"
                "    print((analysis_root / 's2_entanglement_report.json').read_text())\n"
            ),
        ]
    elif bundle == "06_b1_controller_frame":
        cells[-2:-2] = [
            markdown("## 4. Merge, analyze, plot, and report completed B1 cases\n"),
            code(
                "RUN_B1_ANALYSIS = False\n"
                "if RUN_B1_ANALYSIS:\n"
                "    archive_root = output_root / OUTPUT_BUNDLE\n"
                "    analysis_root = archive_root / 'B1_analysis'\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'src' / 'b1_analysis.py'), '--bundle-root', str(bundle_root), '--archive-root', str(archive_root), '--output-root', str(analysis_root), '--mode', 'production'], check=True)\n"
                "    print((analysis_root / 'b1_analysis_summary.json').read_text())\n"
            ),
        ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def existing_p1_notebook() -> dict:
    cells = [
        markdown(
            "# Finish the existing 25-trajectory P1 campaign\n\n"
            "This notebook verifies the transferred `production_25sample_v1` archives and "
            "finishes only the missing P1 shards. It is pinned to the original audit and "
            "engine and cannot launch W1. The first pass is report-only."
        ),
        markdown("## 1. Mount Drive and locate the frozen completion bundle\n"),
        code(
            "from pathlib import Path\n"
            "import subprocess, sys\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n"
            "MYDRIVE = Path('/content/drive/MyDrive')\n"
            "ROOT = MYDRIVE / 'final_production_ready_figure_scripts' / 'prior_designs' / '01_p1_existing_completion'\n"
            "RUNNER = ROOT / 'resume_existing_p1.py'\n"
            "if not RUNNER.is_file():\n"
            "    raise FileNotFoundError('Upload the complete updated campaign package')\n"
            "print(f'[frozen P1 source] {ROOT}')\n"
        ),
        markdown("## 2. Verify first, then finish only the missing shards\n"),
        code(
            "RUN_MISSING_SHARDS = False  # first run False; then review the exact verified/pending report\n"
            "HEARTBEAT_SECONDS = 60\n"
            "command = [sys.executable, '-u', str(RUNNER), '--drive-root', str(MYDRIVE), '--heartbeat-seconds', str(HEARTBEAT_SECONDS)]\n"
            "if RUN_MISSING_SHARDS:\n"
            "    command.append('--run')\n"
            "print(f'[launch exact P1 resume] {\" \".join(command)}', flush=True)\n"
            "subprocess.run(command, check=True)\n"
        ),
        markdown("## Final. Disconnect this completed Colab runtime\n"),
        code(DISCONNECT_SOURCE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def readme(bundle: str) -> str:
    if bundle in ("02_wall_cft_windows", "03_h1_modular_response"):
        # This standalone redesign has a contract-specific README rather than the
        # legacy ten-sample campaign boilerplate.
        return (bundle_path(ROOT, bundle) / "README.md").read_text(encoding="utf-8")
    if bundle == "01_p1_chern_dynamics":
        return (
            "# 01_p1_chern_dynamics\n\n"
            "Standalone versioned P1 experiment for trajectory-resolved real-space Chern "
            "convergence only. It preserves every older P1 and W1 bundle and writes to "
            "`MyDrive/classA_final_production_outputs/production_25sample_p1_chern_v1/01_p1_chern_dynamics`.\n\n"
            "The immutable matrix has `L=16,24,32,64`, `n_shell=1,2,None`, 25 random "
            "half-filled Slater trajectories per case, perfect correction, random serial "
            "updates, `n_a=0.5`, and exactly `L` cycles. At the initial state and every "
            "completed cycle, ten distinct periodic trijunction centers are freshly and "
            "deterministically sampled for each trajectory. The radius-`0.4L` Chern values "
            "and their within-trajectory center mean are the only scientific products.\n\n"
            "Open `run_production_bundle.ipynb` on an A100. First run the required `L=64` "
            "preflight cell and review its time/memory projection. Then inspect the default "
            "resume report before setting `RESUME_REPORT_ONLY=False`. Each case has five "
            "immutable shards of five trajectories. No covariance, Bott index, density, "
            "entropy, tangent, convergence history, purity product, or ordered Born record "
            "is archived. After all 60 archives verify, enable the analysis cell to merge "
            "the 25 samples and produce PDF/PNG figures and compact CSV summaries.\n\n"
            "The runner calls `classA_U1FGTN_gpu.run_markov_circuit` and uses occupied-frame "
            "states without dense covariance materialization. `production_config.json` is "
            "immutable run intent; regenerate copied sources with "
            "`_maintenance/sync_bundle_sources.py`.\n"
        )
    if bundle == "06_b1_controller_frame":
        return (
            "# 06_b1_controller_frame\n\n"
            "Required final B1 controller-frame surrogate-attraction diagnostic. It runs "
            "the fixed `Nx=20`, `Ny=40` interface and matched-trivial cases with 10 "
            "trajectories each in two immutable five-trajectory shards. The calculation "
            "uses exactly `2*Ny` cycles, random sequence, perfect correction, and "
            "complex128 through `classA_U1FGTN_gpu.run_markov_circuit`.\n\n"
            "The experiment tests finite-window approach to the artificial signed "
            "controller-frame ground-state manifold. It has no charge-sector weights, "
            "training/test split, manuscript gate, or infinite-time relaxation claim. "
            "Run a case's selected shard queue in one invocation: the A100 diagonalizes "
            "the frame once and reuses it in memory while archiving every five-trajectory "
            "shard independently.\n\n"
            "Upload the entire folder to Drive, select an A100 runtime, and run the "
            "notebook. It defaults to a production resume report; after checking the "
            "verified and pending counts, disable report-only mode to launch missing shards. "
            "The resumable queue skips verified prior "
            "archives, and checks the active-output storage budget before starting each case. "
            "The corrected charge-validation revision writes accepted archives under "
            "`06_b1_controller_frame_frame_native_v2` and preserves any future rejected "
            "compact products under that directory's `_rejected_validation/` tier. "
            "Permanent archives retain compact replay records, static spectra, "
            "residual maps, and scalar histories only; no dense covariance, controller "
            "operator, or ground-state projector is saved. After all four archives exist, "
            "enable the separate merge/analysis/report cell.\n"
        )
    width = "All active cases resolve `Nx=20` directly; no accepted-width file is consulted.\n\n"
    dual_s2 = (
        "Every explicit-interface S2 `(Ny, alpha_in)` point has a pure and max-mix arm, "
        "and every point also has a pure support-terminated mirror. The pure constructions "
        "record identical entropy, topology, correlator, and tangent products; the max-mix "
        "arm records purification and the physical tangent spectrum. Dense Choi covariance "
        "tracking is disabled throughout, and distinct Born ensembles are never pooled. "
        "After all pure cases complete, the CPU analysis cell verifies/merges at least 10 "
        "trajectories, performs the locked log-chord and area-law fits, bootstraps whole "
        "trajectories, and produces the publication figure.\n\n"
        if bundle == "05_scans_and_controls"
        else ""
    )
    p2_policy = (
        "Dense Choi covariance tracking is disabled; the archive retains compact "
        "purification, topology, record, and physical tangent products only.\n\n"
        if bundle == "04_maxmix_master"
        else ""
    )
    h3_policy = (
        "Calibration selects one exact record from one five-record `20x40` parent shard "
        "and runs a 33-point closed frozen-flux circle. Production runs record index `0` "
        "from parent shard `0` for each of the explicit-interface and matched-trivial "
        "cases: two representative circles total, not an ensemble-frequency estimate. "
        "Each point checkpoints elapsed time and A100 peak memory, and the manifest "
        "projects every denser-grid cost.\n\n"
        "For an independent local check, `cpu_one_record_pilot/` uses the canonical CPU "
        "engine for one `20x40` record on a conservative nine-point closed circle. It is "
        "not part of the A100 queues.\n\n"
        if bundle == "03_chirality_replay"
        else ""
    )
    g5_policy = (
        "After all primary explicit-interface and matched-trivial master words are "
        "complete at `Ny=20,30,40,50,60`, the optional G5 cell runs a deterministic, "
        "receipt-resumable descendant. It initializes the one-particle correlation "
        "matrix at `identity/2` only as an algebraic probe of `K K^dagger / Z_K`, "
        "records checkpoint natural occupations, oriented caps, factorized spectra, "
        "and charge-resolved low levels, and never propagates a Choi state or changes "
        "a parent archive. The optional R1 merger remains separately labeled as a "
        "supplemental record large-deviation diagnostic. The already versioned R1 "
        "stochastic cases remain byte-for-byte unchanged, but G5 does not depend on "
        "them and never treats their SCGF as an operator spectrum. The G5 cell searches "
        "the current v4 parent directory first and the completed v3 fixed-geometry "
        "directory second, so an already finished v3 parent campaign is not rerun.\n\n"
        "The compact formula dictionary is: "
        "$\\rho_R^K=\\widehat K\\widehat K^\\dagger/Z_K$; "
        "$p_K(\\mathbf{1}/2)=2^{-N_{\\rm orb}}Z_K$; "
        "$j_i=2\\sqrt{\\nu_i(1-\\nu_i)}$; "
        "$\\lambda_{\\boldsymbol n}=\\prod_i\\nu_i^{n_i}(1-\\nu_i)^{1-n_i}$; and "
        "$\\sigma_{\\boldsymbol n}=\\sqrt{Z_K\\lambda_{\\boldsymbol n}}$. "
        "A flip relative to the dominant occupation carries "
        "$q_i=1-2n_i^{(0)}$ and singular-amplitude gap "
        "$\\delta_i=\\tfrac12|\\log[\\nu_i/(1-\\nu_i)]|=\\operatorname{arcosh}(1/j_i)$. "
        "The tower fit is $\\Delta_{q,n}=2\\pi vL^{-1}(q^2/2k+n)+O(L^{-2})$; "
        "the matched-control per-wall leading-level fit is "
        "$f_K=f_\\infty-\\pi c_{\\rm eff}v/(12L^2)+O(L^{-4})$.\n\n"
        if bundle == "02_pure_wall_master"
        else ""
    )
    pilot_allocation = (
        "one preregistered record" if bundle == "03_chirality_replay" else "shard zero (five trajectories)"
    )
    production_allocation = (
        "H3 has one record and one shard for each of its two 33-point protocols. "
        if bundle == "03_chirality_replay"
        else "Stochastic production cases have two fixed shards of five trajectories. "
    )
    return (
        f"# {bundle}\n\n{DESCRIPTIONS[bundle]}.\n\n"
        f"{p2_policy}"
        f"{dual_s2}"
        f"{h3_policy}"
        f"{g5_policy}"
        "The notebook defaults to production in checksum-verification/report-only mode. "
        "Inspect the resolved queue first, then set `RESUME_REPORT_ONLY=False` to compute "
        "only missing shards. Pilot profiles remain available in the runner but are not "
        "the default operational path.\n\n"
        "Open `run_production_bundle.ipynb` in an A100 Colab runtime. One invocation runs "
        f"one complete shard and atomically archives it to `MyDrive/{PRODUCTION_COLLECTION}`. "
        f"{production_allocation}Do not change "
        "sample count, duration, sequence, dtype, or physical protocol inside the notebook; select "
        "cases through the generated resumable queue. By default it queues every currently listed "
        "case and derives the valid shard count for each case; verified existing archives are skipped safely after "
        "an interruption. Before every new shard, the storage guard reserves 1 GB of headroom under "
        "the 12 GB active-output budget and refuses to launch when outputs must be offloaded.\n\n"
        f"{width}"
        "`production_config.json` is immutable run intent. `src/source_manifest.json` records the "
        "canonical engine and helper hashes. Run `_maintenance/sync_bundle_sources.py` from the "
        "repository root whenever canonical source changes; never hand-edit the copied engine.\n"
        + (
            "Production preflight also requires an accepted launch-decision JSON; copy "
            "`gate_decisions_template.json` to the output path shown in the notebook and "
            "set a decision true only after its cited analysis passes.\n"
            if bundle in ("04_maxmix_master", "05_scans_and_controls")
            else ""
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Regenerate audited Colab notebooks")
    parser.add_argument(
        "--bundle", action="append", choices=BUNDLES,
        help="regenerate only this bundle (repeatable); default regenerates every bundle",
    )
    args = parser.parse_args(argv)
    pilot_plan_text = (ROOT / "pilot_plan.json").read_text(encoding="utf-8")
    selected = args.bundle or BUNDLES
    for bundle in selected:
        root = bundle_path(ROOT, bundle)
        if bundle != "03_h1_modular_response":
            (root / "pilot_plan.json").write_text(pilot_plan_text, encoding="utf-8")
        (root / "run_production_bundle.ipynb").write_text(
            json.dumps(bundle_notebook(bundle), indent=1) + "\n", encoding="utf-8"
        )
        if bundle not in ("04_maxmix_operator_cft", "05_pure_tangent_stability"):
            (root / "README.md").write_text(readme(bundle), encoding="utf-8")
        print(f"[built] {bundle}")
    completion = bundle_path(ROOT, "01_p1_existing_completion") / "finish_existing_P1.ipynb"
    notebook_paths = [
        *(bundle_path(ROOT, bundle) / "run_production_bundle.ipynb" for bundle in selected),
        completion,
    ]
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        final_source = "".join(notebook["cells"][-1].get("source", []))
        if final_source != DISCONNECT_SOURCE:
            raise RuntimeError(
                f"nonstandard Colab disconnect cell in {path.relative_to(ROOT)}"
            )
    print(f"[validated disconnect cell] {len(notebook_paths)} notebooks")
    print(f"[preserved byte-for-byte] {completion.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
