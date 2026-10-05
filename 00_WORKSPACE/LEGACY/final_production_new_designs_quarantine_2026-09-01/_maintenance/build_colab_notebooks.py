#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bundle_layout import bundle_path  # noqa: E402

PRODUCTION_COLLECTION = (
    "classA_final_production_outputs/production_10sample_v4_occupied_frame_cycle_resolved"
)
PILOT_COLLECTION = (
    "classA_pilot_outputs/production_10sample_v4_occupied_frame_cycle_resolved"
)
BUNDLES = [
    "01_p1_chern_dynamics",
    "02_wall_cft_windows",
    "03_h1_modular_response",
    "04_maxmix_operator_cft",
    "05_pure_tangent_stability",
    "07_log_gram_alpha_scan",
    "08_h1_endpoint_packet",
]
V4_OPERATIONAL_RELEASE = "v4-drivefs-independent-20260901-r2"

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
    "08_h1_endpoint_packet": "translated-cut endpoint packets with exact-benchmark wall orientation",
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


def standalone_redesign_notebook(bundle: str) -> dict:
    analysis = "g4_analysis.py" if bundle.startswith("04_") else "g5_analysis.py"
    example = "G4_N20x20_wall" if bundle.startswith("04_") else "G5_N20x20_wall"
    cells = [
        markdown(
            f"# {bundle}: standalone resumable A100 campaign\n\n"
            f"{DESCRIPTIONS[bundle]}. This bundle uses its own versioned 25-trajectory "
            "output collection and never imports or pools legacy G4/G5 shards. The "
            "runtime check below must pass before any pilot or production work starts."
        ),
        markdown("## 1. Mount Drive and inspect the immutable contract\n"),
        code(
            "from pathlib import Path\n"
            "import importlib.util, json, subprocess, sys\n"
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n"
            "MYDRIVE = Path('/content/drive/MyDrive')\n"
            "CAMPAIGN_ROOT = MYDRIVE / 'final_production_new_designs'\n"
            f"bundle_root = CAMPAIGN_ROOT / {bundle!r}\n"
            "if not (bundle_root / 'run_bundle.py').is_file():\n"
            "    raise FileNotFoundError(f'Missing uploaded bundle: {bundle_root}')\n"
            "config = json.loads((bundle_root / 'production_config.json').read_text())\n"
            "runner_path = bundle_root / 'run_bundle.py'\n"
            f"runner_spec = importlib.util.spec_from_file_location({'_colab_' + bundle + '_runner'!r}, runner_path)\n"
            "if runner_spec is None or runner_spec.loader is None:\n"
            "    raise ImportError(f'Cannot load runner: {runner_path}')\n"
            "runner_module = importlib.util.module_from_spec(runner_spec)\n"
            "sys.modules[runner_spec.name] = runner_module\n"
            "runner_spec.loader.exec_module(runner_module)\n"
            "print(json.dumps(config, indent=2))\n"
        ),
        markdown("## 2. Verify the A100 runtime\n"),
        code(
            "RUN_A100_PREFLIGHT = True\n"
            "if RUN_A100_PREFLIGHT:\n"
            "    subprocess.run([sys.executable, '-u', str(bundle_root / 'run_bundle.py'), 'preflight'], check=True)\n"
        ),
        markdown("## 3. Select and run the checksum-resumable queue\n"),
        code(
            "RUN_MODE = 'production'  # or 'pilot'\n"
            f"CASE_ID = None           # e.g. {example!r}; None runs every case\n"
            "SHARD_INDEX = None        # production: 0..4; None runs every shard\n"
            "RUN_QUEUE = False\n"
            "if RUN_MODE not in ('production', 'pilot'):\n"
            '    raise ValueError("RUN_MODE must be \'production\' or \'pilot\'")\n' 
            "if RUN_MODE == 'pilot' and SHARD_INDEX is not None:\n"
            "    raise ValueError('SHARD_INDEX is not used in pilot mode')\n"
            "collection = (config['production_output_collection'] if RUN_MODE == 'production' else config['pilot_output_collection'])\n"
            "output_root = MYDRIVE / collection\n"
            "runner_argv = [RUN_MODE, '--output-root', str(output_root)]\n"
            "if CASE_ID is not None:\n"
            "    runner_argv += ['--case-id', CASE_ID]\n"
            "if SHARD_INDEX is not None:\n"
            "    runner_argv += ['--shard-index', str(SHARD_INDEX)]\n"
            "command = [sys.executable, '-u', str(runner_path), *runner_argv]\n"
            "print('[output root]', output_root)\n"
            "print('[launch]', ' '.join(command))\n"
            "if RUN_QUEUE:\n"
            "    returncode = runner_module.main(runner_argv)\n"
            "    if returncode not in (None, 0):\n"
            "        raise RuntimeError(f'Runner returned nonzero status {returncode}')\n"
        ),
        markdown("## 4. Analyze only checksum-verified complete shards\n"),
        code(
            "RUN_ANALYSIS = False\n"
            "if RUN_ANALYSIS:\n"
            "    subprocess.run([sys.executable, '-u', str(bundle_root / 'src' / "
            + repr(analysis)
            + "), '--archive-root', str(output_root), '--bundle-root', str(bundle_root)], check=True)\n"
        ),
        markdown("## Final. Disconnect this completed Colab runtime\n"),
        code(DISCONNECT_SOURCE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def log_gram_notebook() -> dict:
    bundle = "07_log_gram_alpha_scan"
    cells = [
        markdown(
            "# Ambient Log-Gram Alpha Scan\n\n"
            "Runs the checkpoint-only max-mix/pure campaign through the canonical GPU "
            "circuit runner. The hard exterior preparation is replayable but excluded "
            "from the cocycle; Choi tracking is disabled. Pilot and production are both "
            "opt-in and independently checksum-resumable."
        ),
        markdown("## 1. Mount Drive and inspect the immutable contract\n"),
        code(
            "from google.colab import drive\n"
            "drive.mount('/content/drive', force_remount=False)\n"
            "from pathlib import Path\n"
            "import importlib.util, json, subprocess, sys\n"
            "MYDRIVE = Path('/content/drive/MyDrive')\n"
            "CAMPAIGN_ROOT = MYDRIVE / 'final_production_new_designs'\n"
            "BUNDLE_ROOT = CAMPAIGN_ROOT / '07_log_gram_alpha_scan'\n"
            "if not (BUNDLE_ROOT / 'run_bundle.py').is_file():\n"
            "    raise FileNotFoundError(f'Missing uploaded bundle: {BUNDLE_ROOT}')\n"
            "config = json.loads((BUNDLE_ROOT / 'production_config.json').read_text())\n"
            "runner_path = BUNDLE_ROOT / 'run_bundle.py'\n"
            "runner_spec = importlib.util.spec_from_file_location('_colab_log_gram_runner', runner_path)\n"
            "if runner_spec is None or runner_spec.loader is None:\n"
            "    raise ImportError(f'Cannot load runner: {runner_path}')\n"
            "runner_module = importlib.util.module_from_spec(runner_spec)\n"
            "sys.modules[runner_spec.name] = runner_module\n"
            "runner_spec.loader.exec_module(runner_module)\n"
            "print(json.dumps(config, indent=2))\n"
        ),
        markdown("## 2. Verify the A100 runtime\n"),
        code(
            "RUN_A100_PREFLIGHT = True\n"
            "if RUN_A100_PREFLIGHT:\n"
            "    subprocess.run([sys.executable, '-u', str(BUNDLE_ROOT / 'run_bundle.py'), 'preflight'], check=True)\n"
        ),
        markdown(
            "## 3. Optional pilot\n\n"
            "The 12 pilot cases use two trajectories each and replay every saved record "
            "before accepting the shard. The pilot is disabled by default.\n"
        ),
        code(
            "PILOT_ROOT = MYDRIVE / 'classA_pilot_outputs' / config['campaign_revision']\n"
            "PILOT_CASE_ID = None\n"
            "RUN_PILOT = False\n"
            "pilot_argv = ['pilot', '--output-root', str(PILOT_ROOT), '--replay-check']\n"
            "if PILOT_CASE_ID is not None:\n"
            "    pilot_argv += ['--case-id', PILOT_CASE_ID]\n"
            "pilot_command = [sys.executable, '-u', str(runner_path), *pilot_argv]\n"
            "print('[pilot]', ' '.join(pilot_command))\n"
            "if RUN_PILOT:\n"
            "    returncode = runner_module.main(pilot_argv)\n"
            "    if returncode not in (None, 0):\n"
            "        raise RuntimeError(f'Runner returned nonzero status {returncode}')\n"
        ),
        markdown(
            "## 4. Resumable production queue\n\n"
            "Set `RUN_PRODUCTION=True` only after the pilot manifests validate. Existing "
            "immutable shards are checksum-verified and skipped.\n"
        ),
        code(
            "PRODUCTION_ROOT = MYDRIVE / config['output_collection']\n"
            "CASE_ID = None       # exact case ID, or None for all 220 cases\n"
            "SHARD_INDEX = None   # 0..4, or None for all five shards\n"
            "RUN_PRODUCTION = False\n"
            "production_argv = ['production', '--output-root', str(PRODUCTION_ROOT)]\n"
            "if CASE_ID is not None:\n"
            "    production_argv += ['--case-id', CASE_ID]\n"
            "if SHARD_INDEX is not None:\n"
            "    production_argv += ['--shard-index', str(SHARD_INDEX)]\n"
            "production_command = [sys.executable, '-u', str(runner_path), *production_argv]\n"
            "print('[production]', ' '.join(production_command))\n"
            "if RUN_PRODUCTION:\n"
            "    returncode = runner_module.main(production_argv)\n"
            "    if returncode not in (None, 0):\n"
            "        raise RuntimeError(f'Runner returned nonzero status {returncode}')\n"
        ),
        markdown("## Final. Disconnect this completed Colab runtime\n"),
        code(DISCONNECT_SOURCE),
    ]
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def bundle_notebook(bundle: str) -> dict:
    if bundle in ("04_maxmix_operator_cft", "05_pure_tangent_stability"):
        return standalone_redesign_notebook(bundle)
    if bundle == "07_log_gram_alpha_scan":
        return log_gram_notebook()
    trajectory_text = (
        "25 independent trajectories: five-trajectory final archives at L=16,24,32 "
        "and one-trajectory final archives at L=64, with one rolling per-cycle resume checkpoint"
        if bundle == "01_p1_chern_dynamics"
        else "25 independent trajectories in five immutable five-trajectory shards"
        if bundle in ("02_wall_cft_windows", "03_h1_modular_response", "04_maxmix_operator_cft", "05_pure_tangent_stability", "08_h1_endpoint_packet")
        else "10 independent trajectories in two immutable five-trajectory shards"
    )
    session_hours_default = "7.5" if bundle == "01_p1_chern_dynamics" else "None"
    server_commit_bundle = bundle in ("01_p1_chern_dynamics", "08_h1_endpoint_packet")
    drive_import_source = (
        "from google.colab import auth, drive\n"
        "drive.mount('/content/drive', force_remount=False)\n\n"
        "auth.authenticate_user()  # one Drive API authorization per fresh runtime\n\n"
        if server_commit_bundle
        else
        "from google.colab import drive\n"
        "drive.mount('/content/drive', force_remount=False)\n\n"
    )
    if server_commit_bundle:
        campaign_setup_source = (
            "DRIVE_CAMPAIGN_ROOT = MYDRIVE / 'final_production_new_designs_v4'\n"
            f"EXPECTED_OPERATIONAL_RELEASE = {V4_OPERATIONAL_RELEASE!r}\n"
            "import os, shutil, tempfile\n"
            "import google.auth, google_auth_httplib2, httplib2\n"
            "from googleapiclient.discovery import build as _build_google_service\n"
            "_credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive'])\n"
            "_authorized_http = google_auth_httplib2.AuthorizedHttp(_credentials, http=httplib2.Http(timeout=60))\n"
            "_drive_service = _build_google_service('drive', 'v3', http=_authorized_http, cache_discovery=False)\n"
            "_drive_fields = 'nextPageToken,files(id,name,parents,size,sha256Checksum,trashed,mimeType)'\n"
            "def _drive_query_escape(value):\n"
            "    return str(value).replace('\\\\', '\\\\\\\\').replace(\"'\", \"\\\\'\")\n"
            "def _drive_named(parent_id, name):\n"
            "    query = f\"'{_drive_query_escape(parent_id)}' in parents and name = '{_drive_query_escape(name)}' and trashed = false\"\n"
            "    rows, page_token = [], None\n"
            "    while True:\n"
            "        response = _drive_service.files().list(q=query, spaces='drive', fields=_drive_fields, pageSize=100, pageToken=page_token).execute()\n"
            "        rows.extend(response.get('files', []))\n"
            "        page_token = response.get('nextPageToken')\n"
            "        if not page_token:\n"
            "            break\n"
            "    if len(rows) != 1:\n"
            "        raise RuntimeError(f'Drive API expected one {name!r} below parent {parent_id}, found {len(rows)}')\n"
            "    return rows[0]\n"
            "def _drive_folder(parent_id, name):\n"
            "    row = _drive_named(parent_id, name)\n"
            "    if row.get('mimeType') != 'application/vnd.google-apps.folder':\n"
            "        raise RuntimeError(f'Drive deployment component is not a folder: {name!r}')\n"
            "    return row\n"
            "def _drive_download_verified(row, expected=None):\n"
            "    raw = _drive_service.files().get_media(fileId=str(row['id'])).execute()\n"
            "    if not isinstance(raw, (bytes, bytearray)):\n"
            "        raise RuntimeError(f'Drive API returned non-binary content for {row.get(\"name\")!r}')\n"
            "    raw = bytes(raw)\n"
            "    actual_sha = hashlib.sha256(raw).hexdigest()\n"
            "    if len(raw) != int(row.get('size', -1)) or actual_sha != row.get('sha256Checksum'):\n"
            "        raise RuntimeError(f'Drive server metadata/readback mismatch for {row.get(\"name\")!r}')\n"
            "    if expected is not None and (len(raw) != int(expected['bytes']) or actual_sha != expected['sha256']):\n"
            "        raise RuntimeError(f'Deployment manifest mismatch for {row.get(\"name\")!r}')\n"
            "    return raw\n"
            "_campaign_row = _drive_folder('root', DRIVE_CAMPAIGN_ROOT.name)\n"
            "_manifest_row = _drive_named(str(_campaign_row['id']), 'deployment_manifest.json')\n"
            "_manifest_raw = _drive_download_verified(_manifest_row)\n"
            "deployment_manifest = json.loads(_manifest_raw.decode('utf-8'))\n"
            "if deployment_manifest.get('schema') != 'classA_v4_deployment_manifest_v1':\n"
            "    raise RuntimeError('Server deployment manifest has the wrong schema')\n"
            "if deployment_manifest.get('operational_release') != EXPECTED_OPERATIONAL_RELEASE:\n"
            "    raise RuntimeError(f'Wrong server deployment release: {deployment_manifest.get(\"operational_release\")!r}')\n"
            "if deployment_manifest.get('campaign_parent') != DRIVE_CAMPAIGN_ROOT.name:\n"
            "    raise RuntimeError('Server deployment manifest names the wrong campaign parent')\n"
            "if set(deployment_manifest.get('bundles', [])) != {'01_p1_chern_dynamics', '08_h1_endpoint_packet'}:\n"
            "    raise RuntimeError('Server deployment manifest has the wrong bundle set')\n"
            "deployment_files = deployment_manifest.get('files', {})\n"
            "if not deployment_files:\n"
            "    raise RuntimeError('The v4 deployment manifest is empty or missing its file table')\n"
            "if int(deployment_manifest.get('file_count', -1)) != len(deployment_files):\n"
            "    raise RuntimeError('The v4 deployment manifest file count is inconsistent')\n"
            "_expected_notebooks = {'01_p1_chern_dynamics/run_production_bundle.ipynb', '08_h1_endpoint_packet/run_production_bundle.ipynb'}\n"
            "notebook_manifest_files = {relative: expected for relative, expected in deployment_files.items() if str(relative).endswith('.ipynb')}\n"
            "if set(notebook_manifest_files) != _expected_notebooks:\n"
            "    raise RuntimeError('The v4 deployment manifest has the wrong notebook set')\n"
            "runtime_deployment_files = {relative: expected for relative, expected in deployment_files.items() if relative not in notebook_manifest_files}\n"
            "if not runtime_deployment_files:\n"
            "    raise RuntimeError('The v4 deployment manifest has no executable payload')\n"
            "def _deployment_path(root, relative):\n"
            "    clean = Path(str(relative))\n"
            "    if clean.is_absolute() or clean == Path('.') or '..' in clean.parts:\n"
            "        raise RuntimeError(f'Unsafe deployment-manifest path: {relative!r}')\n"
            "    return root / clean\n"
            "deployment_key = hashlib.sha256(_manifest_raw).hexdigest()[:16]\n"
            "LOCAL_CAMPAIGN_ROOT = Path('/content') / f'final_production_new_designs_v4_{deployment_key}'\n"
            "def _verified_deployment(root):\n"
            "    for relative, expected in runtime_deployment_files.items():\n"
            "        candidate = _deployment_path(root, relative)\n"
            "        if not candidate.is_file() or candidate.stat().st_size != int(expected['bytes']):\n"
            "            return False\n"
            "        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected['sha256']:\n"
            "            return False\n"
            "    return True\n"
            "if not _verified_deployment(LOCAL_CAMPAIGN_ROOT):\n"
            "    stage_root = Path(tempfile.mkdtemp(prefix='.classA_v4_stage.', dir='/content'))\n"
            "    try:\n"
            "        _folder_ids = {Path('.'): str(_campaign_row['id'])}\n"
            "        for relative, expected in runtime_deployment_files.items():\n"
            "            clean = Path(str(relative))\n"
            "            parent_id = str(_campaign_row['id'])\n"
            "            walked = Path('.')\n"
            "            for component in clean.parts[:-1]:\n"
            "                walked = walked / component\n"
            "                if walked not in _folder_ids:\n"
            "                    _folder_ids[walked] = str(_drive_folder(parent_id, component)['id'])\n"
            "                parent_id = _folder_ids[walked]\n"
            "            remote_row = _drive_named(parent_id, clean.name)\n"
            "            if remote_row.get('mimeType') == 'application/vnd.google-apps.folder':\n"
            "                raise RuntimeError(f'Deployment file is unexpectedly a folder: {relative}')\n"
            "            raw = _drive_download_verified(remote_row, expected)\n"
            "            destination = _deployment_path(stage_root, relative)\n"
            "            destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "            destination.write_bytes(raw)\n"
            "            if destination.stat().st_size != int(expected['bytes']) or hashlib.sha256(destination.read_bytes()).hexdigest() != expected['sha256']:\n"
            "                raise RuntimeError(f'Deployment API download failed verification: {relative}')\n"
            "        (stage_root / 'deployment_manifest.json').write_bytes(_manifest_raw)\n"
            "        if LOCAL_CAMPAIGN_ROOT.exists():\n"
            "            shutil.rmtree(LOCAL_CAMPAIGN_ROOT)\n"
            "        os.replace(stage_root, LOCAL_CAMPAIGN_ROOT)\n"
            "    except BaseException:\n"
            "        shutil.rmtree(stage_root, ignore_errors=True)\n"
            "        raise\n"
            "if not _verified_deployment(LOCAL_CAMPAIGN_ROOT):\n"
            "    raise RuntimeError('Local v4 deployment failed its final hash verification')\n"
            "CAMPAIGN_ROOT = LOCAL_CAMPAIGN_ROOT\n"
            "print(f'[server deployment release] {EXPECTED_OPERATIONAL_RELEASE}')\n"
            "print(f'[verified executable payloads] {len(runtime_deployment_files)}; notebook records are audit-only because Colab mutates open notebook outputs')\n"
            "print(f'[local verified deployment] {CAMPAIGN_ROOT}')\n"
        )
    else:
        campaign_setup_source = (
            "CAMPAIGN_ROOT = MYDRIVE / 'final_production_new_designs'\n"
        )
    collection_source = (
        f"production_collection = bundle_config.get('production_output_collection', {PRODUCTION_COLLECTION!r})\n"
        f"pilot_collection = bundle_config.get('pilot_output_collection', {PILOT_COLLECTION!r})\n"
        if bundle in ("01_p1_chern_dynamics", "02_wall_cft_windows", "03_h1_modular_response", "04_maxmix_operator_cft", "05_pure_tangent_stability", "08_h1_endpoint_packet")
        else
        f"production_collection = {PRODUCTION_COLLECTION!r}\n"
        f"pilot_collection = {PILOT_COLLECTION!r}\n"
    )
    session_dir_source = (
        "session_dir = runner_module.local_session_root(output_collection=collection, bundle=BUNDLE)\n"
        if server_commit_bundle
        else
        "session_dir = output_root / '_bundle_sessions'\n"
    )
    failure_diagnostics_source = (
        "except BaseException:\n"
        "    try:\n"
        "        sessions = sorted(session_dir.glob(f'*_{BUNDLE}_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)\n"
        "        if sessions:\n"
        "            latest = json.loads(sessions[0].read_text())\n"
        "            print('[saved bundle failure: local telemetry]')\n"
        "            print(json.dumps({'session': str(sessions[0]), 'failure': latest.get('failure'), 'current': latest.get('current'), 'log': latest.get('session_log'), 'telemetry_errors': latest.get('telemetry_errors')}, indent=2))\n"
        "    except Exception as diagnostic_error:\n"
        "        print(f'[local failure diagnostics unavailable] {type(diagnostic_error).__name__}: {diagnostic_error}')\n"
        "    raise\n"
        if server_commit_bundle
        else
        "except BaseException:\n"
        "    sessions = sorted(session_dir.glob(f'*_{BUNDLE}_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)\n"
        "    if sessions:\n"
        "        latest = json.loads(sessions[0].read_text())\n"
        "        print('[saved bundle failure]')\n"
        "        print(json.dumps({'session': str(sessions[0]), 'failure': latest.get('failure'), 'current': latest.get('current'), 'log': latest.get('session_log')}, indent=2))\n"
        "    raise\n"
    )
    cells = [
        markdown(
            f"# {bundle}: independent resumable A100 bundle\n\n"
            f"{DESCRIPTIONS[bundle]}. This notebook launches only `{bundle}`; it never "
            f"continues into another experiment. The sampling layout uses {trajectory_text}. Run the "
            "default report-only pass first, inspect the exact verified/pending inventory, "
            "then set `RESUME_REPORT_ONLY=False` to compute only missing shards."
        ),
        markdown("## 1. Mount Drive and locate the campaign package\n"),
        code(
            "from pathlib import Path\n"
            "import hashlib, importlib.util, json, subprocess, sys\n\n"
            f"{drive_import_source}"
            "MYDRIVE = Path('/content/drive/MyDrive')\n"
            f"{campaign_setup_source}"
            "runner = CAMPAIGN_ROOT / 'colab_bundle_runner.py'\n"
            f"bundle_root = CAMPAIGN_ROOT / {bundle!r}\n"
            "if not runner.is_file() or not (bundle_root / 'run_bundle.py').is_file():\n"
            f"    raise FileNotFoundError('Upload this bundle with the {('final_production_new_designs_v4' if bundle in ('01_p1_chern_dynamics', '08_h1_endpoint_packet') else 'final_production_new_designs')} parent support files')\n"
            "def _load_runner_module(module_path, module_name):\n"
            "    module_spec = importlib.util.spec_from_file_location(module_name, module_path)\n"
            "    if module_spec is None or module_spec.loader is None:\n"
            "        raise ImportError(f'Cannot load runner: {module_path}')\n"
            "    module = importlib.util.module_from_spec(module_spec)\n"
            "    sys.modules[module_spec.name] = module\n"
            "    module_spec.loader.exec_module(module)\n"
            "    return module\n"
            "runner_module = _load_runner_module(runner, '_colab_bundle_runner')\n"
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
            f"MAX_SESSION_GPU_HOURS_OVERRIDE = {session_hours_default}\n"
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
            f"## {5 if bundle in ('01_p1_chern_dynamics', '08_h1_endpoint_packet') else 4 if bundle == '03_h1_modular_response' else 3}. "
            "Verify, resume, and stop at the end of this bundle\n"
        ),
        code(
            "if not RUN_QUEUE:\n"
            "    raise RuntimeError('RUN_QUEUE is False')\n"
            "runner_argv = [\n"
            "    '--drive-root', str(MYDRIVE),\n"
            "    '--bundle', BUNDLE, '--profile', RUN_PROFILE,\n"
            "    '--heartbeat-seconds', str(HEARTBEAT_SECONDS),\n"
            "]\n"
            "for prefix in CASE_PREFIXES:\n"
            "    runner_argv += ['--case-prefix', str(prefix)]\n"
            "for case_id in CASE_IDS:\n"
            "    runner_argv += ['--case-id', str(case_id)]\n"
            "if MAX_SESSION_GPU_HOURS_OVERRIDE is not None:\n"
            "    runner_argv += ['--max-session-hours', str(MAX_SESSION_GPU_HOURS_OVERRIDE)]\n"
            "if PREFLIGHT_ONLY:\n"
            "    runner_argv.append('--preflight-only')\n"
            "if RESUME_REPORT_ONLY:\n"
            "    runner_argv.append('--resume-report-only')\n"
            "command = [sys.executable, '-u', str(runner), *runner_argv]\n"
            f"{collection_source}"
            "collection = production_collection if RUN_PROFILE == 'production' else pilot_collection\n"
            "output_root = MYDRIVE / collection\n"
            f"{session_dir_source}"
            "build_id = hashlib.sha256(runner.read_bytes()).hexdigest()[:16]\n"
            "print('[bundle dashboard]')\n"
            "print(json.dumps({\n"
            "    'bundle': BUNDLE, 'profile': RUN_PROFILE, 'output_root': str(output_root),\n"
            "    'runner_build_id': build_id, 'heartbeat_seconds': HEARTBEAT_SECONDS,\n"
            "    'session_dir': str(session_dir), 'case_prefixes': CASE_PREFIXES,\n"
            "    'case_ids': CASE_IDS, 'resume_report_only': RESUME_REPORT_ONLY,\n"
            "}, indent=2))\n"
            "print(f'[launch bundle] {\" \".join(command)}', flush=True)\n"
            "try:\n"
            "    returncode = runner_module.main(runner_argv)\n"
            "    if returncode not in (None, 0):\n"
            "        raise RuntimeError(f'Runner returned nonzero status {returncode}')\n"
            f"{failure_diagnostics_source}"
        ),
        markdown("## Final. Disconnect this completed Colab runtime\n"),
        code(DISCONNECT_SOURCE),
    ]
    if bundle in ("01_p1_chern_dynamics", "08_h1_endpoint_packet"):
        cells[5:5] = [
            markdown(
                "## 3. Verify server-side Drive commits\n\n"
                "This writes a tiny probe through the Drive API, verifies its server "
                "parent, byte count, and SHA-256, then removes it. DriveFS visibility "
                "alone is never accepted as durability.\n"
            ),
            code(
                "RUN_REMOTE_COMMIT_PROBE = True\n"
                "if RUN_REMOTE_COMMIT_PROBE:\n"
                "    remote_commit_module = _load_runner_module(CAMPAIGN_ROOT / 'drive_remote_commit.py', '_colab_operational_drive_remote_commit')\n"
                "    committer = remote_commit_module.DriveRemoteCommitter(drive_root=MYDRIVE)\n"
                "    probe_path = MYDRIVE / bundle_config['production_output_collection'] / OUTPUT_BUNDLE / '_remote_commit_probe.json'\n"
                "    probe = {'schema': 'classA_remote_commit_probe_v1', 'bundle': BUNDLE}\n"
                "    remote_commit_module.publish_json(committer, probe, probe_path, replace=True, required_headroom_bytes=0)\n"
                "    verified_probe = committer.path_commit_record(probe_path)\n"
                "    committer.verify_record_for_path(verified_probe, probe_path)\n"
                "    committer.delete_verified(verified_probe)\n"
                "    print('[remote commit probe] passed and cleaned up')\n"
            ),
        ]
    if bundle == "01_p1_chern_dynamics":
        cells[7:7] = [
            markdown(
                "## 4. Required A100 cost and memory preflight\n\n"
                "The production queue now runs or resumes this qualification automatically before its first missing shard. This optional cell lets you start it early and inspect the result. It advances the "
                "`L=64`, `n_shell=1` sample-0 trajectory through atomic Drive checkpoints after every completed physical cycle. "
                "Rerunning continues from the next cycle with the exact occupied frame, RNG, and observer state. Once cycle 64 finishes, "
                "the trajectory is archived and verified as real production work, and a version-matched safety receipt references its archive hash. "
                "The queue budgets one resumable cycle at a time within the 7.5-hour session cap. The production runner refuses to launch the remaining matrix without a current safe receipt.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "P1_PREFLIGHT_MAX_RUNTIME_SECONDS = int(6.75 * 3600)\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    storage = runner_module.server_storage_status(drive_root=MYDRIVE, output_collection=bundle_config['production_output_collection'], working_limit_gb=12.0, absolute_edge_gb=14.0, required_headroom_gb=1.25)\n"
                "    print(json.dumps(storage, indent=2, sort_keys=True))\n"
                "    if not storage['clear_to_run']:\n"
                "        raise RuntimeError('Server-authoritative Drive storage guard blocked P1 preflight')\n"
                "    preflight_module = _load_runner_module(bundle_root / 'run_bundle.py', '_colab_p1_preflight_runner')\n"
                "    preflight_argv = ['--bundle-root', str(bundle_root), '--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight', '--max-runtime-seconds', str(P1_PREFLIGHT_MAX_RUNTIME_SECONDS)]\n"
                "    returncode = preflight_module.main(preflight_argv)\n"
                "    if returncode not in (None, 0):\n"
                "        raise RuntimeError(f'Preflight runner returned nonzero status {returncode}')\n"
            ),
        ]
        cells[-2:-2] = [
            markdown("## 6. Merge the completed matrix and create the reference-style figure\n"),
            code(
                "RUN_P1_ANALYSIS = False\n"
                "if RUN_P1_ANALYSIS:\n"
                "    subprocess.run([sys.executable, '-u', str(CAMPAIGN_ROOT / 'server_verified_analysis.py'), '--campaign-root', str(CAMPAIGN_ROOT), '--drive-root', str(MYDRIVE), '--bundle', BUNDLE], check=True)\n"
            ),
        ]
    elif bundle == "02_wall_cft_windows":
        cells[5:5] = [
            markdown(
                "## 3. Required Ny=60 hard/soft A100 preflight\n\n"
                "The production queue now runs or reuses this qualification automatically before its first missing shard. This optional cell lets you run it early and inspect the receipt. It evaluates a complete five-trajectory "
                "shard for each wall construction at the largest geometry, including all "
                "Ay=30 origins, and writes a version-matched safety receipt. No scientific "
                "parameter is reduced if the preflight is unsafe.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    preflight_module = _load_runner_module(bundle_root / 'run_bundle.py', '_colab_wall_preflight_runner')\n"
                "    preflight_argv = ['--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight']\n"
                "    returncode = preflight_module.main(preflight_argv)\n"
                "    if returncode not in (None, 0):\n"
                "        raise RuntimeError(f'Preflight runner returned nonzero status {returncode}')\n"
            ),
        ]
    elif bundle == "03_h1_modular_response":
        cells[5:5] = [
            markdown(
                "## 3. Required complete-shard A100 preflight\n\n"
                "The production queue now runs or reuses this qualification automatically before its first missing shard. This optional cell lets you run it early and inspect the receipt. It executes all 80 cycles and all six "
                "H1 observations for one complete five-trajectory soft-wall shard, then "
                "writes a version-matched memory/runtime/storage receipt. Production "
                "remains locked if the receipt is missing, stale, or unsafe; no scientific "
                "parameter is reduced automatically.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    preflight_module = _load_runner_module(bundle_root / 'run_bundle.py', '_colab_h1_preflight_runner')\n"
                "    preflight_argv = ['--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight']\n"
                "    returncode = preflight_module.main(preflight_argv)\n"
                "    if returncode not in (None, 0):\n"
                "        raise RuntimeError(f'Preflight runner returned nonzero status {returncode}')\n"
            ),
        ]
        cells[-2:-2] = [
            markdown("## 5. Merge completed H1 shards, analyze, and create figures\n"),
            code(
                "RUN_H1_ANALYSIS = False\n"
                "H1_ANALYSIS_PROFILE = 'production'  # or 'pilot'\n"
                "if RUN_H1_ANALYSIS:\n"
                "    if H1_ANALYSIS_PROFILE not in ('production', 'pilot'):\n"
                "        raise ValueError(\"H1_ANALYSIS_PROFILE must be 'production' or 'pilot'\")\n"
                "    collection_key = f'{H1_ANALYSIS_PROFILE}_output_collection'\n"
                "    archive_root = MYDRIVE / bundle_config[collection_key] / OUTPUT_BUNDLE\n"
                "    analysis_root = archive_root / 'analysis_outputs'\n"
                "    subprocess.run([sys.executable, '-u', str(bundle_root / 'src' / 'h1_modular_analysis.py'), '--archive-root', str(archive_root), '--output-root', str(analysis_root), '--bundle-root', str(bundle_root)], check=True)\n"
                "    print((analysis_root / 'h1_analysis_summary.json').read_text())\n"
            ),
        ]
    elif bundle == "08_h1_endpoint_packet":
        cells[7:7] = [
            markdown(
                "## 4. Required complete-shard A100 preflight\n\n"
                "The production queue runs or reuses this qualification automatically before "
                "its first missing shard. This optional cell executes all 80 circuit cycles and "
                "all 40 translated cuts at every observation cycle for one complete five-trajectory "
                "soft-wall, alpha_1=1 shard. A safe v4 result is archived as the real production shard zero, "
                "so the calculation is not repeated. H1-v4 records raw A100 packet-norm error, "
                "normalizes the exact-unitary packets before extracting handedness, and treats "
                "warning-level raw residuals as non-gating. It writes a version-matched memory, "
                "runtime, storage, and numerical receipt. A hard numerical failure is preserved "
                "under `_failed_qualifications` before production remains locked.\n"
            ),
            code(
                "RUN_A100_PREFLIGHT = False\n"
                "if RUN_A100_PREFLIGHT:\n"
                "    storage = runner_module.server_storage_status(drive_root=MYDRIVE, output_collection=bundle_config['production_output_collection'], working_limit_gb=12.0, absolute_edge_gb=14.0, required_headroom_gb=1.25)\n"
                "    print(json.dumps(storage, indent=2, sort_keys=True))\n"
                "    if not storage['clear_to_run']:\n"
                "        raise RuntimeError('Server-authoritative Drive storage guard blocked H1 preflight')\n"
                "    preflight_module = _load_runner_module(bundle_root / 'run_bundle.py', '_colab_h1_endpoint_preflight_runner')\n"
                "    preflight_argv = ['--drive-root', str(MYDRIVE), '--mode', 'production', '--a100-preflight']\n"
                "    returncode = preflight_module.main(preflight_argv)\n"
                "    if returncode not in (None, 0):\n"
                "        raise RuntimeError(f'Preflight runner returned nonzero status {returncode}')\n"
            ),
        ]
        cells[-2:-2] = [
            markdown("## 6. Merge 11 server-verified H1-v3 shards with 9 H1-v4 shards\n"),
            code(
                "RUN_H1_ENDPOINT_ANALYSIS = False\n"
                "if RUN_H1_ENDPOINT_ANALYSIS:\n"
                "    subprocess.run([sys.executable, '-u', str(CAMPAIGN_ROOT / 'server_verified_analysis.py'), '--campaign-root', str(CAMPAIGN_ROOT), '--drive-root', str(MYDRIVE), '--bundle', BUNDLE], check=True)\n"
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



def readme(bundle: str) -> str:
    if bundle in ("02_wall_cft_windows", "03_h1_modular_response", "08_h1_endpoint_packet"):
        # This standalone redesign has a contract-specific README rather than the
        # legacy ten-sample campaign boilerplate.
        return (bundle_path(ROOT, bundle) / "README.md").read_text(encoding="utf-8")
    if bundle == "01_p1_chern_dynamics":
        return (bundle_path(ROOT, bundle) / "README.md").read_text(encoding="utf-8")
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
        if bundle not in (
            "03_h1_modular_response", "07_log_gram_alpha_scan",
            "08_h1_endpoint_packet",
        ):
            (root / "pilot_plan.json").write_text(pilot_plan_text, encoding="utf-8")
        (root / "run_production_bundle.ipynb").write_text(
            json.dumps(bundle_notebook(bundle), indent=1) + "\n", encoding="utf-8"
        )
        if bundle not in (
            "01_p1_chern_dynamics", "04_maxmix_operator_cft",
            "05_pure_tangent_stability", "07_log_gram_alpha_scan",
            "08_h1_endpoint_packet",
        ):
            (root / "README.md").write_text(readme(bundle), encoding="utf-8")
        print(f"[built] {bundle}")
    notebook_paths = [
        bundle_path(ROOT, bundle) / "run_production_bundle.ipynb"
        for bundle in selected
    ]
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        final_source = "".join(notebook["cells"][-1].get("source", []))
        if final_source != DISCONNECT_SOURCE:
            raise RuntimeError(
                f"nonstandard Colab disconnect cell in {path.relative_to(ROOT)}"
            )
    print(f"[validated disconnect cell] {len(notebook_paths)} notebooks")


if __name__ == "__main__":
    main()
