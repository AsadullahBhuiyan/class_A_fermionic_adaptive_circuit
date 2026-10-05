"""Rebuild the standalone notebook, config and local deployment checksums."""
from pathlib import Path
import hashlib
import json
import pprint

from run_campaign import expected_config

ROOT = Path(__file__).resolve().parent
NOTEBOOK = "run_hard_wall_full_measurement_purification.ipynb"


def cell(source, kind="code", identity="cell"):
    value = {"cell_type": kind, "metadata": {}, "id": identity,
             "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build():
    config = expected_config()
    cells = [
        cell("# Hard-wall full-measurement purification: alpha1=1 and 3\n\n"
             "Nx=20, Ny=30, T=60, 100 independent trajectories per alpha. "
             "Hard support truncation, **meas_slab_only=False**, perfect correction, no postselection. "
             "The entire physical layer starts maximally mixed; no exterior projection before cycle zero. "
             "Both cases save every-cycle full-system occupation spectra and scalar entropy/charge. "
             "Alpha1=1 additionally saves every-cycle entropy/charge-variance contours and one final "
             "minimum-magnitude Lyapunov eigenmode per sample. Both save final covariance. "
             "The notebook runs alpha1=1 then alpha1=3, in separate 100-sample execution batches. "
             "New output collection; old campaigns stay untouched. Allow roughly 5–8 A100 hours as "
             "an unbenchmarked planning estimate; use the measured cycle timing printed during execution.\n\n"
             "For C=(G+I)/2, save its raw eigenvalues nu at t=0,...,60. "
             "The slow mode minimizes abs(log((1-nu)/nu)/(2T)) at T=60, not the most negative signed rate. "
             "The finite-mode selection is 1e-9 < nu < 1-1e-9; ties/unresolved modes are flagged. "
             "Entropy uses natural logarithms and the historical 1e-12 kernel cutoff; tiny nonzero "
             "entropy in a purified state is a numerical floor, not residual mixedness.\n",
             "markdown", "intro"),
        cell("from google.colab import drive\ndrive.mount('/content/drive')\n", identity="mount"),
        cell("from pathlib import Path\nimport json\nimport torch\n\n"
             "REPORT_ONLY = False\n"
             "MAX_NEW_EXECUTION_BATCHES = None  # set 1 to limit this session\n"
             f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{ROOT.name}')\n"
             f"LOCAL_BUNDLE_DIR = Path('/content/{ROOT.name}')\n"
             "SCRATCH_ROOT = Path('/content/hard_wall_full_measurement_purification_scratch')\n"
             "# Locked scientific contract; resume/session controls are above.\n"
             "CONFIG = " + pprint.pformat(config, sort_dicts=False) + "\n"
             "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n"
             "if not REPORT_ONLY:\n"
             "    if not torch.cuda.is_available():\n"
             "        raise RuntimeError('An A100 GPU is required.')\n"
             "    name = torch.cuda.get_device_name(0)\n"
             "    if 'A100' not in name or torch.cuda.get_device_properties(0).total_memory / 1024**3 < 35:\n"
             "        raise RuntimeError(f'Expected A100 40-GB-class runtime, got {name}')\n"
             "    print('[GPU]', name, '; dtype:', torch.complex128, flush=True)\n"
             "print(json.dumps({'config': CONFIG, 'bundle': str(BUNDLE_DRIVE_DIR), 'output': str(OUTPUT_ROOT)}, indent=2), flush=True)\n",
             identity="config"),
        cell("## Stage locally and run/resume\n\nNative outer shard and inner cycle bars. "
             "An interruption resumes the last verified ten-cycle checkpoint. "
             "Use a fresh runtime after updating the uploaded code. "
             "The occupation-check hotfix accepts the exact original source identity so completed "
             "alpha1=1 results are skipped and alpha1=3 can resume its cycle-30 checkpoint. "
             "Suspect spectra are rechecked independently on CPU at the unchanged 1e-9 tolerance; "
             "a confirmed violation stops with full-precision diagnostics, not a relaxed threshold.\n",
             "markdown", "run-explanation"),
        cell("import importlib.util\nimport shutil\nimport sys\n\n"
             "if not BUNDLE_DRIVE_DIR.is_dir():\n"
             "    raise FileNotFoundError(f'Missing uploaded folder: {BUNDLE_DRIVE_DIR}')\n"
             "LOCAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)\n"
             "for relative in ['run_campaign.py', 'purification_observer.py', 'src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py']:\n"
             "    destination = LOCAL_BUNDLE_DIR / relative\n"
             "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
             "    shutil.copy2(BUNDLE_DRIVE_DIR / relative, destination)\n"
             "for name in ['classA_U1FGTN_gpu', 'occupied_frame_gpu', 'purification_observer']:\n"
             "    loaded = sys.modules.get(name)\n"
             "    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(LOCAL_BUNDLE_DIR):\n"
             "        raise RuntimeError(f'{name} was imported from another campaign. Restart the runtime first.')\n"
             "# Reload the observer after copying an updated bundle in this runtime.\n"
             "sys.modules.pop('purification_observer', None)\n"
             "config_path = LOCAL_BUNDLE_DIR / 'resolved_config.json'\n"
             "config_path.write_text(json.dumps(CONFIG, indent=2) + '\\n')\n"
             "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
             "spec = importlib.util.spec_from_file_location('_full_measurement_purification_runner', LOCAL_BUNDLE_DIR / 'run_campaign.py')\n"
             "runner = importlib.util.module_from_spec(spec)\n"
             "sys.modules[spec.name] = runner\n"
             "spec.loader.exec_module(runner)\n"
             "args = ['--config', str(config_path), '--output-root', str(OUTPUT_ROOT), '--scratch-root', str(SCRATCH_ROOT)]\n"
             "if REPORT_ONLY:\n"
             "    args.append('--report-only')\n"
             "if MAX_NEW_EXECUTION_BATCHES is not None:\n"
             "    args.extend(['--max-new-execution-batches', str(int(MAX_NEW_EXECUTION_BATCHES))])\n"
             "print('[launch in notebook kernel]', runner.__file__, flush=True)\n"
             "runner.main(args)\n"
             "print('[notebook] requested queue completed successfully', flush=True)\n", identity="run"),
        cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n", identity="disconnect"),
    ]
    return {"cells": cells, "metadata": {"accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}


def main():
    (ROOT / "campaign_config.json").write_text(json.dumps(expected_config(), indent=2) + "\n")
    (ROOT / NOTEBOOK).write_text(json.dumps(build(), indent=1) + "\n")
    names = ["README.md", "build_notebook.py", "campaign_config.json", NOTEBOOK,
             "run_campaign.py", "purification_observer.py",
             "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py"]
    files = {}
    for name in names:
        raw = (ROOT / name).read_bytes()
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (ROOT / "deployment_manifest.json").write_text(json.dumps({
        "schema": "simple_colab_bundle_manifest_v1", "bundle": ROOT.name,
        "sampling_revision": expected_config()["sampling_revision"], "files": files,
    }, indent=2) + "\n")
    print(ROOT / NOTEBOOK)


if __name__ == '__main__':
    main()
