"""Generate notebook/config/manifest; optionally synchronize canonical sources."""
import argparse
import hashlib
import json
from pathlib import Path
import pprint
import shutil

from run_campaign import default_config, SOURCE_FILES

ROOT = Path(__file__).resolve().parent
NOTEBOOK = 'run_square_hard_wall_random_center_chern.ipynb'


def cell(source, kind='code', name='cell'):
    out = dict(cell_type=kind, id=name, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        out.update(execution_count=None, outputs=[])
    return out


def build():
    cells = [
        cell('# Square hard-wall Chern dynamics: ten random centers\n\n'
             'Nx=Ny=L=20/30/40; 100 pure trajectories each, exactly 40 cycles (observations 0–40). '
             'Perfect correction, hard support truncation, slab-only raster-y measurements, '
             'alpha1=1, alpha2=30, nshell=1, periodic boundaries, complex128. '
             'Cycle zero is after canonical exterior preparation.\n\n'
             'At each cycle and for each trajectory, choose ten distinct random y centers, '
             'with x=L/2 and R=0.2L (4, 6, 8). Interfaces are (5,15), (8,22), (10,30). Periodic wrapping preserves disks crossing y=0. '
             'Compute the three-sector estimator from Gamma=(V V†)^T. '
             'Save every center individually and average centers within each trajectory '
             'before forming the trajectory mean and standard error. No covariance history.\n\n'
             'The first run benchmarks ascending batch sizes 5/10/25/50/100 (2/1 fallback), stopping growth at a measured limit, using the actual '
             'observer. Qualified means <32 GiB peak reserved and <45 minutes forecast including '
             '25% margin. Calibration is not production data. There is no prequalified batch size. '
             'Completed batches resume; an interrupted batch restarts. Use only one writer.\n', 'markdown', 'intro'),
        cell("from google.colab import drive\ndrive.mount('/content/drive')\n", name='mount'),
        cell("from pathlib import Path\n\n"
             "# All scientific and execution settings are visible here. This revision's\n"
             "# scientific/calibration contract is fixed; edits require a new revision.\n"
             "REPORT_ONLY = False\nMAX_NEW_BATCHES = None  # e.g. 1 to stop after one production batch\n"
             f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{ROOT.name}')\n"
             f"LOCAL_BUNDLE_DIR = Path('/content/{ROOT.name}')\n"
             "SCRATCH_ROOT = Path('/content/square_random_center_chern_scratch')\n"
             "CONFIG = " + pprint.pformat(default_config(), sort_dicts=False) + "\n"
             "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n",
             name='config'),
        cell('## Stage locally and run/resume\n\n'
             'Prints source/config identity, measured timings, memory, frozen batch sizes and '
             'remaining-time forecast. Both stderr and stdout are relayed, including tqdm '
             'carriage-return updates. A saved batch longer than one hour stops the queue. '
             'DriveFS readback is checked; it is not independent server-side verification.\n',
             'markdown', 'run-intro'),
        cell("import codecs\nimport json\nimport os\nimport shutil\nimport subprocess\nimport sys\n\n"
             "LOCAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)\n"
             f"for relative in {list(SOURCE_FILES)!r}:\n"
             "    target = LOCAL_BUNDLE_DIR / relative\n"
             "    target.parent.mkdir(parents=True, exist_ok=True)\n"
             "    shutil.copy2(BUNDLE_DRIVE_DIR / relative, target)\n"
             "config_path = LOCAL_BUNDLE_DIR / 'resolved_config.json'\n"
             "config_path.write_text(json.dumps(CONFIG, indent=2))\n"
             "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR / 'run_campaign.py'),\n"
             "           '--config', str(config_path), '--output-root', str(OUTPUT_ROOT),\n"
             "           '--scratch-root', str(SCRATCH_ROOT)]\n"
             "if REPORT_ONLY:\n    command.append('--report-only')\n"
             "if MAX_NEW_BATCHES is not None:\n    command += ['--max-new-batches', str(MAX_NEW_BATCHES)]\n"
             "print('[launch]', ' '.join(command), flush=True)\n"
             "process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)\n"
             "decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')\n"
             "try:\n"
             "    while True:\n"
             "        block = os.read(process.stdout.fileno(), 4096)\n"
             "        if not block:\n            break\n"
             "        sys.stdout.write(decoder.decode(block))\n        sys.stdout.flush()\n"
             "    sys.stdout.write(decoder.decode(b'', final=True))\n"
             "    if process.wait():\n        raise subprocess.CalledProcessError(process.returncode, command)\n"
             "finally:\n"
             "    if process.poll() is None:\n"
             "        process.terminate()\n"
             "        try:\n            process.wait(timeout=10)\n"
             "        except subprocess.TimeoutExpired:\n            process.kill()\n            process.wait()\n"
             "print('[notebook] requested session finished', flush=True)\n", name='run'),
        cell('## Preview: center-averaged trajectories\n\n'
             'Verified completed batches only; labels report actual S. Shading is '
             'std(c_xi, ddof=1)/sqrt(S), not an error over individual centers.\n', 'markdown', 'preview-intro'),
        cell("import importlib.util\nimport matplotlib.pyplot as plt\n"
             "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
             "sys.modules.pop('random_center_observer', None)\n"
             "spec = importlib.util.spec_from_file_location('chern_preview', LOCAL_BUNDLE_DIR / 'run_campaign.py')\n"
             "runner = importlib.util.module_from_spec(spec)\nspec.loader.exec_module(runner)\n"
             "products = runner.preview(OUTPUT_ROOT)\n"
             "fig, ax = plt.subplots(figsize=(5, 3.3))\n"
             "for ny, row in products.items():\n"
             "    line, = ax.plot(row['cycles'], row['mean'], label=f'{ny} x {ny}, S={row[\"samples\"]}')\n"
             "    ax.fill_between(row['cycles'], row['mean']-row['sem'], row['mean']+row['sem'], alpha=.2, color=line.get_color())\n"
             "ax.axhline(1, color='gray', ls='--', lw=.8)\n"
             "ax.set(xlabel='cycle', ylabel='center-averaged Chern number')\n"
             "ax.tick_params(direction='in', top=True, right=True)\n"
             "if products:\n    ax.legend(frameon=False)\n"
             "else:\n    print('No verified production batches yet.')\n"
             "fig.tight_layout()\nplt.show()\n", name='preview'),
        cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n", name='disconnect')]
    return dict(cells=cells, metadata=dict(accelerator='GPU', colab=dict(gpuType='A100', provenance=[]),
                kernelspec=dict(name='python3', display_name='Python 3'), language_info=dict(name='python')),
                nbformat=4, nbformat_minor=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sync-sources', action='store_true')
    args = parser.parse_args()
    if args.sync_sources:
        repo = next(p for p in ROOT.parents if (p/'PROJECT_ADMIN/REPO_POLICY.md').is_file())
        (ROOT/'src').mkdir(exist_ok=True)
        for name in ('classA_U1FGTN_gpu.py', 'occupied_frame_gpu.py'):
            shutil.copyfile(repo/'src/fgtn'/name, ROOT/'src'/name)
    (ROOT/'campaign_config.json').write_text(json.dumps(default_config(), indent=2)+'\n')
    (ROOT/NOTEBOOK).write_text(json.dumps(build(), indent=1)+'\n')
    files = {}
    for name in (*SOURCE_FILES, NOTEBOOK, 'build_notebook.py', 'campaign_config.json', 'README.md'):
        raw = (ROOT/name).read_bytes()
        files[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (ROOT/'deployment_manifest.json').write_text(json.dumps(dict(
        bundle=ROOT.name, schema='simple_colab_bundle_manifest_v1',
        sampling_revision=default_config()['sampling_revision'], files=files), indent=2)+'\n')
    print(ROOT/NOTEBOOK)


if __name__ == '__main__':
    main()
