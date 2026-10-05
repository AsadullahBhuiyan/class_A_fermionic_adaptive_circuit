"""Generate the small Colab frontend from the runner's locked scientific contract."""
import json
from pathlib import Path
from pprint import pformat

from run_campaign import BUNDLE, EXPECTED_REVISION, expected_config

ROOT = Path(__file__).resolve().parent
NOTEBOOK = 'run_hard_wall_alpha3_correlator.ipynb'


def cell(kind, text):
    result = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code': result.update(execution_count=None, outputs=[])
    return result


def build():
    config_text = pformat(expected_config(), sort_dicts=False, width=95)
    cells = [cell('markdown', r'''# Hard-wall correlators: $\alpha_1=3$, $20\times60$, S100

Matched comparison to the completed alpha-1 ensemble: hard/support-truncated walls at
$x=5,15$, $n_{\rm shell}=1$, $\alpha_2=30$, pure initialization,
raster-$y$, perfect correction, complex128, and **120 cycles ($2N_y$)**.
There is no soft-wall mode. This is a separate ensemble with a new seed and output folder.

At the endpoint, save each sample's squared correlator
$C_G(x,r)=\sum_{y,\mu,\nu}|C_{x,y,\mu;x,y+r,\nu}|^2/(2N_y)$,
its $x$ average, and total charge. No endpoint states, covariance matrices or spectra.
Frames/RNG are retained only in the rolling checkpoint until results finish.

Select an **A100 40-GB-class runtime**. The resident batches are **40+40+20**, limited to
30 GiB in the PyTorch allocator. This is not a measured 30-GiB occupancy target or a cap
on allocations outside PyTorch. Results are 20 immutable five-sample NPZ/JSON pairs.
Restarting skips valid pairs and resumes the active batch from a valid five-cycle checkpoint.
'''), cell('markdown', '## 1. Mount Drive and configure\n\nEdit paths and session controls here. The displayed scientific contract is locked for this ensemble.\n')]
    cells.append(cell('code', f'''from pathlib import Path
import json
import torch
from google.colab import drive

drive.mount('/content/drive')
BUNDLE = {BUNDLE!r}
REVISION = {EXPECTED_REVISION!r}
BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs') / BUNDLE
LOCAL_BUNDLE_DIR = Path('/content') / BUNDLE
OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / REVISION
SCRATCH_ROOT = Path('/content/alpha3_hard_wall_correlator_scratch')
REPORT_ONLY = False
MAX_NEW_EXECUTION_BATCHES = None  # set 1 to stop after one resident batch

CONFIG = {config_text}

if not torch.cuda.is_available():
    raise RuntimeError('Select an A100 GPU runtime')
gpu = torch.cuda.get_device_properties(0)
if 'A100' not in gpu.name.upper() or gpu.total_memory < 38 * 1024**3:
    raise RuntimeError(f'Expected A100 40-GB-class memory, found {{gpu.name}}')
print(json.dumps({{'contract': CONFIG, 'gpu': gpu.name, 'dtype': CONFIG['dtype'],
                  'bundle_source': str(BUNDLE_DRIVE_DIR), 'local_stage': str(LOCAL_BUNDLE_DIR),
                  'output_root': str(OUTPUT_ROOT), 'scratch_root': str(SCRATCH_ROOT),
                  'resident_batches': [40,40,20], 'durable_shards': 20, 'trajectories': 100,
                  'cycles': 120, 'sample_cycles': 12000}}, indent=2), flush=True)
'''))
    cells.append(cell('markdown', '''## 2. Stage locally and run/resume

Only the four executable files are copied from Drive, never previous data.
Run inside the notebook kernel so `tqdm.auto` displays Colab-native progress bars:
20 result shards outside, 120 physical cycles inside. Do not run a second copy against
the same output folder. Space checks inspect the mounted filesystem, not cloud quota;
keep at least 5 GiB available on Drive for checkpoint replacement.
'''))
    cells.append(cell('code', '''import importlib.util
import shutil
import sys

FILES = ['run_campaign.py', 'alpha3_correlator.py',
         'src/classA_U1FGTN_gpu.py', 'src/occupied_frame_gpu.py']
for relative in FILES:
    source = BUNDLE_DRIVE_DIR / relative
    if not source.is_file():
        raise FileNotFoundError(f'Missing bundle file: {source}')
    target = LOCAL_BUNDLE_DIR / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
config_path = Path('/content/alpha3_hard_wall_correlator_config.json')
config_path.write_text(json.dumps(CONFIG, indent=2) + '\\n')
runner_args = ['--config', str(config_path), '--output-root', str(OUTPUT_ROOT),
               '--scratch-root', str(SCRATCH_ROOT)]
if REPORT_ONLY: runner_args.append('--report-only')
if MAX_NEW_EXECUTION_BATCHES is not None:
    runner_args += ['--max-new-execution-batches', str(int(MAX_NEW_EXECUTION_BATCHES))]
module_name = '_alpha3_hard_wall_correlator_runner'
for name in (module_name, 'alpha3_correlator', 'classA_U1FGTN_gpu', 'occupied_frame_gpu'):
    sys.modules.pop(name, None)
runner_path = LOCAL_BUNDLE_DIR / 'run_campaign.py'
print('[launch in notebook kernel]', runner_path, flush=True)
spec = importlib.util.spec_from_file_location(module_name, runner_path)
runner = importlib.util.module_from_spec(spec)
sys.modules[module_name] = runner
spec.loader.exec_module(runner)
returncode = runner.main(runner_args)
if returncode: raise RuntimeError(f'Runner exited: {returncode}')
print('[notebook] requested work finished; inspect the completion summary above', flush=True)
'''))
    cells.append(cell('markdown', '## 3. Release the runtime\n\nAfter the requested work finishes, release the GPU.\n'))
    cells.append(cell('code', "from google.colab import runtime\nruntime.unassign()\nprint('done')\n"))
    notebook = dict(cells=cells, metadata={'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
                                         'language_info': {'name':'python'}, 'accelerator':'GPU'},
                    nbformat=4, nbformat_minor=4)
    path = ROOT/NOTEBOOK
    path.write_text(json.dumps(notebook, indent=1)+'\n')
    print(path)


if __name__ == '__main__': build()
