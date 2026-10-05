import json
from pathlib import Path
import pprint
from run_campaign import default_config,SOURCE_FILES

HERE=Path(__file__).resolve().parent


def cell(source,kind='code',name='cell'):
    row=dict(cell_type=kind,id=name,metadata={},source=source.splitlines(keepends=True))
    if kind=='code':row.update(execution_count=None,outputs=[])
    return row


def build():
    cells=[cell(r'''# Full-measurement purification: fixed 40 cycles, 100 samples

Nx=20; Ny=30,36,42,48,54,60; hard walls x=5,15; alpha1=1, alpha2=30,
nshell=1, raster-y, perfect correction, complex128. The entire physical layer
starts maximally mixed. meas_slab_only=False: measure both the slab and exterior,
without Born-conditioned exterior preparation. Requires A100 40-GB-class.

At every cycle t=6,...,40, save each sample's full occupation spectrum, raw
modular gap, finite-time gap divided by 2t, and all complex128 eigenvectors with
$10^{-9}<\nu<1-10^{-9}$. This includes zero-rate modes, not just the smallest-gap mode.
Vectors use the full spatial basis, with counts and spectrum indices for padding.
Gaps are computed sample by sample, not from the ensemble-averaged spectrum.

600 trajectories in 18 execution batches. 4,200 five-sample/cycle result pairs.
Covariance/RNG checkpoints are published every five cycles. Spectral shards
are saved at every cycle 6–40 and skipped on restart; at most five dynamics
cycles need replay. Spectral matrix batches 1,2,5 are timed on the actual GPU;
the fastest safe candidate is selected independently for each size.
No covariance histories, entropy contours, bootstrap, or old-data migration.
Never run two writers on this output folder.

Saving modes at 35 cycles requires many more eigendecompositions than endpoint-only.
The uncompressed worst case is about 1.15 TB of eigenvectors; actual output depends
on the number of mixed modes. Budget roughly 60–90 A100-hours by extrapolation from campaign 21;
this new workload has not been benchmarked on A100. Set MAX_NEW_EXECUTION_BATCHES=1 to measure the first Ny=60 batch.
''','markdown','intro'),
        cell("from google.colab import drive\ndrive.mount('/content/drive')\n",name='mount'),
        cell("from pathlib import Path\n\nREPORT_ONLY = False\n"
             "MAX_NEW_EXECUTION_BATCHES = None  # use 1 for the first Ny=60 batch\n"
             "RUN_ANALYSIS = False  # requires all 4,200 cycle result shards\n"
             "# Complete configuration: scientific changes require a new revision.\n"
             "CONFIG = "+pprint.pformat(default_config(),sort_dicts=False)+"\n"
             f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{HERE.name}')\n"
             f"LOCAL_BUNDLE_DIR = Path('/content/{HERE.name}')\n"
             "SCRATCH_ROOT = Path('/content/full_measurement_gap_t40_scratch')\n"
             "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n",
             name='config'),
        cell('## Stage locally and run/resume\n\nThe largest geometry runs first. '
             'The runner prints checkpoint status, measured cycle/checkpoint times and GPU memory. '
             'A failed write leaves the task incomplete; five-cycle state checkpoints bound replay; completed spectral shards are retained.',
             'markdown','run-intro'),
        cell("import codecs\nimport json\nimport shutil\nimport subprocess\nimport sys\n\n"
             "LOCAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)\n"
             f"for relative in {list(SOURCE_FILES)+['analyze_campaign.py']!r}:\n"
             "    destination = LOCAL_BUNDLE_DIR / relative\n"
             "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
             "    shutil.copy2(BUNDLE_DRIVE_DIR / relative, destination)\n"
             "config_path = LOCAL_BUNDLE_DIR / 'resolved_config.json'\n"
             "config_path.write_text(json.dumps(CONFIG, indent=2))\n\n"
             "def run_streamed(command):\n"
             "    print('[launch]', ' '.join(command), flush=True)\n"
             "    decoder = codecs.getincrementaldecoder('utf-8')('replace')\n"
             "    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) as process:\n"
             "        while True:\n"
             "            chunk = process.stdout.read(4096)\n"
             "            if not chunk:\n                break\n"
             "            sys.stdout.write(decoder.decode(chunk)); sys.stdout.flush()\n"
             "        sys.stdout.write(decoder.decode(b'', final=True)); sys.stdout.flush()\n"
             "        returncode = process.wait()\n"
             "        if returncode:\n            raise subprocess.CalledProcessError(returncode, command)\n\n"
             "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR / 'run_campaign.py'),\n"
             "           '--config', str(config_path), '--output-root', str(OUTPUT_ROOT),\n"
             "           '--scratch-root', str(SCRATCH_ROOT)]\n"
             "if REPORT_ONLY:\n    command.append('--report-only')\n"
             "if MAX_NEW_EXECUTION_BATCHES is not None:\n"
             "    command.extend(['--max-new-execution-batches', str(MAX_NEW_EXECUTION_BATCHES)])\n"
             "run_streamed(command)\n",name='run'),
        cell('## Optional cycle-resolved gap analysis\n\nMean of sample-resolved gaps with ordinary SEM; '
             'raw modular gap and normalized Lyapunov half gap are plotted separately. No fitted exponent.',
             'markdown','analysis-intro'),
        cell("if RUN_ANALYSIS and not REPORT_ONLY:\n"
             "    run_streamed([sys.executable, '-u', str(LOCAL_BUNDLE_DIR / 'analyze_campaign.py'),\n"
             "                  '--output-root', str(OUTPUT_ROOT)])\n"
             "else:\n    print('Analysis skipped.')\n",name='analysis'),
        cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n",name='disconnect')]
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,
            metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    (HERE/'run_full_measurement_purification_gap_t40.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    (HERE/'campaign_config.json').write_text(json.dumps(default_config(),indent=2)+'\n')


if __name__=='__main__':build()
