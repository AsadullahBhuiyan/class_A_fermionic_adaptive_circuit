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
    cells=[cell(r'''# Square-system finite-time Lyapunov gap: T=10, ten samples

Hard walls, slab-only measurements, alpha1=1, alpha2=30, nshell=1;
L=Nx=Ny=20,24,28,32,36,40,44. Each case starts maximally mixed with the
canonical Born-conditioned exterior preparation, then runs ten physical
raster-y cycles with perfect correction. Production requires A100 40-GB-class.

At cycle ten only, save each active-slab occupation spectrum. Define
$g_{\mathrm{mod},\xi}=\min_j|\log[(1-\nu_{\xi,j})/\nu_{\xi,j}]|$ and
$\Delta_\xi=g_{\mathrm{mod},\xi}/(2T)=g_{\mathrm{mod},\xi}/20$.
Average gaps over ten trajectories; uncertainty is ordinary SEM, not bootstrap.
This is a finite-time diagnostic, not proof of a converged scaling exponent.
Both wall separation and circumference increase with L.

One ten-sample batch per size; two five-sample result shards. Rolling covariance
and RNG checkpoints every five cycles. No intermediate spectral observations,
contours, eigenvectors, or covariance histories. Never run two writers concurrently.
''','markdown','intro'),
        cell("from google.colab import drive\ndrive.mount('/content/drive')\n",name='mount'),
        cell("from pathlib import Path\n\nREPORT_ONLY = False\n"
             "MAX_NEW_EXECUTION_BATCHES = None  # use 1 for the first L=44 batch\n"
             "RUN_ANALYSIS = False  # requires all 14 result shards\n"
             "# Complete configuration: scientific changes require a new revision.\n"
             "CONFIG = "+pprint.pformat(default_config(),sort_dicts=False)+"\n"
             f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{HERE.name}')\n"
             f"LOCAL_BUNDLE_DIR = Path('/content/{HERE.name}')\n"
             "SCRATCH_ROOT = Path('/content/square_gap_scratch')\n"
             "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n",
             name='config'),
        cell('## Stage locally and run/resume\n\nThe largest geometry runs first. '
             'The runner prints checkpoint status, measured cycle/checkpoint times and GPU memory. '
             'A failed write leaves the task incomplete; cycle-10 checkpoints survive until both result shards verify.',
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
        cell('## Optional endpoint analysis\n\nMean of sample-resolved gaps with ordinary SEM; '
             'raw modular gap and normalized Lyapunov half gap are plotted separately. No fitted exponent.',
             'markdown','analysis-intro'),
        cell("if RUN_ANALYSIS and not REPORT_ONLY:\n"
             "    run_streamed([sys.executable, '-u', str(LOCAL_BUNDLE_DIR / 'analyze_campaign.py'),\n"
             "                  '--output-root', str(OUTPUT_ROOT)])\n"
             "else:\n    print('Analysis skipped.')\n",name='analysis'),
        cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n",name='disconnect')]
    nb=dict(cells=cells,nbformat=4,nbformat_minor=5,
            metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
    (HERE/'run_square_hard_wall_gap.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    (HERE/'campaign_config.json').write_text(json.dumps(default_config(),indent=2)+'\n')


if __name__=='__main__':build()
