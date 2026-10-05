"""Generate the small drop-in Colab notebook and its visible configuration."""
import json
from pathlib import Path
import pprint
from run_campaign import default_config, SOURCE_FILES

HERE=Path(__file__).resolve().parent


def cell(text,kind='code',name='cell'):
    result=dict(cell_type=kind,id=name,metadata={},source=text.splitlines(keepends=True))
    if kind=='code':result.update(execution_count=None,outputs=[])
    return result


def build():
    cells=[cell('''# Square hard-wall purification: one trajectory each at 30×30 and 40×40

60 physical cycles; alpha1=1, alpha2=30, nshell=1, hard support-truncated walls,
global maximally mixed initialization, full-system measurements, raster-y,
perfect correction, complex128. No postselection or state clipping.
Use an A100 40-GB-class runtime. Each size is independently resumable.

At every cycle including zero, save total mixed-state von Neumann entropy,
cell-resolved entropy contour, occupations and gap diagnostics. For C=(G+I)/2,
S=sum h(nu) and s_i=sum_j |U_ij|² h(nu_j), summed over the two orbitals per cell.
This is purification entropy, not the entanglement of a spatial subsystem.
Natural logs; the 1e-12 entropy endpoint regulator matches the preceding campaign.
The state itself is never clipped. A fully capped spectrum has an undefined gap.

Five-cycle rolling covariance/RNG/observer checkpoints limit restart loss.
Only one writer may use this output folder. No covariance or eigenvector history.
No sample error bars are meaningful with one trajectory per size.
''','markdown','intro'),
        cell("from google.colab import drive\ndrive.mount('/content/drive')\n",name='mount'),
        cell("from pathlib import Path\n\nREPORT_ONLY = False\nMAX_NEW_TASKS = None  # 1 runs only the first unfinished size\n"
            "# Scientific edits require a new sampling_revision/output folder.\n"
            "CONFIG = "+pprint.pformat(default_config(),sort_dicts=False)+"\n"
            f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{HERE.name}')\n"
            f"LOCAL_BUNDLE_DIR = Path('/content/{HERE.name}')\n"
            "SCRATCH_ROOT = Path('/content/square_contour_scratch')\n"
            "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n",
            name='configuration'),
        cell('## Stage locally and run/resume\n\nVisible outer task and inner cycle progress includes eigensolving. '
            'Completed result pairs are verified and skipped. Active tasks resume from their own five-cycle checkpoints.',
            'markdown','run-heading'),
        cell("import codecs\nimport json\nimport shutil\nimport subprocess\nimport sys\n\n"
            "LOCAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)\n"
            f"for relative in {list(SOURCE_FILES)!r}:\n"
            "    destination = LOCAL_BUNDLE_DIR / relative\n"
            "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(BUNDLE_DRIVE_DIR / relative, destination)\n"
            "config_path = LOCAL_BUNDLE_DIR / 'resolved_config.json'\n"
            "config_path.write_text(json.dumps(CONFIG, indent=2))\n"
            "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR/'run_campaign.py'),\n"
            "           '--config', str(config_path), '--output-root', str(OUTPUT_ROOT), '--scratch-root', str(SCRATCH_ROOT)]\n"
            "if REPORT_ONLY: command.append('--report-only')\n"
            "if MAX_NEW_TASKS is not None: command.extend(['--max-new-tasks', str(MAX_NEW_TASKS)])\n"
            "print('[launch]', ' '.join(command), flush=True)\n"
            "decoder = codecs.getincrementaldecoder('utf-8')('replace')\n"
            "with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) as process:\n"
            "    try:\n"
            "        while True:\n"
            "            chunk = process.stdout.read(4096)\n"
            "            if not chunk: break\n"
            "            sys.stdout.write(decoder.decode(chunk)); sys.stdout.flush()\n"
            "        sys.stdout.write(decoder.decode(b'', final=True)); sys.stdout.flush()\n"
            "        code = process.wait()\n"
            "        if code: raise subprocess.CalledProcessError(code, command)\n"
            "    except BaseException:\n"
            "        if process.poll() is None:\n"
            "            process.terminate()\n"
            "            try: process.wait(timeout=10)\n"
            "            except subprocess.TimeoutExpired: process.kill(); process.wait()\n"
            "        raise\n",name='run'),
        cell('## Quick view of completed trajectories\n\nNo averaging or uncertainty bands; these are individual trajectories. '
            'Cycle zero is omitted only from logarithmic time plots.', 'markdown','plot-heading'),
        cell("import numpy as np\nimport matplotlib.pyplot as plt\n"
            "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
            "import run_campaign as runner\n"
            "for L in CONFIG['sizes']:\n"
            "    data = runner.load_pair(OUTPUT_ROOT, L, 'result', runner.identity(CONFIG, L))\n"
            "    if data is None:\n"
            "        print(f'L={L}: not yet complete'); continue\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(3.375, 4.8), layout='constrained')\n"
            "    axes[0].loglog(data['cycles'][1:], data['total_entropy'][1:]/L, color='#1f77b4')\n"
            "    axes[0].set(xlabel='cycle t', ylabel='S(t)/Ny')\n"
            "    image = axes[1].imshow(data['entropy_contour'][-1].T, origin='lower', cmap='magma')\n"
            "    axes[1].set(xlabel='x', ylabel='y')\n"
            "    fig.colorbar(image, ax=axes[1], label='s(x,y), final cycle')\n"
            "    for ax in axes: ax.tick_params(direction='in', top=True, right=True)\n"
            "    print(f'L={L}, walls={data[\"walls\"].tolist()}, final entropy={data[\"total_entropy\"][-1]:.6g}')\n"
            "    plt.show()\n",name='plots'),
        cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n",name='disconnect')]
    notebook=dict(cells=cells,nbformat=4,nbformat_minor=5,
        metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                      accelerator='GPU'))
    (HERE/'run_square_purification_contour.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    (HERE/'campaign_config.json').write_text(json.dumps(default_config(),indent=2)+'\n')


if __name__=='__main__':build()
