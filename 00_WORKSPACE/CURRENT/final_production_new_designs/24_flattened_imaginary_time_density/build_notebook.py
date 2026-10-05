"""Reproducible config/notebook generation, without bundling historical code."""
import json
from pathlib import Path
import pprint

from run_campaign import default_config, SOURCE_FILES

ROOT = Path(__file__).resolve().parent


def cell(text, kind='code', name='cell'):
    out = dict(cell_type=kind, id=name, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        out.update(execution_count=None, outputs=[])
    return out


def build():
    cells = [cell(r'''# Imaginary-time density memory: flattened-parent ground state

One deterministic 20×30 hard-wall ground state, not a stochastic circuit campaign.
The signed OW-projector parent retains its wall dispersion; no additional spectral flattening.

For a quadratic density observable, evaluate
$C_O(\tau)=\sum_{a\in\mathrm{occ},b\in\mathrm{empty}}|O_{ba}|^2e^{-\tau(\epsilon_b-\epsilon_a)}$.
Local density is summed over the two orbitals and averaged over origins y.
Column charge sums all y before forming the correlator and is divided by Ny.
Imaginary time is in inverse parent-energy units, not circuit cycles.
The ground state and propagation use the same regulated parent (occupation twist 1e-7).
No error bars or power-law fits are imposed. See `theory.tex` for conventions.
''', 'markdown', 'intro'),
        cell("from google.colab import drive\ndrive.mount('/content/drive')\n", name='mount'),
        cell("from pathlib import Path\n\n# All settings. CPU is sufficient; no A100 required.\n"
             "REPORT_ONLY = False\nCPU_THREADS = 2\n"
             "# Scientific settings are visible but locked to this named revision.\n"
             "CONFIG = "+pprint.pformat(default_config(), sort_dicts=False)+"\n"
             f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{ROOT.name}')\n"
             f"LOCAL_BUNDLE_DIR = Path('/content/{ROOT.name}')\n"
             "SCRATCH_ROOT = Path('/content/imaginary_time_density_scratch')\n"
             "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['revision']\n",
             name='config'),
        cell('## Stage locally and run/resume\n\nThe one completed result is verified before skipping. '
             'An interrupted task restarts; there are no trajectory or state histories. '
             'Progress counts imaginary times. DriveFS readback is not an independent server-side check.',
             'markdown', 'run-intro'),
        cell("import codecs\nimport json\nimport shutil\nimport subprocess\nimport sys\n\n"
             "LOCAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)\n"
             f"for relative in {list(SOURCE_FILES)!r}:\n"
             "    destination = LOCAL_BUNDLE_DIR / relative\n"
             "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
             "    shutil.copy2(BUNDLE_DRIVE_DIR / relative, destination)\n"
             "config_path = LOCAL_BUNDLE_DIR / 'resolved_config.json'\n"
             "config_path.write_text(json.dumps(CONFIG, indent=2))\n"
             "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR / 'run_campaign.py'),\n"
             "           '--config', str(config_path), '--output-root', str(OUTPUT_ROOT),\n"
             "           '--scratch-root', str(SCRATCH_ROOT), '--threads', str(CPU_THREADS)]\n"
             "if REPORT_ONLY:\n    command.append('--report-only')\n"
             "print('[launch]', ' '.join(command), flush=True)\n"
             "decoder = codecs.getincrementaldecoder('utf-8')('replace')\n"
             "with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) as process:\n"
             "    while True:\n"
             "        chunk = process.stdout.read(4096)\n"
             "        if not chunk:\n            break\n"
             "        sys.stdout.write(decoder.decode(chunk)); sys.stdout.flush()\n"
             "    sys.stdout.write(decoder.decode(b'', final=True)); sys.stdout.flush()\n"
             "    code = process.wait()\n"
             "    if code:\n        raise subprocess.CalledProcessError(code, command)\n", name='run'),
        cell('## Read verified data\n\nEach plot has its own editable cell. Earlier results are never pooled.',
             'markdown', 'load-intro'),
        cell("import numpy as np\nimport matplotlib.pyplot as plt\nfrom matplotlib.colors import LogNorm\n"
             "sys.path.insert(0, str(LOCAL_BUNDLE_DIR))\n"
             "from run_campaign import identity, paths, verified\n"
             "from plot_results import configure_style, draw_curves\n"
             "configure_style()\nDATA = None\n"
             "ident = identity(CONFIG)\n"
             "if verified(OUTPUT_ROOT, ident):\n"
             "    result_path, receipt_path = paths(OUTPUT_ROOT, ident)\n"
             "    with np.load(result_path, allow_pickle=False) as saved:\n"
             "        DATA = {key: saved[key] for key in saved.files}\n"
             "    print(json.loads(str(DATA['diagnostics_json'])))\n"
             "else:\n    print('No verified result yet; plots skipped.')\n", name='load')]
    for name, kind, log in [('local-loglog', 'local', True), ('local-semilog', 'local', False),
                             ('column-loglog', 'column', True)]:
        cells += [cell(f'## {name.replace("-", " ").capitalize()}\n\nRaw connected autocorrelation; zeros are omitted on logarithmic axes.',
                       'markdown', name+'-intro'),
                  cell("if DATA is not None:\n"
                       "    fig, ax = plt.subplots(figsize=(3.375, 2.6), layout='constrained')\n"
                       f"    draw_curves(ax, DATA, kind={kind!r}, log_time={log!r})\n"
                       "    plt.show()\n", name=name)]
    cells += [cell('## Spatially resolved memory\n\nNormalize by each x slice’s own equal-time variance; undefined ratios are masked.',
                   'markdown', 'map-intro'),
              cell("if DATA is not None:\n"
                   "    fig, ax = plt.subplots(figsize=(3.375, 2.6), layout='constrained')\n"
                   "    values = np.ma.masked_invalid(DATA['local_normalized'][1:])\n"
                   "    values = np.ma.masked_less_equal(values, 0)\n"
                   "    image = ax.pcolormesh(DATA['x'], DATA['tau'][1:], values, shading='nearest', cmap='Blues', norm=LogNorm(1e-12, 1))\n"
                   "    ax.set(yscale='log', xlabel='$x$', ylabel=r'imaginary time $\\tau$')\n"
                   "    fig.colorbar(image, ax=ax, label=r'$C_x(\\tau)/C_x(0)$', extend='min')\n"
                   "    plt.show()\n", name='map'),
              cell('## Raw diagnostics\n\nNo fitted exponent, no independent samples, no sampling uncertainty.',
                   'markdown', 'diagnostic-intro'),
              cell("if DATA is not None:\n"
                   "    print('Local normalization valid:', DATA['local_normalization_valid'])\n"
                   "    print('Column normalization valid:', DATA['column_normalization_valid'])\n"
                   "    print('Diagnostics:', json.loads(str(DATA['diagnostics_json'])))\n", name='diagnostics'),
              cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n", name='disconnect')]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
                    nbformat=4, nbformat_minor=5)
    (ROOT/'run_flattened_imaginary_time_density.ipynb').write_text(json.dumps(notebook, indent=1)+'\n')
    (ROOT/'campaign_config.json').write_text(json.dumps(default_config(), indent=2)+'\n')


if __name__ == '__main__':
    build()
