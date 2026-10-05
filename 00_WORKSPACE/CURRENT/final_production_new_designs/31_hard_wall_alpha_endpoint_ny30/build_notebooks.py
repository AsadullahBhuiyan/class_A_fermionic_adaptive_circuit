"""Generate clean, fixed-lane notebooks and their standalone configuration."""
import json
from pathlib import Path
import pprint
from run_campaign import default_config, extension_config, SOURCE_FILES, tasks, sha

HERE=Path(__file__).resolve().parent


def cell(source,kind='code',name='cell'):
    row=dict(cell_type=kind,id=name,metadata={},source=source.splitlines(keepends=True))
    if kind=='code':row.update(execution_count=None,outputs=[])
    return row


def build():
    for lane in 'ABC':
        alpha=[t.alpha_1 for t in tasks(extension_config(),lane)]
        cells=[cell(f'''# Hard-wall endpoint alpha sweep — lane {lane}

Nx=20, Ny=30, T=60, 90 additional independent Born samples per alpha.
Global sample IDs 10--99 extend the completed IDs 0--9 to S=100.
The completed S10 files are not changed, copied, or rerun. New outputs use an independent add90 folder.
This lane runs {alpha}, in that order. Run A/B/C simultaneously on **three separate A100s**;
never launch the same lane twice. Hard walls x=5,15; alpha2=30, nshell=1,
maximally mixed with canonical exterior preparation, slab-only raster-y,
perfect correction, complex128, no postselection. Allocator ceiling: 38 GiB.

Save complete endpoint spectra and one gap eigenmode per sample, no intermediate
entropy or spectra. Rates are lambda=-atanh(a)/60 and gap=min(abs(lambda)).
Centered caps within 1e-9 give infinite rates. Infinite gaps have a flagged display
placeholder 100; this is never a finite measurement or included in averages.
Mode vectors are zero padded with valid=False when all modes are capped.

One resident 90-sample batch per alpha; 18 durable five-sample shards.
Each lane runs seven batches: 630 NEW trajectories and 126 shards (1,890 new samples overall).
Rolling covariance/RNG checkpoints every ten cycles, including cycle 60.
Relaunch this notebook to resume. Interruptions repeat only work since the last verified checkpoint.
The saved S10 benchmark was about 13 minutes/alpha. Linear scaling gives about
14 hours/lane plus I/O; batching may reduce that, but no measured 90-sample speedup is promised.
The first batch prints revised ETA and peak memory. The 38-GiB allocator ceiling stays enforced.
''','markdown','intro'),
            cell("from google.colab import drive\ndrive.mount('/content/drive')\n",name='mount'),
            cell("from pathlib import Path\nREPORT_ONLY = False\nMAX_NEW_EXECUTION_BATCHES = None  # optionally 1 for a first-batch check\n"
                 f"LANE = {lane!r}  # fixed lane; do not change\n"
                 "# Full scientific configuration (changes require a new revision).\n"
                 "CONFIG = "+pprint.pformat(extension_config(),sort_dicts=False)+"\n"
                 f"BUNDLE_DRIVE_DIR = Path('/content/drive/MyDrive/final_production_new_designs/{HERE.name}')\n"
                 f"LOCAL_BUNDLE_DIR = Path('/content/{HERE.name}_lane_{lane}')\n"
                 f"SCRATCH_ROOT = Path('/content/alpha_endpoint_ny30_lane_{lane}_scratch')\n"
                 "OUTPUT_ROOT = Path('/content/drive/MyDrive/classA_final_production_outputs') / CONFIG['sampling_revision']\n",name='config'),
            cell('## Stage locally and run/resume\nLive child stdout/stderr, cycle progress, checkpoint status and ETA. '
                 'Each alpha has disjoint result/checkpoint paths.','markdown','run-intro'),
            cell("import codecs\nimport json\nimport shutil\nimport subprocess\nimport sys\n\n"
                 f"assert LANE == {lane!r}, 'Use the appropriate fixed-lane notebook'\n"
                 "LOCAL_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)\n"
                 f"for relative in {list(SOURCE_FILES)+['analyze_campaign.py']!r}:\n"
                 "    destination = LOCAL_BUNDLE_DIR / relative\n"
                 "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
                 "    shutil.copy2(BUNDLE_DRIVE_DIR / relative, destination)\n"
                 "config_path = LOCAL_BUNDLE_DIR / 'resolved_config.json'\n"
                 "config_path.write_text(json.dumps(CONFIG, indent=2))\n"
                 "command = [sys.executable, '-u', str(LOCAL_BUNDLE_DIR / 'run_campaign.py'),\n"
                 "           '--lane', LANE, '--config', str(config_path), '--output-root', str(OUTPUT_ROOT),\n"
                 "           '--scratch-root', str(SCRATCH_ROOT)]\n"
                 "if REPORT_ONLY:\n    command.append('--report-only')\n"
                 "if MAX_NEW_EXECUTION_BATCHES is not None:\n"
                 "    command.extend(['--max-new-execution-batches', str(MAX_NEW_EXECUTION_BATCHES)])\n"
                 "print('[launch]', ' '.join(command), flush=True)\n"
                 "decoder = codecs.getincrementaldecoder('utf-8')('replace')\n"
                 "with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0) as process:\n"
                 "    try:\n"
                 "        while True:\n"
                 "            chunk = process.stdout.read(4096)\n"
                 "            if not chunk:\n                break\n"
                 "            sys.stdout.write(decoder.decode(chunk)); sys.stdout.flush()\n"
                 "        sys.stdout.write(decoder.decode(b'', final=True)); sys.stdout.flush()\n"
                 "        returncode = process.wait()\n"
                 "    except BaseException:\n"
                 "        process.terminate()\n"
                 "        try:\n            process.wait(timeout=10)\n"
                 "        except subprocess.TimeoutExpired:\n            process.kill(); process.wait()\n"
                 "        raise\n"
                 "if returncode:\n    raise subprocess.CalledProcessError(returncode, command)\n"
                 "print('[done] lane completed or requested batch limit reached', flush=True)\n",name='run'),
            cell('## Combined analysis\nAfter all three lanes finish, run analyze_campaign.py locally '
                 'against the downloaded output collection with --config campaign_config.json. '
                 'It requires all 378 NEW verified pairs and uses '
                 'ordinary sample SEM, with infinite-gap placeholders excluded. No analysis writer runs concurrently here.',
                 'markdown','analysis'),
            cell("from google.colab import runtime\nruntime.unassign()\nprint('done')\n",name='disconnect')]
        nb=dict(cells=cells,nbformat=4,nbformat_minor=5,
                metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')))
        (HERE/f'run_alpha_endpoint_lane_{lane}.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    (HERE/'campaign_config.json').write_text(json.dumps(extension_config(),indent=2)+'\n')
    (HERE/'original_s10_config.json').write_text(json.dumps(default_config(),indent=2)+'\n')
    files=list(SOURCE_FILES)+['analyze_campaign.py','analyze_endpoint_diagnostics.py','build_notebooks.py','README.md','campaign_config.json','original_s10_config.json']
    files += [f'run_alpha_endpoint_lane_{lane}.ipynb' for lane in 'ABC']
    (HERE/'deployment_manifest.json').write_text(json.dumps(dict(files={f:dict(bytes=(HERE/f).stat().st_size,sha256=sha(HERE/f)) for f in files}),indent=2)+'\n')


if __name__=='__main__':build()
