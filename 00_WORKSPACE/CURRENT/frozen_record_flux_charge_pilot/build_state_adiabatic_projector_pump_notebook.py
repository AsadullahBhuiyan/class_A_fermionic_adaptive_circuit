#!/usr/bin/env python3
"""Build the editable local notebook for the state-derived projector pump."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "state_adiabatic_projector_pump.ipynb"


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# State-derived adiabatic projector pump\n\n"
            "This notebook launches and monitors the S10 static spectral-flow diagnostic. "
            "It does **not** evolve the monitored circuit again. Each verified burn-in occupied "
            "frame defines its own flattened parent $h_\\xi=\\mathbf{1}-2P_\\xi$, and the fixed-rank "
            "occupied projector is continued by overlap through one flux quantum."
        ),
        nbf.v4.new_markdown_cell(
            "## Theory and estimators\n\n"
            "For a saved orthonormal occupied frame $F_\\xi$, we form "
            "$P_\\xi=F_\\xi F_\\xi^\\dagger$ and $h_\\xi=\\mathbf{1}-2P_\\xi$. "
            "A minimum-image Peierls phase inserts flux along the periodic $y$ direction. "
            "At each neighboring flux point the rank-$r$ subspace with greatest overlap with the "
            "previous occupied subspace is retained. The charge-transfer estimator is "
            "$q_x=(\\Delta N_R-\\Delta N_L)/2$. Independently filling the lowest $r$ eigenvectors "
            "at each flux is saved as the instantaneous closure control."
        ),
        nbf.v4.new_markdown_cell(
            "## Runtime and editable launch configuration\n\n"
            "The calculation runs directly from this notebook. One process handles one "
            "trajectory/direction path and each process uses one BLAS thread."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json, os, subprocess\n\n"
            f"PROJECT_ROOT = Path({str(ROOT)!r})\n"
            "CONFIG = PROJECT_ROOT / 'campaign_config.state_adiabatic_projector_pump_s10_v1.json'\n"
            "OUTPUT_ROOT = PROJECT_ROOT / 'results/N16x20_state_adiabatic_projector_pump_s10_v1'\n"
            "CPU_LIST = '56-63'       # editable logical CPU range\n"
            "WORKERS = 8              # do not exceed the number of selected CPUs\n"
            "BLAS_THREADS = 1\n"
            "print(json.dumps({\n"
            "    'project_root': str(PROJECT_ROOT), 'config': str(CONFIG),\n"
            "    'output_root': str(OUTPUT_ROOT),\n"
            "    'cpu_list': CPU_LIST, 'workers': WORKERS, 'blas_threads': BLAS_THREADS,\n"
            "}, indent=2))"
        ),
        nbf.v4.new_markdown_cell(
            "## Verified input and resume inventory\n\n"
            "This cell verifies the 20 source burn-in result/completion pairs and every existing "
            "state-pump result before reporting progress."
        ),
        nbf.v4.new_code_cell(
            "command = [\n"
            "    'taskset', '-c', CPU_LIST, 'python', '-u',\n"
            "    str(PROJECT_ROOT / 'run_state_adiabatic_projector_pump.py'), 'report',\n"
            "    '--config', str(CONFIG), '--output-root', str(OUTPUT_ROOT),\n"
            "]\n"
            "subprocess.run(command, check=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## Run or resume directly\n\n"
            "This cell blocks until the static calculation finishes and surfaces both `tqdm` bars. "
            "Re-running it verifies all result/completion pairs and skips valid paths."
        ),
        nbf.v4.new_code_cell(
            "environment = os.environ.copy()\n"
            "for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',\n"
            "             'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):\n"
            "    environment[name] = str(BLAS_THREADS)\n"
            "run_command = [\n"
            "    'taskset', '-c', CPU_LIST, 'python', '-u',\n"
            "    str(PROJECT_ROOT / 'run_state_adiabatic_projector_pump.py'), 'run', '--resume',\n"
            "    '--config', str(CONFIG), '--output-root', str(OUTPUT_ROOT),\n"
            "    '--workers', str(WORKERS),\n"
            "]\n"
            "subprocess.run(run_command, env=environment, check=True)\n"
            "subprocess.run([\n"
            "    'taskset', '-c', CPU_LIST.split(',')[0].split('-')[0], 'python', '-u',\n"
            "    str(PROJECT_ROOT / 'analyze_state_adiabatic_projector_pump.py'),\n"
            "    '--config', str(CONFIG), '--output-root', str(OUTPUT_ROOT),\n"
            "], env=environment, check=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## Progress\n\n"
            "Run this cell whenever you want a checksum-verified completion count and the latest "
            "analysis summary."
        ),
        nbf.v4.new_code_cell(
            "subprocess.run(command, check=True)\n"
            "summary = OUTPUT_ROOT / 'analysis/analysis_summary.json'\n"
            "if summary.is_file(): print(summary.read_text())"
        ),
        nbf.v4.new_markdown_cell(
            "## Continued projector response\n\n"
            "Once all 40 paths verify, the analysis writes the ensemble mean with one-standard-"
            "deviation trajectory bands. Each figure has a separate display cell for easy editing."
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "figure = OUTPUT_ROOT / 'analysis/figures/state_adiabatic_projector_qx.png'\n"
            "if figure.is_file():\n"
            "    display(Image(filename=str(figure)))\n"
            "else:\n"
            "    print('Pending: analysis runs automatically after 40/40 verified paths.')"
        ),
        nbf.v4.new_markdown_cell("## Left/right subsystem charge"),
        nbf.v4.new_code_cell(
            "figure = OUTPUT_ROOT / 'analysis/figures/state_adiabatic_projector_regional_charge.png'\n"
            "if figure.is_file(): display(Image(filename=str(figure)))\n"
            "else: print('Pending.')"
        ),
        nbf.v4.new_markdown_cell("## Endpoint distributions"),
        nbf.v4.new_code_cell(
            "figure = OUTPUT_ROOT / 'analysis/figures/state_adiabatic_projector_endpoints.png'\n"
            "if figure.is_file(): display(Image(filename=str(figure)))\n"
            "else: print('Pending.')"
        ),
        nbf.v4.new_markdown_cell("## Raw summary and numerical diagnostics"),
        nbf.v4.new_code_cell(
            "summary = OUTPUT_ROOT / 'analysis/analysis_summary.json'\n"
            "if summary.is_file():\n"
            "    payload = json.loads(summary.read_text())\n"
            "    print(json.dumps(payload, indent=2, sort_keys=True))\n"
            "else:\n"
            "    print('Pending.')"
        ),
    ]
    nbf.write(nb, OUTPUT)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
