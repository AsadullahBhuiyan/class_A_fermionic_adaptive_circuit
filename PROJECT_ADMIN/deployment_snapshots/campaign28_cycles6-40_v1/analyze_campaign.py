"""Mean of trajectory gaps at each saved cycle, ordinary SEM; no automatic exponent."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from run_campaign import default_config, tasks, slots, identity, result_paths, result_verified, sha


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        # Never silently discard a completely purified trajectory.
        return dict(mean=None, sem=None, finite_count=int(np.isfinite(values).sum()))
    return dict(mean=float(values.mean()), sem=float(values.std(ddof=1)/np.sqrt(len(values))),
                finite_count=len(values))


def analyze(output, destination):
    config = default_config()
    groups = {}
    inputs = []
    sample_rows = []
    for task in tqdm(tasks(config),desc='verify and analyze batches',unit='batch'):
        ident = identity(task,config)
        for cycle,start in slots(task,config):
            if not result_verified(output,task,start,ident,cycle):
                raise RuntimeError('Analysis requires all 4,200 verified cycle shards')
            path,receipt = result_paths(output,task,start,cycle)
            inputs.append(dict(path=str(path),sha256=sha(path),receipt_sha256=sha(receipt)))
            with np.load(path,allow_pickle=False) as data:
                for i,sample in enumerate(data['sample_indices']):
                    row = dict(Nx=task.Nx,Ny=task.Ny,cycle=cycle,sample=int(sample),
                               modular_gap=float(data['modular_gap'][i]),
                               lyapunov_gap=float(data['lyapunov_gap'][i]),
                               finite_modes=int(data['finite_mode_count'][i]))
                    sample_rows.append(row)
                    groups.setdefault((task.Ny,cycle),[]).append(row)
    summary = []
    for (ny,cycle),rows in sorted(groups.items()):
        if sorted(r['sample'] for r in rows) != list(range(100)):
            raise ValueError('Expected every sample 0..99 exactly once at each cycle')
        row = dict(Nx=20,Ny=ny,cycle=cycle,samples=len(rows))
        for observable in ('modular_gap','lyapunov_gap'):
            row.update({observable+'_'+key:value for key,value in summarize([r[observable] for r in rows]).items()})
        summary.append(row)
    destination = Path(destination); destination.mkdir(parents=True,exist_ok=True)
    for name,rows in (('sample_gaps.csv',sample_rows),('gap_summary.csv',summary)):
        with (destination/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif','DejaVu Sans'],
                         'font.size':9,'xtick.direction':'in','ytick.direction':'in',
                         'xtick.top':True,'ytick.right':True})
    for observable,label in (('modular_gap',r'$g_{\mathrm{mod}}$'),('lyapunov_gap',r'$\Delta=g_{\mathrm{mod}}/(2t)$')):
        fig,ax=plt.subplots(figsize=(3.375,2.65))
        for ny in config['Ny_values']:
            rows=[r for r in summary if r['Ny']==ny and r[observable+'_mean'] is not None]
            ax.errorbar([r['cycle'] for r in rows],[r[observable+'_mean'] for r in rows],
                        yerr=[r[observable+'_sem'] for r in rows],label=f'{ny}',linewidth=1)
        ax.set(xlabel='physical cycle',ylabel=label)
        ax.legend(title=r'$N_y$',frameon=False,ncol=2,fontsize=7)
        fig.tight_layout()
        fig.savefig(destination/(observable+'_vs_cycle.pdf'))
        fig.savefig(destination/(observable+'_vs_cycle.png'),dpi=300)
        plt.close(fig)
    outputs={p.name:sha(p) for p in destination.iterdir() if p.suffix in ('.csv','.pdf','.png')}
    (destination/'analysis_manifest.json').write_text(json.dumps(dict(
        revision=config['sampling_revision'],uncertainty='sample-wise SEM; no bootstrap',
        gap_convention='min(abs(log((1-nu)/nu)))/(2*cycle)',inputs=inputs,outputs=outputs,
        fitted_exponent=None,summary=summary),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--analysis-root',type=Path)
    args=parser.parse_args()
    analyze(args.output_root,args.analysis_root or args.output_root/'analysis_outputs')

