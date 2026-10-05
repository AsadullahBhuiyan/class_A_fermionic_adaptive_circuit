"""Trajectory-wise endpoint gaps, ordinary SEM; no size-exponent fit."""
import argparse
import csv
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_campaign import default_config, tasks, identity, result_paths, result_verified, sha


def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def analyze(output, destination):
    config=default_config()
    sample_rows, summaries, inputs = [], [], []
    for task in reversed(tasks(config)):
        ident=identity(task,config)
        indices, raw, rates = [], [], []
        for start in (0,5):
            if not result_verified(output,task,start,ident):
                raise RuntimeError(f'Analysis requires all 14 verified shards; missing/invalid {task.name}, start={start}')
            result,receipt=result_paths(output,task,start)
            inputs.extend(dict(path=str(p.resolve()),sha256=sha(p)) for p in (result,receipt))
            with np.load(result,allow_pickle=False) as z:
                indices.extend(z['sample_indices'].tolist())
                raw.extend(z['modular_gap'].tolist());rates.extend(z['lyapunov_gap'].tolist())
        np.testing.assert_array_equal(indices,np.arange(10))
        raw,rates=np.array(raw),np.array(rates)
        np.testing.assert_array_equal(rates,raw/20)
        for sample,gap,rate in zip(indices,raw,rates):
            sample_rows.append(dict(L=task.L,Nx=task.L,Ny=task.L,T=10,sample_index=sample,
                                    modular_gap=gap,lyapunov_gap=rate,finite=bool(np.isfinite(rate))))
        finite=int(np.isfinite(rates).sum())
        if finite != 10:
            warnings.warn(f'L={task.L}: only {finite}/10 finite gaps; no mean or SEM is reported for this size')
        summaries.append(dict(L=task.L,T=10,samples=10,finite_samples=finite,
                              mean_modular_gap=float(raw.mean()) if finite==10 else None,
                              sem_modular_gap=float(raw.std(ddof=1)/np.sqrt(10)) if finite==10 else None,
                              mean_lyapunov_gap=float(rates.mean()) if finite==10 else None,
                              sem_lyapunov_gap=float(rates.std(ddof=1)/np.sqrt(10)) if finite==10 else None))
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    write_csv(destination/'sample_gaps.csv',sample_rows)
    write_csv(destination/'gap_summary.csv',summaries)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['CMU Sans Serif','DejaVu Sans'],
        'font.size':8,'mathtext.fontset':'cm','xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,'pdf.fonttype':42})
    for kind,label in [('lyapunov',r'$\overline{\Delta}(T=10)$'),('modular',r'$\overline{g}_{\mathrm{mod}}(T=10)$')]:
        fig,ax=plt.subplots(figsize=(3.375,2.5),layout='constrained')
        values=[r[f'mean_{kind}_gap'] if r['finite_samples']==10 else np.nan for r in summaries]
        errors=[r[f'sem_{kind}_gap'] if r['finite_samples']==10 else np.nan for r in summaries]
        ax.errorbar(config['sizes'],values,yerr=errors,color='#1f77b4',marker='o',ls='--',
                    mfc='white',ms=4,lw=1,capsize=2,label=r'$T=10$: mean $\pm$ SEM')
        ax.set(xlabel=r'square size $L=N_x=N_y$',ylabel=label,xticks=config['sizes'])
        ax.legend(frameon=False)
        for ext in ('pdf','png'):
            fig.savefig(destination/f'{kind}_gap_vs_L.{ext}',dpi=300)
        plt.close(fig)
    files=[p for p in destination.iterdir() if p.suffix in ('.pdf','.png','.csv')]
    manifest=dict(sampling_revision=config['sampling_revision'],independent_samples_per_size=10,
                  estimator='mean of per-trajectory min absolute modular energy; divide by 2T=20 for Lyapunov half gap',
                  uncertainty='sample standard deviation / sqrt(10); no bootstrap',
                  interpretation='fixed depth, not a converged rate or a fixed-width circumference sweep',
                  fitted_exponent=None,inputs=inputs,summary=summaries,
                  outputs=[dict(path=p.name,sha256=sha(p)) for p in files])
    (destination/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--analysis-root',type=Path)
    a=p.parse_args();analyze(a.output_root,a.analysis_root or a.output_root/'analysis')
