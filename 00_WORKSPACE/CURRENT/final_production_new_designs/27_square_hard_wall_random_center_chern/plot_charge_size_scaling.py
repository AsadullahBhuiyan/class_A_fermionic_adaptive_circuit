"""Late-time absolute charge deviation, with trajectory-first SEMs."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import run_campaign as R

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'analysis_outputs/charge_variance_S100'
OUT = ROOT/'analysis_outputs/charge_size_scaling_S100'


def main():
    provenance = json.loads((SOURCE/'summary.json').read_text())
    source = SOURCE/'charge_samples.npz'
    assert R.digest(source) == provenance['outputs'][source.name]
    with np.load(source, allow_pickle=False) as z:
        sizes, q = z['sizes'], z['global_charge']
        np.testing.assert_array_equal(sizes, [20,30,40])
        np.testing.assert_array_equal(z['cycles'], np.arange(41))
        np.testing.assert_array_equal(z['sample_ids'], np.arange(100))
    assert q.shape == (3,100,41) and q.dtype.kind in 'iu'
    rows, sample_rows = [], []
    for l, charge in zip(sizes, q):
        delta = charge-l*l
        # Time is averaged within each trajectory; there are 100, not 2000,
        # independent units when estimating the standard error.
        absolute = np.abs(delta[:,21:41]).mean(axis=1)
        fraction = absolute/(2*l*l)
        signed = delta[:,21:41].mean(axis=1)
        mean,sem = float(absolute.mean()),float(absolute.std(ddof=1)/10)
        np.testing.assert_allclose(sem, np.sqrt(((absolute-mean)**2).sum()/(100*99)),atol=1e-14)
        rows.append(dict(L=int(l),samples=100,cycle_start=21,cycle_stop=40,
            mean_absolute_charge=mean,sem_absolute_charge=sem,
            mean_absolute_filling_deviation=float(fraction.mean()),sem_absolute_filling_deviation=float(fraction.std(ddof=1)/10),
            mean_signed_charge=float(signed.mean()),sem_signed_charge=float(signed.std(ddof=1)/10),
            endpoint_mean_absolute_charge=float(np.abs(delta[:,-1]).mean()),
            endpoint_sem_absolute_charge=float(np.abs(delta[:,-1]).std(ddof=1)/10)))
        for s in range(100):
            sample_rows.append(dict(L=int(l),sample_id=s,late_mean_absolute_charge=absolute[s],
                                   late_mean_absolute_filling_deviation=fraction[s],late_mean_signed_charge=signed[s]))
    OUT.mkdir(parents=True,exist_ok=True)
    for name, table in [('size_statistics.csv',rows),('trajectory_statistics.csv',sample_rows)]:
        with (OUT/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,'mathtext.fontset':'cm',
        'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(3.375,4.4),layout='constrained',sharex=True)
    for ax,key,err,color,marker in zip(axes,
            ('mean_absolute_charge','mean_absolute_filling_deviation'),
            ('sem_absolute_charge','sem_absolute_filling_deviation'),
            ('#1675bd','#d94738'),('o','^')):
        ax.errorbar(sizes,[r[key] for r in rows],yerr=[r[err] for r in rows],color=color,
            marker=marker,mfc='white',ms=4,lw=1,capsize=3,ls='-')
        ax.set_ylim(bottom=0);ax.set_xlim(18,42);ax.set_xticks(sizes)
        ax.tick_params(top=True,right=True)
    axes[0].set_ylabel(r'$\langle|Q-L^2|\rangle$')
    axes[1].set_ylabel(r'$\langle|f-\frac{1}{2}|\rangle$')
    axes[1].set_xlabel(r'$L$')
    axes[0].set_title(r'cycles 21–40, $S=100$',fontsize=9)
    axes[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    for label,ax in zip('ab',axes):
        ax.text(-.19,1.035,f'({label})',transform=ax.transAxes,fontsize=10)
    for ext in ('pdf','png'):fig.savefig(OUT/f'slab_charge_size_scaling.{ext}',dpi=300)
    plt.close(fig)
    (OUT/'caption.txt').write_text(
        'Charge deviation from half filling. (a) Mean absolute total-charge deviation |Q-L^2|; '
        '(b) mean absolute filling-fraction deviation |Q/(2L^2)-1/2|. '
        'Cycles 21–40 are averaged within each trajectory before computing the sample mean and SEM '
        '(ddof=1) over S=100 independent trajectories per size. Lines are guides to the eye, not fits. '
        'Hard walls, nshell=1, alpha1=1, alpha2=30, slab-only measurements, perfect correction, '
        'raster-y updates, periodic boundaries, pure half-filled random initialization followed by exterior preparation, '
        '40 total cycles. This is mean absolute deviation, not the signed mean or variance.\n')
    summary=dict(results=rows,input=str(source),input_checksum=R.digest(source),
        upstream_provenance=str(SOURCE/'summary.json'),upstream_provenance_checksum=R.digest(SOURCE/'summary.json'),
        estimator='Per trajectory: mean over cycles21..40 of abs(Q-L^2); ensemble mean and std(ddof=1)/sqrt(100). Filling deviation divides by 2L^2.',
        script=R.digest(Path(__file__)),outputs={p.name:R.digest(p) for p in OUT.iterdir() if p.suffix in ('.csv','.txt','.png','.pdf')})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
