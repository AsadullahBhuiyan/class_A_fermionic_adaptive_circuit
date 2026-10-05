"""Sample-to-sample filling-fraction variance; no dynamics or time pooling."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import run_campaign as R

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / R.REVISION
OUT = ROOT / 'analysis_outputs/charge_variance_S100'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((DATA/'execution_plan.json').read_text())
    R.validate_plan(plan, R.default_config(), R.identity(R.default_config()))
    groups, sources = {l: [] for l in (20,30,40)}, []
    for task in tqdm(plan['tasks'], desc='Verify charge inputs', unit='batch'):
        path = DATA/(task['id']+'.npz')
        receipt = json.loads(path.with_suffix('.json').read_text())
        assert receipt['task'] == task and receipt['identity'] == plan['identity']
        assert receipt['result'] == path.name and receipt['file'] == R.digest(path)
        with np.load(path, allow_pickle=False) as z:
            assert json.loads(str(z['metadata_json'])) == dict(task=task, identity=plan['identity'],
                config=plan['config'], entry_point='classA_U1FGTN_gpu.run_markov_circuit')
            np.testing.assert_array_equal(z['cycles'], np.arange(41))
            np.testing.assert_array_equal(z['sample_ids'], task['sample_ids'])
            q = z['global_charge']
            assert q.shape == (len(task['sample_ids']),41) and q.dtype.kind in 'iu'
            assert np.all((q >= 0) & (q <= 2*task['nx']**2))
            np.testing.assert_array_equal(q[:,-1], z['final_ranks'])
            groups[task['nx']].append((z['sample_ids'],q))
        sources.append(dict(path=str(path), **receipt['file']))
    curves, rows, summaries, charges = {}, [], [], []
    for l, parts in groups.items():
        ids = np.concatenate([p[0] for p in parts])
        order = np.argsort(ids)
        np.testing.assert_array_equal(ids[order],np.arange(100))
        q = np.concatenate([p[1] for p in parts])[order]
        charges.append(q)
        # Subtract integer half filling first, avoiding cancellation near f=1/2.
        deviation = (q-l*l)/(2*l*l)
        mean = deviation.mean(axis=0)
        variance = deviation.var(axis=0, ddof=1)
        np.testing.assert_allclose(variance, q.var(axis=0,ddof=1)/(4*l**4),atol=1e-20,rtol=1e-12)
        np.testing.assert_allclose(variance, ((deviation-mean)**2).sum(axis=0)/99,atol=1e-20,rtol=1e-12)
        mse = (deviation**2).mean(axis=0)
        np.testing.assert_allclose(mse, .99*variance+mean**2, atol=1e-20,rtol=1e-12)
        curves[l] = variance
        for t in range(41):
            rows.append(dict(L=l,samples=100,cycle=t,mean_filling_deviation=mean[t],
                sample_variance_filling_deviation=variance[t],sample_std_filling_deviation=np.sqrt(variance[t]),
                mean_squared_deviation_from_half_filling=mse[t],mean_absolute_deviation=np.abs(deviation[:,t]).mean(),
                sample_variance_relative_half_filling=4*variance[t],sample_variance_total_charge=q[:,t].var(ddof=1)))
        summaries.append(dict(L=l,endpoint_variance=float(variance[-1]),
            endpoint_std=float(np.sqrt(variance[-1])),endpoint_mean_deviation=float(mean[-1]),
            endpoint_mean_squared_deviation=float(mse[-1]),
            mean_cyclewise_variance_cycles21_40=float(variance[21:41].mean())))
    with (OUT/'charge_statistics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'charge_samples.npz',sizes=[20,30,40],sample_ids=np.arange(100),cycles=np.arange(41),
                        global_charge=np.stack(charges))
    plt.rcParams.update({'font.family':'CMU Sans Serif','font.size':9,'mathtext.fontset':'cm',
        'xtick.direction':'in','ytick.direction':'in','legend.frameon':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(3.375,2.7),layout='constrained')
    for l,c,m,ls in zip((20,30,40),('#d94738','#249b57','#1675bd'),('^','s','o'),(':','--','-')):
        ax.plot(np.arange(41),curves[l],color=c,marker=m,ls=ls,ms=3,lw=1,mfc='white',mew=.7,label=rf'${l}\times{l}$')
    ax.set(xlabel='cycle',ylabel=r'$\mathrm{Var}_{\xi}[f_\xi-\frac{1}{2}]$',xlim=(0,40),ylim=(0,None))
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    ax.tick_params(top=True,right=True)
    ax.legend(loc='center right',fontsize=8,handlelength=2.5)
    for ext in ('pdf','png'): fig.savefig(OUT/f'slab_charge_variance.{ext}',dpi=300)
    plt.close(fig)
    (OUT/'caption.txt').write_text(
        'Sample-to-sample charge fluctuations for square hard-wall samples. '
        'At each cycle we plot the unbiased variance over S=100 independent trajectories of '
        'f_xi-1/2, where f_xi=Q_xi/(2L^2). The divisor is S-1=99, not S or S(S-1). '
        'This is the variance about the ensemble mean, not the mean squared deviation from half filling '
        'and not the uncertainty of the mean. Curves have no added uncertainty bands. '
        'All samples use nshell=1, alpha1=1, alpha2=30, hard walls, slab-only measurements, '
        'perfect correction, raster-y order, periodic boundaries, and 40 cycles. '
        'Pure half-filled random initialization is followed by exterior preparation; cycle zero is the prepared state. '
        'No cycles are pooled. Each trajectory has definite integer charge; this is record-to-record variation.\n')
    summary=dict(revision=R.REVISION,config=plan['config'],sources=sources,results=summaries,
        statistic='sum_xi [(Q_xi-Qbar)/(2L^2)]^2/(S-1), separately at each cycle',
        alternative_normalization='Variance of Q/L^2-1 is four times the plotted variance',
        script=R.digest(Path(__file__)),outputs={p.name:R.digest(p) for p in OUT.iterdir() if p.suffix in ('.csv','.npz','.png','.pdf','.txt')})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
    print(OUT,flush=True)


if __name__ == '__main__':
    main()
