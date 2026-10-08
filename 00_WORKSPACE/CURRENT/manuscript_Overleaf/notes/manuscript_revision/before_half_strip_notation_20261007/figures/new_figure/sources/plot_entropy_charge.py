#!/usr/bin/env python3
"""Reproduce Figure 6 from bundled trajectory curves; no simulation or external data."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json
import numpy as np
import entropy_charge_support as base
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography
BUNDLE=Path(__file__).resolve().parent.parent
DATA=BUNDLE/'data/entropy_charge'
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

STYLES = {
    30: ("#D92725", "^"), 35: ("#F08050", "<"),
    40: ("#8FC1E3", "v"), 45: ("#2CA02C", "s"),
    50: ("#6B6B6B", "D"), 55: ("#000000", "P"),
    60: ("#1F77B4", "o"),
}

def main(output):
    output.mkdir(parents=True, exist_ok=True)
    stem=output/'Figure_06_entropy_charge'
    provenance=json.loads((DATA/'input_provenance.json').read_text())
    assert sha256(DATA/'sample_curves.npz')==provenance['compact_input_sha256']
    analyzer=SimpleNamespace(CURVE_KEYS={'c1':'endpoint__entropy_von_neumann','k':'endpoint__charge_variance'}, PREFACTOR={'c1':3.,'k':np.pi**2})
    cases={}
    with np.load(DATA/'sample_curves.npz',allow_pickle=False) as z:
        for ny in base.NY_VALUES:
            cases[ny]={'ay_values':np.arange(ny//2+1)}
            for key in analyzer.CURVE_KEYS.values():
                value=z[f'{key}_Ny{ny}'].copy()
                assert value.shape==(100,ny//2+1) and np.isfinite(value).all()
                cases[ny][key]=value
    base.configure_matplotlib()
    dimensions = (3.375, 4.6)
    fig, axes = base.plt.subplots(2, 1,
                                  figsize=dimensions, sharex=True)
    summaries = {}
    for ax, label, letter, symbol, ylabel in zip(
        axes, ("c1", "k"), ("(a)", "(b)"), ("c", "k"),
        (r"$\Delta\overline{S}_1$",
         r"$\Delta\overline{F}_A$"),
    ):
        by_size, summary = base.anchored_mean_curve_fit(analyzer, cases, label)
        summaries[label] = summary
        # Display-only exclusion; the locked Ay>=8 fitting inputs are untouched.
        for ny, item in by_size.items():
            ay = np.asarray(cases[ny]['ay_values'])
            keep = ay[ay >= 1] >= 2
            for key in ('x_all', 'mean_all', 'sem_all'):
                item[key] = item[key][keep]
        ax.axvspan(min(v["x_fit"].min() for v in by_size.values()), 0,
                   color="0.5", alpha=0.15, linewidth=0)
        for ny, (color, marker) in STYLES.items():
            item = by_size[ny]
            ax.errorbar(item["x_all"], item["mean_all"], yerr=item["sem_all"],
                        color=color, marker=marker, linestyle="none",
                        markerfacecolor="white", markeredgewidth=0.8,
                        markersize=3.4, elinewidth=0.45, capsize=0, zorder=3,
                        label=rf"$N_y={ny}$")
            assert item["mean_all"][-1] == item["sem_all"][-1] == 0
        x = np.linspace(min(v["x_all"].min() for v in by_size.values()), 0, 300)
        ax.plot(x, summary["slope"] * x, "k--", linewidth=0.9)
        ax.set(xlim=(float(x.min()) - .06, 0.05), ylabel=ylabel)
        ax.text(0.035, 0.92,
                rf"${symbol}={summary['converted_coefficient']:.4f}"
                rf"\pm{summary['converted_covariance_SEM']:.4f}$"
                "\n" + rf"$R^2={summary['R0_squared']:.6f}$",
                transform=ax.transAxes, fontsize=8, va="top", linespacing=1.35)
        base.panel_letter(ax, letter)
        assert not ax.get_title()
    axes[0].legend(ncol=2, loc="lower right", fontsize=8,
                   columnspacing=0.7, handletextpad=0.3, borderaxespad=0.45)
    axes[1].set_xlabel(
        r"$\log[D(A_y)/D(A_y^\star)]$")
    prepare_figure(fig, "Figure_06_entropy_charge")
    fig.subplots_adjust(left=0.19, right=0.975,
                        bottom=0.105,
                        top=0.935,
                        hspace=0.26)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        bounds = ax.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    record_typography(fig, "Figure_06_entropy_charge")
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    base.plt.close(fig)
    archived=json.loads((DATA/'original_figure_manifest.json').read_text())
    for label, fit in summaries.items():
        for key,value in fit.items():
            if isinstance(value,(int,float)):
                np.testing.assert_allclose(value,archived['fits'][label][key],rtol=2e-13,atol=1e-14)
            else:
                assert value==archived['fits'][label][key]
    checks={'input_hash_verified':True,'samples_per_size':100,'sizes':list(base.NY_VALUES),'display_min_Ay':2,'fit_window':'8 <= Ay <= floor(Ny/2)','fits_unchanged':True,'fits':summaries,'uncertainty':'ordinary trajectory SEM; full within-trajectory width covariance','renderer_sha256':sha256(__file__),'helper_sha256':sha256(base.__file__),'outputs':{f'Figure_06_entropy_charge.{ext}':sha256(output/f'Figure_06_entropy_charge.{ext}') for ext in ('pdf','png')}}
    target=output/'data/entropy_charge';target.mkdir(parents=True,exist_ok=True)
    (target/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BUNDLE)
    main(parser.parse_args().output_dir)
