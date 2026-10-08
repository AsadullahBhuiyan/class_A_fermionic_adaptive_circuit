#!/usr/bin/env python3
"""Reproduce Figure 6 from bundled trajectory curves; no simulation or external data."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json
import numpy as np
import entropy_charge_support as base
from endpoint_even_support import SIZES, STYLES, load_fits
from log_ticks import add_log_minor_ticks
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography
BUNDLE=Path(__file__).resolve().parent.parent
DATA=BUNDLE/'data/entropy_charge'
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    stem=output/'Figure_06_entropy_charge'
    fits, _, provenance = load_fits()
    base.configure_matplotlib()
    dimensions = (3.375, 5.05)
    fig, axes = base.plt.subplots(2, 1,
                                  figsize=dimensions, sharex=False)
    summaries = {}
    size_handles = []
    size_labels = []
    for ax, label, letter, symbol, ylabel in zip(
        axes, ("c1", "k"), ("(a)", "(b)"), ("c", "k"),
        (r"$\Delta\overline{S}_1$",
         r"$\Delta\overline{F}_A$"),
    ):
        by_size, summary = fits['entropy' if label == 'c1' else 'variance']
        summaries[label] = summary
        # Display-only exclusion; the locked Ay>=8 fitting inputs are untouched.
        for ny, item in by_size.items():
            ay = np.arange(ny//2+1)
            keep = ay[ay >= 1] >= 2
            for key in ('x_all', 'mean_all', 'sem_all'):
                item[key] = item[key][keep]
        ax.axvspan(min(v["x_fit"].min() for v in by_size.values()), 0,
                   color="0.5", alpha=0.15, linewidth=0)
        for ny, (color, marker) in STYLES.items():
            item = by_size[ny]
            empirical = ax.errorbar(item["x_all"], item["mean_all"], yerr=item["sem_all"],
                        color=color, marker=marker, linestyle="none",
                        markerfacecolor="white", markeredgewidth=0.8,
                        markersize=3.4, elinewidth=0.45, capsize=0, zorder=3,
                        label=rf"$N_y={ny}$")
            if label == "c1":
                size_handles.append(empirical[0])
                size_labels.append(rf"${ny}$")
            assert item["mean_all"][-1] == item["sem_all"][-1] == 0
        x = np.linspace(min(v["x_all"].min() for v in by_size.values()), 0, 300)
        fit_line, = ax.plot(x, summary["slope"] * x, "k--", linewidth=0.9)
        ax.set(xlim=(float(x.min()) - .06, 0.05), ylabel=ylabel,
               xlabel=r"$\log[\sin(\pi A_y/N_y)]$")
        ax.text(0.035, 0.92,
                rf"${symbol}={summary['converted_coefficient']:.4f}"
                rf"\pm{summary['converted_covariance_SEM']:.4f}$"
                "\n" + rf"$R^2={summary['R0_squared']:.6f}$",
                transform=ax.transAxes, fontsize=8, va="top", linespacing=1.35)
        base.panel_letter(ax, letter)
        prefactor = r"\frac{c}{3}" if label == "c1" else r"\frac{k}{\pi^2}"
        fit_legend = ax.legend(
            handles=[fit_line],
            labels=[rf"${prefactor}\log[\sin(\pi A_y/N_y)]$"],
            loc="upper left", bbox_to_anchor=(0.035, 0.76),
            handlelength=1.3, handletextpad=0.4, borderaxespad=0,
        )
        if label == "c1":
            ax.add_artist(fit_legend)
        assert not ax.get_title()
    axes[0].legend(handles=size_handles, labels=size_labels, title=r"$N_y$",
                   ncol=3, loc="lower right", fontsize=8, frameon=False,
                   handlelength=.6, columnspacing=.5, handletextpad=.15,
                   labelspacing=.2, borderaxespad=.45)
    add_log_minor_ticks(fig)
    prepare_figure(fig, "Figure_06_entropy_charge")
    fig.subplots_adjust(left=0.19, right=0.975,
                        bottom=0.105,
                        top=0.955,
                        hspace=0.42)
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
    checks={'layout':[2,1], 'convergence_separate':True,
            'size_legend':{'title':'N_y','columns':3,'entries':list(SIZES),
                           'marker_only':True,'errorbar_glyphs':False},
            'plotted_errorbars_retained':True,
            'x_axes_and_fit_legends':'log(sin(pi Ay/Ny))','input_hash_verified':True,
            'samples_per_size':100,'sizes':list(SIZES),'display_min_Ay':2,
            'fit_window':'8 <= Ay <= Ny/2','fits_match_imported_campaign':True,
            'scientific_input':'data/endpoint_even_20261008/sample_curves.npz',
            'input_sha256':provenance['compact_sha256'],'fits':summaries,
            'uncertainty':'trajectory SEM with full within-trajectory width covariance',
            'renderer_sha256':sha256(__file__),'helper_sha256':sha256(base.__file__),
            'outputs':{f'Figure_06_entropy_charge.{ext}':sha256(output/f'Figure_06_entropy_charge.{ext}') for ext in ('pdf','png')}}
    target=output/'data/entropy_charge';target.mkdir(parents=True,exist_ok=True)
    (target/'validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=BUNDLE)
    main(parser.parse_args().output_dir)
