"""Enlarge the ensemble-mean entropy contour alone to one journal column."""
import json
import numpy as np
import make_contour_scaling_figures as base


def main():
    analyzer = base.load_bundle_analysis()
    cases = analyzer.load_cases(analyzer.discover(base.OUTPUT_ROOT))
    samples = cases[60][analyzer.CONTOUR_KEYS["c1"]]
    assert samples.shape == (100, 20, 30)
    contour = samples.mean(axis=0)
    base.configure_matplotlib()
    base.plt.rcParams.update({"axes.labelsize": 10, "xtick.labelsize": 9,
                             "ytick.labelsize": 9})
    fig = base.plt.figure(figsize=(3.375, 5.05))
    ax = fig.add_axes((.20, .23, .72, .7217821782))
    cax = fig.add_axes((.20, .10, .72, .024))
    scale = base.contour_panel(fig, ax, contour, cax=cax,
                               colorbar_orientation="horizontal", cmap="Blues",
                               colorbar_label=r"$\langle s_1(x,y)\rangle_\xi$")
    cax.tick_params(labelsize=8, pad=2)
    cax.set_xlabel(r"$\langle s_1(x,y)\rangle_\xi$", fontsize=10, labelpad=3)
    assert not ax.get_title()
    # Uniformly reduce the original standalone figure by the requested 1.5x.
    reduction = 1.5
    fig.set_size_inches(3.375 / reduction, 5.05 / reduction)
    for text in fig.findobj(match=base.mpl.text.Text):
        text.set_fontsize(text.get_fontsize() / reduction)
    for axis in (ax, cax):
        for spine in axis.spines.values():
            spine.set_linewidth(spine.get_linewidth() / reduction)
        axis.tick_params(which="major", length=2.8 / reduction, width=.65 / reduction)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for axis in (ax, cax):
        bounds = axis.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    stem = base.FIGURE_DIR / "endpoint_entropy_contour_single_column"
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    base.plt.close(fig)
    metadata = {
        "campaign": str(base.OUTPUT_ROOT), "verified_result_pairs": 140,
        "Nx": 20, "Ny": 60, "endpoint_cycle": 120, "samples": 100,
        "region": "[0,20) x [0,30)",
        "averaging": "Cellwise mean of 100 trajectory contours; no origin average",
        "figure_inches": [3.375 / reduction, 5.05 / reduction],
        "reduction_factor": reduction, "color_scale": scale,
        "unit_cell_grid": "gray, alpha=0.28, linewidth=0.22",
        "contour_sum": float(contour.sum()),
        "script_sha256": base.sha256(base.Path(__file__)),
        "outputs": {stem.with_suffix(s).name: base.sha256(stem.with_suffix(s))
                    for s in (".pdf", ".png")},
    }
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
