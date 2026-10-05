#!/usr/bin/env python3
"""Combine the verified mean von Neumann and charge collapses without titles."""

import json

import numpy as np

import make_contour_scaling_figures as base


STEM = base.FIGURE_DIR / "endpoint_entropy_charge_mean_curve_collapse_2x1"
STYLES = {
    30: ("#D92725", "^"), 35: ("#F08050", "<"),
    40: ("#8FC1E3", "v"), 45: ("#2CA02C", "s"),
    50: ("#6B6B6B", "D"), 55: ("#000000", "P"),
    60: ("#1F77B4", "o"),
}


def load_archived_curve_cases(analyzer):
    """Read immutable historical curves against their pinned download manifest.

    Current engine sources may have changed since acquisition; no current-code
    source identity is substituted into the archived receipts or result files.
    """
    path = base.OUTPUT_ROOT / 'DOWNLOAD_MANIFEST.json'
    expected_sha = 'ac8f8f233f5a4ecf2868181fad99810cb676590ea30adb49d9039658c2b3eeb4'
    if base.sha256(path) != expected_sha:
        raise RuntimeError('Historical download-manifest checksum mismatch')
    archive_manifest = json.loads(path.read_text())
    for entry in archive_manifest['files']:
        file = base.OUTPUT_ROOT / entry['path']
        if file.stat().st_size != entry['bytes'] or base.sha256(file) != entry['sha256']:
            raise RuntimeError(f'Historical file checksum mismatch: {file}')
    results = analyzer.discover(base.OUTPUT_ROOT)
    cases = {}
    for ny in base.NY_VALUES:
        keys = [analyzer.CURVE_KEYS[label] for label in ('c1', 'k')]
        arrays = {key: [] for key in keys}
        ids = []
        for path, receipt in results:
            if int(receipt['Ny']) != ny:
                continue
            assert receipt['config_sha256'] == archive_manifest['config_sha256']
            with np.load(path, allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload['ay_values'], np.arange(ny // 2 + 1))
                np.testing.assert_array_equal(payload['sample_ids'], receipt['global_sample_indices'])
                ids.extend(payload['sample_ids'].tolist())
                for key in keys:
                    value = payload[key].copy()
                    assert value.shape == (5, ny // 2 + 1) and np.isfinite(value).all()
                    arrays[key].append(value)
        np.testing.assert_array_equal(sorted(ids), np.arange(100))
        order = np.argsort(ids)
        cases[ny] = {key: np.concatenate(arrays[key])[order] for key in keys}
        cases[ny]['ay_values'] = np.arange(ny // 2 + 1)
    return cases


def main(include_spectrum=False):
    stem = (base.FIGURE_DIR / "endpoint_entropy_charge_spectrum_3x1"
            if include_spectrum else STEM)
    analyzer = base.load_bundle_analysis()
    cases = load_archived_curve_cases(analyzer)
    base.configure_matplotlib()
    dimensions = (3.375, 6.8) if include_spectrum else (3.375, 4.6)
    fig, axes = base.plt.subplots(3 if include_spectrum else 2, 1,
                                  figsize=dimensions, sharex=not include_spectrum)
    if include_spectrum:
        axes[0].sharex(axes[1])
        axes[0].tick_params(labelbottom=False)
    summaries = {}
    for ax, label, letter, symbol, ylabel in zip(
        axes, ("c1", "k"), ("(a)", "(b)"), ("c_1", "k"),
        (r"$\Delta\langle\overline{S}_1\rangle_\xi$",
         r"$\Delta\langle\overline{F}_A\rangle_\xi$"),
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
                "\n" + rf"$R_0^2={summary['R0_squared']:.6f}$",
                transform=ax.transAxes, fontsize=7.5, va="top", linespacing=1.35)
        base.panel_letter(ax, letter)
        assert not ax.get_title()
    axes[0].legend(ncol=2, loc="lower right", fontsize=6.4,
                   columnspacing=0.7, handletextpad=0.3, borderaxespad=0.45)
    axes[1].set_xlabel(
        r"$\log\!\left[\sin(\pi A_y/N_y)/\sin(\pi A_y^\star/N_y)\right]$")
    spectrum_metadata = None
    if include_spectrum:
        source = (base.HERE.parent.parent / "final_production_new_designs" /
                  "09_pure_tangent_replay_acquisition/analysis_outputs/" /
                  "half_system_centered_spectrum_n20_sizes_hard_alpha1_v1/" /
                  "mixed_mode_density_overlay.csv")
        table = np.genfromtxt(source, delimiter=",", names=True)
        ax = axes[2]
        integrals = {}
        for ny, color, marker, style in ((24, "#c62828", "^", ":"),
                                        (28, "#2e7d32", "s", "--"),
                                        (32, "#1565c0", "o", "-")):
            rows = table[table["Ny"] == ny]
            edges = np.r_[rows["left_edge"], rows["right_edge"][-1]]
            density = rows["conditional_probability_density"]
            integrals[str(ny)] = float(density @ np.diff(edges))
            np.testing.assert_allclose(integrals[str(ny)], 1, atol=1e-14)
            values = np.where(rows["count"] > 0, density, np.nan)
            ax.stairs(values, edges, color=color, linestyle=style, linewidth=.85,
                      label=rf"$N_y={ny}$")
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers[::8], values[::8], linestyle="none", marker=marker,
                    markersize=2.6, markerfacecolor="none", color=color)
        ax.set(yscale="log", xlim=(-1, 1), xlabel=r"Centered occupation $\lambda=2\nu-1$",
               ylabel=r"$\rho_{\mathrm{mixed}}(\lambda)$")
        ax.set_xticks([-1, -.5, 0, .5, 1])
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=6.8,
                  columnspacing=.8, handlelength=1.6, handletextpad=.4)
        base.panel_letter(ax, "(c)")
        spectrum_metadata = {"source": str(source), "sha256": base.sha256(source),
                             "campaign": "09_pure_tangent_replay_acquisition",
                             "sizes": [24, 28, 32], "samples_per_size": 100,
                             "cut": "all x, y in [0, Ny/2), both orbitals",
                             "filter": "abs(lambda) < 1-1e-8",
                             "normalization": "pooled retained levels, unit area per size",
                             "density_integrals": integrals}
    fig.subplots_adjust(left=0.19, right=0.975,
                        bottom=0.065 if include_spectrum else 0.105,
                        top=0.965 if include_spectrum else 0.935,
                        hspace=0.40 if include_spectrum else 0.26)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in axes:
        bounds = ax.get_tightbbox(renderer)
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), dpi=300)
    base.plt.close(fig)
    manifest = {
        "schema": "endpoint_entropy_charge_spectrum_3x1_v1" if include_spectrum else "endpoint_entropy_charge_mean_collapse_2x1_v1",
        "campaign": base.OUTPUT_ROOT.name,
        "verified_result_pairs": 140,
        "archive_validation": "all files checked against pinned historical DOWNLOAD_MANIFEST.json; current engine identity not substituted",
        "sizes": list(base.NY_VALUES), "samples_per_size": 100,
        "size_inches": list(dimensions), "titles": False,
        "spectrum_panel": spectrum_metadata,
        "fit_window": "8 <= Ay <= floor(Ny/2)",
        "display_window": "2 <= Ay <= floor(Ny/2); Ay=1 omitted from panels a,b only",
        "uncertainty": "ordinary trajectory SEM, full covariance across widths",
        "fits": summaries,
        "source_hashes": {__file__: base.sha256(base.Path(__file__)),
                          str(base.Path(base.__file__)): base.sha256(base.Path(base.__file__))},
        "outputs": {stem.with_suffix(s).name: base.sha256(stem.with_suffix(s))
                    for s in (".pdf", ".png")},
    }
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
