#!/usr/bin/env python3
"""Render Figure 07 from its portable, trajectory-resolved saved inputs."""

from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from manuscript_palette import ALPHA_COLORS
from manuscript_typography import configure_style as manuscript_style, prepare_figure, record_typography


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "entanglement_spectrum"
STEM = ROOT / "Figure_07_entanglement_spectrum"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    occupation_provenance = json.loads((DATA / 'pooled_half_strip_provenance.json').read_text())
    assert sha256(DATA / 'pooled_half_strip_spectra.npz') == occupation_provenance['compact_input_sha256']
    with np.load(DATA / 'pooled_half_strip_spectra.npz', allow_pickle=False) as payload:
        full_occupations = payload['centered_occupations'].copy()
        np.testing.assert_array_equal(payload['alpha_1'], [1, 3])
        np.testing.assert_array_equal(payload['sample_ids'], np.arange(100))
        np.testing.assert_array_equal(payload['origins'], np.arange(32))
        for name, expected_value in dict(Nx=20, Ny=32, Ay=16, cycle=64).items():
            assert payload[name].item() == expected_value
    assert full_occupations.shape == (2, 100, 32, 640) and np.isfinite(full_occupations).all()
    assert full_occupations.min() >= -1-1e-8 and full_occupations.max() <= 1+1e-8
    occupation_edges = np.linspace(-1, 1, 101)
    # Roundoff at the physical endpoints must not discard almost-pure modes.
    occupation_counts = np.array([np.histogram(np.clip(v.ravel(), -1, 1), occupation_edges)[0]
                                  for v in full_occupations])
    np.testing.assert_array_equal(occupation_counts.sum(axis=1), [2048000, 2048000])
    # Map the original bin edges to occupations; retain identical bin membership.
    occupation_edges = (occupation_edges + 1) / 2
    occupation_density = occupation_counts / (2048000 * np.diff(occupation_edges))
    np.testing.assert_allclose((occupation_density * np.diff(occupation_edges)).sum(axis=1), 1,
                               rtol=0, atol=1e-14)
    provenance = json.loads((DATA / "provenance.json").read_text())
    assert sha256(DATA / "inputs.npz") == provenance["compact_input_sha256"]
    with np.load(DATA / "inputs.npz", allow_pickle=False) as payload:
        values = {name: payload[name].copy() for name in payload.files}
    np.testing.assert_array_equal(values["sample_ids"], np.arange(100))
    np.testing.assert_array_equal(values["origins"], np.arange(32))
    np.testing.assert_array_equal(values["widths"], np.arange(1, 17))
    assert all(values[name].item() == expected for name, expected in
               (("L", .99), ("Nx", 20), ("Ny", 32), ("Ay", 16), ("cycle", 64)))
    occupation = values["retained_centered_occupations"]
    energy = np.log1p(-occupation) - np.log1p(occupation)
    assert np.isfinite(occupation).all() and np.all(np.abs(occupation) <= .99)
    np.testing.assert_array_equal(energy, values["retained_energies"])
    np.testing.assert_allclose(-np.tanh(energy / 2), occupation, rtol=0, atol=5e-16)
    assert len(energy) == 95398
    energy_limit = np.log(199)
    edges = np.linspace(-energy_limit, energy_limit, 102)
    histogram, _ = np.histogram(energy, bins=edges)
    assert histogram.sum() == len(energy) and len(histogram) == 101
    energy_histograms = []
    window_counts = []
    central_energy_counts = []
    for alpha, full in zip((1, 3), full_occupations):
        assert not np.any(np.abs(full) == .99), "Boundary modes require explicit cutoff reconciliation"
        window = np.abs(full) < .99
        selected = full[window]
        energies = np.log1p(-selected) - np.log1p(selected)
        np.testing.assert_allclose(-np.tanh(energies/2), selected, atol=5e-16, rtol=0)
        bins, _ = np.histogram(energies, edges)
        assert bins.sum() == energies.size
        n = window.sum(axis=-1)
        np.testing.assert_array_equal(n[:, :16], n[:, 16:])
        energy_histograms.append(bins)
        window_counts.append(n)
        central_energy_counts.append(int((abs(energies)<1).sum()))
        if alpha == 1:
            np.testing.assert_array_equal(energies, energy)
            np.testing.assert_array_equal(bins, histogram)
    energy_histograms = np.array(energy_histograms)
    window_counts = np.array(window_counts)
    retained_totals = energy_histograms.sum(axis=1)
    energy_densities = energy_histograms / (retained_totals[:, None] * np.diff(edges))
    np.testing.assert_allclose((energy_densities * np.diff(edges)).sum(axis=1), 1, atol=1e-14, rtol=0)

    # These are correlated cut origins, not 3200 independent trajectories.
    counts = values["counts_by_sample_width_origin"]
    assert counts.shape == (100, 16, 32)
    half_counts = np.zeros((100, 32), dtype=int)
    np.add.at(half_counts, (values["retained_sample_id"], values["retained_origin"]), 1)
    np.testing.assert_array_equal(half_counts, counts[:, -1, :])
    np.testing.assert_array_equal(half_counts, window_counts[0])
    np.testing.assert_array_equal(half_counts[:, :16], half_counts[:, 16:])
    trajectory_counts = counts.mean(axis=-1)
    means = trajectory_counts.mean(axis=0)
    sems = trajectory_counts.std(axis=0, ddof=1) / np.sqrt(100)
    assert means[-1] == 29.811875
    widths = values["widths"]
    log_chord = np.log(32 / np.pi * np.sin(np.pi * widths / 32))
    fit_mask = (widths >= 5) & (widths <= 16)
    design = np.column_stack([np.ones(fit_mask.sum()), log_chord[fit_mask]])
    projection = np.linalg.pinv(design)
    trajectory_coefficients = trajectory_counts[:, fit_mask] @ projection.T
    coefficients = trajectory_coefficients.mean(axis=0)
    coefficient_sem = trajectory_coefficients.std(axis=0, ddof=1) / np.sqrt(100)
    np.testing.assert_allclose(coefficients, projection @ means[fit_mask], atol=1e-12, rtol=0)
    residual = means[fit_mask] - design @ coefficients
    r_squared = 1 - (residual @ residual) / np.sum(
        (means[fit_mask] - means[fit_mask].mean()) ** 2)
    expected = provenance["expected_fit"]
    for value, key in ((coefficients[0], "intercept"), (coefficients[1], "slope"),
                       (coefficient_sem[0], "intercept_sem"),
                       (coefficient_sem[1], "slope_sem"), (r_squared, "R_squared"),
                       (sems[-1], "mean_modes_sem_at_reference_Ay")):
        np.testing.assert_allclose(value, expected[key], atol=1e-12, rtol=0)

    manuscript_style({'pdf.fonttype': 42, 'savefig.dpi': 300, 'text.color': 'black', 'axes.labelcolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in'})
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 5.8))
    color = ALPHA_COLORS[1]
    ax = axes[0]
    for density, alpha, style, hue in zip(occupation_density, (1, 3), ('-', ':'), (color, ALPHA_COLORS[3])):
        ax.stairs(density, occupation_edges, baseline=None, color=hue, linestyle=style,
                  linewidth=1, label=rf'$\alpha_1={alpha}$')
    positive_density = occupation_density[occupation_density > 0]
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set(xlabel=r'Occupation $\nu$',
           ylabel=r'Probability density $\rho(\nu)$', xlim=(-.01, 1.01),
           yscale='log', ylim=(positive_density.min()/2, positive_density.max()*1.5))
    ax.text(.5, .96, r'$N_y=32$, $A_y=16$',
            transform=ax.transAxes, ha='center', va='top')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, .82), ncol=2, frameon=False,
              handlelength=1.6, columnspacing=1)
    ax = axes[1]
    for density, alpha, style, hue in zip(energy_densities, (1, 3), ('-', ':'), (color, ALPHA_COLORS[3])):
        ax.stairs(density, edges, fill=True, color=hue, alpha=.08)
        ax.stairs(density, edges, color=hue, linestyle=style, linewidth=1.1,
                  label=rf'$\alpha_1={alpha}$')
    ax.set(xlabel=r"Entanglement energy $\varepsilon$", ylabel=r"Conditional density $p_W(\varepsilon)$",
           xlim=(-energy_limit, energy_limit), ylim=(0, energy_densities.max() * 1.32))
    ax.text(.5, .93, r"$W:\ |2\nu-1|<0.99$",
            transform=ax.transAxes, ha="center", va="top")
    ax.legend(loc='upper center', bbox_to_anchor=(.5, .80), ncol=2, frameon=False,
              handlelength=1.6, columnspacing=1)
    ax = axes[2]
    ax.axvspan(log_chord[fit_mask].min(), log_chord[fit_mask].max(),
               color="0.92", linewidth=0, zorder=0)
    xfit = np.linspace(log_chord.min(), log_chord.max(), 250)
    ax.plot(xfit, coefficients[0] + coefficients[1] * xfit,
            color="black", linestyle="--", linewidth=.9, zorder=1)
    ax.errorbar(log_chord, means, yerr=sems, color=color, marker="o",
                markerfacecolor="white", markeredgewidth=.8, linestyle="none",
                markersize=3.5, capsize=2, elinewidth=.7, zorder=3)
    ax.set(xlabel=r"$\log D(A_y)$", ylabel=r"$\overline{N}_{0.99}$")
    ax.text(.05, .52,
            rf"$b={coefficients[1]:.3f}\pm{coefficient_sem[1]:.3f}$" + "\n" +
            rf"$R^2={r_squared:.4f}$",
            transform=ax.transAxes, ha="left", va="top", linespacing=1.35)
    ax.text(.97, .05, r"$\alpha_1=1$"+'\n'+r"Fit: $5\leq A_y\leq16$",
            transform=ax.transAxes, ha="right", va="bottom")
    for ax, letter in zip(axes, ("(a)", "(b)", "(c)")):
        ax.tick_params(which="both", top=True, right=True)
        ax.text(-.08, 1.045, letter, transform=ax.transAxes, fontsize=9,
                fontweight="normal", va="bottom")
        assert ax.get_xscale() == "linear"
        assert ax.get_yscale() == ('log' if letter == '(a)' else 'linear')
    prepare_figure(fig, STEM.name)
    fig.tight_layout(pad=.75, h_pad=1.0)
    for ax in axes:
        bounds = ax.get_tightbbox(fig.canvas.get_renderer())
        assert bounds.x0 >= 0 and bounds.y0 >= 0
        assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
    record_typography(fig, STEM.name)
    fig.savefig(STEM.with_suffix(".pdf"))
    plt.close(fig)
    # Rasterize the vector artifact directly; this also avoids a dvipng dependency.
    subprocess.run(["pdftoppm", "-r", "300", "-singlefile", "-png",
                    str(STEM.with_suffix(".pdf")), str(STEM)], check=True)

    np.savetxt(DATA / 'occupation_histogram.csv', np.column_stack([
        occupation_edges[:-1], occupation_edges[1:], occupation_counts.T, occupation_density.T]),
        delimiter=',', header='nu_left,nu_right,alpha1_1_count,alpha1_3_count,alpha1_1_density,alpha1_3_density',
        comments='')
    np.savetxt(DATA / "energy_histogram.csv", np.column_stack([
        edges[:-1], edges[1:], histogram]), delimiter=",",
        header="energy_left,energy_right,raw_count", comments="")
    np.savetxt(DATA / 'energy_histogram_comparison.csv', np.column_stack([
        edges[:-1], edges[1:], energy_histograms.T, energy_densities.T]), delimiter=',',
        header='energy_left,energy_right,alpha1_1_count,alpha1_3_count,alpha1_1_density,alpha1_3_density', comments='')
    np.savez_compressed(DATA / 'half_strip_window_counts.npz', alpha_1=np.array([1, 3]),
                        sample_ids=np.arange(100), origins=np.arange(32), counts=window_counts)
    np.savetxt(DATA / "mean_mode_counts.csv", np.column_stack([
        widths, log_chord, means, sems]), delimiter=",",
        header="Ay,log_chord,mean_count,trajectory_SEM", comments="")
    validation = {
        "figure": STEM.name, "all_checks_passed": True, "samples": 100,
        "figure_inches": [3.375, 5.8], "layout": [3, 1],
        "occupation_panel": {
            "alpha_1": [1, 3], "samples_per_alpha": 100, "cut_origins": list(range(32)),
            "Ny": 32, "Ay": 16, "cycle": 64,
            "observations_per_alpha": occupation_counts.sum(axis=1).tolist(),
            "normalization": "unit-area probability density separately per alpha",
            "density_integrals": (occupation_density * np.diff(occupation_edges)).sum(axis=1).tolist(),
            "bins": 100, "range": [0, 1], "coordinate": "nu", "density_jacobian": 2, "y_axis": "log; empty bins not shown; no pseudocounts",
            "raw_extrema": [float(full_occupations.min()), float(full_occupations.max())],
            "roundoff_values_clipped": int((np.abs(full_occupations) > 1).sum()),
            "spectral_window_exclusion": False,
        },
        "energy_panel": {
            "alpha_1": [1, 3], "retained_observations_per_alpha": retained_totals.tolist(),
            "normalization": "unit-area conditional density after pooling samples and all cut origins",
            "density_integrals": (energy_densities*np.diff(edges)).sum(axis=1).tolist(),
            "bins": 101, "range": [-float(energy_limit), float(energy_limit)],
            "window": "abs(2nu-1)<0.99", "abs_energy_below_one": central_energy_counts,
            "half_strip_mean_counts": window_counts.mean(axis=2).mean(axis=1).tolist(),
            "half_strip_count_SEMs": (window_counts.mean(axis=2).std(axis=1,ddof=1)/10).tolist(),
        },
        "origins_per_trajectory": 32, "independent_sampling_unit": "Born trajectory",
        "retained_observations": int(histogram.sum()), "histogram_bins": 101,
        "histogram_normalization": "occupation: unit area over [0,1]; energy: unit area within window",
        "count_panel": "alpha_1=1, raw origin-averaged mean count; existing fit and trajectory SEM unchanged",
        "axes": {"a": "linear x, logarithmic y", "b": "linear", "c": "linear"},
        "display_widths": [1, 16], "fit_widths": [5, 16],
        "half_system_mean_count": float(means[-1]),
        "half_system_trajectory_SEM": float(sems[-1]),
        "intercept": float(coefficients[0]), "slope": float(coefficients[1]),
        "intercept_SEM": float(coefficient_sem[0]), "slope_SEM": float(coefficient_sem[1]),
        "R_squared": float(r_squared),
        "checks": ["compact source hashes", "full occupation counts conserved and densities integrate to one", "exact unique sample/origin IDs",
                   "window selection and energy inverse transform", "raw histogram count conservation",
                   "half-system per-origin counts equal retained spectrum counts",
                   "complementary half-system cut counts", "fit values and SEM agree with historical source",
                   "fitting ensemble mean equals ensemble mean of linear fits", "figure bounds"],
        "source_sha256": sha256(Path(__file__)),
        "outputs": {STEM.with_suffix(s).name: sha256(STEM.with_suffix(s)) for s in (".pdf", ".png")},
    }
    (DATA / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
