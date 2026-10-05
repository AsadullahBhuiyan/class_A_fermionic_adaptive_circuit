"""Plot the spatially twirled endpoint, explicitly without time averaging."""
from pathlib import Path
import argparse
import hashlib
import json
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from observables import exact_y_twirl, ky_spectrum_and_x_weights, translation_residual

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/matched_markov_lindblad_perfect_correction_v1_20260821T222115Z_afb274a039e6"
OUT = ROOT / "analysis_outputs/endpoint_ky_spectrum_n20x64_nsh1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('markov_channel', 'lindblad'), default='markov_channel')
    family = parser.parse_args().family
    out = OUT if family == 'markov_channel' else ROOT / 'analysis_outputs/lindblad_dephasing_endpoint_ky_n20x64_nsh1'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "CMU Sans Serif", "font.size": 9,
                         "axes.labelsize": 10, "xtick.direction": "in",
                         "ytick.direction": "in", "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), sharey=True)
    summaries, products = [], {}
    for index, truncated in enumerate((True, False)):
        case_id = f"N20x64_ain1p0_aout30p0_nsh1_dwtrunc{int(truncated)}_{family}_deph1_pc1"
        case_root = RUN / "cases" / case_id
        meta = json.loads((case_root / "metadata.json").read_text())
        path = case_root / "observables.npz"
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        assert digest == meta["observables_sha256"]
        with np.load(path, allow_pickle=False) as saved:
            final = saved["G_final"]
            late = saved['G_late_cycle_average']
            last_step_distance = float(saved['successive_state_distance'][0, -1])
        assert final.shape == (1, 2560, 2560)
        twirled = exact_y_twirl(final[0], 20, 64)
        ky, occupations, weights = ky_spectrum_and_x_weights(twirled, 20, 64)
        assert occupations.min() >= -1e-10 and occupations.max() <= 1 + 1e-10
        trace_error = abs(occupations.sum() - np.trace(final[0]).real)
        assert trace_error < 1e-8
        order = np.argsort(ky)
        ax = axes[index]
        ax.scatter(np.repeat(ky[order] / np.pi, 40), occupations[order].ravel(),
                   s=6, c="#2468ad", linewidths=0, alpha=0.8)
        ax.axhline(0.5, c="0.45", ls="--", lw=0.8, zorder=0)
        ax.set(xlim=(-1.03, 1.03), ylim=(-0.035, 1.035), xlabel=r"$k_y/\pi$",
               title="DW-support truncation " + ("on" if truncated else "off"))
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.tick_params(top=True, right=True)
        ax.text(-0.13, 1.05, f"({chr(97+index)})", transform=ax.transAxes)
        name = "truncation_on" if truncated else "truncation_off"
        products[name + "_occupations"] = occupations[order]
        products[name + "_x_weights"] = weights[order]
        summaries.append({"case_id": case_id, "source": str(path), "sha256": digest,
                          "endpoint_cycle": 128, "schedule_samples": 1,
                          "dynamics_family": family,
                          "stationarity_residual_late": meta['adapter'].get('stationarity_residual_late'),
                          "last_step_distance": last_step_distance,
                          "endpoint_vs_late_relative_frobenius": float(np.linalg.norm(final-late)/np.linalg.norm(final)),
                          "sample_seed": meta["case"]["dynamics"]["sample_seeds"],
                          "input_array": "G_final", "time_averaging": False,
                          "translation_average": "(1/Ny) sum_s T_y^s G_final T_y^-s",
                          "raw_translation_residual": translation_residual(final[0], 20, 64),
                          "twirled_translation_residual": translation_residual(twirled, 20, 64),
                          "trace_error": float(trace_error),
                          "minimum_distance_to_half": float(np.min(abs(occupations-0.5))),
                          "k0_central_occupations": occupations[np.argmin(abs(ky)), 19:21].tolist()})
        print(json.dumps(summaries[-1]), flush=True)
    axes[0].set_ylabel(r"Occupation $\nu_a(k_y)$")
    title = (r"Perfect-correction channel endpoint: $20\times64$, $n_{\rm shell}=1$, cycle 128"
             if family == 'markov_channel' else
             r"Lindblad with dephasing: $20\times64$, $n_{\rm shell}=1$, $t=128$")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(out / f"endpoint_ky_occupations.{extension}", bbox_inches="tight")
    np.savez_compressed(out / "endpoint_ky_occupations.npz", ky=ky[order], **products)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    ensemble = ("Each panel uses one prescribed random schedule with measurement outcomes averaged analytically. "
                if family == 'markov_channel' else
                "Each panel is one deterministic Lindblad evolution with unit gain, loss, and number-dephasing rates. "
                "The endpoint is stationary to numerical precision; residuals are saved in summary.json. ")
    (out / "caption.txt").write_text(
        "Occupation spectrum of the exact y-translation twirl of G_final, at cycle 128. "
        "Nx=20, Ny=64, nshell=1, alpha_run,in=1, alpha_run,out=30, walls x=5,15; "
        "maximally mixed initialization, dephasing and perfect correction. "
        + ensemble +
        "All 40 occupations at each of 64 momenta are shown. Spatial averaging precedes "
        "diagonalization; no temporal averaging, fits, or uncertainty estimates are applied.\n"
    )


if __name__ == "__main__":
    main()
