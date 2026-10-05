"""Figure-12 analogue for the squared Born-averaged two-point function.

This is NOT the Born average of a trajectory-wise squared correlator.
No dynamics, fits, covariance flattening, or Gaussian Wick closure is used.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results/raster_y_channel_endpoints_v1_20260915T025739Z"
OUT = HERE / "analysis_outputs/raster_y_mean_channel_squared_correlator_v1"
SITES = (5, 6, 10, 14, 15)
DISPLAY_CUTOFF = 1e-8


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def square_correlator(c, nx, ny):
    """Sum |C[(x,y,mu),(x,y+r,nu)]|^2 over y,mu,nu / (2 Ny)."""
    c = np.asarray(c)
    if c.shape != (2*nx*ny, 2*nx*ny) or not np.isfinite(c).all():
        raise ValueError("Invalid occupation matrix")
    blocks = c.reshape(ny, nx, 2, ny, nx, 2)
    y = np.arange(ny)
    values = np.empty((nx, ny//2+1))
    for x in range(nx):
        for r in range(ny//2+1):
            values[x, r] = np.sum(abs(blocks[y, x, :, (y+r) % ny, x, :])**2)/(2*ny)
    return values


def load_case(alpha):
    receipt_path = ROOT / f"alpha{alpha}_hard/completion.json"
    receipt = json.loads(receipt_path.read_text())
    path = receipt_path.parent / receipt["result_filename"]
    assert receipt["status"] == "complete"
    assert path.stat().st_size == receipt["result_bytes"]
    assert sha(path) == receipt["result_sha256"]
    cfg = receipt["config"]
    assert (cfg["Nx"], cfg["Ny"], cfg["cycles"], cfg["alpha_1"]) == (20, 64, 128, alpha)
    assert cfg["wall"] == "hard" and cfg["site_schedule"] == "raster_y"
    assert cfg["family"] == "markov_channel" and cfg["evolution_domain"] == "full_system"
    assert cfg["active_initialization"] == "maxmix" and cfg["dtype"] == "complex128"
    with np.load(path, allow_pickle=False) as data:
        assert json.loads(str(data["config_json"])) == cfg
        np.testing.assert_array_equal(data["schedule_words"], np.tile(
            [x+20*y for x in range(20) for y in range(64)], (128, 1)))
        c = data["G_final"][0]  # Saved G_final is C=(engine G + I)/2.
        np.testing.assert_allclose(c, c.conj().T, atol=1e-13, rtol=0)
        values = square_correlator(c, 20, 64)
        twirled = square_correlator(data["G_final_twirl"][0], 20, 64)
        # Jensen's inequality: squaring after an additional spatial twirl
        # cannot exceed the origin-average of the squared amplitudes.
        assert np.all(twirled <= values + 1e-15)
        residual = float(np.max(data["successive_state_distance"].ravel()[-10:]))
        assert residual < 1e-12
    avg = values.mean(axis=0)
    return values, twirled, {
        "input": str(path), "input_sha256": receipt["result_sha256"],
        "receipt_sha256": sha(receipt_path), "config": cfg,
        "last_ten_cycle_max_normalized_frobenius_change": residual,
        "xavg_at_selected_separations": {str(r): float(avg[r]) for r in (1, 2, 3, 4, 5, 8, 16, 32)},
        "displayed_xavg_separations": (np.flatnonzero(avg[1:] > DISPLAY_CUTOFF)+1).tolist(),
        "additional_spatial_twirl_max_absolute_change": float(np.max(abs(values-twirled))),
    }


def plot(curves, *, uncut):
    plt.rcParams.update({"font.family": "CMU Sans Serif", "font.size": 8,
                         "axes.labelsize": 8, "legend.fontsize": 8})
    fig, axes = plt.subplots(3, 1, figsize=(3.375, 6.0))
    r = np.arange(1, 33)
    chord = (64/np.pi)*np.sin(np.pi*r/64)
    xx = np.log(chord)

    def line(ax, curve, label, color, marker, ls):
        values = curve[1:]
        visible = values > (0 if uncut else DISPLAY_CUTOFF)
        yy = np.full(values.shape, np.nan)
        yy[visible] = np.log(values[visible])
        ax.plot(xx, yy, color=color, marker=marker, ls=ls, ms=3,
                mfc="white", mew=.65, lw=.75, label=label)

    for alpha, color, marker, ls in [(1, "#0072B2", "o", "-"), (3, "#D55E00", "^", ":")]:
        line(axes[0], curves[alpha].mean(0), rf"$\alpha_1={alpha}$", color, marker, ls)
    axes[0].set_ylabel(r"$\log C_{\overline{G}}^{\mathrm{av}}(r_y)$")
    axes[0].legend(loc="lower left" if uncut else "lower right", frameon=False)
    for ax, alpha in zip(axes[1:], (1, 3)):
        for site, color, marker, ls in zip(SITES,
                ("#0072B2", "#E69F00", "#555555", "#009E73", "#D55E00"),
                ("o", "s", "D", "^", "v"), ("-", "--", ":", "-.", ":")):
            line(ax, curves[alpha][site], rf"$x={site}$", color, marker, ls)
        ax.text(.96, .96, rf"$\alpha_1={alpha}$", transform=ax.transAxes, ha="right", va="top")
        ax.set_ylabel(r"$\log C_{\overline{G}}(x,r_y)$")
        ax.legend(loc="lower left" if uncut else "lower right", ncol=2, frameon=False, columnspacing=.7,
                  handlelength=1.3, handletextpad=.3, labelspacing=.25)
    for ax, letter in zip(axes, "abc"):
        ax.set(xlim=(-.06, 3.08), xlabel=r"$\log d_{64}(r_y)$")
        ax.set_xticks([0, 1, 2, 3])
        ax.tick_params(direction="in", top=True, right=True)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.text(-.17, 1.035, f"({letter})", transform=ax.transAxes, fontsize=9)
        if not uncut:
            ax.set_ylim(np.log(DISPLAY_CUTOFF)-.3, -2.5)
    fig.subplots_adjust(left=.205, right=.975, bottom=.075, top=.975, hspace=.53)
    stem = "hard_wall_mean_channel_correlator_3x1" + ("_uncut" if uncut else "")
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    curves, twirled, diagnostics = {}, {}, {}
    for alpha in (1, 3):
        curves[alpha], twirled[alpha], diagnostics[str(alpha)] = load_case(alpha)
    # This comparison changes alpha_1 only; retain the entire source contract.
    configs = [dict(diagnostics[str(a)]["config"]) for a in (1, 3)]
    for cfg in configs:
        cfg.pop("alpha_1")
    assert configs[0] == configs[1]
    np.savez_compressed(OUT / "curves.npz", ry=np.arange(33), x=np.arange(20),
        **{f"alpha{a}_xresolved": curves[a] for a in (1, 3)},
        **{f"alpha{a}_xavg": curves[a].mean(0) for a in (1, 3)},
        **{f"alpha{a}_spatial_twirl_before_square": twirled[a] for a in (1, 3)})
    with (OUT / "curves.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["alpha_1", "x", "ry", "correlator", "shown_at_1e-8_cutoff"])
        for alpha in (1, 3):
            for x in ["average", *range(20)]:
                row = curves[alpha].mean(0) if x == "average" else curves[alpha][x]
                for r, value in enumerate(row):
                    writer.writerow([alpha, x, r, value, r > 0 and value > DISPLAY_CUTOFF])
    for uncut in (False, True):
        plot(curves, uncut=uncut)
    summary = {
        "estimator": "sum_y,mu,nu |Cbar[(x,y,mu),(x,y+r,nu)]|^2 / (2 Ny); then optional mean_x; then log",
        "notation": "Cbar is the occupation matrix; plot C_{Gbar} denotes its squared correlator, not centered covariance",
        "spatial_twirl_before_square": False, "fit": None, "display_cutoff": DISPLAY_CUTOFF,
        "size_collapse": "not available: these raster-y endpoints have only Ny=64",
        "cases": diagnostics,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = [p for p in OUT.iterdir() if p.name != "manifest.json"]
    (OUT / "manifest.json").write_text(json.dumps({
        "script": str(Path(__file__)), "script_sha256": sha(Path(__file__)),
        "files_sha256": {p.name: sha(p) for p in sorted(files)},
    }, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
