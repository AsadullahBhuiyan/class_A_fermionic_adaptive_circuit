# --- project bootstrap ---
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)
# -------------------------

import time

# Keep CPU allocation behavior aligned with other Markov scripts
cpu_cap = 40
os.environ["MY_CPU_COUNT"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["MKL_NUM_THREADS"] = str(1)
os.environ["NUMEXPR_MAX_THREADS"] = str(1)
try:
    os.sched_setaffinity(0, set(range(cpu_cap)))
except Exception as exc:
    print(f"CPU affinity not set: {exc}")
try:
    print(os.sched_getaffinity(0))
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from fgtn.classA_U1FGTN import classA_U1FGTN

# ---- Config ----
NX = 16
NY = 21
CYCLES = 20
SAMPLES = 250
NSHELL = None
ALPHA_1 = 30.0
ALPHA_2 = 1.0
SEQUENCE = "dw_symmetric_random"
INIT_MODE = "default"  # default or maxmix
N_A = 0.5
PARALLEL_SAMPLES = True
# Let run_markov_circuit auto-throttle worker batches.
MAX_IN_FLIGHT = None
# Separate threading cap for post-processing fits.
FIT_N_JOBS = max(1, min(cpu_cap, int(os.environ.get("HYBRID_FIT_NJOBS", "4"))))
# --------------


def entanglement_contour_single(G_sub, nx, ny_sub):
    eye = np.eye(G_sub.shape[0], dtype=np.complex128)
    G2 = 0.5 * (eye + G_sub)
    evals, vecs = np.linalg.eigh(G2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
    f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    diagF = np.einsum("ik,k,ik->i", vecs, f_eigs, vecs.conj(), optimize=True).real
    diagF = diagF.reshape(2, nx, ny_sub, order="F")
    return diagF.sum(axis=0)


def build_sub_indices_from_ycut(nx, ny, y_cut):
    if not (0 <= y_cut < ny):
        raise ValueError(f"invalid y_cut={y_cut} for Ny={ny}")
    sub_indices = []
    for y in range(y_cut, ny):
        base = 2 * nx * y
        for x in range(nx):
            sub_indices.append(base + 2 * x)
            sub_indices.append(base + 2 * x + 1)
    return np.asarray(sub_indices, dtype=int)


def pair_sum_for_sample_final(G_final, s_idx, sub_idx, nx, ny_sub, pair):
    G_sub = G_final[s_idx][np.ix_(sub_idx, sub_idx)]
    s_map = entanglement_contour_single(G_sub, nx, ny_sub)
    return float(np.sum(s_map[pair, :]))


def mean_pair_curve_for_ycuts(G_final, y_cut_list, sample_idx, sub_idx_map, ny_sub_map, nx, pair):
    vals = np.empty(len(y_cut_list), dtype=float)
    for i, y_cut in enumerate(tqdm(y_cut_list, desc="integrated contour cuts", leave=False)):
        sub_idx = sub_idx_map[int(y_cut)]
        ny_sub = ny_sub_map[int(y_cut)]
        if PARALLEL_SAMPLES and len(sample_idx) > 1:
            per_s = Parallel(n_jobs=FIT_N_JOBS, backend="threading")(
                delayed(pair_sum_for_sample_final)(
                    G_final, int(s), sub_idx, nx, ny_sub, pair
                )
                for s in sample_idx
            )
            vals[i] = float(np.mean(per_s))
        else:
            sum_v = 0.0
            for s in sample_idx:
                sum_v += pair_sum_for_sample_final(G_final, int(s), sub_idx, nx, ny_sub, pair)
            vals[i] = sum_v / len(sample_idx)
    return vals


def fit_line_log_chord_with_error(Ay_vals, curve_vals, ny):
    x = np.log(np.sin(np.pi * Ay_vals / ny))
    y = np.asarray(curve_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan, x, y
    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        m, b = float(coeffs[0]), float(coeffs[1])
        m_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    except Exception:
        m, b, m_err = np.nan, np.nan, np.nan
    if not np.isfinite(m):
        return np.nan, np.nan, np.nan, np.nan, x, y
    y_hat = m * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    return m, m_err, r2, b, x, y


def main():
    t0 = time.time()

    model = classA_U1FGTN(NX, NY, nshell=NSHELL, DW=True, alpha_1=ALPHA_1, alpha_2=ALPHA_2)
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc not set; need DW=True with valid alpha profile")
    x0 = int(model.DW_loc[0]) % NX
    x1 = int(model.DW_loc[1]) % NX
    left_pair = np.array([x0 % NX, (x0 + 1) % NX], dtype=int)
    right_pair = np.array([(x1 - 1) % NX, x1 % NX], dtype=int)

    print(
        "[info] post_select_triv was removed from the canonical CPU Markov driver; "
        "this script now runs perfect_correction everywhere.",
        flush=True,
    )
    print(f"[info] DW slab x-range: [{x0}, {x1}] (inclusive)")
    print("[info] Parallel sample workers (max_in_flight) = auto")
    print(f"[info] Fit threading workers (FIT_N_JOBS) = {FIT_N_JOBS}")

    res = model.run_markov_circuit(
        G_history=False,
        progress=True,
        cycles=CYCLES,
        samples=SAMPLES,
        n_jobs=cpu_cap,
        parallelize_samples=PARALLEL_SAMPLES,
        init_mode=INIT_MODE,
        save=True,
        n_a=N_A,
        sequence=SEQUENCE,
        perfect_correction=True,
        max_in_flight=MAX_IN_FLIGHT,
        save_suffix="_perfect_correction_from_legacy_hybrid_topopc_trivps",
    )
    G_final = res.get("G_final")
    if G_final is None:
        save_path = res.get("save_path")
        if save_path and os.path.exists(save_path):
            with np.load(save_path) as data:
                if "G_final" in data:
                    G_final = data["G_final"]
                elif "G_hist" in data:
                    G_final = data["G_hist"][:, -1]
                else:
                    raise RuntimeError(f"No G_final/G_hist in {save_path}")
        else:
            raise RuntimeError("G_final not available and no readable save_path.")
    G_final = np.asarray(G_final, dtype=np.complex128)

    nlayer = 2 * NX * NY
    if G_final.ndim != 3 or G_final.shape[1:] != (nlayer, nlayer):
        raise ValueError(f"Unexpected G_final shape: {G_final.shape}")
    sample_count = G_final.shape[0]
    final_path = res.get("save_path")

    y_cut_base = np.arange(2, NY // 2, dtype=int)
    y_cut_list_1 = y_cut_base.copy()
    y_cut_list_2 = np.sort(NY - y_cut_base)
    Ay_list_1 = NY - y_cut_list_1
    Ay_list_2 = NY - y_cut_list_2

    all_y_cuts = np.unique(np.concatenate([y_cut_list_1, y_cut_list_2]))
    sub_idx_map = {}
    ny_sub_map = {}
    for y_cut in all_y_cuts:
        sub_idx_map[int(y_cut)] = build_sub_indices_from_ycut(NX, NY, int(y_cut))
        ny_sub_map[int(y_cut)] = NY - int(y_cut)

    sample_idx = np.arange(sample_count, dtype=int)
    left_curve_1 = mean_pair_curve_for_ycuts(G_final, y_cut_list_1, sample_idx, sub_idx_map, ny_sub_map, NX, left_pair)
    right_curve_1 = mean_pair_curve_for_ycuts(G_final, y_cut_list_1, sample_idx, sub_idx_map, ny_sub_map, NX, right_pair)
    left_curve_2 = mean_pair_curve_for_ycuts(G_final, y_cut_list_2, sample_idx, sub_idx_map, ny_sub_map, NX, left_pair)
    right_curve_2 = mean_pair_curve_for_ycuts(G_final, y_cut_list_2, sample_idx, sub_idx_map, ny_sub_map, NX, right_pair)

    cfgs = [
        ("Left DW, y_cut_list_1", Ay_list_1, left_curve_1, "#1f77b4", "o"),
        ("Right DW, y_cut_list_1", Ay_list_1, right_curve_1, "#ff7f0e", "s"),
        ("Left DW, y_cut_list_2", Ay_list_2, left_curve_2, "#2ca02c", "^"),
        ("Right DW, y_cut_list_2", Ay_list_2, right_curve_2, "#d62728", "D"),
    ]

    fit_stats = {}
    for name, Ay_vals, curve, _, _ in cfgs:
        m, m_err, r2, b, x_fit, y_fit = fit_line_log_chord_with_error(Ay_vals, curve, NY)
        fit_stats[name] = {
            "m": m,
            "m_err": m_err,
            "r2": r2,
            "b": b,
            "x": x_fit,
            "y": y_fit,
        }

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.1, 1.1])
    ax_fit = fig.add_subplot(gs[0, 0])
    ax_slope = fig.add_subplot(gs[1, 0])
    ax_r2 = fig.add_subplot(gs[2, 0])

    for name, _, _, color, marker in cfgs:
        stats = fit_stats[name]
        ax_fit.plot(
            stats["x"],
            stats["y"],
            linestyle="None",
            marker=marker,
            ms=5,
            color=color,
            label=f"{name}: m={stats['m']:.4g} +/- {stats['m_err']:.2g}, R^2={stats['r2']:.4f}",
        )
        if np.isfinite(stats["m"]) and np.isfinite(stats["b"]) and stats["x"].size > 0:
            xx = np.linspace(np.min(stats["x"]), np.max(stats["x"]), 200)
            ax_fit.plot(xx, stats["m"] * xx + stats["b"], "-", lw=1.5, color=color)

    ax_fit.set_xlabel("log(sin(pi |A_y| / N_y))")
    ax_fit.set_ylabel("trajectory-avg boundary integrated contour")
    ax_fit.set_title(
        f"Hybrid dynamics via run_markov_circuit (N={NX}x{NY}, C={CYCLES}, S={sample_count})\\n"
        "Topological slab: perfect correction | Trivial regions: post-selection"
    )
    ax_fit.grid(alpha=0.3)
    ax_fit.legend(fontsize=8)

    labels = [name for name, _, _, _, _ in cfgs]
    x_pos = np.arange(len(labels), dtype=int)
    slopes = [fit_stats[name]["m"] for name in labels]
    slope_errs = [fit_stats[name]["m_err"] for name in labels]
    r2_errs = [1.0 - fit_stats[name]["r2"] for name in labels]

    ax_slope.errorbar(x_pos, slopes, yerr=slope_errs, fmt="o", capsize=3, lw=1.4)
    ax_slope.set_xticks(x_pos)
    ax_slope.set_xticklabels(labels, rotation=20, ha="right")
    ax_slope.set_ylabel("slope m +/- fit stderr")
    ax_slope.set_title("Extracted slope by configuration")
    ax_slope.grid(alpha=0.3)

    ax_r2.plot(x_pos, r2_errs, "s-", lw=1.4)
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels(labels, rotation=20, ha="right")
    ax_r2.set_ylabel("R^2 error (1 - R^2)")
    ax_r2.set_title("Fit error by configuration")
    ax_r2.grid(alpha=0.3)

    figs_dir = ROOT / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    nshell_tag = "None" if NSHELL is None else str(NSHELL)
    pdf_name = (
        "markov_circuit_boundary_EC_fit_hybrid_topoPC_trivPS_"
        f"N{NX}x{NY}_C{int(CYCLES)}_S{int(sample_count)}_nsh{nshell_tag}.pdf"
    )
    pdf_path = figs_dir / pdf_name

    fig.tight_layout()
    fig.savefig(pdf_path, dpi=220)
    plt.close(fig)

    print(f"Saved final states: {final_path}")
    print(f"Saved characterization PDF: {pdf_path}")
    print("Fit summary:")
    for name in labels:
        st = fit_stats[name]
        print(
            f"  {name}: slope={st['m']:.8e}, slope_err={st['m_err']:.8e}, "
            f"intercept={st['b']:.8e}, R^2={st['r2']:.8f}"
        )

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
