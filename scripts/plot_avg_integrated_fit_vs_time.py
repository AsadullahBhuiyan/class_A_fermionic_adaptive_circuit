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

import hashlib
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from fgtn.classA_U1FGTN import classA_U1FGTN
# ---- Config ----
DATA_PATH = (
    "cache/G_history_samples/"
    "N12x31/"
    "N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_slope_drift_testing.npz"
)
PLOT_PATH = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_avg_curve_fit_vs_t.pdf"
)
STATS_PATH = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_avg_curve_fit_vs_t.npz"
)
NX = 12
NY = 31
ALPHA_1 = 30.0
ALPHA_2 = 1.0
MAX_SAMPLES = None
CPU_USE =10
AFFINITY_START = 0
PARALLEL_SAMPLES = False
# ----------------


def configure_cpu(cpu_use, affinity_start):
    os.environ["MY_CPU_COUNT"] = str(cpu_use)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"
    try:
        os.sched_setaffinity(0, set(range(affinity_start, affinity_start + cpu_use)))
    except Exception as exc:
        print(f"CPU affinity not set: {exc}")
    try:
        print(os.sched_getaffinity(0))
    except Exception:
        pass


def path_cache_key(path):
    apath = os.path.abspath(path)
    exists = os.path.exists(apath)
    stat = os.stat(apath) if exists else None
    payload = "|".join(
        [
            apath,
            str(exists),
            str(stat.st_size if stat else -1),
            str(int(stat.st_mtime) if stat else -1),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_G_hist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "G_hist" not in data:
            raise KeyError(f"'G_hist' not found in {path}")
        return data["G_hist"]


def sample_indices(total_samples, max_samples):
    if max_samples is None or max_samples >= total_samples:
        return np.arange(total_samples, dtype=int)
    return np.linspace(0, total_samples - 1, max_samples, dtype=int)


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


def entanglement_contour_single(G_sub, nx, ny_sub):
    I = np.eye(G_sub.shape[0], dtype=np.complex128)
    G2 = 0.5 * (I + G_sub)
    evals, vecs = np.linalg.eigh(G2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
    f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    diagF = np.einsum("ik,k,ik->i", vecs, f_eigs, vecs.conj(), optimize=True).real
    diagF = diagF.reshape(2, nx, ny_sub, order="F")
    return diagF.sum(axis=0)


def fit_line_log_chord_with_error(Ay_vals, curve_vals, ny):
    x = np.log(np.sin(np.pi * Ay_vals / ny))
    y = np.asarray(curve_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    try:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        m, b = float(coeffs[0]), float(coeffs[1])
        m_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    except Exception:
        m, b, m_err = np.nan, np.nan, np.nan
    if not np.isfinite(m):
        return np.nan, np.nan, np.nan, np.nan
    y_hat = m * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    return m, m_err, r2, b


def pair_sum_for_sample(G_hist, s_idx, t_idx, sub_idx, nx, ny_sub, pair):
    G_sub = G_hist[s_idx, t_idx][np.ix_(sub_idx, sub_idx)]
    s_map = entanglement_contour_single(G_sub, nx, ny_sub)
    return float(np.sum(s_map[pair, :]))


def mean_pair_curve_for_ycuts(
    G_hist,
    t_idx,
    y_cut_list,
    sample_idx,
    sub_idx_map,
    ny_sub_map,
    nx,
    pair,
):
    vals = np.empty(len(y_cut_list), dtype=float)
    for i, y_cut in enumerate(y_cut_list):
        sub_idx = sub_idx_map[int(y_cut)]
        ny_sub = ny_sub_map[int(y_cut)]
        if PARALLEL_SAMPLES and len(sample_idx) > 1:
            per_s = Parallel(n_jobs=CPU_USE, backend="threading")(
                delayed(pair_sum_for_sample)(
                    G_hist, int(s), int(t_idx), sub_idx, nx, ny_sub, pair
                )
                for s in sample_idx
            )
            vals[i] = float(np.mean(per_s))
        else:
            sum_v = 0.0
            for s in sample_idx:
                sum_v += pair_sum_for_sample(
                    G_hist, int(s), int(t_idx), sub_idx, nx, ny_sub, pair
                )
            vals[i] = sum_v / len(sample_idx)
    return vals


def main():
    configure_cpu(CPU_USE, AFFINITY_START)
    t0 = time.time()

    cache_keys = {
        "data_path": path_cache_key(DATA_PATH),
        "plot_path": path_cache_key(PLOT_PATH),
        "stats_path": path_cache_key(STATS_PATH),
    }

    G_hist = load_G_hist(DATA_PATH)
    S_tot, T, Nlayer, Nlayer2 = G_hist.shape
    if Nlayer != Nlayer2:
        raise ValueError(f"G_hist last dims must be square; got {Nlayer} x {Nlayer2}")
    if 2 * NX * NY != Nlayer:
        raise ValueError(f"2*NX*NY={2*NX*NY} != Nlayer={Nlayer}")

    sample_idx = sample_indices(S_tot, MAX_SAMPLES)
    time_steps = np.arange(T, dtype=int)

    model = classA_U1FGTN(
        NX, NY, nshell=None, DW=True, alpha_1=ALPHA_1, alpha_2=ALPHA_2
    )
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc not set; need DW=True with valid alpha profile")
    x0 = int(model.DW_loc[0]) % NX
    x1 = int(model.DW_loc[1]) % NX
    left_pair = np.array([x0 % NX, (x0 + 1) % NX], dtype=int)
    right_pair = np.array([(x1 - 1) % NX, x1 % NX], dtype=int)

    y_cut_list_1 = np.arange(2, NY // 2, dtype=int)
    y_cut_list_2 = np.arange(NY // 2, NY - 1, dtype=int)
    n = min(len(y_cut_list_1), len(y_cut_list_2))
    y_cut_list_1 = y_cut_list_1[:n]
    y_cut_list_2 = y_cut_list_2[:n]
    Ay_list_1 = NY - y_cut_list_1
    Ay_list_2 = NY - y_cut_list_2

    all_y_cuts = np.unique(np.concatenate([y_cut_list_1, y_cut_list_2]))
    sub_idx_map = {}
    ny_sub_map = {}
    for y_cut in all_y_cuts:
        sub_idx_map[int(y_cut)] = build_sub_indices_from_ycut(NX, NY, int(y_cut))
        ny_sub_map[int(y_cut)] = NY - int(y_cut)

    cfg_names = [
        "Left DW, y_cut_list_1",
        "Right DW, y_cut_list_1",
        "Left DW, y_cut_list_2",
        "Right DW, y_cut_list_2",
    ]
    slope = {k: np.full(len(time_steps), np.nan, dtype=float) for k in cfg_names}
    slope_err = {k: np.full(len(time_steps), np.nan, dtype=float) for k in cfg_names}
    r2 = {k: np.full(len(time_steps), np.nan, dtype=float) for k in cfg_names}

    tbar = tqdm(time_steps, desc="time-steps", unit="t")
    for it, t_idx in enumerate(tbar):
        tbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)

        left_curve_1 = mean_pair_curve_for_ycuts(
            G_hist, t_idx, y_cut_list_1, sample_idx, sub_idx_map, ny_sub_map, NX, left_pair
        )
        right_curve_1 = mean_pair_curve_for_ycuts(
            G_hist,
            t_idx,
            y_cut_list_1,
            sample_idx,
            sub_idx_map,
            ny_sub_map,
            NX,
            right_pair,
        )
        left_curve_2 = mean_pair_curve_for_ycuts(
            G_hist, t_idx, y_cut_list_2, sample_idx, sub_idx_map, ny_sub_map, NX, left_pair
        )
        right_curve_2 = mean_pair_curve_for_ycuts(
            G_hist,
            t_idx,
            y_cut_list_2,
            sample_idx,
            sub_idx_map,
            ny_sub_map,
            NX,
            right_pair,
        )

        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_1, left_curve_1, NY)
        slope[cfg_names[0]][it], slope_err[cfg_names[0]][it], r2[cfg_names[0]][it] = m, e, q
        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_1, right_curve_1, NY)
        slope[cfg_names[1]][it], slope_err[cfg_names[1]][it], r2[cfg_names[1]][it] = m, e, q
        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_2, left_curve_2, NY)
        slope[cfg_names[2]][it], slope_err[cfg_names[2]][it], r2[cfg_names[2]][it] = m, e, q
        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_2, right_curve_2, NY)
        slope[cfg_names[3]][it], slope_err[cfg_names[3]][it], r2[cfg_names[3]][it] = m, e, q

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    for name in cfg_names:
        ax0.errorbar(
            time_steps,
            slope[name],
            yerr=slope_err[name],
            marker="o",
            lw=1.4,
            capsize=3,
            label=name,
        )
    ax0.set_ylabel("Slope m +/- fit stderr")
    ax0.set_title("Slope from fit to sample-averaged integrated contour")
    ax0.grid(alpha=0.3)
    ax0.legend(ncol=2, fontsize=9)

    for name in cfg_names:
        ax1.plot(time_steps, r2[name], marker="o", lw=1.4, label=name)
    ax1.set_xlabel("t")
    ax1.set_ylabel("R^2")
    ax1.set_title("R^2 vs time-step")
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)

    np.savez(
        STATS_PATH,
        time_steps=time_steps,
        cfg_names=np.asarray(cfg_names, dtype=object),
        slope=np.asarray([slope[k] for k in cfg_names], dtype=float),
        slope_stderr=np.asarray([slope_err[k] for k in cfg_names], dtype=float),
        r2=np.asarray([r2[k] for k in cfg_names], dtype=float),
        data_path=np.asarray(DATA_PATH),
        plot_path=np.asarray(PLOT_PATH),
        stats_path=np.asarray(STATS_PATH),
        data_path_cache_key=np.asarray(cache_keys["data_path"]),
        plot_path_cache_key=np.asarray(cache_keys["plot_path"]),
        stats_path_cache_key=np.asarray(cache_keys["stats_path"]),
    )

    elapsed = time.time() - t0
    print(f"Data path:   {DATA_PATH}")
    print(f"Data key:    {cache_keys['data_path']}")
    print(f"Plot path:   {PLOT_PATH}")
    print(f"Plot key:    {cache_keys['plot_path']}")
    print(f"Stats path:  {STATS_PATH}")
    print(f"Stats key:   {cache_keys['stats_path']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
