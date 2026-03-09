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
# ---- Config (edit here) ----
DATA_PATH = (
    "cache/G_history_samples/"
    "N12x31/"
    "N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_slope_drift_testing.npz"
)
PLOT_PATH = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_sample_fit_stats_memopt.pdf"
)
STATS_PATH = (
    "figs/N12x31_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_random_"
    "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_sample_fit_stats_memopt.npz"
)
NX = 12
NY = 31
ALPHA_1 = 30.0
ALPHA_2 = 1.0
MAX_SAMPLES = None
CPU_USE = 30
AFFINITY_START = 60
PARALLEL_SAMPLES = False
# ---------------------------


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


def sample_indices(total_samples, max_samples):
    if max_samples is None or max_samples >= total_samples:
        return np.arange(total_samples, dtype=int)
    return np.linspace(0, total_samples - 1, max_samples, dtype=int)


def load_G_hist(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, mmap_mode="r")
    if ext == ".npz":
        with np.load(path) as data:
            if "G_hist" not in data:
                raise KeyError(f"'G_hist' not found in {path}")
            return data["G_hist"]
    raise ValueError(f"Unsupported data extension: {ext}")


def build_sub_indices_bottom_trace(nx, ny, ay):
    if not (1 <= ay <= ny):
        raise ValueError(f"invalid Ay={ay} for Ny={ny}")
    sub_indices = []
    for y in range(ay):
        base = 2 * nx * y
        for x in range(nx):
            sub_indices.append(base + 2 * x)
            sub_indices.append(base + 2 * x + 1)
    return np.asarray(sub_indices, dtype=int)


def build_subsystem_maps(nx, ny, ay_list):
    sub_idx_map = {}
    ny_sub_map = {}
    for ay in ay_list:
        sub_idx_map[int(ay)] = build_sub_indices_bottom_trace(nx, ny, int(ay))
        ny_sub_map[int(ay)] = int(ay)
    return sub_idx_map, ny_sub_map


def entanglement_contour_single(G_sub, nx, ny_sub):
    I = np.eye(G_sub.shape[0], dtype=np.complex128)
    G2 = 0.5 * (I + G_sub)
    evals, vecs = np.linalg.eigh(G2)
    evals = np.clip(np.real_if_close(evals), 1e-12, 1 - 1e-12)
    f_eigs = -(evals * np.log(evals) + (1.0 - evals) * np.log(1.0 - evals))
    diagF = np.einsum("ik,k,ik->i", vecs, f_eigs, vecs.conj(), optimize=True).real
    diagF = diagF.reshape(2, nx, ny_sub, order="F")
    return diagF.sum(axis=0)


def fit_slope_r2_log_chord(ay_vals, curve_vals, ny):
    x = np.log(np.sin(np.pi * ay_vals / ny))
    y = np.asarray(curve_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot == 0 else (1.0 - ss_res / ss_tot)
    return float(m), float(r2)


def summarize(arr):
    arr = np.asarray(arr, dtype=float)
    mask = np.isfinite(arr)
    valid = arr[mask]
    if valid.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(valid))
    if valid.size == 1:
        return mean, 0.0, 0.0
    std = float(np.std(valid, ddof=1))
    stderr = float(std / np.sqrt(valid.size))
    return mean, std, stderr


def fit_one_sample_one_t(
    G_st,
    nx,
    ny,
    ay_list_1,
    ay_list_2,
    sub_idx_map,
    ny_sub_map,
    left_pair,
    right_pair,
):
    left_curve_1 = np.empty(len(ay_list_1), dtype=float)
    right_curve_1 = np.empty(len(ay_list_1), dtype=float)
    left_curve_2 = np.empty(len(ay_list_2), dtype=float)
    right_curve_2 = np.empty(len(ay_list_2), dtype=float)

    for k, ay in enumerate(ay_list_1):
        idx = sub_idx_map[int(ay)]
        G_sub = G_st[np.ix_(idx, idx)]
        s_map = entanglement_contour_single(G_sub, nx, ny_sub_map[int(ay)])
        left_curve_1[k] = float(np.sum(s_map[left_pair, :]))
        right_curve_1[k] = float(np.sum(s_map[right_pair, :]))

    for k, ay in enumerate(ay_list_2):
        idx = sub_idx_map[int(ay)]
        G_sub = G_st[np.ix_(idx, idx)]
        s_map = entanglement_contour_single(G_sub, nx, ny_sub_map[int(ay)])
        left_curve_2[k] = float(np.sum(s_map[left_pair, :]))
        right_curve_2[k] = float(np.sum(s_map[right_pair, :]))

    m_l1, r2_l1 = fit_slope_r2_log_chord(ay_list_1, left_curve_1, ny)
    m_r1, r2_r1 = fit_slope_r2_log_chord(ay_list_1, right_curve_1, ny)
    m_l2, r2_l2 = fit_slope_r2_log_chord(ay_list_2, left_curve_2, ny)
    m_r2, r2_r2 = fit_slope_r2_log_chord(ay_list_2, right_curve_2, ny)
    return (m_l1, r2_l1, m_r1, r2_r1, m_l2, r2_l2, m_r2, r2_r2)


def main():
    configure_cpu(CPU_USE, AFFINITY_START)

    t_start = time.time()
    nx, ny = int(NX), int(NY)
    cache_keys = {
        "data_path": path_cache_key(DATA_PATH),
        "plot_path": path_cache_key(PLOT_PATH),
        "stats_path": path_cache_key(STATS_PATH),
    }

    G_hist = load_G_hist(DATA_PATH)
    S_tot, T, Nlayer, Nlayer2 = G_hist.shape
    if Nlayer != Nlayer2:
        raise ValueError(f"G_hist last dims must be square, got {Nlayer} x {Nlayer2}")
    if 2 * nx * ny != Nlayer:
        raise ValueError(f"2*Nx*Ny={2*nx*ny} but data layer is {Nlayer}")
    time_steps = np.arange(5,T, dtype=int)

    model = classA_U1FGTN(
        nx,
        ny,
        nshell=None,
        DW=True,
        alpha_1=ALPHA_1,
        alpha_2=ALPHA_2,
    )
    if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
        raise ValueError("DW_loc missing; ensure DW=True for model construction.")
    x0 = int(model.DW_loc[0]) % nx
    x1 = int(model.DW_loc[1]) % nx
    left_pair = np.array([x0 % nx, (x0 + 1) % nx], dtype=int)
    right_pair = np.array([(x1 - 1) % nx, x1 % nx], dtype=int)

    y_cut_list_1 = np.arange(2, ny // 2, dtype=int)
    y_cut_list_2 = np.arange(ny // 2, ny - 1, dtype=int)
    n_cuts = min(len(y_cut_list_1), len(y_cut_list_2))
    y_cut_list_1 = y_cut_list_1[:n_cuts]
    y_cut_list_2 = y_cut_list_2[:n_cuts]
    ay_list_1 = ny - y_cut_list_1
    ay_list_2 = ny - y_cut_list_2

    all_ay = np.unique(np.concatenate([ay_list_1, ay_list_2]))
    sub_idx_map, ny_sub_map = build_subsystem_maps(nx, ny, all_ay)

    s_idx = sample_indices(S_tot, MAX_SAMPLES)
    S_use = len(s_idx)

    cfg_names = [
        "Left DW, y_cut_list_1",
        "Right DW, y_cut_list_1",
        "Left DW, y_cut_list_2",
        "Right DW, y_cut_list_2",
    ]
    slopes_by_cfg = {k: np.full((len(time_steps), S_use), np.nan) for k in cfg_names}
    r2_by_cfg = {k: np.full((len(time_steps), S_use), np.nan) for k in cfg_names}

    tbar = tqdm(time_steps, desc="time-steps", unit="t")
    for it, t_idx in enumerate(tbar):
        if not (0 <= t_idx < T):
            raise ValueError(f"time-step {t_idx} out of range for T={T}")
        tbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)

        if PARALLEL_SAMPLES and S_use > 1:
            out = Parallel(n_jobs=CPU_USE, backend="threading")(
                delayed(fit_one_sample_one_t)(
                    G_hist[s, t_idx],
                    nx,
                    ny,
                    ay_list_1,
                    ay_list_2,
                    sub_idx_map,
                    ny_sub_map,
                    left_pair,
                    right_pair,
                )
                for s in s_idx
            )
            for isamp, row in enumerate(out):
                slopes_by_cfg[cfg_names[0]][it, isamp] = row[0]
                r2_by_cfg[cfg_names[0]][it, isamp] = row[1]
                slopes_by_cfg[cfg_names[1]][it, isamp] = row[2]
                r2_by_cfg[cfg_names[1]][it, isamp] = row[3]
                slopes_by_cfg[cfg_names[2]][it, isamp] = row[4]
                r2_by_cfg[cfg_names[2]][it, isamp] = row[5]
                slopes_by_cfg[cfg_names[3]][it, isamp] = row[6]
                r2_by_cfg[cfg_names[3]][it, isamp] = row[7]
            continue

        sbar = tqdm(s_idx, desc=f"t={t_idx} samples", unit="sample", leave=False)
        for isamp, s in enumerate(sbar):
            sbar.set_postfix_str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), refresh=False)
            row = fit_one_sample_one_t(
                G_hist[s, t_idx],
                nx,
                ny,
                ay_list_1,
                ay_list_2,
                sub_idx_map,
                ny_sub_map,
                left_pair,
                right_pair,
            )
            slopes_by_cfg[cfg_names[0]][it, isamp] = row[0]
            r2_by_cfg[cfg_names[0]][it, isamp] = row[1]
            slopes_by_cfg[cfg_names[1]][it, isamp] = row[2]
            r2_by_cfg[cfg_names[1]][it, isamp] = row[3]
            slopes_by_cfg[cfg_names[2]][it, isamp] = row[4]
            r2_by_cfg[cfg_names[2]][it, isamp] = row[5]
            slopes_by_cfg[cfg_names[3]][it, isamp] = row[6]
            r2_by_cfg[cfg_names[3]][it, isamp] = row[7]

    slope_mean = {k: [] for k in cfg_names}
    slope_std = {k: [] for k in cfg_names}
    slope_stderr = {k: [] for k in cfg_names}
    r2_mean = {k: [] for k in cfg_names}
    r2_std = {k: [] for k in cfg_names}
    r2_stderr = {k: [] for k in cfg_names}

    for k in cfg_names:
        for it in range(len(time_steps)):
            m_mean, m_std, m_stderr = summarize(slopes_by_cfg[k][it, :])
            q_mean, q_std, q_stderr = summarize(r2_by_cfg[k][it, :])
            slope_mean[k].append(m_mean)
            slope_std[k].append(m_std)
            slope_stderr[k].append(m_stderr)
            r2_mean[k].append(q_mean)
            r2_std[k].append(q_std)
            r2_stderr[k].append(q_stderr)
        slope_mean[k] = np.asarray(slope_mean[k], dtype=float)
        slope_std[k] = np.asarray(slope_std[k], dtype=float)
        slope_stderr[k] = np.asarray(slope_stderr[k], dtype=float)
        r2_mean[k] = np.asarray(r2_mean[k], dtype=float)
        r2_std[k] = np.asarray(r2_std[k], dtype=float)
        r2_stderr[k] = np.asarray(r2_stderr[k], dtype=float)

    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 13), sharex=True)

    ax0 = axes[0]
    for k in cfg_names:
        ax0.errorbar(
            time_steps,
            slope_mean[k],
            yerr=slope_stderr[k],
            marker="o",
            lw=1.4,
            capsize=3,
            label=k,
        )
    ax0.set_ylabel("Slope mean +/- stderr")
    ax0.set_title("Mean fit slope vs time-step")
    ax0.grid(alpha=0.3)
    ax0.legend(ncol=2, fontsize=9)

    ax1 = axes[1]
    for k in cfg_names:
        ax1.plot(time_steps, slope_std[k], marker="o", lw=1.4, label=k)
    ax1.set_ylabel("Slope stdev")
    ax1.set_title("Slope standard deviation vs time-step")
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=2, fontsize=9)

    ax2 = axes[2]
    for k in cfg_names:
        ax2.errorbar(
            time_steps,
            r2_mean[k],
            yerr=r2_stderr[k],
            marker="o",
            lw=1.4,
            capsize=3,
            label=k,
        )
    ax2.set_xlabel("t")
    ax2.set_ylabel("R^2 mean +/- stderr")
    ax2.set_title("Average R^2 vs time-step")
    ax2.grid(alpha=0.3)
    ax2.legend(ncol=2, fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)

    np.savez(
        STATS_PATH,
        time_steps=np.asarray(time_steps, dtype=int),
        cfg_names=np.asarray(cfg_names, dtype=object),
        data_path=np.asarray(DATA_PATH),
        plot_path=np.asarray(PLOT_PATH),
        stats_path=np.asarray(STATS_PATH),
        data_path_cache_key=np.asarray(cache_keys["data_path"]),
        plot_path_cache_key=np.asarray(cache_keys["plot_path"]),
        stats_path_cache_key=np.asarray(cache_keys["stats_path"]),
        slope_mean=np.asarray([slope_mean[k] for k in cfg_names], dtype=float),
        slope_std=np.asarray([slope_std[k] for k in cfg_names], dtype=float),
        slope_stderr=np.asarray([slope_stderr[k] for k in cfg_names], dtype=float),
        r2_mean=np.asarray([r2_mean[k] for k in cfg_names], dtype=float),
        r2_std=np.asarray([r2_std[k] for k in cfg_names], dtype=float),
        r2_stderr=np.asarray([r2_stderr[k] for k in cfg_names], dtype=float),
        slopes_per_sample=np.asarray([slopes_by_cfg[k] for k in cfg_names], dtype=float),
        r2_per_sample=np.asarray([r2_by_cfg[k] for k in cfg_names], dtype=float),
    )

    elapsed = time.time() - t_start
    print(f"Data path:   {DATA_PATH}")
    print(f"Data key:    {cache_keys['data_path']}")
    print(f"Plot path:   {PLOT_PATH}")
    print(f"Plot key:    {cache_keys['plot_path']}")
    print(f"Stats path:  {STATS_PATH}")
    print(f"Stats key:   {cache_keys['stats_path']}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
