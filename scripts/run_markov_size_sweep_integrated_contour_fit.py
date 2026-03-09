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

import os
import time
from datetime import datetime

# Keep CPU allocation behavior aligned with run_markov_entropy_boundary_timeseries.py
cpu_cap = 30
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
NX = 12
NY_SIM_LIST = [16, 24]
NY_LOAD_LIST = None
NY_SIM_LIST = [] if NY_SIM_LIST is None else list(NY_SIM_LIST)
NY_LOAD_LIST = [] if NY_LOAD_LIST is None else list(NY_LOAD_LIST)
NY_LIST = sorted(set(NY_SIM_LIST + NY_LOAD_LIST))
CYCLES = 20
SAMPLES = 500
NSHELL = None
ALPHA_1 = 30.0
ALPHA_2 = 1.0
P_MEAS = 1.0
SEQUENCE = "random"
INIT_MODE = "default"
TOP_TRIV_BACK_FORTH = True
PARALLEL_SAMPLES = True
def cache_path_for_ny(ny):
    return (
        "cache/G_history_samples/"
        f"N12x{ny}/"
        f"N12x{ny}_C20_S250_nshNone_DW1_init-default_n_a0.5_seq-random_"
        "exclNone_pm1.00_tbtf1_tbtflm0_markov_circuit_final.npz"
    )


DATA_PATH_BY_NY = {ny: cache_path_for_ny(ny) for ny in NY_LOAD_LIST}
# --------------


def entanglement_contour_single(G_sub, nx, ny_sub):
    I = np.eye(G_sub.shape[0], dtype=np.complex128)
    G2 = 0.5 * (I + G_sub)
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


def pair_sum_for_sample_final(G_final, s_idx, sub_idx, nx, ny_sub, pair):
    G_sub = G_final[s_idx][np.ix_(sub_idx, sub_idx)]
    s_map = entanglement_contour_single(G_sub, nx, ny_sub)
    return float(np.sum(s_map[pair, :]))


def mean_pair_curve_for_ycuts(G_final, y_cut_list, sample_idx, sub_idx_map, ny_sub_map, nx, pair):
    vals = np.empty(len(y_cut_list), dtype=float)
    for i, y_cut in enumerate(y_cut_list):
        sub_idx = sub_idx_map[int(y_cut)]
        ny_sub = ny_sub_map[int(y_cut)]
        if PARALLEL_SAMPLES and len(sample_idx) > 1:
            per_s = Parallel(n_jobs=cpu_cap, backend="threading")(
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


def load_final_from_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as data:
        if "G_final" in data:
            G_final = data["G_final"]
        elif "G_hist" in data:
            G_hist = data["G_hist"]
            if G_hist.ndim != 4:
                raise ValueError(f"G_hist must have shape (S,T,N,N); got {G_hist.shape}")
            G_final = G_hist[:, -1]
        else:
            raise KeyError(f"'G_final' or 'G_hist' not found in {path}")
    if G_final.ndim != 3:
        raise ValueError(f"G_final must have shape (S,N,N); got {G_final.shape}")
    return G_final


def main():
    t0 = time.time()

    cfg_names = [
        "Left DW, y_cut_list_1",
        "Right DW, y_cut_list_1",
        "Left DW, y_cut_list_2",
        "Right DW, y_cut_list_2",
    ]
    slope = {k: np.full(len(NY_LIST), np.nan, dtype=float) for k in cfg_names}
    slope_err = {k: np.full(len(NY_LIST), np.nan, dtype=float) for k in cfg_names}
    r2 = {k: np.full(len(NY_LIST), np.nan, dtype=float) for k in cfg_names}

    ny_bar = tqdm(NY_LIST, desc="Ny sweep", unit="Ny")
    for i_ny, NY in enumerate(ny_bar):
        ny_bar.set_postfix({"Ny": NY, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, refresh=False)

        model = classA_U1FGTN(NX, NY, nshell=NSHELL, DW=True, alpha_1=ALPHA_1, alpha_2=ALPHA_2)
        if not (hasattr(model, "DW_loc") and len(model.DW_loc) >= 2):
            raise ValueError("DW_loc not set; need DW=True with valid alpha profile")
        x0 = int(model.DW_loc[0]) % NX
        x1 = int(model.DW_loc[1]) % NX
        left_pair = np.array([x0 % NX, (x0 + 1) % NX], dtype=int)
        right_pair = np.array([(x1 - 1) % NX, x1 % NX], dtype=int)

        if NY in NY_LOAD_LIST:
            G_final = load_final_from_cache(DATA_PATH_BY_NY[NY])
        else:
            res = model.run_markov_circuit(
                G_history=False,
                cycles=CYCLES,
                progress=True,
                init_mode=INIT_MODE,
                save=True,
                samples=SAMPLES,
                p_meas=P_MEAS,
                n_jobs=cpu_cap,
                parallelize_samples=True,
                sequence=SEQUENCE,
                top_triv_back_forth=TOP_TRIV_BACK_FORTH,
                max_in_flight=cpu_cap,
            )

            G_final = res.get("G_final")
            if G_final is None:
                raise RuntimeError("G_final not available; set G_history=False to keep only final state.")

        S_tot, Nlayer, Nlayer2 = G_final.shape
        if Nlayer != Nlayer2:
            raise ValueError(f"G_final last dims must be square; got {Nlayer} x {Nlayer2}")
        if 2 * NX * NY != Nlayer:
            raise ValueError(f"2*NX*NY={2*NX*NY} != Nlayer={Nlayer}")

        sample_idx = np.arange(S_tot, dtype=int)

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

        with tqdm(total=4, desc="fit curves", leave=False) as fit_bar:
            left_curve_1 = mean_pair_curve_for_ycuts(
                G_final, y_cut_list_1, sample_idx, sub_idx_map, ny_sub_map, NX, left_pair
            )
            fit_bar.update(1)
            right_curve_1 = mean_pair_curve_for_ycuts(
                G_final, y_cut_list_1, sample_idx, sub_idx_map, ny_sub_map, NX, right_pair
            )
            fit_bar.update(1)
            left_curve_2 = mean_pair_curve_for_ycuts(
                G_final, y_cut_list_2, sample_idx, sub_idx_map, ny_sub_map, NX, left_pair
            )
            fit_bar.update(1)
            right_curve_2 = mean_pair_curve_for_ycuts(
                G_final, y_cut_list_2, sample_idx, sub_idx_map, ny_sub_map, NX, right_pair
            )
            fit_bar.update(1)

        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_1, left_curve_1, NY)
        slope[cfg_names[0]][i_ny], slope_err[cfg_names[0]][i_ny], r2[cfg_names[0]][i_ny] = m, e, q
        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_1, right_curve_1, NY)
        slope[cfg_names[1]][i_ny], slope_err[cfg_names[1]][i_ny], r2[cfg_names[1]][i_ny] = m, e, q
        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_2, left_curve_2, NY)
        slope[cfg_names[2]][i_ny], slope_err[cfg_names[2]][i_ny], r2[cfg_names[2]][i_ny] = m, e, q
        m, e, q, _ = fit_line_log_chord_with_error(Ay_list_2, right_curve_2, NY)
        slope[cfg_names[3]][i_ny], slope_err[cfg_names[3]][i_ny], r2[cfg_names[3]][i_ny] = m, e, q

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 9))
    for name in cfg_names:
        ax0.errorbar(
            NY_LIST,
            slope[name],
            yerr=slope_err[name],
            marker="o",
            lw=1.4,
            capsize=3,
            label=name,
        )
    ax0.set_xlabel("Ny")
    ax0.set_ylabel("Slope m +/- fit stderr")
    ax0.set_title("Slope from fit to sample-averaged integrated contour (final state)")
    ax0.grid(alpha=0.3)
    ax0.legend(ncol=2, fontsize=9)

    for name in cfg_names:
        ax1.plot(NY_LIST, 1.0 - r2[name], marker="o", lw=1.4, label=name)
    ax1.set_xlabel("Ny")
    ax1.set_ylabel("R^2 error (1 - R^2)")
    ax1.set_title("Fit error vs system size (final state)")
    ax1.grid(alpha=0.3)
    ax1.legend(ncol=2, fontsize=9)

    figs_dir = os.path.join("figs")
    os.makedirs(figs_dir, exist_ok=True)
    nshell_tag = "None" if NSHELL is None else str(NSHELL)
    ny_max = int(max(NY_LIST))
    pdf_name = (
        "markov_circuit_boundary_EC_slope_fit_size_sweep_"
        f"Nyleq{ny_max}_integrated_contour_fit_C{int(CYCLES)}_S{int(SAMPLES)}"
        f"_nsh{nshell_tag}.pdf"
    )
    pdf_path = os.path.join(figs_dir, pdf_name)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=200)
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"Saved: {pdf_path}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
