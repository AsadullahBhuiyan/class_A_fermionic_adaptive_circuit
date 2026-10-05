from __future__ import annotations

import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ------------------------- CPU / experiment controls -------------------------
# Override any of these from the shell, e.g.:
#   DWCONV_CPU_START=20 DWCONV_CPU_COUNT=20 python run_n20x20_dwtrunc_convergence_cpu.py
CPU_START = int(os.environ.get("DWCONV_CPU_START", "0"))
CPU_COUNT = int(os.environ.get("DWCONV_CPU_COUNT", "20"))
N_JOBS = int(os.environ.get("DWCONV_N_JOBS", str(CPU_COUNT)))

NX = int(os.environ.get("DWCONV_NX", "20"))
NY = int(os.environ.get("DWCONV_NY", "20"))
CYCLES = int(os.environ.get("DWCONV_CYCLES", "40"))
SAMPLES = int(os.environ.get("DWCONV_SAMPLES", "10"))
NSHELL = int(os.environ.get("DWCONV_NSHELL", "1"))

ALPHA_1 = float(os.environ.get("DWCONV_ALPHA_1", "1"))
ALPHA_2 = float(os.environ.get("DWCONV_ALPHA_2", "30"))
TRIAL_ORBITALS = os.environ.get("DWCONV_TRIAL_ORBITALS", "X")
SEQUENCE = os.environ.get("DWCONV_SEQUENCE", "raster_y")

INIT_MODE = "maxmix"
POSTSELECT = False
PERFECT_CORRECTION = True
N_A = float(os.environ.get("DWCONV_N_A", "0.5"))
MEAS_SLAB_ONLY = os.environ.get("DWCONV_MEAS_SLAB_ONLY", "1") not in {"0", "false", "False"}
SAVE_RAW_HISTORY = os.environ.get("DWCONV_SAVE_RAW_HISTORY", "1") not in {"0", "false", "False"}

CANONICAL_DYNAMICS_ENTRY_POINT = "classA_U1FGTN.run_markov_circuit"


def parse_dw_truncation_values(raw: str) -> tuple[bool, ...]:
    values = []
    for token in raw.split(","):
        normalized = token.strip().lower()
        if normalized in {"0", "false"}:
            values.append(False)
        elif normalized in {"1", "true"}:
            values.append(True)
        elif normalized:
            raise ValueError(f"Invalid DWCONV_DW_TRUNCATION_VALUES token: {token!r}")
    if not values:
        raise ValueError("DWCONV_DW_TRUNCATION_VALUES must select at least one condition.")
    return tuple(dict.fromkeys(values))


DW_TRUNCATION_VALUES = parse_dw_truncation_values(
    os.environ.get("DWCONV_DW_TRUNCATION_VALUES", "0,1")
)
HERM_TOL = float(os.environ.get("DWCONV_HERM_TOL", "1e-8"))
INIT_MAXMIX_TOL = float(os.environ.get("DWCONV_INIT_MAXMIX_TOL", "1e-8"))
ENTROPY_EPS = float(os.environ.get("DWCONV_ENTROPY_EPS", "1e-12"))

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
SRC = ROOT / "src"
OUTPUT_DIR = Path(os.environ.get("DWCONV_OUTPUT_DIR", str(SCRIPT_DIR))).resolve()
FIG_DIR = OUTPUT_DIR / "figures"

os.environ["MY_CPU_COUNT"] = str(CPU_COUNT)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".tmp" / "cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from fgtn.classA_U1FGTN import classA_U1FGTN


def configure_cpu() -> dict:
    requested = list(range(CPU_START, CPU_START + CPU_COUNT))
    affinity_error = None
    try:
        os.sched_setaffinity(0, set(requested))
    except Exception as exc:  # pragma: no cover - platform dependent
        affinity_error = str(exc)
        print(f"[warn] CPU affinity not set: {exc}", flush=True)

    actual_affinity = None
    try:
        actual_affinity = sorted(os.sched_getaffinity(0))
        print(f"[info] CPU affinity: {actual_affinity}", flush=True)
    except Exception as exc:  # pragma: no cover - platform dependent
        affinity_error = str(exc) if affinity_error is None else affinity_error
        print(f"[warn] Could not query CPU affinity: {exc}", flush=True)

    return {
        "cpu_start": CPU_START,
        "cpu_count": CPU_COUNT,
        "n_jobs": N_JOBS,
        "requested_affinity": requested,
        "actual_affinity": actual_affinity,
        "affinity_error": affinity_error,
    }


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp_path.replace(path)


def write_csv_atomic(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp_path.replace(path)


def git_metadata() -> dict:
    def _run(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                args,
                cwd=ROOT,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "branch", "--show-current"]),
        "status_short": _run(["git", "status", "--short"]),
    }


def sem(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def gaussian_entropy_from_covariance(G: np.ndarray) -> tuple[float, float, float, float]:
    G = np.asarray(G, dtype=np.complex128)
    herm_err = float(np.max(np.abs(G - G.conj().T))) if G.size else 0.0
    G_herm = 0.5 * (G + G.conj().T)
    occ = 0.5 * (np.eye(G_herm.shape[0], dtype=np.complex128) + G_herm)
    evals = np.linalg.eigvalsh(occ)
    evals_real = np.real_if_close(evals).astype(np.float64, copy=False)
    eval_min = float(np.min(evals_real))
    eval_max = float(np.max(evals_real))
    clipped = np.clip(evals_real, ENTROPY_EPS, 1.0 - ENTROPY_EPS)
    entropy_nats = -float(np.sum(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)))
    entropy_bits = entropy_nats / math.log(2.0)
    return entropy_nats, entropy_bits, herm_err, max(abs(eval_min), abs(eval_max - 1.0))


def fit_log_slope(x_values: np.ndarray, y_values: np.ndarray) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
    if int(np.sum(mask)) < 3:
        return float("nan")
    coeff = np.polyfit(x[mask], np.log(y[mask]), deg=1)
    return float(coeff[0])


def threshold_crossing(cycles: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    for cycle, value in zip(cycles, values):
        if np.isfinite(value) and value <= threshold:
            return int(cycle)
    return None


def preferred_font_family() -> str:
    for name in ("CMU Sans Serif", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "sans-serif"


def run_condition(dw_truncation: bool) -> dict:
    label = f"dwtrunc{int(dw_truncation)}"
    if dw_truncation and not MEAS_SLAB_ONLY:
        raise ValueError(
            "dw_truncation=True requires meas_slab_only=True for the intended stable "
            "domain-wall-truncated dynamics."
        )
    print(f"[run] {label}: N={NX}x{NY}, cycles={CYCLES}, samples={SAMPLES}", flush=True)
    t0 = time.time()

    model = classA_U1FGTN(
        NX,
        NY,
        DW=True,
        nshell=NSHELL,
        alpha_1=ALPHA_1,
        alpha_2=ALPHA_2,
        trial_orbitals=TRIAL_ORBITALS,
        dw_truncation=dw_truncation,
    )
    active_indices = model.active_top_layer_indices(meas_slab_only=MEAS_SLAB_ONLY)
    result = model.run_markov_circuit(
        G_history=True,
        progress=True,
        cycles=CYCLES,
        samples=SAMPLES,
        n_jobs=N_JOBS,
        backend="loky",
        parallelize_samples=True,
        init_mode=INIT_MODE,
        save=SAVE_RAW_HISTORY,
        save_init=True,
        save_history_stride=None,
        save_suffix="_n20x20_dwtrunc_convergence_cpu",
        n_a=N_A,
        sequence=SEQUENCE,
        meas_slab_only=MEAS_SLAB_ONLY,
        postselect=POSTSELECT,
        perfect_correction=PERFECT_CORRECTION,
    )

    G_hist = result.get("G_hist")
    if G_hist is None:
        save_path = result.get("save_path")
        if not save_path:
            raise RuntimeError(f"{label}: run_markov_circuit did not return G_hist or save_path.")
        with np.load(save_path, allow_pickle=False) as data:
            G_hist = np.asarray(data["G_hist"], dtype=np.complex128)

    G_hist = np.asarray(G_hist, dtype=np.complex128)
    expected_shape = (SAMPLES, CYCLES + 1, 2 * NX * NY, 2 * NX * NY)
    if tuple(G_hist.shape) != expected_shape:
        raise ValueError(f"{label}: expected G_hist shape {expected_shape}, got {tuple(G_hist.shape)}")

    init_active = G_hist[:, 0][:, active_indices][:, :, active_indices]
    init_active_max_abs = float(np.max(np.abs(init_active)))
    init_global_max_abs = float(np.max(np.abs(G_hist[:, 0])))
    if init_active_max_abs > INIT_MAXMIX_TOL:
        raise ValueError(
            f"{label}: expected active-region maxmix cycle 0, "
            f"max |G_active|={init_active_max_abs:.3e}"
        )

    per_sample_rows: list[dict] = []
    frob_rows: list[dict] = []
    ensemble_rows: list[dict] = []

    entropy_nats = np.full((SAMPLES, CYCLES + 1), np.nan, dtype=np.float64)
    entropy_bits = np.full_like(entropy_nats, np.nan)
    herm_errors = np.full_like(entropy_nats, np.nan)
    occupation_clip_errors = np.full_like(entropy_nats, np.nan)

    for sample_index in range(SAMPLES):
        for cycle in range(CYCLES + 1):
            ent_nats, ent_bits, herm_err, clip_err = gaussian_entropy_from_covariance(
                G_hist[sample_index, cycle]
            )
            entropy_nats[sample_index, cycle] = ent_nats
            entropy_bits[sample_index, cycle] = ent_bits
            herm_errors[sample_index, cycle] = herm_err
            occupation_clip_errors[sample_index, cycle] = clip_err
            per_sample_rows.append(
                {
                    "config_id": label,
                    "dw_truncation": int(dw_truncation),
                    "Nx": NX,
                    "Ny": NY,
                    "nshell": NSHELL,
                    "protocol": "perfect_correction",
                    "sample_index": sample_index,
                    "cycle": cycle,
                    "global_entropy_nats": ent_nats,
                    "global_entropy_bits": ent_bits,
                    "hermitian_max_err": herm_err,
                    "occupation_clip_error": clip_err,
                }
            )

        for cycle in range(1, CYCLES + 1):
            delta = G_hist[sample_index, cycle] - G_hist[sample_index, cycle - 1]
            frob_rows.append(
                {
                    "config_id": label,
                    "dw_truncation": int(dw_truncation),
                    "Nx": NX,
                    "Ny": NY,
                    "nshell": NSHELL,
                    "protocol": "perfect_correction",
                    "sample_index": sample_index,
                    "cycle": cycle,
                    "cycle_prev": cycle - 1,
                    "frob_successive_delta": float(np.linalg.norm(delta, ord="fro")),
                }
            )

    frob_by_cycle = {
        cycle: np.asarray(
            [row["frob_successive_delta"] for row in frob_rows if int(row["cycle"]) == cycle],
            dtype=np.float64,
        )
        for cycle in range(1, CYCLES + 1)
    }

    mean_G_prev = None
    for cycle in range(CYCLES + 1):
        mean_G = np.mean(G_hist[:, cycle], axis=0)
        ensemble_frob = float("nan")
        if mean_G_prev is not None:
            ensemble_frob = float(np.linalg.norm(mean_G - mean_G_prev, ord="fro"))
        mean_G_prev = mean_G

        frob_vals = frob_by_cycle.get(cycle, np.asarray([], dtype=np.float64))
        ensemble_rows.append(
            {
                "config_id": label,
                "dw_truncation": int(dw_truncation),
                "Nx": NX,
                "Ny": NY,
                "nshell": NSHELL,
                "protocol": "perfect_correction",
                "cycle": cycle,
                "global_entropy_nats_mean": float(np.mean(entropy_nats[:, cycle])),
                "global_entropy_nats_std": float(np.std(entropy_nats[:, cycle], ddof=1)) if SAMPLES > 1 else 0.0,
                "global_entropy_nats_sem": sem(entropy_nats[:, cycle]),
                "global_entropy_bits_mean": float(np.mean(entropy_bits[:, cycle])),
                "global_entropy_bits_std": float(np.std(entropy_bits[:, cycle], ddof=1)) if SAMPLES > 1 else 0.0,
                "global_entropy_bits_sem": sem(entropy_bits[:, cycle]),
                "frob_successive_delta_mean": float(np.mean(frob_vals)) if frob_vals.size else float("nan"),
                "frob_successive_delta_std": float(np.std(frob_vals, ddof=1)) if frob_vals.size > 1 else 0.0,
                "frob_successive_delta_sem": sem(frob_vals) if frob_vals.size else float("nan"),
                "ensemble_mean_frob_successive_delta": ensemble_frob,
                "sample_count": SAMPLES,
            }
        )

    max_herm_err = float(np.nanmax(herm_errors))
    if max_herm_err > HERM_TOL:
        raise ValueError(f"{label}: Hermiticity error {max_herm_err:.3e} exceeds {HERM_TOL:.3e}")

    save_path = result.get("save_path")
    elapsed_sec = time.time() - t0
    print(f"[done] {label}: elapsed={elapsed_sec:.2f}s, raw_history={save_path}", flush=True)

    del G_hist
    gc.collect()

    return {
        "label": label,
        "dw_truncation": bool(dw_truncation),
        "save_path": save_path,
        "elapsed_sec": elapsed_sec,
        "history_shape": list(expected_shape),
        "active_top_layer_size": int(active_indices.size),
        "init_active_max_abs": init_active_max_abs,
        "init_global_max_abs": init_global_max_abs,
        "max_hermitian_error": max_herm_err,
        "max_occupation_clip_error": float(np.nanmax(occupation_clip_errors)),
        "per_sample_rows": per_sample_rows,
        "frob_rows": frob_rows,
        "ensemble_rows": ensemble_rows,
    }


def build_comparison_summary(ensemble_rows: list[dict]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    verdict: dict[str, str | None] = {}
    by_label = sorted({str(row["config_id"]) for row in ensemble_rows})
    cycles = np.arange(CYCLES + 1, dtype=np.int64)
    delta_cycles = np.arange(1, CYCLES + 1, dtype=np.int64)

    for label in by_label:
        sub = [row for row in ensemble_rows if str(row["config_id"]) == label]
        by_cycle = {int(row["cycle"]): row for row in sub}
        ent = np.asarray([by_cycle[c]["global_entropy_nats_mean"] for c in range(CYCLES + 1)], dtype=np.float64)
        frob = np.asarray(
            [by_cycle[c]["frob_successive_delta_mean"] for c in range(1, CYCLES + 1)],
            dtype=np.float64,
        )
        late_start = max(1, CYCLES - 9)
        late_delta_mask = delta_cycles >= late_start
        late_entropy_mask = cycles >= late_start
        initial_entropy = float(ent[0])
        thresholds = {
            "entropy_first_cycle_le_10pct_initial": threshold_crossing(cycles, ent, 0.10 * initial_entropy),
            "entropy_first_cycle_le_5pct_initial": threshold_crossing(cycles, ent, 0.05 * initial_entropy),
            "entropy_first_cycle_le_1pct_initial": threshold_crossing(cycles, ent, 0.01 * initial_entropy),
            "frob_first_cycle_le_10pct_cycle1": threshold_crossing(delta_cycles, frob, 0.10 * float(frob[0])),
            "frob_first_cycle_le_5pct_cycle1": threshold_crossing(delta_cycles, frob, 0.05 * float(frob[0])),
            "frob_first_cycle_le_1pct_cycle1": threshold_crossing(delta_cycles, frob, 0.01 * float(frob[0])),
        }
        rows.append(
            {
                "config_id": label,
                "dw_truncation": int(label.endswith("1")),
                "frob_auc_cycles_1_40": float(np.trapz(frob, x=delta_cycles)),
                "frob_late_mean_cycles_31_40": float(np.mean(frob[late_delta_mask])),
                "frob_log_decay_slope": fit_log_slope(delta_cycles, frob),
                "entropy_auc_cycles_0_40": float(np.trapz(ent, x=cycles)),
                "entropy_late_mean_cycles_31_40": float(np.mean(ent[late_entropy_mask])),
                "entropy_log_decay_slope": fit_log_slope(cycles, ent),
                **thresholds,
            }
        )

    if len(rows) == 2:
        verdict["frob_late_mean_winner"] = min(rows, key=lambda row: row["frob_late_mean_cycles_31_40"])[
            "config_id"
        ]
        verdict["frob_auc_winner"] = min(rows, key=lambda row: row["frob_auc_cycles_1_40"])["config_id"]
        verdict["entropy_late_mean_winner"] = min(
            rows, key=lambda row: row["entropy_late_mean_cycles_31_40"]
        )["config_id"]
        verdict["entropy_auc_winner"] = min(rows, key=lambda row: row["entropy_auc_cycles_0_40"])[
            "config_id"
        ]
    return rows, verdict


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def create_figures_from_csv(output_dir: Path = OUTPUT_DIR) -> None:
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv_rows(output_dir / "ensemble_cycle_curves.csv")
    labels = sorted({row["config_id"] for row in rows})
    colors = {"dwtrunc0": "#4C72B0", "dwtrunc1": "#DD8452"}

    plt.rcParams.update(
        {
            "font.family": preferred_font_family(),
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    def _series(label: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        sub = sorted((row for row in rows if row["config_id"] == label), key=lambda row: int(row["cycle"]))
        x = np.asarray([int(row["cycle"]) for row in sub], dtype=np.int64)
        y = np.asarray([float(row[key]) if row[key] else np.nan for row in sub], dtype=np.float64)
        return x, y

    fig, ax = plt.subplots(figsize=(3.375, 2.45), dpi=300)
    for label in labels:
        x, mean = _series(label, "frob_successive_delta_mean")
        _, err = _series(label, "frob_successive_delta_sem")
        mask = x >= 1
        ax.plot(x[mask], mean[mask], marker="o", ms=2.2, lw=1.0, color=colors.get(label), label=label)
        ax.fill_between(x[mask], mean[mask] - err[mask], mean[mask] + err[mask], color=colors.get(label), alpha=0.18)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$\|G_c-G_{c-1}\|_F$")
    ax.set_title("Successive covariance differences")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "frob_successive_delta_vs_cycle.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.375, 2.45), dpi=300)
    for label in labels:
        x, mean = _series(label, "global_entropy_bits_mean")
        _, err = _series(label, "global_entropy_bits_sem")
        ax.plot(x, mean, marker="o", ms=2.2, lw=1.0, color=colors.get(label), label=label)
        ax.fill_between(x, mean - err, mean + err, color=colors.get(label), alpha=0.18)
    ax.set_xlabel("cycle")
    ax.set_ylabel(r"$S / \log 2$")
    ax.set_title("Global entropy")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "global_entropy_vs_cycle.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.45), dpi=300)
    for label in labels:
        x, frob = _series(label, "frob_successive_delta_mean")
        _, frob_err = _series(label, "frob_successive_delta_sem")
        mask = x >= 1
        axes[0].plot(x[mask], frob[mask], marker="o", ms=2.0, lw=1.0, color=colors.get(label), label=label)
        axes[0].fill_between(
            x[mask], frob[mask] - frob_err[mask], frob[mask] + frob_err[mask], color=colors.get(label), alpha=0.18
        )
        x, ent = _series(label, "global_entropy_bits_mean")
        _, ent_err = _series(label, "global_entropy_bits_sem")
        axes[1].plot(x, ent, marker="o", ms=2.0, lw=1.0, color=colors.get(label), label=label)
        axes[1].fill_between(x, ent - ent_err, ent + ent_err, color=colors.get(label), alpha=0.18)
    axes[0].set_xlabel("cycle")
    axes[0].set_ylabel(r"$\|G_c-G_{c-1}\|_F$")
    axes[0].set_title("Frobenius convergence")
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel(r"$S / \log 2$")
    axes[1].set_title("Entropy convergence")
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "convergence_comparison_panel.png", dpi=300)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cpu_info = configure_cpu()

    manifest = {
        "analysis_name": "n20x20_maxmix_dwtrunc_convergence_cpu",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_dynamics_entry_point": CANONICAL_DYNAMICS_ENTRY_POINT,
        "root": str(ROOT),
        "output_dir": str(OUTPUT_DIR),
        "cpu": cpu_info,
        "git": git_metadata(),
        "config": {
            "Nx": NX,
            "Ny": NY,
            "cycles": CYCLES,
            "samples": SAMPLES,
            "nshell": NSHELL,
            "DW": True,
            "alpha_1": ALPHA_1,
            "alpha_2": ALPHA_2,
            "trial_orbitals": TRIAL_ORBITALS,
            "dw_truncation_values": list(DW_TRUNCATION_VALUES),
            "init_mode": INIT_MODE,
            "postselect": POSTSELECT,
            "perfect_correction": PERFECT_CORRECTION,
            "n_a": N_A,
            "sequence": SEQUENCE,
            "meas_slab_only": MEAS_SLAB_ONLY,
            "save_raw_history": SAVE_RAW_HISTORY,
        },
        "notes": [
            "dw_truncation=True is guarded to require meas_slab_only=True.",
            "With dw_truncation=True and perfect correction, the active DW slab starts maximally mixed while the exterior is prepared as a deterministic onsite product state.",
            "Primary Frobenius curves are samplewise ||G_s(c)-G_s(c-1)||_F averaged over samples.",
            "Global entropy is the Gaussian entropy of the full top-layer covariance matrix.",
        ],
        "runs": [],
    }
    write_json_atomic(OUTPUT_DIR / "analysis_manifest.json", manifest)

    per_sample_rows: list[dict] = []
    frob_rows: list[dict] = []
    ensemble_rows: list[dict] = []
    for dw_truncation in DW_TRUNCATION_VALUES:
        run_payload = run_condition(dw_truncation)
        per_sample_rows.extend(run_payload.pop("per_sample_rows"))
        frob_rows.extend(run_payload.pop("frob_rows"))
        ensemble_rows.extend(run_payload.pop("ensemble_rows"))
        manifest["runs"].append(run_payload)
        write_json_atomic(OUTPUT_DIR / "analysis_manifest.json", manifest)

    per_sample_fields = [
        "config_id",
        "dw_truncation",
        "Nx",
        "Ny",
        "nshell",
        "protocol",
        "sample_index",
        "cycle",
        "global_entropy_nats",
        "global_entropy_bits",
        "hermitian_max_err",
        "occupation_clip_error",
    ]
    frob_fields = [
        "config_id",
        "dw_truncation",
        "Nx",
        "Ny",
        "nshell",
        "protocol",
        "sample_index",
        "cycle",
        "cycle_prev",
        "frob_successive_delta",
    ]
    ensemble_fields = [
        "config_id",
        "dw_truncation",
        "Nx",
        "Ny",
        "nshell",
        "protocol",
        "cycle",
        "global_entropy_nats_mean",
        "global_entropy_nats_std",
        "global_entropy_nats_sem",
        "global_entropy_bits_mean",
        "global_entropy_bits_std",
        "global_entropy_bits_sem",
        "frob_successive_delta_mean",
        "frob_successive_delta_std",
        "frob_successive_delta_sem",
        "ensemble_mean_frob_successive_delta",
        "sample_count",
    ]

    write_csv_atomic(OUTPUT_DIR / "per_sample_cycle_metrics.csv", per_sample_rows, per_sample_fields)
    write_csv_atomic(OUTPUT_DIR / "frob_successive_deltas.csv", frob_rows, frob_fields)
    write_csv_atomic(OUTPUT_DIR / "ensemble_cycle_curves.csv", ensemble_rows, ensemble_fields)

    comparison_rows, verdict = build_comparison_summary(ensemble_rows)
    comparison_fields = [
        "config_id",
        "dw_truncation",
        "frob_auc_cycles_1_40",
        "frob_late_mean_cycles_31_40",
        "frob_log_decay_slope",
        "entropy_auc_cycles_0_40",
        "entropy_late_mean_cycles_31_40",
        "entropy_log_decay_slope",
        "entropy_first_cycle_le_10pct_initial",
        "entropy_first_cycle_le_5pct_initial",
        "entropy_first_cycle_le_1pct_initial",
        "frob_first_cycle_le_10pct_cycle1",
        "frob_first_cycle_le_5pct_cycle1",
        "frob_first_cycle_le_1pct_cycle1",
    ]
    write_csv_atomic(OUTPUT_DIR / "comparison_summary.csv", comparison_rows, comparison_fields)

    create_figures_from_csv(OUTPUT_DIR)

    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["elapsed_sec"] = time.time() - t0
    manifest["row_counts"] = {
        "per_sample_cycle_metrics": len(per_sample_rows),
        "frob_successive_deltas": len(frob_rows),
        "ensemble_cycle_curves": len(ensemble_rows),
        "comparison_summary": len(comparison_rows),
    }
    manifest["validation"] = {
        "expected_per_sample_cycle_metrics": len(DW_TRUNCATION_VALUES) * SAMPLES * (CYCLES + 1),
        "expected_frob_successive_deltas": len(DW_TRUNCATION_VALUES) * SAMPLES * CYCLES,
        "expected_history_shape_per_run": [SAMPLES, CYCLES + 1, 2 * NX * NY, 2 * NX * NY],
        "init_maxmix_tol": INIT_MAXMIX_TOL,
        "herm_tol": HERM_TOL,
    }
    manifest["verdict"] = verdict
    write_json_atomic(OUTPUT_DIR / "analysis_manifest.json", manifest)

    print("[done] wrote outputs:", flush=True)
    for name in [
        "per_sample_cycle_metrics.csv",
        "frob_successive_deltas.csv",
        "ensemble_cycle_curves.csv",
        "comparison_summary.csv",
        "analysis_manifest.json",
        "figures/frob_successive_delta_vs_cycle.png",
        "figures/global_entropy_vs_cycle.png",
        "figures/convergence_comparison_panel.png",
    ]:
        print(f"  {OUTPUT_DIR / name}", flush=True)


if __name__ == "__main__":
    main()
