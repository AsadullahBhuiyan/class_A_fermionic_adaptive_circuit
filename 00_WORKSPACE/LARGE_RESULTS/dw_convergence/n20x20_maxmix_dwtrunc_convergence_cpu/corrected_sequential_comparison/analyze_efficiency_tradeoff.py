from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


BASE = Path(__file__).resolve().parent
SOURCE_ROOT = BASE.parent
FIG_DIR = BASE / "figures"

CONDITIONS = ("dwtrunc0", "dwtrunc1")
CONDITION_LABELS = {
    "dwtrunc0": "dw_truncation=False",
    "dwtrunc1": "dw_truncation=True",
}
CONDITION_DIRS = {
    "dwtrunc0": SOURCE_ROOT / "dwtrunc0_sequential_run",
    "dwtrunc1": SOURCE_ROOT / "dwtrunc1_sequential_run",
}
COLORS = {
    "dwtrunc0": "#4C72B0",
    "dwtrunc1": "#DD8452",
}

ABS_ENTROPY_THRESHOLDS = (100.0, 50.0, 20.0, 10.0, 5.0, 1.0, 0.5)
REL_ENTROPY_THRESHOLDS = (0.10, 0.05, 0.01, 0.005, 0.001)
FROB_THRESHOLDS = (20.0, 10.0, 8.0, 7.0)
EPS = 1e-12


def preferred_font_family() -> str:
    for name in ("CMU Sans Serif", "DejaVu Sans"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "sans-serif"


def load_manifest(condition: str) -> dict:
    return json.loads((CONDITION_DIRS[condition] / "analysis_manifest.json").read_text())


def load_inputs() -> tuple[pd.DataFrame, dict[str, dict]]:
    ensemble_path = BASE / "combined_ensemble_cycle_curves.csv"
    if not ensemble_path.exists():
        raise FileNotFoundError(ensemble_path)
    ensemble = pd.read_csv(ensemble_path)
    manifests = {condition: load_manifest(condition) for condition in CONDITIONS}
    return ensemble, manifests


def timing_table(manifests: dict[str, dict]) -> dict[str, dict[str, float]]:
    timing: dict[str, dict[str, float]] = {}
    for condition, manifest in manifests.items():
        cfg = manifest["config"]
        elapsed_sec = float(manifest["elapsed_sec"])
        cycles = int(cfg["cycles"])
        samples = int(cfg["samples"])
        timing[condition] = {
            "elapsed_sec": elapsed_sec,
            "cycles": float(cycles),
            "samples": float(samples),
            "intrinsic_sec_per_trajectory_cycle": elapsed_sec / (samples * cycles),
            "wall_sec_per_cycle": elapsed_sec / cycles,
        }
    return timing


def first_crossing(cycles: np.ndarray, values: np.ndarray, threshold: float) -> tuple[float, float, str]:
    cycles = np.asarray(cycles, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(cycles) & np.isfinite(values)
    cycles = cycles[mask]
    values = values[mask]
    if cycles.size == 0:
        return math.nan, math.nan, "no_finite_data"
    if values[0] <= threshold:
        return float(cycles[0]), float(values[0]), "initial_or_first_point"
    for idx in range(1, cycles.size):
        v_prev = float(values[idx - 1])
        v_cur = float(values[idx])
        if v_cur <= threshold:
            c_prev = float(cycles[idx - 1])
            c_cur = float(cycles[idx])
            if np.isclose(v_cur, v_prev):
                return c_cur, v_cur, "step_crossing"
            frac = (threshold - v_prev) / (v_cur - v_prev)
            frac = float(np.clip(frac, 0.0, 1.0))
            cycle_interp = c_prev + frac * (c_cur - c_prev)
            return cycle_interp, float(threshold), "linear_interpolation"
    return math.nan, math.nan, "not_reached"


def sustained_crossing(cycles: np.ndarray, values: np.ndarray, threshold: float) -> tuple[float, float, str]:
    cycles = np.asarray(cycles, dtype=float)
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(cycles) & np.isfinite(values)
    cycles = cycles[mask]
    values = values[mask]
    if cycles.size == 0:
        return math.nan, math.nan, "no_finite_data"
    for idx in range(cycles.size):
        if np.all(values[idx:] <= threshold):
            return float(cycles[idx]), float(values[idx]), "sustained_discrete_crossing"
    return math.nan, math.nan, "not_reached"


def build_entropy_rate_table(
    ensemble: pd.DataFrame,
    manifests: dict[str, dict],
    timing: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        sub = ensemble[ensemble["condition"] == condition].sort_values("cycle")
        cfg = manifests[condition]["config"]
        cycles = int(cfg["cycles"])
        s0 = float(sub[sub["cycle"] == 0]["global_entropy_bits_mean"].iloc[0])
        s_final = float(sub[sub["cycle"] == cycles]["global_entropy_bits_mean"].iloc[0])
        entropy_loss = s0 - s_final
        entropy_loss_per_cycle = entropy_loss / cycles
        wall_sec_per_cycle = timing[condition]["wall_sec_per_cycle"]
        intrinsic_sec_per_cycle = timing[condition]["intrinsic_sec_per_trajectory_cycle"]
        rows.append(
            {
                "condition": condition,
                "S0_bits": s0,
                "S_final_bits": s_final,
                "entropy_loss_bits": entropy_loss,
                "entropy_loss_bits_per_cycle": entropy_loss_per_cycle,
                "wall_sec_per_cycle": wall_sec_per_cycle,
                "intrinsic_sec_per_trajectory_cycle": intrinsic_sec_per_cycle,
                "bits_lost_per_wall_sec": entropy_loss_per_cycle / wall_sec_per_cycle,
                "wall_sec_per_bit_lost": wall_sec_per_cycle / entropy_loss_per_cycle,
                "bits_lost_per_intrinsic_sec": entropy_loss_per_cycle / intrinsic_sec_per_cycle,
                "intrinsic_sec_per_bit_lost": intrinsic_sec_per_cycle / entropy_loss_per_cycle,
            }
        )
    table = pd.DataFrame(rows)
    false = table[table["condition"] == "dwtrunc0"].iloc[0]
    true = table[table["condition"] == "dwtrunc1"].iloc[0]
    table["cycle_time_speedup_false_over_true"] = false["wall_sec_per_cycle"] / true["wall_sec_per_cycle"]
    table["entropy_loss_per_cycle_true_over_false"] = (
        true["entropy_loss_bits_per_cycle"] / false["entropy_loss_bits_per_cycle"]
    )
    table["bits_per_wall_sec_true_over_false"] = (
        true["bits_lost_per_wall_sec"] / false["bits_lost_per_wall_sec"]
    )
    table["wall_sec_per_bit_true_over_false"] = (
        true["wall_sec_per_bit_lost"] / false["wall_sec_per_bit_lost"]
    )
    return table


def build_sustained_entropy_table(
    ensemble: pd.DataFrame,
    timing: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for metric, thresholds, column in (
        ("absolute_entropy_bits", ABS_ENTROPY_THRESHOLDS, "global_entropy_bits_mean"),
        ("relative_entropy", REL_ENTROPY_THRESHOLDS, "relative_entropy_mean"),
    ):
        for threshold in thresholds:
            per_condition = {}
            for condition in CONDITIONS:
                sub = ensemble[ensemble["condition"] == condition].sort_values("cycle").copy()
                if metric == "relative_entropy":
                    init_entropy = float(
                        ensemble[
                            (ensemble["condition"] == condition)
                            & (ensemble["cycle"] == 0)
                        ]["global_entropy_bits_mean"].iloc[0]
                    )
                    sub["relative_entropy_mean"] = sub["global_entropy_bits_mean"] / init_entropy
                cycle_to_threshold, value_at_crossing, method = sustained_crossing(
                    sub["cycle"].to_numpy(dtype=float),
                    sub[column].to_numpy(dtype=float),
                    threshold,
                )
                wall_hr = (
                    cycle_to_threshold * timing[condition]["wall_sec_per_cycle"] / 3600.0
                    if np.isfinite(cycle_to_threshold)
                    else math.nan
                )
                per_condition[condition] = {
                    "cycle": cycle_to_threshold,
                    "wall_hr": wall_hr,
                    "method": method,
                    "value": value_at_crossing,
                }
            false_time = per_condition["dwtrunc0"]["wall_hr"]
            true_time = per_condition["dwtrunc1"]["wall_hr"]
            speedup = false_time / true_time if np.isfinite(false_time) and np.isfinite(true_time) and true_time > 0 else math.nan
            rows.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "dwtrunc0_sustained_cycle": per_condition["dwtrunc0"]["cycle"],
                    "dwtrunc1_sustained_cycle": per_condition["dwtrunc1"]["cycle"],
                    "dwtrunc0_sustained_wall_hr": false_time,
                    "dwtrunc1_sustained_wall_hr": true_time,
                    "speedup_false_over_true_wall": speedup,
                    "winner": (
                        "dwtrunc1"
                        if np.isfinite(speedup) and speedup > 1.0
                        else "dwtrunc0"
                        if np.isfinite(speedup) and speedup < 1.0
                        else "tie_or_not_comparable"
                    ),
                    "dwtrunc0_method": per_condition["dwtrunc0"]["method"],
                    "dwtrunc1_method": per_condition["dwtrunc1"]["method"],
                }
            )
    return pd.DataFrame(rows)


def build_threshold_rows(
    ensemble: pd.DataFrame,
    timing: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for metric, thresholds, column, start_cycle in (
        ("absolute_entropy_bits", ABS_ENTROPY_THRESHOLDS, "global_entropy_bits_mean", 0),
        ("relative_entropy", REL_ENTROPY_THRESHOLDS, "relative_entropy_mean", 0),
        ("frobenius_successive_delta", FROB_THRESHOLDS, "frob_successive_delta_mean", 1),
    ):
        for threshold in thresholds:
            per_condition = {}
            for condition in CONDITIONS:
                sub = ensemble[ensemble["condition"] == condition].sort_values("cycle").copy()
                if start_cycle > 0:
                    sub = sub[sub["cycle"] >= start_cycle].copy()
                if metric == "relative_entropy":
                    init_entropy = float(
                        ensemble[
                            (ensemble["condition"] == condition)
                            & (ensemble["cycle"] == 0)
                        ]["global_entropy_bits_mean"].iloc[0]
                    )
                    sub["relative_entropy_mean"] = sub["global_entropy_bits_mean"] / init_entropy
                cycle_to_threshold, value_at_crossing, method = first_crossing(
                    sub["cycle"].to_numpy(dtype=float),
                    sub[column].to_numpy(dtype=float),
                    threshold,
                )
                intrinsic_sec = (
                    cycle_to_threshold * timing[condition]["intrinsic_sec_per_trajectory_cycle"]
                    if np.isfinite(cycle_to_threshold)
                    else math.nan
                )
                wall_hr = (
                    cycle_to_threshold * timing[condition]["wall_sec_per_cycle"] / 3600.0
                    if np.isfinite(cycle_to_threshold)
                    else math.nan
                )
                per_condition[condition] = {
                    "cycle_to_threshold": cycle_to_threshold,
                    "value_at_crossing": value_at_crossing,
                    "method": method,
                    "intrinsic_sec_to_threshold": intrinsic_sec,
                    "wall_hr_to_threshold": wall_hr,
                }
            false_time = per_condition["dwtrunc0"]["wall_hr_to_threshold"]
            true_time = per_condition["dwtrunc1"]["wall_hr_to_threshold"]
            speedup = false_time / true_time if np.isfinite(false_time) and np.isfinite(true_time) and true_time > 0 else math.nan
            rows.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "dwtrunc0_cycle": per_condition["dwtrunc0"]["cycle_to_threshold"],
                    "dwtrunc1_cycle": per_condition["dwtrunc1"]["cycle_to_threshold"],
                    "dwtrunc0_wall_hr": per_condition["dwtrunc0"]["wall_hr_to_threshold"],
                    "dwtrunc1_wall_hr": per_condition["dwtrunc1"]["wall_hr_to_threshold"],
                    "dwtrunc0_intrinsic_sec": per_condition["dwtrunc0"]["intrinsic_sec_to_threshold"],
                    "dwtrunc1_intrinsic_sec": per_condition["dwtrunc1"]["intrinsic_sec_to_threshold"],
                    "speedup_false_over_true_wall": speedup,
                    "winner": (
                        "dwtrunc1"
                        if np.isfinite(speedup) and speedup > 1.0
                        else "dwtrunc0"
                        if np.isfinite(speedup) and speedup < 1.0
                        else "tie_or_not_comparable"
                    ),
                    "dwtrunc0_method": per_condition["dwtrunc0"]["method"],
                    "dwtrunc1_method": per_condition["dwtrunc1"]["method"],
                }
            )
    return pd.DataFrame(rows)


def add_time_columns(ensemble: pd.DataFrame, timing: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = ensemble.copy()
    out["relative_entropy_mean"] = np.nan
    out["relative_entropy_sem"] = np.nan
    out["wall_time_hr"] = np.nan
    out["intrinsic_trajectory_time_sec"] = np.nan
    for condition in CONDITIONS:
        idx = out["condition"] == condition
        init_entropy = float(out[idx & (out["cycle"] == 0)]["global_entropy_bits_mean"].iloc[0])
        out.loc[idx, "relative_entropy_mean"] = out.loc[idx, "global_entropy_bits_mean"] / init_entropy
        out.loc[idx, "relative_entropy_sem"] = out.loc[idx, "global_entropy_bits_sem"] / init_entropy
        out.loc[idx, "wall_time_hr"] = (
            out.loc[idx, "cycle"] * timing[condition]["wall_sec_per_cycle"] / 3600.0
        )
        out.loc[idx, "intrinsic_trajectory_time_sec"] = (
            out.loc[idx, "cycle"] * timing[condition]["intrinsic_sec_per_trajectory_cycle"]
        )
    return out


def safe_yerr(mean: np.ndarray, sem: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    sem = np.asarray(sem, dtype=float)
    lower = np.maximum(mean - sem, EPS)
    upper = np.maximum(mean + sem, EPS)
    return np.vstack([mean - lower, upper - mean])


def plot_metric(
    timed: pd.DataFrame,
    *,
    filename: str,
    y_col: str,
    sem_col: str,
    ylabel: str,
    title: str,
    loglog: bool = False,
    start_cycle: int = 0,
) -> None:
    fig, ax = plt.subplots(figsize=(3.375, 2.45), dpi=300)
    for condition in CONDITIONS:
        sub = timed[(timed["condition"] == condition) & (timed["cycle"] >= start_cycle)].sort_values("cycle")
        x = sub["wall_time_hr"].to_numpy(dtype=float)
        y = np.maximum(sub[y_col].to_numpy(dtype=float), EPS if loglog else -np.inf)
        sem = sub[sem_col].to_numpy(dtype=float)
        if loglog:
            mask = (x > 0.0) & (y > 0.0)
            x = x[mask]
            y = y[mask]
            sem = sem[mask]
            yerr = safe_yerr(y, sem)
        else:
            yerr = sem
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            ms=2.1,
            lw=1.0,
            capsize=1.5,
            color=COLORS[condition],
            label=CONDITION_LABELS[condition],
        )
    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("wall-clock ensemble time (hr)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.5, which="both")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)


def plot_entropy_rate_bars(rate_table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.45), dpi=300)
    x = np.arange(len(CONDITIONS))
    labels = [CONDITION_LABELS[c] for c in CONDITIONS]
    colors = [COLORS[c] for c in CONDITIONS]
    ordered = rate_table.set_index("condition").loc[list(CONDITIONS)]

    axes[0].bar(x, ordered["entropy_loss_bits_per_cycle"], color=colors)
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("bits lost / cycle")
    axes[0].set_title("Entropy loss per cycle")
    axes[0].grid(alpha=0.25, axis="y", linewidth=0.5)

    axes[1].bar(x, ordered["bits_lost_per_wall_sec"], color=colors)
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("bits lost / wall sec")
    axes[1].set_title("Entropy throughput")
    axes[1].grid(alpha=0.25, axis="y", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "entropy_loss_rate_efficiency.png", dpi=300)
    plt.close(fig)


def plot_sustained_entropy_thresholds(sustained_table: pd.DataFrame) -> None:
    for metric, filename, xlabel in (
        ("absolute_entropy_bits", "sustained_absolute_entropy_threshold_walltime.png", "absolute entropy threshold (bits)"),
        ("relative_entropy", "sustained_relative_entropy_threshold_walltime.png", "relative entropy threshold"),
    ):
        sub = sustained_table[sustained_table["metric"] == metric].copy()
        fig, ax = plt.subplots(figsize=(3.375, 2.45), dpi=300)
        for condition in CONDITIONS:
            x = sub["threshold"].to_numpy(dtype=float)
            y = sub[f"{condition}_sustained_wall_hr"].to_numpy(dtype=float)
            mask = np.isfinite(y)
            ax.plot(
                x[mask],
                y[mask],
                marker="o",
                ms=2.2,
                lw=1.0,
                color=COLORS[condition],
                label=CONDITION_LABELS[condition],
            )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel("wall-clock ensemble time (hr)")
        ax.set_title("Sustained threshold crossing")
        ax.grid(alpha=0.25, linewidth=0.5, which="both")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG_DIR / filename, dpi=300)
        plt.close(fig)


def write_summary(
    table: pd.DataFrame,
    timing: dict[str, dict[str, float]],
    rate_table: pd.DataFrame,
    sustained_table: pd.DataFrame,
) -> None:
    lines = [
        "# Efficiency Tradeoff Summary",
        "",
        "Corrected sequential-site CPU runs only. Times are measured from completed run manifests.",
        "",
        "## Mean Cycle Cost",
        "",
        "| condition | intrinsic sec / trajectory-cycle | wall sec / ensemble-cycle | total runtime hr |",
        "|---|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        lines.append(
            f"| {condition} | "
            f"{timing[condition]['intrinsic_sec_per_trajectory_cycle']:.6g} | "
            f"{timing[condition]['wall_sec_per_cycle']:.6g} | "
            f"{timing[condition]['elapsed_sec'] / 3600.0:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Direct Entropy-Loss Efficiency",
            "",
            rate_table.to_markdown(index=False, floatfmt=".6g"),
            "",
            "Direct-rate interpretation: `dwtrunc1` is cheaper per cycle, but its entropy loss per cycle is smaller. The `bits_lost_per_wall_sec` column is the net entropy-throughput measure.",
            "",
            "## Threshold Crossings",
            "",
            table.to_markdown(index=False, floatfmt=".6g"),
            "",
            "## Sustained Entropy Threshold Crossings",
            "",
            sustained_table.to_markdown(index=False, floatfmt=".6g"),
            "",
            "Interpretation: speedup is `dwtrunc0_wall_hr / dwtrunc1_wall_hr`; values above 1 mean `dw_truncation=True` reaches that threshold faster in wall-clock ensemble time.",
            "Threshold times are first crossings of the ensemble-mean curve, linearly interpolated between saved cycles. A later increase above the same threshold does not change the reported first-crossing time.",
        ]
    )
    (BASE / "efficiency_tradeoff_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
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
    ensemble, manifests = load_inputs()
    timing = timing_table(manifests)
    timed = add_time_columns(ensemble, timing)
    timed.to_csv(BASE / "efficiency_time_curves.csv", index=False)
    rate_table = build_entropy_rate_table(ensemble, manifests, timing)
    rate_table.to_csv(BASE / "entropy_loss_rate_efficiency_table.csv", index=False)
    table = build_threshold_rows(ensemble, timing)
    table.to_csv(BASE / "efficiency_tradeoff_table.csv", index=False)
    sustained_table = build_sustained_entropy_table(ensemble, timing)
    sustained_table.to_csv(BASE / "sustained_entropy_threshold_table.csv", index=False)
    write_summary(table, timing, rate_table, sustained_table)

    plot_metric(
        timed,
        filename="entropy_residual_vs_walltime.png",
        y_col="global_entropy_bits_mean",
        sem_col="global_entropy_bits_sem",
        ylabel=r"$S/\log 2$",
        title="Absolute entropy residual",
    )
    plot_metric(
        timed,
        filename="relative_entropy_residual_vs_walltime.png",
        y_col="relative_entropy_mean",
        sem_col="relative_entropy_sem",
        ylabel=r"$S(c)/S(0)$",
        title="Relative entropy residual",
    )
    plot_metric(
        timed,
        filename="frobenius_residual_vs_walltime.png",
        y_col="frob_successive_delta_mean",
        sem_col="frob_successive_delta_sem",
        ylabel=r"$\langle\|G_c-G_{c-1}\|_F\rangle$",
        title="Frobenius convergence residual",
        start_cycle=1,
    )
    plot_metric(
        timed,
        filename="entropy_residual_vs_walltime_loglog.png",
        y_col="global_entropy_bits_mean",
        sem_col="global_entropy_bits_sem",
        ylabel=r"$S/\log 2$",
        title="Absolute entropy residual",
        loglog=True,
        start_cycle=1,
    )
    plot_metric(
        timed,
        filename="relative_entropy_residual_vs_walltime_loglog.png",
        y_col="relative_entropy_mean",
        sem_col="relative_entropy_sem",
        ylabel=r"$S(c)/S(0)$",
        title="Relative entropy residual",
        loglog=True,
        start_cycle=1,
    )
    plot_metric(
        timed,
        filename="frobenius_residual_vs_walltime_loglog.png",
        y_col="frob_successive_delta_mean",
        sem_col="frob_successive_delta_sem",
        ylabel=r"$\langle\|G_c-G_{c-1}\|_F\rangle$",
        title="Frobenius convergence residual",
        loglog=True,
        start_cycle=1,
    )
    plot_entropy_rate_bars(rate_table)
    plot_sustained_entropy_thresholds(sustained_table)
    print(table.to_markdown(index=False, floatfmt=".6g"))
    print("\nDirect entropy-loss efficiency")
    print(rate_table.to_markdown(index=False, floatfmt=".6g"))
    print("\nSustained entropy thresholds")
    print(sustained_table.to_markdown(index=False, floatfmt=".6g"))


if __name__ == "__main__":
    main()
