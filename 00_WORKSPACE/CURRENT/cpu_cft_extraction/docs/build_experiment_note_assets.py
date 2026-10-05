#!/usr/bin/env python3
"""Rebuild the numerical audit and figures used by the CPU CFT experiment note.

This is deliberately a read-only analysis of completed production artifacts.  The
live L=26,28,30 extension is recorded as excluded until its atomic campaign output
exists; no partial worker state is interpreted as data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "generated"
FIG = HERE / "figures"

RUNS = {
    "T2": [
        REPO / "cpu_cft_extraction/outputs/tmux_cft_20260806_000032/production/20260806_001959_88d7bf97e4"
    ],
    "T5": [
        REPO / "cpu_cft_extraction/outputs/tmux_L20_T100_20260807_183409/production/20260807_183409_6ea93098a1",
        REPO / "cpu_cft_extraction/outputs/tmux_L30L40_T5L_20260808_133322/production/20260808_133322_0188a5388e",
    ],
    "T20": [
        REPO / "cpu_cft_extraction/outputs/tmux_Nx20_L20_22_24_TNxNy_20260810_161330/production/20260810_161331_bcda51e8d2"
    ],
}

LIVE_EXTENSION = (
    REPO
    / "cpu_cft_extraction/outputs/tmux_Nx20_L26_28_30_TNxNy_20260812_195004"
    / "production/20260812_195043_62b43481c4"
)

RANK_SCAN = (
    RUNS["T2"][0] / "rank_scan/rank_fit_summary.csv"
)


def finite_or_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite_or_none(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_or_none(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_or_none(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def ols(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return {"n": int(x.size), "intercept": None, "slope": None,
                "intercept_se": None, "slope_se": None}
    design = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept_se = slope_se = None
    if x.size > 2:
        residual = y - design @ beta
        covariance = np.linalg.inv(design.T @ design) * (residual @ residual) / (x.size - 2)
        intercept_se, slope_se = np.sqrt(np.diag(covariance))
    return {
        "n": int(x.size),
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "intercept_se": None if intercept_se is None else float(intercept_se),
        "slope_se": None if slope_se is None else float(slope_se),
    }


def wls(x: np.ndarray, y: np.ndarray, sigma: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(sigma) & (sigma > 0)
    x, y, sigma = x[mask], y[mask], sigma[mask]
    if x.size < 2:
        return {"n": int(x.size), "intercept": None, "slope": None,
                "intercept_se": None, "slope_se": None, "chi2": None, "dof": None}
    design = np.column_stack([np.ones_like(x), x])
    weight = 1.0 / sigma**2
    covariance = np.linalg.inv(design.T @ (weight[:, None] * design))
    beta = covariance @ (design.T @ (weight * y))
    residual = (y - design @ beta) / sigma
    errors = np.sqrt(np.diag(covariance))
    return {
        "n": int(x.size), "intercept": float(beta[0]), "slope": float(beta[1]),
        "intercept_se": float(errors[0]), "slope_se": float(errors[1]),
        "chi2": float(residual @ residual), "dof": int(x.size - 2),
    }


def through_origin(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if not x.size:
        return {"n": 0, "slope": None}
    return {"n": int(x.size), "slope": float((x @ y) / (x @ x))}


def derived_fits(rows: pd.DataFrame) -> dict[str, Any]:
    inv_l2 = 1.0 / rows.L.to_numpy(float) ** 2
    f0 = rows.f0.to_numpy(float)
    f0_se = rows.f0_se.to_numpy(float)
    ffit = ols(inv_l2, f0)
    fwls = wls(inv_l2, f0, f0_se)
    for fit in (ffit, fwls):
        fit["c_eff"] = None if fit["slope"] is None else -6.0 * float(fit["slope"]) / np.pi
        fit["c_eff_se"] = None if fit["slope_se"] is None else 6.0 * float(fit["slope_se"]) / np.pi

    gaps: dict[str, Any] = {}
    for route, gap_col, se_col in (
        ("tangent", "tangent_fock_gap", "tangent_fock_gap_se"),
        ("choi", "choi_fock_gap", "choi_fock_gap_se"),
    ):
        gap = rows[gap_col].to_numpy(float)
        alpha = rows.alpha.to_numpy(float)
        scaled = gap / (alpha * rows.L.to_numpy(float))
        scaled_se = rows[se_col].to_numpy(float) / (alpha * rows.L.to_numpy(float))
        free = ols(inv_l2, scaled)
        weighted = wls(inv_l2, scaled, scaled_se)
        origin = through_origin(inv_l2, scaled)
        for fit in (free, weighted, origin):
            fit["x"] = None if fit["slope"] is None else float(fit["slope"]) / (2.0 * np.pi)
            if "slope_se" in fit:
                fit["x_se"] = None if fit["slope_se"] is None else float(fit["slope_se"]) / (2.0 * np.pi)
        gaps[route] = {"free_intercept": free, "weighted": weighted, "through_origin": origin}
    return {"f0": {"free_intercept": ffit, "weighted": fwls}, "gaps": gaps}


def load_rows() -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for label, roots in RUNS.items():
        pieces = []
        for root in roots:
            scalar_file = root / "scalars_by_size.csv"
            if not scalar_file.is_file():
                raise FileNotFoundError(f"completed campaign is missing {scalar_file}")
            pieces.append(pd.read_csv(scalar_file))
        rows = pd.concat(pieces, ignore_index=True).sort_values("L").reset_index(drop=True)
        expected_ratio = int(label[1:])
        actual_ratio = rows.cycles.to_numpy(float) / rows.L.to_numpy(float)
        if not np.allclose(actual_ratio, expected_ratio):
            raise ValueError(f"{label} mixes incompatible time/aspect ratios: {actual_ratio}")
        if rows.L.duplicated().any():
            raise ValueError(f"{label} contains duplicate sizes")
        result[label] = rows
    return result


def size_root(label: str, L: int) -> Path:
    dirname = f"N20x{L}"
    matches = [root / dirname for root in RUNS[label] if (root / dirname).is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label}, L={L} directory; got {matches}")
    return matches[0]


def quarter_drift(label: str, L: int) -> dict[str, float | None]:
    root = size_root(label, L)
    weights = pd.read_csv(root / "trajectory_weights.csv")
    pivot = weights.pivot(index="sample_index", columns="cycle", values="minus_log_p")
    cycles = pivot.columns.to_numpy(float)
    f0_curve = np.nanmean(pivot.to_numpy(float) / (L * cycles[None, :]), axis=0)

    tangent = np.load(root / "tangent_lyapunov.npz", allow_pickle=False)["spectra"]
    tangent_gap = np.nanmedian(np.nanmin(np.abs(tangent), axis=2), axis=0)
    choi_gap = np.load(root / "choi_rapidity.npz", allow_pickle=False)["choi_gap"]
    choi_curve = np.asarray([
        np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else np.nan
        for column in choi_gap.T
    ])

    def compare(curve: np.ndarray) -> float | None:
        n = curve.size
        previous_window = curve[n // 2: 3 * n // 4]
        final_window = curve[3 * n // 4:]
        previous_finite = previous_window[np.isfinite(previous_window)]
        final_finite = final_window[np.isfinite(final_window)]
        if not previous_finite.size or not final_finite.size:
            return None
        previous = np.mean(previous_finite)
        final = np.mean(final_finite)
        if previous == 0:
            return None
        return float(100.0 * (final / previous - 1.0))

    return {"f0_percent": compare(f0_curve), "tangent_percent": compare(tangent_gap),
            "choi_percent": compare(choi_curve)}


def late_half_f0(label: str, L: int) -> dict[str, float]:
    weights = pd.read_csv(size_root(label, L) / "trajectory_weights.csv")
    rates = []
    for _, sample in weights.groupby("sample_index"):
        sample = sample.sort_values("cycle")
        half = sample.iloc[len(sample) // 2:]
        rates.append(np.polyfit(half.cycle.to_numpy(float), half.minus_log_p.to_numpy(float) / L, 1)[0])
    rates = np.asarray(rates)
    return {"mean": float(rates.mean()), "se": float(rates.std(ddof=1) / np.sqrt(rates.size)),
            "sample_values": rates.tolist()}


def choi_saturation(L: int) -> dict[str, Any]:
    data = np.load(size_root("T20", L) / "choi_rapidity.npz", allow_pickle=False)
    gap = data["choi_gap"]
    eigenvalues = data["eigenvalues"]
    endpoint = data["endpoint_plus"] + data["endpoint_minus"]
    finite_by_cycle = np.isfinite(gap).sum(axis=0)
    finite_indices = np.flatnonzero(finite_by_cycle > 0)
    final_departure = np.max(1.0 - np.abs(eigenvalues[:, -1, :]))
    return {
        "dimension": int(eigenvalues.shape[2]),
        "finite_gap_observations": int(np.isfinite(gap).sum()),
        "total_sample_cycles": int(gap.size),
        "last_cycle_with_any_finite_gap": None if not finite_indices.size else int(finite_indices[-1] + 1),
        "finite_samples_final_cycle": int(finite_by_cycle[-1]),
        "endpoint_mean_final_cycle": float(endpoint[:, -1].mean()),
        "max_final_departure_from_endpoint": float(final_departure),
        "finite_samples_by_cycle": finite_by_cycle.tolist(),
    }


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 120, "savefig.dpi": 300, "axes.linewidth": 0.7,
        "lines.linewidth": 1.1, "lines.markersize": 4.0,
    })


def make_figures(rows: dict[str, pd.DataFrame], audit: dict[str, Any]) -> None:
    set_plot_style()

    fig, ax = plt.subplots(figsize=(3.375, 2.45))
    positions = np.arange(3)
    ceff = [audit["campaigns"][k]["fits"]["f0"]["free_intercept"]["c_eff"] for k in rows]
    ceff_se = [audit["campaigns"][k]["fits"]["f0"]["free_intercept"]["c_eff_se"] for k in rows]
    ax.errorbar(positions, ceff, yerr=ceff_se, fmt="o", color="#2b6cb0", capsize=2.5)
    ax.axhline(0, color="0.55", lw=0.7)
    ax.set_xticks(positions, [r"$T/L=2$", r"$T/L=5$", r"$T/L=20$"])
    ax.set_ylabel(r"free-intercept $c_{\mathrm{eff}}$")
    ax.set_title("Cylinder-fit instability")
    fig.tight_layout()
    fig.savefig(FIG / "central_charge_stability.png", bbox_inches="tight")
    plt.close(fig)

    colors = {"T2": "#9c4221", "T5": "#2f855a", "T20": "#2b6cb0"}
    fig, ax = plt.subplots(figsize=(3.375, 2.55))
    for label, frame in rows.items():
        L = frame.L.to_numpy(float)
        ax.plot(1.0 / L, frame.x_tangent_fock_size, "o-", color=colors[label], label=rf"tangent, $T/L={label[1:]}$")
        finite = np.isfinite(frame.x_choi_fock_size.to_numpy(float))
        if finite.any():
            ax.plot(1.0 / L[finite], frame.x_choi_fock_size.to_numpy(float)[finite], "s--",
                    color=colors[label], alpha=0.8, label=rf"Choi, $T/L={label[1:]}$")
    ax.set_xlabel(r"$1/L$")
    ax.set_ylabel(r"direct size estimate $L\Delta/(2\pi\alpha)$")
    ax.set_title("Gap estimates before extrapolation")
    ax.legend(ncol=2, frameon=False, columnspacing=0.7, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "direct_gap_estimates.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.375, 2.55))
    for L in (20, 22, 24):
        record = audit["choi_saturation"][str(L)]
        finite = np.asarray(record["finite_samples_by_cycle"], dtype=float)
        ax.plot(np.arange(1, finite.size + 1) / L, finite, label=rf"$L={L}$")
    ax.axhline(0, color="0.55", lw=0.7)
    ax.set_xlabel(r"$t/L$")
    ax.set_ylabel("samples with a finite Choi gap")
    ax.set_title("Late-time Choi rapidity saturation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "choi_saturation.png", bbox_inches="tight")
    plt.close(fig)

    rank = pd.read_csv(RANK_SCAN)
    fig, ax = plt.subplots(figsize=(3.375, 2.55))
    ax.plot(rank["rank"], rank["x_tangent_fock_fit"], "o-", label="tangent")
    ax.plot(rank["rank"], rank["x_choi_fock_fit"], "s--", label="Choi")
    ax.set_xlabel("postprocessed additive rank")
    ax.set_ylabel(r"free-intercept $x_R$")
    ax.set_title(r"Additive rank scan at $T/L=2$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rank_scan_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    audit: dict[str, Any] = {
        "scope": {
            "completed_only": True,
            "included_runs": {k: [str(p.relative_to(REPO)) for p in v] for k, v in RUNS.items()},
            "live_extension": str(LIVE_EXTENSION.relative_to(REPO)),
            "live_extension_complete": (LIVE_EXTENSION / "scalars_by_size.csv").is_file(),
            "live_extension_used": False,
        },
        "campaigns": {},
        "choi_saturation": {},
    }

    combined_rows = []
    for label, frame in rows.items():
        fit = derived_fits(frame)
        drift = {str(int(L)): quarter_drift(label, int(L)) for L in frame.L}
        campaign = {"rows": frame.to_dict(orient="records"), "fits": fit, "quarter_drift": drift}
        if label == "T20":
            campaign["late_half_f0"] = {str(int(L)): late_half_f0(label, int(L)) for L in frame.L}
            late = campaign["late_half_f0"]
            late_frame = pd.DataFrame({
                "L": frame.L.to_numpy(float),
                "f0": [late[str(int(L))]["mean"] for L in frame.L],
                "f0_se": [late[str(int(L))]["se"] for L in frame.L],
            })
            inv_l2 = 1.0 / late_frame.L.to_numpy(float) ** 2
            late_fit = wls(inv_l2, late_frame.f0.to_numpy(float), late_frame.f0_se.to_numpy(float))
            late_fit["c_eff"] = -6.0 * float(late_fit["slope"]) / np.pi
            late_fit["c_eff_se"] = 6.0 * float(late_fit["slope_se"]) / np.pi
            campaign["late_half_f0_fit"] = late_fit
        audit["campaigns"][label] = campaign
        exported = frame.copy()
        exported.insert(0, "time_ratio", int(label[1:]))
        combined_rows.append(exported)

    for L in (20, 22, 24):
        audit["choi_saturation"][str(L)] = choi_saturation(L)

    rank = pd.read_csv(RANK_SCAN)
    audit["rank_scan"] = rank.to_dict(orient="records")
    clean = finite_or_none(audit)
    (OUT / "audit_summary.json").write_text(json.dumps(clean, indent=2) + "\n")
    pd.concat(combined_rows, ignore_index=True).to_csv(OUT / "completed_campaign_rows.csv", index=False)
    make_figures(rows, audit)
    print(f"wrote {OUT / 'audit_summary.json'}")
    print(f"wrote {OUT / 'completed_campaign_rows.csv'}")
    for path in sorted(FIG.glob("*.png")):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
