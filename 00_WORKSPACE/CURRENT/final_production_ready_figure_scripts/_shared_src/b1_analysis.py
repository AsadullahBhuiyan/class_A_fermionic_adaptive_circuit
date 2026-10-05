from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

from b1_controller_frame import b1_cases, validate_b1_config
from production_runtime import (
    load_config,
    save_npz_atomic,
    sha256_file,
    verify_archive_receipt,
    write_json_atomic,
)


def _npz_from_tar(archive: tarfile.TarFile, suffix: str) -> dict[str, np.ndarray]:
    matches = [member for member in archive.getmembers() if member.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"archive contains {len(matches)} products ending in {suffix!r}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError(f"cannot read {matches[0].name}")
    with np.load(io.BytesIO(handle.read()), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _manifest_from_tar(archive: tarfile.TarFile) -> dict[str, Any]:
    matches = [
        member
        for member in archive.getmembers()
        if member.name.removeprefix("./") == "manifest.json"
    ]
    if len(matches) != 1:
        raise RuntimeError("B1 archive must contain one root manifest.json")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError("cannot read B1 archive manifest")
    return json.loads(handle.read().decode("utf-8"))


def discover_case_archives(
    archive_root: Path, *, case_id: str, minimum_shards: int
) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(archive_root.glob("*.tar.gz")):
        verify_archive_receipt(path)
        with tarfile.open(path, "r:gz") as archive:
            manifest = _manifest_from_tar(archive)
        if manifest.get("case_id") == case_id:
            found.append((path, manifest))
    if len(found) < int(minimum_shards):
        raise RuntimeError(
            f"{case_id} requires at least {minimum_shards} complete archives; found {len(found)}"
        )
    shard_indices = [int(manifest["shard_index"]) for _, manifest in found]
    if len(shard_indices) != len(set(shard_indices)) or sorted(shard_indices) != list(range(len(found))):
        raise RuntimeError(f"{case_id} has incomplete or duplicate shard indices: {shard_indices}")
    return sorted(found, key=lambda row: int(row[1]["shard_index"]))


def merge_case(
    archive_root: Path,
    *,
    case: dict[str, Any],
    minimum_samples: int,
    output_root: Path,
) -> dict[str, Any]:
    minimum_shards = minimum_samples // 5
    rows = discover_case_archives(
        archive_root, case_id=case["case_id"], minimum_shards=minimum_shards
    )
    actual_samples = 5 * len(rows)
    observables: dict[str, list[np.ndarray]] = {}
    records: dict[str, list[np.ndarray]] = {}
    sample_indices: list[int] = []
    static_reference: dict[str, np.ndarray] | None = None
    static_hash: str | None = None
    archive_hashes: dict[str, str] = {}
    for path, manifest in rows:
        if manifest.get("status") != "complete_local":
            raise RuntimeError(f"incomplete manifest in {path}")
        indices = [int(value) for value in manifest.get("global_sample_indices", [])]
        if len(indices) != 5:
            raise RuntimeError(f"{path} does not contain five global sample indices")
        sample_indices.extend(indices)
        archive_hashes[path.name] = sha256_file(path)
        with tarfile.open(path, "r:gz") as archive:
            obs = _npz_from_tar(archive, "controller_observables.npz")
            static = _npz_from_tar(archive, "controller_frame_static.npz")
            record = _npz_from_tar(archive, "ordered_record.npz")
        current_hash = str(np.asarray(static["frame_sha256"]).item())
        if static_hash is None:
            static_hash, static_reference = current_hash, static
        elif current_hash != static_hash:
            raise RuntimeError(f"controller-frame hash changes across {case['case_id']} shards")
        for key in (
            "total_charge",
            "charge_integer_residual",
            "controller_cost",
            "ky_fan_excess",
            "half_filled_manifold_distance",
            "purity_defect",
            "successive_covariance_delta",
        ):
            observables.setdefault(key, []).append(np.asarray(obs[key]))
        for key in (
            "wrong_outcome_per_cycle",
            "signed_transfer_per_cycle",
            "absolute_transfer_per_cycle",
            "self_information_per_cycle",
        ):
            records.setdefault(key, []).append(np.asarray(record[key]))
    if sorted(sample_indices) != list(range(actual_samples)):
        raise RuntimeError(f"{case['case_id']} does not contain consecutive samples 0..{actual_samples-1}")
    merged = {key: np.concatenate(value, axis=0) for key, value in observables.items()}
    merged.update({key: np.concatenate(value, axis=0) for key, value in records.items()})
    merged["sample_index"] = np.asarray(sample_indices, dtype=np.int64)
    order = np.argsort(merged["sample_index"])
    for key, value in list(merged.items()):
        if key != "sample_index" and value.ndim >= 1 and value.shape[0] == actual_samples:
            merged[key] = value[order]
    merged["sample_index"] = merged["sample_index"][order]
    cycles = int(case["run"]["cycles"])
    if merged["ky_fan_excess"].shape != (actual_samples, cycles + 1):
        raise RuntimeError("merged B1 observable shape is inconsistent with the configuration")
    if static_reference is None:
        raise RuntimeError("missing static controller-frame product")
    output_root.mkdir(parents=True, exist_ok=True)
    merged_path = output_root / f"{case['case_id']}_merged.npz"
    save_npz_atomic(
        merged_path,
        case_id=np.asarray(case["case_id"]),
        cycle=np.arange(cycles + 1, dtype=np.int64),
        **merged,
    )
    static_path = output_root / f"{case['case_id']}_static.npz"
    save_npz_atomic(static_path, **static_reference)
    return {
        "case_id": case["case_id"],
        "merged_path": str(merged_path),
        "merged_sha256": sha256_file(merged_path),
        "static_path": str(static_path),
        "static_sha256": sha256_file(static_path),
        "frame_sha256": static_hash,
        "samples": actual_samples,
        "minimum_samples": minimum_samples,
        "shards": len(rows),
        "input_archives": archive_hashes,
    }


def _bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    generator = np.random.default_rng(int(seed))
    estimates = np.empty((int(draws),), dtype=np.float64)
    for index in range(int(draws)):
        estimates[index] = np.mean(generator.choice(values, size=values.size, replace=True))
    return (
        float(np.mean(values)),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _bootstrap_independent_difference(
    left: np.ndarray, right: np.ndarray, *, draws: int, seed: int
) -> tuple[float, float, float]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    generator = np.random.default_rng(int(seed))
    estimates = np.empty((int(draws),), dtype=np.float64)
    for index in range(int(draws)):
        estimates[index] = np.mean(generator.choice(left, size=left.size, replace=True)) - np.mean(
            generator.choice(right, size=right.size, replace=True)
        )
    observed = float(np.mean(left) - np.mean(right))
    return observed, float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _trajectory_slopes(values: np.ndarray, start: int, stop: int) -> np.ndarray:
    time = np.arange(start, stop + 1, dtype=np.float64)
    centered = time - np.mean(time)
    denominator = float(np.sum(centered * centered))
    return np.sum((values[:, start : stop + 1] - np.mean(values[:, start : stop + 1], axis=1, keepdims=True)) * centered[None], axis=1) / denominator


def analyze_case(
    merged_path: Path,
    *,
    ny: int,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(merged_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    late = slice(3 * ny // 2, 2 * ny + 1)
    metrics = {
        "ky_fan_excess": np.asarray(arrays["ky_fan_excess"], dtype=np.float64),
        "half_filled_manifold_distance": np.asarray(
            arrays["half_filled_manifold_distance"], dtype=np.float64
        ),
    }
    summary: dict[str, Any] = {
        "case_id": str(np.asarray(arrays["case_id"]).item()),
        "samples": int(metrics["ky_fan_excess"].shape[0]),
        "Ny": int(ny),
        "cycles": 2 * int(ny),
        "late_window": [3 * ny // 2, 2 * ny],
        "slope_window": [ny, 2 * ny],
        "metrics": {},
    }
    derived: dict[str, np.ndarray] = {}
    attraction = True
    for metric_index, (name, values) in enumerate(metrics.items()):
        late_mean = np.mean(values[:, late], axis=1)
        initial = values[:, 0]
        change = late_mean - initial
        slopes = _trajectory_slopes(values, ny, 2 * ny)
        initial_ci = _bootstrap_mean(initial, draws=draws, seed=seed + 10 * metric_index)
        late_ci = _bootstrap_mean(late_mean, draws=draws, seed=seed + 10 * metric_index + 1)
        change_ci = _bootstrap_mean(change, draws=draws, seed=seed + 10 * metric_index + 2)
        slope_ci = _bootstrap_mean(slopes, draws=draws, seed=seed + 10 * metric_index + 3)
        attraction = attraction and change_ci[2] < 0.0 and slope_ci[2] < 0.0
        summary["metrics"][name] = {
            "initial_mean_ci95": initial_ci,
            "late_mean_ci95": late_ci,
            "late_minus_initial_mean_ci95": change_ci,
            "late_slope_mean_ci95": slope_ci,
            "late_median": float(np.median(late_mean)),
            "late_q05": float(np.quantile(late_mean, 0.05)),
            "late_q95": float(np.quantile(late_mean, 0.95)),
        }
        derived[f"{name}_late_per_trajectory"] = late_mean
        derived[f"{name}_slope_per_trajectory"] = slopes
        derived[f"{name}_mean"] = np.mean(values, axis=0)
        derived[f"{name}_median"] = np.median(values, axis=0)
        derived[f"{name}_q05"] = np.quantile(values, 0.05, axis=0)
        derived[f"{name}_q95"] = np.quantile(values, 0.95, axis=0)
    summary["classification"] = (
        "finite-window attraction signal"
        if attraction
        else f"no evidence of attraction over 0<=t<={2 * ny}"
    )
    summary["interpretation_limit"] = "finite-window diagnostic; no infinite-time extrapolation"
    return summary, derived


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _configure_plotting() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def make_figures(
    *,
    output_root: Path,
    cases: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    derived: dict[str, dict[str, np.ndarray]],
    merged_products: dict[str, dict[str, Any]],
) -> list[str]:
    _configure_plotting()
    import matplotlib.pyplot as plt

    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    colors = ("#1f77b4", "#d62728")
    labels = ("interface", "matched trivial")
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.6), constrained_layout=True)
    for case, color, label in zip(cases, colors, labels):
        with np.load(merged_products[case["case_id"]]["static_path"], allow_pickle=False) as data:
            eigenvalues = np.asarray(data["operator_eigenvalues"])
            residual_map = np.asarray(data["residual_map_half_filling"])
        axes[0].plot(np.arange(eigenvalues.size), eigenvalues, lw=0.8, color=color, label=label)
        axes[1].plot(np.nanmean(residual_map, axis=(1, 2)), lw=1.1, color=color, label=label)
    axes[0].set(xlabel="mode index", ylabel=r"$\epsilon_a(H_{\rm frame})$", title="Controller-frame spectrum")
    axes[1].set(xlabel=r"$x$", ylabel="mean residual", title="Half-filled residual profile")
    for axis in axes:
        axis.legend(frameon=False)
    static_stem = figure_root / "b1_static_controller_frame"
    fig.savefig(static_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(static_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.0), constrained_layout=True, sharex=True)
    for case, color, label in zip(cases, colors, labels):
        item = derived[case["case_id"]]
        for row, metric in enumerate(("ky_fan_excess", "half_filled_manifold_distance")):
            time = np.arange(item[f"{metric}_mean"].size)
            axes[row, 0].plot(time, item[f"{metric}_mean"], color=color, lw=1.0, label=label)
            axes[row, 0].fill_between(
                time, item[f"{metric}_q05"], item[f"{metric}_q95"], color=color, alpha=0.14
            )
            axes[row, 1].hist(
                item[f"{metric}_late_per_trajectory"], bins=18, histtype="step", color=color, label=label
            )
    axes[0, 0].set(ylabel=r"$\Delta F(t)$", title="Trajectory evolution")
    axes[1, 0].set(xlabel="cycle", ylabel="manifold distance")
    axes[0, 1].set(ylabel="count", title="Late-window distributions")
    axes[1, 1].set(xlabel="late-window value", ylabel="count")
    for axis in axes.flat:
        axis.legend(frameon=False)
    dynamics_stem = figure_root / "b1_surrogate_attraction"
    fig.savefig(dynamics_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(dynamics_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(static_stem.with_suffix(".pdf")), str(dynamics_stem.with_suffix(".pdf"))]


def write_report(
    *, output_root: Path, cases: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    report_root = output_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        summary = summaries[case["case_id"]]
        gap = summary["metrics"]["ky_fan_excess"]["late_mean_ci95"]
        distance = summary["metrics"]["half_filled_manifold_distance"]["late_mean_ci95"]
        label = case["protocol"].replace("_", r"\_")
        rows.append(
            f"{label} & {gap[0]:.6g} [{gap[1]:.6g},{gap[2]:.6g}] & "
            f"{distance[0]:.6g} [{distance[1]:.6g},{distance[2]:.6g}] & "
            f"{summary['classification']} \\\\"
        )
    tex = r"""\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}
\usepackage{amsmath,amssymb,amsthm,mathtools,bm}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage[colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue]{hyperref}
\usepackage{microtype}
\setcounter{tocdepth}{2}
\begin{document}
\title{B1 Controller-Frame Surrogate-Attraction Diagnostic}
\author{Numerical campaign report}
\date{\today}
\begin{abstract}
We test whether adaptive Gaussian trajectories approach the artificial fixed-rank
ground-state manifold of the signed controller-frame operator.  The calculation is an
required finite-window mechanism diagnostic and is not an equilibrium description of
the circuit.
\end{abstract}
\maketitle
\tableofcontents
\section{Definition}
For normalized controller orbitals $P_j$ and targets $s_j$, we define
\begin{equation}
H_{\rm frame}=\sum_j(1-2s_j)P_j,
\qquad
F[C]=\sum_j s_j+\operatorname{Tr}(H_{\rm frame}C).
\end{equation}
At the instantaneous integer rank $N(t)$, Ky Fan minimization gives
$F_\star(N)=\sum_j s_j+\sum_{a=1}^{N}\epsilon_a^\uparrow$.
No charge-sector frequency or fitted weight enters this comparison.
\section{Protocol and result}
Both cases use $N_x=20$, $N_y=40$, $n_{\rm shell}=1$, 10 independent trajectories
per protocol (50 trajectories total),
the random schedule, perfect correction, complex128, and exactly $2N_y$ cycles.
\begin{table}[b]
\caption{Late-window means and whole-trajectory 95\% bootstrap intervals.}
\begin{ruledtabular}
\begin{tabular}{llll}
protocol & $\Delta F$ & manifold distance & finite-window classification \\
\hline
""" + "\n".join(rows) + r"""
\end{tabular}
\end{ruledtabular}
\end{table}
\begin{figure}[t]
\includegraphics[width=0.95\linewidth]{../figures/b1_static_controller_frame.pdf}
\caption{Static signed controller-frame spectrum and half-filled residual profile.}
\end{figure}
\begin{figure}[t]
\includegraphics[width=0.95\linewidth]{../figures/b1_surrogate_attraction.pdf}
\caption{Finite-window Ky Fan excess and distance to the half-filled ground-state manifold.}
\end{figure}
\section{Interpretation}
The circuit is not generated by $H_{\rm frame}$.  Required campaign status does not
strengthen the scientific claim: these data license only a statement
about approach over $0\leq t\leq2N_y$ and never an infinite-time relaxation claim.
\end{document}
"""
    tex_path = report_root / "b1_controller_frame_report.tex"
    tex_path.write_text(tex, encoding="utf-8")
    compiled = False
    if shutil.which("pdflatex"):
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=report_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        compiled = True
    return {
        "tex": str(tex_path),
        "tex_sha256": sha256_file(tex_path),
        "pdf": str(tex_path.with_suffix(".pdf")) if compiled else None,
        "compiled": compiled,
    }


def analyze_bundle(
    *, bundle_root: Path, archive_root: Path, output_root: Path, smoke: bool
) -> dict[str, Any]:
    config = load_config(bundle_root)
    validate_b1_config(config)
    cases = b1_cases(config, smoke=smoke)
    minimum_samples = 5 if smoke else int(config["locked_contract"]["samples"])
    merged_products = {
        case["case_id"]: merge_case(
            archive_root,
            case=case,
            minimum_samples=minimum_samples,
            output_root=output_root / "merged",
        )
        for case in cases
    }
    summaries, derived = {}, {}
    for index, case in enumerate(cases):
        summary, arrays = analyze_case(
            Path(merged_products[case["case_id"]]["merged_path"]),
            ny=int(case["model"]["Ny"]),
            draws=int(config["B1"]["bootstrap_draws"]),
            seed=int(config["root_seed"]) + index,
        )
        summaries[case["case_id"]] = summary
        derived[case["case_id"]] = arrays
        save_npz_atomic(output_root / "analysis" / f"{case['case_id']}_analysis.npz", **arrays)
    rows = []
    for case in cases:
        summary = summaries[case["case_id"]]
        for metric, values in summary["metrics"].items():
            rows.append(
                {
                    "case_id": case["case_id"],
                    "metric": metric,
                    "initial_mean": values["initial_mean_ci95"][0],
                    "late_mean": values["late_mean_ci95"][0],
                    "late_ci_low": values["late_mean_ci95"][1],
                    "late_ci_high": values["late_mean_ci95"][2],
                    "slope_mean": values["late_slope_mean_ci95"][0],
                    "slope_ci_low": values["late_slope_mean_ci95"][1],
                    "slope_ci_high": values["late_slope_mean_ci95"][2],
                    "classification": summary["classification"],
                }
            )
    wall_id, control_id = cases[0]["case_id"], cases[1]["case_id"]
    control_comparison = {}
    comparison_rows = []
    for metric_index, metric in enumerate(
        ("ky_fan_excess", "half_filled_manifold_distance")
    ):
        difference = _bootstrap_independent_difference(
            derived[wall_id][f"{metric}_late_per_trajectory"],
            derived[control_id][f"{metric}_late_per_trajectory"],
            draws=int(config["B1"]["bootstrap_draws"]),
            seed=int(config["root_seed"]) + 100 + metric_index,
        )
        control_comparison[metric] = {
            "definition": "interface minus matched trivial late-window mean",
            "difference_mean_ci95": difference,
        }
        comparison_rows.append(
            {
                "metric": metric,
                "interface_minus_control_mean": difference[0],
                "ci_low": difference[1],
                "ci_high": difference[2],
            }
        )
    (output_root / "analysis").mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "analysis" / "b1_summary.csv", rows)
    _write_csv(output_root / "analysis" / "b1_control_comparison.csv", comparison_rows)
    figures = make_figures(
        output_root=output_root,
        cases=cases,
        summaries=summaries,
        derived=derived,
        merged_products=merged_products,
    )
    report = write_report(output_root=output_root, cases=cases, summaries=summaries)
    payload = {
        "schema": "b1_controller_frame_bundle_analysis_v1",
        "mode": "smoke" if smoke else "production",
        "merged_products": merged_products,
        "case_summaries": summaries,
        "interface_control_comparison": control_comparison,
        "figures": figures,
        "report": report,
        "charge_sector_weights_used": False,
        "training_test_split_used": False,
        "infinite_time_claim_permitted": False,
    }
    write_json_atomic(output_root / "b1_analysis_summary.json", payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and analyze complete B1 archives")
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("production", "smoke"), default="production")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_bundle(
        bundle_root=args.bundle_root.resolve(),
        archive_root=args.archive_root.resolve(),
        output_root=args.output_root.resolve(),
        smoke=args.mode == "smoke",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
