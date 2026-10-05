"""Merge, characterize, and plot the standalone H1 modular-response archives."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from h1_io import sha256_file, write_json_atomic
from h1_modular_runner import AUDIT, BUNDLE, REVISION, load_config


def _member_bytes(archive: tarfile.TarFile, suffix: str) -> bytes:
    matches = [
        member for member in archive.getmembers()
        if member.name.lstrip("./").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r} member, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError(f"cannot read {matches[0].name}")
    return handle.read()


def _npz_member(archive: tarfile.TarFile, suffix: str) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(_member_bytes(archive, suffix)), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _verify_receipt(path: Path) -> dict[str, Any]:
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    if not receipt_path.is_file():
        raise RuntimeError(f"missing H1 archive receipt: {path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("archive") != path.name or receipt.get("archive_sha256") != sha256_file(path):
        raise RuntimeError(f"H1 archive receipt failed: {path}")
    return receipt


def load_archive(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    receipt = _verify_receipt(path)
    with tarfile.open(path, "r:gz") as archive:
        manifest = json.loads(_member_bytes(archive, "manifest.json").decode("utf-8"))
        common = _npz_member(archive, "h1_modular/common.npz")
        summary = _npz_member(archive, "h1_modular/retarded_summary.npz")
        packet = _npz_member(archive, "h1_modular/packet_drift.npz")
        fields = _npz_member(archive, "h1_modular/retarded_fields.npz")
    if manifest.get("bundle") != BUNDLE:
        raise RuntimeError(f"wrong bundle in {path}")
    if manifest.get("sampling_revision") != REVISION or manifest.get("audit_sha256") != AUDIT:
        raise RuntimeError(f"wrong H1 revision or audit in {path}")
    forbidden = {"static_susceptibility", "covariance", "eigenvectors", "entropy", "chern", "bott", "tangent"}
    found = forbidden.intersection({*common, *summary, *packet, *fields})
    if found:
        raise RuntimeError(f"retired H1 products found in {path}: {sorted(found)}")
    if set(fields) != {"schema", "retarded_density_xy"}:
        raise RuntimeError(f"unexpected source-resolved response schema in {path}")
    return {
        "path": path,
        "receipt": receipt,
        "manifest": manifest,
        "case": manifest["run_config"]["case"],
        "common": common,
        "summary": summary,
        "packet": packet,
    }


def merge_archives(archive_root: Path | str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(archive_root).glob("*.tar.gz")):
        try:
            item = load_archive(path)
        except RuntimeError as exc:
            if "wrong bundle" in str(exc) or "wrong H1 revision" in str(exc):
                continue
            raise
        groups[item["case"]["case_id"]].append(item)
    if len(groups) != 4:
        raise RuntimeError(f"expected four complete H1 cases, found {len(groups)}")

    merged: dict[str, dict[str, Any]] = {}
    for case_id, rows in groups.items():
        by_shard = {int(row["manifest"]["shard_index"]): row for row in rows}
        if set(by_shard) != set(range(5)) or len(rows) != 5:
            raise RuntimeError(f"{case_id}: expected exactly shards 0..4")
        ordered = [by_shard[index] for index in range(5)]
        sample_ids = np.concatenate([
            row["common"]["global_sample_ids"] for row in ordered
        ])
        if sample_ids.tolist() != list(range(25)):
            raise RuntimeError(f"{case_id}: incomplete or unordered sample IDs")
        common_keys = (
            "source_xy", "restricted_occupation_spectrum", "spectral_clip_counts",
            "covariance_hermiticity_error", "observer_seconds",
        )
        summary_keys = (
            "retarded_source_mean_aligned_density",
            "retarded_source_mean_aligned_profile",
            "retarded_wall_velocity", "retarded_wall_velocity_r2",
            "handed_response", "retarded_total_charge",
            "retarded_transverse_leakage",
        )
        packet_keys = (
            "packet_source_mean_profile", "packet_wall_velocity",
            "packet_wall_fit_points", "packet_handed_velocity",
            "packet_wall_retention",
        )
        arrays = {
            key: np.concatenate([row["common"][key] for row in ordered], axis=0)
            for key in common_keys
        }
        arrays.update({
            key: np.concatenate([row["summary"][key] for row in ordered], axis=0)
            for key in summary_keys
        })
        arrays.update({
            key: np.concatenate([row["packet"][key] for row in ordered], axis=0)
            for key in packet_keys
        })
        reference = ordered[0]["common"]
        merged[case_id] = {
            "case": ordered[0]["case"],
            "sample_ids": sample_ids,
            "checkpoints": reference["checkpoints"].astype(np.int64),
            "modular_times": reference["modular_times"].astype(np.float64),
            "packet_modular_times": reference["packet_modular_times"].astype(np.float64),
            "spectral_clip_eps": reference["spectral_clip_eps"].astype(np.float64),
            "wall_half_widths": reference["wall_half_widths"].astype(np.int64),
            "primary_epsilon_index": int(reference["primary_epsilon_index"]),
            "primary_wall_width_index": int(reference["primary_wall_width_index"]),
            **arrays,
            "archives": [str(row["path"]) for row in ordered],
        }
    source_reference = next(iter(merged.values()))["source_xy"]
    for case_id, item in merged.items():
        if not np.array_equal(item["source_xy"], source_reference):
            raise RuntimeError(f"{case_id}: source coordinates do not match other cases")
    return merged


def bootstrap_mean_ci(
    values: np.ndarray, *, repetitions: int, seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite trajectory vector")
    rng = np.random.default_rng(int(seed))
    draws = values[rng.integers(0, len(values), size=(int(repetitions), len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def tukey_window(length: int, alpha: float = 0.25) -> np.ndarray:
    if length <= 1:
        return np.ones(length)
    x = np.linspace(0.0, 1.0, length)
    result = np.ones(length)
    left = x < alpha / 2
    right = x > 1 - alpha / 2
    result[left] = 0.5 * (1 + np.cos(np.pi * (2 * x[left] / alpha - 1)))
    result[right] = 0.5 * (1 + np.cos(np.pi * (2 * x[right] / alpha - 2 / alpha + 1)))
    return result


def finite_window_spectrum(
    aligned_density: np.ndarray, *, wall_x: int, wall_half_width: int,
    modular_times: np.ndarray, eta: float, omega_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct finite-aperture transform of one source-averaged response field."""

    aligned_density = np.asarray(aligned_density, dtype=np.float64)
    _, nd, nx = aligned_density.shape
    ay = (nd + 1) // 2
    displacements = np.arange(-(ay - 1), ay, dtype=np.float64)
    xmask = np.asarray([
        min((x - wall_x) % nx, (wall_x - x) % nx) <= int(wall_half_width)
        for x in range(nx)
    ])
    wall_field = aligned_density[..., xmask].sum(axis=-1)
    k_values = np.sort(2.0 * np.pi * np.fft.fftfreq(ay))
    spatial = np.exp(-1j * k_values[:, None] * displacements[None]) @ wall_field.T
    taper = tukey_window(len(modular_times), alpha=0.25) * np.exp(-float(eta) * modular_times)
    temporal = np.exp(1j * omega_values[:, None] * modular_times[None])
    spectrum = temporal @ (spatial * taper[None]).T
    if len(modular_times) > 1:
        spectrum *= modular_times[1] - modular_times[0]
    return k_values, spectrum


def extract_ridge(
    spectrum: np.ndarray, *, k_values: np.ndarray, omega_values: np.ndarray,
    sector_sign: int, omega_window: tuple[float, float], k_max: float,
) -> dict[str, Any]:
    kmask = (k_values * int(sector_sign) > 0.1) & (np.abs(k_values) <= float(k_max))
    omask = (
        (omega_values >= float(omega_window[0]) - 1e-12)
        & (omega_values <= float(omega_window[1]) + 1e-12)
    )
    amplitude = np.abs(spectrum[np.ix_(omask, kmask)])
    peak = omega_values[omask][np.argmax(amplitude, axis=0)]
    abs_k = np.abs(k_values[kmask])
    design = np.column_stack((np.ones(len(abs_k)), abs_k))
    intercept, velocity = np.linalg.lstsq(design, peak, rcond=None)[0]
    predicted = intercept + velocity * abs_k
    denominator = float(np.sum((peak - peak.mean()) ** 2))
    r2 = float("nan") if denominator <= 0 else 1.0 - float(np.sum((peak - predicted) ** 2)) / denominator
    boundary_fraction = float(np.mean(np.isclose(peak, omega_window[0])))
    return {
        "k": k_values[kmask],
        "abs_k": abs_k,
        "omega_peak": peak,
        "intercept": float(intercept),
        "velocity": float(velocity),
        "r2": r2,
        "boundary_fraction": boundary_fraction,
        "resolved": bool(boundary_fraction < 0.5 and np.isfinite(r2)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _case_label(item: dict[str, Any]) -> str:
    return f"{item['case']['protocol']}, $\\alpha_1={item['case']['model']['alpha_1']:g}$"


def analyze(
    archive_root: Path | str, output_root: Path | str,
    *, bundle_root: Path | str | None = None,
) -> dict[str, Any]:
    merged = merge_archives(archive_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if bundle_root is None:
        bundle_root = Path(__file__).resolve().parents[1]
    config = load_config(bundle_root)
    analysis_config = config["analysis"]
    repetitions = int(analysis_config["bootstrap_repetitions"])
    seed = int(analysis_config["bootstrap_seed"])
    ordered_ids = [
        f"H1_N20x40_{protocol}_a1-{alpha}"
        for alpha in (1, 3) for protocol in ("hard", "soft")
    ]
    styles = {
        ordered_ids[0]: ("#b2182b", "o", "-"),
        ordered_ids[1]: ("#ef8a62", "^", "--"),
        ordered_ids[2]: ("#2166ac", "s", "-"),
        ordered_ids[3]: ("#67a9cf", "D", "--"),
    }
    summary_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    late_values: dict[str, np.ndarray] = {}
    for case_index, case_id in enumerate(ordered_ids):
        item = merged[case_id]
        pe, pw = item["primary_epsilon_index"], item["primary_wall_width_index"]
        handed = item["handed_response"][:, :, pe, pw]
        late = handed.mean(axis=1)
        late_values[case_id] = late
        mean, low, high = bootstrap_mean_ci(
            late, repetitions=repetitions, seed=seed + case_index
        )
        summary_rows.append({
            "case_id": case_id,
            "protocol": item["case"]["protocol"],
            "alpha_1": item["case"]["model"]["alpha_1"],
            "trajectories": 25,
            "late_handed_mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "checkpoint_reduction": "mean_six_within_trajectory",
        })
        for checkpoint_index, checkpoint in enumerate(item["checkpoints"]):
            center, lo, hi = bootstrap_mean_ci(
                handed[:, checkpoint_index], repetitions=repetitions,
                seed=seed + 100 + 10 * case_index + checkpoint_index,
            )
            checkpoint_rows.append({
                "case_id": case_id,
                "checkpoint": int(checkpoint),
                "handed_mean": center,
                "ci95_low": lo,
                "ci95_high": hi,
            })

    contrast_rows = []
    contrast_specs = [
        ("hard_minus_soft_alpha1_1", ordered_ids[0], ordered_ids[1]),
        ("hard_minus_soft_alpha1_3", ordered_ids[2], ordered_ids[3]),
        ("alpha3_minus_alpha1_hard", ordered_ids[2], ordered_ids[0]),
        ("alpha3_minus_alpha1_soft", ordered_ids[3], ordered_ids[1]),
    ]
    rng = np.random.default_rng(seed + 1000)
    for label, left, right in contrast_specs:
        a, b = late_values[left], late_values[right]
        draws = (
            a[rng.integers(0, len(a), size=(repetitions, len(a)))].mean(axis=1)
            - b[rng.integers(0, len(b), size=(repetitions, len(b)))].mean(axis=1)
        )
        contrast_rows.append({
            "contrast": label,
            "left_case": left,
            "right_case": right,
            "difference": float(a.mean() - b.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "bootstrap": "unpaired_trajectory",
        })
    _write_csv(output_root / "h1_late_handed_summary.csv", summary_rows)
    _write_csv(output_root / "h1_checkpoint_handed_summary.csv", checkpoint_rows)
    _write_csv(output_root / "h1_unpaired_contrasts.csv", contrast_rows)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })
    figure, axis = plt.subplots(figsize=(3.375, 2.7), dpi=300)
    for case_id in ordered_ids:
        rows = [row for row in checkpoint_rows if row["case_id"] == case_id]
        color, marker, linestyle = styles[case_id]
        x = np.asarray([row["checkpoint"] for row in rows])
        y = np.asarray([row["handed_mean"] for row in rows])
        lo = np.asarray([row["ci95_low"] for row in rows])
        hi = np.asarray([row["ci95_high"] for row in rows])
        axis.errorbar(
            x, y, yerr=(y - lo, hi - y), color=color, marker=marker,
            linestyle=linestyle, linewidth=0.9, markersize=3.2, capsize=1.5,
            label=_case_label(merged[case_id]),
        )
    axis.axhline(0.0, color="0.4", linestyle=":", linewidth=0.7)
    axis.set(xlabel="circuit checkpoint", ylabel=r"handed modular response $H$")
    axis.legend(frameon=False, fontsize=6.4, ncol=2)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_root / f"h1_handed_response_checkpoints.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.375, 2.7), dpi=300)
    x = np.arange(len(contrast_rows))
    center = np.asarray([row["difference"] for row in contrast_rows])
    low = np.asarray([row["ci95_low"] for row in contrast_rows])
    high = np.asarray([row["ci95_high"] for row in contrast_rows])
    labels = [
        "hard$-$soft\n$\\alpha_1=1$",
        "hard$-$soft\n$\\alpha_1=3$",
        "$\\alpha_1:3-1$\nhard",
        "$\\alpha_1:3-1$\nsoft",
    ]
    axis.errorbar(
        x, center, yerr=(center - low, high - center), fmt="o",
        color="0.15", markersize=4, capsize=2, linewidth=0.9,
    )
    axis.axhline(0.0, color="0.45", linestyle="--", linewidth=0.7)
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"unpaired difference in late $H$")
    axis.tick_params(axis="x", labelsize=6.5)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_root / f"h1_handed_contrasts.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    response_time = next(iter(merged.values()))["modular_times"]
    time_index = int(np.argmin(np.abs(response_time - 0.8)))
    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.7), constrained_layout=True)
    image = None
    for panel, case_id in enumerate(ordered_ids):
        item = merged[case_id]
        field = item["retarded_source_mean_aligned_density"][:, :, 0, time_index].mean(axis=(0, 1))
        axis = axes.flat[panel]
        limit = max(float(np.max(np.abs(field))), 1e-12)
        image = axis.imshow(
            field, origin="lower", aspect="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, extent=(-0.5, 19.5, -19.5, 19.5),
        )
        axis.set_title(_case_label(item))
        axis.set(xlabel=r"$x$", ylabel=r"source-relative $d_y$")
        axis.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=axis.transAxes, fontweight="bold")
    if image is not None:
        figure.colorbar(image, ax=axes, label=rf"$\chi^R(x,d_y,\tau={response_time[time_index]:g})$", shrink=0.84)
    for suffix in ("pdf", "png"):
        figure.savefig(output_root / f"h1_real_space_retarded_response.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    omega_values = np.linspace(
        float(analysis_config["fourier_omega_min"]),
        float(analysis_config["fourier_omega_max"]),
        int(analysis_config["fourier_omega_points"]),
    )
    eta_values = [float(value) for value in analysis_config["fourier_eta_sensitivity"]]
    ridge_rows: list[dict[str, Any]] = []
    ensemble_spectra: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for case_index, case_id in enumerate(ordered_ids):
        item = merged[case_id]
        density = item["retarded_source_mean_aligned_density"].mean(axis=(0, 1))
        for wall, sector in ((0, -1), (1, +1)):
            for eta in eta_values:
                k_values, spectrum = finite_window_spectrum(
                    density[wall], wall_x=(5, 15)[wall], wall_half_width=2,
                    modular_times=item["modular_times"], eta=eta,
                    omega_values=omega_values,
                )
                fit = extract_ridge(
                    spectrum, k_values=k_values, omega_values=omega_values,
                    sector_sign=sector,
                    omega_window=tuple(analysis_config["ridge_omega_window"]),
                    k_max=float(analysis_config["ridge_k_max"]),
                )
                ridge_rows.append({
                    "case_id": case_id,
                    "wall": wall,
                    "momentum_sector": "k<0" if sector < 0 else "k>0",
                    "eta": eta,
                    "resolved": fit["resolved"],
                    "velocity": fit["velocity"],
                    "intercept": fit["intercept"],
                    "r2": fit["r2"],
                    "boundary_fraction": fit["boundary_fraction"],
                })
                if math.isclose(eta, float(analysis_config["fourier_eta"])):
                    ensemble_spectra[(case_id, wall)] = (k_values, spectrum, fit)
    _write_csv(output_root / "h1_fourier_ridge_summary.csv", ridge_rows)

    figure, axes = plt.subplots(2, 2, figsize=(7.05, 4.8), constrained_layout=True)
    image = None
    for panel, case_id in enumerate(ordered_ids):
        k0, spectrum0, fit0 = ensemble_spectra[(case_id, 0)]
        k1, spectrum1, fit1 = ensemble_spectra[(case_id, 1)]
        oriented = 0.5 * (np.abs(spectrum0) + np.abs(spectrum1[:, ::-1]))
        oriented /= max(float(oriented.max()), 1e-300)
        axis = axes.flat[panel]
        image = axis.pcolormesh(k0, omega_values, oriented, shading="nearest", cmap="magma", vmin=0, vmax=1)
        if fit0["resolved"]:
            axis.plot(fit0["k"], fit0["omega_peak"], "wo", ms=2.7, mec="k", mew=0.3)
        if fit1["resolved"]:
            axis.plot(-fit1["k"], fit1["omega_peak"], "w^", ms=2.7, mec="k", mew=0.3)
        axis.set_title(_case_label(merged[case_id]))
        axis.set(xlabel="oriented finite-window $k$", ylabel=r"modular frequency $\omega$")
        axis.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=axis.transAxes, fontweight="bold")
        if not (fit0["resolved"] and fit1["resolved"]):
            axis.text(0.04, 0.90, "one or more ridges unresolved", transform=axis.transAxes, color="white", fontsize=6.5)
    if image is not None:
        figure.colorbar(image, ax=axes, label=r"normalized $|\chi_w(k,\omega)|$", shrink=0.84)
    for suffix in ("pdf", "png"):
        figure.savefig(output_root / f"h1_fourier_spectra.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(3.375, 2.7), dpi=300)
    for case_id in ordered_ids:
        item = merged[case_id]
        pe, pw = item["primary_epsilon_index"], item["primary_wall_width_index"]
        packet_handed = item["packet_handed_velocity"][:, :, pe, pw]
        finite_count = np.sum(np.isfinite(packet_handed), axis=0)
        mean = np.nanmean(packet_handed, axis=0)
        sem = np.nanstd(packet_handed, axis=0, ddof=1) / np.sqrt(
            np.maximum(finite_count, 1)
        )
        color, marker, linestyle = styles[case_id]
        axis.errorbar(
            item["checkpoints"], mean, yerr=1.96 * sem, color=color,
            marker=marker, linestyle=linestyle, linewidth=0.9, markersize=3.2,
            capsize=1.5, label=_case_label(item),
        )
    axis.axhline(0.0, color="0.4", linestyle=":", linewidth=0.7)
    axis.set(xlabel="circuit checkpoint", ylabel="supplemental packet handed velocity")
    axis.legend(frameon=False, fontsize=6.4, ncol=2)
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_root / f"h1_packet_drift_supplement.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    table_lines = [
        r"\begin{tabular}{lccc}",
        r"\hline\hline",
        r"case & $\overline H$ & 95\% CI & $S$\\",
        r"\hline",
    ]
    for row in summary_rows:
        label = f"{row['protocol']}, $\\alpha_1={row['alpha_1']:g}$"
        table_lines.append(
            f"{label} & {row['late_handed_mean']:.4g} & "
            f"[{row['ci95_low']:.4g},{row['ci95_high']:.4g}] & 25\\\\"
        )
    table_lines.extend((r"\hline\hline", r"\end{tabular}"))
    (output_root / "h1_results_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    diagnostics = {
        case_id: {
            "retarded_total_charge_max": float(np.max(np.abs(item["retarded_total_charge"]))),
            "covariance_hermiticity_error_max": float(np.max(item["covariance_hermiticity_error"])),
            "mean_primary_transverse_leakage": float(np.mean(
                item["retarded_transverse_leakage"][:, :, item["primary_epsilon_index"], item["primary_wall_width_index"]]
            )),
        }
        for case_id, item in merged.items()
    }
    result = {
        "schema": "h1_modular_response_analysis_v1",
        "status": "complete",
        "case_count": 4,
        "samples_per_case": 25,
        "checkpoints": [40, 48, 56, 64, 72, 80],
        "sources_per_wall_checkpoint": 10,
        "primary_estimator": "average_sources_within_wall_then_fit; form H per trajectory-checkpoint; average checkpoints within trajectory; bootstrap trajectories",
        "static_susceptibility": "omitted",
        "summary_rows": summary_rows,
        "contrast_rows": contrast_rows,
        "diagnostics": diagnostics,
        "fourier_interpretation": "finite-window exploratory ridge; require checkpoint and eta stability",
        "outputs": sorted(path.name for path in output_root.iterdir()),
    }
    write_json_atomic(output_root / "h1_analysis_summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    result = analyze(args.archive_root, args.output_root, bundle_root=args.bundle_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
