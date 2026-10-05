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

from p1_chern_runner import (
    BUNDLE,
    CONTRACT_AUDIT_SHA256,
    SAMPLING_REVISION,
    sha256_file,
    write_json_atomic,
)


def _receipt(archive: Path) -> dict[str, Any]:
    path = archive.with_suffix(archive.suffix + ".receipt.json")
    if not path.is_file():
        raise RuntimeError(f"missing receipt for {archive}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("archive") != archive.name:
        raise RuntimeError(f"receipt names a different archive: {archive}")
    if receipt.get("archive_sha256") != sha256_file(archive):
        raise RuntimeError(f"archive checksum mismatch: {archive}")
    return receipt


def _member_bytes(archive: tarfile.TarFile, suffix: str) -> bytes:
    matches = [
        member for member in archive.getmembers()
        if member.name.lstrip("./").endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {suffix!r} member, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError(f"cannot read archive member {matches[0].name}")
    return handle.read()


def load_archive(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    receipt = _receipt(path)
    with tarfile.open(path, "r:gz") as archive:
        manifest = json.loads(_member_bytes(archive, "manifest.json").decode("utf-8"))
        with np.load(io.BytesIO(_member_bytes(archive, "p1_chern.npz")), allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
    case = manifest.get("run_config", {}).get("case", {})
    if manifest.get("bundle") != BUNDLE:
        raise RuntimeError(f"wrong bundle in {path}")
    if manifest.get("audit_sha256") != CONTRACT_AUDIT_SHA256:
        raise RuntimeError(f"wrong P1 contract in {path}")
    if manifest.get("run_config", {}).get("sampling_revision") != SAMPLING_REVISION:
        raise RuntimeError(f"wrong P1 sampling revision in {path}")
    required = {
        "cycles", "global_sample_ids", "center_x", "center_y",
        "chern_by_center", "chern_center_mean", "observer_seconds",
    }
    if not required.issubset(arrays):
        raise RuntimeError(f"missing raw P1 arrays in {path}: {sorted(required - arrays.keys())}")
    forbidden = {
        "covariance", "bott_index", "density", "entropy_contour",
        "tangent", "ordered_record", "purity_gap",
    }
    if forbidden.intersection(arrays):
        raise RuntimeError(f"retired P1 products found in {path}")
    raw = arrays["chern_by_center"].astype(np.float64)
    mean = arrays["chern_center_mean"].astype(np.float64)
    sample_ids = arrays["global_sample_ids"].astype(np.int64)
    expected_width = int(case.get("execution", {}).get("samples_per_shard", 5))
    if sample_ids.shape != (expected_width,):
        raise RuntimeError(f"wrong global sample ID shape in {path}: {sample_ids.shape}")
    if manifest.get("global_sample_indices") != sample_ids.tolist():
        raise RuntimeError(f"manifest/product sample IDs differ in {path}")
    if raw.shape != (expected_width, int(case["model"]["Nx"]) + 1, 10):
        raise RuntimeError(f"wrong raw Chern shape in {path}: {raw.shape}")
    expected_leading = (expected_width, int(case["model"]["Nx"]) + 1)
    for key in ("center_x", "center_y"):
        if arrays[key].shape != (*expected_leading, 10):
            raise RuntimeError(f"wrong {key} shape in {path}: {arrays[key].shape}")
    if mean.shape != expected_leading or arrays["observer_seconds"].shape != expected_leading:
        raise RuntimeError(f"wrong per-sample P1 array shape in {path}")
    np.testing.assert_allclose(mean, raw.mean(axis=-1), rtol=0.0, atol=1e-13)
    if not np.isfinite(raw).all():
        raise RuntimeError(f"non-finite Chern values in {path}")
    return {"path": path, "receipt": receipt, "manifest": manifest, "case": case, "arrays": arrays}


def merge_archives(archive_root: Path | str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path(archive_root).glob("*.tar.gz")):
        try:
            row = load_archive(path)
        except RuntimeError as exc:
            if "wrong bundle" in str(exc) or "wrong P1 contract" in str(exc) or "wrong P1 sampling" in str(exc):
                continue
            raise
        groups[str(row["case"]["case_id"])].append(row)
    if len(groups) != 12:
        raise RuntimeError(f"expected 12 completed P1 cases, found {len(groups)}")

    merged: dict[str, dict[str, Any]] = {}
    for case_id, rows in groups.items():
        by_shard = {int(row["manifest"]["shard_index"]): row for row in rows}
        case = rows[0]["case"]
        width = int(case.get("execution", {}).get("samples_per_shard", 5))
        expected_shards = 25 // width
        if 25 % width or set(by_shard) != set(range(expected_shards)) or len(rows) != expected_shards:
            raise RuntimeError(
                f"{case_id}: expected exactly shards 0..{expected_shards - 1}"
            )
        ordered = [by_shard[index] for index in range(expected_shards)]
        arrays = {
            key: np.concatenate([row["arrays"][key] for row in ordered], axis=0)
            for key in (
                "global_sample_ids", "center_x", "center_y", "chern_by_center",
                "chern_center_mean", "observer_seconds",
            )
        }
        size = int(case["model"]["Nx"])
        if arrays["global_sample_ids"].tolist() != list(range(25)):
            raise RuntimeError(f"{case_id}: global sample IDs are incomplete or unordered")
        if arrays["chern_by_center"].shape != (25, size + 1, 10):
            raise RuntimeError(f"{case_id}: merged Chern shape is wrong")
        merged[case_id] = {
            "case": ordered[0]["case"],
            "cycles": ordered[0]["arrays"]["cycles"].astype(np.int64),
            **arrays,
            "archives": [str(row["path"]) for row in ordered],
            "archive_sha256": [row["receipt"]["archive_sha256"] for row in ordered],
        }

    by_size: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in merged.values():
        by_size[int(item["case"]["model"]["Nx"])].append(item)
    for size, items in by_size.items():
        reference = items[0]
        for item in items[1:]:
            np.testing.assert_array_equal(item["center_x"], reference["center_x"])
            np.testing.assert_array_equal(item["center_y"], reference["center_y"])
    return merged


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(archive_root: Path | str, output_root: Path | str) -> dict[str, Any]:
    merged = merge_archives(archive_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    series: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for item in merged.values():
        case = item["case"]
        size = int(case["model"]["Nx"])
        shell = "None" if case["model"]["nshell"] is None else str(int(case["model"]["nshell"]))
        values = np.asarray(item["chern_center_mean"], dtype=np.float64)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        sem = std / math.sqrt(25.0)
        ci = 2.0639 * sem
        cycles = np.asarray(item["cycles"], dtype=np.int64)
        series[(size, shell)] = {"cycles": cycles, "mean": mean}
        for cycle, center, spread, error, half_width in zip(cycles, mean, std, sem, ci):
            rows.append({
                "L": size,
                "n_shell": shell,
                "cycle": int(cycle),
                "t_over_L": float(cycle / size),
                "chern_mean": float(center),
                "sample_std": float(spread),
                "sample_sem": float(error),
                "ci95_low": float(center - half_width),
                "ci95_high": float(center + half_width),
                "absolute_target_error": float(abs(1.0 - center)),
            })
        final_rows.append({
            "L": size,
            "n_shell": shell,
            "samples": 25,
            "centers_per_sample_cycle": 10,
            "final_chern_mean": float(mean[-1]),
            "final_sample_std": float(std[-1]),
            "final_absolute_target_error": float(abs(1.0 - mean[-1])),
        })
    rows.sort(key=lambda row: (row["L"], row["n_shell"], row["cycle"]))
    final_rows.sort(key=lambda row: (row["L"], row["n_shell"]))
    _write_csv(output_root / "p1_chern_timeseries.csv", rows)
    _write_csv(output_root / "p1_chern_final_summary.csv", final_rows)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    colors = {"1": "#d62728", "2": "#2ca02c", "None": "#1f77b4"}
    markers = {"1": "^", "2": "s", "None": "o"}
    labels = {"1": r"$n_{\rm shell}=1$", "2": r"$n_{\rm shell}=2$", "None": r"$n_{\rm shell}=\mathrm{None}$"}
    styles = {16: ":", 24: "-.", 32: "--", 64: "-"}
    figure, axis = plt.subplots(figsize=(3.375, 3.0), dpi=300)
    inset = axis.inset_axes([0.43, 0.43, 0.54, 0.54])
    for size in (16, 24, 32, 64):
        for shell in ("1", "2", "None"):
            item = series[(size, shell)]
            cycles, mean = item["cycles"], item["mean"]
            early = cycles <= 10
            axis.plot(
                cycles[early], mean[early], color=colors[shell], linestyle=styles[size],
                marker=markers[shell], markerfacecolor="none", markersize=3.0,
                linewidth=0.9, markevery=1,
            )
            inset.plot(
                cycles / size,
                np.maximum(np.abs(1.0 - mean), np.finfo(np.float64).tiny),
                color=colors[shell], linestyle=styles[size], marker=markers[shell],
                markerfacecolor="none", markersize=2.0, linewidth=0.7,
                markevery=max(1, size // 8),
            )
    shell_handles = [
        axis.plot([], [], color=colors[shell], marker=markers[shell], linestyle="none", markerfacecolor="none", label=labels[shell])[0]
        for shell in ("1", "2", "None")
    ]
    size_handles = [
        axis.plot([], [], color="0.2", linestyle=styles[size], label=rf"$L={size}$")[0]
        for size in (16, 24, 32, 64)
    ]
    first_legend = axis.legend(handles=shell_handles, loc="lower right", frameon=False, fontsize=6.5)
    axis.add_artist(first_legend)
    axis.legend(handles=size_handles, loc="center right", frameon=False, fontsize=6.5)
    axis.axhline(1.0, color="0.4", linestyle="--", linewidth=0.7, zorder=0)
    axis.set(xlabel=r"cycle $t$", ylabel=r"$\overline{C_G}$", xlim=(-0.2, 10.2))
    inset.set(xlabel=r"$t/L$", ylabel=r"$|1-\overline{C_G}|$", xlim=(0.0, 1.0), yscale="log")
    inset.tick_params(labelsize=6)
    inset.xaxis.label.set_size(7)
    inset.yaxis.label.set_size(7)
    figure.tight_layout()
    pdf = output_root / "p1_chern_dynamics_reference_style.pdf"
    png = output_root / "p1_chern_dynamics_reference_style.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "schema": "p1_chern_analysis_v1",
        "status": "complete",
        "case_count": 12,
        "samples_per_case": 25,
        "centers_per_sample_cycle": 10,
        "averaging_order": "center_mean_within_sample_cycle_then_mean_over_25_samples",
        "chern_target": 1.0,
        "final_rows": final_rows,
        "outputs": {
            "timeseries_csv": str(output_root / "p1_chern_timeseries.csv"),
            "final_summary_csv": str(output_root / "p1_chern_final_summary.csv"),
            "figure_pdf": str(pdf),
            "figure_png": str(png),
        },
    }
    write_json_atomic(output_root / "p1_chern_analysis_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge and plot lean P1 Chern archives")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.archive_root, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
