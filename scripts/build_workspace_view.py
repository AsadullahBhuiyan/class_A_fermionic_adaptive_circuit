#!/usr/bin/env python3
"""Validate and index the physical high-level workspace tree."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "00_WORKSPACE"

PROJECTS: dict[str, tuple[str, ...]] = {
    "CURRENT": (
        "cpu_cft_extraction",
        "experiment_review",
        "final_production_ready_figure_scripts",
        "prxq_draft",
        "tangent_cocycle_flux_snapshot",
        "tangent_edge_channel",
        "topological_frustration_diagnostics",
        "validation_campaigns",
    ),
    "COLAB": (
        "colab_charge_fluctuations",
        "colab_large_entanglement_scaling_N20",
        "colab_lyapunov",
        "colab_no_feedback_alpha_sweep_transfer",
        "colab_partial_post-select",
        "colab_regularized_choi_transfer_matrix",
        "colab_small_system_testing",
    ),
    "LARGE_RESULTS": (
        "choi_covariance_cpu",
        "dw_convergence",
        "experiments",
        "lyapunov_analysis_v2",
    ),
    "LEGACY": (
        "exact_DW_benchmark_notes",
        "form_factor_analysis",
        "markov_transfer_operators",
        "monitored_fermion_reference_sheet",
        "perturbative_expansion",
        "prl_draft",
        "projector_form_factor_product",
        "repo_synthesis",
        "sample_average_testing",
        "summary_of_results",
        "tangent_edge_channel_note",
        "topological_dynamics_introduction",
    ),
    "EXTERNAL": ("Haining_code", "Haoyu_code", "OSG"),
}

COMPATIBILITY_LINKS = ("src", "scripts", "tests", "notebooks", "cache", "figs", ".tmp")


def inventory(target: Path) -> tuple[float, int, int]:
    """Return newest mtime, apparent bytes, and regular-file count."""
    newest = target.stat().st_mtime
    total = 0
    count = 0
    for path in target.rglob("*"):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            newest = max(newest, stat.st_mtime)
            total += stat.st_size
            count += 1
    return newest, total, count


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError


def validate_tree() -> list[tuple[float, str, int, int]]:
    start = WORKSPACE / "START_HERE"
    if not start.is_dir() or start.is_symlink():
        raise FileNotFoundError(f"physical start directory missing: {start}")

    rows: list[tuple[float, str, int, int]] = []
    for category, names in PROJECTS.items():
        category_root = WORKSPACE / category
        if not category_root.is_dir() or category_root.is_symlink():
            raise FileNotFoundError(f"physical category missing: {category_root}")
        for name in names:
            project = category_root / name
            if not project.is_dir() or project.is_symlink():
                raise FileNotFoundError(f"physical project missing: {project}")
            newest, size, count = inventory(project)
            rows.append((newest, f"{category}/{name}", size, count))
        for shared in COMPATIBILITY_LINKS:
            link = category_root / shared
            expected = (ROOT / shared).resolve()
            if not link.is_symlink() or link.resolve() != expected:
                raise ValueError(f"invalid compatibility link: {link} -> {expected}")
    return rows


def main() -> None:
    rows = sorted(validate_tree(), reverse=True)
    lines = [
        "# Workspace recency index",
        "",
        "Generated from the physical project directories. The category tree is the",
        "canonical layout; this table provides the same projects ordered by recency.",
        "",
        "| Newest | Physical project | Size | Files |",
        "|---|---|---:|---:|",
    ]
    for newest, project, size, count in rows:
        timestamp = datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {timestamp} | `{project}` | {human_bytes(size)} | {count:,} |")
    (WORKSPACE / "RECENCY_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"physical_projects={len(rows)}")
    print(f"categories={len(PROJECTS)}")
    print(f"recency_index={WORKSPACE / 'RECENCY_INDEX.md'}")


if __name__ == "__main__":
    main()
