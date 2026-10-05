#!/usr/bin/env python3
"""Build a categorized, non-destructive documentation library."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "NOTES"
WORKSPACE = ROOT / "00_WORKSPACE"
CURRENT = WORKSPACE / "CURRENT"
COLAB = WORKSPACE / "COLAB"
LEGACY = WORKSPACE / "LEGACY"
EXTERNAL = WORKSPACE / "EXTERNAL"
TEXT_EXTENSIONS = {".md", ".tex", ".txt", ".bib"}
EXCLUDED_ROOTS = {
    ".git",
    ".pytest_cache",
    "NOTES",
    "__pycache__",
    "cache",
}
PDF_PROJECT_ROOTS = {
    "Haoyu_code",
    "OSG",
    "Paper Methods",
    "colab_charge_fluctuations",
    "colab_large_entanglement_scaling_N20",
    "colab_lyapunov",
    "colab_regularized_choi_transfer_matrix",
    "exact_DW_benchmark_notes",
    "experiment_review",
    "form_factor_analysis",
    "monitored_fermion_reference_sheet",
    "markov_transfer_operators",
    "perturbative_expansion",
    "prl_draft",
    "prxq_draft",
    "repo_synthesis",
    "src",
    "summary_of_results",
    "tangent_edge_channel_note",
    "topological_dynamics_introduction",
    "topological_frustration_diagnostics",
}
MANAGED_DOCUMENT_ROOTS = (
    NOTES / "20_THEORY_AND_METHODS/standalone",
    NOTES / "40_EXTERNAL_PAPERS/sources",
    NOTES / "90_LEGACY/root_imports",
)


CURATED: dict[str, Path] = {
    # The small set normally worth opening first.
    "00_CURRENT/00_numerical_campaign_working.pdf": CURRENT / "experiment_review/numerical_campaign_legacy_working.pdf",
    "00_CURRENT/01_numerical_campaign_working.tex": CURRENT / "experiment_review/numerical_campaign_legacy_working.tex",
    "00_CURRENT/02_production_bundles_README.md": CURRENT / "final_production_ready_figure_scripts/README.md",
    "00_CURRENT/03_completion_audit.md": ROOT / "PROJECT_ADMIN/COMPLETION_AUDIT.md",
    "00_CURRENT/04_workspace_layout.md": ROOT / "PROJECT_ADMIN/WORKSPACE_LAYOUT.md",
    "00_CURRENT/05_current_manuscript": CURRENT / "prxq_draft",
    "00_CURRENT/06_experiment_review_README.md": CURRENT / "experiment_review/README.md",

    # Manuscript-scale narratives.
    "10_MANUSCRIPTS/00_PRXQ_CURRENT": CURRENT / "prxq_draft",
    "10_MANUSCRIPTS/10_PRL_DRAFT": LEGACY / "prl_draft",
    "10_MANUSCRIPTS/20_TOPOLOGICAL_DYNAMICS_INTRODUCTION": LEGACY / "topological_dynamics_introduction",
    "10_MANUSCRIPTS/30_SUMMARY_OF_RESULTS": LEGACY / "summary_of_results",
    "10_MANUSCRIPTS/40_REPOSITORY_SYNTHESIS": LEGACY / "repo_synthesis",

    # Internal theory and method notes.
    "20_THEORY_AND_METHODS/00_MONITORED_FERMION_REFERENCE_SHEET": LEGACY / "monitored_fermion_reference_sheet",
    "20_THEORY_AND_METHODS/10_EXACT_DW_BENCHMARK_NOTES": LEGACY / "exact_DW_benchmark_notes",
    "20_THEORY_AND_METHODS/20_TANGENT_EDGE_CHANNEL_NOTE": LEGACY / "tangent_edge_channel_note",
    "20_THEORY_AND_METHODS/25_OPEN_SYSTEM_MONITORED_RESPONSE_THEORY": CURRENT / "Paper Methods/open_system_monitored_system_response_theory",
    "20_THEORY_AND_METHODS/27_KAC_MOODY_RENYI_VALIDATION": CURRENT / "Paper Methods/kac_moody_renyi_validation",
    "20_THEORY_AND_METHODS/30_CANONICAL_SOURCE_DOCS": ROOT / "src/docs",
    "20_THEORY_AND_METHODS/40_FORM_FACTOR_DOCS": LEGACY / "form_factor_analysis/docs",
    "20_THEORY_AND_METHODS/50_CHARGE_FLUCTUATION_DOCS": COLAB / "colab_charge_fluctuations/docs",
    "20_THEORY_AND_METHODS/60_ENTANGLEMENT_SCALING_DOCS": COLAB / "colab_large_entanglement_scaling_N20/docs",
    "20_THEORY_AND_METHODS/70_LYAPUNOV_DOCS": COLAB / "colab_lyapunov/docs",
    "20_THEORY_AND_METHODS/80_REGULARIZED_CHOI_DOCS": COLAB / "colab_regularized_choi_transfer_matrix/docs",
    "20_THEORY_AND_METHODS/90_Lyapunov_Spectrum_from_Choi-Covariance.pdf": ROOT / "NOTES/20_THEORY_AND_METHODS/standalone/Lyapunov Spectrum from Choi-Covariance.pdf",
    "20_THEORY_AND_METHODS/91_Regularized_Choi_Covariance.pdf": ROOT / "NOTES/20_THEORY_AND_METHODS/standalone/Regularized_Choi_Covariance.pdf",
    "20_THEORY_AND_METHODS/92_U1_Transfer_Matrix_Formalism_v2.pdf": ROOT / "NOTES/20_THEORY_AND_METHODS/standalone/U(1) Symmetric Transfer Matrix Formalism v2.pdf",
    "20_THEORY_AND_METHODS/93_U1_Transfer_Matrix_Formalism_6.pdf": ROOT / "NOTES/20_THEORY_AND_METHODS/standalone/U(1) Symmetric Transfer Matrix Formalism 6.pdf",
    "20_THEORY_AND_METHODS/94_covariance_note.pdf": ROOT / "NOTES/20_THEORY_AND_METHODS/standalone/covariance_note (2).pdf",
    "20_THEORY_AND_METHODS/95_free_fermion_trajectory_lyapunov_cft.pdf": LEGACY / "markov_transfer_operators/free_fermion_trajectory_lyapunov_cft.pdf",
    "20_THEORY_AND_METHODS/96_markov_transfer_operators_note.pdf": LEGACY / "markov_transfer_operators/markov_transfer_operators_note.pdf",
    "20_THEORY_AND_METHODS/97_u1_symmetric_choi_transfer_matrix_formalism.pdf": LEGACY / "markov_transfer_operators/u1_symmetric_choi_transfer_matrix_formalism.pdf",
    "20_THEORY_AND_METHODS/98_handwritten_perturbative_expansion_about_fixed_point.pdf": LEGACY / "perturbative_expansion/handwritten_perturbative_expansion_about_fixed _point.pdf",

    # Documentation attached to runnable projects/campaigns.
    "30_PROJECT_DOCUMENTATION/00_PRODUCTION_BUNDLES": CURRENT / "final_production_ready_figure_scripts",
    "30_PROJECT_DOCUMENTATION/10_EXPERIMENT_REVIEW": CURRENT / "experiment_review",
    "30_PROJECT_DOCUMENTATION/20_VALIDATION": CURRENT / "validation_campaigns",
    "30_PROJECT_DOCUMENTATION/30_TOPOLOGICAL_FRUSTRATION": CURRENT / "topological_frustration_diagnostics/docs",
    "30_PROJECT_DOCUMENTATION/40_CPU_CFT_EXTRACTION": CURRENT / "cpu_cft_extraction",
    "30_PROJECT_DOCUMENTATION/50_MARKOV_TRANSFER_OPERATORS": LEGACY / "markov_transfer_operators",

    # External or source papers kept at repository root/in temporary references.
    "40_EXTERNAL_PAPERS/2107.10279v2.pdf": ROOT / "NOTES/40_EXTERNAL_PAPERS/sources/2107.10279v2.pdf",
    "40_EXTERNAL_PAPERS/2406.07673v2.pdf": ROOT / "NOTES/40_EXTERNAL_PAPERS/sources/2406.07673v2.pdf",
    "40_EXTERNAL_PAPERS/Non_equilibrium_topological_boundary.pdf": ROOT / "NOTES/40_EXTERNAL_PAPERS/sources/Non_equilibrium_topological_boundary.pdf",
    "40_EXTERNAL_PAPERS/XH10947W_6.pdf": ROOT / "NOTES/40_EXTERNAL_PAPERS/sources/XH10947W (6).pdf",
    "40_EXTERNAL_PAPERS/TEMP_REFERENCE_PAPERS": ROOT / ".tmp/ref_papers",

    # Useful but not part of the current scientific contract.
    "90_LEGACY/2406.txt": ROOT / "NOTES/40_EXTERNAL_PAPERS/sources/2406.txt",
    "90_LEGACY/choi_transfer_matrix_chatlog.md": ROOT / "NOTES/90_LEGACY/root_imports/choi_transfer_matrix_chatlog.md",
    "90_LEGACY/paper_style_draft.pdf": ROOT / "NOTES/90_LEGACY/root_imports/paper_style_draft.pdf",
    "90_LEGACY/top_last_protocol.txt": ROOT / "NOTES/90_LEGACY/root_imports/top_last_protocol.txt",
    "90_LEGACY/translation_invariance_monitor_log.md": ROOT / "NOTES/90_LEGACY/root_imports/Log of translation-invariance of monitor.md",
}

OPTIONAL_CURATED = {"40_EXTERNAL_PAPERS/TEMP_REFERENCE_PAPERS"}


def is_document(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.is_symlink() or relative.parts[0] in EXCLUDED_ROOTS:
        return False
    if path.name.endswith("Notes.bib") or path.name.endswith(".orig"):
        return False
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.suffix.lower() != ".pdf":
        return False
    if "figures" in relative.parts or "results" in relative.parts:
        return False
    contextual = {part.lower() for part in relative.parts}
    project_root = relative.parts[0]
    if (
        len(relative.parts) >= 3
        and relative.parts[0] == "00_WORKSPACE"
        and relative.parts[1] in {"CURRENT", "COLAB", "LEGACY", "EXTERNAL"}
    ):
        project_root = relative.parts[2]
    return (
        len(relative.parts) == 1
        or project_root in PDF_PROJECT_ROOTS
        or bool(contextual & {"docs", "doc", "note", "notes", "references", "ref_papers"})
    )


def link(relative_link: Path, target: Path) -> None:
    destination = NOTES / relative_link
    destination.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, destination.parent)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink():
            raise FileExistsError(f"nonmatching notes path: {destination}")
        if destination.resolve() == target.resolve():
            return
        # Every symlink created here is generated navigation.  Retarget it after an
        # intentional canonical move, but never replace a regular user file/directory.
        destination.unlink()
    destination.symlink_to(relative_target)


def managed_documents() -> list[Path]:
    """Return physical documents intentionally housed inside NOTES itself."""
    result = []
    for directory in MANAGED_DOCUMENT_ROOTS:
        for path in directory.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            if path.name.endswith("Notes.bib") or path.name.endswith(".orig"):
                continue
            if path.suffix.lower() in TEXT_EXTENSIONS | {".pdf"}:
                result.append(path)
    return sorted(result)


def remove_stale_source_links(expected: set[Path]) -> None:
    """Remove obsolete generated mirror links while preserving physical files."""
    mirror = NOTES / "ALL_BY_SOURCE"
    for path in sorted(mirror.rglob("*"), reverse=True):
        if path.is_symlink() and path not in expected:
            path.unlink()
    for path in sorted(mirror.rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink():
            try:
                path.rmdir()
            except OSError:
                pass


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    raise AssertionError


def main() -> None:
    missing = [
        str(path)
        for relative, path in CURATED.items()
        if relative not in OPTIONAL_CURATED and not path.exists() and not path.is_symlink()
    ]
    if missing:
        raise FileNotFoundError("missing curated note targets:\n" + "\n".join(missing))

    available_curated = {
        relative: path
        for relative, path in CURATED.items()
        if path.exists() or path.is_symlink()
    }

    external_documents = sorted(path for path in ROOT.rglob("*") if is_document(path))
    documents = external_documents + managed_documents()
    expected_source_links = {
        NOTES / "ALL_BY_SOURCE" / document.relative_to(ROOT)
        for document in external_documents
    }
    remove_stale_source_links(expected_source_links)
    for relative_link, target in available_curated.items():
        link(Path(relative_link), target)
    for document in external_documents:
        link(Path("ALL_BY_SOURCE") / document.relative_to(ROOT), document)

    rows = sorted(
        ((path.stat().st_mtime, path.relative_to(ROOT), path.stat().st_size) for path in documents),
        reverse=True,
    )
    lines = [
        "# Notes recency index",
        "",
        "Generated from canonical documentation files. LaTeX build debris, generated figures,",
        "results, caches, and `*Notes.bib` files are excluded.",
        "",
        "| Modified | Canonical document | Size |",
        "|---|---|---:|",
    ]
    for modified, relative, size in rows:
        timestamp = datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")
        lines.append(f"| {timestamp} | `{relative.as_posix()}` | {human_bytes(size)} |")
    (NOTES / "RECENCY_INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"documents={len(documents)}")
    print(f"curated_links={len(available_curated)}")
    print(f"notes_root={NOTES}")


if __name__ == "__main__":
    main()
