#!/usr/bin/env python3
"""Build the single-source Jupyter notebook for the legacy evidence figure atlas.

This helper only serializes documented notebook cells.  It does not load scientific
data or generate figures; those operations occur exclusively when the resulting
``legacy_evidence_figure_atlas.ipynb`` is executed.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "legacy_evidence_figure_atlas.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


cells = [
    md(
        r"""
        # Legacy numerical evidence: a BPJ-style twenty-four-figure atlas

        This notebook is the **single plotting source** for the companion RevTeX
        working note.  It turns the strongest numerical findings identified in
        `numerical_campaign_legacy_working.tex` into twenty-four clean multi-panel figures.

        The scientific rule is conservative: every plotted point is read from an
        explicit repository artifact.  In particular, Result 5 now reads a compact,
        provenance-complete reduction of the original exact-domain-wall calculation and
        the historical 100-trajectory covariance archive; it no longer transcribes fit
        coefficients from a rendered figure.  The exact local-Chern heat map remains
        identified as a rendered legacy calibration because its numerical field was not
        separately archived.

        The visual grammar follows Bhuiyan--Pan--Jian (BPJ), *Physical Review Research* 8,
        023147 (2026): Times-style serif text, Computer-Modern mathematics, red triangles,
        green squares, blue circles, dotted/dashed/solid size ordering, boxed axes,
        inward ticks, compact legends, and panel letters outside the axes.

        Result 24 is a compact deterministic reconstruction with the audited continuous-
        time solver; no stochastic circuit is simulated here, no covariance history is
        saved, and no path under `erroneous_gpu_stuff/` is read.
        """
    ),
    code(
        r"""
        # Runtime / CPU allocation.  Edit CPU_RANGE directly or set
        # LEGACY_FIGURE_CPU_RANGE (examples: "0-7", "2,4,6", or "auto:4").
        import os
        from pathlib import Path

        CPU_RANGE = os.environ.get("LEGACY_FIGURE_CPU_RANGE", "auto:4")

        def parse_cpu_range(spec, allowed):
            allowed = sorted(allowed)
            if spec.startswith("auto:"):
                return allowed[: max(1, int(spec.split(":", 1)[1]))]
            requested = []
            for token in spec.split(","):
                token = token.strip()
                if not token:
                    continue
                if "-" in token:
                    lo, hi = map(int, token.split("-", 1))
                    requested.extend(range(lo, hi + 1))
                else:
                    requested.append(int(token))
            selected = sorted(set(requested).intersection(allowed))
            if not selected:
                raise ValueError(f"CPU_RANGE={spec!r} selects no allowed CPU from {allowed}")
            return selected

        ALLOWED_CPUS = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        SELECTED_CPUS = parse_cpu_range(CPU_RANGE, ALLOWED_CPUS)
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, SELECTED_CPUS)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(len(SELECTED_CPUS))

        def find_repo_root(start):
            start = Path(start).resolve()
            for candidate in (start, *start.parents):
                if (candidate / "PROJECT_ADMIN/REPO_POLICY.md").exists():
                    return candidate
                if ((candidate / "AGENTS.md").exists()
                        and (candidate / "experiment_review").is_dir()
                        and (candidate / "form_factor_analysis").is_dir()):
                    return candidate
            raise FileNotFoundError("Could not locate the repository root")

        ROOT = find_repo_root(Path.cwd())
        EXPERIMENT_REVIEW = ROOT / "00_WORKSPACE/CURRENT/experiment_review"
        OUT = EXPERIMENT_REVIEW / "legacy_evidence_figure_atlas/figures"
        OUT.mkdir(parents=True, exist_ok=True)
        print({"CPU_RANGE": CPU_RANGE, "selected_cpus": SELECTED_CPUS, "repo_root": str(ROOT), "figure_dir": str(OUT)})
        """
    ),
    md(
        r"""
        ## Shared BPJ visual grammar and provenance helpers

        Colors and markers deliberately reproduce BPJ's recurring visual encoding:

        - red upward triangles and dotted lines for the first/smallest member;
        - green squares and dashed lines for the second/intermediate member;
        - blue circles and solid lines for the third/largest member;
        - black dashed lines for theoretical targets or transition locations.

        The helpers below also reject excluded paths, save both vector PDF and 300-dpi PNG,
        and record a source manifest for every figure.
        """
    ),
    code(
        r"""
        import json
        import math
        import re
        from datetime import datetime, timezone

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display

        BPJ_RED = "#D92725"
        BPJ_GREEN = "#2CA02C"
        BPJ_BLUE = "#1F77B4"
        BPJ_BLACK = "#000000"
        BPJ_GRAY = "#6B6B6B"
        BPJ_LIGHT_BLUE = "#8FC1E3"
        BPJ_ORANGE = "#F08050"
        BPJ_STYLES = [
            dict(color=BPJ_RED, marker="^", linestyle=":"),
            dict(color=BPJ_GREEN, marker="s", linestyle="--"),
            dict(color=BPJ_BLUE, marker="o", linestyle="-"),
        ]

        mpl.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "font.family": "serif",
                "font.serif": ["Times", "Nimbus Roman", "Times New Roman", "Liberation Serif"],
                "mathtext.fontset": "cm",
                "font.size": 8.0,
                "axes.labelsize": 8.0,
                "axes.titlesize": 8.0,
                "xtick.labelsize": 7.0,
                "ytick.labelsize": 7.0,
                "legend.fontsize": 6.6,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.0,
                "lines.markersize": 4.0,
                "xtick.direction": "in",
                "ytick.direction": "in",
                "xtick.major.width": 0.7,
                "ytick.major.width": 0.7,
                "xtick.major.size": 3.2,
                "ytick.major.size": 3.2,
                "axes.spines.top": True,
                "axes.spines.right": True,
                "legend.frameon": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            }
        )

        PHYSICAL_PROJECT_PREFIXES = {
            "experiment_review": "00_WORKSPACE/CURRENT/experiment_review",
            "topological_frustration_diagnostics": "00_WORKSPACE/CURRENT/topological_frustration_diagnostics",
            "validation_campaigns": "00_WORKSPACE/CURRENT/validation_campaigns",
            "tangent_cocycle_flux_snapshot": "00_WORKSPACE/CURRENT/tangent_cocycle_flux_snapshot",
            "tangent_edge_channel": "00_WORKSPACE/CURRENT/tangent_edge_channel",
            "cpu_cft_extraction": "00_WORKSPACE/CURRENT/cpu_cft_extraction",
            "final_production_ready_figure_scripts": "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts",
            "mean_channel_lindblad_cpu_campaign": "00_WORKSPACE/CURRENT/mean_channel_lindblad_cpu_campaign",
            "prxq_draft": "00_WORKSPACE/CURRENT/prxq_draft",
            "colab_charge_fluctuations": "00_WORKSPACE/COLAB/colab_charge_fluctuations",
            "colab_large_entanglement_scaling_N20": "00_WORKSPACE/COLAB/colab_large_entanglement_scaling_N20",
            "colab_lyapunov": "00_WORKSPACE/COLAB/colab_lyapunov",
            "colab_no_feedback_alpha_sweep_transfer": "00_WORKSPACE/COLAB/colab_no_feedback_alpha_sweep_transfer",
            "colab_partial_post-select": "00_WORKSPACE/COLAB/colab_partial_post-select",
            "colab_regularized_choi_transfer_matrix": "00_WORKSPACE/COLAB/colab_regularized_choi_transfer_matrix",
            "colab_small_system_testing": "00_WORKSPACE/COLAB/colab_small_system_testing",
            "choi_covariance_cpu": "00_WORKSPACE/LARGE_RESULTS/choi_covariance_cpu",
            "dw_convergence": "00_WORKSPACE/LARGE_RESULTS/dw_convergence",
            "experiments": "00_WORKSPACE/LARGE_RESULTS/experiments",
            "lyapunov_analysis_v2": "00_WORKSPACE/LARGE_RESULTS/lyapunov_analysis_v2",
            "exact_DW_benchmark_notes": "00_WORKSPACE/LEGACY/exact_DW_benchmark_notes",
            "form_factor_analysis": "00_WORKSPACE/LEGACY/form_factor_analysis",
            "markov_transfer_operators": "00_WORKSPACE/LEGACY/markov_transfer_operators",
            "repo_synthesis": "00_WORKSPACE/LEGACY/repo_synthesis",
            "sample_average_testing": "00_WORKSPACE/LEGACY/sample_average_testing",
            "summary_of_results": "00_WORKSPACE/LEGACY/summary_of_results",
        }

        def physical_relative(relative):
            relative = Path(relative)
            prefix = PHYSICAL_PROJECT_PREFIXES.get(relative.parts[0])
            return Path(prefix, *relative.parts[1:]) if prefix else relative

        def require(relative):
            path = (ROOT / physical_relative(relative)).resolve()
            if "erroneous_gpu_stuff" in path.parts:
                raise RuntimeError(f"Excluded evidence path: {path}")
            if not path.exists():
                raise FileNotFoundError(path)
            return path

        def panel(ax, letter, title=None):
            ax.text(-0.16, 1.08, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
            if title:
                ax.set_title(title, pad=4)
            ax.tick_params(direction="in")
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        def unit_cell_grid(ax, nx=None, ny=None):
            # Match the exact-DW benchmark heat-map grammar: integer cell centers,
            # labeled every two sites, with white boundaries at half-integers.
            if nx is not None:
                ax.set_xticks(np.arange(0, nx, 2))
                ax.set_xticks(np.arange(nx + 1) - 0.5, minor=True)
            if ny is not None:
                ax.set_yticks(np.arange(0, ny, 2))
                ax.set_yticks(np.arange(ny + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.35)
            ax.tick_params(which="minor", bottom=False, left=False)

        def save_figure(fig, stem, sources, evidence_class, claim, format_mode="rebuilt from machine-readable artifacts"):
            pdf = OUT / f"{stem}.pdf"
            png = OUT / f"{stem}.png"
            fig.savefig(pdf, bbox_inches="tight")
            fig.savefig(png, dpi=300, bbox_inches="tight")
            FIGURE_MANIFEST["figures"].append(
                {
                    "stem": stem,
                    "pdf": str(pdf.relative_to(ROOT)),
                    "png": str(png.relative_to(ROOT)),
                    "sources": [str(require(s).relative_to(ROOT)) for s in sources],
                    "evidence_class": evidence_class,
                    "claim": claim,
                    "format_mode": format_mode,
                }
            )
            display(fig)
            plt.close(fig)

        def sem(values, axis=0):
            values = np.asarray(values, float)
            return np.nanstd(values, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(values), axis=axis))

        def binary_entropy(p):
            p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
            return -p * np.log(p) - (1 - p) * np.log(1 - p)

        FIGURE_MANIFEST = {
            "schema_version": 1,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "notebook": "00_WORKSPACE/CURRENT/experiment_review/legacy_evidence_figure_atlas/legacy_evidence_figure_atlas.ipynb",
            "style_reference": "00_WORKSPACE/LEGACY/repo_synthesis/PRR_paper.pdf",
            "style": {
                "font": "Times-compatible serif with Computer Modern mathematics",
                "colors": {"red": BPJ_RED, "green": BPJ_GREEN, "blue": BPJ_BLUE, "black": BPJ_BLACK},
                "markers": ["triangle", "square", "circle"],
                "line_styles": ["dotted", "dashed", "solid"],
            },
            "figures": [],
        }
        DIAGNOSTICS = {}
        print("BPJ visual grammar configured; excluded directory reads are guarded.")
        """
    ),
    md(
        r"""
        # Group I — Deterministic target and adaptive topology

        ## Result 1 — The exact wall realizes the clean $c=1$ chiral target

        **Premise.**  Before interpreting the adaptive data, the same geometry and
        estimators must recover the known class-A edge: $c=1$, an $r^{-2}$ squared
        correlator, opposite wall velocities, and negligible transverse hybridization.

        **Evidence class.** Completed, audited deterministic B0 campaign.  These panels
        are the strongest evidence in the atlas and are not stochastic-circuit results.
        """
    ),
    code(
        r"""
        B0 = "experiment_review/b0_exact_domain_wall/results/20260816_191957"
        b0_sources = [
            f"{B0}/manifest.json",
            f"{B0}/processed/tables/entropy_fits.csv",
            f"{B0}/processed/tables/correlator_fits.csv",
            f"{B0}/processed/tables/modular_velocities.csv",
            f"{B0}/processed/tables/response_velocities.csv",
            f"{B0}/processed/tables/twist_flow_counts.csv",
            f"{B0}/processed/tables/width_selection.csv",
        ]
        b0_manifest = json.loads(require(b0_sources[0]).read_text())
        b0_entropy = pd.read_csv(require(b0_sources[1]))
        b0_corr = pd.read_csv(require(b0_sources[2]))
        b0_mod = pd.read_csv(require(b0_sources[3]))
        b0_resp = pd.read_csv(require(b0_sources[4]))
        b0_twist = pd.read_csv(require(b0_sources[5]))
        b0_width = pd.read_csv(require(b0_sources[6]))

        b0_e = b0_entropy.query("nx == 20 and q == 1 and quantity == 'full_strip'").sort_values(["construction", "ny"])
        b0_c = b0_corr.query("nx == 20 and curve == 'wall_0' and model == 'chord_power' and endpoint_shift == 0").sort_values(["construction", "ny"])
        b0_m = b0_mod.query("nx == 20 and ny == 48 and is_primary == True and source_width == 3")
        b0_p = b0_resp.query("nx == 20 and ny == 48")[["construction", "wall_index", "wavefront_velocity_stderr"]]
        b0_m = b0_m.merge(b0_p, on=["construction", "wall_index"], how="left", validate="one_to_one")
        b0_w = b0_width.sort_values("nx")
        print("B0 manifest keys:", sorted(b0_manifest)[:12])
        display(b0_e[["construction", "ny", "c_estimate", "r2"]])
        display(b0_m[["construction", "wall_index", "physical_velocity", "modular_velocity", "minimum_wall_retention"]])
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.2))
        construction_style = {
            "coupled": dict(color=BPJ_RED, marker="^", linestyle="-", label="coupled interface"),
            "hard_exterior": dict(color=BPJ_BLUE, marker="o", linestyle="--", label="hard exterior"),
        }

        ax = axes[0, 0]
        for construction, group in b0_e.groupby("construction"):
            ax.plot(group.ny, group.c_estimate, **construction_style[construction], mfc="white")
        ax.axhline(1, color=BPJ_BLACK, lw=0.8, ls=":")
        ax.set(xlabel=r"$N_y$", ylabel=r"$c_1$", ylim=(0.9999, 1.0006))
        ax.legend(loc="upper right")
        panel(ax, "a", r"entropy coefficient")

        ax = axes[0, 1]
        for construction, group in b0_c.groupby("construction"):
            style = construction_style[construction].copy()
            ax.errorbar(group.ny, group.parameter_1, yerr=group.stderr_1, capsize=1.8, mfc="white", **style)
        ax.axhline(2, color=BPJ_BLACK, lw=0.8, ls=":")
        ax.set(xlabel=r"$N_y$", ylabel=r"$\beta$ in $C_G\sim\ell_y^{-\beta}$", ylim=(1.98, 2.11))
        panel(ax, "b", r"wall correlator")

        ax = axes[1, 0]
        offsets = {"coupled": -0.05, "hard_exterior": 0.05}
        for construction, group in b0_m.groupby("construction"):
            group = group.sort_values("wall_index")
            style = construction_style[construction]
            short_label = {"coupled": "coupled", "hard_exterior": "hard"}[construction]
            x = np.array([-1.0, 1.0]) + offsets[construction]
            ax.errorbar(x, group.modular_velocity, yerr=group.velocity_stderr, color=style["color"], marker=style["marker"],
                        ls="none", capsize=2, mfc="white", label=f"{short_label}: modular")
            ax.errorbar(x, group.physical_velocity, yerr=group.wavefront_velocity_stderr,
                        color=style["color"], marker=style["marker"], ls=style["linestyle"], capsize=2,
                        mfc=style["color"], label=f"{short_label}: physical")
        ax.axhline(0, color=BPJ_BLACK, lw=0.7)
        ax.set_xticks([-1, 1], ["left wall", "right wall"])
        ax.set_ylabel(r"signed velocity")
        ax.legend(ncol=2, loc="upper left", handletextpad=0.35, columnspacing=0.55, fontsize=5.9)
        panel(ax, "c", r"two clocks, opposite signs")

        ax = axes[1, 1]
        ax.semilogy(b0_w.nx, b0_w.ratio_coupled, color=BPJ_RED, marker="^", ls=":", mfc="white", label="coupled")
        ax.semilogy(b0_w.nx, b0_w.ratio_hard_exterior, color=BPJ_BLUE, marker="o", ls="-", mfc="white", label="hard exterior")
        ax.axhline(float(b0_w.threshold.iloc[0]), color=BPJ_BLACK, ls="--", lw=0.8, label="mass gate")
        ax.axvline(20, color=BPJ_BLACK, ls=":", lw=0.7)
        ax.text(0.04, 0.07, r"accepted $N_x^*=20$" + "\n" + r"$\nu_{\rm flow}=(-1,+1)$" + "\n" + r"$\xi_\perp=(0.296,0.223)$",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
        ax.set(xlabel=r"$N_x$ at $N_y=48$", ylabel=r"$m/(2\pi v/N_y)$", ylim=(5e-9, 0.3))
        ax.legend(loc="upper right")
        panel(ax, "d", r"transverse convergence")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_01"] = {
            "c1_Nx20_Ny48": b0_e.query("ny == 48").set_index("construction").c_estimate.to_dict(),
            "beta_Nx20_Ny48": b0_c.query("ny == 48").set_index("construction").parameter_1.to_dict(),
            "twist_flow_Nx20_Ny48": b0_twist.query("nx == 20 and ny == 48").signed_flow.tolist(),
        }
        save_figure(fig, "result_01_exact_wall_benchmark", b0_sources, "completed deterministic audit",
                    "Exact wall recovers c=1, beta about 2.07, opposite physical/modular velocities, unit signed flow, and an accepted width N_x=20.")
        """
    ),
    md(
        r"""
        ## Result 2 — The adaptive wall carries a sharply localized slow tangent sector

        **Premise.**  A topological boundary should be dynamically distinguished from a
        uniform bulk.  The relevant legacy statement is finite-size softening plus spatial
        localization—not a proven thermodynamic zero gap.

        **Evidence class.** Two independent canonical perfect-correction campaigns:
        provenance-complete CPU data at $N_y=20,22,24$ and broader GPU data at
        $N_y=30,40,50$.
        """
    ),
    code(
        r"""
        CPU_TANGENT = "topological_frustration_diagnostics/results/campaigns/20260813_121340_Nx20_perfect_Ny20_22_24"
        GPU_TANGENT = "colab_lyapunov/gpu_data/lyapunov_spectra/campaigns/N20_Ny30-50_nsh1_a1-1_S25_cyclesNy"
        tangent_sources = [f"{GPU_TANGENT}/campaign_manifest.json"]

        cpu_rows = []
        for ny in (20, 22, 24):
            for geometry, dirname in (("wall", f"N20x{ny}_dw_dwtrunc1_nsh1_perfect_correction"),
                                      ("uniform", f"N20x{ny}_uniform_dwtrunc0_nsh1_perfect_correction")):
                rel = f"{CPU_TANGENT}/spectral_Ny{ny}_perfect/spectral/{dirname}/scalar_metrics.csv"
                diag_rel = rel.rsplit("/", 1)[0] + "/spectral_diagnostics.npz"
                tangent_sources += [rel, diag_rel]
                row = pd.read_csv(require(rel)).iloc[0].copy()
                z = np.load(require(diag_rel), allow_pickle=True)
                gap_samples = np.abs(z["lyapunov_final_value"])
                interface_index = list(z["region_names"]).index("interface")
                interface_samples = z["lyapunov_mode_region_weight"][:, 0, interface_index]
                row["geometry_plot"] = geometry
                row["lyapunov_gap_final_sem"] = sem(gap_samples)
                row["lyapunov_interface_weight_sem"] = sem(interface_samples)
                cpu_rows.append(row)
        cpu_tangent = pd.DataFrame(cpu_rows)

        gpu_rows, gpu_profiles = [], {}
        for ny in (30, 40, 50):
            for geometry, dw, a2 in (("wall", 1, 30), ("uniform", 0, 1)):
                run = f"N20x{ny}_DW{dw}_dwtrunc{dw}_a2-{a2}_nsh1_perfect_correction"
                scalar_rel = f"{GPU_TANGENT}/runs/{run}/scalar_metrics.csv"
                vector_rel = f"{GPU_TANGENT}/runs/{run}/lyapunov_min_abs_vector.npz"
                tangent_sources += [scalar_rel, vector_rel]
                z = np.load(require(vector_rel), allow_pickle=True)
                values = np.abs(z["lyapunov_min_abs_value"])
                vectors = z["lyapunov_min_abs_vector"]
                profile_samples = (np.abs(vectors.reshape(len(vectors), ny, 20, 2)) ** 2).sum(axis=(1, 3))
                profile = profile_samples.mean(axis=0)
                gpu_profiles[(ny, geometry)] = {"mean": profile, "sem": sem(profile_samples, axis=0)}
                interface_samples = profile_samples[:, [5, 15]].sum(axis=1)
                gpu_rows.append({"Ny": ny, "geometry_plot": geometry, "gap_mean": values.mean(), "gap_sem": sem(values),
                                 "interface_weight": interface_samples.mean() if geometry == "wall" else np.nan,
                                 "interface_sem": sem(interface_samples) if geometry == "wall" else np.nan})
        gpu_tangent = pd.DataFrame(gpu_rows)
        display(cpu_tangent[["Ny", "geometry_plot", "lyapunov_gap_final_mean", "lyapunov_interface_weight_mean"]])
        display(gpu_tangent)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        geom_style = {
            "wall": dict(color=BPJ_RED, marker="^", linestyle="-", label="wall"),
            "uniform": dict(color=BPJ_BLUE, marker="o", linestyle="--", label="uniform"),
        }

        ax = axes[0, 0]
        for geometry, group in cpu_tangent.groupby("geometry_plot"):
            ax.errorbar(group.Ny, group.lyapunov_gap_final_mean, yerr=group.lyapunov_gap_final_sem,
                        capsize=2, mfc="white", **geom_style[geometry])
        ax.set_yscale("log")
        ax.set(xlabel=r"$N_y$", ylabel=r"finite-time tangent gap")
        ax.legend()
        panel(ax, "a", "CPU, $T=2N_y$")

        ax = axes[0, 1]
        for geometry, group in gpu_tangent.groupby("geometry_plot"):
            ax.errorbar(group.Ny, group.gap_mean, yerr=group.gap_sem, capsize=2, mfc="white", **geom_style[geometry])
        ax.set(xlabel=r"$N_y$", ylabel=r"finite-time tangent gap")
        ax.set_yscale("log")
        ax.legend()
        panel(ax, "b", "GPU, $T=N_y$")

        ax = axes[1, 0]
        for i, ny in enumerate((30, 40, 50)):
            style = BPJ_STYLES[i]
            profile = gpu_profiles[(ny, "wall")]
            ax.plot(np.arange(20), profile["mean"], label=rf"$N_y={ny}$", mfc="white", **style)
            ax.fill_between(np.arange(20), profile["mean"] - profile["sem"], profile["mean"] + profile["sem"],
                            color=style["color"], alpha=.10, lw=0)
        profile = gpu_profiles[(50, "uniform")]
        ax.plot(np.arange(20), profile["mean"], color=BPJ_BLACK, ls="--", lw=0.8, label="uniform, 50")
        ax.fill_between(np.arange(20), profile["mean"] - profile["sem"], profile["mean"] + profile["sem"],
                        color=BPJ_BLACK, alpha=.08, lw=0)
        ax.axvline(5, color=BPJ_GRAY, lw=0.6, ls=":")
        ax.axvline(15, color=BPJ_GRAY, lw=0.6, ls=":")
        ax.set(xlabel=r"transverse coordinate $x$", ylabel=r"normalized slow-mode weight")
        ax.legend(ncol=2)
        panel(ax, "c", "GPU slow-mode profile")

        ax = axes[1, 1]
        wall_cpu = cpu_tangent.query("geometry_plot == 'wall'")
        wall_gpu = gpu_tangent.query("geometry_plot == 'wall'")
        ax.errorbar(wall_cpu.Ny, wall_cpu.lyapunov_interface_weight_mean,
                    yerr=wall_cpu.lyapunov_interface_weight_sem, color=BPJ_RED, marker="^", ls=":",
                    capsize=2, mfc="white", label="CPU interface window")
        ax.errorbar(wall_gpu.Ny, wall_gpu.interface_weight, yerr=wall_gpu.interface_sem,
                    color=BPJ_BLUE, marker="o", ls="-", capsize=2, mfc="white", label="GPU wall columns")
        ax.axhline(0.70, color=BPJ_BLACK, ls="--", lw=0.8, label="70% reference")
        ax.set(xlabel=r"$N_y$", ylabel=r"interface weight", ylim=(0.70, 1.015))
        ax.legend()
        panel(ax, "d", "spatial localization")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_02"] = {"cpu": cpu_tangent.to_dict("records"), "gpu": gpu_tangent.to_dict("records")}
        save_figure(fig, "result_02_tangent_slow_sector", tangent_sources, "canonical CPU and GPU finite-time precursors",
                    "Wall tangent gaps are far below uniform controls and the selected slow sector is sharply wall localized.")
        """
    ),
    md(
        r"""
        ## Result 3 — Max-mix purification leaves a wall-localized late residual

        **Premise.**  A maximally mixed initial covariance rapidly purifies in the bulk,
        while a topological wall retains a slow, spatially concentrated residual.

        **Evidence class.** $S=100$ GPU purification histories, a canonical CPU
        topological/trivial mass control, and a smaller entropy-contour campaign.  The
        datasets have different sample counts and are juxtaposed, not pooled.
        """
    ),
    code(
        r"""
        PUR_GPU = "colab_charge_fluctuations/gpu_data/purification_dynamics_maxmix/campaigns/N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny"
        PUR_CPU_TABLE = "colab_charge_fluctuations/analysis_outputs/purification_charge_sharpening_alpha_sweep_cpu/N16_alpha-fine21_nsh1_dwtrunc1_init-maxmix_S10_cycles-2Ny/tables/steady_state_summary.csv"
        PUR_CONTOUR_BASE = "colab_small_system_testing/analysis_outputs/purification_entropy_contours_maxmix_cpu"
        purification_sources = [f"{PUR_GPU}/campaign_manifest.json", PUR_CPU_TABLE]

        pur_frames = []
        for ny in (30, 40, 50):
            rel = f"{PUR_GPU}/runs/N20x{ny}_nsh1_init-maxmix_perfect_correction/scalar_metrics.csv"
            purification_sources.append(rel)
            pur_frames.append(pd.read_csv(require(rel)))
        pur_gpu = pd.concat(pur_frames, ignore_index=True)
        pur_stats = pur_gpu.groupby(["Ny", "cycle_label"]).agg(
            entropy_mean=("total_entropy", "mean"), entropy_sem=("total_entropy", sem),
            variance_mean=("total_charge_variance", "mean"), variance_sem=("total_charge_variance", sem),
            chern_mean=("real_space_chern", "mean")
        ).reset_index()

        pur_cpu = pd.read_csv(require(PUR_CPU_TABLE)).query("protocol == 'perfect_correction'")
        pur_cpu = pur_cpu[pur_cpu.alpha_topological_region.isin([1.0, 3.0])]
        contour_frames = []
        for trunc in (0, 1):
            rel = f"{PUR_CONTOUR_BASE}/N16x32_C100_dwtrunc{trunc}/y_integrated_entropy_curves.csv"
            purification_sources.append(rel)
            frame = pd.read_csv(require(rel))
            frame["dw_truncation"] = trunc
            contour_frames.append(frame)
        pur_contour = pd.concat(contour_frames, ignore_index=True)
        display(pur_stats.groupby("Ny").tail(1))
        display(pur_cpu[["Ny", "alpha_topological_region", "total_entropy_mean", "total_charge_variance_mean"]])
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        for ax, mean_col, sem_col, ylabel, letter, title in [
            (axes[0, 0], "entropy_mean", "entropy_sem", r"$S_{\rm tot}$", "a", "max-mix entropy"),
            (axes[0, 1], "variance_mean", "variance_sem", r"$\mathrm{Tr}[C(\mathbf{1}-C)]$", "b", "intrinsic charge variance"),
        ]:
            for i, ny in enumerate((30, 40, 50)):
                group = pur_stats.query("Ny == @ny")
                style = BPJ_STYLES[i]
                ax.plot(group.cycle_label / ny, group[mean_col], label=rf"$N_y={ny}$", markevery=max(1, len(group)//8), mfc="white", **style)
                ax.fill_between(group.cycle_label / ny, np.maximum(group[mean_col] - group[sem_col], 1e-14), group[mean_col] + group[sem_col],
                                color=style["color"], alpha=0.12, lw=0)
            ax.set(xlabel=r"$t/N_y$", ylabel=ylabel, yscale="log")
            ax.legend()
            panel(ax, letter, title)

        ax = axes[1, 0]
        for i, ny in enumerate((16, 24, 32)):
            style = BPJ_STYLES[i]
            q = pur_cpu.query("Ny == @ny").sort_values("alpha_topological_region")
            for observable, error_col, ls, open_marker in (("total_entropy_mean", "total_entropy_std", "-", False),
                                                            ("total_charge_variance_mean", "total_charge_variance_std", "--", True)):
                y = np.maximum(q[observable].to_numpy(), 1e-14)
                err = q[error_col].to_numpy() / np.sqrt(q.samples.to_numpy())
                ax.errorbar(q.alpha_topological_region, y, yerr=np.vstack([np.minimum(err, .999 * y), err]),
                            color=style["color"], marker=style["marker"], ls=ls, capsize=1.5,
                            mfc="white" if open_marker else style["color"], label=rf"$N_y={ny}$, " + (r"$S$" if observable.startswith("total_entropy") else r"$F^{\rm q}$"))
        ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel="final residual", yscale="log", xticks=[1, 3])
        ax.legend(ncol=2, handletextpad=0.35, columnspacing=0.6)
        panel(ax, "c", "topological versus trivial")

        ax = axes[1, 1]
        for trunc, color, marker, label in ((1, BPJ_RED, "^", "support terminated"), (0, BPJ_BLUE, "o", "untruncated")):
            q = pur_contour.query("case_id == 'N16x32_nsh1_perfect_correction' and cycle == 100 and dw_truncation == @trunc")
            ax.semilogy(q.x, q.y_integrated_entropy, color=color, marker=marker, ls="-" if trunc else "--", mfc="white", label=label)
        ax.axvline(5, color=BPJ_BLACK, ls=":", lw=0.8)
        ax.text(0.98, 0.93, "94.25% on both wall columns\n(half-profile shown)", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set(xlabel=r"$x$ on retained half", ylabel=r"$\sum_y s(x,y)$")
        ax.legend(loc="lower right")
        panel(ax, "d", "late entropy profile")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_03"] = {"gpu_final": pur_stats.groupby("Ny").tail(1).to_dict("records")}
        save_figure(fig, "result_03_purification_wall_residual", purification_sources,
                    "canonical ensemble dynamics plus smaller spatial precursor",
                    "The topological wall purifies much more slowly than a trivial control and the late entropy is concentrated at the wall.")
        """
    ),
    md(
        r"""
        ## Result 4 — Adaptive-state entropy approaches the two-wall $1/3$ slope

        **Premise.**  For a $c=1$ complex fermion, the two-wall full-strip entropy has
        log-chord slope $c/3=1/3$.

        **Evidence class.** Canonical GPU entropy campaign at $S=100$ and fixed depth
        $C=50$, supplemented by the broader $S=10$, $C=N_y/2$ sequence at
        $N_y=40,60,80,100,120$.  Regression errors across interval sizes diagnose curve
        shape; they are not Born-ensemble confidence intervals.  Empirical points are
        never connected: lines denote only the log-chord fit or the $c=1$ target.
        """
    ),
    code(
        r"""
        ENT_BASE = "colab_large_entanglement_scaling_N20/gpu_data/pure_state_entanglement_slope_vs_system_size/runs"
        ENT_CYCLE = "colab_large_entanglement_scaling_N20/gpu_data/pure_state_entanglement_slope_vs_cycle/runs/N20x40_nsh1_dwtrunc1_C100_S10/slope_vs_cycle.csv"
        ENT_TIME_BASE = "colab_large_entanglement_scaling_N20/gpu_data/pure_state_entanglement_slope_vs_system_size_time_scaling/runs"
        entropy_sources = [ENT_CYCLE]
        entropy_curves, entropy_fits = {}, []
        for ny in (30, 40, 50):
            run = f"N20x{ny}_nsh1_dwtrunc1_C50_S100"
            curve_rel = f"{ENT_BASE}/{run}/entropy_curves.csv"
            fit_rel = f"{ENT_BASE}/{run}/fit_rows.csv"
            entropy_sources += [curve_rel, fit_rel]
            entropy_curves[ny] = pd.read_csv(require(curve_rel))
            entropy_fits.append(pd.read_csv(require(fit_rel)).iloc[0])
        entropy_fits = pd.DataFrame(entropy_fits).sort_values("Ny")
        entropy_time_scaled = []
        for ny in (40, 60, 80, 100, 120):
            fit_rel = f"{ENT_TIME_BASE}/N20x{ny}_nsh1_dwtrunc1_C{ny // 2}_S10/fit_rows.csv"
            entropy_sources.append(fit_rel)
            entropy_time_scaled.append(pd.read_csv(require(fit_rel)).iloc[0])
        entropy_time_scaled = pd.DataFrame(entropy_time_scaled).sort_values("Ny")
        entropy_cycle = pd.read_csv(require(ENT_CYCLE))
        display(entropy_fits[["Ny", "slope", "slope_err", "r2", "percent_diff"]])
        display(entropy_time_scaled[["Ny", "cycle", "slope", "slope_err", "r2", "percent_diff"]])
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        ax = axes[0, 0]
        ny = 50
        curve = entropy_curves[ny].query("Ay >= 1")
        fit = entropy_fits.query("Ny == @ny").iloc[0]
        X = curve.log_sin_pi_Ay_over_Ny + np.log(ny / np.pi)
        mask = curve.Ay.between(fit.Ay_fit_min, fit.Ay_fit_max)
        xfit = X[mask]
        b_chord = fit.intercept - fit.slope * np.log(ny / np.pi)
        ax.axvspan(float(xfit.min()), float(xfit.max()), color="0.5", alpha=.18,
                   label=rf"fit window $A_y={int(fit.Ay_fit_min)}\ldots{int(fit.Ay_fit_max)}$")
        ax.errorbar(X, curve.entropy_mean, yerr=curve.entropy_sem, linestyle="none",
                    color=BPJ_BLUE, marker="o", mfc="white", capsize=1.2, label="GPU data")
        # Fit only inside the shaded window, but display that fitted model over the
        # complete measured x-domain so the slope is easy to compare with every point.
        xline = np.linspace(float(X.min()), float(X.max()), 300)
        ax.plot(xline, fit.slope * xline + b_chord, color=BPJ_BLACK, ls="--", lw=1.1,
                label=rf"fit (gray window): $c_{{\rm eff}}={3 * fit.slope:.4f}$")
        ax.set(xlabel=r"$\log[(N_y/\pi)\sin(\pi A_y/N_y)]$", ylabel=r"$\overline{S(A_y)}$")
        ax.legend(loc="upper left")
        panel(ax, "a", r"$N_y=50$, $S=100$, $C=50$")

        ax = axes[0, 1]
        ax.errorbar(entropy_fits.Ny, 3 * entropy_fits.slope, yerr=3 * entropy_fits.slope_err,
                    linestyle="none", color=BPJ_BLUE, marker="o", mfc="white", capsize=2,
                    label=r"$S=100$, fixed $C=50$")
        ax.errorbar(entropy_time_scaled.Ny, 3 * entropy_time_scaled.slope,
                    yerr=3 * entropy_time_scaled.slope_err, linestyle="none", color=BPJ_RED,
                    marker="^", mfc="white", capsize=2, label=r"$S=10$, $C=N_y/2$")
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=0.8, label=r"$c=1$")
        ax.set(xlabel=r"circumference $N_y$", ylabel=r"$c_{\rm eff}=3m_1$", ylim=(0.995, 1.095))
        ax.legend(loc="upper right", fontsize=6.0)
        panel(ax, "b", "GPU central-charge sequence")

        ax = axes[1, 0]
        ax.semilogy(entropy_fits.Ny, 1 - entropy_fits.r2, linestyle="none", color=BPJ_BLUE,
                    marker="o", mfc="white", label=r"$S=100$, fixed $C=50$")
        ax.semilogy(entropy_time_scaled.Ny, 1 - entropy_time_scaled.r2, linestyle="none",
                    color=BPJ_RED, marker="^", mfc="white", label=r"$S=10$, $C=N_y/2$")
        ax.set(xlabel=r"$N_y$", ylabel=r"$1-R^2$")
        panel(ax, "c", "fit-shape residual")

        ax = axes[1, 1]
        q = entropy_cycle.query("cycle >= 5")
        shown = q.iloc[::5]
        ax.errorbar(shown.cycle, 3 * shown.slope, yerr=3 * shown.slope_err, linestyle="none",
                    color=BPJ_BLUE, marker="o", capsize=1.2, mfc="white")
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=0.8)
        ax.set(xlabel="cycle", ylabel=r"$c_{\rm eff}(C)=3m_1(C)$",
               ylim=(.90, max(1.95, 3 * q.query("cycle >= 10").slope.max())))
        panel(ax, "d", r"$20\times40$, $S=10$")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_04"] = {
            "fixed_C50_S100": entropy_fits.to_dict("records"),
            "time_scaled_S10": entropy_time_scaled.to_dict("records"),
        }
        save_figure(fig, "result_04_entropy_log_chord", entropy_sources, "canonical GPU precursor plus small large-circumference pilot",
                    "Direct GPU central-charge estimates c_eff=3m remain near one across both the controlled S=100 sequence and the broader time-scaled S=10 sequence.")
        """
    ),
    md(
        r"""
        ## Result 5 — Tripartite information resolves one wall versus two walls

        For three adjacent intervals $A,B,C$ around the periodic $y$ direction, define
        $$
        I_2(A:C)=S_A+S_C-S_{AC},\qquad
        I_3(A:B:C)=S_A+S_B+S_C-S_{AB}-S_{AC}-S_{BC}+S_{ABC}.
        $$
        For the full $x$ width, the plotted positive quantity is the conditional mutual
        information
        $$
        \Delta I\equiv I_2-I_3=S_{AB}+S_{BC}-S_B-S_{ABC}=I(A:C\mid B)\geq0.
        $$
        Thus the requested full-width $I_3-I_2$ is simply $-\Delta I$.  To resolve a
        wall, first evaluate each Gaussian entropy on the full-$x$ subsystem and then sum
        its entanglement contour over an $x$ window $X$:
        $$
        S_R^{(X)}=\sum_{x\in X}\sum_{y\in R}s_R(x,y),\qquad
        \Delta I^{(X)}=S_{AB}^{(X)}+S_{BC}^{(X)}-S_B^{(X)}-S_{ABC}^{(X)}.
        $$
        When $X$ is the full width, $S_R^{(X)}=S_R$ and this is the genuine CMI above.
        For a proper wall window, $\Delta I^{(X)}$ is an algebraic contour contribution,
        not itself a von Neumann entropy or CMI; strong subadditivity does not protect its
        sign.  The wall contributions plotted here are empirically positive.

        On a circle of circumference
        $N_y$, with chord length $d(\ell)=(N_y/\pi)\sin(\pi\ell/N_y)$, the cross ratio and
        fit coordinate are
        $$
        x=\frac{d(L_A)d(L_C)}{d(L_A+L_B)d(L_B+L_C)},\qquad
        z=\log\!\frac{1}{1-x},\qquad \Delta I=mz+b.
        $$
        A single chiral wall predicts $m=c/6$; summing both walls predicts $m=c/3$.

        **Fit-window sensitivity.** The archived baseline is not a post-hoc residual cut:
        it uses every stored row satisfying
        $$
        L_A+L_C\leq L_B\leq N_y-2(L_A+L_C).
        $$
        Equivalently, both $B$ and the complementary interval
        $D=N_y-(L_A+L_B+L_C)$ are at least $L_A+L_C$.  The diagnostics additionally scan
        the symmetry-preserving restriction $L_B,D\geq q$.  A candidate is reported only
        when at least three distinct $z$ values remain.  This is a sensitivity analysis,
        not a rule for selecting whichever window gives $c$ closest to one.

        **Reduction order.** Every entropy is evaluated on the full-$x$ subsystem.  The
        entanglement contour is restricted in $x$ only after this evaluation.  For the
        stochastic archive, the translated origins are averaged *inside each trajectory*,
        and only then are the 100 trajectory curves averaged.  Point bars are trajectory
        SEMs.  Fit errors are ordinary regression errors across the displayed $z$ rows;
        the diagnostics also retain a fixed-seed whole-trajectory bootstrap.  Translation
        averaging gives $\overline S_{AB}=\overline S_{BC}$, so the implemented identity
        is $\Delta I=2\overline S_{AB}-\overline S_B-\overline S_{ABC}$; no $S_A$ or
        disconnected-$AC$ contour is needed.

        **Evidence class.** The exact $20\times40$ curves are a deterministic calibration.
        The historical $12\times31$, $C=100$, $S=100$, unwindowed-controller curves are a
        reusable estimator pilot with a protocol-dependent coefficient bias.  No partial
        post-selection data, stale $S=250$ notebook output, reduced-strip estimator, or
        digitized figure values enter this result.
        """
    ),
    code(
        r"""
        # Result-5 cache controls.  Ordinary atlas runs validate and preload the compact
        # reduction.  Set LEGACY_ATLAS_REBUILD_TMI=1 (or =archive) to make one
        # sequential pass through the authoritative 43.64-GB source archive and select
        # only each trajectory's final cycle.  A historical extracted .npy predates the
        # archive and is deliberately not accepted as a rebuild source.
        import hashlib
        import sys
        import tempfile
        import zipfile
        from concurrent.futures import ThreadPoolExecutor
        from numpy.lib import format as npformat
        from threadpoolctl import threadpool_limits

        TMI_CACHE_SCHEMA = 2
        TMI_BOOTSTRAP_SEED = 20260817
        TMI_BOOTSTRAP_RESAMPLES = 2000
        TMI_REBUILD_MODE = os.environ.get("LEGACY_ATLAS_REBUILD_TMI", "0").strip().lower()
        TMI_REBUILD_WORKERS = max(1, min(int(os.environ.get("LEGACY_ATLAS_TMI_WORKERS", len(SELECTED_CPUS))), len(SELECTED_CPUS)))

        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))

        TMI_CACHE_REL = "experiment_review/legacy_evidence_figure_atlas/data/result_05_tripartite_information_reduction.npz"
        TMI_META_REL = "experiment_review/legacy_evidence_figure_atlas/data/result_05_tripartite_information_reduction.json"
        TMI_ARCHIVE_REL = (
            "cache/G_history_samples/N12x31/"
            "N12x31_C100_S100_nshNone_DW1_init-default_n_a0.5_"
            "seq-dw_symmetric_exclNone_ps0_pst0_fm0_pc0_markov_circuit.npz"
        )
        TMI_ARCHIVE_MANIFEST_REL = "PROJECT_ADMIN/archive_manifests/cache_G_history_samples.json"
        TMI_EXACT_NOTEBOOK_REL = "notebooks/exact_DW_GS_benchmark.ipynb"
        TMI_CLASS_SOURCE_REL = "src/fgtn/classA_U1FGTN.py"
        TMI_LEGACY_REDUCER_REL = "sample_average_testing/sample_average_entanglement_contour_fit_testing_v2.py"
        TMI_EXACT_RENDER_REL = "exact_DW_benchmark_notes/DW_GS_Benchmark_Figs/MI_linear_fits.png"
        TMI_STOCH_RENDER_L4_REL = (
            "sample_average_testing/figs/"
            "N12x31_C100_S100_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_"
            "exclNone_ps0_pst0_fm0_pc0_markov_circuit_xsum_window_integrated_contour_and_mi_fits_v2.pdf"
        )
        TMI_STOCH_RENDER_L3_REL = (
            "sample_average_testing/figs/"
            "N12x31_C100_S100_nshNone_DW1_init-default_n_a0.5_seq-dw_symmetric_"
            "exclNone_ps0_pst0_fm0_pc0_markov_circuit_sample_average_y0_average_integrated_contour_and_mi_fits_v2.pdf"
        )

        def tmi_sha256(path, chunk_bytes=8 * 1024 * 1024):
            digest = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(chunk_bytes), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def tmi_fit_line(x, y):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            coefficients, covariance = np.polyfit(x, y, 1, cov=True)
            slope, intercept = map(float, coefficients)
            slope_error = float(np.sqrt(covariance[0, 0]))
            prediction = slope * x + intercept
            ss_res = float(np.square(y - prediction).sum())
            ss_tot = float(np.square(y - y.mean()).sum())
            return slope, slope_error, intercept, 1.0 - ss_res / ss_tot

        def tmi_chord(length, circumference):
            return circumference / np.pi * np.sin(np.pi * length / circumference)

        def tmi_cross_ratio(l_a, l_b, l_c, circumference):
            return (
                tmi_chord(l_a, circumference) * tmi_chord(l_c, circumference)
                / (tmi_chord(l_a + l_b, circumference) * tmi_chord(l_b + l_c, circumference))
            )

        def tmi_geometry(l_a, l_c, l_b_values, circumference):
            l_b_values = np.asarray(l_b_values, int)
            cross_ratio = np.asarray(
                [tmi_cross_ratio(l_a, int(l_b), l_c, circumference) for l_b in l_b_values], float
            )
            return cross_ratio, np.log(1.0 / (1.0 - cross_ratio))

        def tmi_subsystem_indices(nx, ny, y_values):
            y_values = np.asarray(y_values, int) % ny
            return np.concatenate(
                [np.arange(2 * nx, dtype=int) + 2 * nx * int(y) for y in y_values]
            )

        def tmi_contour_x_sums(G, nx, ny, y_values):
            y_values = np.asarray(y_values, int) % ny
            indices = tmi_subsystem_indices(nx, ny, y_values)
            restricted = np.asarray(G[np.ix_(indices, indices)], np.complex128)
            occupation = 0.5 * (np.eye(len(indices), dtype=np.complex128) + restricted)
            occupation = 0.5 * (occupation + occupation.conj().T)
            eigenvalues, eigenvectors = np.linalg.eigh(occupation)
            eigenvalues = np.clip(np.real_if_close(eigenvalues).real, 1e-12, 1.0 - 1e-12)
            entropy_eigenvalues = -(
                eigenvalues * np.log(eigenvalues)
                + (1.0 - eigenvalues) * np.log(1.0 - eigenvalues)
            )
            diagonal = np.einsum(
                "ik,k,ik->i", eigenvectors, entropy_eigenvalues, eigenvectors.conj(), optimize=True
            ).real
            # The subsystem index is i=mu+2*x+2*Nx*y.  Sum over orbitals and all
            # selected y sites, retaining the x-resolved post-contour contribution.
            return diagonal.reshape(len(y_values), nx, 2).sum(axis=(0, 2))

        def tmi_contiguous_x_sums(G, nx, ny, length, average_y0):
            origins = range(ny) if average_y0 else (0,)
            values = [
                tmi_contour_x_sums(G, nx, ny, (y0 + np.arange(length)) % ny)
                for y0 in origins
            ]
            return np.mean(values, axis=0)

        _TMI_EXACT_G = None
        _TMI_STOCH_GHIST = None
        _TMI_STOCH_WALL_X = None

        def tmi_exact_pattern_worker(task):
            kind, value = task
            return task, tmi_contiguous_x_sums(_TMI_EXACT_G, 20, 40, int(value), False)

        def tmi_stochastic_trajectory_worker(sample_index):
            source = _TMI_STOCH_GHIST
            G = np.asarray(source[sample_index, -1] if source.ndim == 4 else source[sample_index])
            nx, ny = 12, 31
            configurations = ((4, np.arange(8, 16, dtype=int)), (3, np.arange(6, 20, dtype=int)))
            needed_lengths = set()
            for l_a, l_b_values in configurations:
                needed_lengths.add(l_a)
                for l_b in l_b_values:
                    needed_lengths.update((int(l_b), int(l_a + l_b), int(2 * l_a + l_b)))
            contiguous = {
                length: tmi_contiguous_x_sums(G, nx, ny, length, True)
                for length in sorted(needed_lengths)
            }
            curves = {}
            for l_a, l_b_values in configurations:
                delta_x = []
                for l_b in l_b_values:
                    # Delta I=S_AB+S_BC-S_B-S_ABC.  Translation averaging makes
                    # S_AB=S_BC trajectory by trajectory.
                    delta_x.append(
                        2.0 * contiguous[int(l_a + l_b)]
                        - contiguous[int(l_b)]
                        - contiguous[int(2 * l_a + l_b)]
                    )
                delta_x = np.asarray(delta_x)
                wall = delta_x[:, _TMI_STOCH_WALL_X].sum(axis=1)
                full = delta_x.sum(axis=1)
                curves[l_a] = (wall, full)
            return int(sample_index), curves[4][0], curves[4][1], curves[3][1]

        def tmi_parallel_map(function, tasks, workers):
            tasks = list(tasks)
            if workers == 1:
                with threadpool_limits(limits=1, user_api="blas"):
                    return [function(task) for task in tasks]
            # The 45-GB stochastic history is memory mapped and must remain shared.
            # Threads avoid both a second copy and the unsafe fork-after-BLAS pattern;
            # each dense eigensolve is single-threaded, while independent tasks run in
            # parallel at this outer level.
            with threadpool_limits(limits=1, user_api="blas"):
                with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                    return list(pool.map(function, tasks))

        def tmi_read_npy_header(stream):
            version = npformat.read_magic(stream)
            if version == (1, 0):
                return npformat.read_array_header_1_0(stream)
            if version in {(2, 0), (3, 0)}:
                return npformat.read_array_header_2_0(stream)
            raise ValueError(f"Unsupported NPY version {version}")

        def tmi_stream_final_states(archive_path, destination):
            # One forward decompression pass: discard cycles 0..C-1 and retain only C.
            with zipfile.ZipFile(archive_path) as archive, archive.open("G_hist.npy") as stream:
                shape, fortran_order, dtype_description = tmi_read_npy_header(stream)
                if fortran_order or len(shape) != 4:
                    raise ValueError(f"Expected C-order (S,T,N,N) G_hist; received {shape}")
                samples, time_steps, nlayer, nlayer_2 = map(int, shape)
                if nlayer != nlayer_2:
                    raise ValueError(f"Non-square covariance history: {shape}")
                dtype = np.dtype(dtype_description)
                output = npformat.open_memmap(
                    destination, mode="w+", dtype=dtype, shape=(samples, nlayer, nlayer)
                )
                slice_bytes = nlayer * nlayer * dtype.itemsize
                discard_bytes = (time_steps - 1) * slice_bytes
                scratch = bytearray(8 * 1024 * 1024)
                for sample in range(samples):
                    remaining = discard_bytes
                    while remaining:
                        count = stream.readinto(memoryview(scratch)[: min(len(scratch), remaining)])
                        if not count:
                            raise EOFError("Unexpected end of G_hist while skipping pre-final cycles")
                        remaining -= count
                    final_buffer = bytearray(slice_bytes)
                    view = memoryview(final_buffer)
                    received = 0
                    while received < slice_bytes:
                        count = stream.readinto(view[received:])
                        if not count:
                            raise EOFError("Unexpected end of G_hist while reading a final covariance")
                        received += count
                    output[sample] = np.frombuffer(final_buffer, dtype=dtype).reshape(nlayer, nlayer)
                    if (sample + 1) % 10 == 0 or sample + 1 == samples:
                        print(
                            f"[TMI archive] retained final covariance {sample + 1}/{samples}",
                            flush=True,
                        )
                output.flush()
            return Path(destination)

        def tmi_bootstrap_slopes(trajectory_curves, z_values, resamples, seed):
            trajectory_curves = np.asarray(trajectory_curves, float)
            generator = np.random.default_rng(seed)
            sample_count = trajectory_curves.shape[0]
            slopes = np.empty((resamples, trajectory_curves.shape[1]), float)
            for bootstrap_index in range(resamples):
                selection = generator.integers(0, sample_count, size=sample_count)
                means = trajectory_curves[selection].mean(axis=0)
                for curve_index, curve in enumerate(means):
                    slopes[bootstrap_index, curve_index] = np.polyfit(z_values[curve_index], curve, 1)[0]
            return slopes

        def rebuild_tmi_cache(mode):
            from fgtn.classA_U1FGTN import classA_U1FGTN

            cache_path = EXPERIMENT_REVIEW / "legacy_evidence_figure_atlas/data/result_05_tripartite_information_reduction.npz"
            metadata_path = EXPERIMENT_REVIEW / "legacy_evidence_figure_atlas/data/result_05_tripartite_information_reduction.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Exact 20x40 calibration, fixed y0=0.  Exact y-translation invariance is
            # used only to reuse shifted contiguous spectra; the estimator remains the
            # full-subsystem contour followed by the x-window sum.
            print("[TMI rebuild] constructing exact 20x40 covariance", flush=True)
            exact_model = classA_U1FGTN(20, 40, nshell=None, DW=True, alpha_1=1.0, alpha_2=30.0)
            # Reproduce the source notebook's legacy Nx//5 slab exactly.  The current
            # class default uses a wider slab, so relying on its default here would be a
            # protocol mismatch even though the Hamiltonian builder itself is unchanged.
            exact_legacy_w = 20 // 5
            exact_legacy_x0 = 20 // 2 - exact_legacy_w
            exact_legacy_x1 = 20 // 2 + exact_legacy_w + 1
            exact_alpha = np.full((20, 40), 30.0, dtype=np.complex128)
            exact_alpha[exact_legacy_x0:exact_legacy_x1, :] = 1.0
            exact_model.DW_loc = [exact_legacy_x0, exact_legacy_x1 - 1]
            with threadpool_limits(limits=min(8, TMI_REBUILD_WORKERS), user_api="blas"):
                exact_g = np.asarray(
                    exact_model.G_CI_domain_wall(
                        periodic=True, alpha=exact_alpha, triv_region_local_mode=False
                    ), np.complex128
                )
            exact_l_b = np.arange(6, 29, dtype=int)
            exact_x, exact_z = tmi_geometry(3, 3, exact_l_b, 40)
            exact_tasks = [("contiguous", length) for length in range(6, 35)]
            global _TMI_EXACT_G
            _TMI_EXACT_G = exact_g
            print(f"[TMI rebuild] reducing {len(exact_tasks)} exact interval lengths", flush=True)
            exact_patterns = dict(
                tmi_parallel_map(tmi_exact_pattern_worker, exact_tasks, min(TMI_REBUILD_WORKERS, 24))
            )
            exact_delta_x = np.asarray([
                2.0 * exact_patterns[("contiguous", int(l_b + 3))]
                - exact_patterns[("contiguous", int(l_b))]
                - exact_patterns[("contiguous", int(l_b + 6))]
                for l_b in exact_l_b
            ])
            exact_wall_x = [
                np.asarray([(int(exact_model.DW_loc[0]) + offset) % 20 for offset in (-1, 0, 1)]),
                np.asarray([(int(exact_model.DW_loc[1]) + offset) % 20 for offset in (-1, 0, 1)]),
            ]
            exact_delta = np.vstack(
                [exact_delta_x[:, xs].sum(axis=1) for xs in exact_wall_x]
                + [exact_delta_x.sum(axis=1)]
            )
            exact_fits = np.asarray([tmi_fit_line(exact_z, curve) for curve in exact_delta])
            expected_exact = np.asarray([0.166967668, 0.166967668, 0.334232464])
            if not np.allclose(exact_fits[:, 0], expected_exact, atol=5e-10, rtol=0):
                raise AssertionError({"exact_slopes": exact_fits[:, 0], "expected": expected_exact})
            print(f"[TMI rebuild] exact slopes {exact_fits[:, 0].tolist()}", flush=True)

            # Historical stochastic pilot.  Every rebuild performs exactly one
            # sequential scan of the authoritative archive.  The older extracted NPY
            # is not used because its timestamp predates the archived source.
            archive_path = require(TMI_ARCHIVE_REL)
            temporary_directory = tempfile.TemporaryDirectory(prefix="atlas_tmi_finals_")
            final_path = tmi_stream_final_states(
                archive_path, Path(temporary_directory.name) / "final_covariances.npy"
            )
            stochastic_history = np.load(final_path, mmap_mode="r")
            if stochastic_history.shape not in ((100, 51, 744, 744), (100, 744, 744)):
                raise ValueError(f"Unexpected stochastic covariance shape {stochastic_history.shape}")
            global _TMI_STOCH_GHIST, _TMI_STOCH_WALL_X
            _TMI_STOCH_GHIST = stochastic_history
            stochastic_legacy_w = 12 // 5
            stochastic_legacy_x0 = 12 // 2 - stochastic_legacy_w
            _TMI_STOCH_WALL_X = np.asarray(
                [(stochastic_legacy_x0 + offset) % 12 for offset in (-1, 0, 1)], int
            )
            print(
                f"[TMI rebuild] reducing 100 final trajectories with "
                f"{min(TMI_REBUILD_WORKERS, 100)} workers",
                flush=True,
            )
            trajectory_rows = tmi_parallel_map(
                tmi_stochastic_trajectory_worker, range(100), TMI_REBUILD_WORKERS
            )
            trajectory_rows.sort(key=lambda row: row[0])
            stochastic_l4_trajectory = np.asarray([[row[1], row[2]] for row in trajectory_rows])
            stochastic_l3_full_trajectory = np.asarray([row[3] for row in trajectory_rows])
            temporary_directory.cleanup()

            stochastic_l4_l_b = np.arange(8, 16, dtype=int)
            stochastic_l3_l_b = np.arange(6, 20, dtype=int)
            stochastic_l4_x, stochastic_l4_z = tmi_geometry(4, 4, stochastic_l4_l_b, 31)
            stochastic_l3_x, stochastic_l3_z = tmi_geometry(3, 3, stochastic_l3_l_b, 31)
            stochastic_l4_mean = stochastic_l4_trajectory.mean(axis=0)
            stochastic_l4_sem = stochastic_l4_trajectory.std(axis=0, ddof=1) / np.sqrt(100)
            stochastic_l3_full_mean = stochastic_l3_full_trajectory.mean(axis=0)
            stochastic_l3_full_sem = stochastic_l3_full_trajectory.std(axis=0, ddof=1) / np.sqrt(100)
            stochastic_l4_fits = np.asarray(
                [tmi_fit_line(stochastic_l4_z, curve) for curve in stochastic_l4_mean]
            )
            stochastic_l3_full_fit = np.asarray(tmi_fit_line(stochastic_l3_z, stochastic_l3_full_mean))
            rendered_reference_slopes = np.asarray([0.2487, 0.5112, 0.5251])
            reconstructed_slopes = np.asarray(
                [stochastic_l4_fits[0, 0], stochastic_l4_fits[1, 0], stochastic_l3_full_fit[0]]
            )
            rendered_reference_match = np.isclose(
                reconstructed_slopes, rendered_reference_slopes, atol=5e-5, rtol=0
            )
            print(
                "[TMI rebuild] stochastic slopes "
                f"L4={stochastic_l4_fits[:, 0].tolist()}, "
                f"L3-full={float(stochastic_l3_full_fit[0])}",
                flush=True,
            )

            combined_trajectories = np.empty((100, 3, 14), float)
            combined_z = np.empty((3, 14), float)
            combined_trajectories[:, :2, :] = np.nan
            combined_z[:2, :] = np.nan
            combined_trajectories[:, 0, :8] = stochastic_l4_trajectory[:, 0]
            combined_trajectories[:, 1, :8] = stochastic_l4_trajectory[:, 1]
            combined_trajectories[:, 2, :] = stochastic_l3_full_trajectory
            combined_z[0, :8] = stochastic_l4_z
            combined_z[1, :8] = stochastic_l4_z
            combined_z[2, :] = stochastic_l3_z
            # Bootstrap each estimator on its native z grid; resampling always selects
            # whole trajectories and therefore preserves all within-trajectory z covariance.
            generator = np.random.default_rng(TMI_BOOTSTRAP_SEED)
            bootstrap_slopes = np.empty((TMI_BOOTSTRAP_RESAMPLES, 3), float)
            for bootstrap_index in range(TMI_BOOTSTRAP_RESAMPLES):
                selection = generator.integers(0, 100, size=100)
                for curve_index in (0, 1):
                    curve = stochastic_l4_trajectory[selection, curve_index].mean(axis=0)
                    bootstrap_slopes[bootstrap_index, curve_index] = np.polyfit(
                        stochastic_l4_z, curve, 1
                    )[0]
                curve = stochastic_l3_full_trajectory[selection].mean(axis=0)
                bootstrap_slopes[bootstrap_index, 2] = np.polyfit(stochastic_l3_z, curve, 1)[0]
            bootstrap_ci = np.percentile(bootstrap_slopes, [2.5, 97.5], axis=0).T
            bootstrap_se = bootstrap_slopes.std(axis=0, ddof=1)

            fit_rows = [
                ("exact", "left wall", 3, exact_fits[0], 1.0 / 6.0, 6.0),
                ("exact", "right wall", 3, exact_fits[1], 1.0 / 6.0, 6.0),
                ("exact", "full width", 3, exact_fits[2], 1.0 / 3.0, 3.0),
                ("stochastic", "one wall", 4, stochastic_l4_fits[0], 1.0 / 6.0, 6.0),
                ("stochastic", "full width", 4, stochastic_l4_fits[1], 1.0 / 3.0, 3.0),
                ("stochastic", "full width", 3, stochastic_l3_full_fit, 1.0 / 3.0, 3.0),
            ]
            fit_evidence = np.asarray([row[0] for row in fit_rows], dtype="U16")
            fit_window = np.asarray([row[1] for row in fit_rows], dtype="U16")
            fit_l_a = np.asarray([row[2] for row in fit_rows], int)
            fit_values = np.asarray([row[3] for row in fit_rows], float)
            fit_target = np.asarray([row[4] for row in fit_rows], float)
            fit_c_factor = np.asarray([row[5] for row in fit_rows], float)

            np.savez_compressed(
                cache_path,
                schema_version=np.asarray(TMI_CACHE_SCHEMA, np.int64),
                exact_l_b=exact_l_b,
                exact_cross_ratio=exact_x,
                exact_z=exact_z,
                exact_delta_i=exact_delta,
                exact_window_names=np.asarray(["left wall", "right wall", "full width"], dtype="U16"),
                exact_fit=exact_fits,
                stochastic_l4_l_b=stochastic_l4_l_b,
                stochastic_l4_cross_ratio=stochastic_l4_x,
                stochastic_l4_z=stochastic_l4_z,
                stochastic_l4_trajectory_delta_i=stochastic_l4_trajectory,
                stochastic_l4_mean_delta_i=stochastic_l4_mean,
                stochastic_l4_sem_delta_i=stochastic_l4_sem,
                stochastic_l4_fit=stochastic_l4_fits,
                stochastic_l3_l_b=stochastic_l3_l_b,
                stochastic_l3_cross_ratio=stochastic_l3_x,
                stochastic_l3_z=stochastic_l3_z,
                stochastic_l3_full_trajectory_delta_i=stochastic_l3_full_trajectory,
                stochastic_l3_full_mean_delta_i=stochastic_l3_full_mean,
                stochastic_l3_full_sem_delta_i=stochastic_l3_full_sem,
                stochastic_l3_full_fit=stochastic_l3_full_fit,
                stochastic_bootstrap_slopes=bootstrap_slopes,
                stochastic_bootstrap_slope_se=bootstrap_se,
                stochastic_bootstrap_slope_ci95=bootstrap_ci,
                stochastic_rendered_reference_slopes=rendered_reference_slopes,
                stochastic_rendered_reference_match=rendered_reference_match,
                stochastic_rendered_reference_difference=reconstructed_slopes - rendered_reference_slopes,
                fit_evidence=fit_evidence,
                fit_window=fit_window,
                fit_l_a=fit_l_a,
                fit_values=fit_values,
                fit_target=fit_target,
                fit_c_factor=fit_c_factor,
            )

            archive_manifest = json.loads(require(TMI_ARCHIVE_MANIFEST_REL).read_text())
            archive_suffix = str(Path(TMI_ARCHIVE_REL).relative_to("cache/G_history_samples"))
            archive_row = next(row for row in archive_manifest["files"] if row["path"] == archive_suffix)
            source_rows = []
            for source in (TMI_EXACT_NOTEBOOK_REL, TMI_CLASS_SOURCE_REL, TMI_LEGACY_REDUCER_REL):
                path = require(source)
                source_rows.append(
                    {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": tmi_sha256(path)}
                )
            source_rows.append(
                {
                    "path": str(require(TMI_ARCHIVE_REL).relative_to(ROOT)),
                    "bytes": int(archive_row["bytes"]),
                    "sha256": archive_row["sha256"],
                    "fingerprint_source": str(require(TMI_ARCHIVE_MANIFEST_REL).relative_to(ROOT)),
                }
            )
            metadata = {
                "schema_version": TMI_CACHE_SCHEMA,
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "cache_npz": str(cache_path.relative_to(ROOT)),
                "cache_npz_sha256": tmi_sha256(cache_path),
                "definition": "For the full x width, Delta I = I2 - I3 = I(A:C|B) >= 0; I3-I2 = -Delta I",
                "windowed_definition": "Delta I^(X) is the post-contour x-window contribution S_AB^(X)+S_BC^(X)-S_B^(X)-S_ABC^(X)",
                "positivity_scope": "Strong subadditivity protects only the full-width CMI; proper-window contributions are empirically positive here but have no independent sign theorem",
                "entropy_estimator": "full-x restricted covariance; x window applied only to the final entanglement-contour sum",
                "exact_protocol": {
                    "Nx": 20, "Ny": 40,
                    "source_notebook_arguments": {"alpha_1": 30.0, "alpha_2": 1.0},
                    "resolved_mass_profile": {
                        "alpha_in": 1.0, "alpha_out": 30.0,
                        "topological_x_inclusive": list(range(exact_legacy_x0, exact_legacy_x1)),
                    },
                    "compatibility_override": "reproduces the source-era Nx//5 slab and alpha_2-inside semantics instead of the current class default",
                    "L_A": 3, "L_C": 3, "y0": 0, "periodic_y": True,
                    "windows": {"left": exact_wall_x[0].tolist(), "right": exact_wall_x[1].tolist(), "full": list(range(20))},
                },
                "stochastic_protocol": {
                    "Nx": 12, "Ny": 31, "cycles": 100, "trajectories": 100,
                    "n_shell": None, "domain_wall": True, "init": "default",
                    "feedback": "nonperfect stochastic", "sequence": "dw_symmetric",
                    "postselect": False, "partial_postselect": False,
                    "L4_windows": {"wall": _TMI_STOCH_WALL_X.tolist(), "full": list(range(12))},
                    "averaging_order": "y0 mean within each trajectory, then mean/SEM over 100 trajectories",
                    "rebuild_source": "one sequential pass through the authoritative compressed G_hist.npy member; only final-cycle covariances are retained temporarily",
                },
                "fit": {"coordinate": "z=log(1/(1-x))", "window": "all stored rows", "model": "Delta I=m z+b"},
                "uncertainty": {
                    "points": "SEM over 100 trajectory-level y0-averaged curves",
                    "fit": "unweighted OLS regression standard error over z rows",
                    "bootstrap": "whole-trajectory resampling with replacement",
                    "bootstrap_seed": TMI_BOOTSTRAP_SEED,
                    "bootstrap_resamples": TMI_BOOTSTRAP_RESAMPLES,
                },
                "exclusions": ["partial post-selection", "stale S=250 output", "reduced-strip estimator", "digitization"],
                "sources": source_rows,
                "array_shapes": {
                    "exact_delta_i": list(exact_delta.shape),
                    "stochastic_l4_trajectory_delta_i": list(stochastic_l4_trajectory.shape),
                    "stochastic_l3_full_trajectory_delta_i": list(stochastic_l3_full_trajectory.shape),
                    "stochastic_bootstrap_slopes": list(bootstrap_slopes.shape),
                },
                "fit_rows": [
                    {
                        "evidence": row[0], "window": row[1], "L_A": int(row[2]),
                        "slope": float(row[3][0]), "slope_error": float(row[3][1]),
                        "intercept": float(row[3][2]), "r2": float(row[3][3]),
                        "target": float(row[4]), "c_factor": float(row[5]),
                    }
                    for row in fit_rows
                ],
                "rendered_legacy_comparison": {
                    "reference_slopes": rendered_reference_slopes.tolist(),
                    "reconstructed_slopes": reconstructed_slopes.tolist(),
                    "difference": (reconstructed_slopes - rendered_reference_slopes).tolist(),
                    "matches_archived_rounding": rendered_reference_match.tolist(),
                    "comparison_tolerance": 5e-5,
                },
            }
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            print(f"[TMI rebuild] wrote {cache_path} and {metadata_path}", flush=True)
            return cache_path, metadata_path

        if TMI_REBUILD_MODE in {"1", "true", "yes", "archive"}:
            rebuild_tmi_cache("archive")
        elif TMI_REBUILD_MODE not in {"0", "false", "no", ""}:
            raise ValueError("LEGACY_ATLAS_REBUILD_TMI must be 0, 1, or archive")
        """
    ),
    code(
        r"""
        # Validate and preload the compact reduction.  This is the ordinary fast path.
        tmi_cache_path = require(TMI_CACHE_REL)
        tmi_metadata_path = require(TMI_META_REL)
        tmi_metadata = json.loads(tmi_metadata_path.read_text())
        assert tmi_metadata["schema_version"] == TMI_CACHE_SCHEMA
        assert tmi_sha256(tmi_cache_path) == tmi_metadata["cache_npz_sha256"]
        assert tmi_metadata["stochastic_protocol"]["trajectories"] == 100
        assert tmi_metadata["stochastic_protocol"]["postselect"] is False
        assert tmi_metadata["stochastic_protocol"]["partial_postselect"] is False
        assert tmi_metadata["uncertainty"]["bootstrap_seed"] == TMI_BOOTSTRAP_SEED
        assert tmi_metadata["uncertainty"]["bootstrap_resamples"] == TMI_BOOTSTRAP_RESAMPLES
        assert all("erroneous_gpu_stuff" not in row["path"] for row in tmi_metadata["sources"])
        assert all("partial_post-select" not in row["path"] for row in tmi_metadata["sources"])
        for source_row in tmi_metadata["sources"]:
            source_path = require(source_row["path"])
            assert source_path.stat().st_size == source_row["bytes"]
            if source_path.stat().st_size < 100_000_000:
                assert tmi_sha256(source_path) == source_row["sha256"]

        with np.load(tmi_cache_path, allow_pickle=False) as payload:
            tmi_cache = {key: payload[key] for key in payload.files}
        assert int(tmi_cache["schema_version"]) == TMI_CACHE_SCHEMA
        assert tmi_cache["exact_delta_i"].shape == (3, 23)
        assert tmi_cache["stochastic_l4_trajectory_delta_i"].shape == (100, 2, 8)
        assert tmi_cache["stochastic_l3_full_trajectory_delta_i"].shape == (100, 14)
        assert np.isfinite(tmi_cache["exact_delta_i"]).all()
        assert np.isfinite(tmi_cache["stochastic_l4_trajectory_delta_i"]).all()

        tmi_fit_table = pd.DataFrame(
            {
                "evidence": tmi_cache["fit_evidence"],
                "window": tmi_cache["fit_window"],
                "L_A=L_C": tmi_cache["fit_l_a"],
                "slope": tmi_cache["fit_values"][:, 0],
                "slope_error": tmi_cache["fit_values"][:, 1],
                "intercept": tmi_cache["fit_values"][:, 2],
                "r2": tmi_cache["fit_values"][:, 3],
                "target": tmi_cache["fit_target"],
                "c_factor": tmi_cache["fit_c_factor"],
            }
        )
        tmi_fit_table["c_tmi"] = tmi_fit_table.c_factor * tmi_fit_table.slope
        tmi_fit_table["c_tmi_error"] = tmi_fit_table.c_factor * tmi_fit_table.slope_error
        exact_ratio = tmi_fit_table.iloc[2].slope / tmi_fit_table.iloc[:2].slope.mean()
        stochastic_ratio = tmi_fit_table.iloc[4].slope / tmi_fit_table.iloc[3].slope

        def tmi_symmetric_window_scan(
            evidence, geometry, l_b, z, curves, circumference, l_a, labels, c_factors,
            trajectory_curves=None,
        ):
            # Scan predeclared symmetric windows L_B,D >= q; require >=3 unique z.
            l_b = np.asarray(l_b, int)
            z = np.asarray(z, float)
            curves = np.atleast_2d(np.asarray(curves, float))
            complement = int(circumference) - (2 * int(l_a) + l_b)
            trajectory_curves = None if trajectory_curves is None else np.asarray(trajectory_curves, float)
            rows = []
            for q in range(int(min(l_b.min(), complement.min())), int(max(l_b.max(), complement.max())) + 1):
                mask = (l_b >= q) & (complement >= q)
                unique_z = int(np.unique(np.round(z[mask], 14)).size)
                if unique_z < 3:
                    continue
                for curve_index, (label, factor) in enumerate(zip(labels, c_factors)):
                    fit = tmi_fit_line(z[mask], curves[curve_index, mask])
                    bootstrap_low = bootstrap_high = np.nan
                    if trajectory_curves is not None:
                        rng = np.random.default_rng(TMI_BOOTSTRAP_SEED + 1009 * q + curve_index)
                        bootstrap_c = np.empty(TMI_BOOTSTRAP_RESAMPLES, float)
                        for bootstrap_index in range(TMI_BOOTSTRAP_RESAMPLES):
                            ids = rng.integers(0, trajectory_curves.shape[0], trajectory_curves.shape[0])
                            mean_curve = trajectory_curves[ids, curve_index].mean(axis=0)
                            bootstrap_c[bootstrap_index] = factor * np.polyfit(
                                z[mask], mean_curve[mask], 1
                            )[0]
                        bootstrap_low, bootstrap_high = np.percentile(bootstrap_c, [2.5, 97.5])
                    rows.append(
                        {
                            "evidence": evidence,
                            "geometry": geometry,
                            "quantity": label,
                            "q_min": int(q),
                            "L_B_min": int(l_b[mask].min()),
                            "L_B_max": int(l_b[mask].max()),
                            "row_count": int(mask.sum()),
                            "unique_z_count": unique_z,
                            "c_tmi": float(factor * fit[0]),
                            "regression_se": float(factor * fit[1]),
                            "r2": float(fit[3]),
                            "bootstrap_ci95_low": float(bootstrap_low),
                            "bootstrap_ci95_high": float(bootstrap_high),
                        }
                    )
            return rows

        tmi_window_sensitivity = pd.DataFrame(
            tmi_symmetric_window_scan(
                "exact", "20x40, L_A=L_C=3", tmi_cache["exact_l_b"], tmi_cache["exact_z"],
                tmi_cache["exact_delta_i"], 40, 3, ["left wall", "right wall", "full width"], [6, 6, 3],
            )
            + tmi_symmetric_window_scan(
                "stochastic", "12x31, L_A=L_C=4", tmi_cache["stochastic_l4_l_b"],
                tmi_cache["stochastic_l4_z"], tmi_cache["stochastic_l4_mean_delta_i"],
                31, 4, ["one wall", "full width"], [6, 3],
                tmi_cache["stochastic_l4_trajectory_delta_i"],
            )
            + tmi_symmetric_window_scan(
                "stochastic", "12x31, L_A=L_C=3", tmi_cache["stochastic_l3_l_b"],
                tmi_cache["stochastic_l3_z"], tmi_cache["stochastic_l3_full_mean_delta_i"],
                31, 3, ["full width"], [3],
                tmi_cache["stochastic_l3_full_trajectory_delta_i"][:, None, :],
            )
        )

        tmi_ratio_sensitivity_rows = []
        for evidence, geometry, l_b_key, z_key, curve_key, circumference, l_a in [
            ("exact", "20x40, L_A=L_C=3", "exact_l_b", "exact_z", "exact_delta_i", 40, 3),
            ("stochastic", "12x31, L_A=L_C=4", "stochastic_l4_l_b", "stochastic_l4_z", "stochastic_l4_mean_delta_i", 31, 4),
        ]:
            l_b = np.asarray(tmi_cache[l_b_key], int)
            z = np.asarray(tmi_cache[z_key], float)
            curves = np.asarray(tmi_cache[curve_key], float)
            complement = circumference - (2 * l_a + l_b)
            for q in range(int(min(l_b.min(), complement.min())), int(max(l_b.max(), complement.max())) + 1):
                mask = (l_b >= q) & (complement >= q)
                if np.unique(np.round(z[mask], 14)).size < 3:
                    continue
                wall_slope = np.mean([tmi_fit_line(z[mask], curves[i, mask])[0] for i in range(curves.shape[0] - 1)])
                full_slope = tmi_fit_line(z[mask], curves[-1, mask])[0]
                tmi_ratio_sensitivity_rows.append(
                    {"evidence": evidence, "geometry": geometry, "q_min": int(q),
                     "full_to_wall_slope_ratio": float(full_slope / wall_slope)}
                )
        tmi_ratio_sensitivity = pd.DataFrame(tmi_ratio_sensitivity_rows)
        display(tmi_fit_table)
        display(tmi_window_sensitivity)
        display(tmi_ratio_sensitivity)
        print(
            {
                "exact_full_to_wall_ratio": exact_ratio,
                "stochastic_full_to_wall_ratio": stochastic_ratio,
                "trajectory_curve_shape": tmi_cache["stochastic_l4_trajectory_delta_i"].shape,
                "cache_sha256": tmi_metadata["cache_npz_sha256"],
            }
        )
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        axes = axes.ravel()
        exact_z = tmi_cache["exact_z"]
        exact_curves = tmi_cache["exact_delta_i"]
        exact_fits = tmi_cache["exact_fit"]

        ax = axes[0]
        ax.axvspan(exact_z.min(), exact_z.max(), color="0.5", alpha=.14, label="fit: all rows")
        exact_margin = 0.03 * np.ptp(exact_z)
        ax.set_xlim(exact_z.min() - exact_margin, exact_z.max() + exact_margin)
        exact_styles = [
            dict(color=BPJ_RED, marker="^", ms=3.7, mfc="white", label="left wall"),
            dict(color=BPJ_GREEN, marker="s", ms=5.0, mfc="none", label="right wall"),
            dict(color=BPJ_BLUE, marker="o", ms=3.7, mfc="white", label="full width"),
        ]
        line_styles = [":", "--", "-"]
        for curve, fit, style, line_style in zip(exact_curves, exact_fits, exact_styles, line_styles):
            order = np.argsort(exact_z)
            ax.plot(exact_z[order], curve[order], linestyle="none", markeredgewidth=.8, **style)
            xline = np.linspace(*ax.get_xlim(), 200)
            ax.plot(xline, fit[0] * xline + fit[2], color=style["color"], ls=line_style, lw=1.0)
        ax.set(xlabel=r"$z=\log[1/(1-x)]$", ylabel=r"$\Delta I=I_2-I_3$")
        ax.legend(loc="upper left", fontsize=5.7, ncol=2)
        panel(ax, "a", r"exact $20\times40$, $L_A=L_C=3$")

        ax = axes[1]
        stochastic_z = tmi_cache["stochastic_l4_z"]
        stochastic_mean = tmi_cache["stochastic_l4_mean_delta_i"]
        stochastic_sem = tmi_cache["stochastic_l4_sem_delta_i"]
        stochastic_fits = tmi_cache["stochastic_l4_fit"]
        ax.axvspan(stochastic_z.min(), stochastic_z.max(), color="0.5", alpha=.14, label="fit: all rows")
        stochastic_margin = 0.03 * np.ptp(stochastic_z)
        ax.set_xlim(stochastic_z.min() - stochastic_margin, stochastic_z.max() + stochastic_margin)
        for index, (label, color, marker) in enumerate(
            (("one wall", BPJ_RED, "^"), ("full width", BPJ_BLUE, "o"))
        ):
            order = np.argsort(stochastic_z)
            ax.errorbar(
                stochastic_z[order], stochastic_mean[index, order], yerr=stochastic_sem[index, order],
                linestyle="none", color=color, marker=marker, mfc="white", ms=3.5,
                capsize=1.2, label=label,
            )
            xline = np.linspace(*ax.get_xlim(), 200)
            ax.plot(
                xline, stochastic_fits[index, 0] * xline + stochastic_fits[index, 2],
                color=color, ls=(":", "-")[index], lw=1.0,
            )
        ax.set(xlabel=r"$z=\log[1/(1-x)]$", ylabel=r"$\Delta I=I_2-I_3$")
        ax.legend(loc="upper left", fontsize=5.7, ncol=2)
        panel(ax, "b", r"$12\times31$, $C=100$, $S=100$, $L_A=L_C=4$")

        ax = axes[2]
        positions = np.arange(len(tmi_fit_table))
        c_values = tmi_fit_table.c_tmi.to_numpy()
        c_regression_error = tmi_fit_table.c_tmi_error.to_numpy()
        colors = [BPJ_RED, BPJ_GREEN, BPJ_BLUE, BPJ_RED, BPJ_BLUE, BPJ_GREEN]
        markers = ["^", "s", "o", "^", "o", "s"]
        bootstrap_ci = tmi_cache["stochastic_bootstrap_slope_ci95"]
        for i in range(3, 6):
            factor = tmi_fit_table.iloc[i].c_factor
            low, high = factor * bootstrap_ci[i - 3]
            ax.vlines(i, low, high, color="0.72", lw=3.0, zorder=1)
        for i, (value, error, color, marker) in enumerate(zip(c_values, c_regression_error, colors, markers)):
            ax.errorbar(i, value, yerr=error, linestyle="none", color=color, marker=marker,
                        mfc="white", capsize=1.8, zorder=2)
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set_xticks(
            positions,
            [
                "exact\nleft", "exact\nright", "exact\nfull",
                "stoch.\nwall, $L=4$", "stoch.\nfull, $L=4$", "stoch.\nfull, $L=3$",
            ],
            rotation=22,
            ha="right",
        )
        ax.set_ylabel(r"$c_{\rm TMI}$")
        panel(ax, "c", "regression bars; gray bootstrap CI")

        ax = axes[3]
        exact_ratio_error = exact_ratio * np.sqrt(
            (exact_fits[2, 1] / exact_fits[2, 0]) ** 2
            + ((np.hypot(exact_fits[0, 1], exact_fits[1, 1]) / 2) / exact_fits[:2, 0].mean()) ** 2
        )
        bootstrap_ratio = (
            tmi_cache["stochastic_bootstrap_slopes"][:, 1]
            / tmi_cache["stochastic_bootstrap_slopes"][:, 0]
        )
        stochastic_ratio_ci = np.percentile(bootstrap_ratio, [2.5, 97.5])

        def plot_tmi_window_scan(rows, label, color, marker, linestyle, bootstrap):
            rows = rows.sort_values("q_min")
            x = rows.q_min.to_numpy(float)
            y = rows.c_tmi.to_numpy(float)
            if bootstrap:
                low = rows.bootstrap_ci95_low.to_numpy(float)
                high = rows.bootstrap_ci95_high.to_numpy(float)
                yerr = np.vstack([y - low, high - y])
            else:
                yerr = rows.regression_se.to_numpy(float)
            ax.errorbar(
                x, y, yerr=yerr, color=color, marker=marker, mfc="white",
                linestyle=linestyle, lw=.9, ms=3.5, capsize=1.5, label=label,
            )

        exact_scan = tmi_window_sensitivity.query("evidence == 'exact'")
        stochastic_scan = tmi_window_sensitivity.query("evidence == 'stochastic'")
        plot_tmi_window_scan(
            exact_scan.query("quantity == 'left wall'"), "exact wall", BPJ_RED, "^", ":", False,
        )
        plot_tmi_window_scan(
            exact_scan.query("quantity == 'full width'"), "exact full", BPJ_BLUE, "o", "-", False,
        )
        plot_tmi_window_scan(
            stochastic_scan.query("geometry == '12x31, L_A=L_C=4' and quantity == 'one wall'"),
            "stoch. wall, $L=4$", BPJ_RED, "^", "--", True,
        )
        plot_tmi_window_scan(
            stochastic_scan.query("geometry == '12x31, L_A=L_C=4' and quantity == 'full width'"),
            "stoch. full, $L=4$", BPJ_BLUE, "o", "--", True,
        )
        plot_tmi_window_scan(
            stochastic_scan.query("geometry == '12x31, L_A=L_C=3' and quantity == 'full width'"),
            "stoch. full, $L=3$", BPJ_GREEN, "s", "-.", True,
        )
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"symmetric cutoff $q$: $L_B,D\geq q$", ylabel=r"$c_{\rm TMI}(q)$")
        ax.legend(loc="upper right", fontsize=4.6, ncol=2)
        panel(ax, "d", "fit-window sensitivity")

        fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.94, wspace=.34, hspace=.40)
        tmi_sources = [
            TMI_CACHE_REL, TMI_META_REL, TMI_ARCHIVE_REL, TMI_ARCHIVE_MANIFEST_REL,
            TMI_EXACT_NOTEBOOK_REL, TMI_CLASS_SOURCE_REL, TMI_LEGACY_REDUCER_REL, TMI_EXACT_RENDER_REL,
            TMI_STOCH_RENDER_L4_REL, TMI_STOCH_RENDER_L3_REL,
        ]
        DIAGNOSTICS["result_05"] = {
            "definition": {
                "full_width": "Delta I=I2-I3=I(A:C|B)>=0; I3-I2=-Delta I",
                "proper_x_window": "Delta I^(X) is a post-contour algebraic contribution, not a von Neumann entropy or CMI",
                "positivity_scope": "SSA guarantees the full-width sign only; saved wall contributions are empirically positive",
            },
            "fit_rows": tmi_fit_table.to_dict("records"),
            "symmetric_window_rule": "L_B >= q and D=N_y-(L_A+L_B+L_C) >= q; at least three distinct z values",
            "window_sensitivity": tmi_window_sensitivity.to_dict("records"),
            "ratio_window_sensitivity": tmi_ratio_sensitivity.to_dict("records"),
            "ratios": {"exact": exact_ratio, "stochastic": stochastic_ratio},
            "point_uncertainty": "trajectory SEM after y0 averaging inside each trajectory",
            "fit_uncertainty": "OLS regression standard error across stored z rows",
            "bootstrap": {
                "seed": TMI_BOOTSTRAP_SEED,
                "resamples": TMI_BOOTSTRAP_RESAMPLES,
                "slope_se": tmi_cache["stochastic_bootstrap_slope_se"],
                "slope_ci95": tmi_cache["stochastic_bootstrap_slope_ci95"],
                "ratio_ci95": stochastic_ratio_ci,
            },
            "trajectory_curve_shapes": {
                "L4_wall_full": tmi_cache["stochastic_l4_trajectory_delta_i"].shape,
                "L3_full": tmi_cache["stochastic_l3_full_trajectory_delta_i"].shape,
            },
            "rendered_legacy_comparison": tmi_metadata["rendered_legacy_comparison"],
            "source_fingerprints": tmi_metadata["sources"],
        }
        save_figure(
            fig, "result_05_tripartite_information", tmi_sources,
            "machine-readable exact calibration and trajectory-resolved historical stochastic pilot",
            "Raw full-width CMI and wall-contour contributions recover c_TMI=1.001806 from each exact wall contribution and 1.002697 across both walls. A symmetry-preserving fit-window scan moves the exact values still closer to one but does not remove the historical stochastic pilot's coefficient bias; stochastic sensitivity bars are whole-trajectory bootstrap intervals.",
            format_mode="recomputed raw curves; no transcribed coefficients or digitized points",
        )
        """
    ),
    md(
        r"""
        ## Result 6 — Uniform adaptive trajectories converge close to the Chern-one target

        **Premise.**  The uniform topological controller should steer individual retained
        trajectories toward the intended topological state.

        **Evidence class.** Canonical GPU perfect-correction runs, $S=25$, at
        $N_x=20$ and $N_y=30,40,50$.  The plotted observable is the saved real-space
        tripartition Chern estimator.
        """
    ),
    code(
        r"""
        CHERN_BASE = "colab_lyapunov/gpu_data/lyapunov_spectra/campaigns/N20_Ny30-50_nsh1_a1-1_S25_cyclesNy"
        chern_sources = [f"{CHERN_BASE}/campaign_manifest.json"]
        chern_data = {}
        for ny in (30, 40, 50):
            rel = f"{CHERN_BASE}/runs/N20x{ny}_DW0_dwtrunc0_a2-1_nsh1_perfect_correction/real_space_chern.npz"
            chern_sources.append(rel)
            chern_data[ny] = np.load(require(rel), allow_pickle=True)["real_space_chern"]
        chern_final = pd.DataFrame([
            {"Ny": ny, "mean": values[:, -1].mean(), "sd": values[:, -1].std(ddof=1), "sem": sem(values[:, -1]),
             "minimum": values[:, -1].min(), "maximum": values[:, -1].max()}
            for ny, values in chern_data.items()
        ])
        display(chern_final)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        ax = axes[0, 0]
        for i, ny in enumerate((30, 40, 50)):
            values = chern_data[ny]
            mean = values.mean(axis=0)
            err = sem(values, axis=0)
            x = np.arange(1, values.shape[1] + 1) / ny
            ax.plot(x, mean, label=rf"$N_y={ny}$", markevery=max(1, ny // 8), mfc="white", **BPJ_STYLES[i])
            ax.fill_between(x, mean - err, mean + err, color=BPJ_STYLES[i]["color"], alpha=.10, lw=0)
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=0.8)
        ax.set(xlabel=r"$t/N_y$", ylabel=r"$\overline{\mathcal{C}}_R$", ylim=(0.72, 1.012))
        ax.legend()
        panel(ax, "a", "trajectory mean")

        ax = axes[0, 1]
        for i, ny in enumerate((30, 40, 50)):
            values = chern_data[ny]
            mean = values.mean(axis=0)
            err = sem(values, axis=0)
            deviation = np.maximum(np.abs(1 - mean), 1e-8)
            x = np.arange(1, values.shape[1] + 1) / ny
            ax.semilogy(x, deviation, label=rf"$N_y={ny}$", markevery=max(1, ny // 8), mfc="white", **BPJ_STYLES[i])
            ax.fill_between(x, np.maximum(deviation - err, 1e-8), deviation + err,
                            color=BPJ_STYLES[i]["color"], alpha=.10, lw=0)
        ax.set(xlabel=r"$t/N_y$", ylabel=r"$|1-\overline{\mathcal{C}}_R|$")
        ax.legend()
        panel(ax, "b", "approach to unity")

        ax = axes[1, 0]
        rng = np.random.default_rng(12345)
        for i, ny in enumerate((30, 40, 50)):
            vals = chern_data[ny][:, -1]
            jitter = rng.uniform(-0.11, 0.11, len(vals))
            style = BPJ_STYLES[i]
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=13, facecolors="white", edgecolors=style["color"], marker=style["marker"], linewidths=0.7)
            ax.hlines(vals.mean(), i - 0.22, i + 0.22, color=style["color"], lw=1.2)
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=0.8)
        ax.set_xticks(range(3), [30, 40, 50])
        ax.set(xlabel=r"$N_y$", ylabel=r"final $\mathcal{C}_{R,\xi}$", ylim=(0.989, 1.002))
        panel(ax, "c", "trajectory distribution")

        ax = axes[1, 1]
        for i, row in chern_final.iterrows():
            style = BPJ_STYLES[i]
            ax.errorbar(row.Ny, row["mean"], yerr=row.sd, color=style["color"], marker=style["marker"], mfc="white", capsize=2)
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=0.8)
        ax.text(0.04, 0.08, rf"global minimum $={chern_final.minimum.min():.6f}$", transform=ax.transAxes, fontsize=7)
        ax.set(xlabel=r"$N_y$", ylabel=r"final mean $\pm$ SD", ylim=(0.989, 1.003))
        panel(ax, "d", "final summary")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_06"] = chern_final.to_dict("records")
        save_figure(fig, "result_06_uniform_topology", chern_sources, "canonical GPU trajectory precursor",
                    "The uniform adaptive controller drives the saved trajectory-resolved Chern estimator very close to one at all three tested sizes.")
        """
    ),
    md(
        r"""
        # Group II — Feedback activity and the transition axis

        ## Result 7 — Feedback activity is sharply concentrated at the wall

        **Premise.**  Wrong outcomes measure how much local corrective work the controller
        performs.  The wall should show an excess relative to both its own interior and a
        uniform controller.

        **Evidence class.** Canonical CPU raster-order perfect-correction pilot,
        $N_x=20$, $N_y=20,22,24$, $S=25$.  Activity and tangent spectra were generated in
        separate trajectory suites, so their ratios demonstrate coexistence, not
        trajectory-level correlation or causation.
        """
    ),
    code(
        r"""
        ACT_BASE = "topological_frustration_diagnostics/results/campaigns/20260813_121340_Nx20_perfect_Ny20_22_24"
        activity_sources, activity_rows, activity_profiles, activity_profile_sem = [], [], {}, {}
        spectral_by_geom = {}
        for ny in (20, 22, 24):
            for geometry, dirname in (("wall", f"N20x{ny}_dw_dwtrunc1_nsh1_perfect_correction"),
                                      ("uniform", f"N20x{ny}_uniform_dwtrunc0_nsh1_perfect_correction")):
                scalar_rel = f"{ACT_BASE}/activity_Ny{ny}_perfect/activity/{dirname}/scalar_metrics.csv"
                analysis_rel = f"{ACT_BASE}/activity_Ny{ny}_perfect/activity/{dirname}/activity_analysis.npz"
                raw_rel = f"{ACT_BASE}/activity_Ny{ny}_perfect/activity/{dirname}/activity_raw.npz"
                spec_rel = f"{ACT_BASE}/spectral_Ny{ny}_perfect/spectral/{dirname}/scalar_metrics.csv"
                spec_diag_rel = f"{ACT_BASE}/spectral_Ny{ny}_perfect/spectral/{dirname}/spectral_diagnostics.npz"
                activity_sources += [scalar_rel, analysis_rel, raw_rel, spec_rel, spec_diag_rel]
                df = pd.read_csv(require(scalar_rel))
                df["geometry_plot"] = geometry
                z = np.load(require(analysis_rel), allow_pickle=True)
                raw = np.load(require(raw_rel), allow_pickle=True)
                burn = int(z["burn_in"])
                sample_rates = z["rates_by_cycle"][0, :, burn:, :, -1].mean(axis=1)
                df["mean_sem"] = np.nan
                for region_index, region in enumerate(z["region_names"]):
                    df.loc[(df.activity == "defect_X") & (df.region == region), "mean_sem"] = sem(sample_rates[:, region_index])
                activity_rows.append(df)
                activity_profiles[(ny, geometry)] = z["spatial_profile_x"][0, :, -1]
                profile_samples = np.full((int(raw["samples"]), int(raw["nx"])), np.nan)
                for x in range(int(raw["nx"])):
                    site_mask = raw["site_x"] == x
                    valid = raw["valid"][:, burn:, site_mask, :]
                    counts = raw["defect_X"][:, burn:, site_mask, :]
                    denominator = valid.sum(axis=(1, 2, 3))
                    profile_samples[:, x] = np.divide(
                        (counts * valid).sum(axis=(1, 2, 3)), denominator,
                        out=np.full(len(denominator), np.nan), where=denominator > 0,
                    )
                activity_profile_sem[(ny, geometry)] = sem(profile_samples, axis=0)
                spec = np.load(require(spec_diag_rel), allow_pickle=True)
                gap_samples = np.abs(spec["lyapunov_final_value"])
                spectral_by_geom[(ny, geometry)] = {"mean": gap_samples.mean(), "sem": sem(gap_samples)}
        activity = pd.concat(activity_rows, ignore_index=True)
        display(activity.query("activity == 'defect_X' and region in ['all','interface','interior']")[["Ny", "geometry_plot", "region", "mean_rate"]])
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        ax = axes[0, 0]
        for geometry, style in (("wall", geom_style["wall"]), ("uniform", geom_style["uniform"])):
            mean = activity_profiles[(24, geometry)]
            err = activity_profile_sem[(24, geometry)]
            ax.plot(np.arange(20), mean, mfc="white", **style)
            ax.fill_between(np.arange(20), np.maximum(mean - err, 0), mean + err,
                            color=style["color"], alpha=.10, lw=0)
        ax.axvline(4, color=BPJ_GRAY, ls=":", lw=0.7)
        ax.axvline(16, color=BPJ_GRAY, ls=":", lw=0.7)
        ax.set(xlabel=r"$x$", ylabel="wrong-outcome rate")
        ax.legend()
        panel(ax, "a", r"$N_y=24$ profile")

        ax = axes[0, 1]
        qall = activity.query("activity == 'defect_X' and region == 'all'")
        for geometry, group in qall.groupby("geometry_plot"):
            ax.errorbar(group.Ny, group.mean_rate, yerr=group.mean_sem, capsize=2,
                        mfc="white", **geom_style[geometry])
        ax.set(xlabel=r"$N_y$", ylabel="all-channel rate")
        ax.legend()
        panel(ax, "b", "wall versus uniform")

        ax = axes[1, 0]
        qwall = activity.query("activity == 'defect_X' and geometry_plot == 'wall' and region in ['interface','interior']")
        for region, color, marker, ls in (("interface", BPJ_RED, "^", ":"), ("interior", BPJ_BLUE, "o", "-")):
            group = qwall.query("region == @region")
            ax.errorbar(group.Ny, group.mean_rate, yerr=group.mean_sem, color=color, marker=marker,
                        ls=ls, capsize=2, mfc="white", label=region)
        ax.set(xlabel=r"$N_y$", ylabel="wrong-outcome rate")
        ax.legend()
        panel(ax, "c", "inside the wall geometry")

        ax = axes[1, 1]
        nys = np.array([20, 22, 24])
        act_ratio, act_ratio_sem, gap_ratio, gap_ratio_sem = [], [], [], []
        for ny in nys:
            wall = qall.query("Ny == @ny and geometry_plot == 'wall'").iloc[0]
            uniform = qall.query("Ny == @ny and geometry_plot == 'uniform'").iloc[0]
            ratio = wall.mean_rate / uniform.mean_rate
            ratio_sem = abs(ratio) * np.hypot(wall.mean_sem / wall.mean_rate, uniform.mean_sem / uniform.mean_rate)
            act_ratio.append(ratio)
            act_ratio_sem.append(ratio_sem)
            wall_gap = spectral_by_geom[(ny, "wall")]
            uniform_gap = spectral_by_geom[(ny, "uniform")]
            ratio = uniform_gap["mean"] / wall_gap["mean"]
            ratio_sem = abs(ratio) * np.hypot(uniform_gap["sem"] / uniform_gap["mean"], wall_gap["sem"] / wall_gap["mean"])
            gap_ratio.append(ratio)
            gap_ratio_sem.append(ratio_sem)
        ax.errorbar(nys, act_ratio, yerr=act_ratio_sem, color=BPJ_RED, marker="^", ls=":",
                    capsize=2, mfc="white", label="activity enhancement")
        ax.errorbar(nys, gap_ratio, yerr=gap_ratio_sem, color=BPJ_BLUE, marker="o", ls="-",
                    capsize=2, mfc="white", label="gap suppression")
        ax.set(xlabel=r"$N_y$", ylabel="wall / control contrast")
        ax.legend()
        ax.text(0.06, 0.48, "separate trajectory suites", transform=ax.transAxes, fontsize=7)
        panel(ax, "d", "coexisting contrasts")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_07"] = {
            "activity_ratio": list(map(float, act_ratio)), "activity_ratio_sem": list(map(float, act_ratio_sem)),
            "gap_ratio": list(map(float, gap_ratio)), "gap_ratio_sem": list(map(float, gap_ratio_sem)),
        }
        save_figure(fig, "result_07_feedback_activity", activity_sources, "canonical CPU finite-size precursor",
                    "Wrong outcomes are strongly enhanced at the wall and coexist with a much smaller wall tangent gap.")
        """
    ),
    md(
        r"""
        ## Result 8 — Independent sweeps locate the dynamical crossover near $\alpha=2$

        **Premise.**  The exact positive-mass transition occurs at $\alpha_c=2$.  Distinct
        dynamical diagnostics should change in the same neighborhood without being
        conflated as one observable.

        **Evidence class.** Canonical GPU particle--Choi transfer proxies and canonical
        CPU max-mix purification observables.  Open transfer markers indicate fewer than
        ten finite samples; loss of a finite sector is censoring, not an infinite measured
        gap.
        """
    ),
    code(
        r"""
        ALPHA_GPU = "colab_no_feedback_alpha_sweep_transfer/analysis_outputs/gpu_data_summary/tables/run_summary_metrics.csv"
        ALPHA_CPU = "colab_charge_fluctuations/analysis_outputs/purification_charge_sharpening_alpha_sweep_cpu/N16_alpha-fine21_nsh1_dwtrunc1_init-maxmix_S10_cycles-2Ny/tables/steady_state_summary.csv"
        alpha_sources = [ALPHA_GPU, ALPHA_CPU]
        alpha_gpu = pd.read_csv(require(ALPHA_GPU))
        alpha_cpu = pd.read_csv(require(ALPHA_CPU)).query("protocol == 'perfect_correction'")
        display(alpha_gpu[["Ny", "correction", "alpha", "gap_mean", "gap_sem", "gap_finite_sample_count"]].head())
        display(alpha_cpu[["Ny", "alpha_topological_region", "total_entropy_mean", "total_charge_variance_mean"]].head())
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        for ax, correction, letter, title in ((axes[0, 0], "none", "a", "no feedback: transfer gap"),
                                               (axes[0, 1], "perfect", "b", "perfect correction: transfer gap")):
            for i, ny in enumerate((20, 30, 40)):
                q = alpha_gpu.query("correction == @correction and Ny == @ny").sort_values("alpha")
                style = BPJ_STYLES[i]
                ax.errorbar(q.alpha, q.gap_mean, yerr=q.gap_sem, label=rf"$N_y={ny}$", capsize=1.5, mfc="white", **style)
                censored = q.gap_finite_sample_count < q.sample_count
                ax.scatter(q.loc[censored, "alpha"], q.loc[censored, "gap_mean"], s=32, facecolors="white", edgecolors=style["color"], marker=style["marker"], linewidths=1.0)
            ax.axvline(2, color=BPJ_BLACK, ls="--", lw=0.8)
            ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel="finite transfer exponent", yscale="log")
            ax.legend()
            panel(ax, letter, title)

        for ax, observable, letter, title in ((axes[1, 0], "total_entropy_mean", "c", "max-mix entropy"),
                                               (axes[1, 1], "total_charge_variance_mean", "d", "max-mix charge variance")):
            for i, ny in enumerate((16, 24, 32)):
                q = alpha_cpu.query("Ny == @ny").sort_values("alpha_topological_region")
                style = BPJ_STYLES[i]
                y = np.maximum(q[observable].to_numpy(), 1e-14)
                err = q[observable.replace("_mean", "_std")].to_numpy() / np.sqrt(q.samples.to_numpy())
                ax.errorbar(q.alpha_topological_region, y, yerr=np.vstack([np.minimum(err, .999 * y), err]),
                            label=rf"$N_y={ny}$", capsize=1.2, mfc="white", **style)
            ax.axvline(2, color=BPJ_BLACK, ls="--", lw=0.8)
            ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel=observable.replace("_mean", "").replace("_", " "), yscale="log")
            ax.legend()
            panel(ax, letter, title)

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_08"] = {
            "gpu_rows": int(len(alpha_gpu)), "cpu_rows": int(len(alpha_cpu)),
            "critical_alpha": 2.0,
        }
        save_figure(fig, "result_08_alpha_transition", alpha_sources, "canonical GPU and CPU transition precursors",
                    "Transfer and max-mix purification diagnostics independently change sharply in the neighborhood of the exact alpha=2 transition.")
        """
    ),
    md(
        r"""
        # Group III — Handedness and record observables

        ## Result 9 — Modular dynamics show orientation-dependent handedness

        **Premise.**  A localized packet evolved with the state-derived modular generator
        should drift with opposite orientation on oppositely directed walls.

        **Evidence class.** Completed exact B0 calibration plus a canonical GPU-state / CPU
        observer precursor.  Modular time is not monitored circuit time, and translated
        cuts within one trajectory are correlated variance-reduction samples.  Accordingly,
        the stochastic reduction first averages the 40 translated origins inside each of
        the 10 trajectories, and only then takes the trajectory mean and the
        sample-standard-deviation (`ddof=1`) trajectory SEM.  The effective
        independent sample count for the uncertainty is therefore 10, not $10\times40$.
        """
    ),
    code(
        r"""
        MOD_EXACT_RAW = "experiment_review/b0_exact_domain_wall/results/20260816_191957/raw/coupled__Nx020__Ny048.npz"
        MOD_EXACT_TABLE = "experiment_review/b0_exact_domain_wall/results/20260816_191957/processed/tables/modular_velocities.csv"
        MOD_EXACT_RESPONSE = "experiment_review/b0_exact_domain_wall/results/20260816_191957/processed/tables/response_velocities.csv"
        MOD_STOCH_BASE = "colab_small_system_testing/analysis_outputs/dynamic_modular_charge_spreading/sample_y0_averaged_dy_com/runs"
        modular_sources = [MOD_EXACT_RAW, MOD_EXACT_TABLE, MOD_EXACT_RESPONSE]
        mod_exact = np.load(require(MOD_EXACT_RAW), allow_pickle=True)
        mod_exact_table = pd.read_csv(require(MOD_EXACT_TABLE)).query("construction == 'coupled' and nx == 20 and ny == 48 and is_primary == True and source_width == 3")
        mod_exact_response = pd.read_csv(require(MOD_EXACT_RESPONSE)).query("construction == 'coupled' and nx == 20 and ny == 48")
        mod_exact_table = mod_exact_table.merge(
            mod_exact_response[["wall_index", "wavefront_velocity_stderr"]], on="wall_index", how="left", validate="one_to_one"
        )
        mod_stoch = {}
        mod_stoch_reduced = {}
        for nshell in (1, 2):
            rel = f"{MOD_STOCH_BASE}/N16x40_nsh{nshell}_perfect_correction/sample_y0_averaged_dy_com.npz"
            modular_sources.append(rel)
            mod_stoch[nshell] = np.load(require(rel), allow_pickle=True)
            raw_curves = np.asarray(mod_stoch[nshell]["dy_com_curves"], float)
            # Shape: (cycle selection, trajectory, translated y0, packet, modular time).
            # The 40 origins share one trajectory and are variance-reduction samples,
            # not independent ensemble members.
            trajectory_curves = raw_curves.mean(axis=2)
            trajectory_mean = trajectory_curves.mean(axis=1)
            trajectory_sem = trajectory_curves.std(axis=1, ddof=1) / np.sqrt(
                trajectory_curves.shape[1]
            )
            if not np.allclose(
                trajectory_mean, mod_stoch[nshell]["dy_com_mean"], atol=5e-13, rtol=0
            ):
                raise AssertionError(f"Result 9 trajectory-first mean mismatch for n_shell={nshell}")
            old_pooled_sem = np.asarray(mod_stoch[nshell]["dy_com_sem"], float)
            mod_stoch_reduced[nshell] = {
                "trajectory_curves": trajectory_curves,
                "mean": trajectory_mean,
                "sem": trajectory_sem,
                "old_pooled_sem": old_pooled_sem,
            }
        display(mod_exact_table[["wall_index", "modular_velocity", "velocity_stderr", "minimum_wall_retention"]])
        for nshell, z in mod_stoch.items():
            reduced = mod_stoch_reduced[nshell]
            print(
                "n_shell", nshell,
                "final displacement", reduced["mean"][0, :, -1],
                "+/- trajectory SEM", reduced["sem"][0, :, -1],
            )
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        ax = axes[0, 0]
        t = mod_exact["modular_times"]
        keep = t <= 2.5
        primary = int(mod_exact["modular_primary_width_index"])
        for wall, color, marker, label in ((0, BPJ_RED, "^", "left wall"), (1, BPJ_BLUE, "o", "right wall")):
            ax.plot(t[keep], mod_exact["modular_handedness"][primary, wall, keep], color=color, marker=marker,
                    markevery=10, ls=":" if wall == 0 else "-", mfc="white", label=label)
        ax.axhline(0, color=BPJ_BLACK, lw=0.7)
        ax.set(xlabel=r"modular time $t_{\rm mod}$", ylabel=r"$D_w(t_{\rm mod})$")
        ax.legend()
        panel(ax, "a", "exact endpoint-resolved drift")

        ax = axes[0, 1]
        q = mod_exact_table.sort_values("wall_index")
        ax.errorbar([-1, 1], q.modular_velocity, yerr=q.velocity_stderr, color=BPJ_RED, marker="^", ls="none", mfc="white", capsize=2, label="modular")
        ax.errorbar([-1, 1], q.physical_velocity, yerr=q.wavefront_velocity_stderr,
                    color=BPJ_BLUE, marker="o", ls="-", capsize=2, mfc="white", label="physical")
        ax.axhline(0, color=BPJ_BLACK, lw=0.7)
        ax.set_xticks([-1, 1], ["left", "right"])
        ax.set_ylabel("signed velocity")
        ax.legend()
        panel(ax, "b", "exact handed target")

        ax = axes[1, 0]
        z = mod_stoch[2]
        reduced = mod_stoch_reduced[2]
        times = z["times"]
        for packet in range(4):
            wall = 0 if packet < 2 else 1
            color = BPJ_RED if wall == 0 else BPJ_BLUE
            ls = "-" if packet % 2 == 0 else "--"
            label = str(z["packet_labels"][packet])
            mean = reduced["mean"][0, packet]
            err = reduced["sem"][0, packet]
            ax.plot(times, mean, color=color, ls=ls, label=label)
            ax.fill_between(times, mean - err, mean + err, color=color, alpha=0.10, lw=0)
        ax.axhline(0, color=BPJ_BLACK, lw=0.7)
        ax.set(xlabel=r"$t_{\rm mod}$", ylabel=r"$\langle\Delta y\rangle$")
        ax.legend(ncol=2)
        panel(ax, "c", r"stochastic snapshots, $n_{\rm shell}=2$")

        ax = axes[1, 1]
        x = np.arange(4)
        for j, nshell in enumerate((1, 2)):
            z = mod_stoch[nshell]
            reduced = mod_stoch_reduced[nshell]
            color, marker = ((BPJ_GREEN, "s") if nshell == 1 else (BPJ_BLUE, "o"))
            ax.errorbar(x + (j - .5) * .12, reduced["mean"][0, :, -1], yerr=reduced["sem"][0, :, -1],
                        color=color, marker=marker, ls="none", mfc="white", capsize=2, label=rf"$n_{{\rm shell}}={nshell}$")
        ax.axhline(0, color=BPJ_BLACK, lw=0.7)
        ax.set_xticks(x, mod_stoch[2]["packet_labels"], rotation=30)
        ax.set_ylabel(r"final $\langle\Delta y\rangle$")
        ax.legend()
        panel(ax, "d", "endpoint sign pattern")

        fig.subplots_adjust(wspace=0.34, hspace=0.36)
        DIAGNOSTICS["result_09"] = {
            "exact_modular_velocity": mod_exact_table.modular_velocity.tolist(),
            "stochastic_nshell2_final": mod_stoch_reduced[2]["mean"][0, :, -1].tolist(),
            "stochastic_averaging_order": "mean over 40 translated y0 origins within each trajectory, then mean and ddof=1 SEM over 10 trajectories",
            "independent_trajectory_count": 10,
            "translated_origins_per_trajectory": 40,
            "endpoint_uncertainty": {
                str(nshell): {
                    "trajectory_sem": mod_stoch_reduced[nshell]["sem"][0, :, -1].tolist(),
                    "superseded_pooled_cut_sem": mod_stoch_reduced[nshell]["old_pooled_sem"][0, :, -1].tolist(),
                    "trajectory_to_pooled_ratio": (
                        mod_stoch_reduced[nshell]["sem"][0, :, -1]
                        / mod_stoch_reduced[nshell]["old_pooled_sem"][0, :, -1]
                    ).tolist(),
                }
                for nshell in (1, 2)
            },
        }
        save_figure(fig, "result_09_modular_handedness", modular_sources, "completed exact calibration plus canonical stochastic precursor",
                    "Exact and trajectory-derived modular packet motion exhibits the expected orientation-dependent sign structure; stochastic bands and endpoint bars are trajectory SEM after within-trajectory origin averaging.")
        """
    ),
    md(
        r"""
        ## Result 10 — The classical probability record detects an active wall sector

        **Premise.**  Sequential premeasurement target probabilities determine a
        chain-rule conditional Shannon entropy.  Its leading term is extensive, while its
        spatial distribution can reveal where the record remains uncertain.

        **Evidence class.** Canonical GPU perfect-correction streaming probability maps,
        $S=100$, $N_x=20$, $N_y=30,40,50$.  Outcome bits were not saved, so this computes
        the Born mean chain entropy, not individual record self-information or a record CFT
        coefficient.
        """
    ),
    code(
        r"""
        RECORD_BASE = "colab_charge_fluctuations/gpu_data/streaming_covariance_observables/campaigns/N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
        record_sources, record_data = [], {}
        for ny in (30, 40, 50):
            rel = f"{RECORD_BASE}/N20x{ny}_nsh1_init-default_perfect_correction/measurement_frustration.npz"
            record_sources.append(rel)
            z = np.load(require(rel), allow_pickle=True)
            H = np.zeros((len(z["sample_indices"]), len(z["cycle_labels"])), float)
            h_x_sample_cycle = np.zeros((len(z["sample_indices"]), len(z["cycle_labels"]), 20), float)
            for key in ("s_Ap", "s_Am", "s_Bp", "s_Bm"):
                h = binary_entropy(z[key])
                H += h.sum(axis=(2, 3))
                h_x_sample_cycle += h.mean(axis=3) / 4
            record_data[ny] = {
                "cycle": z["cycle_labels"], "H": H,
                "h_x_cycle": h_x_sample_cycle.mean(axis=0),
                "h_x_sample_cycle": h_x_sample_cycle,
            }
        record_summary = []
        for ny, payload in record_data.items():
            late = payload["H"][:, payload["H"].shape[1] // 2 :].mean(axis=1)
            record_summary.append({"Ny": ny, "H_late": late.mean(), "H_late_sem": sem(late), "H_over_Ny": late.mean() / ny,
                                   "H_over_Ny_sem": sem(late) / ny})
        record_summary = pd.DataFrame(record_summary)
        display(record_summary)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        ax = axes[0, 0]
        for i, ny in enumerate((30, 40, 50)):
            payload = record_data[ny]
            mean = payload["H"].mean(axis=0)
            err = sem(payload["H"], axis=0)
            x = payload["cycle"] / ny
            style = BPJ_STYLES[i]
            ax.plot(x, mean, label=rf"$N_y={ny}$", markevery=max(1, len(x)//8), mfc="white", **style)
            ax.fill_between(x, mean - err, mean + err, color=style["color"], alpha=.10, lw=0)
        ax.set(xlabel=r"$t/N_y$", ylabel=r"$\widehat H_{\rm chain}$ per cycle")
        ax.legend()
        panel(ax, "a", "transient and plateau")

        ax = axes[0, 1]
        for i, row in record_summary.iterrows():
            style = BPJ_STYLES[i]
            ax.errorbar(row.Ny, row.H_over_Ny, yerr=row.H_over_Ny_sem, color=style["color"], marker=style["marker"], mfc="white", capsize=2)
        ax.set(xlabel=r"$N_y$", ylabel=r"late $\widehat H_{\rm chain}/N_y$", ylim=(2.97, 3.01))
        panel(ax, "b", "extensive background")

        payload = record_data[40]
        late_x_samples = payload["h_x_sample_cycle"][:, payload["h_x_sample_cycle"].shape[1] // 2 :].mean(axis=1)
        late_x = late_x_samples.mean(axis=0)
        late_x_sem = sem(late_x_samples, axis=0)
        ax = axes[1, 0]
        ax.errorbar(np.arange(20), late_x, yerr=late_x_sem, color=BPJ_BLUE, marker="o",
                    capsize=1.5, mfc="white")
        ax.axvline(5, color=BPJ_RED, ls=":", lw=0.8)
        ax.axvline(15, color=BPJ_RED, ls=":", lw=0.8)
        wall_mask = np.zeros(20, dtype=bool)
        for wall in (5, 15):
            wall_mask[np.minimum(np.abs(np.arange(20) - wall), 20 - np.abs(np.arange(20) - wall)) <= 1] = True
        wall_h, nonwall_h = late_x[wall_mask].mean(), late_x[~wall_mask].mean()
        ax.text(0.04, .94, rf"wall/nonwall $={wall_h:.4f}/{nonwall_h:.4f}$", transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.set(xlabel=r"$x$", ylabel="binary entropy per event")
        panel(ax, "c", r"$20\times40$ spatial profile")

        ax = axes[1, 1]
        im = ax.imshow(payload["h_x_cycle"].T, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest",
                       extent=[payload["cycle"][0] / 40, payload["cycle"][-1] / 40, -0.5, 19.5])
        unit_cell_grid(ax, ny=20)
        ax.set(xlabel=r"$t/N_y$", ylabel=r"$x$")
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("binary entropy / event", fontsize=7)
        cb.ax.tick_params(labelsize=6, direction="in")
        panel(ax, "d", "where uncertainty lives")

        fig.subplots_adjust(wspace=0.36, hspace=0.36)
        DIAGNOSTICS["result_10"] = {"plateau": record_summary.to_dict("records"), "wall_event_entropy": wall_h, "nonwall_event_entropy": nonwall_h}
        save_figure(fig, "result_10_record_uncertainty", record_sources, "canonical GPU chain-entropy feasibility result",
                    "The sequential probability record has a stable extensive Shannon background and roughly tenfold enhanced uncertainty near the walls.")
        """
    ),
    md(
        r"""
        # Group IV — Effective parent Hamiltonians, spatial structure, and ensemble spread

        ## Result 11 — Translation-resolved flattened spectra expose the wall modes

        **Premise.**  The archived flattened-parent construction preserves translation
        symmetry along the periodic direction, so the single-particle problem decomposes
        into $k_y$ blocks.  In that representation a chiral wall is visible directly as
        in-gap spectral flow rather than only through a fitted scalar.

        **Evidence class.** Rendered deterministic benchmark.  The source notebook and
        PNG are both retained, but the numerical eigenvalue array was not exported as a
        separate machine-readable artifact.  The atlas therefore republishes a cropped,
        full-width view of the archived $k_y$ panels and labels this provenance explicitly.
        """
    ),
    code(
        r"""
        FLAT_SPEC_RENDER = "exact_DW_benchmark_notes/DW_GS_Benchmark_Figs/flattened_hamiltonian_spec.png"
        FLAT_SPEC_NOTEBOOK = "notebooks/exact_DW_GS_benchmark.ipynb"
        flat_spec_sources = [FLAT_SPEC_RENDER, FLAT_SPEC_NOTEBOOK]
        flat_image = plt.imread(require(FLAT_SPEC_RENDER))
        crop_start = int(round(0.515 * flat_image.shape[0]))
        flat_ky_image = flat_image[crop_start:, ...]

        fig, ax = plt.subplots(figsize=(7.05, 3.05))
        ax.imshow(flat_ky_image, interpolation="nearest")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        DIAGNOSTICS["result_11"] = {
            "source_pixel_shape": list(flat_image.shape),
            "published_crop_start_row": crop_start,
            "construction": "translation invariant along y; k_y-resolved spectrum",
            "trial_orbitals": ["X", "Y", "Z"],
        }
        save_figure(fig, "result_11_flattened_wall_spectrum", flat_spec_sources,
                    "rendered deterministic flattened-Hamiltonian benchmark",
                    "The translation-resolved flattened spectrum contains counter-propagating in-gap branches crossing zero at the two walls; X and Y give the cleanest separation.",
                    format_mode="preserved legacy render; cropped and resized only")
        """
    ),
    md(
        r"""
        ## Result 12 — The finite-range uniform parent is already a gapped Chern insulator

        For outcome projectors $P_{\sigma,+}$ and $P_{\sigma,-}$, define
        $$
        H_{\rm flat}=\sum_\sigma(P_{\sigma,+}-P_{\sigma,-}).
        $$
        At fixed particle number, the additive wrong-outcome loss is
        $$
        \mathcal L_{\rm wrong}(P)=\mathrm{const}+\mathrm{Tr}(P H_{\rm flat}).
        $$
        Ky Fan's minimum principle therefore says that the rank-$N_{\rm occ}$ projector
        minimizing this loss is the span of the $N_{\rm occ}$ lowest eigenvectors of
        $H_{\rm flat}$.  This is a sum of marginal wrong-outcome probabilities, not the
        probability of one joint all-correct event, because the overlapping local
        projectors generally do not commute.

        **Evidence class.** Direct deterministic reconstruction of the uniform
        $20\times40$ flattened parent for $n_{\rm sh}=0,1,2,$ and full support, with both
        the real-space tripartition estimator and an independent momentum-space FHS
        integer evaluated on the same occupied lower-band projector $P_{\rm occ}$.  In
        the covariance convention expected by the real-space routine, this projector is
        supplied as $G=2P_{\rm occ}^{*}-\mathbf{1}$; the extra minus sign used in an
        earlier atlas build instead reconstructed the complementary projector
        $\mathbf{1}-P_{\rm occ}$ and was a computational bug, not a Chern-sign
        convention.  The topological $\alpha=1$ parent is compared with the trivial
        $\alpha=3$ control.  Separate domain-wall tables are retained only for their
        entropy and wall-correlator convergence; the former cross-interface scalar is not
        used as a bulk Chern diagnostic.
        """
    ),
    code(
        r"""
        FLAT_BASE = "notebooks/flattened_hamiltonian_analysis"
        flat_sources = [
            f"{FLAT_BASE}/flattened_hamiltonian_analysis.ipynb",
            f"{FLAT_BASE}/data/slope_fit_summary_df.csv",
            f"{FLAT_BASE}/data/slope_fit_df.csv",
            f"{FLAT_BASE}/data/cavg_fit_summary_df.csv",
            "src/fgtn/classA_U1FGTN.py",
        ]
        flat_entropy = pd.read_csv(require(flat_sources[1]))
        flat_entropy_rows = pd.read_csv(require(flat_sources[2]))
        flat_corr = pd.read_csv(require(flat_sources[3]))

        import sys
        if str(ROOT / "src") not in sys.path:
            sys.path.insert(0, str(ROOT / "src"))
        from fgtn.classA_U1FGTN import classA_U1FGTN

        FLAT_NX, FLAT_NY = 20, 40
        FLAT_SHELLS = [0, 1, 2, None]

        def flat_uniform_parent(alpha, nshell):
            model = classA_U1FGTN(
                FLAT_NX, FLAT_NY, nshell=nshell, DW=False,
                alpha_1=alpha, alpha_2=alpha,
            )
            model.construct_OW_projectors(
                nshell=nshell, DW=False, trial_orbitals="X", dw_truncation=False
            )
            nlayer = 2 * FLAT_NX * FLAT_NY
            modes = [
                mode.reshape(nlayer, -1)
                for mode in (model.WF_Ap, model.WF_Bp, model.WF_Am, model.WF_Bm)
            ]
            projectors = [mode @ mode.conj().T for mode in modes]
            h_flat = projectors[0] + projectors[1] - projectors[2] - projectors[3]
            h_flat = 0.5 * (h_flat + h_flat.conj().T)
            evals, evecs = np.linalg.eigh(h_flat)
            occupied = evals < 0
            assert int(occupied.sum()) == nlayer // 2
            u_occ = evecs[:, occupied]
            p_occ = u_occ @ u_occ.conj().T
            # `real_space_chern_number` reconstructs its particle projector from this
            # covariance block.  This sign evaluates p_occ itself; a leading minus sign
            # would instead evaluate the complementary projector I-p_occ.
            g_chern = 2 * p_occ.conj() - np.eye(nlayer, dtype=np.complex128)
            g_full = model._block_diag2(g_chern, np.zeros_like(g_chern))
            chern_rs = float(np.real(model.real_space_chern_number(g_full)))
            return model, h_flat, {
                "alpha": float(alpha),
                "nshell": nshell,
                "shell_label": "full" if nshell is None else str(nshell),
                "chern_rs": chern_rs,
                "min_abs_energy": float(np.min(np.abs(evals))),
                "occupied_bandwidth": float(np.ptp(evals[occupied])),
            }

        def flat_uniform_fhs(h_flat, nx=FLAT_NX, ny=FLAT_NY, nk=101):
            def index(mu, x, y):
                return mu + 2 * x + 2 * nx * y

            kernel = np.empty((nx, ny, 2, 2), dtype=np.complex128)
            for x in range(nx):
                for y in range(ny):
                    for mu in range(2):
                        for nu in range(2):
                            kernel[x, y, mu, nu] = h_flat[
                                index(mu, 0, 0), index(nu, x, y)
                            ]
            dx = np.where(np.arange(nx) <= nx // 2, np.arange(nx), np.arange(nx) - nx)
            dy = np.where(np.arange(ny) <= ny // 2, np.arange(ny), np.arange(ny) - ny)
            momenta = np.linspace(-np.pi, np.pi, nk, endpoint=False)
            lower = np.empty((nk, nk, 2), dtype=np.complex128)
            for ix, kx in enumerate(momenta):
                for iy, ky in enumerate(momenta):
                    phase = np.exp(1.0j * (kx * dx[:, None] + ky * dy[None, :]))
                    h_k = np.einsum("xy,xyab->ab", phase, kernel, optimize=True)
                    _, vectors = np.linalg.eigh(0.5 * (h_k + h_k.conj().T))
                    lower[ix, iy] = vectors[:, 0]
            link_x = np.einsum(
                "...a,...a->...", lower.conj(), np.roll(lower, -1, axis=0)
            )
            link_y = np.einsum(
                "...a,...a->...", lower.conj(), np.roll(lower, -1, axis=1)
            )
            link_x /= np.abs(link_x)
            link_y /= np.abs(link_y)
            plaquette = (
                link_x * np.roll(link_y, -1, axis=0)
                * np.conj(np.roll(link_x, -1, axis=1)) * np.conj(link_y)
            )
            return float(np.rint(np.angle(plaquette).sum() / (2 * np.pi)))

        uniform_rows = []
        uniform_hamiltonians = {}
        for alpha in (1.0, 3.0):
            for nshell in FLAT_SHELLS:
                _, h_flat, row = flat_uniform_parent(alpha, nshell)
                uniform_hamiltonians[(alpha, nshell)] = h_flat
                uniform_rows.append(row)
        uniform_flat = pd.DataFrame(uniform_rows)
        for alpha in (1.0, 3.0):
            mask = (uniform_flat.alpha == alpha) & (uniform_flat.shell_label == "1")
            uniform_flat.loc[mask, "fhs_lower_band"] = flat_uniform_fhs(
                uniform_hamiltonians[(alpha, 1)]
            )
        assert abs(float(uniform_flat.query("alpha == 1 and shell_label == '1'").chern_rs.iloc[0]) + 1) < 1e-5
        assert float(uniform_flat.query("alpha == 1 and shell_label == '1'").fhs_lower_band.iloc[0]) == -1
        assert abs(float(uniform_flat.query("alpha == 3 and shell_label == '1'").chern_rs.iloc[0])) < 1e-5
        assert float(uniform_flat.query("alpha == 3 and shell_label == '1'").fhs_lower_band.iloc[0]) == 0
        display(uniform_flat)

        def shell_coordinate(frame):
            return frame["nshell"].fillna(3).astype(float)

        shell_ticks = [0, 1, 2, 3]
        shell_labels = ["0", "1", "2", "full"]
        variant_specs = [
            ("dw_truncation", "DW truncated", BPJ_STYLES[0]),
            ("standard", "standard", BPJ_STYLES[2]),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        ax = axes[0, 0]
        for i, (alpha, label) in enumerate(((1.0, r"topological $\alpha=1$"),
                                             (3.0, r"trivial $\alpha=3$"))):
            g = uniform_flat.query("alpha == @alpha").copy()
            g["shell_x"] = g.shell_label.map({"0": 0, "1": 1, "2": 2, "full": 3})
            g = g.sort_values("shell_x")
            ax.plot(g.shell_x, g.chern_rs, label=label, mfc="white", **BPJ_STYLES[2 * i])
        ax.axhline(-1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xticks=shell_ticks, xticklabels=shell_labels, xlabel=r"shell depth $n_{\rm sh}$", ylabel=r"real-space $\mathcal{C}$")
        ax.legend()
        panel(ax, "a", "same occupied lower band")

        ax = axes[0, 1]
        entropy_specs = [
            ("full_x", r"full strip; target $1/3$", BPJ_STYLES[0], 1 / 3),
            ("wall_window", r"wall window; target $1/6$", BPJ_STYLES[2], 1 / 6),
        ]
        for kind, label, style, target in entropy_specs:
            g = flat_entropy_rows.query("case_key == 'dw_benchmark' and configuration_label == 'dw_benchmark_dw_trunc' and run_name == 'X' and x_interval_kind == @kind").copy()
            g["shell_x"] = shell_coordinate(g)
            g = g.groupby("shell_x", as_index=False).agg(
                mean_slope=("slope", "mean"),
                representative_fit_sem=("slope_err", lambda values: np.sqrt(np.mean(np.square(values)))),
            ).sort_values("shell_x")
            ax.errorbar(g.shell_x, g.mean_slope, yerr=g.representative_fit_sem,
                        label=label, capsize=1.5, mfc="white", **style)
            ax.axhline(target, color=style["color"], ls=":", lw=.65, alpha=.65)
        ax.set(xticks=shell_ticks, xticklabels=shell_labels, xlabel=r"shell depth $n_{\rm sh}$", ylabel=r"entropy slope $m$")
        ax.legend(loc="lower right")
        panel(ax, "b", "universal slopes appear early")

        ax = axes[1, 0]
        for variant, label, style in variant_specs:
            g = flat_corr.query("case_key == 'dw_benchmark' and variant == @variant and run_name == 'X' and observable == 'fixed_dw' and fixed_x == 6").copy()
            g["shell_x"] = shell_coordinate(g)
            g = g.sort_values("shell_x")
            ax.plot(g.shell_x, np.abs(g.slope), label=label, mfc="white", **style)
        ax.axhline(2, color=BPJ_BLACK, ls="--", lw=.8, label=r"$r_y^{-2}$")
        ax.set(xticks=shell_ticks, xticklabels=shell_labels, xlabel=r"shell depth $n_{\rm sh}$", ylabel=r"correlator exponent $\beta$")
        ax.legend()
        panel(ax, "c", "local tails remain nonuniversal")

        ax = axes[1, 1]
        for i, (alpha, label) in enumerate(((1.0, r"topological $\alpha=1$"),
                                             (3.0, r"trivial $\alpha=3$"))):
            g = uniform_flat.query("alpha == @alpha").copy()
            g["shell_x"] = g.shell_label.map({"0": 0, "1": 1, "2": 2, "full": 3})
            g = g.sort_values("shell_x")
            ax.plot(g.shell_x, g.min_abs_energy, label=label, mfc="white", **BPJ_STYLES[2 * i])
        ax.set(xticks=shell_ticks, xticklabels=shell_labels, xlabel=r"shell depth $n_{\rm sh}$", ylabel=r"$\min |E(H_{\rm flat})|$")
        ax.legend()
        panel(ax, "d", "uniform spectral gap")

        fig.subplots_adjust(wspace=.36, hspace=.38)
        DIAGNOSTICS["result_12"] = {
            "occupied_projector_covariance": "G=2*P_occ.conj()-I; real-space and FHS estimators evaluate the same lower-band P_occ",
            "corrected_bug": "the superseded leading minus sign reconstructed I-P_occ in the real-space estimator and was not a sign convention",
            "uniform_flattened_parent": uniform_flat.to_dict("records"),
            "entropy": flat_entropy.query("case_key == 'dw_benchmark' and configuration_label == 'dw_benchmark_dw_trunc' and run_name == 'X'").to_dict("records"),
            "correlator": flat_corr.query("case_key == 'dw_benchmark' and variant in ['dw_truncation', 'standard'] and run_name == 'X' and observable == 'fixed_dw' and fixed_x == 6").to_dict("records"),
        }
        save_figure(fig, "result_12_flattened_shell_benchmark", flat_sources,
                    "deterministic uniform-topology and domain-wall flattened-Hamiltonian benchmark",
                    "The uniform n_shell=1 flattened parent is spectrally gapped, and the same occupied lower-band projector gives real-space Chern response -0.999999829 and FHS integer -1; the alpha=3 control is trivial, while separate domain-wall tables show that entropy coefficients converge earlier than wall-correlator tails.")
        """
    ),
    md(
        r"""
        ## Result 13 — Spatial maps locate topology, slow modes, entanglement, and charge noise

        **Premise.**  Scalar averages hide whether an estimator is attached to the two
        walls, spread through the bulk, or dominated by an artifact.  Four archived maps
        provide complementary spatial checks.

        **Evidence class.** One rendered exact local-Chern calibration and three
        machine-readable adaptive arrays: a $25$-trajectory slow-vector density, a
        $10$-trajectory strip-entanglement contour averaged over all longitudinal
        origins, and a $100$-trajectory purification charge-variance map.
        """
    ),
    code(
        r"""
        CHERN_RENDER = "exact_DW_benchmark_notes/DW_GS_Benchmark_Figs/chern_marker.png"
        LYAP_MAP = "colab_lyapunov/gpu_data/lyapunov_spectra/campaigns/N20_Ny30-50_nsh1_a1-1_S25_cyclesNy/runs/N20x40_DW1_dwtrunc1_a2-30_nsh1_perfect_correction/lyapunov_min_abs_vector.npz"
        ENT_CONTOUR = "colab_small_system_testing/gpu_data/pure_state_strip_entropy_contours/runs/N16x40_nsh1_perfect_correction/Ay_020/strip_entropy_contour_Ay020.npz"
        CHARGE_VAR_MAP = "colab_charge_fluctuations/gpu_data/purification_dynamics_maxmix/campaigns/N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny/runs/N20x40_nsh1_init-maxmix_perfect_correction/local_charge_cell_variance.npz"
        spatial_sources = [
            CHERN_RENDER,
            FLAT_SPEC_NOTEBOOK,
            LYAP_MAP,
            "colab_lyapunov/notebooks/characterization/characterize_lyapunov_spectra_N20_Ny30-50_S25_cyclesNy.ipynb",
            ENT_CONTOUR,
            "colab_small_system_testing/notebooks/characterization/run_pure_state_strip_entropy_contours.ipynb",
            CHARGE_VAR_MAP,
            "colab_charge_fluctuations/notebooks/data_generation/run_purification_dynamics_maxmix_multi_geometry.ipynb",
            "colab_charge_fluctuations/notebooks/characterization/load_purification_charge_fluctuation_data_cpu_edited.ipynb",
        ]

        chern_render = plt.imread(require(CHERN_RENDER))
        lyap_z = np.load(require(LYAP_MAP), allow_pickle=True)
        lyap_vectors = lyap_z["lyapunov_min_abs_vector"].reshape(-1, 40, 20, 2)
        lyap_density_each = (np.abs(lyap_vectors) ** 2).sum(axis=-1)
        lyap_density_each /= lyap_density_each.sum(axis=(1, 2), keepdims=True)
        lyap_density = lyap_density_each.mean(axis=0)

        ent_z = np.load(require(ENT_CONTOUR), allow_pickle=True)
        ent_map = ent_z["entropy_contour_avg"][-1].T / np.log(2)
        ent_cycle = int(ent_z["snapshot_cycles"][-1])

        qvar_z = np.load(require(CHARGE_VAR_MAP), allow_pickle=True)
        qvar_map = qvar_z["local_charge_cell_variance"][:, -1].mean(axis=0).T
        qvar_cycle = int(qvar_z["cycle_labels"][-1])

        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.35))
        ax = axes[0, 0]
        ax.imshow(chern_render, interpolation="nearest")
        ax.axis("off")
        panel(ax, "a")

        ax = axes[0, 1]
        im = ax.imshow(lyap_density, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest", extent=[-.5, 19.5, -.5, 39.5])
        unit_cell_grid(ax, nx=20, ny=40)
        for wall in (5, 15):
            ax.axvline(wall, color=BPJ_RED, ls=":", lw=.7)
        ax.set(xlabel=r"$x$", ylabel=r"$y$")
        cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.05)
        cb.set_label("mean normalized density", fontsize=7)
        cb.ax.tick_params(labelsize=6, direction="in")
        panel(ax, "b", "slow Lyapunov vector")

        ax = axes[1, 0]
        im = ax.imshow(ent_map, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest", extent=[-.5, 15.5, -.5, 19.5])
        unit_cell_grid(ax, nx=16, ny=20)
        for wall in (5, 11):
            ax.axvline(wall, color=BPJ_RED, ls=":", lw=.7)
        ax.set(xlabel=r"$x$", ylabel=r"$\Delta y$ in half strip")
        cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.05)
        cb.set_label(r"entropy contour / $\ln 2$", fontsize=7)
        cb.ax.tick_params(labelsize=6, direction="in")
        panel(ax, "c", rf"adaptive contour, cycle {ent_cycle}")

        ax = axes[1, 1]
        im = ax.imshow(qvar_map, origin="lower", aspect="auto", cmap="Blues", interpolation="nearest", extent=[-.5, 19.5, -.5, 39.5])
        unit_cell_grid(ax, nx=20, ny=40)
        for wall in (5, 15):
            ax.axvline(wall, color=BPJ_RED, ls=":", lw=.7)
        ax.set(xlabel=r"$x$", ylabel=r"$y$")
        cb = fig.colorbar(im, ax=ax, pad=.02, fraction=.05)
        cb.set_label(r"$\overline{\mathrm{Var}(N_{xy})}$", fontsize=7)
        cb.ax.tick_params(labelsize=6, direction="in")
        panel(ax, "d", rf"max-mix charge noise, cycle {qvar_cycle}")

        fig.subplots_adjust(wspace=.34, hspace=.34)
        lyap_wall = lyap_density[:, [5, 15]].sum()
        ent_wall = ent_map[:, [5, 11]].sum() / ent_map.sum()
        DIAGNOSTICS["result_13"] = {
            "lyapunov_two_column_weight": lyap_wall,
            "entanglement_two_column_fraction": ent_wall,
            "charge_variance_peak": float(qvar_map.max()),
            "rendered_chern_source_shape": list(chern_render.shape),
        }
        save_figure(fig, "result_13_spatial_evidence_maps", spatial_sources,
                    "rendered exact calibration plus canonical adaptive spatial precursors",
                    "Independent topology, tangent, entanglement, and charge-noise maps all resolve the domain-wall geometry rather than a featureless bulk.",
                    format_mode="mixed: preserved local-Chern render plus three rebuilt numeric maps")
        """
    ),
    md(
        r"""
        ## Result 14 — Modular charge propagation is visible before center-of-mass reduction

        **Premise.**  The signed velocities in Result 9 are compressed summaries.  The
        underlying density movies should visibly show how a localized modular packet
        reorganizes along the two-wall subsystem.

        **Evidence class.** Canonical CPU analysis of $S=10$ GPU covariance snapshots at
        $16\times40$, averaged over samples and all longitudinal origins at circuit cycle
        $50$.  The horizontal coordinate is the physical transverse direction and the
        vertical coordinate is position relative to the half-strip origin.
        """
    ),
    code(
        r"""
        MODULAR_MAP = "colab_small_system_testing/analysis_outputs/dynamic_modular_charge_spreading/selected_cycle50/N16x40_nsh1_perfect_correction/sample_y0_avg_hmod_cycle50_charge_spreading.npz"
        modular_map_sources = [
            MODULAR_MAP,
            "colab_small_system_testing/notebooks/characterization/analyze_dynamic_modular_charge_spreading.ipynb",
        ]
        mod_z = np.load(require(MODULAR_MAP), allow_pickle=True)
        mod_times = mod_z["times"]
        requested_times = [0, 8, 16, 32]
        mod_indices = [int(np.argmin(np.abs(mod_times - t))) for t in requested_times]
        mod_maps = [mod_z["N_xy"][0, i].T for i in mod_indices]

        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.05))
        for letter, ax, density, i in zip("abcd", axes.ravel(), mod_maps, mod_indices):
            im = ax.imshow(density, origin="lower", aspect="auto", cmap="Blues", vmin=0, vmax=2,
                           interpolation="nearest", extent=[-.5, 15.5, -.5, 19.5])
            unit_cell_grid(ax, nx=16, ny=20)
            for wall in (5, 11):
                ax.axvline(wall, color=BPJ_RED, ls=":", lw=.7)
            ax.set(xlabel=r"$x$", ylabel=r"$y-y_0$")
            panel(ax, letter, rf"$t_{{\rm mod}}={mod_times[i]:.0f}$")
        fig.subplots_adjust(left=.08, right=.89, bottom=.09, top=.96, wspace=.28, hspace=.34)
        cax = fig.add_axes([.915, .17, .018, .68])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(r"modular density $N(x,y-y_0)$", fontsize=7)
        cb.ax.tick_params(labelsize=6, direction="in")
        DIAGNOSTICS["result_14"] = {
            "times": [float(mod_times[i]) for i in mod_indices],
            "total_charges": [float(d.sum()) for d in mod_maps],
            "minima": [float(d.min()) for d in mod_maps],
            "maxima": [float(d.max()) for d in mod_maps],
        }
        save_figure(fig, "result_14_modular_density_heatmaps", modular_map_sources,
                    "canonical modular-density precursor",
                    "The archived modular evolution resolves spatial propagation and redistribution along the wall subsystem underlying the signed center-of-mass velocities.")
        """
    ),
    md(
        r"""
        ## Result 15 — Typical trajectories cluster near the target, but the tails matter

        **Premise.**  Ensemble means do not show whether every record behaves similarly.
        The final-cycle trajectory table permits a direct distributional audit of the
        effective central charge $c_{\rm eff}=3m_s$, charge drift, real-space Chern
        response, and the mixed-state charge variance.

        **Evidence class.** Canonical $S=100$ perfect-correction trajectories at
        $N_x=20$ and $N_y=30,40,50$, evaluated at $C=2N_y$ (cycles $60,80,100$,
        respectively).  The first three observables come from the pure-state feature
        table; $V_Q$ comes from the independently saved max-mix purification trajectories.
        """
    ),
    code(
        r"""
        FEATURE_TABLE = "colab_charge_fluctuations/analysis_outputs/perfect_correction_slope_excess_diagnostics/tables/slope_excess_feature_table.csv"
        VQ_BASE = "colab_charge_fluctuations/gpu_data/purification_dynamics_maxmix/campaigns/N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny/runs"
        distribution_sources = [
            FEATURE_TABLE,
            "colab_charge_fluctuations/notebooks/characterization/diagnose_perfect_correction_slope_excess_cpu.ipynb",
            "colab_charge_fluctuations/notebooks/characterization/load_purification_charge_fluctuation_data_cpu_edited.ipynb",
        ]
        features = pd.read_csv(require(FEATURE_TABLE))
        final_features = features.sort_values("cycle_label").groupby(["Ny", "sample_index"], as_index=False).tail(1).copy()
        final_features["c_eff"] = 3 * final_features["entropy_slope"]
        distribution_cycles = final_features.groupby("Ny").cycle_label.unique().apply(lambda values: int(values[0])).to_dict()
        assert distribution_cycles == {30: 60, 40: 80, 50: 100}

        final_vq = {}
        for ny in (30, 40, 50):
            rel = f"{VQ_BASE}/N20x{ny}_nsh1_init-maxmix_perfect_correction/total_charge_variance.npz"
            distribution_sources.append(rel)
            z = np.load(require(rel), allow_pickle=True)
            final_vq[ny] = np.asarray(z["total_charge_variance"][:, -1], float)

        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.05))
        ranges = {
            "c_eff": np.linspace(.75, 1.6, 24),
            "abs_q_pct": np.linspace(0, 1.6, 18),
            "real_space_chern": np.linspace(.4, 1.01, 25),
            "log_vq": np.linspace(-3.0, -.2, 24),
        }
        histogram_colors = [BPJ_STYLES[i]["color"] for i in range(3)]
        histogram_labels = [rf"$N_y={ny}$" for ny in (30, 40, 50)]

        def grouped_histogram(ax, value_sets, bins):
            return ax.hist(
                value_sets, bins=bins, density=True, histtype="bar", rwidth=.86,
                color=histogram_colors, edgecolor=BPJ_BLACK, linewidth=.35,
                label=histogram_labels,
            )

        ax = axes[0, 0]
        grouped_histogram(
            ax, [final_features.query("Ny == @ny").c_eff for ny in (30, 40, 50)], ranges["c_eff"]
        )
        ax.axvline(1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"trajectory $c_{\rm eff}=3m_s$", ylabel="density")
        ax.legend()
        panel(ax, "a", "central-charge fit")

        ax = axes[0, 1]
        grouped_histogram(
            ax, [final_features.query("Ny == @ny").abs_q_pct for ny in (30, 40, 50)], ranges["abs_q_pct"]
        )
        ax.set(xlabel=r"$|\Delta Q|/(N_xN_y)$ [percent]", ylabel="density")
        panel(ax, "b", "global charge drift")

        ax = axes[1, 0]
        grouped_histogram(
            ax, [final_features.query("Ny == @ny").real_space_chern for ny in (30, 40, 50)], ranges["real_space_chern"]
        )
        ax.axvline(1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set_yscale("log")
        ax.set(xlabel=r"trajectory real-space $\mathcal{C}$", ylabel="density")
        panel(ax, "c", "rare topological tail")

        ax = axes[1, 1]
        grouped_histogram(
            ax, [np.log10(np.clip(final_vq[ny], 1e-12, None)) for ny in (30, 40, 50)], ranges["log_vq"]
        )
        ax.set(xlabel=r"$\log_{10}V_Q$ at $2N_y$", ylabel="density")
        panel(ax, "d", "purification spread")

        fig.subplots_adjust(wspace=.34, hspace=.36)
        distribution_summary = final_features.groupby("Ny")[["c_eff", "abs_q_pct", "real_space_chern"]].agg(["mean", "std", "median", "min", "max"])
        vq_summary = {ny: {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "median": float(np.median(v)),
                           "min": float(v.min()), "max": float(v.max())} for ny, v in final_vq.items()}
        DIAGNOSTICS["result_15"] = {
            "cycles_by_Ny": distribution_cycles,
            "pure_state": distribution_summary.to_dict(),
            "purification_vq": vq_summary,
        }
        save_figure(fig, "result_15_trajectory_distributions", distribution_sources,
                    "canonical trajectory-resolved ensemble diagnostics",
                    "Most trajectories lie near c=1 and C=1, but finite-size central-charge excess, discrete charge drift, rare low-Chern records, and a broad purification tail remain visible.")
        """
    ),
    md(
        r"""
        ## Result 16 — Ensemble cleaning is observable-dependent and must remain explicit

        **Premise.**  If the central-charge excess were caused only by a small set of bad
        records, conditioning on independently motivated quality metrics would remove it
        robustly.  Ranking trajectories by three candidate metrics tests that idea without
        hiding the retained fraction.

        **Evidence class.** Post hoc diagnostic on the same $S=100$ final-cycle feature
        table.  It is not a production reweighted ensemble: the $10\%$ points contain only
        ten trajectories per circumference and are shown with their standard errors.
        """
    ),
    code(
        r"""
        CLEANING_TABLE = "colab_charge_fluctuations/analysis_outputs/perfect_correction_slope_excess_diagnostics/tables/test2_conditioned_subset_slopes.csv"
        cleaning_sources = [
            CLEANING_TABLE,
            FEATURE_TABLE,
            "colab_charge_fluctuations/notebooks/characterization/diagnose_perfect_correction_slope_excess_cpu.ipynb",
        ]
        cleaning = pd.read_csv(require(CLEANING_TABLE))
        metric_specs = [
            ("abs_q_pct", r"lowest $|\Delta Q|$", BPJ_STYLES[0]),
            ("wall_charge_rms", "lowest wall charge RMS", BPJ_STYLES[1]),
            ("chern_error", r"lowest $|\mathcal{C}-1|$", BPJ_STYLES[2]),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.35), sharey=True)
        for ax, ny, letter in zip(axes, (30, 40, 50), "abc"):
            for metric, label, style in metric_specs:
                g = cleaning.query("Ny == @ny and metric == @metric").sort_values("retained_fraction")
                ax.errorbar(100 * g.retained_fraction, 3 * g.mean_slope, yerr=3 * g.sem_slope,
                            label=label, capsize=1.8, mfc="white", **style)
            ax.axhline(1, color=BPJ_BLACK, ls="--", lw=.8)
            ax.set(xlabel="retained records [percent]", xticks=[10, 30, 50, 100])
            panel(ax, letter, rf"$N_y={ny}$")
        axes[0].set_ylabel(r"conditioned $c_{\rm eff}=3\overline{m}_s$")
        axes[-1].legend(loc="upper right", fontsize=6.1)
        fig.subplots_adjust(wspace=.10, left=.09, right=.99, bottom=.20, top=.88)
        DIAGNOSTICS["result_16"] = cleaning.query("metric in ['abs_q_pct', 'wall_charge_rms', 'chern_error']").to_dict("records")
        save_figure(fig, "result_16_ensemble_cleaning_sensitivity", cleaning_sources,
                    "post hoc trajectory-conditioning diagnostic",
                    "The inferred central-charge excess is not removed uniformly by cleaning: charge-drift conditioning can worsen it, while wall-charge or Chern conditioning helps only at selected sizes and retention fractions.")
        """
    ),
    md(
        r"""
        ## Result 17 — The filling-distribution variance decreases with circumference

        **Premise.**  The trajectory histograms in Result 15 establish typicality only at
        the final cycle.  The archived cell-resolved charge histories permit the sharper
        finite-size question: does the ensemble variance of the size-normalized filling
        deviation narrow as the wall circumference grows?

        We define, for trajectory $\xi$,
        $$
        \delta\nu_\xi(C)=100\,
        \frac{Q_\xi(C)-N_xN_y}{N_xN_y},\qquad
        \sigma_\nu^2(C)=\operatorname{Var}_\xi[\delta\nu_\xi(C)].
        $$
        This is a **sample-to-sample filling-distribution variance in squared percentage
        points**, not the intrinsic quantum variance
        $V_Q=\langle Q^2\rangle-\langle Q\rangle^2$ within a trajectory.

        **Evidence class.** Two independent canonical $S=100$, $N_x=20$ campaigns:
        pure-state perfect correction and maximally mixed purification, each run to
        $C=2N_y$ at $N_y=30,40,50$.  Bands and final-point error bars are leave-one-record-
        out jackknife standard errors of the population-variance estimator.
        """
    ),
    code(
        r"""
        PURE_CHARGE_BASE = "colab_charge_fluctuations/gpu_data/streaming_covariance_observables/campaigns/N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/runs"
        MAXMIX_CHARGE_BASE = "colab_charge_fluctuations/gpu_data/purification_dynamics_maxmix/campaigns/N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny/runs"
        filling_sources = [
            "colab_charge_fluctuations/notebooks/characterization/diagnose_perfect_correction_slope_excess_cpu.ipynb",
            "colab_charge_fluctuations/notebooks/characterization/load_purification_charge_fluctuation_data_cpu_edited.ipynb",
            "colab_charge_fluctuations/analysis_outputs/streaming_covariance_characterization/N20_selected_nsh1_dwtrunc1_init-default_S100_cycles-2Ny/tables/perfect_cycle_summary.csv",
            "colab_charge_fluctuations/analysis_outputs/purification_total_charge_histogram_video_cpu/N20_multi_geometry_nsh1_dwtrunc1_init-maxmix_S100_cycles-2Ny/tables/purification_centered_total_charge_sample_stats_vs_cycle.csv",
        ]

        def load_filling_deviation(base, ny, initial_state, filename, array_key):
            rel = f"{base}/N20x{ny}_nsh1_init-{initial_state}_perfect_correction/{filename}"
            filling_sources.append(rel)
            with np.load(require(rel), allow_pickle=True) as z:
                cycles = np.asarray(z["cycle_labels"], int)
                charge_cells = np.asarray(z[array_key], float)
            assert charge_cells.shape == (100, 2 * ny, 20, ny)
            # Each saved cell charge sums to Q; half filling is Q=Nx*Ny.
            q_pct = 100 * (charge_cells.sum(axis=(2, 3)) - 20 * ny) / (20 * ny)
            return cycles, q_pct

        def variance_and_jackknife_sem(records):
            records = np.asarray(records, float)
            n = records.shape[0]
            assert n == 100 and np.isfinite(records).all()
            total = records.sum(axis=0)
            total_sq = np.square(records).sum(axis=0)
            loo_mean = (total[None, :] - records) / (n - 1)
            loo_var = (total_sq[None, :] - np.square(records)) / (n - 1) - np.square(loo_mean)
            variance = np.var(records, axis=0, ddof=0)
            jackknife_sem = np.sqrt((n - 1) / n * np.square(loo_var - loo_var.mean(axis=0)).sum(axis=0))
            return variance, jackknife_sem

        filling = {"pure": {}, "maxmix": {}}
        for ny in (30, 40, 50):
            filling["pure"][ny] = load_filling_deviation(
                PURE_CHARGE_BASE, ny, "default", "local_charge_cell.npz", "local_charge_cell"
            )
            filling["maxmix"][ny] = load_filling_deviation(
                MAXMIX_CHARGE_BASE, ny, "maxmix", "local_charge_cell_mean.npz", "local_charge_cell_mean"
            )

        fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.48))
        final_rows = []
        for ax, protocol, letter, title in zip(
            axes[:2], ("pure", "maxmix"), "ab", ("pure-state correction", "max-mix purification")
        ):
            for ny, style in zip((30, 40, 50), BPJ_STYLES):
                cycles, q_pct = filling[protocol][ny]
                variance, variance_sem = variance_and_jackknife_sem(q_pct)
                x = cycles / ny
                ax.plot(x, variance, label=rf"$N_y={ny}$", markevery=max(1, len(x) // 8),
                        mfc="white", **style)
                ax.fill_between(x, np.maximum(0, variance - variance_sem), variance + variance_sem,
                                color=style["color"], alpha=.12, linewidth=0)
                final_rows.append(
                    {
                        "protocol": protocol,
                        "Ny": ny,
                        "cycle": int(cycles[-1]),
                        "mean_pct": float(q_pct[:, -1].mean()),
                        "variance_pct2": float(variance[-1]),
                        "variance_jackknife_sem_pct2": float(variance_sem[-1]),
                        "sd_pct": float(np.sqrt(variance[-1])),
                    }
                )
            ax.set(xlabel=r"normalized cycle $C/N_y$", ylabel=r"$\sigma_\nu^2$ [percentage points$^2$]",
                   xlim=(0, 2.02), ylim=(0, None))
            panel(ax, letter, title)
        axes[0].legend(loc="upper right")

        final_filling = pd.DataFrame(final_rows)
        ax = axes[2]
        campaign_styles = {
            "pure": dict(color=BPJ_RED, marker="^", linestyle=":"),
            "maxmix": dict(color=BPJ_BLUE, marker="o", linestyle="-"),
        }
        campaign_labels = {"pure": "pure-state correction", "maxmix": "max-mix purification"}
        for protocol in ("pure", "maxmix"):
            g = final_filling.query("protocol == @protocol").sort_values("Ny")
            ax.errorbar(g.Ny, g.variance_pct2, yerr=g.variance_jackknife_sem_pct2,
                        label=campaign_labels[protocol], capsize=2, mfc="white", **campaign_styles[protocol])
        ax.set(xlabel=r"circumference $N_y$", ylabel=r"$\sigma_\nu^2(2N_y)$ [percentage points$^2$]",
               xticks=[30, 40, 50], ylim=(0, None))
        ax.legend(loc="upper right", fontsize=6.0)
        panel(ax, "c", r"final $C=2N_y$")
        fig.subplots_adjust(left=.075, right=.995, bottom=.22, top=.85, wspace=.38)

        reductions = {}
        for protocol in ("pure", "maxmix"):
            g = final_filling.query("protocol == @protocol").set_index("Ny")
            reductions[protocol] = float(100 * (1 - g.loc[50, "variance_pct2"] / g.loc[30, "variance_pct2"]))
        DIAGNOSTICS["result_17"] = {
            "definition": "100*(Q-Nx*Ny)/(Nx*Ny); population variance across S=100 records",
            "uncertainty": "leave-one-record-out jackknife standard error of population variance",
            "final_rows": final_filling.to_dict("records"),
            "variance_reduction_percent_Ny30_to_Ny50": reductions,
        }
        display(final_filling)
        save_figure(fig, "result_17_filling_variance_scaling", filling_sources,
                    "two canonical trajectory-resolved S=100 charge campaigns",
                    "The size-normalized filling distribution narrows with circumference in both pure-state correction and max-mix purification; its final variance falls by about 44% and 40%, respectively, from Ny=30 to Ny=50.")
        """
    ),
    md(
        r"""
        ## Result 18 — An independent pure-state archive jointly checks entropy and correlations

        **Premise.**  The large-$N_y$ entropy campaign is not the only physical-state
        archive in the Colab folders.  Saved covariance snapshots at two shell depths
        permit an independent joint check of the late-window entropy coefficient, the
        wall correlator, and a spatially matched nonwall control.

        **Evidence class.** Canonical GPU perfect-correction snapshots at $N_x=16$,
        $N_y=30,40$, $n_{\rm shell}=1,2$, $S=10$, and $C=50$, followed by local CPU
        characterization.  Entropy error bars are regression standard errors over the
        $A_y$ fit points.  Correlator error bars are standard errors over ten trajectory
        exponents.  The fit-window audit is shown explicitly because the correlator
        exponent is not stable under arbitrary endpoint changes.
        """
    ),
    code(
        r"""
        SMALL_ENTROPY = "colab_small_system_testing/analysis_outputs/pure_state_entanglement_vs_system_size_cpu/full_x_late_window_log_chord_fit_rows.csv"
        SMALL_CORR_FITS = "colab_charge_fluctuations/analysis_outputs/correlator_scaling_diagnostics/tables/n16_wall_localized_fit_sweep.csv"
        SMALL_CORR_CURVES = "colab_charge_fluctuations/analysis_outputs/correlator_scaling_diagnostics/tables/n16_wall_localized_correlator_curves.csv"
        small_state_sources = [
            SMALL_ENTROPY,
            SMALL_CORR_FITS,
            SMALL_CORR_CURVES,
            "colab_small_system_testing/notebooks/characterization/analyze_pure_state_entanglement_vs_system_size_cpu.ipynb",
            "colab_small_system_testing/notebooks/characterization/analyze_pure_state_square_correlations_cpu.ipynb",
            "colab_charge_fluctuations/notebooks/characterization/diagnose_correlator_scaling_windows_cpu.ipynb",
            "colab_small_system_testing/gpu_data/pure_state_covariance_snapshots/campaign_manifest.json",
        ]
        small_entropy = pd.read_csv(require(SMALL_ENTROPY))
        small_entropy["c_eff"] = 3 * np.log(2) * small_entropy.slope_over_ln2
        small_entropy["c_eff_err"] = 3 * np.log(2) * small_entropy.slope_err_over_ln2
        small_corr = pd.read_csv(require(SMALL_CORR_FITS))
        small_curves = pd.read_csv(require(SMALL_CORR_CURVES))

        fixed_corr = (
            small_corr.query("cycle_label == 50 and r_min == 2 and r_max_label == 'Ny//4'")
            .groupby(["Ny", "nshell", "x_window"], as_index=False)
            .agg(exponent=("correlator_exponent", "mean"),
                 exponent_sem=("correlator_exponent", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
                 mean_r2=("correlator_loglog_r2", "mean"), n=("sample_index", "nunique"))
        )

        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.05))
        shell_styles = {1: BPJ_STYLES[0], 2: BPJ_STYLES[2]}
        shell_labels = {1: r"$n_{\rm shell}=1$", 2: r"$n_{\rm shell}=2$"}

        ax = axes[0, 0]
        for nshell in (1, 2):
            q = small_entropy.query("nshell == @nshell").sort_values("Ny")
            ax.errorbar(q.Ny, q.c_eff, yerr=q.c_eff_err, label=shell_labels[nshell],
                        linestyle="none", capsize=2, mfc="white",
                        color=shell_styles[nshell]["color"], marker=shell_styles[nshell]["marker"])
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"circumference $N_y$", ylabel=r"late-window $c_{\rm eff}$",
               xticks=[30, 40], ylim=(.99, 1.105))
        ax.legend()
        panel(ax, "a", "independent entropy check")

        ax = axes[0, 1]
        spatial_specs = [("both_walls", "wall windows", "white"),
                         ("nonwall_control", "nonwall control", None)]
        for nshell in (1, 2):
            for x_window, label, face in spatial_specs:
                q = fixed_corr.query("nshell == @nshell and x_window == @x_window").sort_values("Ny")
                marker = shell_styles[nshell]["marker"]
                color = shell_styles[nshell]["color"]
                ax.errorbar(q.Ny, q.exponent, yerr=q.exponent_sem, linestyle="none",
                            marker=marker, color=color, markerfacecolor=(face or color), capsize=2,
                            label=f"{shell_labels[nshell]}, {label}")
        ax.axhline(2, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"circumference $N_y$", ylabel=r"squared-correlator exponent $\beta$",
               xticks=[30, 40], ylim=(1.8, 6.8))
        ax.legend(fontsize=5.8)
        panel(ax, "b", r"fixed $2\leq r_y\leq N_y/4$")

        ax = axes[1, 0]
        sensitivity = (
            small_corr.query("Ny == 40 and nshell == 1 and cycle_label == 50 and x_window == 'both_walls' and r_min in [1,2,3,4]")
            .groupby(["r_min", "r_max_label"], as_index=False)
            .correlator_exponent.mean()
        )
        rmax_order = ["Ny//4", "Ny//3", "Ny//2"]
        heat = np.full((4, 3), np.nan)
        for _, row in sensitivity.iterrows():
            if row.r_max_label in rmax_order:
                heat[int(row.r_min) - 1, rmax_order.index(row.r_max_label)] = row.correlator_exponent
        im = ax.imshow(heat, origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=1.5, vmax=5.5)
        ax.set(xticks=np.arange(3), xticklabels=[r"$N_y/4$", r"$N_y/3$", r"$N_y/2$"],
               yticks=np.arange(4), yticklabels=[1, 2, 3, 4], xlabel=r"fit endpoint $r_{\max}$",
               ylabel=r"fit start $r_{\min}$")
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        ax.grid(which="minor", color="white", lw=.65, alpha=.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        fig.colorbar(im, ax=ax, fraction=.047, pad=.03, label=r"mean $\beta$")
        panel(ax, "c", r"window sensitivity, $N_y=40$")

        ax = axes[1, 1]
        q = small_curves.query("Ny == 40 and nshell == 1 and cycle_label == 50 and x_window == 'both_walls'")
        curve = q.groupby("ry").square_correlator.agg(["mean", "sem"]).reset_index()
        curve = curve.query("ry >= 1")
        chord = 40 / np.pi * np.sin(np.pi * curve.ry.to_numpy() / 40)
        ax.errorbar(chord, curve["mean"], yerr=curve["sem"], linestyle="none", marker="o",
                    ms=3.0, color=BPJ_BLUE, mfc="white", capsize=1.4, label="trajectory mean")
        fit_mask = (curve.ry >= 2) & (curve.ry <= 10)
        fit_x = chord[fit_mask]
        fit_y = curve.loc[fit_mask, "mean"].to_numpy()
        slope, intercept = np.polyfit(np.log(fit_x), np.log(fit_y), 1)
        xx = np.geomspace(fit_x.min(), fit_x.max(), 100)
        ax.axvspan(fit_x.min(), fit_x.max(), color="0.90", zorder=0, label="fit window")
        ax.plot(xx, np.exp(intercept) * xx ** slope, color=BPJ_BLACK, ls="--", lw=1.0,
                label=rf"fit $\beta={-slope:.2f}$")
        ax.set(xscale="log", yscale="log", xlabel=r"chord distance $d_y$",
               ylabel=r"wall-window $|G(r_y)|^2$")
        ax.legend(fontsize=5.9)
        panel(ax, "d", r"raw curve and declared fit")

        fig.subplots_adjust(left=.085, right=.99, bottom=.11, top=.94, wspace=.36, hspace=.34)
        DIAGNOSTICS["result_18"] = {
            "entropy": small_entropy[["Ny", "nshell", "c_eff", "c_eff_err", "r2"]].to_dict("records"),
            "fixed_window_correlators": fixed_corr.to_dict("records"),
            "representative_mean_curve_fit_beta": float(-slope),
            "window_sensitivity_matrix": heat,
        }
        save_figure(fig, "result_18_small_system_state_crosscheck", small_state_sources,
                    "canonical small-system pure-state covariance cross-check",
                    "A separate perfect-correction archive gives near-c=1 entropy and a wall-localized near-r^-2 correlator at two shell depths, while the explicit endpoint sweep shows why the correlator is a qualified rather than precision exponent result.")
        """
    ),
    md(
        r"""
        ## Result 19 — Feedback reshapes the finite particle–Choi transfer mode

        **Premise.**  Result 8 used only the scalar transfer exponent.  The same Colab
        campaign saved the finite transfer eigenvectors, allowing their participation and
        spatial edge weight to be audited across the mass sweep.

        **Evidence class.** Canonical GPU $N_x=20$, $N_y=20,30,40$, $S=10$ particle--Choi
        transfer sweep, with per-sample modes mapped from the reduced slab basis back to
        physical unit cells.  These are conditioned transfer modes, despite the legacy
        filenames calling them Lyapunov modes; they are not physical covariance-tangent
        Oseledec vectors.  Open points contain fewer than ten finite samples.
        """
    ),
    code(
        r"""
        TRANSFER_MODE_TABLE = "colab_no_feedback_alpha_sweep_transfer/analysis_outputs/gpu_data_summary/lyapunov_modes/tables/gap_lyapunov_mode_metrics.csv"
        TRANSFER_MODE_NOTEBOOK = "colab_no_feedback_alpha_sweep_transfer/analyze_lyapunov_modes.executed.ipynb"
        TRANSFER_RUNS = "colab_no_feedback_alpha_sweep_transfer/gpu_data/alpha_sweep_transfer_dwtrunc1_no_feedback_and_perfect_correction/campaigns/N20_Ny20-30-40_nsh1_dwtrunc1_init-default_S10_cycles2Ny_alpha-sweep_corr-none-perfect/runs"
        transfer_mode_sources = [TRANSFER_MODE_TABLE, TRANSFER_MODE_NOTEBOOK]
        transfer_modes = pd.read_csv(require(TRANSFER_MODE_TABLE))
        transfer_agg = (
            transfer_modes.groupby(["Ny", "correction", "alpha"], as_index=False)
            .agg(sample_count=("sample_index", "nunique"),
                 edge_weight=("edge_weight_1", "mean"),
                 edge_weight_sem=("edge_weight_1", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
                 participation=("site_participation", "mean"),
                 participation_sem=("site_participation", lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
                 edge_distance=("mean_edge_distance", "mean"))
        )

        def alpha_tag(alpha):
            return f"{float(alpha):g}".replace(".", "p")

        def averaged_gap_density(ny, correction, alpha):
            run = f"N20x{ny}_dwtrunc1_corr_{correction}_alpha_{alpha_tag(alpha)}"
            sample_rels = [f"{TRANSFER_RUNS}/{run}/samples/sample_{i:04d}.npz" for i in range(10)]
            density_sum = np.zeros((20, ny), float)
            count = 0
            for rel in sample_rels:
                path = require(rel)
                transfer_mode_sources.append(rel)
                with np.load(path, allow_pickle=False) as z:
                    exponents = np.asarray(z["finite_lyapunov_exponents"], float)
                    if not len(exponents):
                        continue
                    vectors = np.asarray(z["finite_lyapunov_modes"], complex)
                    basis = np.asarray(z["lyapunov_mode_basis_indices"], int)
                    vector = vectors[:, int(np.argmin(np.abs(exponents)))]
                weights = np.abs(vector) ** 2
                weights /= weights.sum()
                mu = basis % 2
                sites = basis // 2
                x = sites % 20
                y = sites // 20
                density = np.zeros((20, ny), float)
                np.add.at(density, (x, y), weights)
                density_sum += density
                count += 1
            if count == 0:
                raise RuntimeError(f"No finite modes for {run}")
            return density_sum / count, count

        density_none, n_none = averaged_gap_density(40, "none", 1.0)
        density_perfect, n_perfect = averaged_gap_density(40, "perfect", 1.0)
        vmax = max(float(density_none.max()), float(density_perfect.max()))

        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.25))
        ax = axes[0, 0]
        for i, ny in enumerate((20, 30, 40)):
            for correction, suffix, face in (("none", "no feedback", "white"),
                                              ("perfect", "perfect correction", None)):
                q = transfer_agg.query("Ny == @ny and correction == @correction").sort_values("alpha")
                style = BPJ_STYLES[i]
                ax.errorbar(q.alpha, q.edge_weight, yerr=q.edge_weight_sem, linestyle=style["linestyle"],
                            color=style["color"], marker=style["marker"],
                            mfc=(face or style["color"]), capsize=1.5,
                            label=rf"$N_y={ny}$, {suffix}")
                censored = q.sample_count < 10
                ax.scatter(q.loc[censored, "alpha"], q.loc[censored, "edge_weight"], s=30,
                           facecolors="white", edgecolors=style["color"], marker=style["marker"], zorder=4)
        ax.axvline(2, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel="weight within one slab-edge column",
               ylim=(0, 1.02))
        ax.legend(ncol=2, fontsize=5.3)
        panel(ax, "a", "transfer-mode edge localization")

        ax = axes[0, 1]
        for i, ny in enumerate((20, 30, 40)):
            for correction, suffix, face in (("none", "no feedback", "white"),
                                              ("perfect", "perfect correction", None)):
                q = transfer_agg.query("Ny == @ny and correction == @correction").sort_values("alpha")
                style = BPJ_STYLES[i]
                ax.errorbar(q.alpha, q.participation, yerr=q.participation_sem,
                            linestyle=style["linestyle"], color=style["color"], marker=style["marker"],
                            mfc=(face or style["color"]), capsize=1.5,
                            label=rf"$N_y={ny}$, {suffix}")
        ax.axvline(2, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel="site participation ratio")
        panel(ax, "b", "finite-mode participation")

        for ax, density, letter, title in ((axes[1, 0], density_none, "c", r"no feedback, $\alpha=1$"),
                                           (axes[1, 1], density_perfect, "d", r"perfect correction, $\alpha=1$")):
            im = ax.imshow(density, origin="lower", aspect="equal", cmap="magma", vmin=0, vmax=vmax,
                           extent=(-.5, 39.5, -.5, 19.5), interpolation="none")
            ax.set(xticks=[0, 10, 20, 30, 39], yticks=[0, 5, 10, 15, 19],
                   xlabel=r"periodic coordinate $y$", ylabel=r"transverse coordinate $x$")
            ax.set_xticks(np.arange(-.5, 40, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 20, 1), minor=True)
            ax.grid(which="minor", color="white", lw=.24, alpha=.38)
            ax.tick_params(which="minor", bottom=False, left=False)
            panel(ax, letter, title)
        fig.subplots_adjust(left=.075, right=.91, bottom=.10, top=.94, wspace=.30, hspace=.34)
        cax = fig.add_axes([.93, .105, .017, .285])
        fig.colorbar(im, cax=cax, label="mean gap-mode probability per unit cell")
        DIAGNOSTICS["result_19"] = {
            "aggregate": transfer_agg.to_dict("records"),
            "heatmaps": {"Ny": 40, "alpha": 1.0, "no_feedback_samples": n_none,
                         "perfect_correction_samples": n_perfect,
                         "common_vmax": vmax, "each_density_sum": [float(density_none.sum()), float(density_perfect.sum())]},
            "terminology": "finite particle-Choi transfer modes in the reduced slab basis; not physical tangent Oseledec modes",
        }
        save_figure(fig, "result_19_transfer_mode_localization", transfer_mode_sources,
                    "canonical GPU finite particle-Choi transfer-mode analysis",
                    "Perfect correction sharply localizes the topological-side finite transfer mode near the reduced-slab edges; the localization weakens approaching alpha=2, but this is a conditioned Choi-transfer result rather than a physical tangent eigenmode.")
        """
    ),
    md(
        r"""
        ## Result 20 — The untruncated overlapping interface was executed with correct feedback

        **Premise.**  Turning off `dw_truncation` does not remove the interface when
        `DW=True` and the two spatial controller parameters remain unequal.  It instead
        retains the full top layer and allows the controller projectors to overlap across
        the interface.

        **Evidence class.** Completed canonical CPU perfect-correction comparison at
        $N_x=16$, $N_y=16,24,32$, $S=10$, and $T=2N_y$.  The untruncated arm fixes
        $\alpha_{\rm out}=30$ and sweeps $\alpha_{\rm in}$ through nine values; it is a
        genuine two-interface construction, not a homogeneous `DW=False` control.  Raw
        observables and per-measured-mode ratios are shown separately because the two
        constructions have different active supports.  No post-selection data are used.

        The spatial controller is
        $$
        \alpha(x)=\begin{cases}
        \alpha_{\rm in},&x\in\mathcal I_{\rm top},\\
        \alpha_{\rm out}=30,&x\in\mathcal I_{\rm triv},
        \end{cases}
        $$
        in both arms.  For $O\in\{S_{\rm tot},\mathrm{Tr}[C(\mathbf{1}-C)]\}$, the heat
        maps display
        $$
        \overline O_\gamma=\frac{O_\gamma}{N_{\rm mode}^\gamma},\qquad
        R_O(\alpha,N_y)=\frac{\overline O_{\rm untruncated}}
        {\overline O_{\rm terminated}}.
        $$
        """
    ),
    code(
        r"""
        OVERLAP_TABLE = "colab_charge_fluctuations/analysis_outputs/purification_charge_sharpening_alpha_sweep_cpu/dwtrunc0_vs_dwtrunc1_N16_alpha_sweep/tables/steady_state_comparison.csv"
        overlap_sources = [OVERLAP_TABLE]
        overlap = pd.read_csv(require(OVERLAP_TABLE)).query("protocol == 'perfect_correction'").copy()
        assert set(overlap.truncation) == {"dwtrunc0", "dwtrunc1"}
        assert set(overlap.Ny) == {16, 24, 32}
        assert set(overlap.samples) == {10}
        assert len(overlap.query("truncation == 'dwtrunc0'")) == 27
        display(overlap[["truncation", "Ny", "alpha_topological_region", "total_entropy_mean",
                         "total_charge_variance_mean", "measured_modes"]].head())
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        style_by_trunc = {
            "dwtrunc1": dict(color=BPJ_RED, marker="^", linestyle=":", mfc=BPJ_RED,
                              label="support terminated"),
            "dwtrunc0": dict(color=BPJ_BLUE, marker="o", linestyle="-", mfc="white",
                              label="untruncated overlap"),
        }
        ny_show = 32
        alpha_max = 2.2
        for ax, mean_col, std_col, floor, letter, title, ylabel in (
            (axes[0, 0], "total_entropy_mean", "total_entropy_std", 1e-9,
             "a", rf"$N_y={ny_show}$: entropy", r"final $S_{\rm tot}$"),
            (axes[0, 1], "total_charge_variance_mean", "total_charge_variance_std", 1e-13,
             "b", rf"$N_y={ny_show}$: charge variance", r"final $\mathrm{Tr}[C(\mathbf{1}-C)]$"),
        ):
            for trunc in ("dwtrunc1", "dwtrunc0"):
                q = overlap.query("truncation == @trunc and Ny == @ny_show and alpha_topological_region <= @alpha_max").sort_values("alpha_topological_region")
                y = np.maximum(q[mean_col].to_numpy(), floor)
                err = q[std_col].to_numpy() / np.sqrt(q.samples.to_numpy())
                style = style_by_trunc[trunc]
                ax.errorbar(q.alpha_topological_region, y,
                            yerr=np.vstack([np.minimum(err, .999 * y), err]),
                            color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                            mfc=style["mfc"], capsize=1.5, label=style["label"])
            ax.axvline(2, color=BPJ_BLACK, ls="--", lw=.8, label=r"$\alpha_c=2$")
            ax.set(xlabel=r"$\alpha_{\rm in}$", ylabel=ylabel, yscale="log")
            ax.legend(loc="lower left", fontsize=6.0)
            panel(ax, letter, title)

        subcritical = overlap.query("alpha_topological_region < 2").copy()
        alphas = sorted(subcritical.alpha_topological_region.unique())
        nys = sorted(subcritical.Ny.unique())
        ratio_fields = ("entropy_per_mode_mean", "charge_variance_per_mode_mean")
        ratio_heatmaps = {}
        for field in ratio_fields:
            heat = np.empty((len(alphas), len(nys)))
            for ia, alpha in enumerate(alphas):
                for iny, ny in enumerate(nys):
                    q = subcritical.query("alpha_topological_region == @alpha and Ny == @ny").set_index("truncation")
                    heat[ia, iny] = np.log10((float(q.loc["dwtrunc0", field]) + 1e-14) /
                                             (float(q.loc["dwtrunc1", field]) + 1e-14))
            ratio_heatmaps[field] = heat
        common_lim = max(float(np.max(np.abs(heat))) for heat in ratio_heatmaps.values())
        for ax, field, letter, title in (
            (axes[1, 0], "entropy_per_mode_mean", "c", "entropy density ratio"),
            (axes[1, 1], "charge_variance_per_mode_mean", "d", "charge-variance density ratio"),
        ):
            im = ax.imshow(ratio_heatmaps[field], origin="lower", aspect="auto", cmap="RdBu_r",
                           vmin=-common_lim, vmax=common_lim, interpolation="none")
            ax.set(xticks=np.arange(len(nys)), xticklabels=nys,
                   yticks=np.arange(len(alphas)), yticklabels=[f"{a:g}" for a in alphas],
                   xlabel=r"$N_y$", ylabel=r"$\alpha_{\rm in}$")
            ax.set_xticks(np.arange(-.5, len(nys), 1), minor=True)
            ax.set_yticks(np.arange(-.5, len(alphas), 1), minor=True)
            ax.grid(which="minor", color="white", lw=.55, alpha=.72)
            ax.tick_params(which="minor", bottom=False, left=False)
            panel(ax, letter, title)
        fig.subplots_adjust(left=.085, right=.90, bottom=.11, top=.94, wspace=.34, hspace=.38)
        cax = fig.add_axes([.92, .11, .018, .31])
        fig.colorbar(im, cax=cax, label=r"$\log_{10}$(untruncated / terminated)")
        DIAGNOSTICS["result_20"] = {
            "untruncated_configs": 27,
            "untruncated_trajectories": 270,
            "Ny": nys,
            "alpha_in": sorted(overlap.alpha_topological_region.unique()),
            "alpha_out": 30.0,
            "dw": True,
            "dw_truncation": False,
            "Ny32": overlap.query("Ny == 32").to_dict("records"),
            "ratio_heatmaps": {field: ratio_heatmaps[field].tolist() for field in ratio_fields},
        }
        save_figure(fig, "result_20_untruncated_interface", overlap_sources,
                    "canonical CPU perfect-correction overlapping-interface comparison",
                    "A completed DW=True, alpha_out=30 campaign retains the explicit interface with domain-wall truncation disabled; its purification crossover is sharp near alpha=2 and its subcritical residual is generally smaller than in the support-terminated construction.")
        """
    ),
    md(
        r"""
        ## Result 21 — Subsystem charge fluctuations resolve a near-level-one current sector

        **Premise.**  The covariance snapshots used in Result 18 contain more than the
        archived entropy and correlator reductions.  They also determine the quantum
        charge variance of every full-$x$ interval without rerunning the circuit.  For
        $C=(G+\mathbf{1})/2$ and an interval $A$,
        $$
        F_A^{\rm q}=\operatorname{Var}(Q_A)
        =\operatorname{Tr}[C_A(\mathbf{1}_A-C_A)].
        $$
        Writing $d=2N_x$, $\overline n=N_y^{-1}\operatorname{Tr}C$, and
        $K_r=N_y^{-1}\sum_y\lVert C_{y,y+r}\rVert_F^2$, the average over all periodic
        interval origins is evaluated exactly as
        $$
        \overline F^{\rm q}(A_y)=A_y(\overline n-K_0)
        -\sum_{r=1}^{A_y-1}(A_y-r)(K_r+K_{-r}).
        $$
        A two-wall charged CFT predicts
        $$
        \overline F^{\rm q}(A_y)=\frac{k}{\pi^2}
        \log\!\left[\frac{N_y}{\pi}\sin\!\left(\frac{\pi A_y}{N_y}\right)\right]+b,
        \qquad k_{\rm eff}=\pi^2m_F.
        $$

        **Evidence class.** Canonical GPU perfect-correction raw covariance snapshots at
        $N_x=16$, $N_y=30,40$, $n_{\rm shell}=1,2$, $S=10$, and cycle 50.  Every
        interval curve is computed trajectory first and then averaged over trajectories;
        periodic-origin averaging remains internal to one trajectory.  The size-panel
        bars and histogram use the ten trajectory-resolved fits.  No post-selection data
        are used.  This is a direct charge-sector precursor, but it has only two sizes and
        shares its ensemble with Result 18 rather than providing an independent archive.
        """
    ),
    code(
        r"""
        CHARGE_SNAPSHOT_BASE = "colab_small_system_testing/gpu_data/pure_state_covariance_snapshots"
        CHARGE_ENTROPY_TABLE = "colab_small_system_testing/analysis_outputs/pure_state_entanglement_vs_system_size_cpu/full_x_late_window_log_chord_fit_rows.csv"
        charge_cases = [
            {"case_id": f"N16x{ny}_nsh{nshell}_perfect_correction", "Nx": 16, "Ny": ny, "nshell": nshell}
            for ny in (30, 40) for nshell in (1, 2)
        ]
        charge_sources = [
            f"{CHARGE_SNAPSHOT_BASE}/campaign_manifest.json",
            CHARGE_ENTROPY_TABLE,
            "colab_small_system_testing/notebooks/characterization/analyze_pure_state_entanglement_vs_system_size_cpu.ipynb",
        ]

        def periodic_full_x_charge_curve(G, nx, ny):
            # Exact y-origin average of Tr[C_A(1_A-C_A)] for every Ay <= Ny/2.
            nlayer = 2 * nx * ny
            if G.shape != (nlayer, nlayer):
                raise ValueError(f"Expected {(nlayer, nlayer)}, received {G.shape}")
            C = np.asarray(G, np.complex128).copy()
            C.flat[:: nlayer + 1] += 1.0
            C *= 0.5
            d = 2 * nx
            blocks = C.reshape(ny, d, ny, d)
            rows = np.arange(ny)
            K = np.empty(ny, float)
            for r in range(ny):
                block = blocks[rows, :, (rows + r) % ny, :]
                K[r] = np.square(np.abs(block)).sum(axis=(1, 2)).mean()
            nbar = float(np.trace(C).real / ny)
            ay = np.arange(1, ny // 2 + 1, dtype=int)
            variance = np.empty(len(ay), float)
            for i, ell in enumerate(ay):
                cross = sum((ell - r) * (K[r] + K[-r]) for r in range(1, ell))
                variance[i] = ell * (nbar - K[0]) - cross
            return ay, variance, {"nbar": nbar, "K0": float(K[0])}

        charge_curve_rows = []
        charge_fit_rows = []
        charge_formula_checks = []
        for case in charge_cases:
            latest_rel = f"{CHARGE_SNAPSHOT_BASE}/runs/{case['case_id']}/latest_run.json"
            latest = json.loads(require(latest_rel).read_text())
            run_rel = f"colab_small_system_testing/gpu_data/{latest['run_dir_relative']}"
            manifest_rel = f"{run_rel}/manifest.json"
            shard_rel = f"{run_rel}/batch_00000_snapshots.npy"
            charge_sources.extend([latest_rel, manifest_rel, shard_rel])
            manifest = json.loads(require(manifest_rel).read_text())
            cfg = manifest["config"]
            assert cfg["DW"] and cfg["dw_truncation"] and cfg["perfect_correction"]
            assert not cfg["postselect"] and cfg["samples"] == 10
            cycle_index = list(cfg["snapshot_cycles"]).index(50)
            snapshots = np.load(require(shard_rel), mmap_mode="r")
            for sample_index in range(cfg["samples"]):
                ay, variance, auxiliaries = periodic_full_x_charge_curve(
                    snapshots[sample_index, cycle_index], case["Nx"], case["Ny"]
                )
                x = np.log(case["Ny"] / np.pi * np.sin(np.pi * ay / case["Ny"]))
                fit_mask = ay >= 8
                slope, intercept = np.polyfit(x[fit_mask], variance[fit_mask], 1)
                predicted = slope * x[fit_mask] + intercept
                ss_res = float(np.square(variance[fit_mask] - predicted).sum())
                ss_tot = float(np.square(variance[fit_mask] - variance[fit_mask].mean()).sum())
                r2 = 1.0 - ss_res / ss_tot
                charge_fit_rows.append(
                    {**case, "sample_index": sample_index, "cycle": 50,
                     "Ay_fit_min": 8, "Ay_fit_max": int(ay.max()),
                     "slope": float(slope), "intercept": float(intercept),
                     "k_eff": float(np.pi ** 2 * slope), "r2": r2}
                )
                charge_curve_rows.extend(
                    {**case, "sample_index": sample_index, "cycle": 50,
                     "Ay": int(a), "log_chord": float(xi), "charge_variance": float(fi)}
                    for a, xi, fi in zip(ay, x, variance)
                )
                if sample_index == 0:
                    # Direct submatrix traces guard the block-correlation reduction at
                    # the shortest, fit-start, and longest interval.
                    G = np.asarray(snapshots[sample_index, cycle_index])
                    nlayer = G.shape[0]
                    C = G.copy()
                    C.flat[:: nlayer + 1] += 1.0
                    C *= 0.5
                    direct = []
                    for ell in sorted(set((1, 8, case["Ny"] // 2))):
                        vals = []
                        for y0 in range(case["Ny"]):
                            ys = (y0 + np.arange(ell)) % case["Ny"]
                            idx = np.concatenate([np.arange(2 * case["Nx"]) + 2 * case["Nx"] * y for y in ys])
                            CA = C[np.ix_(idx, idx)]
                            vals.append(float(np.trace(CA).real - np.square(np.abs(CA)).sum()))
                        direct.append(abs(float(np.mean(vals)) - float(variance[ell - 1])))
                    charge_formula_checks.append({**case, "max_abs_error": max(direct), **auxiliaries})

        charge_curves = pd.DataFrame(charge_curve_rows)
        charge_fits = pd.DataFrame(charge_fit_rows)
        charge_fit_summary = (
            charge_fits.groupby(["case_id", "Nx", "Ny", "nshell", "cycle"], as_index=False)
            .agg(k_eff=("k_eff", "mean"),
                 k_eff_sem=("k_eff", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))),
                 k_eff_min=("k_eff", "min"), k_eff_max=("k_eff", "max"),
                 mean_r2=("r2", "mean"), trajectories=("sample_index", "nunique"))
        )
        charge_curve_summary = (
            charge_curves.groupby(["case_id", "Nx", "Ny", "nshell", "cycle", "Ay", "log_chord"], as_index=False)
            .agg(charge_variance=("charge_variance", "mean"),
                 charge_variance_sem=("charge_variance", lambda s: float(s.std(ddof=1) / np.sqrt(len(s)))))
        )
        charge_entropy = pd.read_csv(require(CHARGE_ENTROPY_TABLE))
        charge_entropy["c_eff"] = 3 * np.log(2) * charge_entropy.slope_over_ln2
        charge_entropy["c_eff_err"] = 3 * np.log(2) * charge_entropy.slope_err_over_ln2
        charge_joint = charge_fit_summary.merge(
            charge_entropy[["case_id", "c_eff", "c_eff_err"]], on="case_id", validate="one_to_one"
        )
        assert max(row["max_abs_error"] for row in charge_formula_checks) < 1e-10
        display(charge_joint)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
        shell_styles = {1: BPJ_STYLES[0], 2: BPJ_STYLES[2]}
        shell_labels = {1: r"$n_{\rm shell}=1$", 2: r"$n_{\rm shell}=2$"}

        ax = axes[0, 0]
        representative = charge_curve_summary.query("Ny == 40 and nshell == 1").sort_values("Ay")
        fit = charge_joint.query("Ny == 40 and nshell == 1").iloc[0]
        ax.errorbar(representative.log_chord, representative.charge_variance,
                    yerr=representative.charge_variance_sem, linestyle="none", marker="o",
                    ms=3.2, color=BPJ_BLUE, mfc="white", capsize=1.4,
                    label="trajectory mean")
        fit_mask = representative.Ay >= 8
        ax.axvspan(float(representative.loc[fit_mask, "log_chord"].min()),
                   float(representative.loc[fit_mask, "log_chord"].max()),
                   color="0.90", zorder=0, label=r"fit window $A_y\geq8$")
        fit_points = charge_curves.query("Ny == 40 and nshell == 1 and Ay >= 8")
        slope, intercept = np.polyfit(fit_points.log_chord, fit_points.charge_variance, 1)
        xx = np.linspace(representative.log_chord.min(), representative.log_chord.max(), 160)
        ax.plot(xx, slope * xx + intercept, color=BPJ_BLACK, ls="--", lw=1.0,
                label=rf"fit $k_{{\rm eff}}={fit.k_eff:.3f}$")
        ax.set(xlabel=r"$\log[(N_y/\pi)\sin(\pi A_y/N_y)]$",
               ylabel=r"$\langle F_A^{\rm q}\rangle_\xi$")
        ax.legend(loc="upper left", fontsize=5.9)
        panel(ax, "a", r"raw interval curve, cycle 50")

        ax = axes[0, 1]
        for nshell in (1, 2):
            q = charge_joint.query("nshell == @nshell").sort_values("Ny")
            style = shell_styles[nshell]
            ax.errorbar(q.Ny, q.k_eff, yerr=q.k_eff_sem, linestyle="none",
                        color=style["color"], marker=style["marker"], mfc="white",
                        capsize=2, label=shell_labels[nshell])
        ax.axhline(1, color=BPJ_BLACK, ls="--", lw=.8, label=r"$U(1)_1$")
        ax.set(xlabel=r"circumference $N_y$", ylabel=r"charge level $k_{\rm eff}$",
               xticks=[30, 40], ylim=(.94, 1.15))
        ax.legend(fontsize=6.0)
        panel(ax, "b", r"trajectory-resolved fits, cycle 50")

        ax = axes[1, 0]
        for nshell in (1, 2):
            q = charge_joint.query("nshell == @nshell").sort_values("Ny")
            style = shell_styles[nshell]
            ax.errorbar(q.c_eff, q.k_eff, xerr=q.c_eff_err, yerr=q.k_eff_sem,
                        linestyle="none", color=style["color"], marker=style["marker"],
                        mfc="white", capsize=2, label=shell_labels[nshell])
            for _, row in q.iterrows():
                ax.annotate(rf"${int(row.Ny)}$", (row.c_eff, row.k_eff),
                            xytext=(3, -6), textcoords="offset points", fontsize=6)
        lo, hi = .98, 1.15
        ax.plot([lo, hi], [lo, hi], color=BPJ_BLACK, ls="--", lw=.8, label=r"$k=c$")
        ax.set(xlabel=r"entropy coefficient $c_{\rm eff}$",
               ylabel=r"charge level $k_{\rm eff}$", xlim=(lo, hi), ylim=(lo, hi))
        ax.legend(fontsize=6.0)
        panel(ax, "c", r"matched entropy--charge test")

        ax = axes[1, 1]
        histogram_values = [
            charge_fits.query("Ny == 40 and nshell == @nshell").k_eff.to_numpy()
            for nshell in (1, 2)
        ]
        bins = np.linspace(.80, 1.52, 10)
        ax.hist(histogram_values, bins=bins, histtype="bar", rwidth=.84,
                color=[BPJ_RED, BPJ_BLUE], edgecolor=BPJ_BLACK, linewidth=.55,
                alpha=.72, label=[shell_labels[1], shell_labels[2]])
        ax.axvline(1, color=BPJ_BLACK, ls="--", lw=.8)
        ax.set(xlabel=r"trajectory $k_{\rm eff,\xi}$", ylabel="trajectory count",
               xticks=[.8, 1.0, 1.2, 1.4], ylim=(0, None))
        ax.legend(fontsize=6.0)
        panel(ax, "d", r"$N_y=40$, cycle 50")

        fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.94, wspace=.34, hspace=.36)
        DIAGNOSTICS["result_21"] = {
            "definition": "full-x periodic-interval quantum variance, averaged over y origins within trajectory",
            "fit": "Ay >= 8 against log[(Ny/pi) sin(pi Ay/Ny)]; k_eff=pi^2*slope",
            "uncertainty": "standard error over ten trajectory-resolved fit coefficients",
            "formula_checks": charge_formula_checks,
            "fit_summary": charge_joint.to_dict("records"),
            "trajectory_fits": charge_fits.to_dict("records"),
        }
        save_figure(fig, "result_21_subsystem_charge_fluctuations", charge_sources,
                    "canonical GPU perfect-correction trajectory-resolved subsystem charge fluctuations",
                    "The same pure-state snapshots that give near-c=1 entropy also give logarithmic full-strip quantum charge variance with k_eff near one at both sizes and shell depths, providing a direct current-sector precursor without partial post-selection.")
        """
    ),
    md(
        r"""
        ## Result 22 — A three-by-three OW window preserves the target-band overlap obstruction

        **Premise.**  The circuit uses compact overcomplete-Wannier (OW) measurement
        modes, so the finite support must be tested against the target Chern band rather
        than justified only from a tail length.  If $f(\mathbf{k})=\langle
        u_-(\mathbf{k})|\tau\rangle$, a real-space mask $g_n$ produces
        $$
        f_n^{\rm eff}(\mathbf{k}')=\frac{1}{N}\sum_{\mathbf{k}}
        \widetilde g_n(\mathbf{k}'-\mathbf{k})f(\mathbf{k})
        \langle u_-(\mathbf{k}')|u_-(\mathbf{k})\rangle .
        $$
        The zero and local winding of this overlap diagnose whether the measurement mode
        still encounters the target-band obstruction.  They are not the Chern number of
        the compact-support spinor
        $|\psi_n(\mathbf{k})\rangle=\sum_{\mathbf r\in\mathrm{supp}(g_n)}
        e^{-i\mathbf{k}\cdot\mathbf r}C_{\mathbf r}|\tau\rangle$.  If that spinor is
        smooth, periodic, and nonzero, it is a global section and has $C_{\psi_n}=0$.

        **Evidence class.** Deterministic reconstruction of the documented
        `windowed_chern` calculation for the uniform $\alpha=1$ two-band projector and
        $X$ trial spinor.  The notebook recomputes the projector Fourier coefficients,
        retained real-space norm, continuum-$k$ overlap cuts, local phase winding, and
        stable compact-spinor FHS diagnostic.  No circuit trajectory and no
        post-selection data enter this controller-design result.
        """
    ),
    code(
        r"""
        from scipy.optimize import minimize, minimize_scalar

        FORM_FACTOR_NOTE = "form_factor_analysis/docs/windowed_chern.tex"
        FORM_FACTOR_PDF = "form_factor_analysis/docs/windowed_chern.pdf"
        FORM_FACTOR_NOTEBOOK = "form_factor_analysis/form_factor_analysis.ipynb"
        form_factor_sources = [
            FORM_FACTOR_NOTE, FORM_FACTOR_PDF, FORM_FACTOR_NOTEBOOK,
            "form_factor_analysis/figs/continuum_shifted_zeros.png",
            "form_factor_analysis/figs/effective_form_factor_maps.png",
        ]
        for source in form_factor_sources:
            require(source)

        OW_ALPHA = 1.0
        OW_GRID = 181
        ow_tau = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
        ow_ref = np.array([1.0, 0.0], dtype=np.complex128)
        ow_sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        ow_sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        ow_sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        ow_eye = np.eye(2, dtype=np.complex128)

        def ow_projector(kx, ky):
            kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
            nx, ny = np.sin(kx), np.sin(ky)
            nz = OW_ALPHA - np.cos(kx) - np.cos(ky)
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            hhat = (nx[..., None, None] * ow_sx + ny[..., None, None] * ow_sy
                    + nz[..., None, None] * ow_sz) / norm[..., None, None]
            return 0.5 * (ow_eye - hhat)

        def ow_support(nshell):
            if np.isclose(nshell, 0.5):
                return [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
            n = int(round(nshell))
            return [(rx, ry) for rx in range(-n, n + 1) for ry in range(-n, n + 1)]

        ow_k = 2 * np.pi * (np.arange(OW_GRID) / OW_GRID - 0.5)
        OW_KX, OW_KY = np.meshgrid(ow_k, ow_k, indexing="ij")
        ow_projector_grid = ow_projector(OW_KX, OW_KY)

        def ow_coefficients(nshell):
            return {
                (rx, ry): np.mean(
                    np.exp(1.0j * (OW_KX * rx + OW_KY * ry))[..., None, None]
                    * ow_projector_grid,
                    axis=(0, 1),
                )
                for rx, ry in ow_support(nshell)
            }

        def ow_spinor(kx, ky, coeffs):
            kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
            value = np.zeros(kx.shape + (2,), dtype=np.complex128)
            for (rx, ry), coefficient in coeffs.items():
                value += (np.exp(-1.0j * (kx * rx + ky * ry))[..., None]
                          * (coefficient @ ow_tau))
            return value

        def ow_overlap_abs(kx, ky, coeffs=None):
            projector = ow_projector(kx, ky)
            spinor = (np.broadcast_to(ow_tau, projector.shape[:-1]) if coeffs is None
                      else ow_spinor(kx, ky, coeffs))
            weight = np.real(np.einsum(
                "...a,...ab,...b->...", spinor.conj(), projector, spinor, optimize=True
            ))
            return np.sqrt(np.maximum(weight, 0.0))

        # Real-space OW norm fractions from the unwindowed projected trial mode.
        projected_trial = np.einsum("...ab,b->...a", ow_projector_grid, ow_tau, optimize=True)
        ow_real_space = np.fft.ifft2(projected_trial, axes=(0, 1))
        ow_weight = np.square(np.abs(ow_real_space)).sum(axis=-1)
        ow_weight /= ow_weight.sum()

        def retained_norm(nshell):
            return float(sum(ow_weight[rx % OW_GRID, ry % OW_GRID]
                             for rx, ry in ow_support(nshell)))

        def local_overlap_winding(kx_zero, coeffs, radius=0.02):
            theta = np.linspace(0, 2 * np.pi, 721)
            kx = kx_zero + radius * np.cos(theta)
            ky = radius * np.sin(theta)
            projector = ow_projector(kx, ky)
            local_frame = np.einsum("...ab,b->...a", projector, ow_ref, optimize=True)
            local_frame /= np.linalg.norm(local_frame, axis=-1)[..., None]
            overlap = np.einsum(
                "...a,...a->...", local_frame.conj(), ow_spinor(kx, ky, coeffs), optimize=True
            )
            return float(np.rint((np.unwrap(np.angle(overlap))[-1]
                                  - np.unwrap(np.angle(overlap))[0]) / (2 * np.pi)))

        def compact_spinor_fhs(coeffs):
            spinor = ow_spinor(OW_KX, OW_KY, coeffs)
            norm = np.linalg.norm(spinor, axis=-1)
            unit = spinor / norm[..., None]
            overlap_x = np.einsum("...a,...a->...", unit.conj(), np.roll(unit, -1, axis=0))
            overlap_y = np.einsum("...a,...a->...", unit.conj(), np.roll(unit, -1, axis=1))
            Ux, Uy = overlap_x / np.abs(overlap_x), overlap_y / np.abs(overlap_y)
            plaquette = (Ux * np.roll(Uy, -1, axis=0)
                         * np.conj(np.roll(Ux, -1, axis=1)) * np.conj(Uy))
            return float(np.angle(plaquette).sum() / (2 * np.pi))

        ow_shells = [0, 0.5, 1, 2, 4, 8]
        ow_coeffs = {nshell: ow_coefficients(nshell) for nshell in ow_shells}
        ow_rows = []
        for nshell in ow_shells:
            bounds = ((0.05 * np.pi, 0.35 * np.pi) if nshell == 0
                      else (0.40 * np.pi, 0.55 * np.pi))
            root = minimize_scalar(
                lambda kx: float(ow_overlap_abs(kx, 0.0, ow_coeffs[nshell])),
                bounds=bounds, method="bounded", options={"xatol": 1e-14},
            )
            norm_minimum = minimize(
                lambda point: float(np.linalg.norm(ow_spinor(point[0], point[1], ow_coeffs[nshell]))),
                x0=np.array([0.5 * np.pi, 0.0]), method="Nelder-Mead",
                options={"xatol": 1e-12, "fatol": 1e-12, "maxiter": 10000},
            )
            stable_fhs = compact_spinor_fhs(ow_coeffs[nshell]) if nshell <= 2 else np.nan
            ow_rows.append({
                "nshell": nshell,
                "support_cells": len(ow_support(nshell)),
                "retained_norm": retained_norm(nshell),
                "zero_kx_over_pi": float(root.x / np.pi),
                "zero_residual": float(root.fun),
                "overlap_winding": local_overlap_winding(root.x, ow_coeffs[nshell]),
                "compact_spinor_min_norm": float(norm_minimum.fun),
                "compact_spinor_fhs_stable": stable_fhs,
            })
        ow_summary = pd.DataFrame(ow_rows)
        ow_scales = {
            "xi_top_over_a": float(1 / np.sqrt(np.pi)),
            "xi_rms_over_a": float(np.sqrt(np.sum(
                ow_weight * (((np.arange(OW_GRID)[:, None] + OW_GRID // 2) % OW_GRID - OW_GRID // 2) ** 2
                             + ((np.arange(OW_GRID)[None, :] + OW_GRID // 2) % OW_GRID - OW_GRID // 2) ** 2)
            ))),
            "xi_exp_over_a": float(1 / np.log(2)),
        }
        assert abs(float(ow_summary.query("nshell == 1").retained_norm.iloc[0]) - 0.992155) < 2e-6
        assert abs(float(ow_summary.query("nshell == 1").zero_kx_over_pi.iloc[0]) - 0.4922) < 5e-4
        assert set(ow_summary.query("nshell <= 2").overlap_winding) == {-1.0}
        assert np.max(np.abs(ow_summary.query("nshell <= 2").compact_spinor_fhs_stable)) < 1e-10
        display(ow_summary)
        print(ow_scales)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))

        ax = axes[0, 0]
        ax.plot(ow_summary.nshell, ow_summary.retained_norm, color=BPJ_BLUE,
                marker="o", mfc="white", linestyle="-", label="square/cross mask")
        ax.axhline(.99, color=BPJ_BLACK, ls="--", lw=.8, label="99% retained")
        row_one = ow_summary.query("nshell == 1").iloc[0]
        ax.annotate(rf"$n=1:\ {100*row_one.retained_norm:.4f}\%$",
                    (1, row_one.retained_norm), xytext=(8, -18), textcoords="offset points",
                    fontsize=6.4, arrowprops=dict(arrowstyle="-", lw=.6, color=BPJ_BLACK))
        ax.text(.97, .12,
                (rf"$\xi_{{\rm top}}={ow_scales['xi_top_over_a']:.3f}a$" "\n"
                 rf"$\xi_{{\rm rms}}={ow_scales['xi_rms_over_a']:.3f}a$" "\n"
                 rf"$\xi_{{\rm exp}}={ow_scales['xi_exp_over_a']:.3f}a$"),
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3)
        ax.set(xlabel=r"support radius $n_{\rm shell}$", ylabel="retained OW norm",
               xticks=[0, .5, 1, 2, 4, 8], ylim=(.58, 1.015))
        ax.legend(loc="center right", fontsize=5.8)
        panel(ax, "a", r"locality of the $\alpha=1$ OW mode")

        ax = axes[0, 1]
        cut_kx_pi = np.linspace(0.0, .56, 500)
        cut_kx = np.pi * cut_kx_pi
        ax.plot(cut_kx_pi, ow_overlap_abs(cut_kx, 0.0), color=BPJ_BLACK,
                ls="--", lw=1.0, label="unwindowed")
        cut_specs = [(0, BPJ_GRAY, "v", ":"), (1, BPJ_RED, "^", ":"),
                     (2, BPJ_BLUE, "o", "-"), (8, BPJ_GREEN, "s", "--")]
        for nshell, color, marker, linestyle in cut_specs:
            ax.plot(cut_kx_pi, ow_overlap_abs(cut_kx, 0.0, ow_coeffs[nshell]),
                    color=color, marker=marker, markevery=70, ms=2.7, mfc="white",
                    linestyle=linestyle, label=rf"$n_{{\rm shell}}={nshell}$")
            zero = float(ow_summary.query("nshell == @nshell").zero_kx_over_pi.iloc[0])
            ax.plot(zero, 0, marker=marker, color=color, ms=4, clip_on=False)
        ax.set(xlabel=r"$k_x/\pi$ at $k_y=0$", ylabel=r"$|f_n^{\rm eff}(k_x,0)|$",
               xlim=(0, .56), ylim=(0, .76))
        ax.legend(loc="upper right", ncol=2, fontsize=5.5)
        panel(ax, "b", "continuum overlap zeros")

        ax = axes[1, 0]
        ax.axis("off")
        ax.text(-.16, 1.08, "(c)", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
        ax.text(.5, 1.04, r"target overlap amplitude in the BZ", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)
        map_k = np.linspace(-np.pi, np.pi, 121, endpoint=False)
        MAP_KX, MAP_KY = np.meshgrid(map_k, map_k, indexing="ij")
        map_exact = ow_overlap_abs(MAP_KX, MAP_KY)
        map_one = ow_overlap_abs(MAP_KX, MAP_KY, ow_coeffs[1])
        map_axes = [ax.inset_axes([.00, .10, .42, .80]), ax.inset_axes([.48, .10, .42, .80])]
        for map_ax, field, title, zero in (
            (map_axes[0], map_exact, "unwindowed", .5),
            (map_axes[1], map_one, r"$n_{\rm shell}=1$", float(row_one.zero_kx_over_pi)),
        ):
            im = map_ax.imshow(field, origin="lower", extent=(-1, 1, -1, 1),
                               cmap="viridis", vmin=0, vmax=1, interpolation="none", aspect="equal")
            map_ax.plot(0, zero, marker="x", color="white", ms=4.5, mew=.9)
            map_ax.set(title=title, xlabel=r"$k_y/\pi$", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        map_axes[0].set_ylabel(r"$k_x/\pi$")
        map_axes[1].set_yticklabels([])
        cax = ax.inset_axes([.93, .15, .025, .68])
        fig.colorbar(im, cax=cax, label=r"$|f_n^{\rm eff}|$")

        ax = axes[1, 1]
        stable = ow_summary.query("nshell <= 2").copy()
        xpos = np.arange(len(stable))
        ax.plot(xpos, stable.overlap_winding, linestyle="none", color=BPJ_RED,
                marker="^", mfc="white", ms=5, label=r"overlap winding $\nu_n$")
        ax.plot(xpos, stable.compact_spinor_fhs_stable, linestyle="none", color=BPJ_BLUE,
                marker="o", mfc="white", ms=5, label=r"compact-spinor $C_{\psi_n}$")
        ax.axhline(-1, color=BPJ_GRAY, ls=":", lw=.7)
        ax.axhline(0, color=BPJ_GRAY, ls=":", lw=.7)
        ax.set(xlabel=r"support radius $n_{\rm shell}$", ylabel="integer diagnostic",
               xticks=xpos, xticklabels=["0", "1/2", "1", "2"], yticks=[-1, 0],
               ylim=(-1.25, .25))
        ax.legend(loc="center right", fontsize=5.8)
        panel(ax, "d", "overlap obstruction versus bundle topology")

        fig.subplots_adjust(left=.085, right=.985, bottom=.10, top=.94, wspace=.34, hspace=.36)
        DIAGNOSTICS["result_22"] = {
            "model": "uniform two-band alpha=1 projector with X trial spinor",
            "projector_fourier_grid": OW_GRID,
            "localization_scales": ow_scales,
            "shell_summary": ow_summary.to_dict("records"),
            "interpretation": "nu_n is target-band overlap winding; stable C_psi is the compact-spinor FHS Chern number",
            "excluded_interpretation": "coarse-grid nonzero compact-spinor FHS values at n=4,8 are unresolved singular-limit diagnostics",
        }
        save_figure(fig, "result_22_windowed_chern_form_factor", form_factor_sources,
                    "deterministic finite-support overcomplete-Wannier form-factor reconstruction",
                    "At alpha=1 the n_shell=1 three-by-three OW mask retains 99.2155% of the unwindowed norm and preserves the target-band overlap zero with winding -1 near kx/pi=0.4922, while the smooth compact-support spinor itself has Chern number zero.",
                    format_mode="recomputed from the documented deterministic windowed-Chern model")
        """
    ),
    md(
        r"""
        ## Result 23 — The overlap winding follows the target Bloch-bundle phase

        **Bundle statement and proof.**  On overlapping momentum patches, lower-band
        frames obey $|u_j\rangle=e^{i\chi_{ij}}|u_i\rangle$.  For any smooth periodic
        trial field $|\phi_n(\mathbf{k})\rangle$, its band overlap transforms as
        $f_{n,j}=e^{-i\chi_{ij}}f_{n,i}$.  Remove small counterclockwise disks $D_a$
        around its isolated zeros and write $M=T^2\setminus\bigcup_aD_a$.  On $M$, the
        normalized projected trial state is a single-valued occupied-band frame,
        $$
        |v_n\rangle=\frac{P_-|\phi_n\rangle}{\|P_-|\phi_n\rangle\|}
        =e^{i\arg f_{n,i}}|u_i\rangle .
        $$
        With $A_i=-i\langle u_i|d u_i\rangle$, this frame has
        $A_v=A_i+d\arg f_{n,i}$ and the same curvature $F=dA_i$.  Stokes' theorem on
        the punctured torus gives
        $$
        2\pi C=\int_MF=\oint_{\partial M}A_v
        =-\sum_a\oint_{\gamma_a}(A_i+d\arg f_{n,i})
        \longrightarrow-2\pi\sum_a\nu_a.
        $$
        The minus sign appears because each hole is clockwise in $\partial M$, whereas
        $\gamma_a$ defines winding counterclockwise; the smooth $A_i$ contribution
        vanishes as the disks shrink.  Hence $\sum_a\nu_a=-C$ without invoking the
        zero-index theorem as a black box.

        Thus a nonzero target-band Chern number forces overlap zeros even though the
        globally defined compact spinor itself is topologically trivial.  In particular,
        $n_{\rm shell}=0$ has compact-spinor Chern number zero but its target-band overlap
        still winds by $-1$ when $C=+1$.  When $C=0$, zeros are not forbidden, but their
        total index must vanish.

        **Numerical protocol.**  For each requested $\alpha$ and support, the calculation
        evaluates the gauge-invariant magnitude
        $|f_n|=[\langle\phi_n|P_-|\phi_n\rangle]^{1/2}$ on a $181\times181$ periodic grid,
        selects all discrete local minima, refines the lowest candidates by periodic
        two-dimensional minimization, and merges coincident solutions.  Any refined
        minimum below $10^{-6}$ is enclosed by a radius-$0.02$ loop sampled at 721 angles.
        One reference spinor is selected at the loop center and held fixed around the
        loop to make a smooth local frame; the unwrapped phase circulation gives the
        integer $\nu_a$.  The lower-band $C$ is independently checked by an FHS surface
        integral.  A positive global minimum is reported as a gapped overlap with total
        winding zero rather than as a vortex around an arbitrary minimum.

        **Evidence class.** Deterministic parameter sweep of the same two-band projector
        and $X$ trial mode as Result 22.  `none` means the unwindowed trial overlap;
        $n_{\rm shell}=0,1,2$ are literal square real-space masks.  No trajectory or
        post-selection data enter.
        """
    ),
    code(
        r"""
        def ow_projector_for_alpha(kx, ky, alpha):
            kx, ky = np.broadcast_arrays(np.asarray(kx, float), np.asarray(ky, float))
            nx, ny = np.sin(kx), np.sin(ky)
            nz = alpha - np.cos(kx) - np.cos(ky)
            norm = np.sqrt(nx * nx + ny * ny + nz * nz)
            hhat = (nx[..., None, None] * ow_sx + ny[..., None, None] * ow_sy
                    + nz[..., None, None] * ow_sz) / norm[..., None, None]
            return 0.5 * (ow_eye - hhat)

        def ow_coefficients_for_alpha(alpha, nshell):
            projector_grid = ow_projector_for_alpha(OW_KX, OW_KY, alpha)
            return {
                (rx, ry): np.mean(
                    np.exp(1.0j * (OW_KX * rx + OW_KY * ry))[..., None, None]
                    * projector_grid,
                    axis=(0, 1),
                )
                for rx, ry in ow_support(nshell)
            }

        def ow_overlap_squared_alpha(kx, ky, alpha, coeffs=None):
            projector = ow_projector_for_alpha(kx, ky, alpha)
            spinor = (np.broadcast_to(ow_tau, projector.shape[:-1]) if coeffs is None
                      else ow_spinor(kx, ky, coeffs))
            return np.maximum(np.real(np.einsum(
                "...a,...ab,...b->...", spinor.conj(), projector, spinor, optimize=True
            )), 0.0)

        def ow_reference_at(kx, ky, alpha):
            candidates = [
                np.array([1.0, 0.0], dtype=np.complex128),
                np.array([0.0, 1.0], dtype=np.complex128),
                np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0),
                np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2.0),
            ]
            projector = ow_projector_for_alpha(kx, ky, alpha)
            norms = [np.linalg.norm(projector @ candidate) for candidate in candidates]
            return candidates[int(np.argmax(norms))]

        def ow_complex_overlap_alpha(kx, ky, alpha, coeffs, reference):
            projector = ow_projector_for_alpha(kx, ky, alpha)
            frame = np.einsum("...ab,b->...a", projector, reference, optimize=True)
            frame /= np.linalg.norm(frame, axis=-1)[..., None]
            spinor = (np.broadcast_to(ow_tau, frame.shape) if coeffs is None
                      else ow_spinor(kx, ky, coeffs))
            return np.einsum("...a,...a->...", frame.conj(), spinor, optimize=True)

        def ow_wrap_momentum(value):
            return (value + np.pi) % (2 * np.pi) - np.pi

        def ow_periodic_distance(point_a, point_b):
            delta = ow_wrap_momentum(np.asarray(point_a) - np.asarray(point_b))
            return float(np.linalg.norm(delta))

        def ow_refined_overlap_minima(alpha, coeffs, maximum_candidates=32):
            field = np.sqrt(ow_overlap_squared_alpha(OW_KX, OW_KY, alpha, coeffs))
            local = np.ones(field.shape, dtype=bool)
            for axis in (0, 1):
                local &= field <= np.roll(field, 1, axis=axis)
                local &= field <= np.roll(field, -1, axis=axis)
            candidates = np.argwhere(local)
            candidates = sorted(candidates, key=lambda ij: field[tuple(ij)])[:maximum_candidates]
            refined = []
            for ix, iy in candidates:
                start = np.array([OW_KX[ix, iy], OW_KY[ix, iy]])
                fit = minimize(
                    lambda point: float(ow_overlap_squared_alpha(
                        ow_wrap_momentum(point[0]), ow_wrap_momentum(point[1]), alpha, coeffs
                    )),
                    x0=start, method="Nelder-Mead",
                    options={"xatol": 1e-13, "fatol": 1e-18, "maxiter": 5000},
                )
                point = np.array([ow_wrap_momentum(fit.x[0]), ow_wrap_momentum(fit.x[1])])
                reference = ow_reference_at(point[0], point[1], alpha)
                residual = float(abs(ow_complex_overlap_alpha(
                    point[0], point[1], alpha, coeffs, reference
                )))
                if not any(ow_periodic_distance(point, row["point"]) < 1e-5 for row in refined):
                    refined.append({"point": point, "residual": residual, "reference": reference})
            return sorted(refined, key=lambda row: row["residual"])

        def ow_loop_winding(alpha, coeffs, point, reference, radius=0.02):
            theta = np.linspace(0, 2 * np.pi, 721)
            kx = point[0] + radius * np.cos(theta)
            ky = point[1] + radius * np.sin(theta)
            overlap = ow_complex_overlap_alpha(kx, ky, alpha, coeffs, reference)
            phase = np.unwrap(np.angle(overlap))
            return int(np.rint((phase[-1] - phase[0]) / (2 * np.pi)))

        def ow_lower_band_chern(alpha):
            projector = ow_projector_for_alpha(OW_KX, OW_KY, alpha)
            _, eigenvectors = np.linalg.eigh(projector)
            occupied = eigenvectors[..., -1]
            overlap_x = np.einsum(
                "...a,...a->...", occupied.conj(), np.roll(occupied, -1, axis=0)
            )
            overlap_y = np.einsum(
                "...a,...a->...", occupied.conj(), np.roll(occupied, -1, axis=1)
            )
            Ux, Uy = overlap_x / np.abs(overlap_x), overlap_y / np.abs(overlap_y)
            plaquette = (Ux * np.roll(Uy, -1, axis=0)
                         * np.conj(np.roll(Ux, -1, axis=1)) * np.conj(Uy))
            return int(np.rint(np.angle(plaquette).sum() / (2 * np.pi)))

        OW_SWEEP_ALPHAS = [1.0, 1.5, 3.0, 30.0]
        OW_SWEEP_SUPPORTS = [("none", None), ("0", 0), ("1", 1), ("2", 2)]
        ow_sweep_coeffs = {}
        ow_sweep_rows = []
        for alpha in OW_SWEEP_ALPHAS:
            band_chern = ow_lower_band_chern(alpha)
            for support_label, nshell in OW_SWEEP_SUPPORTS:
                coeffs = None if nshell is None else ow_coefficients_for_alpha(alpha, nshell)
                ow_sweep_coeffs[(alpha, support_label)] = coeffs
                minima = ow_refined_overlap_minima(alpha, coeffs)
                zeros = [row for row in minima if row["residual"] < 1e-6]
                windings = [ow_loop_winding(
                    alpha, coeffs, row["point"], row["reference"]
                ) for row in zeros]
                best = minima[0]
                ow_sweep_rows.append({
                    "alpha": alpha,
                    "target_band_chern": band_chern,
                    "support": support_label,
                    "nshell": np.nan if nshell is None else nshell,
                    "minimum_overlap": best["residual"],
                    "minimum_kx_over_pi": float(best["point"][0] / np.pi),
                    "minimum_ky_over_pi": float(best["point"][1] / np.pi),
                    "zero_count": len(zeros),
                    "zero_windings": windings,
                    "total_overlap_winding": int(sum(windings)),
                })
        ow_sweep = pd.DataFrame(ow_sweep_rows)
        expected_winding = {1.0: -1, 1.5: -1, 3.0: 0, 30.0: 0}
        for alpha, expected in expected_winding.items():
            assert set(ow_sweep.query("alpha == @alpha").total_overlap_winding) == {expected}
        assert dict(ow_sweep.groupby("alpha").target_band_chern.first()) == {
            1.0: 1, 1.5: 1, 3.0: 0, 30.0: 0
        }
        assert np.all(ow_sweep.query("alpha >= 3").minimum_overlap > 0.49)
        display(ow_sweep)
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.75), sharex=True)
        cut_kx_pi = np.linspace(0.0, 0.70, 600)
        cut_kx = np.pi * cut_kx_pi
        sweep_styles = {
            "none": dict(color=BPJ_BLACK, marker=None, linestyle="--", label="unwindowed"),
            "0": dict(color=BPJ_GREEN, marker="s", linestyle="-.", label=r"$n_{\rm shell}=0$"),
            "1": dict(color=BPJ_RED, marker="^", linestyle=":", label=r"$n_{\rm shell}=1$"),
            "2": dict(color=BPJ_BLUE, marker="o", linestyle="-", label=r"$n_{\rm shell}=2$"),
        }
        for ax, alpha, letter in zip(axes.flat, OW_SWEEP_ALPHAS, "abcd"):
            subset = ow_sweep.query("alpha == @alpha").set_index("support")
            for support_label, _ in OW_SWEEP_SUPPORTS:
                style = sweep_styles[support_label]
                coeffs = ow_sweep_coeffs[(alpha, support_label)]
                amplitude = np.sqrt(ow_overlap_squared_alpha(
                    cut_kx, np.zeros_like(cut_kx), alpha, coeffs
                ))
                kwargs = dict(color=style["color"], linestyle=style["linestyle"],
                              label=style["label"], lw=1.0)
                if style["marker"] is not None:
                    kwargs.update(marker=style["marker"], markevery=75, ms=2.7, mfc="white")
                ax.plot(cut_kx_pi, amplitude, **kwargs)
                row = subset.loc[support_label]
                if abs(row.minimum_ky_over_pi) < 1e-5:
                    ax.plot(row.minimum_kx_over_pi, row.minimum_overlap,
                            marker=(style["marker"] or "x"), color=style["color"],
                            mfc="white" if style["marker"] else None, ms=4,
                            linestyle="none")
            chern = int(subset.target_band_chern.iloc[0])
            winding_tuple = tuple(int(subset.loc[label].total_overlap_winding)
                                  for label, _ in OW_SWEEP_SUPPORTS)
            chern_text = f"{chern:+d}" if chern else "0"
            common_winding = winding_tuple[0]
            assert winding_tuple == (common_winding,) * len(OW_SWEEP_SUPPORTS)
            ax.text(.03, .08,
                    rf"$C={chern_text}$; all $\nu={common_winding:d}$",
                    transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=.8))
            ax.set(title=rf"$\alpha={alpha:g}$", xlim=(0, .70), ylim=(-.025, .73))
            panel(ax, letter, "")
        axes[0, 0].set_ylabel(r"$|f_n^{\rm eff}(k_x,0)|$")
        axes[1, 0].set_ylabel(r"$|f_n^{\rm eff}(k_x,0)|$")
        axes[1, 0].set_xlabel(r"$k_x/\pi$ at $k_y=0$")
        axes[1, 1].set_xlabel(r"$k_x/\pi$ at $k_y=0$")
        axes[0, 0].legend(loc="upper right", fontsize=5.4)
        fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.94, wspace=.25, hspace=.30)

        DIAGNOSTICS["result_23"] = {
            "definition": "sum of local phase circulations around every refined isolated overlap zero",
            "zero_threshold": 1e-6,
            "loop_radius": 0.02,
            "loop_samples": 721,
            "search_grid": OW_GRID,
            "rows": ow_sweep.to_dict("records"),
        }
        save_figure(fig, "result_23_overlap_winding_alpha_sweep", form_factor_sources,
                    "deterministic target-band overlap-winding parameter sweep",
                    "For unwindowed and n_shell=0,1,2 X-trial modes, total overlap winding is -1 at alpha=1 and 1.5 where the target band has C=+1, while alpha=3 and 30 have a finite overlap gap and zero winding in the trivial target band.",
                    format_mode="recomputed global zero search, local phase circulation, and independent FHS band check")
        """
    ),
    md(
        r"""
        ## Result 24 — Static interface occupations and deterministic density response

        **Premise.**  The finite-success bath channel and its continuous-time weak-success
        limit are both valid averaged descriptions.  The hybrid campaign separates the
        static occupation-band diagnostic from a physical density-phase response and
        tests the latter at three circumferences and both domain-truncation settings.

        **Evidence class.**  New deterministic reconstruction from the CPU production
        solver.  Only the untruncated arm contributes a momentum-resolved spectrum; all
        response products are real-space.  The product is not a Born ensemble and
        contains no saved covariance matrix or covariance history.
        """
    ),
    code(
        r"""
        l1_sources = [
            "experiment_review/legacy_evidence_figure_atlas/data/result_24_continuous_lindblad_reconstruction.npz",
            "experiment_review/legacy_evidence_figure_atlas/data/result_24_continuous_lindblad_reconstruction.json",
            "experiment_review/legacy_evidence_figure_atlas/reconstruct_continuous_lindblad_result.py",
            "mean_channel_lindblad_cpu_campaign/mean_channel_lindblad_cpu.py",
        ]
        with np.load(require(l1_sources[0]), allow_pickle=False) as payload:
            l1 = {key: payload[key] for key in payload.files}
        l1_meta = json.loads(require(l1_sources[1]).read_text())
        assert l1_meta["permanent_covariance_bytes"] == 0
        assert l1_meta["saved_covariance_history"] is False
        assert l1_meta["shells"] == [1, 2, None]
        assert l1_meta["momentum_resolved_shell"] is None
        assert l1_meta["primary_wall_case_count"] == 39
        assert l1_meta["response_covariance_materialized"] is False
        l1_order = np.argsort(l1["ky"])
        l1_colors = [BPJ_RED, BPJ_GREEN, BPJ_BLUE]
        l1_markers = ["^", "s", "o"]
        l1_p_slope = np.polyfit(
            np.log(l1["response_finite_channel_p"]),
            np.log(l1["response_finite_channel_relative_error"]), 1
        )[0]
        print({
            "static_scan": l1_meta["static_scan"],
            "response_sizes": l1_meta["response_sizes"],
            "interpretation_status": l1_meta["interpretation_status"],
            "response_convergence_order": float(l1_p_slope),
            "largest_stationary_residual": l1_meta["largest_stationary_residual"],
        })
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.65))
        ax = axes[0, 0]
        ax.plot(l1["ky"][l1_order], l1["occupation_spectrum_ky"][l1_order], ".",
                color="0.68", ms=1.0, rasterized=True)
        for branch, color in zip(l1["wall_branch_occupations_ky"], (BPJ_RED, BPJ_BLUE)):
            ax.plot(l1["ky"][l1_order], branch[l1_order], color=color, lw=1.15)
        for branch, color in zip(l1["off_wall_branch_occupations_ky"], (BPJ_RED, BPJ_BLUE)):
            ax.plot(l1["ky"][l1_order], branch[l1_order], color=color, lw=.8, ls="--")
        ax.axhline(.5, color=BPJ_BLACK, ls="--", lw=.7)
        ax.set(xlabel=r"$k_y$", ylabel=r"occupation $\nu$", ylim=(-.03, 1.03))
        panel(ax, "a", "stationary wall occupation bands")

        ax = axes[0, 1]
        ax.axis("off")
        shifted = np.fft.fftshift(l1["response_density_ty"], axes=2)
        # Normalize each nonzero time slice only for rendering.  The archived
        # response norms remain absolute, while this exposes the small late-time
        # displacement that the t=0 amplitude would otherwise wash out.
        row_scale = np.max(np.abs(shifted), axis=2, keepdims=True)
        shifted_for_plot = np.divide(
            shifted, row_scale, out=np.zeros_like(shifted), where=row_scale > 1e-15
        )
        vmax = 1.0
        for wall, bounds in enumerate(([.02, .54, .96, .42], [.02, .04, .96, .42])):
            inner = ax.inset_axes(bounds)
            inner.imshow(
                shifted_for_plot[wall], origin="lower", aspect="auto", cmap="RdBu_r",
                vmin=-vmax, vmax=vmax,
                extent=[-l1["response_Ny"][-1] / 2, l1["response_Ny"][-1] / 2,
                        l1["response_times"][0], l1["response_times"][-1]],
                rasterized=True,
            )
            inner.set_ylabel(rf"wall {wall + 1}: $t$")
            if wall == 0:
                inner.set_xticklabels([])
            else:
                inner.set_xlabel(r"$y-y_0$")
        panel(ax, "b", "normalized directional density response")

        ax = axes[1, 0]
        inv_ny = 1.0 / l1["response_Ny"]
        twin = ax.twinx()
        for di, (linestyle, trunc_label) in enumerate((("-", "on"), ("--", "off"))):
            for wall, color in enumerate((BPJ_RED, BPJ_BLUE)):
                ax.plot(inv_ny, l1["response_velocity"][di, 2, :, wall],
                        color=color, marker=("^", "o")[wall], ls=linestyle,
                        label=rf"wall {wall + 1}, DW trunc. {trunc_label}")
                twin.plot(inv_ny, l1["response_mean_directionality"][di, 2, :, wall],
                          color=color, marker=("^", "o")[wall], ls=":", alpha=.38)
        ax.axhline(0, color=BPJ_BLACK, lw=.6)
        twin.axhline(0, color=BPJ_BLACK, lw=.4, ls=":")
        ax.set(xlabel=r"$1/N_y$", ylabel="positive-lobe velocity")
        twin.set_ylabel("signed directionality", color="0.35")
        ax.legend(loc="best", ncol=1, fontsize=5.5)
        panel(ax, "c", "circumference and truncation check")

        ax = axes[1, 1]
        p_order = np.argsort(l1["response_finite_channel_p"])
        p = l1["response_finite_channel_p"][p_order]
        error = l1["response_finite_channel_relative_error"][p_order]
        ax.loglog(p, error, color=BPJ_BLUE, marker="o", lw=1.0, label="density response")
        reference = error[0] * (p / p[0]) ** 2
        ax.loglog(p, reference, color=BPJ_BLACK, ls="--", lw=.8, label=r"$p^2$")
        ax.set(xlabel=r"finite-channel step $p$", ylabel="response relative error")
        ax.legend(loc="upper left")
        panel(ax, "d", "finite channel $\\to$ generator")
        fig.subplots_adjust(left=.09, right=.99, bottom=.11, top=.94, wspace=.28, hspace=.34)

        DIAGNOSTICS["result_24"] = {
            "static_scan": l1_meta["static_scan"],
            "response_sizes": l1_meta["response_sizes"],
            "interpretation_status": l1_meta["interpretation_status"],
            "response_velocity": l1["response_velocity"].tolist(),
            "response_directionality": l1["response_mean_directionality"].tolist(),
            "uniform_response_directionality": l1["uniform_response_directionality"].tolist(),
            "response_convergence_order": float(l1_p_slope),
            "response_finite_channel_relative_errors": l1["response_finite_channel_relative_error"].tolist(),
            "stationary_residual_max": l1_meta["largest_stationary_residual"],
            "permanent_covariance_bytes": 0,
        }
        save_figure(
            fig,
            "result_24_continuous_lindblad_interface",
            l1_sources,
            "deterministic selected-observable reconstruction from the audited L1 solver",
            "The untruncated arm supplies stationary wall occupation bands, while an exact density-phase pulse resolves the deterministic real-space response on both walls. Circumference and domain-truncation controls determine whether directionality survives, and the finite Gaussian channel converges quadratically to the continuous response at fixed physical time.",
            format_mode="hybrid deterministic CPU reconstruction; static and response observables; no covariance archive",
        )
        """
    ),
    md(
        r"""
        # Raw summary, source manifest, and completion checks

        The cell below writes the machine-readable manifest consumed during review and
        prints the scalar diagnostics behind the captions.  Completion requires exactly
        twenty-four PDF/PNG pairs, twenty-four manifest rows, no excluded source path, and no
        missing file.
        """
    ),
    code(
        r"""
        manifest_path = EXPERIMENT_REVIEW / "legacy_evidence_figure_atlas/figure_manifest.json"
        diagnostics_path = EXPERIMENT_REVIEW / "legacy_evidence_figure_atlas/figure_diagnostics.json"
        manifest_path.write_text(json.dumps(FIGURE_MANIFEST, indent=2, sort_keys=True) + "\n")

        def json_safe(value):
            if isinstance(value, dict):
                return {str(k): json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(v) for v in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return None if not np.isfinite(value) else float(value)
            if isinstance(value, np.ndarray):
                return json_safe(value.tolist())
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        diagnostics_path.write_text(json.dumps(json_safe(DIAGNOSTICS), indent=2, sort_keys=True) + "\n")
        assert len(FIGURE_MANIFEST["figures"]) == 24
        assert all("erroneous_gpu_stuff" not in source for row in FIGURE_MANIFEST["figures"] for source in row["sources"])
        for row in FIGURE_MANIFEST["figures"]:
            assert require(row["pdf"]).stat().st_size > 0
            assert require(row["png"]).stat().st_size > 0
        print(f"Generated {len(FIGURE_MANIFEST['figures'])} BPJ-style multipanel figures.")
        print("Manifest:", manifest_path.relative_to(ROOT))
        print("Diagnostics:", diagnostics_path.relative_to(ROOT))
        display(pd.DataFrame(FIGURE_MANIFEST["figures"])[["stem", "format_mode", "evidence_class", "claim"]])
        print(json.dumps(json_safe(DIAGNOSTICS), indent=2, sort_keys=True)[:12000])
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.4",
        },
        "legacy_evidence_atlas": {
            "single_plotting_source": True,
            "figure_count": 24,
            "style_reference": "BPJ, Physical Review Research 8, 023147 (2026)",
        },
    },
)
nbf.write(notebook, NOTEBOOK)
print(NOTEBOOK)
