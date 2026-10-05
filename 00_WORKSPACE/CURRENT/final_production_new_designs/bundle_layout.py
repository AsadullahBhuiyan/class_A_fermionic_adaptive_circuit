"""Canonical flat layout for independent production campaign bundles."""

from __future__ import annotations

import json
from pathlib import Path


NEW_DESIGN_BUNDLES = (
    "01_uniform_bulk_validation",
    "02_domain_wall_bipartite_mutual_information",
    "03_uniform_bulk_validation_large_l",
    "04_maxmix_manybody_lyapunov_pilot",
    "05_hard_wall_entropy_charge_batched_v2",
    "06_domain_wall_flattened_ground_state_reference",
    "07_maxmix_hard_soft_purification",
    "08_domain_wall_correlator_scaling",
    "09_pure_tangent_replay_acquisition",
    "10_soft_wall_entropy_charge_batched_v2",
    "11_wall_pump_width_endpoints",
    "12_wall_diabatic_spectral_pump_gpu",
    "13_maxmix_manybody_lyapunov_4ny",
    "14_hard_wall_xresolved_correlator_scaling",
    "15_hard_wall_alpha3_xresolved_correlator",
    "16_hard_wall_entropy_contour_all_ay",
    "17_hard_wall_tangent_gap_cocycle",
    "18_hard_wall_purification_alpha_endpoint",
    "19_postselected_hard_soft_n20x40",
    "20_hard_wall_alpha3_purification",
    "21_hard_wall_full_measurement_purification",
    "22_hard_wall_full_measurement_clipped",
    "23_hard_wall_random_center_chern",
    "24_flattened_imaginary_time_density",
    "25_square_hard_wall_gap_pilot",
    "26_fixed_width_hard_wall_gap_t40",
    "27_square_hard_wall_random_center_chern",
    "28_full_measurement_purification_gap_t40",
    "29_square_purification_contour_pilot",
    "30_square_purification_contour_t40",
    "31_hard_wall_alpha_endpoint_ny30",
)

REQUIRED_BUNDLE_FILES = {
    "31_hard_wall_alpha_endpoint_ny30": (
        "run_alpha_endpoint_lane_A.ipynb", "run_alpha_endpoint_lane_B.ipynb",
        "run_alpha_endpoint_lane_C.ipynb", "run_campaign.py", "endpoint_spectrum.py",
        "analyze_campaign.py", "build_notebooks.py", "campaign_config.json", "README.md",
        "deployment_manifest.json", "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "30_square_purification_contour_t40": (
        "run_square_purification_contour.ipynb", "run_campaign.py", "contour_observer.py",
        "io_utils.py", "build_notebook.py", "campaign_config.json", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "29_square_purification_contour_pilot": (
        "run_square_purification_contour.ipynb", "run_campaign.py", "contour_observer.py",
        "io_utils.py", "build_notebook.py", "campaign_config.json", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "28_full_measurement_purification_gap_t40": (
        "run_full_measurement_purification_gap_t40.ipynb", "run_campaign.py",
        "endpoint_spectrum.py", "analyze_campaign.py", "campaign_config.json",
        "build_notebook.py", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "27_square_hard_wall_random_center_chern": (
        "deployment_manifest.json", "run_square_hard_wall_random_center_chern.ipynb",
        "run_campaign.py", "random_center_observer.py", "campaign_config.json",
        "build_notebook.py", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "26_fixed_width_hard_wall_gap_t40": (
        "run_fixed_width_hard_wall_gap_t40.ipynb", "run_campaign.py", "endpoint_spectrum.py",
        "analyze_campaign.py", "build_notebook.py", "campaign_config.json", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "25_square_hard_wall_gap_pilot": (
        "run_square_hard_wall_gap.ipynb", "run_campaign.py", "endpoint_spectrum.py",
        "analyze_campaign.py", "build_notebook.py", "campaign_config.json", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "24_flattened_imaginary_time_density": (
        "run_flattened_imaginary_time_density.ipynb", "run_campaign.py",
        "density_correlations.py", "plot_results.py", "campaign_config.json",
        "build_notebook.py", "README.md", "theory.tex",
        "src/classA_U1FGTN.py", "src/occupied_frame.py",
    ),
    "23_hard_wall_random_center_chern": (
        "deployment_manifest.json", "run_hard_wall_random_center_chern.ipynb",
        "run_campaign.py", "random_center_observer.py", "campaign_config.json",
        "build_notebook.py", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "22_hard_wall_full_measurement_clipped": (
        "deployment_manifest.json", "run_hard_wall_full_measurement_clipped.ipynb",
        "run_campaign.py", "purification_observer.py", "campaign_config.json",
        "build_notebook.py", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "21_hard_wall_full_measurement_purification": (
        "deployment_manifest.json",
        "run_hard_wall_full_measurement_purification.ipynb",
        "run_campaign.py", "purification_observer.py", "campaign_config.json",
        "build_notebook.py", "README.md",
        "src/classA_U1FGTN_gpu.py", "src/occupied_frame_gpu.py",
    ),
    "20_hard_wall_alpha3_purification": (
        "deployment_manifest.json",
        "run_hard_wall_alpha3_purification.ipynb",
        "run_campaign.py",
        "purification_observer.py",
        "campaign_config.json",
        "build_notebook.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "01_uniform_bulk_validation": (
        "run_uniform_bulk_validation.ipynb",
        "run_campaign.py",
        "compact_observer.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "02_domain_wall_bipartite_mutual_information": (
        "run_domain_wall_bipartite_mutual_information.ipynb",
        "run_campaign.py",
        "mutual_information_observer.py",
        "make_hard_wall_figures.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "03_uniform_bulk_validation_large_l": (
        "run_uniform_bulk_validation_large_l.ipynb",
        "run_campaign.py",
        "compact_observer.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "04_maxmix_manybody_lyapunov_pilot": (
        "run_maxmix_manybody_lyapunov_pilot.ipynb",
        "run_campaign.py",
        "lyapunov_observer.py",
        "analyze_campaign.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "05_hard_wall_entropy_charge_batched_v2": (
        "run_lane_A_Ny40_Ny60.ipynb",
        "run_lane_B_endpoint_Ny30_35_45_50_55.ipynb",
        "run_campaign.py",
        "entropy_charge_observer.py",
        "analyze_campaign.py",
        "build_notebooks.py",
        "bundle_manifest.json",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "06_domain_wall_flattened_ground_state_reference": (
        "run_flattened_ground_state_reference.py",
        "run_flattened_ground_state_large_ny.py",
        "analyze_wall_projected_ground_state.py",
        "mutual_information_observer.py",
        "README.md",
        "src/classA_U1FGTN.py",
        "src/occupied_frame.py",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "07_maxmix_hard_soft_purification": (
        "deployment_manifest.json",
        "run_hard_wall_purification.ipynb",
        "run_soft_wall_purification.ipynb",
        "run_campaign.py",
        "purification_observer.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "08_domain_wall_correlator_scaling": (
        "run_hard_wall_correlator.ipynb",
        "run_soft_wall_correlator.ipynb",
        "run_campaign.py",
        "correlator_observer.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "09_pure_tangent_replay_acquisition": (
        "run_pure_tangent_replay_acquisition.ipynb",
        "run_pure_tangent_gpu_replay.ipynb",
        "run_campaign.py",
        "replay_record_observer.py",
        "README.md",
        "gpu_tangent_replay/run_gpu_tangent_replay.py",
        "gpu_tangent_replay/campaign_config.json",
        "gpu_tangent_replay/build_notebook.py",
        "gpu_tangent_replay/README.md",
        "gpu_tangent_replay/deployment_manifest.json",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "10_soft_wall_entropy_charge_batched_v2": (
        "run_lane_A_Ny40_Ny60.ipynb",
        "run_lane_B_endpoint_Ny30_35_45_50_55.ipynb",
        "run_campaign.py",
        "entropy_charge_observer.py",
        "analyze_campaign.py",
        "build_notebooks.py",
        "bundle_manifest.json",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "11_wall_pump_width_endpoints": (
        "run_wall_pump_width_endpoints.ipynb",
        "run_campaign.py",
        "campaign_config.json",
        "build_notebook.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "12_wall_diabatic_spectral_pump_gpu": (
        "run_wall_diabatic_spectral_pump_gpu.ipynb",
        "run_campaign.py",
        "gpu_backend.py",
        "spectral_cpu_reference.py",
        "campaign_config.json",
        "build_notebook.py",
        "README.md",
    ),
    "13_maxmix_manybody_lyapunov_4ny": (
        "deployment_manifest.json",
        "run_hard_wall_manybody_lyapunov_4ny.ipynb",
        "run_hard_wall_manybody_lyapunov_4ny_lane_a.ipynb",
        "run_hard_wall_manybody_lyapunov_4ny_lane_b.ipynb",
        "run_soft_wall_manybody_lyapunov_4ny.ipynb",
        "run_campaign.py",
        "lyapunov_observer.py",
        "campaign_config.json",
        "build_notebooks.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "14_hard_wall_xresolved_correlator_scaling": (
        "run_hard_wall_xresolved_correlator.ipynb",
        "run_campaign.py",
        "endpoint_correlator.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "15_hard_wall_alpha3_xresolved_correlator": (
        "run_hard_wall_alpha3_correlator.ipynb",
        "run_campaign.py",
        "alpha3_correlator.py",
        "build_notebook.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "16_hard_wall_entropy_contour_all_ay": (
        "run_lane_A_Ny50_Ny60.ipynb",
        "run_lane_B_Ny30_35_40_45_55.ipynb",
        "run_campaign.py",
        "entropy_contour_observer.py",
        "analyze_campaign.py",
        "build_notebooks.py",
        "bundle_manifest.json",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "17_hard_wall_tangent_gap_cocycle": (
        "deployment_manifest.json",
        "run_hard_wall_tangent_lane_a.ipynb",
        "run_hard_wall_tangent_lane_b.ipynb",
        "run_campaign.py",
        "analyze_campaign.py",
        "replay_record_observer.py",
        "campaign_config.json",
        "build_notebooks.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "18_hard_wall_purification_alpha_endpoint": (
        "deployment_manifest.json",
        "run_hard_wall_purification_alpha_endpoint_lane_a.ipynb",
        "run_hard_wall_purification_alpha_endpoint_lane_b.ipynb",
        "run_campaign.py",
        "endpoint_spectrum_observer.py",
        "campaign_config.json",
        "build_notebooks.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
    "19_postselected_hard_soft_n20x40": (
        "deployment_manifest.json",
        "run_postselected_hard_soft_n20x40.ipynb",
        "run_campaign.py",
        "postselected_observer.py",
        "campaign_config.json",
        "build_notebook.py",
        "README.md",
        "src/classA_U1FGTN_gpu.py",
        "src/occupied_frame_gpu.py",
    ),
}


def bundle_path(root: Path | str, bundle: str) -> Path:
    if bundle not in NEW_DESIGN_BUNDLES:
        raise KeyError(f"unknown redesigned campaign bundle {bundle!r}")
    return Path(root) / bundle


def validate_bundle_layout(root: Path | str) -> tuple[str, ...]:
    root = Path(root)
    actual = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith("__")
        )
    )
    if actual != NEW_DESIGN_BUNDLES:
        missing = sorted(set(NEW_DESIGN_BUNDLES) - set(actual))
        unexpected = sorted(set(actual) - set(NEW_DESIGN_BUNDLES))
        raise RuntimeError(
            f"redesigned bundle mismatch: missing={missing}, unexpected={unexpected}"
        )
    for bundle in actual:
        for relative in REQUIRED_BUNDLE_FILES[bundle]:
            required = root / bundle / relative
            if not required.is_file():
                raise FileNotFoundError(f"missing required bundle file: {required}")

    index = json.loads((root / "bundle_index.json").read_text(encoding="utf-8"))
    if index.get("bundles") != list(NEW_DESIGN_BUNDLES):
        raise RuntimeError("bundle_index.json differs from bundle_layout.py")
    if set(index.get("standalone_contracts", {})) != set(NEW_DESIGN_BUNDLES):
        raise RuntimeError("every redesigned bundle must declare a standalone contract")
    return actual
