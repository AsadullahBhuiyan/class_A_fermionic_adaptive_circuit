#!/usr/bin/env python3
"""Build and optionally execute the frozen P1 bulk-production analysis notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[3]
NOTEBOOK_PATH = PROJECT_DIR / "P1_bulk_production_analysis.ipynb"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "production_25sample_v1"
EXECUTED_PATH = OUTPUT_DIR / "P1_bulk_production_analysis_executed.ipynb"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> nbf.NotebookNode:
    cells = [
        markdown(
            r"""
# P1 frozen bulk-production analysis

This notebook verifies and analyzes the 240 frozen `production_25sample_v1` P1 shards without unpacking or modifying them. It produces the two legacy Fig. 1(c,d) candidates—prepared bulk Chern profiles and finite-size topological indices—together with purity/Bott, temporal-convergence, sensitivity, comprehensive cycle-resolved, logarithmic Chern-error, and compact reference-style convergence diagnostics.

The scientific assessment is recorded as a pass. The formal legacy contract remains qualified because it did not specify a numerical Chern-marker tolerance or a local-marker averaging window. The fixed-geometry amendment is treated as authoritative: this is a P1 analysis and does not reinstate W1 as a width gate.
"""
        ),
        markdown(
            r"""
## Estimators and order of operations

For each Born-rule trajectory, the saved correlation matrix was used during production to evaluate the real-space Chern estimate, Bott index, purity gap $\Delta$ about occupation $1/2$, density, and local Chern marker $C(\boldsymbol r)$. Nonlinear topology is evaluated trajectory by trajectory and only then averaged: the notebook reports $\overline{C_G}$, never $C_{\overline G}$.

For each pure-state size series and overcomplete Wannier (OW) truncation, equal-weight size means are fit to
$$
C(L)=C_\infty+\frac{a}{L},\qquad
C(L)=C_\infty+\frac{b}{L^2}.
$$
Uncertainty is obtained from 20,000 deterministic whole-trajectory bootstrap resamples. Bott success intervals are exact two-sided 95% Clopper--Pearson intervals. The central-half local-marker window is used only as a spatial profile diagnostic, not as the primary finite-size estimator.
"""
        ),
        code(
            r"""
# First executable cell: edit P1_CPU_RANGE (for example "0-7") or set it in the environment.
import os

P1_CPU_RANGE = os.environ.get("P1_CPU_RANGE", "").strip()

def _parse_cpu_range(spec):
    cpus = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = (int(part) for part in token.split("-", 1))
            cpus.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            cpus.add(int(token))
    return cpus

available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
requested_cpus = _parse_cpu_range(P1_CPU_RANGE) if P1_CPU_RANGE else set(available_cpus[: min(4, len(available_cpus))])
selected_cpus = sorted(set(available_cpus).intersection(requested_cpus))
if not selected_cpus:
    raise ValueError(f"P1_CPU_RANGE={P1_CPU_RANGE!r} selects no available CPU from {available_cpus}")
if hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, selected_cpus)
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = str(len(selected_cpus))
print({"P1_CPU_RANGE": P1_CPU_RANGE or "auto", "selected_cpus": selected_cpus})
"""
        ),
        markdown(
            r"""
## Runtime, paths, and plotting style

`P1_ARCHIVE_ROOT` may point directly to another copy of the frozen `01_bulk_width_gate` archive directory. The default is the repository's large-results tree. All generated products are written beside this notebook under `outputs/production_25sample_v1/`.
"""
        ),
        code(
            r"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib as mpl
mpl.use("pdf", force=True)
import matplotlib.pyplot as plt
import nbclient
import nbformat
import numpy as np
import pandas as pd
import scipy
from IPython.display import Image as IPythonImage
from IPython.display import display
from PIL import Image
from scipy.stats import beta


def find_repo_root(start):
    for candidate in (Path(start).resolve(), *Path(start).resolve().parents):
        if (candidate / "PROJECT_ADMIN" / "REPO_POLICY.md").is_file():
            return candidate
    raise FileNotFoundError("Could not locate PROJECT_ADMIN/REPO_POLICY.md")


REPO_ROOT = find_repo_root(Path.cwd())
PROJECT_DIR = REPO_ROOT / "00_WORKSPACE/CURRENT/experiment_review/p1_bulk_production_analysis"
SOURCE_NOTEBOOK = PROJECT_DIR / "P1_bulk_production_analysis.ipynb"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "production_25sample_v1"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
LATEX_SUPPORT_DIR = PROJECT_DIR / "latex_support"
os.environ["TEXINPUTS"] = str(LATEX_SUPPORT_DIR) + os.pathsep + os.environ.get("TEXINPUTS", "")

default_archive_root = REPO_ROOT / "00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/01_bulk_width_gate"
ARCHIVE_ROOT = Path(os.environ.get("P1_ARCHIVE_ROOT", str(default_archive_root))).expanduser().resolve()
VERIFIER_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts/prior_designs/01_p1_existing_completion"

BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 2026082101
SINGLE_COLUMN_WIDTH = 3.375
FIGURE_WIDTH = 7.05
FIGURE_DPI = 300

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": FIGURE_DPI,
    "font.family": "sans-serif",
    "font.sans-serif": ["CMU Sans Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}",
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
})
plt.ioff()

BPJ_RED = "#D92725"
BPJ_GREEN = "#2CA02C"
BPJ_BLUE = "#1F77B4"
BPJ_BLACK = "#000000"

def panel(axis, letter, title=None):
    axis.text(-0.12, 1.04, f"({letter})", transform=axis.transAxes, ha="left", va="bottom", fontsize=9)
    if title:
        axis.set_title(title, pad=4)
    axis.tick_params(direction="in")
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)

def save_latex_figure(fig, pdf_path, png_path):
    fig.savefig(pdf_path)
    plt.close(fig)
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(FIGURE_DPI), "-singlefile", str(pdf_path), str(png_path.with_suffix(""))],
        check=True,
    )
    display(IPythonImage(filename=str(png_path)))

print({"repo_root": str(REPO_ROOT), "archive_root": str(ARCHIVE_ROOT), "output_dir": str(OUTPUT_DIR)})
"""
        ),
        markdown(
            r"""
## Frozen-matrix verification and data loading

The official completion verifier supplies the immutable case expansion and the receipt, audit, engine, run-configuration, shard, seed, and sample-index checks. Archives are opened in memory; no member is extracted to disk.
"""
        ),
        code(
            r"""
if str(VERIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFIER_ROOT))
import resume_existing_p1 as official


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def write_json(path, payload):
    Path(path).write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def root_member(archive, basename):
    matches = [member for member in archive.getmembers() if member.name.lstrip("./") == basename]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {basename!r}, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError(f"Unreadable archive member {basename!r}")
    return handle


def selected_member(archive):
    matches = [member for member in archive.getmembers() if member.name.endswith("/selected_observables.npz")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one selected_observables.npz, found {len(matches)}")
    handle = archive.extractfile(matches[0])
    if handle is None:
        raise RuntimeError("Unreadable selected_observables.npz")
    return handle


config = official.load_config(VERIFIER_ROOT)
cases = [case for case in official.expand_cases(config) if case["campaign"] == "P1"]
expected_slots = {(case["case_id"], shard): case for case in cases for shard in range(5)}
archives = sorted(ARCHIVE_ROOT.glob("*.tar.gz"))
receipts = sorted(ARCHIVE_ROOT.glob("*.tar.gz.receipt.json"))

assert len(cases) == 48
assert len(expected_slots) == 240
assert len(archives) == 240 and len(receipts) == 240
assert config["sampling_revision"] == "production_25sample_v1"
assert config["audit_sha256"] == official.EXPECTED_AUDIT
assert sha256_file(VERIFIER_ROOT / "src/classA_U1FGTN_gpu.py") == official.EXPECTED_ENGINE

frozen_fingerprint_before = {}
trajectory_rows = []
convergence_rows = []
profile_samples = {"topological": [], "trivial": []}
seen_slots = set()
case_sample_indices = {case["case_id"]: set() for case in cases}
audit_hashes = set()
engine_hashes = set()
bott_nan_count = 0
bott_finite_count = 0
required_finite_counts = {key: 0 for key in ("purity_gap", "real_space_chern", "density", "convergence", "local_marker")}

for archive_path in archives:
    receipt = official.verify_archive_receipt(archive_path)
    receipt_path = archive_path.with_suffix(archive_path.suffix + ".receipt.json")
    assert int(receipt["archive_bytes"]) == archive_path.stat().st_size
    frozen_fingerprint_before[str(archive_path.relative_to(REPO_ROOT))] = {
        "sha256": receipt["archive_sha256"], "bytes": archive_path.stat().st_size
    }
    frozen_fingerprint_before[str(receipt_path.relative_to(REPO_ROOT))] = {
        "sha256": sha256_file(receipt_path), "bytes": receipt_path.stat().st_size
    }

    with tarfile.open(archive_path, "r:gz") as archive:
        manifest = json.load(root_member(archive, "manifest.json"))
        slot = (manifest["case_id"], int(manifest["shard_index"]))
        if slot in seen_slots:
            raise RuntimeError(f"Duplicate P1 slot {slot}")
        seen_slots.add(slot)
        case = expected_slots.get(slot)
        if case is None:
            raise RuntimeError(f"Unexpected P1 slot {slot}")
        mismatches = official._validate_manifest(manifest, config, case, slot[1])
        if mismatches:
            raise RuntimeError(f"Frozen identity mismatch for {slot}: {mismatches}")
        expected_path = official._expected_archive(ARCHIVE_ROOT, config, case, slot[1])
        assert expected_path.name == archive_path.name

        audit_hashes.add(manifest["audit_sha256"])
        engine_hashes.add(manifest["run_config"]["canonical_engine_sha256"])
        sample_indices = [int(value) for value in manifest["global_sample_indices"]]
        if case_sample_indices[case["case_id"]].intersection(sample_indices):
            raise RuntimeError(f"Duplicate global sample index in {case['case_id']}")
        case_sample_indices[case["case_id"]].update(sample_indices)

        selected_bytes = io.BytesIO(selected_member(archive).read())
        with np.load(selected_bytes, allow_pickle=False) as selected:
            cycles = selected["cycles"].astype(int)
            gap = np.asarray(selected["purity_gap"], dtype=float)
            chern = np.asarray(selected["real_space_chern"], dtype=float)
            bott = np.asarray(selected["bott_index"], dtype=float)
            density = np.asarray(selected["density"], dtype=float)
            convergence = np.asarray(selected["successive_covariance_frobenius_per_dimension"], dtype=float)
            assert gap.shape == chern.shape == bott.shape == (5, len(cycles))
            assert density.shape[:2] == (5, len(cycles))
            assert convergence.shape == (5, 2 * int(case["model"]["Ny"]))
            assert np.isfinite(gap).all() and np.isfinite(chern).all() and np.isfinite(density).all()
            assert np.isfinite(convergence).all()
            final_mask = cycles == 2 * int(case["model"]["Ny"])
            assert final_mask.sum() == 1
            assert np.isfinite(bott[:, final_mask]).all()
            assert np.isnan(bott[:, ~final_mask]).all()
            bott_nan_count += int(np.isnan(bott).sum())
            bott_finite_count += int(np.isfinite(bott).sum())
            required_finite_counts["purity_gap"] += gap.size
            required_finite_counts["real_space_chern"] += chern.size
            required_finite_counts["density"] += density.size
            required_finite_counts["convergence"] += convergence.size

            model = case["model"]
            L = int(model["Ny"])
            phase = "topological" if float(model["alpha_1"]) == 1.0 else "trivial"
            initialization = "pure" if model["init_mode"] == "default" else "maxmix"
            shell = "dense" if model["nshell"] is None else str(int(model["nshell"]))
            for local_sample, global_sample in enumerate(sample_indices):
                trajectory_id = f"{case['case_id']}:{global_sample:02d}"
                for cycle_index, cycle in enumerate(cycles):
                    trajectory_rows.append({
                        "case_id": case["case_id"],
                        "trajectory_id": trajectory_id,
                        "global_sample": global_sample,
                        "shard_index": slot[1],
                        "phase": phase,
                        "initialization": initialization,
                        "shell": shell,
                        "L": L,
                        "cycle": int(cycle),
                        "normalized_cycle": float(cycle) / L,
                        "purity_gap": gap[local_sample, cycle_index],
                        "real_space_chern": chern[local_sample, cycle_index],
                        "bott_index": bott[local_sample, cycle_index],
                        "density_mean": density[local_sample, cycle_index].mean(),
                    })
                for cycle_index in range(convergence.shape[1]):
                    convergence_rows.append({
                        "case_id": case["case_id"],
                        "trajectory_id": trajectory_id,
                        "global_sample": global_sample,
                        "phase": phase,
                        "initialization": initialization,
                        "shell": shell,
                        "L": L,
                        "cycle": cycle_index + 1,
                        "normalized_cycle": float(cycle_index + 1) / L,
                        "covariance_change": convergence[local_sample, cycle_index],
                    })

            if L == 32 and initialization == "pure" and shell == "1":
                marker_keys = [key for key in selected.files if key.startswith("local_chern_marker_cycle_")]
                assert marker_keys == ["local_chern_marker_cycle_0064"]
                marker = np.asarray(selected[marker_keys[0]], dtype=float)
                assert marker.shape == (5, 32, 32) and np.isfinite(marker).all()
                required_finite_counts["local_marker"] += marker.size
                profile_samples[phase].append(marker.copy())

assert seen_slots == set(expected_slots)
assert audit_hashes == {official.EXPECTED_AUDIT}
assert engine_hashes == {official.EXPECTED_ENGINE}
assert all(indices == set(range(25)) for indices in case_sample_indices.values())
assert bott_finite_count == 1200 and bott_nan_count == 6000

trajectory_df = pd.DataFrame(trajectory_rows).sort_values(["case_id", "global_sample", "cycle"]).reset_index(drop=True)
convergence_df = pd.DataFrame(convergence_rows).sort_values(["case_id", "global_sample", "cycle"]).reset_index(drop=True)
profile_means = {phase: np.concatenate(parts, axis=0).mean(axis=0) for phase, parts in profile_samples.items()}
assert all(np.concatenate(parts, axis=0).shape[0] == 25 for parts in profile_samples.values())

campaign_parameters = pd.DataFrame({
    "parameter": ["sampling revision", "cases", "shards", "trajectories", "sizes", "initializations", "OW shells", "cycles", "audit SHA-256", "engine SHA-256"],
    "value": [
        config["sampling_revision"], len(cases), len(archives), trajectory_df["trajectory_id"].nunique(),
        ", ".join(map(str, config["P1"]["square_sizes"])), "pure; maxmix at L=20,28", "1, 2, dense",
        "2L with six observations", next(iter(audit_hashes)), next(iter(engine_hashes)),
    ],
})
display(campaign_parameters)
print({"trajectory_rows": len(trajectory_df), "convergence_rows": len(convergence_df), "intentional_bott_nan": bott_nan_count})
"""
        ),
        markdown(
            r"""
## Trajectory tables, bootstrap intervals, and finite-size fits

The tidy observation table retains every trajectory before aggregation. Case intervals resample complete trajectories. Finite-size replicates resample the 25 trajectories independently within each size, compute six size means, and refit the full size series.
"""
        ),
        code(
            r"""
def keyed_rng(label):
    raw = f"{BOOTSTRAP_SEED}:{label}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")
    return np.random.default_rng(seed)


def bootstrap_means(values, label):
    values = np.asarray(values, dtype=float)
    rng = keyed_rng(label)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    return values[indices].mean(axis=1)


def mean_ci(values, label):
    values = np.asarray(values, dtype=float)
    replicates = bootstrap_means(values, label)
    lo, hi = np.quantile(replicates, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def clopper_pearson(successes, trials, confidence=0.95):
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else beta.ppf(alpha / 2, successes, trials - successes + 1)
    upper = 1.0 if successes == trials else beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    return float(lower), float(upper)


final_df = trajectory_df.loc[np.isclose(trajectory_df["normalized_cycle"], 2.0)].copy()
assert len(final_df) == 1200

case_summary_rows = []
for case_id, group in final_df.groupby("case_id", sort=True):
    mean, lo, hi = mean_ci(group["real_space_chern"], f"case:{case_id}:chern")
    phase = group["phase"].iloc[0]
    target = 1.0 if phase == "topological" else 0.0
    successes = int(np.isclose(group["bott_index"], target, atol=1e-8, rtol=0).sum())
    cp_lo, cp_hi = clopper_pearson(successes, len(group))
    case_summary_rows.append({
        "case_id": case_id,
        "phase": phase,
        "initialization": group["initialization"].iloc[0],
        "shell": group["shell"].iloc[0],
        "L": int(group["L"].iloc[0]),
        "trajectories": len(group),
        "chern_mean": mean,
        "chern_ci95_low": lo,
        "chern_ci95_high": hi,
        "chern_std": group["real_space_chern"].std(ddof=1),
        "purity_gap_min": group["purity_gap"].min(),
        "density_mean": group["density_mean"].mean(),
        "bott_successes": successes,
        "bott_trials": len(group),
        "bott_success_fraction": successes / len(group),
        "bott_cp95_low": cp_lo,
        "bott_cp95_high": cp_hi,
    })
case_summary_df = pd.DataFrame(case_summary_rows).sort_values(["initialization", "phase", "shell", "L"]).reset_index(drop=True)

fit_rows = []
fit_boot = {}
fit_dictionary = {}
pure_final = final_df.loc[final_df["initialization"] == "pure"]
for phase in ("topological", "trivial"):
    fit_dictionary[phase] = {}
    for shell in ("1", "2", "dense"):
        subset = pure_final.loc[(pure_final["phase"] == phase) & (pure_final["shell"] == shell)]
        sizes = np.array(sorted(subset["L"].unique()), dtype=float)
        values_by_size = [subset.loc[subset["L"] == size, "real_space_chern"].to_numpy() for size in sizes]
        assert sizes.tolist() == [12, 16, 20, 24, 28, 32]
        size_means = np.array([values.mean() for values in values_by_size])
        bootstrap_size_means = np.column_stack([
            bootstrap_means(values, f"fit:{phase}:{shell}:L{int(size)}")
            for size, values in zip(sizes, values_by_size)
        ])
        fit_dictionary[phase][shell] = {}
        for model, power in (("1/L", 1), ("1/L^2", 2)):
            x = sizes ** (-power)
            design = np.column_stack([np.ones_like(x), x])
            coefficients = np.linalg.lstsq(design, size_means, rcond=None)[0]
            bootstrap_coefficients = bootstrap_size_means @ np.linalg.pinv(design).T
            lo, hi = np.quantile(bootstrap_coefficients[:, 0], [0.025, 0.975])
            predictions = design @ coefficients
            residuals = size_means - predictions
            key = (phase, shell, model)
            fit_boot[key] = bootstrap_coefficients[:, 0]
            row = {
                "phase": phase,
                "shell": shell,
                "model": model,
                "power": power,
                "sizes": ";".join(str(int(value)) for value in sizes),
                "C_infinity": coefficients[0],
                "coefficient": coefficients[1],
                "C_infinity_ci95_low": lo,
                "C_infinity_ci95_high": hi,
                "rmse_size_means": np.sqrt(np.mean(residuals**2)),
            }
            fit_rows.append(row)
            fit_dictionary[phase][shell][model] = row

fit_df = pd.DataFrame(fit_rows).sort_values(["phase", "shell", "power"]).reset_index(drop=True)

trajectory_df.to_csv(OUTPUT_DIR / "trajectory_observations.csv", index=False, float_format="%.17g")
convergence_df.to_csv(OUTPUT_DIR / "convergence_histories.csv", index=False, float_format="%.17g")
case_summary_df.to_csv(OUTPUT_DIR / "case_summary.csv", index=False, float_format="%.17g")
fit_df.to_csv(OUTPUT_DIR / "finite_size_fits.csv", index=False, float_format="%.17g")
write_json(OUTPUT_DIR / "finite_size_fits.json", fit_dictionary)

display(case_summary_df.head(8))
display(fit_df)
"""
        ),
        markdown(
            r"""
## Gate ledger and sensitivity summaries

The gate ledger separates the numerical scientific conclusion from the underspecified legacy acceptance contract. The sensitivity table propagates trajectory resampling through shell-extrapolation differences and the max-mix-minus-pure comparison.
"""
        ),
        code(
            r"""
def difference_ci(left, right):
    difference = np.asarray(left) - np.asarray(right)
    lo, hi = np.quantile(difference, [0.025, 0.975])
    return float(difference.mean()), float(lo), float(hi)


sensitivity_rows = []
for model in ("1/L", "1/L^2"):
    for shell in ("1", "2"):
        mean, lo, hi = difference_ci(
            fit_boot[("topological", shell, model)],
            fit_boot[("topological", "dense", model)],
        )
        sensitivity_rows.append({
            "comparison": "shell_intercept_minus_dense",
            "phase": "topological",
            "shell": shell,
            "model": model,
            "L": np.nan,
            "difference": mean,
            "ci95_low": lo,
            "ci95_high": hi,
        })

for phase in ("topological", "trivial"):
    for shell in ("1", "2", "dense"):
        for L in (20, 28):
            pure_values = final_df.loc[
                (final_df["phase"] == phase) & (final_df["initialization"] == "pure") &
                (final_df["shell"] == shell) & (final_df["L"] == L), "real_space_chern"
            ].to_numpy()
            maxmix_values = final_df.loc[
                (final_df["phase"] == phase) & (final_df["initialization"] == "maxmix") &
                (final_df["shell"] == shell) & (final_df["L"] == L), "real_space_chern"
            ].to_numpy()
            maxmix_boot = bootstrap_means(maxmix_values, f"init:maxmix:{phase}:{shell}:L{L}")
            pure_boot = bootstrap_means(pure_values, f"init:pure:{phase}:{shell}:L{L}")
            mean, lo, hi = difference_ci(maxmix_boot, pure_boot)
            sensitivity_rows.append({
                "comparison": "maxmix_minus_pure",
                "phase": phase,
                "shell": shell,
                "model": "none",
                "L": L,
                "difference": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            })
sensitivity_df = pd.DataFrame(sensitivity_rows)
sensitivity_df.to_csv(OUTPUT_DIR / "sensitivity_summary.csv", index=False, float_format="%.17g")

bott_ledger = []
for phase, group in final_df.groupby("phase", sort=True):
    target = 1.0 if phase == "topological" else 0.0
    successes = int(np.isclose(group["bott_index"], target, atol=1e-8, rtol=0).sum())
    lower, upper = clopper_pearson(successes, len(group))
    bott_ledger.append({
        "phase": phase, "target": target, "successes": successes, "trials": len(group),
        "fraction": successes / len(group), "clopper_pearson_95_low": lower,
        "clopper_pearson_95_high": upper,
    })

large_topological = final_df.loc[(final_df["phase"] == "topological") & (final_df["L"] >= 24)]
large_size_chern = {}
for shell, group in large_topological.groupby("shell", sort=True):
    large_size_chern[shell] = {
        "trajectories": len(group),
        "mean": group["real_space_chern"].mean(),
        "minimum": group["real_space_chern"].min(),
        "maximum": group["real_space_chern"].max(),
    }

gate_evaluation = {
    "campaign": "P1",
    "sampling_revision": "production_25sample_v1",
    "scientific_assessment": "pass",
    "formal_contract_status": "qualified_under_specified_tolerance",
    "qualification": "The legacy contract supplies no numerical Chern-marker tolerance and no local-marker window.",
    "fixed_geometry_amendment": "authoritative; W1 is not reinstated as a width gate",
    "primary_estimator": "trajectory-resolved real-space Chern estimate averaged after evaluation",
    "profile_diagnostic": "central-half final local Chern marker; diagnostic only",
    "minimum_final_purity_gap": final_df["purity_gap"].min(),
    "bott": bott_ledger,
    "large_size_topological_chern": large_size_chern,
}
write_json(OUTPUT_DIR / "gate_evaluation.json", gate_evaluation)
display(pd.DataFrame(bott_ledger))
display(sensitivity_df)
"""
        ),
        markdown(
            r"""
## Figure 1 — prepared bulk Chern profiles

These heat maps average the final trajectory-resolved $C(\boldsymbol r)$ only after evaluating it on each of $S=25$ independent trajectories at $L=32$, pure initialization, $n_{\rm shell}=1$, and cycle $2L$. Both panels show the central-half window on a common scale; the crop is a spatial diagnostic and is not used for the finite-size gate.
"""
        ),
        code(
            r"""
profile_png = FIGURE_DIR / "fig01_prepared_bulk_chern_profiles.png"
profile_pdf = FIGURE_DIR / "fig01_prepared_bulk_chern_profiles.pdf"
window = slice(8, 24)

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.65), sharex=True, sharey=True)
for axis, phase, title in zip(axes, ("topological", "trivial"), (r"topological, $\alpha=1$", r"trivial, $\alpha=30$")):
    image_artist = axis.imshow(
        profile_means[phase][window, window], origin="lower", cmap="viridis", vmin=0.0, vmax=1.0,
        interpolation="nearest", extent=(8, 24, 8, 24), aspect="equal",
    )
    axis.set_title(title, pad=2)
    axis.set_xlabel(r"$x$")
axes[0].set_ylabel(r"$y$")
panel(axes[0], "a")
panel(axes[1], "b")
colorbar = fig.colorbar(image_artist, ax=axes, fraction=0.045, pad=0.03)
colorbar.set_label(r"$\overline{C_G(\boldsymbol r)}$")
fig.subplots_adjust(left=0.12, right=0.88, bottom=0.20, top=0.86, wspace=0.12)
save_latex_figure(fig, profile_pdf, profile_png)
"""
        ),
        markdown(
            r"""
## Figure 2 — finite-size topological indices

The primary real-space estimator is shown against $1/L$ for the three OW shell choices. Each size contains $S=25$ independent pure-initialization trajectories observed at cycle $2L$. Whiskers are 95% whole-trajectory bootstrap intervals (20,000 resamples); solid and dashed curves are equal-weight fits over $L=12,16,20,24,28,32$ to the $1/L$ and $1/L^2$ forms, respectively.
"""
        ),
        code(
            r"""
finite_png = FIGURE_DIR / "fig02_finite_size_topological_indices.png"
finite_pdf = FIGURE_DIR / "fig02_finite_size_topological_indices.pdf"
shell_order = ("1", "2", "dense")
shell_labels = {"1": r"$n_{\rm shell}=1$", "2": r"$n_{\rm shell}=2$", "dense": "dense"}
shell_colors = {"1": BPJ_RED, "2": BPJ_GREEN, "dense": BPJ_BLUE}
shell_markers = {"1": "^", "2": "s", "dense": "o"}

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.48), sharex=True)
for axis, phase, target in zip(axes, ("topological", "trivial"), (1.0, 0.0)):
    subset = case_summary_df.loc[(case_summary_df["phase"] == phase) & (case_summary_df["initialization"] == "pure")]
    x_dense = np.linspace(0.0, 1 / 12, 250)
    for shell in shell_order:
        group = subset.loc[subset["shell"] == shell].sort_values("L")
        x = 1.0 / group["L"].to_numpy()
        y = group["chern_mean"].to_numpy()
        yerr = np.vstack([y - group["chern_ci95_low"].to_numpy(), group["chern_ci95_high"].to_numpy() - y])
        axis.errorbar(x, y, yerr=yerr, marker=shell_markers[shell], linestyle="none", markerfacecolor="white", capsize=2, color=shell_colors[shell], label=shell_labels[shell])
        for model, linestyle in (("1/L", "-"), ("1/L^2", "--")):
            row = fit_df.loc[(fit_df["phase"] == phase) & (fit_df["shell"] == shell) & (fit_df["model"] == model)].iloc[0]
            axis.plot(x_dense, row["C_infinity"] + row["coefficient"] * x_dense ** int(row["power"]), color=shell_colors[shell], linestyle=linestyle, alpha=0.9)
    axis.axhline(target, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
    axis.set_xlabel(r"$1/L$")
    axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
axes[0].set_ylabel(r"$\overline{C_G}$")
panel(axes[0], "a", "topological")
panel(axes[1], "b", "trivial")
axes[0].legend(loc="best", handlelength=1.3)
axes[1].text(0.98, 0.06, r"solid: $1/L$\quad dashed: $1/L^2$", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
fig.tight_layout(pad=0.35, w_pad=0.6)
save_latex_figure(fig, finite_pdf, finite_png)
"""
        ),
        markdown(
            r"""
## Figure 3 — purity and Bott gates

The left panel conservatively takes the minimum final purity gap over the two phases for each size, shell, and initialization at cycle $2L$, displayed as its scaled deficit from $1/2$ for legibility. Each case contains $S=25$ independent trajectories. The right panel shows aggregate exact Bott success with two-sided 95% Clopper--Pearson lower bounds.
"""
        ),
        code(
            r"""
gate_png = FIGURE_DIR / "fig03_purity_and_bott_gates.png"
gate_pdf = FIGURE_DIR / "fig03_purity_and_bott_gates.pdf"
purity_plot = case_summary_df.groupby(["initialization", "shell", "L"], as_index=False)["purity_gap_min"].min()
purity_plot["purity_gap_deficit_scaled"] = 1e11 * (0.5 - purity_plot["purity_gap_min"])

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.48))
for shell in shell_order:
    for initialization, marker, linestyle in (("pure", "o", "-"), ("maxmix", "s", "none")):
        group = purity_plot.loc[(purity_plot["shell"] == shell) & (purity_plot["initialization"] == initialization)].sort_values("L")
        axes[0].plot(group["L"], group["purity_gap_deficit_scaled"], marker=shell_markers[shell] if initialization == "pure" else marker, markerfacecolor="white", linestyle=linestyle, color=shell_colors[shell], label=f"{shell_labels[shell]}, {initialization}")
axes[0].axhline(0.0, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
axes[0].set_xlabel(r"$L$")
axes[0].set_ylabel(r"$10^{11}[1/2-\min\Delta(2L)]$")
axes[0].set_ylim(-0.15, 1.12 * purity_plot["purity_gap_deficit_scaled"].max())
axes[0].legend(fontsize=8, ncol=2, handlelength=1.2, columnspacing=0.7, labelspacing=0.3)

bott_plot = pd.DataFrame(bott_ledger)
x = np.arange(len(bott_plot))
y = bott_plot["fraction"].to_numpy()
yerr = np.vstack([y - bott_plot["clopper_pearson_95_low"].to_numpy(), bott_plot["clopper_pearson_95_high"].to_numpy() - y])
axes[1].errorbar(x, y, yerr=yerr, marker="o", linestyle="none", markerfacecolor="white", capsize=2, color=BPJ_BLUE)
axes[1].set_xticks(x, ["trivial", "topological"], rotation=18)
axes[1].set_ylabel("exact Bott success")
axes[1].set_ylim(0.99, 1.0008)
for position, row in zip(x, bott_ledger):
    axes[1].text(position, row["clopper_pearson_95_low"] - 0.0008, f"{row['successes']}/{row['trials']}", ha="center", va="top", fontsize=8)
panel(axes[0], "a")
panel(axes[1], "b")
fig.tight_layout(pad=0.35, w_pad=0.7)
save_latex_figure(fig, gate_pdf, gate_png)
"""
        ),
        markdown(
            r"""
## Figure 4 — temporal convergence

The left panel follows topological pure-state $\overline{C_G}$ for $S=25$ independent trajectories at representative small, intermediate, and large sizes. Color denotes shell and transparency increases with size. The right panel isolates the signed trajectory-paired change between $3L/2$ and $2L$ with 95% whole-trajectory bootstrap intervals.
"""
        ),
        code(
            r"""
temporal_png = FIGURE_DIR / "fig04_temporal_convergence.png"
temporal_pdf = FIGURE_DIR / "fig04_temporal_convergence.pdf"
representative_sizes = (12, 20, 32)
temporal = trajectory_df.loc[
    (trajectory_df["phase"] == "topological") & (trajectory_df["initialization"] == "pure") &
    (trajectory_df["L"].isin(representative_sizes))
]

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.48))
for shell in shell_order:
    for L, alpha_value in zip(representative_sizes, (0.42, 0.68, 1.0)):
        group = temporal.loc[(temporal["shell"] == shell) & (temporal["L"] == L)]
        summary = group.groupby("normalized_cycle", as_index=False)["real_space_chern"].mean()
        axes[0].plot(summary["normalized_cycle"], summary["real_space_chern"], color=shell_colors[shell], alpha=alpha_value, marker=shell_markers[shell], markerfacecolor="white", label=f"{shell_labels[shell]}, $L={L}$")
axes[0].axhline(1.0, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
axes[0].set_xlabel(r"cycle number$/L$")
axes[0].set_ylabel(r"$\overline{C_G}$")
axes[0].legend(fontsize=8, ncol=2, handlelength=1.2, columnspacing=0.7, labelspacing=0.3)

change_rows = []
for shell_index, shell in enumerate(shell_order):
    for L in representative_sizes:
        group = temporal.loc[(temporal["shell"] == shell) & (temporal["L"] == L)]
        pivot = group.pivot(index="trajectory_id", columns="normalized_cycle", values="real_space_chern")
        differences = (pivot[2.0] - pivot[1.5]).to_numpy()
        mean, lo, hi = mean_ci(differences, f"temporal-change:{shell}:L{L}")
        change_rows.append({"shell": shell, "L": L, "difference": mean, "ci95_low": lo, "ci95_high": hi})
        axes[1].errorbar(L + (shell_index - 1) * 0.7, mean, yerr=[[mean - lo], [hi - mean]], marker=shell_markers[shell], linestyle="none", markerfacecolor="white", capsize=2, color=shell_colors[shell], label=shell_labels[shell] if L == representative_sizes[0] else None)
axes[1].axhline(0.0, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
axes[1].set_xlabel(r"$L$")
axes[1].set_ylabel(r"$\overline{C_G(2L)-C_G(3L/2)}$")
axes[1].legend(fontsize=8)
panel(axes[0], "a")
panel(axes[1], "b")
change_df = pd.DataFrame(change_rows)
change_df.to_csv(OUTPUT_DIR / "temporal_change_summary.csv", index=False, float_format="%.17g")
fig.tight_layout(pad=0.35, w_pad=0.55)
save_latex_figure(fig, temporal_pdf, temporal_png)
"""
        ),
        markdown(
            r"""
## Figure 5 — shell and initialization sensitivity

The left panel compares topological extrapolated intercepts to the dense result under both correction models using $S=25$ trajectories per size and 20,000 whole-trajectory bootstrap resamples. The right panel shows max-mix-minus-pure final real-space estimates for both phases at $L=20,28$, with $S=25$ independent trajectories per initialization.
"""
        ),
        code(
            r"""
sensitivity_png = FIGURE_DIR / "fig05_shell_and_initialization_sensitivity.png"
sensitivity_pdf = FIGURE_DIR / "fig05_shell_and_initialization_sensitivity.pdf"

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.48))
intercepts = sensitivity_df.loc[sensitivity_df["comparison"] == "shell_intercept_minus_dense"].copy()
positions = {("1", "1/L"): 0.0, ("2", "1/L"): 1.0, ("1", "1/L^2"): 2.5, ("2", "1/L^2"): 3.5}
for _, row in intercepts.iterrows():
    position = positions[(row["shell"], row["model"])]
    axes[0].errorbar(position, row["difference"], yerr=[[row["difference"] - row["ci95_low"]], [row["ci95_high"] - row["difference"]]], marker=shell_markers[row["shell"]], linestyle="none", markerfacecolor="white", capsize=2, color=shell_colors[row["shell"]])
axes[0].axhline(0.0, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
axes[0].set_xticks([0.5, 3.0], [r"$1/L$", r"$1/L^2$"])
axes[0].set_ylabel(r"$C_\infty^{(n)}-C_\infty^{(\mathrm{dense})}$")
axes[0].text(0.03, 0.96, r"red: $n=1$\quad green: $n=2$", transform=axes[0].transAxes, ha="left", va="top", fontsize=8)

init_sensitivity = sensitivity_df.loc[sensitivity_df["comparison"] == "maxmix_minus_pure"].copy()
base_positions = {(20, "topological"): 0, (28, "topological"): 1, (20, "trivial"): 2.5, (28, "trivial"): 3.5}
offsets = {"1": -0.18, "2": 0.0, "dense": 0.18}
for _, row in init_sensitivity.iterrows():
    position = base_positions[(int(row["L"]), row["phase"])] + offsets[row["shell"]]
    axes[1].errorbar(position, row["difference"], yerr=[[row["difference"] - row["ci95_low"]], [row["ci95_high"] - row["difference"]]], marker=shell_markers[row["shell"]], linestyle="none", markerfacecolor="white", capsize=2, color=shell_colors[row["shell"]], label=shell_labels[row["shell"]] if (int(row["L"]) == 20 and row["phase"] == "topological") else None)
axes[1].axhline(0.0, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
axes[1].set_xticks([0, 1, 2.5, 3.5], [r"T, 20", r"T, 28", r"0, 20", r"0, 28"], rotation=18)
axes[1].set_ylabel(r"$\overline{C_G}^{\rm maxmix}-\overline{C_G}^{\rm pure}$")
axes[1].legend(fontsize=8)
panel(axes[0], "a")
panel(axes[1], "b")
fig.tight_layout(pad=0.35, w_pad=0.65)
save_latex_figure(fig, sensitivity_pdf, sensitivity_png)
"""
        ),
        markdown(
            r"""
## Cycle-resolved real-space Chern summary

For every saved observation cycle, the nonlinear real-space Chern estimator is evaluated on each trajectory first and then averaged over the $S=25$ Born records. The table below retains all phases, initializations, shell choices, and available system sizes, with deterministic 95% whole-trajectory bootstrap intervals.
"""
        ),
        code(
            r"""
cycle_chern_rows = []
for keys, group in trajectory_df.groupby(
    ["initialization", "phase", "shell", "L", "cycle", "normalized_cycle"], sort=True
):
    initialization, phase, shell, L, cycle, normalized_cycle = keys
    mean, lo, hi = mean_ci(
        group["real_space_chern"].to_numpy(),
        f"cycle-chern:{initialization}:{phase}:{shell}:L{int(L)}:cycle{int(cycle)}",
    )
    cycle_chern_rows.append({
        "initialization": initialization,
        "phase": phase,
        "shell": shell,
        "L": int(L),
        "cycle": int(cycle),
        "normalized_cycle": float(normalized_cycle),
        "trajectories": len(group),
        "chern_mean": mean,
        "chern_ci95_low": lo,
        "chern_ci95_high": hi,
    })

cycle_chern_df = pd.DataFrame(cycle_chern_rows).sort_values(
    ["initialization", "phase", "shell", "L", "cycle"]
).reset_index(drop=True)
assert set(cycle_chern_df["trajectories"]) == {25}
cycle_chern_df.to_csv(
    OUTPUT_DIR / "cycle_resolved_chern_summary.csv", index=False, float_format="%.17g"
)
display(cycle_chern_df.head(12))
"""
        ),
        markdown(
            r"""
## Figure 6 — cycle-resolved real-space Chern, pure initialization

Each curve is the mean of the trajectory-resolved real-space Chern number over 25 Born records at the indicated saved cycle. Columns separate the three OW shell choices, rows separate the topological and matched trivial preparations, and all production sizes $L=12,16,20,24,28,32$ are shown. Shaded regions are deterministic 95% whole-trajectory bootstrap intervals.
"""
        ),
        code(
            r"""
cycle_pure_png = FIGURE_DIR / "fig06_cycle_resolved_chern_pure.png"
cycle_pure_pdf = FIGURE_DIR / "fig06_cycle_resolved_chern_pure.pdf"
size_order = (12, 16, 20, 24, 28, 32)
size_colors = {
    12: BPJ_RED,
    16: BPJ_GREEN,
    20: BPJ_BLUE,
    24: "#9467BD",
    28: "#E67E22",
    32: "#595959",
}
size_markers = {12: "^", 16: "s", 20: "o", 24: "D", 28: "v", 32: "P"}
size_linestyles = {12: ":", 16: "--", 20: "-", 24: "-.", 28: "--", 32: "-"}

fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, 4.85), sharex=True, sharey="row")
pure_cycles = cycle_chern_df.loc[cycle_chern_df["initialization"] == "pure"]
letters = iter("abcdef")
legend_handles = []
for row_index, (phase, target) in enumerate((("topological", 1.0), ("trivial", 0.0))):
    for column_index, shell in enumerate(shell_order):
        axis = axes[row_index, column_index]
        for L in size_order:
            group = pure_cycles.loc[
                (pure_cycles["phase"] == phase) & (pure_cycles["shell"] == shell) &
                (pure_cycles["L"] == L)
            ].sort_values("cycle")
            assert len(group) == 6
            line, = axis.plot(
                group["cycle"], group["chern_mean"], color=size_colors[L],
                marker=size_markers[L], markerfacecolor="white",
                linestyle=size_linestyles[L], label=fr"$L={L}$",
            )
            axis.fill_between(
                group["cycle"].to_numpy(), group["chern_ci95_low"].to_numpy(),
                group["chern_ci95_high"].to_numpy(), color=size_colors[L], alpha=0.12,
                linewidth=0,
            )
            if row_index == 0 and column_index == 0:
                legend_handles.append(line)
        axis.axhline(target, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
        if row_index == 1:
            axis.set_xlabel("cycle number")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        panel(axis, next(letters), shell_labels[shell] if row_index == 0 else None)
axes[0, 0].set_ylabel(r"topological $\overline{C_G}$")
axes[1, 0].set_ylabel(r"trivial $\overline{C_G}$")
fig.legend(
    legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center",
    bbox_to_anchor=(0.5, 1.005), ncol=6, handlelength=1.5, columnspacing=0.8,
)
fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.4, w_pad=0.55, h_pad=0.65)
save_latex_figure(fig, cycle_pure_pdf, cycle_pure_png)
"""
        ),
        markdown(
            r"""
## Figure 7 — cycle-resolved real-space Chern, maximally mixed initialization

The same trajectory-before-average construction is shown for every maximally mixed production case. All available sizes, $L=20,28$, appear in each phase and shell panel; shaded regions are deterministic 95% whole-trajectory bootstrap intervals over the 25 Born records.
"""
        ),
        code(
            r"""
cycle_maxmix_png = FIGURE_DIR / "fig07_cycle_resolved_chern_maxmix.png"
cycle_maxmix_pdf = FIGURE_DIR / "fig07_cycle_resolved_chern_maxmix.pdf"
maxmix_sizes = (20, 28)

fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, 4.85), sharex=True, sharey="row")
maxmix_cycles = cycle_chern_df.loc[cycle_chern_df["initialization"] == "maxmix"]
letters = iter("abcdef")
legend_handles = []
for row_index, (phase, target) in enumerate((("topological", 1.0), ("trivial", 0.0))):
    for column_index, shell in enumerate(shell_order):
        axis = axes[row_index, column_index]
        for L in maxmix_sizes:
            group = maxmix_cycles.loc[
                (maxmix_cycles["phase"] == phase) & (maxmix_cycles["shell"] == shell) &
                (maxmix_cycles["L"] == L)
            ].sort_values("cycle")
            assert len(group) == 6
            line, = axis.plot(
                group["cycle"], group["chern_mean"], color=size_colors[L],
                marker=size_markers[L], markerfacecolor="white",
                linestyle=size_linestyles[L], label=fr"$L={L}$",
            )
            axis.fill_between(
                group["cycle"].to_numpy(), group["chern_ci95_low"].to_numpy(),
                group["chern_ci95_high"].to_numpy(), color=size_colors[L], alpha=0.12,
                linewidth=0,
            )
            if row_index == 0 and column_index == 0:
                legend_handles.append(line)
        axis.axhline(target, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
        if row_index == 1:
            axis.set_xlabel("cycle number")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
        panel(axis, next(letters), shell_labels[shell] if row_index == 0 else None)
axes[0, 0].set_ylabel(r"topological $\overline{C_G}$")
axes[1, 0].set_ylabel(r"trivial $\overline{C_G}$")
fig.legend(
    legend_handles, [handle.get_label() for handle in legend_handles], loc="upper center",
    bbox_to_anchor=(0.5, 1.005), ncol=2, handlelength=1.5, columnspacing=1.0,
)
fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.4, w_pad=0.55, h_pad=0.65)
save_latex_figure(fig, cycle_maxmix_pdf, cycle_maxmix_png)
"""
        ),
        markdown(
            r"""
## Figure 8 — logarithmic topological Chern error over the late-cycle window

For the topological preparation, this figure shows the record average of the trajectory-resolved error $|C_G-1|$ on a logarithmic scale. Each size is restricted to its own late-cycle window $L\leq s_{\rm cycle}\leq2L$. Columns separate the three OW shell choices, rows separate pure and maximally mixed initialization, and shaded regions are deterministic 95% whole-trajectory bootstrap intervals.
"""
        ),
        code(
            r"""
cycle_error_rows = []
topological_window = trajectory_df.loc[
    (trajectory_df["phase"] == "topological") &
    (trajectory_df["cycle"] >= trajectory_df["L"]) &
    (trajectory_df["cycle"] <= 2 * trajectory_df["L"])
].copy()
topological_window["absolute_chern_error"] = np.abs(topological_window["real_space_chern"] - 1.0)
assert (topological_window["absolute_chern_error"] > 0).all()

for keys, group in topological_window.groupby(
    ["initialization", "shell", "L", "cycle", "normalized_cycle"], sort=True
):
    initialization, shell, L, cycle, normalized_cycle = keys
    mean, lo, hi = mean_ci(
        group["absolute_chern_error"].to_numpy(),
        f"cycle-chern-error:{initialization}:{shell}:L{int(L)}:cycle{int(cycle)}",
    )
    cycle_error_rows.append({
        "initialization": initialization,
        "shell": shell,
        "L": int(L),
        "cycle": int(cycle),
        "normalized_cycle": float(normalized_cycle),
        "trajectories": len(group),
        "mean_absolute_chern_error": mean,
        "ci95_low": lo,
        "ci95_high": hi,
    })

cycle_error_df = pd.DataFrame(cycle_error_rows).sort_values(
    ["initialization", "shell", "L", "cycle"]
).reset_index(drop=True)
assert set(cycle_error_df["trajectories"]) == {25}
assert (cycle_error_df[["mean_absolute_chern_error", "ci95_low", "ci95_high"]].to_numpy() > 0).all()
cycle_error_df.to_csv(
    OUTPUT_DIR / "cycle_resolved_absolute_chern_error_summary.csv",
    index=False, float_format="%.17g",
)

cycle_error_png = FIGURE_DIR / "fig08_cycle_resolved_absolute_chern_error.png"
cycle_error_pdf = FIGURE_DIR / "fig08_cycle_resolved_absolute_chern_error.pdf"
fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH, 4.85), sharex=True, sharey=True)
letters = iter("abcdef")
for row_index, (initialization, sizes) in enumerate((("pure", size_order), ("maxmix", maxmix_sizes))):
    for column_index, shell in enumerate(shell_order):
        axis = axes[row_index, column_index]
        for L in sizes:
            group = cycle_error_df.loc[
                (cycle_error_df["initialization"] == initialization) &
                (cycle_error_df["shell"] == shell) & (cycle_error_df["L"] == L)
            ].sort_values("cycle")
            assert len(group) >= 2
            axis.plot(
                group["cycle"], group["mean_absolute_chern_error"], color=size_colors[L],
                marker=size_markers[L], markerfacecolor="white",
                linestyle=size_linestyles[L], label=fr"$L={L}$",
            )
            axis.fill_between(
                group["cycle"].to_numpy(), group["ci95_low"].to_numpy(),
                group["ci95_high"].to_numpy(), color=size_colors[L], alpha=0.12,
                linewidth=0,
            )
        axis.set_yscale("log")
        if row_index == 1:
            axis.set_xlabel("cycle number")
        panel(axis, next(letters), shell_labels[shell] if row_index == 0 else None)
        if column_index == 2:
            axis.legend(
                loc="best", fontsize=7, ncol=2 if initialization == "pure" else 1,
                handlelength=1.3, columnspacing=0.6, labelspacing=0.25,
                title=initialization,
            )
axes[0, 0].set_ylabel(r"pure $\langle|C_G-1|\rangle$")
axes[1, 0].set_ylabel(r"maxmix $\langle|C_G-1|\rangle$")
fig.tight_layout(pad=0.4, w_pad=0.55, h_pad=0.65)
save_latex_figure(fig, cycle_error_pdf, cycle_error_png)
display(cycle_error_df.head(12))
"""
        ),
        markdown(
            r"""
## Figure 9 — compact trajectory-resolved Chern convergence with logarithmic inset

This single-column reconstruction follows the supplied reference composition while retaining the frozen P1 definitions and sign convention. The main axis shows the pure-initialization topological $\overline{C_G}$ against cycle number for all six sizes and all three OW shell choices. The inset shows $\langle|C_G-1|\rangle$ over each size's late window $L\leq s_{\rm cycle}\leq2L$ against $(s_{\rm cycle}-L)/L$. Every point is formed by evaluating the nonlinear estimator on each of $S=25$ independent Born trajectories and then averaging. Color/marker identifies shell and line style identifies size; uncertainty bands are omitted in this compact view and are reported in Figures 6 and 8.
"""
        ),
        code(
            r"""
from matplotlib.lines import Line2D

compact_png = FIGURE_DIR / "fig09_compact_cycle_resolved_chern_with_inset.png"
compact_pdf = FIGURE_DIR / "fig09_compact_cycle_resolved_chern_with_inset.pdf"
compact_size_styles = {
    12: ":",
    16: "--",
    20: "-",
    24: "-.",
    28: (0, (5, 1)),
    32: (0, (3, 1, 1, 1)),
}

fig, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 3.72))
compact_main = cycle_chern_df.loc[
    (cycle_chern_df["phase"] == "topological") &
    (cycle_chern_df["initialization"] == "pure")
]
for shell in shell_order:
    for L in size_order:
        group = compact_main.loc[
            (compact_main["shell"] == shell) & (compact_main["L"] == L)
        ].sort_values("cycle")
        axis.plot(
            group["cycle"], group["chern_mean"], color=shell_colors[shell],
            marker=shell_markers[shell], markerfacecolor="white",
            linestyle=compact_size_styles[L], linewidth=0.8, markersize=3.5,
        )
axis.axhline(1.0, color=BPJ_BLACK, linewidth=0.8, linestyle="--")
axis.set_xlabel("cycle number")
axis.set_ylabel(r"$\overline{C_G}$")
axis.tick_params(direction="in")

inset = axis.inset_axes([0.43, 0.39, 0.54, 0.57])
compact_error = cycle_error_df.loc[cycle_error_df["initialization"] == "pure"]
for shell in shell_order:
    for L in size_order:
        group = compact_error.loc[
            (compact_error["shell"] == shell) & (compact_error["L"] == L)
        ].sort_values("normalized_cycle")
        inset.plot(
            group["normalized_cycle"] - 1.0, group["mean_absolute_chern_error"],
            color=shell_colors[shell], marker=shell_markers[shell],
            markerfacecolor="white", linestyle=compact_size_styles[L],
            linewidth=0.7, markersize=3.0,
        )
inset.set_yscale("log")
inset.set_xlim(-0.03, 1.03)
inset.set_xlabel(r"$(s_{\rm cycle}-L)/L$", labelpad=1)
inset.set_ylabel(r"$\langle|C_G-1|\rangle$", labelpad=1)
inset.tick_params(direction="in", labelsize=6.5, pad=1.5)
inset.xaxis.label.set_size(7)
inset.yaxis.label.set_size(7)
for spine in inset.spines.values():
    spine.set_linewidth(0.8)

shell_handles = [
    Line2D(
        [0], [0], color=shell_colors[shell], marker=shell_markers[shell],
        markerfacecolor="white", linestyle="-", linewidth=0.8, markersize=3.5,
        label=shell_labels[shell],
    )
    for shell in shell_order
]
size_handles = [
    Line2D(
        [0], [0], color="#595959", linestyle=compact_size_styles[L],
        linewidth=0.9, label=fr"$L={L}$",
    )
    for L in size_order
]
shell_legend = fig.legend(
    handles=shell_handles, loc="lower center", bbox_to_anchor=(0.5, 0.095),
    ncol=3, handlelength=1.25, columnspacing=0.65, fontsize=7,
)
fig.add_artist(shell_legend)
fig.legend(
    handles=size_handles, loc="lower center", bbox_to_anchor=(0.5, 0.018),
    ncol=3, handlelength=1.4, columnspacing=0.75, fontsize=7,
)
fig.subplots_adjust(left=0.19, right=0.98, bottom=0.27, top=0.98)
save_latex_figure(fig, compact_pdf, compact_png)
"""
        ),
        markdown(
            r"""
## Numerical and artifact validation

This stage reproduces the campaign invariants and known large-size/fitted summaries, checks every figure product, and rehashes every frozen archive and receipt after analysis.
"""
        ),
        code(
            r"""
top_final = final_df.loc[final_df["phase"] == "topological"]
triv_final = final_df.loc[final_df["phase"] == "trivial"]
assert np.isclose(top_final["bott_index"], 1.0, atol=1e-8, rtol=0).all()
assert np.isclose(triv_final["bott_index"], 0.0, atol=1e-8, rtol=0).all()
assert len(top_final) == 600 and len(triv_final) == 600
assert final_df["purity_gap"].min() > 0.4999999999

expected_large = {
    "1": (0.9999577809, 0.9983162941),
    "2": (0.9999949293, 0.9998259240),
    "dense": (0.9999885549, 0.9999839618),
}
for shell, (expected_mean, expected_minimum) in expected_large.items():
    assert np.isclose(large_size_chern[shell]["mean"], expected_mean, atol=1e-9, rtol=0)
    assert np.isclose(large_size_chern[shell]["minimum"], expected_minimum, atol=1e-9, rtol=0)

expected_topological_intercepts = {
    ("1", "1/L"): 1.000432703,
    ("1", "1/L^2"): 0.999859011,
    ("2", "1/L"): 1.000692365,
    ("2", "1/L^2"): 1.000165319,
    ("dense", "1/L"): 1.000462702,
    ("dense", "1/L^2"): 1.000122140,
}
for (shell, model), expected in expected_topological_intercepts.items():
    actual = fit_dictionary["topological"][shell][model]["C_infinity"]
    assert np.isclose(actual, expected, atol=1e-9, rtol=0)

figure_products = sorted(FIGURE_DIR.glob("fig*.png")) + sorted(FIGURE_DIR.glob("fig*.pdf"))
assert len(figure_products) == 18
for png_path in sorted(FIGURE_DIR.glob("fig*.png")):
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi", (0.0, 0.0))
        assert all(abs(float(value) - FIGURE_DPI) < 1.0 for value in dpi)
        expected_width = SINGLE_COLUMN_WIDTH if png_path.name.startswith("fig09_") else FIGURE_WIDTH
        assert abs(image.width - round(expected_width * FIGURE_DPI)) <= 1
for pdf_path in sorted(FIGURE_DIR.glob("fig*.pdf")):
    assert pdf_path.stat().st_size > 1_000 and pdf_path.read_bytes().startswith(b"%PDF")

frozen_fingerprint_after = {}
for relative, before in frozen_fingerprint_before.items():
    path = REPO_ROOT / relative
    frozen_fingerprint_after[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
assert frozen_fingerprint_after == frozen_fingerprint_before

output_hashes = {}
for path in sorted(OUTPUT_DIR.rglob("*")):
    if path.is_file() and path.name not in {"analysis_manifest.json", "P1_bulk_production_analysis_executed.ipynb"}:
        output_hashes[str(path.relative_to(OUTPUT_DIR))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

analysis_manifest = {
    "schema": "p1_bulk_production_analysis_v1",
    "campaign": "P1",
    "sampling_revision": "production_25sample_v1",
    "source_notebook": str(SOURCE_NOTEBOOK.relative_to(REPO_ROOT)),
    "archive_root": str(ARCHIVE_ROOT),
    "input_fingerprint": frozen_fingerprint_before,
    "input_fingerprint_unchanged_after_analysis": True,
    "audit_sha256": official.EXPECTED_AUDIT,
    "canonical_engine_sha256": official.EXPECTED_ENGINE,
    "estimators": {
        "primary": "trajectory-resolved real-space Chern estimate; mean after nonlinear evaluation",
        "finite_size_models": ["C(L)=C_infinity+a/L", "C(L)=C_infinity+b/L^2"],
        "size_weighting": "equal-weight size means",
        "profile": "central-half final local marker, diagnostic only",
        "bott_interval": "exact two-sided 95% Clopper-Pearson",
    },
    "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "root_seed": BOOTSTRAP_SEED, "unit": "whole trajectory"},
    "figure_style": {"single_column_width_inches": SINGLE_COLUMN_WIDTH, "compound_width_inches": FIGURE_WIDTH, "dpi": FIGURE_DPI, "font": "CMU Sans Serif", "latex_text": True, "tight_layout": True, "grammar": "BPJ red-triangle / green-square / blue-circle; boxed axes; inward ticks; outside panel letters"},
    "software": {
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "matplotlib": mpl.__version__, "nbformat": nbformat.__version__,
        "nbclient": nbclient.__version__, "pillow": Image.__version__,
    },
    "assessment": {"scientific_assessment": "pass", "formal_contract_status": "qualified_under_specified_tolerance"},
    "output_hashes": output_hashes,
    "self_hash_excluded": True,
}
write_json(OUTPUT_DIR / "analysis_manifest.json", analysis_manifest)

validation_summary = {
    "cases": len(cases), "shards": len(archives), "trajectories": final_df["trajectory_id"].nunique(),
    "topological_bott": "600/600", "trivial_bott": "600/600",
    "minimum_final_purity_gap": final_df["purity_gap"].min(),
    "intentional_bott_nan_count": bott_nan_count, "finite_bott_count": bott_finite_count,
    "figures": len(figure_products), "frozen_inputs_unchanged": True,
}
display(pd.DataFrame([validation_summary]))
"""
        ),
        markdown(
            r"""
## Raw summary and diagnostics

The final cell exposes the exact fit dictionary, scalar gate rows, finite/null counts, artifact ledger, and gate decision used by the figures.
"""
        ),
        code(
            r"""
scalar_rows = {
    "minimum_final_purity_gap": float(final_df["purity_gap"].min()),
    "topological_final_chern_mean": float(top_final["real_space_chern"].mean()),
    "trivial_final_chern_mean": float(triv_final["real_space_chern"].mean()),
    "topological_bott_successes": int(np.isclose(top_final["bott_index"], 1.0, atol=1e-8, rtol=0).sum()),
    "trivial_bott_successes": int(np.isclose(triv_final["bott_index"], 0.0, atol=1e-8, rtol=0).sum()),
}
finite_null_counts = {
    "trajectory_rows": len(trajectory_df),
    "convergence_rows": len(convergence_df),
    "bott_finite": bott_finite_count,
    "bott_intentional_nan": bott_nan_count,
    "required_finite_counts": required_finite_counts,
}
print("FIT_DICTIONARY")
print(json.dumps(json_ready(fit_dictionary), indent=2, sort_keys=True))
print("SCALAR_ROWS")
print(json.dumps(scalar_rows, indent=2, sort_keys=True))
print("FINITE_NULL_COUNTS")
print(json.dumps(finite_null_counts, indent=2, sort_keys=True))
print("GATE_LEDGER")
print(json.dumps(json_ready(gate_evaluation), indent=2, sort_keys=True))
print("OUTPUT_HASHES")
print(json.dumps(output_hashes, indent=2, sort_keys=True))
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": platform.python_version() if False else "3"},
            "analysis_contract": {
                "campaign": "P1",
                "sampling_revision": "production_25sample_v1",
                "analysis_only": True,
            },
        },
    )
    return notebook


def validate_and_finalize(executed: nbf.NotebookNode) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    if len(image_outputs) != 9:
        raise RuntimeError(f"Expected nine rendered notebook figures, found {len(image_outputs)}")

    manifest_path = OUTPUT_DIR / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_hashes"][EXECUTED_PATH.name] = {
        "sha256": sha256_file(EXECUTED_PATH), "bytes": EXECUTED_PATH.stat().st_size
    }

    pdf_widths = {}
    for pdf_path in sorted((OUTPUT_DIR / "figures").glob("fig*.pdf")):
        report = subprocess.check_output(["pdfinfo", str(pdf_path)], text=True)
        match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", report)
        if not match:
            raise RuntimeError(f"Could not read PDF dimensions for {pdf_path}")
        width_points = float(match.group(1))
        width_inches = width_points / 72.0
        expected_width = 3.375 if pdf_path.name.startswith("fig09_") else 7.05
        if abs(width_inches - expected_width) > 0.01:
            raise RuntimeError(f"Unexpected PDF width {width_inches} for {pdf_path}")
        pdf_widths[pdf_path.name] = width_inches

    png_dpi = {}
    for png_path in sorted((OUTPUT_DIR / "figures").glob("fig*.png")):
        with Image.open(png_path) as image:
            dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
            if not all(abs(value - 300.0) < 1.0 for value in dpi):
                raise RuntimeError(f"Unexpected PNG dpi {dpi} for {png_path}")
            png_dpi[png_path.name] = dpi

    validation = {
        "status": "pass",
        "executed_notebook": str(EXECUTED_PATH.relative_to(REPO_ROOT)),
        "executed_notebook_sha256": sha256_file(EXECUTED_PATH),
        "rendered_figure_outputs_in_notebook": len(image_outputs),
        "pdf_width_inches": pdf_widths,
        "png_dpi": png_dpi,
    }
    validation_path = OUTPUT_DIR / "execution_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["output_hashes"][validation_path.name] = {
        "sha256": sha256_file(validation_path), "bytes": validation_path.stat().st_size
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute and validate the notebook")
    args = parser.parse_args()

    notebook = build_notebook()
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(f"wrote {NOTEBOOK_PATH}")

    if args.execute:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        client = NotebookClient(
            notebook,
            timeout=3600,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
        executed = client.execute()
        nbf.write(executed, EXECUTED_PATH)
        validation = validate_and_finalize(executed)
        print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
