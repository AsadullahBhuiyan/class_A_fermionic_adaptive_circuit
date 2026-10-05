#!/usr/bin/env python3
"""Plot the trajectory-first minimum absolute tangent rate, without rerunning dynamics."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import matplotlib as mpl
import numpy as np
from tqdm import tqdm


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BUNDLES = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
RESULTS = REPO / "00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs"
REVISION_BASE = "hard_wall_pure_tangent_alpha21_ny24-40_s100_c2ny_"
CONFIG_PATH = BUNDLES / "17_hard_wall_tangent_gap_cocycle/campaign_config.json"
STEM = "hard_wall_tangent_minabs_gap_alpha_and_size"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_pair(completion: Path) -> tuple[dict, Path]:
    receipt = json.loads(completion.read_text())
    name = receipt["result_filename"]
    if Path(name).name != name:
        raise ValueError(f"nonlocal result filename: {completion}")
    result = completion.parent / name
    if result.stat().st_size != receipt["result_bytes"]:
        raise ValueError(f"byte-count mismatch: {result}")
    if sha256(result) != receipt["result_sha256"]:
        raise ValueError(f"checksum mismatch: {result}")
    return receipt, result


def trajectory_gaps(rates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Five rates are the saved pairs closest to zero, not the old -2*rate gaps."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2 or rates.shape[1] < 1 or not np.isfinite(rates).all():
        raise ValueError("expected a finite sample-by-mode signed-rate array")
    if np.any(np.diff(np.abs(rates), axis=1) < -1e-12):
        raise ValueError("saved pair rates are not ordered by absolute value")
    nearest = np.argmin(np.abs(rates), axis=1)
    signed = rates[np.arange(len(rates)), nearest]
    return np.abs(signed), signed


def summary(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("need at least two finite trajectory gaps")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect() -> tuple[list[dict], list[dict], list[dict]]:
    config = json.loads(CONFIG_PATH.read_text())
    expected = {
        (ny, float(alpha))
        for nys, alphas in [
            (config["sweep_Ny_values"], config["alpha_1_sweep_values"]),
            (config["endpoint_Ny_values"], config["alpha_1_endpoint_values"]),
        ]
        for ny in nys for alpha in alphas
    }
    groups = {case: {} for case in expected}
    inputs, trajectories, globals_seen = [], [], set()
    completions = sorted(
        p for suffix in ("v1", "v2")
        for p in (RESULTS / (REVISION_BASE + suffix)).rglob("*.complete.json")
    )
    if len(completions) != 375:
        raise ValueError(f"expected 375 logical completed batches, got {len(completions)}")
    for completion in tqdm(completions, desc="Verify tangent outputs", unit="batch"):
        d, path = verify_pair(completion)
        case = (int(d["Ny"]), float(d["alpha_1"]))
        if case not in expected or d["Nx"] != 20 or d["cycles"] != 2 * case[0]:
            raise ValueError(f"unexpected case/geometry/window: {path}")
        if d["nshell"] != 1 or d["alpha_2"] != 30:
            raise ValueError(f"unexpected scientific parameters: {path}")
        if d["sampling_revision"] == REVISION_BASE + "v1":
            if case != (40, 1.0) or d["case_sample_indices"] != list(range(25)):
                raise ValueError("unexpected v1 import; only the pinned 25 rows are allowed")
            if d["result_sha256"] != "ba7579994dd3864fa5577aefa09588fc4f8efc1bdf200bbac3a0c62b02183e54":
                raise ValueError("pinned v1 result changed")
        elif d["sampling_revision"] != REVISION_BASE + "v2":
            raise ValueError("unexpected sampling revision")
        with np.load(path, allow_pickle=False) as z:
            for key in ("Nx", "Ny", "alpha_1", "alpha_2", "nshell", "cycles", "task_id",
                        "configuration_sha256", "sampling_revision"):
                if z[key].item() != d[key]:
                    raise ValueError(f"result/receipt mismatch for {key}: {path}")
            if json.loads(str(z["source_hashes_json"])) != d["source_hashes"]:
                raise ValueError(f"result/receipt source mismatch: {path}")
            if str(z["dtype"]) != "complex128" or int(z["burn_in_cycles"]) != 0:
                raise ValueError("wrong dtype or cycle window")
            ids = z["case_sample_indices"]
            global_ids = z["global_sample_indices"]
            np.testing.assert_array_equal(ids, d["case_sample_indices"])
            np.testing.assert_array_equal(global_ids, d["global_sample_indices"])
            rates = z["slow_pair_rates_per_cycle"]
            np.testing.assert_allclose(z["slow_effective_gaps_per_cycle"], -2 * rates,
                                       rtol=1e-13, atol=1e-14)
            gamma, signed = trajectory_gaps(rates)
            if rates.shape != (len(ids), 5) or len(ids) != d["sample_count"]:
                raise ValueError("unexpected sample count or number of saved modes")
            has_product = "chronological_cocycle_hat" in z.files
            if has_product != (case[0] == 40):
                raise ValueError("endpoint product persistence contract mismatch")
            nulls = z["one_leg_singular_null_counts"]
            blocks = z["occupied_empty_block_sizes"]
            if np.any(nulls < 0) or np.any(nulls > blocks):
                raise ValueError("invalid numerical-null diagnostics")
            for row, sample in enumerate(ids):
                sample, global_id = int(sample), int(global_ids[row])
                if sample in groups[case] or global_id in globals_seen:
                    raise ValueError("duplicate trajectory")
                globals_seen.add(global_id)
                groups[case][sample] = float(gamma[row])
                trajectories.append({
                    "Ny": case[0], "alpha_1": case[1], "cycles": 2 * case[0],
                    "sample_index": sample, "global_sample_index": global_id,
                    "gamma_minabs_per_cycle": float(gamma[row]),
                    "nearest_signed_pair_rate": float(signed[row]),
                    "occupied_null_count": int(nulls[row, 0]),
                    "empty_null_count": int(nulls[row, 1]),
                    "task_id": d["task_id"], "revision": d["sampling_revision"],
                })
        inputs.append({"result": str(path.relative_to(REPO)),
                       "bytes": d["result_bytes"], "sha256": d["result_sha256"],
                       "completion_sha256": sha256(completion),
                       "samples": d["sample_count"], "Ny": case[0], "alpha_1": case[1],
                       "full_product_saved": has_product})
    rows = []
    for (ny, alpha), values in sorted(groups.items()):
        if set(values) != set(range(100)):
            raise ValueError(f"incomplete case: {ny}, {alpha}")
        mean, sem = summary(np.array([values[s] for s in range(100)]))
        rows.append({"Ny": ny, "alpha_1": alpha, "cycles": 2 * ny,
                     "samples": 100, "mean_gamma": mean, "sem_gamma": sem})
    if len(rows) != 67 or len(trajectories) != 6700:
        raise ValueError("incorrect total campaign coverage")
    return rows, sorted(trajectories, key=lambda r: (r["Ny"], r["alpha_1"], r["sample_index"])), inputs


def vector_inventory() -> dict:
    """Inspect older mode-bearing results separately; do not pool them into slot 17."""
    revision = "pure_tangent_cpu_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v3"
    root = BUNDLES / "09_pure_tangent_replay_acquisition/cpu_data" / revision
    groups, inputs = {}, []
    for p in tqdm(sorted(root.rglob("*.complete.json")), desc="Verify saved endpoint modes", unit="sample"):
        d, result = verify_pair(p)
        key = (d["construction"], d["Ny"], d["alpha_1"])
        samples = groups.setdefault(key, set())
        sample = int(d["case_sample_index"])
        if sample in samples:
            raise ValueError("duplicate CPU-replay sample")
        with np.load(result, allow_pickle=False) as z:
            if not bool(z["endpoint_replay_verified"]):
                raise ValueError("unverified endpoint replay")
            for window in ("full", "late"):
                for side in ("input", "output"):
                    for block in ("occupied", "empty"):
                        a = z[f"{window}_slow_{side}_{block}"]
                        if a.shape != (2 * 20 * d["Ny"], 16) or not np.isfinite(a).all():
                            raise ValueError("invalid saved slow-mode vector")
            samples.add(sample)
        inputs.append({"result": str(result.relative_to(REPO)), "sha256": d["result_sha256"]})
    purification = []
    root = BUNDLES / "18_hard_wall_purification_alpha_endpoint/gpu_data"
    for p in sorted(root.rglob("*.complete.json")):
        d, result = verify_pair(p)
        with np.load(result, allow_pickle=False) as z:
            purification.append({"Ny": d["Ny"], "alpha_1": d["alpha_1"],
                                 "samples": len(d["sample_indices"]),
                                 "selected_vector_count": int(z["selected_mode_counts"].sum())})
    return {
        "scope": "local repository files, not a live Drive inventory",
        "slot17": {"standalone_mode_vectors_saved": False,
                   "Ny40_full_products": 200, "smaller_sizes_gap_only": [24, 28, 32, 36]},
        "slot09_cpu_replay_revision": revision,
        "slot09_mode_convention": "input/output occupied/empty singular vectors for 16 slow pairs, full and late windows; not eigenvectors of the nonnormal cocycle",
        "slot09_cpu_replay_groups": [
            {"construction": k[0], "Ny": k[1], "alpha_1": k[2],
             "samples": len(s), "sample_indices": sorted(s)} for k, s in sorted(groups.items())],
        "slot09_cpu_replay_inputs": inputs,
        "slot18_local_samples": sum(r["samples"] for r in purification),
        "slot18_selected_vectors": sum(r["selected_vector_count"] for r in purification),
        "slot18_selection": "only centered-covariance eigenmodes with abs(a)<=0.9; empty selections are valid",
    }


def plot(rows: list[dict]) -> None:
    # Use the repository's LaTeX/Computer Modern sans-serif plotting toolchain.
    mpl.use("Agg")
    os.environ["TEXINPUTS"] = str(HERE / "latex_support") + os.pathsep + os.environ.get("TEXINPUTS", "")
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}",
        "font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif"],
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False,
    })
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullFormatter

    fig, axes = plt.subplots(2, 1, figsize=(3.375, 5.2), layout="constrained")
    colors, markers, styles = ["#d62728", "#2ca02c", "#1f77b4"], ["^", "s", "o"], [":", "--", "-"]
    for ny, color, marker, style in zip((24, 28, 32), colors, markers, styles):
        selected = [r for r in rows if r["Ny"] == ny]
        axes[0].errorbar([r["alpha_1"] for r in selected], [r["mean_gamma"] for r in selected],
                         yerr=[r["sem_gamma"] for r in selected], color=color, marker=marker,
                         linestyle=style, label=rf"$N_y={ny}$", markersize=3,
                         markerfacecolor="white", linewidth=.9, elinewidth=.7, capsize=1.5)
    for alpha, color, marker, style in [(1., colors[0], "^", "--"), (3., colors[2], "o", "-")]:
        selected = [r for r in rows if r["alpha_1"] == alpha]
        axes[1].errorbar([r["Ny"] for r in selected], [r["mean_gamma"] for r in selected],
                         yerr=[r["sem_gamma"] for r in selected], color=color, marker=marker,
                         linestyle=style, label=rf"$\alpha_1={alpha:g}$", markersize=3.8,
                         markerfacecolor="white", linewidth=1, elinewidth=.8, capsize=2)
    axes[0].set(xlabel=r"$\alpha_1$", xlim=(.96, 3.04), xticks=[1, 1.5, 2, 2.5, 3])
    axes[1].set(xlabel=r"$N_y$", xlim=(23, 41), xticks=[24, 28, 32, 36, 40])
    for letter, ax in zip(("(a)", "(b)"), axes):
        ax.set_yscale("log")
        ax.set_ylim(.007, 2)
        ax.set_ylabel(r"$\overline{\gamma^{(T)}}$ (cycle$^{-1}$)")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=[1.0]))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="both", top=True, right=True)
        ax.legend(loc="lower right" if letter == "(a)" else "center right",
                  handlelength=1.7, labelspacing=.3)
        ax.text(-.02, 1.025, letter, transform=ax.transAxes, ha="right", va="bottom")
    axes[0].set_title(r"Hard walls: $N_x=20$, $T=2N_y$, $S=100$", fontsize=8, pad=8)
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    pdf = figures / f"{STEM}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf),
                    str(pdf.with_suffix(""))], check=True)


def main() -> None:
    rows, trajectories, inputs = collect()
    write_csv(HERE / "tables/gap_summary.csv", rows)
    write_csv(HERE / "tables/trajectory_gaps.csv", trajectories)
    inventory = vector_inventory()
    (HERE / "endpoint_mode_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    plot(rows)
    artifacts = [HERE / "tables/gap_summary.csv", HERE / "tables/trajectory_gaps.csv",
                 HERE / "endpoint_mode_inventory.json", HERE / "figures" / f"{STEM}.pdf",
                 HERE / "figures" / f"{STEM}.png"]
    manifest = {
        "estimator": "gamma_xi=min_ij abs(lambda_ij_xi); lambda=(log_sigma_o+log_sigma_e)/T",
        "estimator_order": "minimum absolute signed rate per trajectory, then arithmetic ensemble mean",
        "old_gap_conversion": "gamma=0.5*min(abs(slow_effective_gaps_per_cycle))",
        "uncertainty": "sample standard deviation (ddof=1) / sqrt(100); SEM, not 95% CI",
        "cases": len(rows), "trajectories": len(trajectories), "batches": len(inputs),
        "initialization": "pure half-filled random Slater, followed by hard-wall exterior preparation",
        "window": "cycles 1..2Ny, no burn-in; finite-time, not asymptotic Lyapunov gaps",
        "nulls": "numerical singular nulls excluded by original estimator; no additional cutoff here",
        "fit": None, "yscale": "log in both panels",
        "source_script_sha256": sha256(Path(__file__)),
        "campaign_config_sha256": sha256(CONFIG_PATH), "inputs": inputs,
        "artifacts": {str(p.relative_to(HERE)): sha256(p) for p in artifacts},
    }
    (HERE / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed: {len(rows)} cases, {len(trajectories)} trajectories; figure {HERE / 'figures' / (STEM + '.pdf')}")
    print("Saved endpoint-mode groups:", [{k: v for k, v in group.items() if k != "sample_indices"}
                                          for group in inventory["slot09_cpu_replay_groups"]])


if __name__ == "__main__":
    main()
