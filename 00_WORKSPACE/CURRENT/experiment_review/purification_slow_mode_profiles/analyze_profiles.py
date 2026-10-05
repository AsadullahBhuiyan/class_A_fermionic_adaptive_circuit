#!/usr/bin/env python3
"""Recover resolved slot-07 purification eigenmodes and compare slot-04 profiles."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import subprocess

import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BUNDLES = REPO / "00_WORKSPACE/CURRENT/final_production_new_designs"
CAP = 1e-9
DEGENERACY = 1e-10
NY04 = (20, 22, 24, 26, 28, 30, 36, 40)
NY07 = (20, 30, 40)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path):
    receipt = Path(path).with_suffix(".complete.json")
    d = json.loads(receipt.read_text())
    if d["result_filename"] != Path(path).name:
        raise ValueError("source filename mismatch")
    if Path(path).stat().st_size != d["result_bytes"] or sha256(path) != d["result_sha256"]:
        raise ValueError(f"source checksum mismatch: {path}")
    return d


def finite_mask(nu, cap=CAP):
    nu = np.asarray(nu)
    if not np.isfinite(nu).all() or nu.min(initial=0) < -1e-8 or nu.max(initial=1) > 1 + 1e-8:
        raise ValueError("nonphysical occupation spectrum")
    return (nu > cap) & (nu < 1 - cap)


def profiles(vectors, indices, nx=20):
    x = (np.asarray(indices) // 2) % nx
    return np.stack([np.sum(np.abs(vectors[x == i]) ** 2, axis=0) for i in range(nx)], axis=1)


def extract(centered, active_indices, cycles):
    centered = np.asarray(centered, dtype=np.complex128)
    herm = float(np.max(np.abs(centered - centered.conj().T)))
    if herm > 1e-8:
        raise ValueError(f"non-Hermitian covariance: {herm}")
    c = (centered + centered.conj().T) / 4 + np.eye(len(centered)) / 2
    nu, vectors = eigh(c, check_finite=True, driver="evd")
    valid = finite_mask(nu)
    selected = np.flatnonzero(valid)
    rates = (np.log1p(-nu[valid]) - np.log(nu[valid])) / (2 * cycles)
    order = np.argsort(np.abs(rates), kind="stable")
    selected, rates = selected[order], rates[order]
    v = vectors[:, selected].copy()
    if v.shape[1]:
        pivot = np.argmax(np.abs(v), axis=0)
        phases = v[pivot, np.arange(v.shape[1])]
        v *= (phases.conj() / np.abs(phases))[None, :]
        residuals = np.linalg.norm(c @ v - v * nu[selected], axis=0)
        gram_error = float(np.max(np.abs(v.conj().T @ v - np.eye(v.shape[1]))))
    else:
        residuals = np.empty(0)
        gram_error = 0.
    if residuals.max(initial=0) > 1e-9 or gram_error > 1e-9:
        raise ValueError("finite eigenvector residual or orthogonality check failed")
    # Identify clusters from the FULL spectrum, including neighboring capped modes.
    cluster = np.r_[0, np.cumsum(np.diff(nu) > DEGENERACY)]
    sizes = np.bincount(cluster)
    neighbors = np.minimum(np.r_[np.inf, np.diff(nu)], np.r_[np.diff(nu), np.inf])
    separate = all(np.all(valid[cluster == cluster[i]]) for i in selected)
    p = profiles(v, active_indices)
    np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-10)
    mean_profile = p.mean(axis=0) if len(p) else np.full(20, np.nan)
    return {
        "occupations": nu, "finite_mask": valid, "finite_count": np.asarray(valid.sum()),
        "finite_full_spectrum_indices": selected, "finite_occupations": nu[selected],
        "finite_signed_rates": rates, "finite_eigenvectors": v,
        "finite_mode_x_profiles": p, "finite_subspace_mean_x_profile": mean_profile,
        "finite_mode_cluster_ids": cluster[selected],
        "finite_mode_cluster_sizes": sizes[cluster[selected]],
        "finite_mode_individually_separated": sizes[cluster[selected]] == 1,
        "finite_mode_neighbor_gaps": neighbors[selected],
        "finite_subspace_separated_from_caps": np.asarray(separate),
        "finite_eigensolver_residuals": residuals,
        "eigenvector_gram_error": np.asarray(gram_error), "hermiticity_error": np.asarray(herm),
        "counts_at_caps_1e_8_1e_9_1e_10": np.asarray([finite_mask(nu, t).sum() for t in (1e-8, 1e-9, 1e-10)]),
    }


def slot07_shard(path_string):
    path = Path(path_string)
    d = verify(path)
    rows = []
    with np.load(path, allow_pickle=False) as z:
        ny, wall = int(z["Ny"]), str(z["construction"])
        ids = z["sample_indices"]
        np.testing.assert_array_equal(ids, d["sample_indices"])
        if int(z["cycles"][-1]) != 4 * ny or d["cycles"] != 4 * ny:
            raise ValueError("incorrect slot-07 endpoint")
        if z["G_final"].dtype != np.complex128:
            raise ValueError("incorrect covariance dtype")
        states = z["G_final"]
        saved_spectrum = z["occupation_spectrum"][:, -1]
        indices = np.arange(40 * ny)
        active = indices[((indices // 2) % 20 >= 5) & ((indices // 2) % 20 <= 15)] if wall == "hard" else indices
        exterior = np.setdiff1d(indices, active)
        for row, sample in enumerate(ids):
            full = states[row]
            coupling = float(np.max(np.abs(full[np.ix_(active, exterior)]), initial=0))
            if coupling > 1e-9:
                raise ValueError("hard-wall active-exterior coupling prevents slab extraction")
            with threadpool_limits(limits=2):
                result = extract(full[np.ix_(active, active)], active, 4 * ny)
            expected_nu = saved_spectrum[row]
            if wall == "hard":
                ext_nu = (np.diag(full)[exterior].real + 1) / 2
                if np.any(np.minimum(np.abs(ext_nu), np.abs(1 - ext_nu)) > CAP):
                    raise ValueError("hard-wall exterior is not pure")
                reconstructed = np.sort(np.r_[result["occupations"], ext_nu])
            else:
                reconstructed = result["occupations"]
            error = float(np.max(np.abs(reconstructed - expected_nu)))
            if error > 1e-8:
                raise ValueError(f"CPU eigenspectrum disagrees with saved GPU spectrum: {error}")
            saved_count = int(finite_mask(expected_nu).sum())
            result.update({"Nx": np.asarray(20), "Ny": np.asarray(ny), "construction": np.asarray(wall),
                           "sample_index": np.asarray(sample), "cycles": np.asarray(4 * ny),
                           "alpha_1": np.asarray(1.), "alpha_2": np.asarray(30.), "nshell": np.asarray(1),
                           "active_indices": active, "cap_tolerance": np.asarray(CAP),
                           "degeneracy_tolerance": np.asarray(DEGENERACY),
                           "source_result_sha256": np.asarray(d["result_sha256"]),
                           "source_configuration_hash": np.asarray(d["configuration_hash"]),
                           "source_hashes_json": np.asarray(json.dumps(d["source_hashes"], sort_keys=True)),
                           "source_result": np.asarray(str(path.relative_to(REPO))),
                           "source_saved_finite_count": np.asarray(saved_count),
                           "source_spectrum_max_abs_difference": np.asarray(error),
                           "active_exterior_coupling": np.asarray(coupling),
                           "rate_definition": np.asarray("lambda=[log(1-nu)-log(nu)]/(2*T)")})
            out = HERE / "endpoint_modes" / wall / f"Ny{ny:03d}" / f"sample_{sample:03d}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            temporary = out.with_suffix(".partial.npz")
            np.savez_compressed(temporary, **result)
            os.replace(temporary, out)
            rows.append({"dataset": "slot07", "wall": wall, "Ny": ny, "sample": int(sample),
                         "cycles": 4 * ny, "finite_count": int(result["finite_count"]),
                         "source_finite_count": saved_count,
                         "count_at_cap_1e_8": int(result["counts_at_caps_1e_8_1e_9_1e_10"][0]),
                         "count_at_cap_1e_10": int(result["counts_at_caps_1e_8_1e_9_1e_10"][2]),
                         "individually_separated_modes": int(result["finite_mode_individually_separated"].sum()),
                         "subspace_separated_from_caps": bool(result["finite_subspace_separated_from_caps"]),
                         "profile": result["finite_subspace_mean_x_profile"],
                         "max_eigen_residual": float(result["finite_eigensolver_residuals"].max(initial=0)),
                         "spectrum_max_abs_difference": error,
                         "result_file": str(out.relative_to(HERE)), "result_sha256": sha256(out)})
    return rows, {"path": str(path.relative_to(REPO)), "sha256": d["result_sha256"], "bytes": d["result_bytes"]}


def slot04():
    root = BUNDLES / "04_maxmix_manybody_lyapunov_pilot/gpu_data/maxmix_manybody_lyapunov_nx20_ny20to40_s100_gpu_v2/results"
    rows, history, sources = [], [], []
    for path in tqdm(sorted(root.rglob("*.npz")), desc="Audit slot-04 profiles", unit="shard"):
        d = verify(path)
        sources.append({"path": str(path.relative_to(REPO)), "sha256": d["result_sha256"], "bytes": d["result_bytes"]})
        with np.load(path, allow_pickle=False) as z:
            ny = int(z["Ny"])
            ids = z["global_sample_indices"]
            cycles = z["spectrum_cycles"]
            np.testing.assert_array_equal(ids, d["global_sample_indices"])
            nu, costs, p = z["occupations"], z["soft_mode_flip_costs"], z["soft_mode_x_profiles"]
            selected_nu = z["soft_mode_occupations"]
            np.testing.assert_array_equal(np.isfinite(costs), finite_mask(selected_nu))
            if int(cycles[-1]) != 2 * ny:
                raise ValueError("incorrect slot-04 endpoint")
            for i, sample in enumerate(ids):
                keep = np.isfinite(costs[i, -1])
                count = int(finite_mask(nu[i, -1]).sum())
                if count != keep.sum():
                    raise ValueError("slot-04 endpoint saved modes do not cover all finite modes")
                np.testing.assert_allclose(p[i, -1, keep].sum(axis=1), 1, atol=1e-8)
                rows.append({"dataset": "slot04", "wall": "hard", "Ny": ny, "sample": int(sample),
                             "cycles": 2 * ny, "finite_count": count,
                             "profile": p[i, -1, keep].mean(axis=0),
                             "subspace_separated_from_caps": None})
                for t, cycle in enumerate(cycles):
                    history.append({"Ny": ny, "sample": int(sample), "cycle": int(cycle),
                                    "finite_count": int(finite_mask(nu[i, t]).sum()),
                                    "saved_finite_profiles": int(np.isfinite(costs[i, t]).sum())})
    return rows, history, sources


def save_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["wall"], r["Ny"]), []).append(r)
    summaries, arrays = [], {}
    for key, group in sorted(groups.items()):
        if len(group) != 100 or len({r["sample"] for r in group}) != 100:
            raise ValueError(f"incomplete or duplicate case: {key}")
        counts = np.array([r["finite_count"] for r in group])
        # Equal weight per trajectory; soft rows with no finite subspace are
        # excluded explicitly, never zero-filled and never called S=100.
        usable = [r for r in group if r["finite_count"] > 0 and r["subspace_separated_from_caps"] is not False]
        p = np.array([r["profile"] for r in usable])
        mean = p.mean(axis=0) if len(p) else np.full(20, np.nan)
        sem = p.std(axis=0, ddof=1) / np.sqrt(len(p)) if len(p) > 1 else np.full(20, np.nan)
        arrays[key] = (mean, sem)
        summaries.append({"dataset": key[0], "wall": key[1], "Ny": key[2], "cycles": group[0]["cycles"],
                          "samples": 100, "samples_with_finite_modes": int(np.sum(counts > 0)),
                          "profile_samples": len(usable), "finite_count_min": int(counts.min()),
                          "finite_count_max": int(counts.max()), "finite_count_mean": float(counts.mean()),
                          "finite_count_sem": float(counts.std(ddof=1) / 10),
                          "mean_inner_wall_weight_x5_6_14_15": float(mean[[5, 6, 14, 15]].sum()),
                          "mean_wall_neighborhood_weight_x4to6_14to16": float(mean[[4, 5, 6, 14, 15, 16]].sum()),
                          "mean_bulk_weight_x7to13": float(mean[7:14].sum())})
    return summaries, arrays


def plot(summaries, arrays):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["TEXINPUTS"] = str(HERE.parent / "hard_wall_tangent_gap_analysis/latex_support") + os.pathsep + os.environ.get("TEXINPUTS", "")
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif"],
                         "font.size": 8, "text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}\renewcommand{\familydefault}{\sfdefault}",
                         "axes.linewidth": .7, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True, "legend.frameon": False})
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.1), layout="constrained")
    specs = [("slot04", "hard", r"Slot 04: hard, $T=2N_y$"),
             ("slot07", "hard", r"Slot 07: hard, $T=4N_y$"),
             ("slot07", "soft", r"Slot 07: soft, $T=4N_y$")]
    for ax, (dataset, wall, title) in zip(axes.flat, specs):
        for ny, color, marker, ls in zip(NY07, ["#d62728", "#2ca02c", "#1f77b4"], ["^", "s", "o"], [":", "--", "-"]):
            mean, sem = arrays[(dataset, wall, ny)]
            n = next(r["profile_samples"] for r in summaries if (r["dataset"], r["wall"], r["Ny"]) == (dataset, wall, ny))
            ax.errorbar(np.arange(20), mean, yerr=sem, color=color, marker=marker, linestyle=ls,
                        markersize=2.5, markerfacecolor="white", linewidth=.8, capsize=1,
                        label=rf"$N_y={ny}$, $S_{{\rm res}}={n}$")
        for x in [5, 15]: ax.axvline(x, color=".5", linestyle="--", linewidth=.6, zorder=0)
        ax.set(xlabel=r"$x$", ylabel=r"Mean resolved-subspace weight $p(x)$", title=title, xlim=(0, 19), ylim=(0, None), xticks=[0, 5, 10, 15, 19])
        ax.legend(fontsize=8, loc="upper center")
    for (dataset, wall, _), color, marker, ls in zip(specs, ["#1f77b4", "#d62728", "#2ca02c"], ["o", "^", "s"], ["-", ":", "--"]):
        selected = [r for r in summaries if (r["dataset"], r["wall"]) == (dataset, wall)]
        axes[1, 1].errorbar([r["Ny"] for r in selected], [r["finite_count_mean"] for r in selected],
                            yerr=[r["finite_count_sem"] for r in selected], color=color, marker=marker,
                            linestyle=ls, markersize=3, markerfacecolor="white", capsize=2,
                            label=f"{dataset.replace('slot', 'Slot ')} {wall}")
    axes[1, 1].set(xlabel=r"$N_y$", ylabel="Resolved modes per trajectory", title=r"Occupation cap tolerance $10^{-9}$", ylim=(0, 6.3), xticks=[20, 24, 28, 32, 36, 40])
    axes[1, 1].legend(loc="upper right", bbox_to_anchor=(1, .8), fontsize=8)
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
        ax.text(-.04, 1.035, label, transform=ax.transAxes, ha="right")
    (HERE / "figures").mkdir(exist_ok=True)
    pdf = HERE / "figures/purification_resolved_mode_profiles.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(pdf.with_suffix(""))], check=True)


def main():
    root = BUNDLES / "07_maxmix_hard_soft_purification/gpu_data"
    paths = []
    for wall, rev in [("hard", 2), ("soft", 3)]:
        paths.extend(sorted((root / f"maxmix_hard_soft_purification_nx20_ny20-40_s100_4ny_raster_v{rev}" / wall).rglob("*.npz")))
    if len(paths) != 120: raise ValueError("expected 120 slot-07 shards")
    rows, sources = [], []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(slot07_shard, str(p)) for p in paths]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extract slot-07 eigenmodes", unit="shard"):
            r, s = future.result(); rows.extend(r); sources.append(s)
    old, history, old_sources = slot04()
    rows.extend(old); sources.extend(old_sources)
    summaries, arrays = summarize(rows)
    (HERE / "tables").mkdir(exist_ok=True)
    save_csv(HERE / "tables/summary.csv", summaries)
    save_csv(HERE / "tables/slot04_time_resolved_counts.csv", history)
    seven = [r for r in rows if r["dataset"] == "slot07"]
    save_csv(HERE / "tables/slot07_sample_diagnostics.csv", [{k: v for k, v in r.items() if k != "profile"} for r in seven])
    profile_rows = []
    for key, (mean, sem) in sorted(arrays.items()):
        for x in range(20):
            profile_rows.append({"dataset": key[0], "wall": key[1], "Ny": key[2], "x": x,
                                 "mean": float(mean[x]), "sem": float(sem[x])})
    save_csv(HERE / "tables/mean_profiles.csv", profile_rows)
    plot(summaries, arrays)
    manifest = {"source_inputs": sorted(sources, key=lambda d: d["path"]),
                "slot07_samples": len(seven), "slot04_samples": len(old),
                "cap_tolerance": CAP, "degeneracy_tolerance": DEGENERACY,
                "counts_matching_saved_spectra": sum(r["finite_count"] == r["source_finite_count"] for r in seven),
                "max_eigen_residual": max(r["max_eigen_residual"] for r in seven),
                "max_source_spectrum_difference": max(r["spectrum_max_abs_difference"] for r in seven),
                "profile_estimator": "mean over all resolved modes within trajectory, then equal-weight mean over trajectories with nonempty spectrally separated finite subspace",
                "soft_profile_caveat": "conditional subset; zero-finite-mode trajectories excluded, not zero-filled",
                "comparison_caveat": "independent campaigns at different endpoints 2Ny and 4Ny; not a same-time hard/soft comparison across slots",
                "source_script_sha256": sha256(Path(__file__)), "cases": summaries,
                "outputs": [{k: v for k, v in r.items() if k in ("result_file", "result_sha256")} for r in seven]}
    (HERE / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("source_inputs", "outputs")}, indent=2), flush=True)


if __name__ == "__main__": main()
