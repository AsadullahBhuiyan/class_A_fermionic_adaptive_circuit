#!/usr/bin/env python3
"""Early-time modular packets and paired clipping sensitivity; no new dynamics."""
from __future__ import annotations

import analyze_endpoint_packets as base  # Sets single-threaded BLAS before NumPy.
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

EPSILONS = np.array([1e-8, 1e-10, 1e-12])
TIMES = np.linspace(0, 1, 101)
SNAPSHOTS = np.array([0., .1, .2])
OUTPUT = base.HERE / "outputs/hard_n20x32_a1-1-3_y8_early_time_clip_v1"
CONTRACT = dict(base.CONTRACT, schema="endpoint_modular_early_time_clip_v1",
                epsilon=EPSILONS.tolist(), snapshot_times=SNAPSHOTS.tolist(),
                alpha1_1_curve_times=[0, 1, .01], alpha1_3_curve_times=[0, 1, .01])
SPECTRAL_FIELDS = ["min_abs_energy", "bandwidth", "packet_sigma_energy_left",
                   "packet_sigma_energy_right", "clipped_weight_left",
                   "clipped_weight_right", "clipped_modes"]


def analyze(job):
    source, index, alpha, sample, cache, identity = job
    cache = Path(cache)
    receipt = cache.with_suffix(".json")
    if cache.exists() and receipt.exists():
        old = json.loads(receipt.read_text())
        if old.get("identity") == identity and old.get("sha256") == base.digest(cache):
            return "reused"
    frames, ranks = base.load_frames(source)
    frame = frames[index, :, :int(ranks[index])]
    if frame.dtype != np.complex128 or not np.isfinite(frame).all():
        raise ValueError("Invalid frame")
    gram = float(np.max(abs(frame.conj().T @ frame - np.eye(frame.shape[1]))))
    if gram > 1e-9:
        raise ValueError("Nonorthonormal source frame")
    sums, spectral, diagnostics = {}, [], []
    for y0 in range(base.NY):
        rows = (2*base.NX*((y0+np.arange(base.LENGTH)) % base.NY)[:, None]
                + np.arange(2*base.NX)).ravel()
        reduced = frame[rows]
        g = 2*(reduced @ reduced.conj().T) - np.eye(len(rows))
        herm = float(np.max(abs(g-g.conj().T)))
        if herm > 1e-10:
            raise ValueError("Non-Hermitian covariance")
        vals, vecs = np.linalg.eigh((g+g.conj().T)/2)
        if np.max(abs(vals)) > 1+1e-9:
            raise ValueError("Nonphysical covariance spectrum")
        cols = [2*base.NX*base.Y_START+2*x+o for x in base.WALLS for o in (0, 1)]
        # Spectral weights of the normalized, incoherent two-orbital packet.
        weights = (abs(vecs[cols].reshape(2, 2, len(vals)))**2).sum(axis=1)/2
        for ei, eps in enumerate(EPSILONS):
            result = base.evolve_spectrum(vals, vecs, nx=base.NX, length=base.LENGTH,
                       walls=base.WALLS, y_start=base.Y_START, times=TIMES,
                       snapshot_times=SNAPSHOTS, eps=eps)
            for key in ("density", "dy_window", "dy_full", "window_charge"):
                if key not in sums:
                    sums[key] = np.zeros((len(EPSILONS), *result[key].shape))
                sums[key][ei] += result[key]/base.NY
            energies = -2*np.arctanh(np.clip(vals, -1+eps, 1-eps))
            mean_e = weights @ energies
            sigma_e = np.sqrt(np.maximum(0, weights @ energies**2 - mean_e**2))
            clipped = abs(vals) > 1-eps
            spectral.append([y0, ei, float(np.min(abs(energies))), float(np.ptp(energies)),
                             *sigma_e, *(weights @ clipped), int(clipped.sum())])
            diagnostics.append([y0, ei, herm, result["charge_error"],
                                result["window_charge"].min()])
    base.atomic_npz(cache, **sums, spectral=np.array(spectral), diagnostics=np.array(diagnostics),
                    frame_gram_error=gram, times=TIMES, snapshot_times=SNAPSHOTS,
                    epsilons=EPSILONS, identity_json=json.dumps(identity, sort_keys=True))
    temp = receipt.with_suffix(".tmp")
    temp.write_text(json.dumps({"identity": identity, "sha256": base.digest(cache)}, indent=2)+"\n")
    os.replace(temp, receipt)
    return "computed"


def aggregate(output, jobs, sources, count):
    arrays, ledger, rows, spectrum_rows = {}, {}, [], []
    for alpha in (1, 3):
        records = []
        for job in jobs:
            if job[2] != alpha:
                continue
            if json.loads(Path(job[4]).with_suffix(".json").read_text())["sha256"] != base.digest(job[4]):
                raise ValueError("Cache checksum mismatch")
            with np.load(job[4], allow_pickle=False) as data:
                records.append({k: data[k] for k in data.files if k != "identity_json"})
        assert len(records) == count
        for key in ("density", "dy_window", "dy_full", "window_charge"):
            samples = np.stack([r[key] for r in records])
            arrays[f"alpha{alpha}_{key}_samples"] = samples
            arrays[f"alpha{alpha}_{key}_mean"] = samples.mean(axis=0)
            arrays[f"alpha{alpha}_{key}_sem"] = (samples.std(axis=0, ddof=1)/np.sqrt(count)
                                                  if count > 1 else np.full_like(samples[0], np.nan))
        diagnostics = np.concatenate([r["diagnostics"] for r in records])
        spectral = np.stack([r["spectral"] for r in records])
        arrays[f"alpha{alpha}_spectral_samples"] = spectral
        ledger[str(alpha)] = {"samples": count, "cuts_per_sample": base.NY,
           "max_hermiticity_error": float(diagnostics[:,2].max()),
           "max_charge_error": float(diagnostics[:,3].max()),
           "minimum_window_charge": float(diagnostics[:,4].min()),
           "max_frame_gram_error": max(float(r["frame_gram_error"]) for r in records)}
        for ei, eps in enumerate(EPSILONS):
            for ti, t in enumerate(TIMES):
                for p, wall in enumerate(base.WALLS):
                    paired = (arrays[f"alpha{alpha}_dy_window_samples"][:, ei, p, ti]
                              - arrays[f"alpha{alpha}_dy_window_samples"][:, 1, p, ti])
                    rows.append([alpha, eps, t, wall,
                        arrays[f"alpha{alpha}_dy_window_mean"][ei,p,ti],
                        arrays[f"alpha{alpha}_dy_window_sem"][ei,p,ti],
                        paired.mean(), paired.std(ddof=1)/np.sqrt(count) if count>1 else np.nan])
            selected = spectral[:, ei::len(EPSILONS), 2:].reshape(-1,len(SPECTRAL_FIELDS))
            spectrum_rows.append([alpha, eps, *np.median(selected, axis=0)])
    base.atomic_npz(output/"averaged_observables.npz", **arrays, times=TIMES,
         snapshot_times=SNAPSHOTS, epsilons=EPSILONS, contract_json=json.dumps(CONTRACT, sort_keys=True))
    for name, header, values in (
        ("displacement.csv", ["alpha_1","epsilon","t_mod","wall_x","mean_dy","trajectory_sem",
                               "paired_difference_from_1e-10","paired_difference_sem"], rows),
        ("spectral_medians.csv", ["alpha_1","epsilon", *SPECTRAL_FIELDS], spectrum_rows)):
        with (output/name).open("w", newline="") as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(values)
    baseline = {key: value[1] for key, value in arrays.items() if key.endswith(("_mean","_sem"))}
    base.plot_products(output, baseline, count, snapshot_times=SNAPSHOTS, times=TIMES, time_max=1)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.8), sharex=True)
    for alpha, ax in zip((1,3), axes):
        for ei, (eps, color) in enumerate(zip(EPSILONS, ("#D55E00", "#009E73", "#0072B2"))):
            for p, ls in enumerate(("-", "--")):
                mean = arrays[f"alpha{alpha}_dy_window_mean"][ei,p]
                sem = arrays[f"alpha{alpha}_dy_window_sem"][ei,p]
                ax.plot(TIMES, mean, color=color, ls=ls, lw=1, label=rf"$\epsilon={eps:.0e}$" if p==0 else None)
                ax.fill_between(TIMES, mean-sem, mean+sem, color=color, alpha=.12, lw=0)
        ax.axhline(0, color=".5", lw=.5)
        ax.set(xlim=(0,1), xlabel=r"modular time $t_{\rm mod}$", ylabel=r"$\langle\Delta y\rangle$",
               title=rf"$\alpha_1={alpha}$; solid: $x=5$, dashed: $x=15$")
        ax.legend(fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output/f"clipping_sensitivity.{suffix}", dpi=300)
    plt.close(fig)
    summary = {"contract": CONTRACT, "sources": sources, "diagnostics": ledger,
               "script_sha256": base.digest(__file__), "helper_sha256": base.digest(base.__file__),
               "spectral_summary": "Medians across sample/cut pairs; not spectral gaps of an averaged Hamiltonian.",
               "uncertainty": "Origins averaged within trajectory; paired cutoff differences use the same trajectories."}
    (output/"analysis_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(ledger, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.sample_limit <= 100 or args.workers < 1:
        parser.error("Invalid sample count or workers")
    if args.sample_limit != 100 and args.output == OUTPUT:
        parser.error("Use separate output for smoke test")
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:args.workers])
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"contract": CONTRACT, "output": str(args.output),
                      "cpus": sorted(os.sched_getaffinity(0))}), flush=True)
    jobs, sources = base.collect_jobs(args.output, args.sample_limit)
    jobs = [(*job[:5], dict(job[5], contract=CONTRACT,
             script_sha256=base.digest(__file__), helper_sha256=base.digest(base.__file__))) for job in jobs]
    print(f"Verified {len(sources)} source shards; {len(jobs)} trajectories, 32 cuts, 3 cutoffs", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Early packets / clipping", unit="trajectory"):
            future.result()
    aggregate(args.output, jobs, sources, args.sample_limit)
    print(f"[done] {args.output}", flush=True)


if __name__ == "__main__":
    main()
