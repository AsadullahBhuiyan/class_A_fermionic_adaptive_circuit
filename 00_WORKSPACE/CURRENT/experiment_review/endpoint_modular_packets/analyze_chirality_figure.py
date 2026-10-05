#!/usr/bin/env python3
"""Reduce existing endpoint frames for consistent handedness figures; no dynamics."""
from __future__ import annotations

import analyze_endpoint_packets as base  # Set BLAS thread limits before NumPy.
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

OUTPUT = base.HERE / "outputs/hard_n20x32_a1-1-3_y8_chirality_figure_v2"
REFERENCE = base.HERE / "outputs/hard_n20x32_a1-1-3_y8_early_time_clip_v1"
TIMES = np.linspace(0, 1, 101)
SNAPSHOTS = np.array([0., .1, .2, .5, 1.])
FIELDS = ("density", "dy_window", "dy_full", "window_charge", "longitudinal_probability")
CONTRACT = dict(base.CONTRACT, schema="modular_chirality_figure_v2",
                snapshot_times=SNAPSHOTS.tolist(), alpha1_1_curve_times=[0, 1, .01],
                alpha1_3_curve_times=[0, 1, .01],
                longitudinal_normalization="per cut wall charge, before averaging",
                contrast="(dy_x5-dy_x15)/2 within each trajectory")


def cache_valid(cache, identity):
    cache = Path(cache)
    try:
        receipt = json.loads(cache.with_suffix(".json").read_text())
        return (receipt["identity"] == identity and receipt["filename"] == cache.name
                and receipt["bytes"] == cache.stat().st_size
                and receipt["sha256"] == base.digest(cache))
    except (OSError, ValueError, KeyError):
        return False


def validate_probability(probability, displacement):
    if not np.isfinite(probability).all() or np.min(probability) < -1e-14:
        raise ValueError("Invalid conditional probability")
    np.testing.assert_allclose(probability.sum(axis=-1), 1, rtol=0, atol=1e-10)
    moment = probability @ (np.arange(base.LENGTH) - base.Y_START)
    np.testing.assert_allclose(moment, displacement, rtol=0, atol=1e-10)
    return float(np.max(abs(moment-displacement)))


def analyze_sample(job):
    source, index, alpha, sample, cache, identity = job
    cache = Path(cache)
    if cache_valid(cache, identity):
        return "reused"
    frames, ranks = base.load_frames(source)
    frame = frames[index, :, :int(ranks[index])]
    if frame.dtype != np.complex128 or not np.isfinite(frame).all():
        raise ValueError("Invalid endpoint frame")
    gram = float(np.max(abs(frame.conj().T @ frame - np.eye(frame.shape[1]))))
    if gram > 1e-9:
        raise ValueError("Nonorthonormal endpoint frame")
    sums, diagnostics = {}, []
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
            raise ValueError("Nonphysical covariance")
        result = base.evolve_spectrum(
            vals, vecs, nx=base.NX, length=base.LENGTH, walls=base.WALLS,
            y_start=base.Y_START, times=TIMES, snapshot_times=SNAPSHOTS,
            eps=base.EPS, return_longitudinal=True,
        )
        moment_error = validate_probability(result["longitudinal_probability"], result["dy_window"])
        for key in FIELDS:
            if key not in sums:
                sums[key] = np.zeros_like(result[key])
            sums[key] += result[key]/base.NY
        diagnostics.append([herm, result["charge_error"], result["window_charge"].min(), moment_error])
    validate_probability(sums["longitudinal_probability"], sums["dy_window"])
    base.atomic_npz(cache, **sums, diagnostics=np.array(diagnostics), frame_gram_error=gram,
                    sample_index=sample, alpha_1=alpha, times=TIMES, snapshot_times=SNAPSHOTS,
                    identity_json=json.dumps(identity, sort_keys=True))
    receipt = dict(identity=identity, filename=cache.name, bytes=cache.stat().st_size,
                   sha256=base.digest(cache))
    temp = cache.with_suffix(".tmp")
    temp.write_text(json.dumps(receipt, indent=2)+"\n")
    os.replace(temp, cache.with_suffix(".json"))
    return "computed"


def mean_sem(samples):
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)/np.sqrt(len(samples))


def contrast_samples(displacement):
    return (displacement[:, 0] - displacement[:, 1])/2


def aggregate(output, jobs, sources, count):
    arrays, checks = {}, {}
    reference = REFERENCE / "averaged_observables.npz"
    old_summary = json.loads((REFERENCE/"analysis_summary.json").read_text())
    if old_summary["sources"] != sources:
        raise ValueError("Endpoint sources differ from the reference analysis")
    with np.load(reference, allow_pickle=False) as old:
        ei = int(np.flatnonzero(old["epsilons"] == base.EPS)[0])
        np.testing.assert_allclose(TIMES, old["times"], rtol=0, atol=1e-12)
        old_indices = [int(np.flatnonzero(SNAPSHOTS == t)[0]) for t in old["snapshot_times"]]
        for alpha in (1, 3):
            selected = sorted((job for job in jobs if job[2] == alpha), key=lambda j: j[3])
            if [j[3] for j in selected] != list(range(count)):
                raise ValueError("Missing or duplicate trajectory IDs")
            records = []
            for job in selected:
                if not cache_valid(job[4], job[5]):
                    raise ValueError(f"Unverified cache: {job[4]}")
                with np.load(job[4], allow_pickle=False) as record:
                    if json.loads(str(record["identity_json"])) != job[5]:
                        raise ValueError("Cache identity mismatch")
                    records.append({key: record[key] for key in (*FIELDS, "diagnostics", "frame_gram_error")})
            errors = {}
            for key in FIELDS:
                samples = np.stack([r[key] for r in records])
                arrays[f"alpha{alpha}_{key}_samples"] = samples
                mean, sem = mean_sem(samples)
                arrays[f"alpha{alpha}_{key}_mean"] = mean
                arrays[f"alpha{alpha}_{key}_sem"] = sem
                if key != "longitudinal_probability":
                    compare = samples[:, :, old_indices] if key == "density" else samples
                    expected = old[f"alpha{alpha}_{key}_samples"][:count, ei]
                    errors[key] = float(np.max(abs(compare-expected)))
                    np.testing.assert_allclose(compare, expected, rtol=0, atol=1e-10)
            dy = arrays[f"alpha{alpha}_dy_window_samples"]
            moment = validate_probability(arrays[f"alpha{alpha}_longitudinal_probability_samples"], dy)
            contrast = contrast_samples(dy)
            arrays[f"alpha{alpha}_contrast_samples"] = contrast
            arrays[f"alpha{alpha}_contrast_mean"], arrays[f"alpha{alpha}_contrast_sem"] = mean_sem(contrast)
            diagnostics = np.concatenate([r["diagnostics"] for r in records])
            checks[str(alpha)] = dict(samples=count, cuts_per_sample=base.NY,
                baseline_max_errors=errors, max_moment_error=moment,
                max_charge_error=float(diagnostics[:, 1].max()),
                minimum_window_charge=float(diagnostics[:, 2].min()),
                max_frame_gram_error=max(float(r["frame_gram_error"]) for r in records))
    base.atomic_npz(output/"averaged_observables.npz", **arrays, times=TIMES,
                    snapshot_times=SNAPSHOTS, contract_json=json.dumps(CONTRACT, sort_keys=True))
    summary = dict(contract=CONTRACT, sources=sources, validation=checks,
                   samples_per_alpha=count, reference_sha256=base.digest(reference),
                   helper_sha256=base.digest(base.__file__), script_sha256=base.digest(__file__),
                   data_sha256=base.digest(output/"averaged_observables.npz"))
    (output/"analysis_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(checks, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not 2 <= args.sample_limit <= 100 or args.workers < 1:
        parser.error("Require 2..100 samples per alpha and at least one worker")
    if args.output.resolve() in (REFERENCE.resolve(), base.OUTPUT.resolve()):
        parser.error("Previous analyses must not be overwritten")
    if args.sample_limit != 100 and args.output.resolve() == OUTPUT.resolve():
        parser.error("Smoke tests require a separate output directory")
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, available[:args.workers])
    args.output.mkdir(parents=True, exist_ok=True)
    jobs, sources = base.collect_jobs(args.output, args.sample_limit)
    jobs = [(*j[:5], dict(j[5], contract=CONTRACT, script_sha256=base.digest(__file__),
                         helper_sha256=base.digest(base.__file__))) for j in jobs]
    reused = sum(cache_valid(j[4], j[5]) for j in jobs)
    print(json.dumps(dict(contract=CONTRACT, output=str(args.output), verified_sources=len(sources),
                          workers=args.workers, cpus=sorted(os.sched_getaffinity(0)),
                          tasks=len(jobs), reusable=reused, pending=len(jobs)-reused), indent=2), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(analyze_sample, j) for j in jobs if not cache_valid(j[4], j[5])]
        for future in tqdm(as_completed(pending), initial=reused, total=len(jobs),
                           desc="Conditional packet densities", unit="trajectory"):
            future.result()
    aggregate(args.output, jobs, sources, args.sample_limit)
    print(f"[done] Verified reduction saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
