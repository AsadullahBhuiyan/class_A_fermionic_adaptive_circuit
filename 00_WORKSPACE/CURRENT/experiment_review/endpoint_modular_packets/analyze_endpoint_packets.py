#!/usr/bin/env python3
"""Sample/cut-resolved modular packets from campaign-09 occupied frames."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REVISION = "pure_tangent_replay_nx20_ny24-32_hard-soft_a1-1-3_s100_v1"
SOURCE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/09_pure_tangent_replay_acquisition/gpu_data" / REVISION
OUTPUT = HERE / "outputs/hard_n20x32_a1-1-3_y8_observable_mean_v1"
NX, NY, LENGTH, Y_START = 20, 32, 16, 8
WALLS = (5, 15)
SNAPSHOTS = np.array([0., .5, 1., 1.5])
TIMES = np.linspace(0, 32, 3201)
EPS = 1e-10
CONTRACT = {
    "schema": "endpoint_modular_packet_observable_average_v1",
    "Nx": NX, "Ny": NY, "wall": "hard", "alpha_1": [1, 3], "alpha_2": 30,
    "nshell": 1, "endpoint_cycle": 64, "samples_per_alpha": 100,
    "initial_state": "random pure, with Born-conditioned hard-wall exterior preparation",
    "perfect_correction": True, "sequence": "raster_y", "dtype": "complex128",
    "source_positions_cut_relative": [[x, Y_START] for x in WALLS],
    "subsystem": "all x; 16 consecutive periodic y rows beginning at y0",
    "y0_values": list(range(NY)), "packet_charge": 2,
    "epsilon": EPS, "modular_generator": "-2 atanh(restricted centered covariance)",
    "time_normalization": "none", "snapshot_times": SNAPSHOTS.tolist(),
    "alpha1_1_curve_times": [0, 32, .01], "window_radius": 2,
    "averaging": "evolve separately; average observables over cuts within sample, then samples",
    "uncertainty": "SEM over independent trajectories, not over cuts",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path, **payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(temp, path)


def evolve(g, *, nx, length, walls, y_start, times, snapshot_times, eps=EPS):
    """Exact spectral propagation; compute each packet's normalized COM first."""
    g = np.asarray(g, dtype=np.complex128)
    n = 2 * nx * length
    if g.shape != (n, n) or not 0 <= y_start < length:
        raise ValueError("Invalid reduced covariance or source coordinate")
    herm = float(np.max(np.abs(g - g.conj().T)))
    if herm > 1e-10:
        raise ValueError(f"Non-Hermitian covariance: {herm}")
    eigenvalues, vectors = np.linalg.eigh((g + g.conj().T) / 2)
    if np.max(np.abs(eigenvalues)) > 1 + 1e-9:
        raise ValueError("Covariance eigenvalue outside physical interval")
    result = evolve_spectrum(eigenvalues, vectors, nx=nx, length=length,
                            walls=walls, y_start=y_start, times=times,
                            snapshot_times=snapshot_times, eps=eps)
    result["hermiticity_error"] = herm
    return result


def evolve_spectrum(eigenvalues, vectors, *, nx, length, walls, y_start,
                    times, snapshot_times, eps=EPS, return_longitudinal=False):
    """Reuse one covariance eigensystem for paired clipping-sensitivity checks."""
    n = 2 * nx * length
    if not 0 < eps < 1 or vectors.shape != (n, n):
        raise ValueError("Invalid clipping threshold or eigensystem shape")
    energies = -2 * np.arctanh(np.clip(eigenvalues, -1 + eps, 1 - eps))
    source = [2 * nx * y_start + 2 * x + o for x in walls for o in (0, 1)]
    coefficients = vectors[source].conj().T
    times = np.asarray(times)
    snapshot_indices = [int(np.argmin(abs(times - t))) for t in snapshot_times]
    if not np.allclose(times[snapshot_indices], snapshot_times, rtol=0, atol=1e-12):
        raise ValueError("Snapshot time absent from evolution grid")
    p = len(walls)
    density_snapshots = np.empty((p, len(snapshot_times), length, nx))
    window_com, full_com, window_charge = [np.empty((p, len(times))) for _ in range(3)]
    longitudinal = np.empty((p, len(times), length)) if return_longitudinal else None
    charge_error = 0.
    for start in range(0, len(times), 128):
        stop = min(start + 128, len(times))
        phase = np.exp(-1j * energies[:, None] * times[None, start:stop])
        amplitudes = vectors @ (phase[:, :, None] * coefficients[:, None, :]).reshape(n, -1)
        # Rows: y,x,output orbital. Columns: time,packet,input orbital.
        density = (abs(amplitudes.reshape(length, nx, 2, stop-start, p, 2))**2).sum(axis=(2, 5))
        charge = density.sum(axis=(0, 1))  # time,packet
        charge_error = max(charge_error, float(np.max(abs(charge - 2))))
        full_com[:, start:stop] = (np.einsum("yxtp,y->tp", density, np.arange(length)) / charge).T
        for packet, wall in enumerate(walls):
            distance = abs(np.arange(nx) - wall)
            mask = np.minimum(distance, nx-distance) <= 2
            local = density[:, :, :, packet][:, mask, :]
            q = local.sum(axis=(0, 1))
            if np.min(q) <= 1e-10:
                raise ValueError("Vanishing wall-window packet charge")
            window_charge[packet, start:stop] = q
            window_com[packet, start:stop] = np.einsum("yxt,y->t", local, np.arange(length)) / q
            if return_longitudinal:
                # Normalize each realization BEFORE any cut/trajectory average.
                longitudinal[packet, start:stop] = (local.sum(axis=1) / q).T
        for j, index in enumerate(snapshot_indices):
            if start <= index < stop:
                density_snapshots[:, j] = density[:, :, index-start, :].transpose(2, 0, 1)
    if charge_error > 1e-10:
        raise ValueError(f"Packet charge error: {charge_error}")
    result = {
        "density": density_snapshots,
        "dy_window": window_com - window_com[:, [0]],
        "dy_full": full_com - full_com[:, [0]],
        "window_charge": window_charge,
        "charge_error": charge_error,
        "clipped_modes": int(np.count_nonzero(abs(eigenvalues) > 1-eps)),
    }
    if return_longitudinal:
        result["longitudinal_probability"] = longitudinal
    return result


@lru_cache(maxsize=1)
def load_frames(path):
    with np.load(path, allow_pickle=False) as data:
        return data["final_frame"], data["final_ranks"]


def analyze_sample(job):
    source, index, alpha, sample, cache, identity = job
    cache = Path(cache)
    receipt = cache.with_suffix(".json")
    if cache.is_file() and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("identity") == identity and saved.get("sha256") == digest(cache):
            return alpha, sample, "reused"
    frames, ranks = load_frames(source)
    frame = frames[index, :, :int(ranks[index])]
    if frame.dtype != np.complex128 or not np.isfinite(frame).all():
        raise ValueError("Invalid occupied frame")
    gram_error = float(np.max(abs(frame.conj().T @ frame - np.eye(frame.shape[1]))))
    if gram_error > 1e-9:
        raise ValueError(f"Nonorthonormal frame: {gram_error}")
    times = TIMES if alpha == 1 else SNAPSHOTS
    sums = {}
    diagnostics = []
    for y0 in range(NY):
        rows = (2 * NX * ((y0 + np.arange(LENGTH)) % NY)[:, None] + np.arange(2*NX)).ravel()
        restricted = frame[rows]
        g = 2 * (restricted @ restricted.conj().T) - np.eye(len(rows))
        result = evolve(g, nx=NX, length=LENGTH, walls=WALLS, y_start=Y_START,
                        times=times, snapshot_times=SNAPSHOTS)
        for key in ("density", "dy_window", "dy_full", "window_charge"):
            if key not in sums:
                sums[key] = np.zeros_like(result[key])
            sums[key] += result[key] / NY
        diagnostics.append([result["charge_error"], result["hermiticity_error"],
                            result["clipped_modes"], result["window_charge"].min()])
    atomic_npz(cache, **sums, times=times, snapshot_times=SNAPSHOTS,
               cut_diagnostics=np.asarray(diagnostics), frame_gram_error=gram_error,
               alpha_1=alpha, sample_index=sample, identity_json=json.dumps(identity, sort_keys=True))
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "sha256": digest(cache)}, indent=2)+"\n")
    os.replace(temporary, receipt)
    return alpha, sample, "computed"


def collect_jobs(output, sample_limit):
    jobs, sources = [], []
    script_hash = digest(__file__)
    for alpha in (1, 3):
        seen = []
        for path in sorted((SOURCE / f"hard/Ny032/alpha1_{alpha}").glob("*.npz")):
            receipt = json.loads(path.with_suffix(".complete.json").read_text())
            if receipt["result_filename"] != path.name or receipt["result_bytes"] != path.stat().st_size:
                raise ValueError(f"Invalid source completion: {path}")
            file_hash = digest(path)
            if file_hash != receipt["result_sha256"]:
                raise ValueError(f"Source checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as data:
                for key, expected in (("Nx", NX), ("Ny", NY), ("alpha_1", alpha),
                                      ("alpha_2", 30), ("nshell", 1), ("cycles_total", 64),
                                      ("construction", "hard"), ("sequence", "raster_y")):
                    if data[key].item() != expected:
                        raise ValueError(f"Source contract mismatch: {key}")
                ids = data["case_sample_indices"].tolist()
            if ids != receipt["case_sample_indices"]:
                raise ValueError("Source sample identity mismatch")
            sources.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash,
                            "receipt_sha256": digest(path.with_suffix(".complete.json"))})
            seen.extend(ids)
            for index, sample in enumerate(ids):
                if sample >= sample_limit:
                    continue
                cache = output / f"sample_cache/alpha1_{alpha}/sample_{sample:03d}.npz"
                identity = {"contract": CONTRACT, "source_sha256": file_hash,
                            "script_sha256": script_hash, "sample": sample, "alpha_1": alpha}
                jobs.append((str(path), index, alpha, sample, str(cache), identity))
        if sorted(seen) != list(range(100)):
            raise ValueError(f"alpha1={alpha}: source ensemble incomplete")
    return jobs, sources


def plot_products(output, arrays, samples, *, snapshot_times=SNAPSHOTS,
                  times=TIMES, time_max=4):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
        "mathtext.fontset": "cm", "font.size": 8, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "legend.frameon": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.05, 3.0))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1.5, 1), wspace=.43)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    colors = ("#332288", "#E69F00", "#009E73", "#CC79A7")
    xx, yy = np.meshgrid(np.arange(NX), np.arange(LENGTH))
    for alpha, ax, letter in ((1, axes[0], "a"), (3, axes[2], "c")):
        density = arrays[f"alpha{alpha}_density_mean"]
        for packet in range(2):
            for ti, color in enumerate(colors[:len(snapshot_times)]):
                values = density[packet, ti]
                mask = values > 1e-4
                ax.scatter(xx[mask], yy[mask], s=85*np.sqrt(values[mask]/2),
                           facecolors=color, edgecolors=color, linewidths=.2, zorder=5-ti)
        for wall in WALLS:
            ax.axvline(wall, color=".45", ls=":", lw=.8, zorder=0)
        ax.set(xlim=(-.5, 19.5), ylim=(-.5, 15.5), xlabel="$x$", ylabel=r"$y-y_0$",
               title=rf"$\alpha_1={alpha}$" + (" (control)" if alpha==3 else ""))
        ax.set_aspect("equal")
        ax.set_xticks([0, 5, 15, 19]); ax.set_yticks([0, 4, 8, 12, 15])
        ax.text(-.23, 1.08, f"({letter})", transform=ax.transAxes)
    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=3.5, label=f"{t:g}")
               for t,c in zip(snapshot_times, colors)]
    axes[0].legend(handles=handles, title=r"$t_{\rm mod}$", loc="upper left", fontsize=6.5,
                   title_fontsize=7, handlelength=.5, handletextpad=.3, labelspacing=.2)
    for packet, (wall, color, ls) in enumerate(zip(WALLS, ("#D55E00", "#0072B2"), ("-", "--"))):
        mean, sem = arrays["alpha1_dy_window_mean"][packet], arrays["alpha1_dy_window_sem"][packet]
        visible = times <= time_max
        axes[1].plot(times[visible], mean[visible], color=color, ls=ls, lw=1, label=f"$({wall},8)$")
        axes[1].fill_between(times[visible], (mean-sem)[visible], (mean+sem)[visible], color=color, alpha=.18, lw=0)
    axes[1].axhline(0, color=".5", lw=.6)
    axes[1].set(xlim=(0,time_max), xlabel=r"modular time $t_{\rm mod}$", ylabel=r"$\langle\Delta y\rangle$", title=r"$\alpha_1=1$: mean $\pm$ SEM")
    axes[1].set_xticks(np.linspace(0,time_max,5)); axes[1].legend(fontsize=7, ncol=2, loc="best")
    axes[1].text(-.23,1.08,"(b)",transform=axes[1].transAxes)
    fig.suptitle(rf"$20\times32$, cycle 64; $S={samples}$ per $\alpha_1$; packets at $y-y_0=8$", y=.99, fontsize=9)
    fig.subplots_adjust(left=.07,right=.98,bottom=.2,top=.78)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"modular_packets_n20x32_y8.{suffix}", dpi=300)
    plt.close(fig)


def aggregate(output, jobs, sources, sample_limit):
    arrays, diagnostics = {}, {}
    for alpha in (1, 3):
        records = []
        for job in jobs:
            if job[2] != alpha: continue
            with np.load(job[4], allow_pickle=False) as data:
                records.append({k: data[k] for k in data.files if k != "identity_json"})
        if len(records) != sample_limit:
            raise ValueError("Missing analyzed trajectories")
        for key in ("density", "dy_window", "dy_full", "window_charge"):
            samples = np.stack([r[key] for r in records])
            arrays[f"alpha{alpha}_{key}_samples"] = samples
            arrays[f"alpha{alpha}_{key}_mean"] = samples.mean(axis=0)
            arrays[f"alpha{alpha}_{key}_sem"] = samples.std(axis=0,ddof=1)/np.sqrt(sample_limit) if sample_limit>1 else np.full_like(samples[0],np.nan)
        d = np.concatenate([r["cut_diagnostics"] for r in records])
        diagnostics[str(alpha)] = {"samples": len(records), "cuts_per_sample": NY,
            "max_charge_error": float(d[:,0].max()), "max_hermiticity_error": float(d[:,1].max()),
            "clipped_modes_min_max": [int(d[:,2].min()), int(d[:,2].max())],
            "minimum_window_charge": float(d[:,3].min()),
            "max_frame_gram_error": max(float(r["frame_gram_error"]) for r in records),
            "final_dy_window_mean": arrays[f"alpha{alpha}_dy_window_mean"][:,-1].tolist(),
            "final_dy_window_sem": arrays[f"alpha{alpha}_dy_window_sem"][:,-1].tolist(),
            "final_modular_time": 32 if alpha==1 else 1.5}
    atomic_npz(output/"averaged_observables.npz", **arrays, times_alpha1=TIMES, times_alpha3=SNAPSHOTS,
               snapshot_times=SNAPSHOTS, contract_json=json.dumps(CONTRACT,sort_keys=True))
    with (output/"alpha1_1_displacement.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["t_mod","wall_x","mean_dy","trajectory_sem","mean_window_charge"])
        for i,t in enumerate(TIMES):
            for p,x in enumerate(WALLS):
                writer.writerow([t,x,arrays["alpha1_dy_window_mean"][p,i],arrays["alpha1_dy_window_sem"][p,i],arrays["alpha1_window_charge_mean"][p,i]])
    plot_products(output, arrays, sample_limit)
    summary={"contract":CONTRACT,"samples_analyzed_per_alpha":sample_limit,"sources":sources,
             "diagnostics":diagnostics,"script_sha256":digest(__file__),
             "interpretation":"Modular dynamics, not circuit time. No Hamiltonian or covariance ensemble pre-averaging. No orientation sign applied."}
    (output/"analysis_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(diagnostics,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--cpus",default=None,help="Comma-separated CPU IDs; applied before workers start")
    parser.add_argument("--sample-limit",type=int,default=100)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    if not 1<=args.sample_limit<=100 or args.workers<1: parser.error("Invalid sample/worker count")
    if args.sample_limit!=100 and args.output==OUTPUT: parser.error("Use a distinct output for smoke tests")
    if args.cpus: os.sched_setaffinity(0,{int(x) for x in args.cpus.split(",")})
    args.output.mkdir(parents=True,exist_ok=True)
    print(json.dumps({"contract":CONTRACT,"output":str(args.output),"workers":args.workers,
                      "cpus":sorted(os.sched_getaffinity(0))}),flush=True)
    jobs,sources=collect_jobs(args.output,args.sample_limit)
    print(f"Verified {len(sources)} source shards; analyzing {len(jobs)} trajectories / {len(jobs)*NY} cuts",flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(analyze_sample,job) for job in jobs]
        for future in tqdm(as_completed(futures),total=len(futures),desc="Modular packets",unit="trajectory"):
            alpha,sample,status=future.result()
            print(f"[complete] alpha1={alpha} sample={sample:03d} {status}",flush=True)
    aggregate(args.output,jobs,sources,args.sample_limit)
    print(f"[done] {args.output}",flush=True)


if __name__=="__main__":
    main()
