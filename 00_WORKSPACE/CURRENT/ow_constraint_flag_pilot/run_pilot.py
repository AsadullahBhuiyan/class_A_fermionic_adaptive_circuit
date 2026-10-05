#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT.parents[2]
SRC = REPO_ROOT / "src"
for path in (str(SRC), str(EXPERIMENT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import CHANNELS, CHANNEL_TARGETS, TrajectoryActivityRecorder
from flag import analyze_constraint_flag, cell_order, checkpoint_map

CANONICAL_ENTRY = "classA_U1FGTN.run_markov_circuit"
WALLS = (3, 13)
FIGURE_WIDTH = 3.375


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    os.replace(temporary, path)


def write_frame(frame: pd.DataFrame, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = stem.with_suffix(".csv.tmp")
    parquet_tmp = stem.with_suffix(".parquet.tmp")
    frame.to_csv(csv_tmp, index=False)
    frame.to_parquet(parquet_tmp, index=False)
    os.replace(csv_tmp, stem.with_suffix(".csv"))
    os.replace(parquet_tmp, stem.with_suffix(".parquet"))


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def git_metadata() -> dict[str, Any]:
    def call(*args: str) -> str:
        result = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip()

    return {"commit": call("git", "rev-parse", "HEAD"), "dirty": bool(call("git", "status", "--short"))}


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    smoke = bool(args.smoke or args.command == "smoke")
    return {
        "experiment": "ow_constraint_flag_pilot",
        "nx": 4 if smoke else 16,
        "ny": 6 if smoke else 20,
        "nshell_dynamic": 1,
        "nshell_static": [1] if smoke else [1, 2, None],
        "geometries": ["dw", "uniform"],
        "schedules": ["random", "raster_y"],
        "samples_per_configuration": 2 if smoke else 10,
        "cycles": 4 if smoke else 80,
        "burn_in": 2 if smoke else 40,
        "root_seed": int(args.root_seed),
        "seed_convention": "matched-seed IDs; records are not identical across geometries",
        "perfect_correction": True,
        "init_mode": "maxmix",
        "alpha_1": 1,
        "alpha_2": 30,
        "trial_orbitals": "X",
        "canonical_dynamics_entry_point": CANONICAL_ENTRY,
        "workers": int(args.workers),
        "cpu_list": args.cpu_list,
        "blas_threads_per_worker": 1,
        "bootstrap_samples": int(args.bootstrap_samples),
        "static_center_set": "x=3..13,y=0..19" if not smoke else "central smoke slab",
        "wall_x": list(WALLS if not smoke else (1, 3)),
    }


def ensure_manifest(root: Path, config: dict[str, Any], *, resume: bool) -> None:
    path = root / "manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["config"] != config:
            raise RuntimeError("Refusing to reuse an output directory with a different immutable config")
        return
    root.mkdir(parents=True, exist_ok=True)
    preflight_path = os.environ.get("OW_FLAG_PREFLIGHT_JSON")
    preflight = json.loads(Path(preflight_path).read_text()) if preflight_path and Path(preflight_path).is_file() else None
    write_json(path, {"created_utc": utc_now(), "config": config, "git": git_metadata(), "cpu_preflight": preflight})


def stage(root: Path, name: str, function: Any, *, resume: bool) -> None:
    final = root / "stages" / name
    partial = root / "stages" / f".{name}.partial"
    status_path = root / "stage_status.json"
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    if resume and (final / "SUCCESS").is_file():
        print(f"[resume] {name}: already complete", flush=True)
        return
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=True)
    statuses[name] = {"state": "running", "started_utc": utc_now()}
    write_json(status_path, statuses)
    print(f"[stage] {name}", flush=True)
    try:
        function(partial)
        (partial / "SUCCESS").write_text(utc_now() + "\n")
        if final.exists():
            shutil.rmtree(final)
        os.replace(partial, final)
        statuses[name] = {"state": "success", "finished_utc": utc_now()}
        write_json(status_path, statuses)
    except Exception as exc:
        (partial / "FAILED").write_text(traceback.format_exc())
        statuses[name] = {"state": "failed", "finished_utc": utc_now(), "error": repr(exc)}
        write_json(status_path, statuses)
        (root / "FAILED").write_text(f"stage={name}\n{traceback.format_exc()}")
        raise


def model_for(nx: int, ny: int, geometry: str, nshell: int | None) -> classA_U1FGTN:
    dw = geometry == "dw"
    model = classA_U1FGTN(
        nx,
        ny,
        DW=dw,
        nshell=nshell,
        alpha_1=1,
        alpha_2=30,
        trial_orbitals="X",
        dw_truncation=dw,
    )
    model.construct_OW_projectors(nshell=nshell, DW=dw, trial_orbitals="X", dw_truncation=dw)
    return model


def static_constraints(config: dict[str, Any], geometry: str, nshell: int | None):
    nx, ny = config["nx"], config["ny"]
    model = model_for(nx, ny, geometry, nshell)
    if nx == 16:
        xs = range(3, 14)
        walls = WALLS
    else:
        walls = tuple(config["wall_x"])
        xs = range(walls[0], walls[1] + 1)
    active = np.asarray(
        model.active_top_layer_indices(meas_slab_only=(geometry == "dw")), dtype=np.int64
    )
    columns: list[np.ndarray] = []
    targets: list[int] = []
    channels: list[int] = []
    center_x: list[int] = []
    center_y: list[int] = []
    for x in xs:
        for y in range(ny):
            for channel_index, channel in enumerate(CHANNELS):
                columns.append(np.asarray(getattr(model, f"WF_{channel}")[:, x, y])[active])
                targets.append(CHANNEL_TARGETS[channel])
                channels.append(channel_index)
                center_x.append(x)
                center_y.append(y)
    vectors = np.column_stack(columns)
    if np.any(np.linalg.norm(vectors, axis=0) <= 1e-14):
        raise RuntimeError("A terminated OW constraint vanished on the active basis")
    labels = {
        "channel_index": np.asarray(channels),
        "channel": np.asarray(CHANNELS)[np.asarray(channels)],
        "center_x": np.asarray(center_x),
        "center_y": np.asarray(center_y),
        "wall_distance": np.minimum(np.abs(np.asarray(center_x) - walls[0]), np.abs(np.asarray(center_x) - walls[1])),
    }
    return vectors, np.asarray(targets, dtype=np.int8), labels, active.size // 2, walls


def dynamic_seed_ids(config: dict[str, Any]) -> list[int]:
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(config["root_seed"]).spawn(config["samples_per_configuration"])
    ]


def _case_name(geometry: str, schedule: str, seed_index: int) -> str:
    return f"{geometry}_{schedule}_seed{seed_index:02d}"


def _run_trajectory_task(task: dict[str, Any]) -> dict[str, Any]:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    output = Path(task["output"])
    if task["resume"] and (output / "SUCCESS").is_file():
        return {"case": output.name, "state": "skipped"}
    partial = output.with_name("." + output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    config = task["config"]
    model = model_for(config["nx"], config["ny"], task["geometry"], 1)
    recorder = TrajectoryActivityRecorder.from_model(model, cycles=config["cycles"], samples=1)
    result = model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=config["cycles"],
        samples=1,
        save=False,
        perfect_correction=True,
        init_mode="maxmix",
        sequence=task["schedule"],
        meas_slab_only=True,
        random_seed=int(task["seed"]),
        parallelize_samples=False,
        trajectory_weight_observer=recorder,
    )
    recorder.assert_complete()
    payload = recorder.payload()
    x = np.asarray(payload["site_x"], dtype=int)
    walls = tuple(config["wall_x"])
    distance = np.minimum(np.abs(x - walls[0]), np.abs(x - walls[1]))
    interface_width = 1 if config["nx"] < 5 else 2
    interface = distance < interface_width
    side = np.where(np.abs(x - walls[0]) <= np.abs(x - walls[1]), "left_wall", "right_wall")
    region = np.where(interface, side, "interior")
    save_npz(
        partial / "revisit_raw.npz",
        **payload,
        epsilon=1.0 - payload["success_probability"],
        wall_distance=distance,
        region=region,
        root_seed_id=np.asarray(task["seed"], dtype=np.uint64),
        returned_sample_seed=np.asarray(result["sample_seeds"], dtype=np.uint64),
    )
    write_json(
        partial / "case.json",
        {
            "geometry": task["geometry"],
            "schedule": task["schedule"],
            "seed_index": task["seed_index"],
            "root_seed_id": task["seed"],
            "returned_sample_seeds": result["sample_seeds"],
            "canonical_dynamics_entry_point": CANONICAL_ENTRY,
            "completed_utc": utc_now(),
        },
    )
    (partial / "SUCCESS").write_text(utc_now() + "\n")
    if output.exists():
        shutil.rmtree(output)
    os.replace(partial, output)
    return {"case": output.name, "state": "success"}


def run_dynamics(root: Path, config: dict[str, Any], args: argparse.Namespace) -> None:
    output = root / "cases"
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    for geometry in config["geometries"]:
        for schedule in config["schedules"]:
            for seed_index, seed in enumerate(dynamic_seed_ids(config)):
                tasks.append(
                    {
                        "config": config,
                        "geometry": geometry,
                        "schedule": schedule,
                        "seed_index": seed_index,
                        "seed": seed,
                        "output": str(output / _case_name(geometry, schedule, seed_index)),
                        "resume": bool(args.resume),
                    }
                )
    if args.workers == 1:
        reports = [_run_trajectory_task(task) for task in tasks]
    else:
        reports = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_run_trajectory_task, task) for task in tasks]
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                print(f"  {report['case']}: {report['state']}", flush=True)
    write_json(root / "case_report.json", reports)


def random_words(dynamic_stage: Path, geometry: str, count: int) -> list[list[tuple[int, int]]]:
    words = []
    for seed_index in range(count):
        path = dynamic_stage / "cases" / _case_name(geometry, "random", seed_index) / "revisit_raw.npz"
        with np.load(path) as data:
            order = np.argsort(data["visit_order"][0, 0])
            words.append([(int(data["site_x"][i]), int(data["site_y"][i])) for i in order])
    return words


def run_static(
    destination: Path,
    config: dict[str, Any],
    *,
    random_dynamic_stage: Path | None = None,
    random_only: bool = False,
) -> None:
    all_rows: list[dict[str, Any]] = []
    breakdowns: list[dict[str, Any]] = []
    for geometry in config["geometries"]:
        for nshell in config["nshell_static"]:
            vectors, targets, labels, target_rank, walls = static_constraints(config, geometry, nshell)
            modes = [] if random_only else ["interior_to_wall", "raster_y"]
            random_orders: list[list[tuple[int, int]]] = []
            if random_dynamic_stage is not None and nshell == 1:
                random_orders = random_words(random_dynamic_stage, geometry, config["samples_per_configuration"])
                modes.extend([f"random_seed{index:02d}" for index in range(len(random_orders))])
            for mode in modes:
                random_word = random_orders[int(mode[-2:])] if mode.startswith("random_seed") else None
                order_mode = "random" if random_word is not None else mode
                order = cell_order(
                    labels["center_x"],
                    labels["center_y"],
                    labels["channel_index"],
                    mode=order_mode,
                    wall_x=walls,
                    random_site_order=random_word,
                )
                checkpoints = checkpoint_map(order, labels["center_x"], ordering_mode=order_mode, wall_x=walls)
                scan = analyze_constraint_flag(
                    vectors,
                    targets,
                    labels=labels,
                    target_rank=target_rank,
                    ordering=order,
                    checkpoint_indices=checkpoints,
                )
                nsh = "untruncated" if nshell is None else str(nshell)
                identity = {"geometry": geometry, "nshell": nsh, "ordering": mode, "dimension": vectors.shape[0], "target_rank": target_rank}
                all_rows.extend([{**identity, **row} for row in scan.rows])
                breakdowns.append({**identity, **(scan.first_breakdown or {"prefix": -1, "reason": "none"})})
                print(f"  static {geometry} nshell={nsh} {mode}", flush=True)
    frame = pd.DataFrame(all_rows)
    write_frame(frame, destination / "static_prefix_metrics")
    write_frame(pd.DataFrame(breakdowns), destination / "first_breakdowns")


def _trajectory_rows(dynamic_stage: Path, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    burn = config["burn_in"]
    for case_path in sorted((dynamic_stage / "cases").glob("*")):
        meta = json.loads((case_path / "case.json").read_text())
        with np.load(case_path / "revisit_raw.npz") as data:
            eps = data["epsilon"][0]
            defect = data["defect_X"][0]
            valid = data["valid"][0]
            visit = data["visit_order"][0]
            site_x, site_y = data["site_x"], data["site_y"]
            distance, region = data["wall_distance"], data["region"].astype(str)
            nsites = site_x.size
            for cycle in range(burn, config["cycles"]):
                for site in range(nsites):
                    phase = "early" if visit[cycle, site] < nsites / 2 else "late"
                    broad_region = "interface" if region[site] != "interior" else "interior"
                    for channel_index, channel in enumerate(CHANNELS):
                        if not valid[cycle, site, channel_index]:
                            continue
                        rows.append(
                            {
                                "geometry": meta["geometry"],
                                "schedule": meta["schedule"],
                                "seed_index": meta["seed_index"],
                                "root_seed_id": str(meta["root_seed_id"]),
                                "cycle": cycle + 1,
                                "visit_order": int(visit[cycle, site]),
                                "word_phase": phase,
                                "channel": channel,
                                "x": int(site_x[site]),
                                "y": int(site_y[site]),
                                "wall_distance": int(distance[site]),
                                "side_region": region[site],
                                "region": broad_region,
                                "epsilon": float(eps[cycle, site, channel_index]),
                                "retention": float(1.0 - eps[cycle, site, channel_index]),
                                "defect_X": int(defect[cycle, site, channel_index]),
                            }
                        )
    return pd.DataFrame(rows)


def _bootstrap(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return np.nan, np.nan
    draws = values[rng.integers(0, values.size, size=(samples, values.size))].mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def contrasts(trajectory_means: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for schedule in sorted(trajectory_means.schedule.unique()):
        subset = trajectory_means[trajectory_means.schedule == schedule]
        pivot = subset.pivot_table(index=["geometry", "seed_index"], columns="region", values="epsilon")
        for geometry in ("dw", "uniform"):
            for region in ("interface", "left_wall", "right_wall"):
                values = (pivot.loc[geometry, region] - pivot.loc[geometry, "interior"]).to_numpy()
                low, high = _bootstrap(values, rng, bootstrap_samples)
                records.append({"contrast": f"{region}_minus_interior", "geometry": geometry, "schedule": schedule, "estimate": values.mean(), "ci_low": low, "ci_high": high, "bootstrap": "paired_within_trajectory"})
        for region in ("interface", "interior", "left_wall", "right_wall"):
            dw = subset[(subset.geometry == "dw") & (subset.region == region)].set_index("seed_index").epsilon
            uniform = subset[(subset.geometry == "uniform") & (subset.region == region)].set_index("seed_index").epsilon
            common = sorted(set(dw.index) & set(uniform.index))
            paired = np.asarray([dw[index] - uniform[index] for index in common])
            low, high = _bootstrap(paired, rng, bootstrap_samples)
            records.append({"contrast": "wall_minus_uniform", "geometry": region, "schedule": schedule, "estimate": paired.mean(), "ci_low": low, "ci_high": high, "bootstrap": "matched_seed"})
            draws = np.empty(bootstrap_samples)
            dvals, uvals = dw.to_numpy(), uniform.to_numpy()
            for index in range(bootstrap_samples):
                draws[index] = rng.choice(dvals, dvals.size).mean() - rng.choice(uvals, uvals.size).mean()
            records.append({"contrast": "wall_minus_uniform", "geometry": region, "schedule": schedule, "estimate": dvals.mean() - uvals.mean(), "ci_low": np.quantile(draws, .025), "ci_high": np.quantile(draws, .975), "bootstrap": "unpaired_trajectory"})
    return pd.DataFrame(records)


def configure_plotting() -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"], "font.size": 8, "mathtext.fontset": "cm", "figure.dpi": 150, "savefig.dpi": 300})


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def analyze(root: Path, destination: Path, config: dict[str, Any]) -> None:
    configure_plotting()
    dynamic = root / "stages" / "dynamics"
    rows = _trajectory_rows(dynamic, config)
    # Raw observations remain losslessly in per-case NPZ; these are consolidated analysis tables.
    summary = rows.groupby(["geometry", "schedule", "seed_index", "channel", "x", "wall_distance", "side_region", "region", "word_phase"], as_index=False).agg(epsilon=("epsilon", "mean"), retention=("retention", "mean"), defect_rate=("defect_X", "mean"), attempts=("defect_X", "size"))
    trajectory_broad = rows.groupby(["geometry", "schedule", "seed_index", "region"], as_index=False).agg(epsilon=("epsilon", "mean"), defect_rate=("defect_X", "mean"), attempts=("defect_X", "size"))
    trajectory_sides = rows[rows.side_region != "interior"].groupby(["geometry", "schedule", "seed_index", "side_region"], as_index=False).agg(epsilon=("epsilon", "mean"), defect_rate=("defect_X", "mean"), attempts=("defect_X", "size")).rename(columns={"side_region": "region"})
    trajectory = pd.concat((trajectory_broad, trajectory_sides), ignore_index=True)
    contrast = contrasts(trajectory, config["bootstrap_samples"], config["root_seed"] + 991)
    calibration = rows.groupby(["geometry", "schedule", "channel"], as_index=False).agg(mean_X=("defect_X", "mean"), mean_epsilon=("epsilon", "mean"), attempts=("defect_X", "size"))
    calibration["difference"] = calibration.mean_X - calibration.mean_epsilon
    write_frame(summary, destination / "revisit_summary")
    write_frame(trajectory, destination / "trajectory_region_means")
    write_frame(contrast, destination / "bootstrap_contrasts")
    write_frame(calibration, destination / "defect_calibration")

    static_parts = []
    for stage_name in ("static_deterministic", "static_random"):
        path = root / "stages" / stage_name / "static_prefix_metrics.parquet"
        if path.exists():
            static_parts.append(pd.read_parquet(path))
    static = pd.concat(static_parts, ignore_index=True)
    static_checkpoints = static[static.is_checkpoint].copy()
    write_frame(static_checkpoints, destination / "static_checkpoint_summary")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.4))
    primary = static_checkpoints[(static_checkpoints.nshell == "1") & (static_checkpoints.ordering == "interior_to_wall")]
    for geometry, group in primary.groupby("geometry"):
        ax.plot(group.constraint_fraction, group.sigma_max, marker="o", ms=2.5, label=geometry)
    ax.set(xlabel="constraint fraction", ylabel=r"$\sigma_{\max}(Q_F^\dagger Q_E)$", title="Flag spectral flow")
    ax.legend(frameon=False)
    save_figure(fig, destination / "figures", "flag_spectral_flow")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.4))
    shells = primary[primary.checkpoint_reason.str.contains("shell")]
    for geometry, group in shells.groupby("geometry"):
        ax.plot(group.wall_distance, group.delta_f_star, marker="o", label=geometry)
    ax.set(xlabel="wall distance of entering letter", ylabel=r"$\Delta_m$", title="Marginal shell frustration")
    ax.legend(frameon=False)
    save_figure(fig, destination / "figures", "marginal_shell_frustration")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.4))
    profile = summary.groupby(["geometry", "schedule", "wall_distance"], as_index=False).retention.mean()
    for (geometry, schedule), group in profile.groupby(["geometry", "schedule"]):
        ax.plot(group.wall_distance, group.retention, marker="o", label=f"{geometry}, {schedule}")
    ax.set(xlabel="distance to (pseudo-)wall", ylabel=r"$1-\epsilon$", title="Constraint retention")
    ax.legend(frameon=False, fontsize=6)
    save_figure(fig, destination / "figures", "retention_profiles")

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 2.4))
    shown = contrast[(contrast.contrast == "wall_minus_uniform") & (contrast.bootstrap == "matched_seed") & (contrast.geometry == "interface")]
    xloc = np.arange(len(shown))
    ax.errorbar(xloc, shown.estimate, yerr=[shown.estimate - shown.ci_low, shown.ci_high - shown.estimate], fmt="o")
    ax.axhline(0, color="k", lw=.7)
    ax.set_xticks(xloc, shown.schedule)
    ax.set(ylabel=r"$\epsilon_{\rm wall}-\epsilon_{\rm uniform}$", title="Matched-seed wall excess")
    save_figure(fig, destination / "figures", "wall_excesses")

    dynamic_gate_rows = contrast[(contrast.contrast.isin(["left_wall_minus_interior", "right_wall_minus_interior"])) & (contrast.geometry == "dw")]
    dynamic_gate = bool(len(dynamic_gate_rows) == 4 and np.all(dynamic_gate_rows.ci_low > 0))
    wall_uniform_rows = contrast[(contrast.contrast == "wall_minus_uniform") & (contrast.geometry.isin(["left_wall", "right_wall"])) & (contrast.bootstrap.isin(["matched_seed", "unpaired_trajectory"]))]
    wall_uniform_gate = bool(len(wall_uniform_rows) == 8 and np.all(wall_uniform_rows.ci_low > 0))
    uniform_rows = contrast[(contrast.contrast.isin(["left_wall_minus_interior", "right_wall_minus_interior"])) & (contrast.geometry == "uniform")]
    pseudo_gate = bool(len(uniform_rows) == 4 and np.all(uniform_rows.ci_low <= 0))
    # Static gate requires separately resolved left and right boundary-column excesses.
    boundary = primary[(primary.wall_distance == 0) & primary.checkpoint_reason.str.contains("column")]
    side_deltas: dict[str, dict[str, float]] = {}
    static_checks = []
    for side, wall_x in zip(("left", "right"), config["wall_x"]):
        dw_delta = float(boundary[(boundary.geometry == "dw") & (boundary.center_x == wall_x)].delta_f_star.mean())
        uniform_delta_side = float(boundary[(boundary.geometry == "uniform") & (boundary.center_x == wall_x)].delta_f_star.mean())
        side_deltas[side] = {"wall": dw_delta, "uniform": uniform_delta_side}
        static_checks.append(np.isfinite(dw_delta) and dw_delta > 0 and dw_delta > uniform_delta_side)
    static_gate = bool(all(static_checks))
    wall_delta = float(np.mean([value["wall"] for value in side_deltas.values()]))
    uniform_delta = float(np.mean([value["uniform"] for value in side_deltas.values()]))
    gates = {
        "static_boundary_excess": static_gate,
        "dynamic_both_schedules": dynamic_gate,
        "wall_exceeds_uniform_both_bootstraps": wall_uniform_gate,
        "uniform_pseudo_interface_no_comparable_peak": pseudo_gate,
    }
    verdict = "PROMOTE_TO_GPU_COLAB" if all(gates.values()) else "NULL_OR_SCHEDULE_DEPENDENT_NO_GPU"
    scalar = {
        "verdict": verdict,
        "gates": gates,
        "wall_boundary_delta_f_star": wall_delta,
        "uniform_boundary_delta_f_star": uniform_delta,
        "boundary_delta_by_side": side_deltas,
        "max_abs_calibration_error": float(calibration.difference.abs().max()),
        "statistical_unit": "trajectory (seed ID)",
        "interpretation_guardrails": [
            "overlap winding is not identified with a compact mode Chern number",
            "static incompatibility is not identified with persistent dynamical activity",
        ],
    }
    write_json(destination / "gate_verdict.json", scalar)
    generate_results_fragment(destination, config, scalar, contrast)


def generate_results_fragment(destination: Path, config: dict[str, Any], scalar: dict[str, Any], contrast: pd.DataFrame) -> None:
    # Smoke data must never masquerade as the preregistered 16x20 result.
    generated = (
        EXPERIMENT / "docs" / "generated"
        if config["nx"] == 16 and config["ny"] == 20
        else destination / "generated_note_preview"
    )
    generated.mkdir(parents=True, exist_ok=True)
    for stem in ("flag_spectral_flow", "marginal_shell_frustration", "retention_profiles", "wall_excesses"):
        shutil.copy2(destination / "figures" / f"{stem}.pdf", generated / f"{stem}.pdf")
    rows = contrast[(contrast.contrast == "interface_minus_interior") & (contrast.geometry == "dw")]
    table_rows = "\n".join(
        f"{row.schedule.replace('_', r'\_')} & {row.estimate:.4g} & [{row.ci_low:.4g},{row.ci_high:.4g}] \\\\"
        for row in rows.itertuples()
    )
    content = rf"""\providecommand{{\PilotStatus}}{{completed}}
\providecommand{{\PilotVerdict}}{{\texttt{{{scalar['verdict'].replace('_', r'\_')}}}}}
\providecommand{{\PilotParameters}}{{${config['nx']}\times {config['ny']}$, {config['samples_per_configuration']} matched seed IDs, {config['cycles']} cycles, burn-in {config['burn_in']}}}
\begin{{table}}[ht]
\centering
\caption{{Post-burn-in wall interface-minus-interior revisit loss. Confidence intervals resample trajectories.}}
\begin{{tabular}}{{lcc}}\toprule
schedule & estimate & 95\% CI \\\midrule
{table_rows}
\bottomrule\end{{tabular}}
\end{{table}}
\begin{{figure}}[ht]\centering
\includegraphics[width=0.72\linewidth]{{generated/flag_spectral_flow.pdf}}
\caption{{Static filled/empty flag spectral flow. This overlap diagnostic is not a Chern-number estimator.}}
\end{{figure}}
\begin{{figure}}[ht]\centering
\includegraphics[width=0.72\linewidth]{{generated/retention_profiles.pdf}}
\caption{{Post-burn-in reset-retention profiles. Static incompatibility and stationary activity are tested separately.}}
\end{{figure}}
"""
    temporary = generated / "results_fragment.tex.tmp"
    temporary.write_text(content)
    os.replace(temporary, generated / "results_fragment.tex")


def compile_note(destination: Path, config: dict[str, Any]) -> None:
    docs = EXPERIMENT / "docs"
    source = docs / "constraint_flag_pilot.tex"
    build = destination / "build"
    build.mkdir(parents=True, exist_ok=True)
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex is required by the validation plan")
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(build), str(source)]
    for _ in range(2):
        subprocess.run(command, cwd=docs, check=True, timeout=180)
    if config["nx"] == 16 and config["ny"] == 20:
        shutil.copy2(build / "constraint_flag_pilot.pdf", docs / "constraint_flag_pilot.pdf")


def run_pipeline(args: argparse.Namespace) -> Path:
    if args.output_root:
        root = Path(args.output_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = EXPERIMENT / "outputs" / f"ow_flag_{stamp}"
    config = config_from_args(args)
    ensure_manifest(root, config, resume=args.resume)

    command = "all" if args.command == "smoke" else args.command
    if command in ("static", "all"):
        stage(root, "static_deterministic", lambda path: run_static(path, config), resume=args.resume)
    if command in ("dynamics", "all"):
        stage(root, "dynamics", lambda path: run_dynamics(path, config, args), resume=args.resume)
    if command == "all":
        stage(root, "static_random", lambda path: run_static(path, {**config, "nshell_static": [1]}, random_dynamic_stage=root / "stages" / "dynamics", random_only=True), resume=args.resume)
    if command in ("analyze", "all"):
        stage(root, "analysis", lambda path: analyze(root, path, config), resume=args.resume)
    if command in ("docs", "all"):
        stage(root, "docs", lambda path: compile_note(path, config), resume=args.resume)
    if command == "all":
        (root / "SUCCESS").write_text(utc_now() + "\n")
        failed = root / "FAILED"
        if failed.exists():
            failed.unlink()
    print(json.dumps({"output_root": str(root), "command": args.command}, indent=2), flush=True)
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("smoke", "static", "dynamics", "analyze", "docs", "all"), nargs="?", default="all")
    parser.add_argument("--smoke", action="store_true", help="use 4x6, two seeds, and four cycles")
    parser.add_argument("--resume", action="store_true", help="skip atomically completed stages and cases")
    parser.add_argument("--cpu-list", default="", help="recorded CPU affinity list; affinity is normally set by taskset")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root")
    parser.add_argument("--root-seed", type=int, default=20260817)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    run_pipeline(args)


if __name__ == "__main__":
    main()
