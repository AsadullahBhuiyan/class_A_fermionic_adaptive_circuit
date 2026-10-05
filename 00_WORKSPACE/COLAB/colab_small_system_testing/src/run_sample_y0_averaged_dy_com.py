from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from dynamic_modular_charge_spreading import (
    DEFAULT_EPS,
    eigensystem_from_h_mod,
    log,
    modular_hamiltonian_from_restricted_covariance,
    progress_iter,
    reduced_site_index,
    restrict_covariance,
    save_npz_atomic,
    selected_indices,
    source_run_records,
    write_json_atomic,
)


HELPER_VERSION = "sample_y0_averaged_dy_com_cpu_v2"
DEFAULT_OUTPUT_CAMPAIGN = "sample_y0_averaged_dy_com"
DEFAULT_CYCLES = [50]
DEFAULT_DT = 0.01
DEFAULT_TIME_CHUNK = 32
DEFAULT_X_WINDOW_RADIUS = 2
DEFAULT_Q_MIN_TOL = 1e-10
DEFAULT_CHARGE_DRIFT_TOL = 1e-6
WALL_ORIENTATION = {5: 1.0, 11: -1.0}


def default_output_root(bundle_root: Path) -> Path:
    return (
        Path(bundle_root)
        / "analysis_outputs"
        / "dynamic_modular_charge_spreading"
        / DEFAULT_OUTPUT_CAMPAIGN
    )


def product_dir(output_root: Path, record: dict[str, Any]) -> Path:
    return Path(output_root) / "runs" / str(record["source_key"])


def product_path(output_root: Path, record: dict[str, Any]) -> Path:
    return product_dir(output_root, record) / "sample_y0_averaged_dy_com.npz"


def time_grid(nx: int, *, dt: float = DEFAULT_DT, t_final: float | None = None) -> np.ndarray:
    if t_final is None:
        t_final = 2.0 * int(nx)
    n_steps = int(round(float(t_final) / float(dt)))
    if not np.isclose(n_steps * float(dt), float(t_final)):
        raise ValueError(f"dt={dt} does not divide t_final={t_final}.")
    return float(dt) * np.arange(n_steps + 1, dtype=np.float64)


def wall_orientation_eta(x_dw: int) -> float:
    x_dw = int(x_dw)
    if x_dw not in WALL_ORIENTATION:
        raise ValueError(f"No wall orientation convention for x_dw={x_dw}.")
    return float(WALL_ORIENTATION[x_dw])


def packet_specs(ny: int) -> list[dict[str, Any]]:
    ny_sub = int(ny) // 2
    specs = [
        {"label": "x5_y0", "x_dw": 5, "y_rel": 0},
        {"label": f"x5_y{ny_sub - 1}", "x_dw": 5, "y_rel": ny_sub - 1},
        {"label": "x11_y0", "x_dw": 11, "y_rel": 0},
        {"label": f"x11_y{ny_sub - 1}", "x_dw": 11, "y_rel": ny_sub - 1},
    ]
    for spec in specs:
        spec["wall_eta"] = wall_orientation_eta(int(spec["x_dw"]))
    return specs


def packet_occupied_indices(nx: int, ny: int, packets: list[dict[str, Any]]) -> np.ndarray:
    nx = int(nx)
    ny = int(ny)
    ny_sub = ny // 2
    occ: list[int] = []
    for packet in packets:
        x_dw = int(packet["x_dw"]) % nx
        y_rel = int(packet["y_rel"])
        if not (0 <= y_rel < ny_sub):
            raise ValueError(f"Packet {packet} has y_rel outside reduced half window.")
        for orbital in (0, 1):
            occ.append(reduced_site_index(nx, x_dw, y_rel, orbital))
    occ_arr = np.asarray(occ, dtype=np.int64)
    if len(np.unique(occ_arr)) != len(occ_arr):
        raise ValueError("Packet occupancy indices must be unique.")
    if len(occ_arr) != 2 * len(packets):
        raise ValueError("Each packet must occupy exactly two orbitals.")
    return occ_arr


def periodic_x_window(nx: int, x_center: int, radius: int) -> np.ndarray:
    x = np.arange(int(nx), dtype=np.float64)
    raw = np.abs(x - (int(x_center) % int(nx)))
    dist = np.minimum(raw, int(nx) - raw)
    return dist <= int(radius)


def evolve_packet_dy_com(
    *,
    h_vals: np.ndarray,
    h_vecs: np.ndarray,
    nx: int,
    ny: int,
    packets: list[dict[str, Any]],
    times: np.ndarray,
    x_window_radius: int = DEFAULT_X_WINDOW_RADIUS,
    q_min_tol: float = DEFAULT_Q_MIN_TOL,
    time_chunk: int = DEFAULT_TIME_CHUNK,
) -> dict[str, np.ndarray | float]:
    nx = int(nx)
    ny = int(ny)
    ny_sub = ny // 2
    packet_count = len(packets)
    times = np.asarray(times, dtype=np.float64)
    h_vals = np.asarray(h_vals, dtype=np.float64)
    h_vecs = np.asarray(h_vecs, dtype=np.complex128)

    occ = packet_occupied_indices(nx, ny, packets)
    if not np.isclose(float(len(occ)), 2.0 * packet_count):
        raise ValueError("Single-packet initial charge construction failed.")
    occ_coeff = h_vecs[occ, :].conj().T
    y_vals = np.arange(ny_sub, dtype=np.float64)
    x_masks = [periodic_x_window(nx, int(packet["x_dw"]), x_window_radius) for packet in packets]

    y_com = np.empty((packet_count, len(times)), dtype=np.float64)
    q_window = np.empty_like(y_com)
    charge = np.empty_like(y_com)

    for start in range(0, len(times), int(time_chunk)):
        stop = min(len(times), start + int(time_chunk))
        phase = np.exp(-1j * times[start:stop, None] * h_vals[None, :])
        amplitudes = np.einsum("ik,tk,ka->tia", h_vecs, phase, occ_coeff, optimize=True)
        density_by_occ = np.abs(amplitudes) ** 2
        for p_idx in range(packet_count):
            density = density_by_occ[:, :, 2 * p_idx : 2 * p_idx + 2].sum(axis=2).real
            cell_density = density.reshape(stop - start, ny_sub, nx, 2).sum(axis=-1)
            charge[p_idx, start:stop] = cell_density.sum(axis=(1, 2), dtype=np.float64)
            window = cell_density[:, :, x_masks[p_idx]]
            q = window.sum(axis=(1, 2), dtype=np.float64)
            if np.any(q <= float(q_min_tol)):
                raise FloatingPointError(
                    f"Window charge below tolerance for packet {packets[p_idx]['label']}: min={float(np.min(q))}"
                )
            q_window[p_idx, start:stop] = q
            y_com[p_idx, start:stop] = (window.sum(axis=2) * y_vals[None, :]).sum(axis=1) / q

    dy_com = y_com - y_com[:, [0]]
    charge_drift = np.max(np.abs(charge - charge[:, [0]]), axis=1)
    if not np.all(np.isfinite(dy_com)) or not np.all(np.isfinite(q_window)):
        raise FloatingPointError("Non-finite dy_com or Q_window values.")
    return {
        "dy_com": dy_com,
        "Q_window": q_window,
        "min_Q_window": np.min(q_window, axis=1),
        "charge_drift": charge_drift,
    }


def selected_cycle_indices(record: dict[str, Any], cycles: list[int]) -> list[int]:
    available = [int(c) for c in record["snapshot_cycles"]]
    selected: list[int] = []
    for cycle in cycles:
        if int(cycle) not in available:
            raise ValueError(f"{record['source_key']}: requested cycle {cycle}, available {available}.")
        selected.append(available.index(int(cycle)))
    return selected


def wall_eta_from_packet_x(packet_x_dw: np.ndarray) -> np.ndarray:
    return np.asarray([wall_orientation_eta(int(x_dw)) for x_dw in np.asarray(packet_x_dw)], dtype=np.float64)


def compute_product(
    *,
    record: dict[str, Any],
    output_root: Path,
    cycles: list[int],
    eps: float = DEFAULT_EPS,
    dt: float = DEFAULT_DT,
    t_final: float | None = None,
    time_chunk: int = DEFAULT_TIME_CHUNK,
    x_window_radius: int = DEFAULT_X_WINDOW_RADIUS,
    q_min_tol: float = DEFAULT_Q_MIN_TOL,
    charge_drift_tol: float = DEFAULT_CHARGE_DRIFT_TOL,
    max_samples: int | None = None,
    max_y0: int | None = None,
    overwrite: bool = False,
    progress: bool = True,
    make_plots: bool = True,
) -> Path:
    out_path = product_path(output_root, record)
    if out_path.exists() and not overwrite:
        log(f"Reusing existing averaged dy product after validation: {out_path}", enabled=progress)
        validate_product(out_path)
        if make_plots:
            plot_product(out_path)
        return out_path

    nx = int(record["Nx"])
    ny = int(record["Ny"])
    ny_sub = ny // 2
    dim = nx * ny_sub * 2
    packets = packet_specs(ny)
    packet_labels = np.asarray([p["label"] for p in packets])
    packet_x_dw = np.asarray([p["x_dw"] for p in packets], dtype=np.int64)
    packet_y_rel = np.asarray([p["y_rel"] for p in packets], dtype=np.int64)
    wall_eta = np.asarray([p["wall_eta"] for p in packets], dtype=np.float64)
    times = time_grid(nx, dt=dt, t_final=t_final)

    shard = np.load(record["shard_path"], mmap_mode="r")
    sample_count, time_count, nlayer, nlayer2 = shard.shape
    if nlayer != nlayer2 or nlayer != 2 * nx * ny:
        raise ValueError(f"Unexpected shard shape for {record['source_key']}: {shard.shape}.")
    cycle_indices = selected_cycle_indices(record, cycles)
    sample_indices = selected_indices(sample_count, max_samples)
    y0_values = selected_indices(ny, max_y0)
    if not cycle_indices or not sample_indices or not y0_values:
        raise ValueError("At least one cycle, sample, and y0 must be selected.")

    snapshot_cycles = np.asarray([int(record["snapshot_cycles"][idx]) for idx in cycle_indices], dtype=np.int64)
    C = len(cycle_indices)
    S = len(sample_indices)
    Y = len(y0_values)
    P = len(packets)
    T = len(times)
    log(
        (
            f"Starting averaged dy product {record['source_key']}: Nx={nx}, Ny={ny}, nshell={record['nshell']}, "
            f"dim={dim}, cycles={snapshot_cycles.tolist()}, samples={S}, y0={Y}, packets={P}, times={T}, "
            f"output={out_path}"
        ),
        enabled=progress,
    )

    dy_com_curves = np.empty((C, S, Y, P, T), dtype=np.float64)
    q_window_curves = np.empty_like(dy_com_curves)
    min_q_window = np.empty((C, S, Y, P), dtype=np.float64)
    charge_drift = np.empty_like(min_q_window)
    covariance_hermiticity_error = np.empty((C, S, Y), dtype=np.float64)
    h_mod_hermiticity_error = np.empty_like(covariance_hermiticity_error)
    h_reconstruction_error = np.empty_like(covariance_hermiticity_error)
    clip_low_count = np.empty((C, S, Y), dtype=np.int64)
    clip_high_count = np.empty_like(clip_low_count)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for out_c, cycle_idx in progress_iter(
        list(enumerate(cycle_indices)),
        enabled=progress,
        desc=f"{record['source_key']} cycles",
        unit="cycle",
        leave=True,
    ):
        cycle = int(record["snapshot_cycles"][cycle_idx])
        for out_s, sample_idx in progress_iter(
            list(enumerate(sample_indices)),
            enabled=progress,
            desc=f"{record['source_key']} cycle {cycle} samples",
            unit="sample",
            leave=False,
        ):
            G_sample = np.asarray(shard[int(sample_idx), int(cycle_idx)], dtype=np.complex128)
            for out_y, y0 in progress_iter(
                list(enumerate(y0_values)),
                enabled=progress,
                desc=f"{record['source_key']} cycle {cycle} sample {sample_idx} y0",
                unit="cut",
                leave=False,
            ):
                sub = restrict_covariance(G_sample, nx=nx, ny=ny, y0=int(y0))
                mod = modular_hamiltonian_from_restricted_covariance(sub, eps=eps)
                eig = eigensystem_from_h_mod(mod["h_mod"])
                evolved = evolve_packet_dy_com(
                    h_vals=eig["h_vals"],
                    h_vecs=eig["h_vecs"],
                    nx=nx,
                    ny=ny,
                    packets=packets,
                    times=times,
                    x_window_radius=x_window_radius,
                    q_min_tol=q_min_tol,
                    time_chunk=time_chunk,
                )
                drift = np.asarray(evolved["charge_drift"], dtype=np.float64)
                if float(np.max(drift)) > float(charge_drift_tol):
                    raise ValueError(
                        f"{record['source_key']} cycle={cycle} sample={sample_idx} y0={y0}: "
                        f"charge drift {float(np.max(drift)):.3e} > {charge_drift_tol:.3e}"
                )
                dy_com_curves[out_c, out_s, out_y] = np.asarray(evolved["dy_com"], dtype=np.float64)
                q_window_curves[out_c, out_s, out_y] = np.asarray(evolved["Q_window"], dtype=np.float64)
                min_q_window[out_c, out_s, out_y] = np.asarray(evolved["min_Q_window"], dtype=np.float64)
                charge_drift[out_c, out_s, out_y] = drift
                covariance_hermiticity_error[out_c, out_s, out_y] = float(mod["covariance_hermiticity_error"])
                h_mod_hermiticity_error[out_c, out_s, out_y] = float(eig["h_mod_hermiticity_error"])
                h_reconstruction_error[out_c, out_s, out_y] = float(eig["reconstruction_error"])
                clip_low_count[out_c, out_s, out_y] = int(mod["clip_low_count"])
                clip_high_count[out_c, out_s, out_y] = int(mod["clip_high_count"])

    flat_axis_count = S * Y
    dy_com_mean = dy_com_curves.mean(axis=(1, 2))
    dy_com_std = dy_com_curves.std(axis=(1, 2), ddof=1 if flat_axis_count > 1 else 0)
    dy_com_sem = dy_com_std / np.sqrt(float(flat_axis_count))
    final_dy_com_mean = dy_com_mean[:, :, -1]
    max_abs_dy_com_mean = np.max(np.abs(dy_com_mean), axis=2)
    signed_tangent_dy_mean = wall_eta[None, :, None] * dy_com_mean
    signed_tangent_dy_std = dy_com_std.copy()
    signed_tangent_dy_sem = dy_com_sem.copy()
    final_signed_tangent_dy_mean = signed_tangent_dy_mean[:, :, -1]
    max_abs_signed_tangent_dy_mean = np.max(np.abs(signed_tangent_dy_mean), axis=2)
    if not np.all(np.isfinite(dy_com_curves)) or not np.all(np.isfinite(dy_com_mean)):
        raise FloatingPointError("Non-finite dy_com output.")
    if (
        not np.all(np.isfinite(q_window_curves))
        or not np.all(np.isfinite(dy_com_sem))
        or not np.all(np.isfinite(signed_tangent_dy_mean))
    ):
        raise FloatingPointError("Non-finite Q_window, SEM, or signed tangent output.")

    metadata = {
        "helper_version": HELPER_VERSION,
        "source_campaign": "pure_state_covariance_snapshots",
        "source_key": record["source_key"],
        "source_shard_path": str(record["shard_path"]),
        "source_manifest_path": str(record["manifest_path"]),
        "source_summary_path": str(record["summary_path"]),
        "Nx": nx,
        "Ny": ny,
        "nshell": int(record["nshell"]),
        "protocol": record["protocol"],
        "basis_order": "y_rel, x, orbital",
        "subsystem": "[0,Nx) x [y0,y0+Ny//2)",
        "packet_y_convention": "relative_to_each_cut",
        "averaging": "mean_{sample_idx,y0} dy_com(sample_idx,y0,packet,t)",
        "signed_tangent_convention": "eta_wall * dy_com with eta_wall=+1 for x_dw=5 and -1 for x_dw=11",
        "modular_eps": float(eps),
        "dt": float(dt),
        "t_final": float(times[-1]),
        "time_chunk": int(time_chunk),
        "x_window_radius": int(x_window_radius),
        "q_min_tol": float(q_min_tol),
        "charge_drift_tol": float(charge_drift_tol),
        "cycle_indices": [int(i) for i in cycle_indices],
        "snapshot_cycles": [int(c) for c in snapshot_cycles],
        "sample_indices": [int(i) for i in sample_indices],
        "y0_values": [int(y0) for y0 in y0_values],
        "observation_count_per_cycle": int(flat_axis_count),
        "is_truncated_smoke": bool(max_samples is not None or max_y0 is not None or cycles != DEFAULT_CYCLES),
        "saves_full_C_t": False,
        "saves_full_N_xy_t": False,
        "saves_per_cut_h_mod": False,
    }

    save_npz_atomic(
        out_path,
        times=times,
        snapshot_cycles=snapshot_cycles,
        cycle_indices=np.asarray(cycle_indices, dtype=np.int64),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        y0_values=np.asarray(y0_values, dtype=np.int64),
        packet_labels=packet_labels,
        packet_x_dw=packet_x_dw,
        packet_y_rel=packet_y_rel,
        wall_eta=wall_eta,
        Nx=np.asarray(nx, dtype=np.int64),
        Ny=np.asarray(ny, dtype=np.int64),
        nshell=np.asarray(int(record["nshell"]), dtype=np.int64),
        dy_com_curves=dy_com_curves,
        Q_window_curves=q_window_curves,
        dy_com_mean=dy_com_mean,
        dy_com_std=dy_com_std,
        dy_com_sem=dy_com_sem,
        final_dy_com_mean=final_dy_com_mean,
        max_abs_dy_com_mean=max_abs_dy_com_mean,
        signed_tangent_dy_mean=signed_tangent_dy_mean,
        signed_tangent_dy_std=signed_tangent_dy_std,
        signed_tangent_dy_sem=signed_tangent_dy_sem,
        final_signed_tangent_dy_mean=final_signed_tangent_dy_mean,
        max_abs_signed_tangent_dy_mean=max_abs_signed_tangent_dy_mean,
        min_Q_window=min_q_window,
        charge_drift=charge_drift,
        covariance_hermiticity_error=covariance_hermiticity_error,
        h_mod_hermiticity_error=h_mod_hermiticity_error,
        h_reconstruction_error=h_reconstruction_error,
        clip_low_count=clip_low_count,
        clip_high_count=clip_high_count,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )
    validate_product(out_path, charge_drift_tol=charge_drift_tol)
    if make_plots:
        plot_product(out_path)
    log(f"Saved and validated averaged dy product: {out_path}", enabled=progress)
    return out_path


def validate_product(path: Path, *, charge_drift_tol: float = DEFAULT_CHARGE_DRIFT_TOL) -> None:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        C = len(data["snapshot_cycles"])
        S = len(data["sample_indices"])
        Y = len(data["y0_values"])
        P = len(data["packet_labels"])
        T = len(data["times"])
        expected_curve_shape = (C, S, Y, P, T)
        expected_packet_time_shape = (C, P, T)
        if data["dy_com_curves"].shape != expected_curve_shape:
            raise ValueError(f"{path}: dy_com_curves shape mismatch.")
        if data["Q_window_curves"].shape != expected_curve_shape:
            raise ValueError(f"{path}: Q_window_curves shape mismatch.")
        if "wall_eta" in data and data["wall_eta"].shape != (P,):
            raise ValueError(f"{path}: wall_eta shape mismatch.")
        for key in ("dy_com_mean", "dy_com_std", "dy_com_sem"):
            if data[key].shape != expected_packet_time_shape:
                raise ValueError(f"{path}: {key} shape mismatch.")
        for key in ("final_dy_com_mean", "max_abs_dy_com_mean"):
            if data[key].shape != (C, P):
                raise ValueError(f"{path}: {key} shape mismatch.")
        for key in ("signed_tangent_dy_mean", "signed_tangent_dy_std", "signed_tangent_dy_sem"):
            if key in data and data[key].shape != expected_packet_time_shape:
                raise ValueError(f"{path}: {key} shape mismatch.")
        for key in ("final_signed_tangent_dy_mean", "max_abs_signed_tangent_dy_mean"):
            if key in data and data[key].shape != (C, P):
                raise ValueError(f"{path}: {key} shape mismatch.")
        for key in ("min_Q_window", "charge_drift"):
            if data[key].shape != (C, S, Y, P):
                raise ValueError(f"{path}: {key} shape mismatch.")
        finite_keys = (
            "dy_com_curves",
            "Q_window_curves",
            "dy_com_mean",
            "dy_com_std",
            "dy_com_sem",
            "final_dy_com_mean",
            "max_abs_dy_com_mean",
            "min_Q_window",
            "charge_drift",
        )
        for key in finite_keys:
            if not np.all(np.isfinite(data[key])):
                raise FloatingPointError(f"{path}: non-finite values in {key}.")
        for key in (
            "wall_eta",
            "signed_tangent_dy_mean",
            "signed_tangent_dy_std",
            "signed_tangent_dy_sem",
            "final_signed_tangent_dy_mean",
            "max_abs_signed_tangent_dy_mean",
        ):
            if key in data and not np.all(np.isfinite(data[key])):
                raise FloatingPointError(f"{path}: non-finite values in {key}.")
        if float(np.max(data["charge_drift"])) > float(charge_drift_tol):
            raise ValueError(f"{path}: charge drift exceeds {charge_drift_tol}.")
        if int(metadata["observation_count_per_cycle"]) != S * Y:
            raise ValueError(f"{path}: observation count metadata mismatch.")
        if bool(metadata.get("saves_full_C_t")) or bool(metadata.get("saves_full_N_xy_t")):
            raise ValueError(f"{path}: metadata indicates forbidden full trajectory output.")


def summary_rows_for_product(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        cycles = np.asarray(data["snapshot_cycles"], dtype=np.int64)
        packet_labels = [str(label) for label in data["packet_labels"]]
        wall_eta = np.asarray(data["wall_eta"], dtype=np.float64) if "wall_eta" in data else wall_eta_from_packet_x(data["packet_x_dw"])
        signed_mean = (
            np.asarray(data["signed_tangent_dy_mean"], dtype=np.float64)
            if "signed_tangent_dy_mean" in data
            else wall_eta[None, :, None] * np.asarray(data["dy_com_mean"], dtype=np.float64)
        )
        signed_sem = (
            np.asarray(data["signed_tangent_dy_sem"], dtype=np.float64)
            if "signed_tangent_dy_sem" in data
            else np.asarray(data["dy_com_sem"], dtype=np.float64)
        )
        final_signed_mean = (
            np.asarray(data["final_signed_tangent_dy_mean"], dtype=np.float64)
            if "final_signed_tangent_dy_mean" in data
            else signed_mean[:, :, -1]
        )
        max_abs_signed_mean = (
            np.asarray(data["max_abs_signed_tangent_dy_mean"], dtype=np.float64)
            if "max_abs_signed_tangent_dy_mean" in data
            else np.max(np.abs(signed_mean), axis=2)
        )
        for c_idx, cycle in enumerate(cycles):
            for p_idx, label in enumerate(packet_labels):
                rows.append(
                    {
                        "source_key": metadata["source_key"],
                        "Nx": int(metadata["Nx"]),
                        "Ny": int(metadata["Ny"]),
                        "nshell": int(metadata["nshell"]),
                        "cycle": int(cycle),
                        "packet_label": label,
                        "packet_x_dw": int(data["packet_x_dw"][p_idx]),
                        "packet_y_rel": int(data["packet_y_rel"][p_idx]),
                        "wall_eta": float(wall_eta[p_idx]),
                        "sample_count": int(len(data["sample_indices"])),
                        "y0_count": int(len(data["y0_values"])),
                        "observation_count": int(metadata["observation_count_per_cycle"]),
                        "dt": float(metadata["dt"]),
                        "t_final": float(metadata["t_final"]),
                        "x_window_radius": int(metadata["x_window_radius"]),
                        "final_dy_com_mean": float(data["final_dy_com_mean"][c_idx, p_idx]),
                        "max_abs_dy_com_mean": float(data["max_abs_dy_com_mean"][c_idx, p_idx]),
                        "final_dy_com_sem": float(data["dy_com_sem"][c_idx, p_idx, -1]),
                        "final_signed_tangent_dy_mean": float(final_signed_mean[c_idx, p_idx]),
                        "max_abs_signed_tangent_dy_mean": float(max_abs_signed_mean[c_idx, p_idx]),
                        "final_signed_tangent_dy_sem": float(signed_sem[c_idx, p_idx, -1]),
                        "min_Q_window": float(np.min(data["min_Q_window"][c_idx, :, :, p_idx])),
                        "max_charge_drift": float(np.max(data["charge_drift"][c_idx, :, :, p_idx])),
                        "max_covariance_hermiticity_error": float(np.max(data["covariance_hermiticity_error"][c_idx])),
                        "max_h_mod_hermiticity_error": float(np.max(data["h_mod_hermiticity_error"][c_idx])),
                        "max_h_reconstruction_error": float(np.max(data["h_reconstruction_error"][c_idx])),
                        "clip_low_count_total": int(np.sum(data["clip_low_count"][c_idx])),
                        "clip_high_count_total": int(np.sum(data["clip_high_count"][c_idx])),
                        "product_path": str(path),
                    }
                )
    return rows


def write_summary(output_root: Path, product_paths: list[Path]) -> Path:
    rows: list[dict[str, Any]] = []
    for path in product_paths:
        rows.extend(summary_rows_for_product(path))
    summary_path = Path(output_root) / "sample_y0_averaged_dy_com_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(
            Path(output_root) / "sample_y0_averaged_dy_com_summary.parquet",
            index=False,
        )
    except Exception as exc:
        log(f"parquet skipped: {type(exc).__name__}: {exc}", enabled=True)
    return summary_path


def plot_product(path: Path) -> Path | None:
    try:
        mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/matplotlib"))
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"plot skipped for {path}: {type(exc).__name__}: {exc}", enabled=True)
        return None

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        source_key = str(metadata["source_key"])
        cycles = np.asarray(data["snapshot_cycles"], dtype=np.int64)
        times = np.asarray(data["times"], dtype=np.float64)
        packet_labels = [str(label) for label in data["packet_labels"]]
        dy_mean = np.asarray(data["dy_com_mean"], dtype=np.float64)
        dy_sem = np.asarray(data["dy_com_sem"], dtype=np.float64)
        wall_eta = np.asarray(data["wall_eta"], dtype=np.float64) if "wall_eta" in data else wall_eta_from_packet_x(data["packet_x_dw"])
        signed_mean = (
            np.asarray(data["signed_tangent_dy_mean"], dtype=np.float64)
            if "signed_tangent_dy_mean" in data
            else wall_eta[None, :, None] * dy_mean
        )
        signed_sem = (
            np.asarray(data["signed_tangent_dy_sem"], dtype=np.float64)
            if "signed_tangent_dy_sem" in data
            else dy_sem
        )
        fig, axes = plt.subplots(
            len(cycles),
            1,
            figsize=(7.0, 3.5 * len(cycles)),
            constrained_layout=True,
            squeeze=False,
        )
        for c_idx, cycle in enumerate(cycles):
            ax = axes[c_idx, 0]
            for p_idx, label in enumerate(packet_labels):
                line = ax.plot(times, dy_mean[c_idx, p_idx], lw=1.4, label=label)[0]
                ax.fill_between(
                    times,
                    dy_mean[c_idx, p_idx] - dy_sem[c_idx, p_idx],
                    dy_mean[c_idx, p_idx] + dy_sem[c_idx, p_idx],
                    color=line.get_color(),
                    alpha=0.18,
                    linewidth=0,
                )
            ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
            ax.set_title(f"{source_key} | cycle {int(cycle)}")
            ax.set_xlabel("modular time")
            ax.set_ylabel(r"$\mathrm{mean}_{sample,y0}\,\Delta\langle y_{rel}\rangle$")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"{source_key}: sample/y0-averaged dy trajectory")
        if len(cycles) == 1:
            plot_name = f"{source_key}_cycle{int(cycles[0])}_dy_mean_vs_time.png"
        else:
            cycle_tag = "_".join(str(int(c)) for c in cycles)
            plot_name = f"{source_key}_cycles{cycle_tag}_dy_mean_vs_time.png"
        plot_path = path.parent / plot_name
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(
            len(cycles),
            1,
            figsize=(7.0, 3.5 * len(cycles)),
            constrained_layout=True,
            squeeze=False,
        )
        for c_idx, cycle in enumerate(cycles):
            ax = axes[c_idx, 0]
            for p_idx, label in enumerate(packet_labels):
                curve_label = f"{label}, eta={wall_eta[p_idx]:+.0f}"
                line = ax.plot(times, signed_mean[c_idx, p_idx], lw=1.4, label=curve_label)[0]
                ax.fill_between(
                    times,
                    signed_mean[c_idx, p_idx] - signed_sem[c_idx, p_idx],
                    signed_mean[c_idx, p_idx] + signed_sem[c_idx, p_idx],
                    color=line.get_color(),
                    alpha=0.18,
                    linewidth=0,
                )
            ax.axhline(0.0, color="black", lw=0.8, alpha=0.5)
            ax.set_title(f"{source_key} | cycle {int(cycle)}")
            ax.set_xlabel("modular time")
            ax.set_ylabel(r"$\mathrm{mean}_{sample,y0}\,\eta_{wall}\Delta\langle y_{rel}\rangle$")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="best")
        fig.suptitle(f"{source_key}: sample/y0-averaged wall-oriented dy trajectory")
        if len(cycles) == 1:
            signed_plot_name = f"{source_key}_cycle{int(cycles[0])}_signed_tangent_dy_mean_vs_time.png"
        else:
            cycle_tag = "_".join(str(int(c)) for c in cycles)
            signed_plot_name = f"{source_key}_cycles{cycle_tag}_signed_tangent_dy_mean_vs_time.png"
        signed_plot_path = path.parent / signed_plot_name
        fig.savefig(signed_plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    return plot_path


def write_campaign_manifest(
    *,
    output_root: Path,
    bundle_root: Path,
    product_paths: list[Path],
    args: argparse.Namespace,
) -> Path:
    products = []
    for path in product_paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        products.append(
            {
                "source_key": metadata["source_key"],
                "Nx": int(metadata["Nx"]),
                "Ny": int(metadata["Ny"]),
                "nshell": int(metadata["nshell"]),
                "snapshot_cycles": [int(c) for c in metadata["snapshot_cycles"]],
                "sample_count": len(metadata["sample_indices"]),
                "y0_count": len(metadata["y0_values"]),
                "product_path": str(path),
            }
        )
    manifest = {
        "helper_version": HELPER_VERSION,
        "kind": "derived_sample_y0_averaged_dy_com",
        "bundle_root": str(bundle_root),
        "output_root": str(output_root),
        "default_cycles": DEFAULT_CYCLES,
        "requested_cycles": [int(c) for c in args.cycles],
        "dt": float(args.dt),
        "x_window_radius": int(args.x_window_radius),
        "source_campaign": "pure_state_covariance_snapshots",
        "products": products,
    }
    manifest_path = Path(output_root) / "campaign_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sample/y0-averaged dy_com(t) curves from saved covariance snapshots."
    )
    parser.add_argument("--bundle-root", type=Path, default=Path("colab_small_system_testing"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cycles", type=int, nargs="+", default=DEFAULT_CYCLES)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--t-final", type=float, default=None)
    parser.add_argument("--time-chunk", type=int, default=DEFAULT_TIME_CHUNK)
    parser.add_argument("--x-window-radius", type=int, default=DEFAULT_X_WINDOW_RADIUS)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--q-min-tol", type=float, default=DEFAULT_Q_MIN_TOL)
    parser.add_argument("--charge-drift-tol", type=float, default=DEFAULT_CHARGE_DRIFT_TOL)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-y0", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bundle_root = Path(args.bundle_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else default_output_root(bundle_root).resolve()
    progress = not bool(args.no_progress)
    records = source_run_records(bundle_root)
    if args.max_configs is not None:
        records = records[: max(0, int(args.max_configs))]
    if not records:
        raise ValueError("No source configs selected.")

    log(f"Bundle root: {bundle_root}", enabled=progress)
    log(f"Output root: {output_root}", enabled=progress)
    log(f"Requested cycles: {[int(c) for c in args.cycles]}", enabled=progress)

    product_paths: list[Path] = []
    for record in progress_iter(records, enabled=progress, desc="configs", unit="config", leave=True):
        product_paths.append(
            compute_product(
                record=record,
                output_root=output_root,
                cycles=[int(c) for c in args.cycles],
                eps=float(args.eps),
                dt=float(args.dt),
                t_final=args.t_final,
                time_chunk=int(args.time_chunk),
                x_window_radius=int(args.x_window_radius),
                q_min_tol=float(args.q_min_tol),
                charge_drift_tol=float(args.charge_drift_tol),
                max_samples=args.max_samples,
                max_y0=args.max_y0,
                overwrite=bool(args.overwrite),
                progress=progress,
                make_plots=not bool(args.no_plots),
            )
        )

    summary_path = write_summary(output_root, product_paths)
    manifest_path = write_campaign_manifest(
        output_root=output_root,
        bundle_root=bundle_root,
        product_paths=product_paths,
        args=args,
    )
    expected_rows = len(records) * len(args.cycles) * 4
    rows = []
    for path in product_paths:
        rows.extend(summary_rows_for_product(path))
    if len(rows) != expected_rows:
        raise ValueError(f"summary rows {len(rows)} != expected {expected_rows}.")
    log(f"Saved summary: {summary_path}", enabled=True)
    log(f"Saved manifest: {manifest_path}", enabled=True)


if __name__ == "__main__":
    main()
