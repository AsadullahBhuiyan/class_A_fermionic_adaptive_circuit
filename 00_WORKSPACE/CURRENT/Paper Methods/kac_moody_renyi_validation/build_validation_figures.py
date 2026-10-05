#!/usr/bin/env python3
"""Rebuild the data products for the Kac--Moody/Renyi validation note.

The accepted campaigns are immutable inputs.  This program opens them read-only,
checks their hashes and schemas, and writes only below this note's ``figures`` and
``tables`` directories plus ``analysis_manifest.json``.  Every statistical fit is
trajectory first; periodic origins are variance-reduction samples inside a trajectory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
TITLE = (
    "U(1) Kac--Moody Level and Renyi Entanglement in Monitored Gaussian "
    "Trajectories: Formalism and Numerical Validation"
)
FIT_AY_MIN = 8
RENyi_ORDERS = (1, 2, 3)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BASE_SEED = 20260901
BOOTSTRAP_CONFIDENCE = 0.95
EXPECTED_B0 = {
    "coupled": {
        "k_wall": 0.9985294032972167,
        "c_1": 1.0000917734345278,
        "c_2": 0.9959102751088306,
        "c_3": 0.9933596610739922,
    },
    "hard_exterior": {
        "k_wall": 0.9985293087648404,
        "c_1": 1.000091883831995,
        "c_2": 0.9959072032345465,
        "c_3": 0.9933241042941777,
    },
}
EXPECTED_RESULT21_K = {
    "N16x30_nsh1_perfect_correction": 1.0409,
    "N16x30_nsh2_perfect_correction": 1.0468,
    "N16x40_nsh1_perfect_correction": 1.0328,
    "N16x40_nsh2_perfect_correction": 1.0853,
}
SOURCE_ROOTS_RELATIVE = (
    Path("00_WORKSPACE/CURRENT/experiment_review"),
    Path("00_WORKSPACE/COLAB/colab_small_system_testing"),
    Path("00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs"),
)


def _discover_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "PROJECT_ADMIN" / "REPO_POLICY.md").is_file():
            return candidate
    raise FileNotFoundError("Could not locate repository root from builder path")


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether either resolved path contains the other."""

    left_resolved = left.resolve()
    right_resolved = right.resolve()
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_row(repo_root: Path, path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(repo_root.resolve()).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False, default=_json_default
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return format(float(value), ".17g")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    temporary.replace(path)


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _write_tex_table(
    path: Path,
    columns: Sequence[tuple[str, str, str]],
    rows: Sequence[Mapping[str, Any]],
    alignment: str | None = None,
) -> None:
    """Write a small booktabs fragment.

    ``columns`` contains ``(row_key, heading, format_spec)``.  A format spec of
    ``s`` selects escaped text; all other specs are passed to ``format``.
    """

    if alignment is None:
        alignment = "l" + "c" * (len(columns) - 1)
    lines = [r"\begin{ruledtabular}", rf"\begin{{tabular}}{{{alignment}}}", r"\toprule"]
    lines.append(" & ".join(heading for _, heading, _ in columns) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = []
        for key, _, spec in columns:
            value = row.get(key)
            if spec == "s":
                cells.append(_tex_escape(str(value)))
            elif value is None or not np.isfinite(float(value)):
                cells.append(r"---")
            else:
                cells.append(format(float(value), spec))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{ruledtabular}", ""])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def _finite_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def spectrum_diagnostics(eigenvalues: np.ndarray) -> dict[str, int | float]:
    """Count padding, numerical range excursions, and stability clipping."""

    raw = np.asarray(eigenvalues, dtype=float).reshape(-1)
    finite = raw[np.isfinite(raw)]
    maximum_range_correction = 0.0
    if finite.size:
        maximum_range_correction = max(
            0.0, float(-np.min(finite)), float(np.max(finite) - 1)
        )
    return {
        "total_values": int(raw.size),
        "finite_values": int(finite.size),
        "nonfinite_padding_values": int(raw.size - finite.size),
        "below_zero_values": int(np.count_nonzero(finite < 0)),
        "above_one_values": int(np.count_nonzero(finite > 1)),
        "outside_tolerance_values": int(
            np.count_nonzero((finite < -1e-8) | (finite > 1 + 1e-8))
        ),
        "range_clip_low_values": int(np.count_nonzero(finite < 0)),
        "range_clip_high_values": int(np.count_nonzero(finite > 1)),
        "exact_or_near_zero_values": int(np.count_nonzero(finite <= 1e-12)),
        "exact_or_near_one_values": int(np.count_nonzero(finite >= 1 - 1e-12)),
        "maximum_range_correction": maximum_range_correction,
    }


def _sum_diagnostics(
    rows: Iterable[Mapping[str, int | float]]
) -> dict[str, int | float]:
    total: dict[str, int | float] = {}
    for row in rows:
        for key, value in row.items():
            if key.startswith("maximum_"):
                total[key] = max(float(total.get(key, 0.0)), float(value))
            else:
                total[key] = int(total.get(key, 0)) + int(value)
    return total


def entropy_renyi(eigenvalues: np.ndarray, order: float) -> float:
    """Return the number-conserving Gaussian Renyi entropy in natural units."""

    if not np.isfinite(order) or order <= 0:
        raise ValueError("Renyi order must be finite and positive")
    values = _finite_eigenvalues(eigenvalues)
    if values.size == 0:
        return 0.0
    if np.min(values) < -1e-8 or np.max(values) > 1 + 1e-8:
        raise ValueError("Correlation eigenvalues lie outside [0,1]")
    # Clip only numerical range excursions already accepted by the tolerance above.
    # Exact 0 and 1 eigenvalues are retained and evaluated by endpoint-safe branches.
    values = np.clip(values, 0.0, 1.0)
    if math.isclose(float(order), 1.0, rel_tol=0.0, abs_tol=1e-12):
        interior = (values > 0) & (values < 1)
        selected = values[interior]
        return float(
            -np.sum(
                selected * np.log(selected)
                + (1 - selected) * np.log1p(-selected)
            )
        )
    n = float(order)
    interior = (values > 0) & (values < 1)
    selected = values[interior]
    if selected.size == 0:
        return 0.0
    numerator = np.logaddexp(
        n * np.log(selected), n * np.log1p(-selected)
    )
    return float(np.sum(numerator / (1 - n)))


def susceptibility_kernel(epsilon: np.ndarray | Sequence[float] | float) -> np.ndarray:
    """Return the raw modular-energy F kernel v=1/(4 cosh^2(epsilon/2))."""

    values = np.asarray(epsilon, dtype=float)
    return 1.0 / (4.0 * np.square(np.cosh(values / 2.0)))


def _legacy_b0_entropy_renyi(eigenvalues: np.ndarray, order: float) -> float:
    """Reproduce the locked B0 1e-12 endpoint-floor convention explicitly.

    The canonical helper above is endpoint exact.  B0's published regression targets
    were generated with this floor, so the audit reports both estimators rather than
    silently changing the accepted numbers.
    """

    values = _finite_eigenvalues(eigenvalues)
    values = np.clip(values, 1e-12, 1 - 1e-12)
    n = float(order)
    if math.isclose(n, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return float(
            -np.sum(values * np.log(values) + (1 - values) * np.log1p(-values))
        )
    return float(
        np.sum(
            np.logaddexp(n * np.log(values), n * np.log1p(-values)) / (1 - n)
        )
    )


def charge_cumulants(eigenvalues: np.ndarray) -> dict[str, float]:
    """Return Gaussian full-counting-statistics cumulants kappa_1 through kappa_4."""

    values = _finite_eigenvalues(eigenvalues)
    if values.size == 0:
        return {f"kappa_{index}": 0.0 for index in range(1, 5)}
    if np.min(values) < -1e-8 or np.max(values) > 1 + 1e-8:
        raise ValueError("Correlation eigenvalues lie outside [0,1]")
    values = np.clip(values, 0.0, 1.0)
    susceptibility = values * (1 - values)
    return {
        "kappa_1": float(np.sum(values)),
        "kappa_2": float(np.sum(susceptibility)),
        "kappa_3": float(np.sum(susceptibility * (1 - 2 * values))),
        "kappa_4": float(np.sum(susceptibility * (1 - 6 * susceptibility))),
    }


def coefficient_estimates(
    entropy_slope: float, charge_slope: float, order: float
) -> dict[str, float]:
    """Convert two-wall strip slopes to per-wall c, k, and the paired null residual."""

    n = float(order)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("Renyi order must be finite and positive")
    c_wall = 6 * float(entropy_slope) / (1 + 1 / n)
    k_wall = np.pi**2 * float(charge_slope)
    delta_n = float(entropy_slope) - (
        np.pi**2 / 6 * (1 + 1 / n) * float(charge_slope)
    )
    return {
        "c_wall": float(c_wall),
        "k_wall": float(k_wall),
        "delta_n": float(delta_n),
        "normalized_delta": float(6 * delta_n / (1 + 1 / n)),
    }


def periodic_full_x_charge_curve(
    G: np.ndarray, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Exact periodic-origin average of Tr[C_A(1-C_A)] for every Ay <= Ny/2."""

    nlayer = 2 * int(nx) * int(ny)
    covariance = np.asarray(G)
    if covariance.shape != (nlayer, nlayer):
        raise ValueError(f"Expected {(nlayer, nlayer)}, received {covariance.shape}")
    correlation = np.asarray(covariance, dtype=np.complex128).copy()
    correlation.flat[:: nlayer + 1] += 1.0
    correlation *= 0.5
    block_dimension = 2 * int(nx)
    blocks = correlation.reshape(ny, block_dimension, ny, block_dimension)
    rows = np.arange(ny)
    block_norms = np.empty(ny, dtype=float)
    for displacement in range(ny):
        block = blocks[rows, :, (rows + displacement) % ny, :]
        block_norms[displacement] = np.square(np.abs(block)).sum(axis=(1, 2)).mean()
    density_per_y = float(np.trace(correlation).real / ny)
    ay = np.arange(1, ny // 2 + 1, dtype=int)
    variance = np.empty(ay.size, dtype=float)
    for index, length in enumerate(ay):
        cross = sum(
            (length - displacement)
            * (block_norms[displacement] + block_norms[-displacement])
            for displacement in range(1, length)
        )
        variance[index] = length * (density_per_y - block_norms[0]) - cross
    return ay, variance, {"nbar": density_per_y, "K0": float(block_norms[0])}


def _chord_coordinate(ny: int, ay: np.ndarray) -> np.ndarray:
    values = np.asarray(ay, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log((ny / np.pi) * np.sin(np.pi * values / ny))


def _linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 2:
        raise ValueError("Linear fit needs equal one-dimensional arrays with >=2 points")
    design = np.column_stack((x, np.ones_like(x)))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    residual = y - predicted
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(residual @ residual) / denominator if denominator > 0 else 1.0
    return {
        "slope": float(beta[0]),
        "intercept": float(beta[1]),
        "r2": float(r2),
        "n_points": int(x.size),
    }


def _linear_model_aic(x: np.ndarray, y: np.ndarray) -> tuple[float, dict[str, float | int]]:
    fit = _linear_fit(x, y)
    predicted = float(fit["slope"]) * np.asarray(x) + float(fit["intercept"])
    residual_sum = float(np.sum((np.asarray(y) - predicted) ** 2))
    scale = max(residual_sum / len(y), np.finfo(float).tiny)
    return float(len(y) * np.log(scale) + 4), fit


def _constant_model_aic(y: np.ndarray) -> float:
    values = np.asarray(y, dtype=float)
    residual_sum = float(np.sum((values - np.mean(values)) ** 2))
    scale = max(residual_sum / len(values), np.finfo(float).tiny)
    return float(len(values) * np.log(scale) + 2)


def _mean_sem(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    sem = float(np.std(array, ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return float(np.mean(array)), sem


def _seed_for(label: str) -> int:
    payload = f"{BOOTSTRAP_BASE_SEED}:{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _bootstrap_joint(
    values: np.ndarray, labels: Sequence[str], seed_label: str
) -> dict[str, Any]:
    """Deterministic paired percentile bootstrap over complete trajectories."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != len(labels):
        raise ValueError("Bootstrap matrix must be S-by-M with S>=2")
    seed = _seed_for(seed_label)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, matrix.shape[0], size=(BOOTSTRAP_REPLICATES, matrix.shape[0])
    )
    means = matrix[indices].mean(axis=1)
    tail = (1 - BOOTSTRAP_CONFIDENCE) / 2
    lower, upper = np.quantile(means, (tail, 1 - tail), axis=0)
    covariance = np.cov(means, rowvar=False, ddof=1)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]])
    return {
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence": BOOTSTRAP_CONFIDENCE,
        "mean": {label: float(value) for label, value in zip(labels, matrix.mean(axis=0))},
        "ci_low": {label: float(value) for label, value in zip(labels, lower)},
        "ci_high": {label: float(value) for label, value in zip(labels, upper)},
        "covariance_labels": list(labels),
        "covariance": np.asarray(covariance, dtype=float).tolist(),
    }


def _fit_windows(ny: int) -> tuple[dict[str, int | str], ...]:
    maximum = int(ny) // 2
    if maximum - FIT_AY_MIN < 2:
        raise ValueError(f"Insufficient interval lengths for sensitivity fits at Ny={ny}")
    return (
        {"window": "primary", "Ay_fit_min": FIT_AY_MIN, "Ay_fit_max": maximum},
        {
            "window": "drop_largest",
            "Ay_fit_min": FIT_AY_MIN,
            "Ay_fit_max": maximum - 1,
        },
        {
            "window": "drop_smallest",
            "Ay_fit_min": FIT_AY_MIN + 1,
            "Ay_fit_max": maximum,
        },
    )


def _direct_charge_check(G: np.ndarray, nx: int, ny: int, variance: np.ndarray) -> float:
    nlayer = G.shape[0]
    correlation = np.asarray(G, dtype=np.complex128).copy()
    correlation.flat[:: nlayer + 1] += 1.0
    correlation *= 0.5
    errors = []
    for length in sorted({1, FIT_AY_MIN, ny // 2}):
        origins = []
        for y0 in range(ny):
            ys = (y0 + np.arange(length)) % ny
            indices = (
                ys[:, None] * (2 * nx) + np.arange(2 * nx)[None, :]
            ).reshape(-1)
            restricted = correlation[np.ix_(indices, indices)]
            origins.append(
                float(
                    np.trace(restricted).real
                    - np.square(np.abs(restricted)).sum()
                )
            )
        errors.append(abs(float(np.mean(origins)) - float(variance[length - 1])))
    return max(errors)


def _reduce_exact_b0(repo_root: Path) -> dict[str, Any]:
    base = (
        repo_root
        / "00_WORKSPACE/CURRENT/experiment_review/b0_exact_domain_wall/results/20260816_191957"
    )
    rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    config_path = base / "campaign_config.v2.json"
    sources.append(_source_row(repo_root, config_path, "accepted B0 campaign contract"))
    config = _json_load(config_path)
    if int(config["entropy_fit_ay_min"]) != FIT_AY_MIN:
        raise ValueError("B0 entropy fit window no longer matches the note contract")

    for construction in ("coupled", "hard_exterior"):
        source = base / "raw" / f"{construction}__Nx020__Ny048.npz"
        sources.append(_source_row(repo_root, source, f"accepted B0 {construction} spectra"))
        with np.load(source, allow_pickle=False) as archive:
            nx = int(archive["nx"])
            ny = int(archive["ny"])
            ay = np.asarray(archive["entropy_ay"], dtype=int)
            spectra = np.asarray(archive["entropy_spectra"], dtype=float)
            archived_entropies = np.asarray(archive["entropy_values"], dtype=float)

        mask = ay >= FIT_AY_MIN
        x = _chord_coordinate(ny, ay)
        charge_variance = np.empty(ay.size, dtype=float)
        # The accepted B0 archive was produced with an explicitly documented
        # 1e-12 endpoint floor.  Preserve that historical estimator for the
        # locked regression while also evaluating the canonical endpoint-exact
        # formula used everywhere else in this note.  Keeping both reductions
        # makes the tiny compatibility offset visible instead of silently
        # changing the accepted regression values.
        entropies = {order: np.empty(ay.size, dtype=float) for order in RENyi_ORDERS}
        exact_entropies = {
            order: np.empty(ay.size, dtype=float) for order in RENyi_ORDERS
        }
        finite_counts = []
        spectrum_check_rows = []
        for index, padded_spectrum in enumerate(spectra):
            spectrum_check_rows.append(spectrum_diagnostics(padded_spectrum))
            eigenvalues = _finite_eigenvalues(padded_spectrum)
            finite_counts.append(int(eigenvalues.size))
            charge_variance[index] = charge_cumulants(eigenvalues)["kappa_2"]
            for order in RENyi_ORDERS:
                entropies[order][index] = _legacy_b0_entropy_renyi(
                    eigenvalues, order
                )
                exact_entropies[order][index] = entropy_renyi(eigenvalues, order)

        charge_fit = _linear_fit(x[mask], charge_variance[mask])
        row: dict[str, Any] = {
            "construction": construction,
            "Nx": nx,
            "Ny": ny,
            "Ay_fit_min": FIT_AY_MIN,
            "Ay_fit_max": int(ay[mask].max()),
            "n_fit_points": int(np.count_nonzero(mask)),
            "charge_slope": charge_fit["slope"],
            "k_wall": np.pi**2 * float(charge_fit["slope"]),
            "charge_r2": charge_fit["r2"],
        }
        maximum_entropy_reconstruction_error = 0.0
        maximum_exact_compatibility_difference = 0.0
        for order_index, order in enumerate(RENyi_ORDERS):
            fit = _linear_fit(x[mask], entropies[order][mask])
            exact_fit = _linear_fit(x[mask], exact_entropies[order][mask])
            converted = coefficient_estimates(
                float(fit["slope"]), float(charge_fit["slope"]), order
            )
            exact_converted = coefficient_estimates(
                float(exact_fit["slope"]), float(charge_fit["slope"]), order
            )
            row.update(
                {
                    f"entropy_slope_{order}": fit["slope"],
                    f"c_{order}": converted["c_wall"],
                    f"entropy_r2_{order}": fit["r2"],
                    f"delta_{order}": converted["delta_n"],
                    f"normalized_delta_{order}": converted["normalized_delta"],
                    f"entropy_slope_exact_{order}": exact_fit["slope"],
                    f"c_exact_{order}": exact_converted["c_wall"],
                    f"delta_exact_{order}": exact_converted["delta_n"],
                    f"exact_minus_compatibility_c_{order}": float(
                        exact_converted["c_wall"] - converted["c_wall"]
                    ),
                }
            )
            maximum_entropy_reconstruction_error = max(
                maximum_entropy_reconstruction_error,
                float(
                    np.nanmax(
                        np.abs(entropies[order][1:] - archived_entropies[order_index, 1:])
                    )
                ),
            )
            maximum_exact_compatibility_difference = max(
                maximum_exact_compatibility_difference,
                float(
                    np.nanmax(
                        np.abs(exact_entropies[order] - entropies[order])
                    )
                ),
            )
        rows.append(row)
        for window in _fit_windows(ny):
            window_mask = (
                (ay >= int(window["Ay_fit_min"]))
                & (ay <= int(window["Ay_fit_max"]))
            )
            window_charge_fit = _linear_fit(
                x[window_mask], charge_variance[window_mask]
            )
            sensitivity_row: dict[str, Any] = {
                "construction": construction,
                "Nx": nx,
                "Ny": ny,
                **window,
                "n_fit_points": int(np.count_nonzero(window_mask)),
                "charge_slope": window_charge_fit["slope"],
                "k_wall": np.pi**2 * float(window_charge_fit["slope"]),
                "charge_r2": window_charge_fit["r2"],
            }
            for order in RENyi_ORDERS:
                window_entropy_fit = _linear_fit(
                    x[window_mask], entropies[order][window_mask]
                )
                converted = coefficient_estimates(
                    window_entropy_fit["slope"],
                    window_charge_fit["slope"],
                    order,
                )
                sensitivity_row.update(
                    {
                        f"c_{order}": converted["c_wall"],
                        f"delta_{order}": converted["delta_n"],
                        f"normalized_delta_{order}": converted["normalized_delta"],
                        f"entropy_r2_{order}": window_entropy_fit["r2"],
                    }
                )
            sensitivity_rows.append(sensitivity_row)
        for index, length in enumerate(ay):
            curves.append(
                {
                    "construction": construction,
                    "Nx": nx,
                    "Ny": ny,
                    "Ay": int(length),
                    "log_chord": float(x[index]) if length else None,
                    "charge_variance": charge_variance[index],
                    "S_1": entropies[1][index],
                    "S_2": entropies[2][index],
                    "S_3": entropies[3][index],
                    "S_exact_1": exact_entropies[1][index],
                    "S_exact_2": exact_entropies[2][index],
                    "S_exact_3": exact_entropies[3][index],
                    "finite_spectrum_size": finite_counts[index],
                }
            )
        expected = EXPECTED_B0[construction]
        deviations = {
            key: abs(float(row[key]) - float(expected[key]))
            for key in ("k_wall", "c_1", "c_2", "c_3")
        }
        validations.append(
            {
                "construction": construction,
                "maximum_expected_regression_deviation": max(deviations.values()),
                "maximum_entropy_spectrum_reconstruction_error": maximum_entropy_reconstruction_error,
                "maximum_endpoint_exact_vs_archive_compatibility_entropy_difference": maximum_exact_compatibility_difference,
                "archived_entropy_estimator": "explicit eigenvalue floor at 1e-12 (compatibility reduction)",
                "canonical_entropy_estimator": "endpoint-exact logaddexp reduction",
                "nan_padding_filtered": True,
                "spectrum_diagnostics": _sum_diagnostics(spectrum_check_rows),
                "endpoint_windows_evaluated": [
                    dict(window) for window in _fit_windows(ny)
                ],
                "pass": max(deviations.values()) < 1e-10
                and maximum_entropy_reconstruction_error < 1e-10
                and _sum_diagnostics(spectrum_check_rows)[
                    "outside_tolerance_values"
                ]
                == 0,
            }
        )
    if not all(item["pass"] for item in validations):
        raise AssertionError(f"Exact B0 regression failed: {validations}")
    return {
        "rows": rows,
        "sensitivity_rows": sensitivity_rows,
        "curves": curves,
        "sources": sources,
        "validations": validations,
    }


def _interval_entropies(
    correlation: np.ndarray, nx: int, ny: int, ay: int, y0: int
) -> tuple[tuple[float, float, float], float, dict[str, int | float]]:
    ys = (int(y0) + np.arange(int(ay))) % int(ny)
    indices = (ys[:, None] * (2 * nx) + np.arange(2 * nx)[None, :]).reshape(-1)
    eigenvalues = np.linalg.eigvalsh(correlation[np.ix_(indices, indices)])
    entropies = tuple(entropy_renyi(eigenvalues, order) for order in RENyi_ORDERS)
    compatibility_s1 = _legacy_b0_entropy_renyi(eigenvalues, 1)
    return entropies, compatibility_s1, spectrum_diagnostics(eigenvalues)  # type: ignore[return-value]


def _stochastic_renyi_case(
    snapshots: np.ndarray,
    cycle_index: int,
    nx: int,
    ny: int,
    workers: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, int | float],
]:
    """Exact periodic-origin Renyi curves and trajectory-first fits."""

    ay_values = np.arange(1, ny // 2 + 1, dtype=int)
    x = _chord_coordinate(ny, ay_values)
    fit_mask = ay_values >= FIT_AY_MIN
    curve_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, int | float]] = []
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - all maintained environments include it
        from contextlib import nullcontext

        limits_context = nullcontext()
    else:
        limits_context = threadpool_limits(limits=1, user_api="blas")

    with limits_context:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
            for sample_index in range(int(snapshots.shape[0])):
                G = np.asarray(snapshots[sample_index, cycle_index])
                nlayer = G.shape[0]
                correlation = np.asarray(G, dtype=np.complex128).copy()
                correlation.flat[:: nlayer + 1] += 1.0
                correlation *= 0.5
                futures = []
                for ay in ay_values:
                    for y0 in range(ny):
                        futures.append(
                            (
                                int(ay),
                                executor.submit(
                                    _interval_entropies,
                                    correlation,
                                    nx,
                                    ny,
                                    int(ay),
                                    y0,
                                ),
                            )
                        )
                by_length: dict[int, list[tuple[float, float, float]]] = {
                    int(ay): [] for ay in ay_values
                }
                compatibility_by_length: dict[int, list[float]] = {
                    int(ay): [] for ay in ay_values
                }
                for ay, future in futures:
                    entropies, compatibility_s1, diagnostics = future.result()
                    by_length[ay].append(entropies)
                    compatibility_by_length[ay].append(compatibility_s1)
                    diagnostic_rows.append(diagnostics)
                trajectory_curves = np.asarray(
                    [np.mean(by_length[int(ay)], axis=0) for ay in ay_values],
                    dtype=float,
                )
                compatibility_s1_curve = np.asarray(
                    [
                        np.mean(compatibility_by_length[int(ay)])
                        for ay in ay_values
                    ],
                    dtype=float,
                )
                for ay_index, ay in enumerate(ay_values):
                    curve_rows.append(
                        {
                            "sample_index": sample_index,
                            "Ay": int(ay),
                            "log_chord": float(x[ay_index]),
                            "origin_count": ny,
                            "S_1": trajectory_curves[ay_index, 0],
                            "S_1_archive_compatibility": compatibility_s1_curve[
                                ay_index
                            ],
                            "S_2": trajectory_curves[ay_index, 1],
                            "S_3": trajectory_curves[ay_index, 2],
                        }
                    )
                row: dict[str, Any] = {
                    "sample_index": sample_index,
                    "Ay_fit_min": FIT_AY_MIN,
                    "Ay_fit_max": int(ay_values[-1]),
                    "origin_count": ny,
                }
                for order_index, order in enumerate(RENyi_ORDERS):
                    fit = _linear_fit(
                        x[fit_mask], trajectory_curves[fit_mask, order_index]
                    )
                    row.update(
                        {
                            f"entropy_slope_{order}": fit["slope"],
                            f"c_{order}": 6
                            * float(fit["slope"])
                            / (1 + 1 / order),
                            f"entropy_r2_{order}": fit["r2"],
                        }
                    )
                fit_rows.append(row)
                compatibility_fit = _linear_fit(
                    x[fit_mask], compatibility_s1_curve[fit_mask]
                )
                row["c_1_archive_compatibility"] = 3 * float(
                    compatibility_fit["slope"]
                )
                row["archive_compatibility_entropy_r2_1"] = compatibility_fit[
                    "r2"
                ]
    return curve_rows, fit_rows, _sum_diagnostics(diagnostic_rows)


def _read_entropy_coefficients(path: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["case_id"]] = {
                "c_eff": 3 * np.log(2) * float(row["slope_over_ln2"]),
                "c_eff_err": 3 * np.log(2) * float(row["slope_err_over_ln2"]),
            }
    return rows


def _reduce_result21(
    repo_root: Path, compute_stochastic_renyi: bool, workers: int
) -> dict[str, Any]:
    snapshot_base = (
        repo_root
        / "00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data/pure_state_covariance_snapshots"
    )
    colab_gpu_root = repo_root / "00_WORKSPACE/COLAB/colab_small_system_testing/gpu_data"
    entropy_table = (
        repo_root
        / "00_WORKSPACE/COLAB/colab_small_system_testing/analysis_outputs/"
        "pure_state_entanglement_vs_system_size_cpu/"
        "full_x_late_window_log_chord_fit_rows.csv"
    )
    cases = [
        {
            "case_id": f"N16x{ny}_nsh{nshell}_perfect_correction",
            "Nx": 16,
            "Ny": ny,
            "nshell": nshell,
        }
        for ny in (30, 40)
        for nshell in (1, 2)
    ]
    sources = [
        _source_row(repo_root, snapshot_base / "campaign_manifest.json", "snapshot campaign contract"),
        _source_row(repo_root, entropy_table, "matched archived S1 regression table"),
    ]
    entropy_coefficients = _read_entropy_coefficients(entropy_table)
    charge_curves: list[dict[str, Any]] = []
    charge_fits: list[dict[str, Any]] = []
    renyi_curves: list[dict[str, Any]] = []
    renyi_fits: list[dict[str, Any]] = []
    renyi_spectrum_checks: list[dict[str, Any]] = []
    formula_checks: list[dict[str, Any]] = []
    state_checks: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["case_id"]
        latest_path = snapshot_base / "runs" / case_id / "latest_run.json"
        latest_source = _source_row(
            repo_root, latest_path, f"{case_id} immutable run pointer"
        )
        latest = _json_load(latest_path)
        run_dir = colab_gpu_root / latest["run_dir_relative"]
        manifest_path = run_dir / "manifest.json"
        snapshots_path = run_dir / "batch_00000_snapshots.npy"
        manifest_source = _source_row(
            repo_root, manifest_path, f"{case_id} run manifest"
        )
        snapshot_source = _source_row(
            repo_root, snapshots_path, f"{case_id} raw covariances"
        )
        sources.extend(
            [latest_source, manifest_source, snapshot_source]
        )
        manifest = _json_load(manifest_path)
        config = manifest["config"]
        latest_source["metadata"] = {
            "resolved_run_dir_relative": latest["run_dir_relative"],
            "run_id": manifest["run_id"],
        }
        manifest_source["metadata"] = {
            "case_id": case_id,
            "run_id": manifest["run_id"],
            "sample_indices": list(range(int(config["samples"]))),
            "seed_identity": "not archived in this legacy manifest",
            "engine_sha256": "not archived in this legacy manifest",
        }
        snapshot_source["metadata"] = {
            "case_id": case_id,
            "sample_indices": list(range(int(config["samples"]))),
            "snapshot_cycles": list(config["snapshot_cycles"]),
            "dtype_archived": config["dtype"],
            "dtype_resolved": manifest["batch_size_auto_info"]["dtype"],
            "covariance_convention": "archived G=2*C-I; builder reconstructs C=(G+I)/2",
            "initialization": config["init_mode"],
            "sequence": config["sequence"],
            "seed_identity": "not archived in this legacy manifest",
            "engine_sha256": "not archived in this legacy manifest",
        }
        required = (
            bool(config["DW"])
            and bool(config["dw_truncation"])
            and bool(config["perfect_correction"])
            and not bool(config["postselect"])
            and int(config["samples"]) == 10
            and int(config["Nx"]) == case["Nx"]
            and int(config["Ny"]) == case["Ny"]
            and int(config["nshell"]) == case["nshell"]
            and str(config["dtype"]) == "c128"
            and str(manifest["batch_size_auto_info"]["dtype"]) == "complex128"
            and str(config["init_mode"]) == "default"
            and str(config["sequence"]) == "raster_y"
        )
        if not required:
            raise ValueError(f"Result 21 contract mismatch for {case_id}")
        cycle_index = list(config["snapshot_cycles"]).index(50)
        snapshots = np.load(snapshots_path, mmap_mode="r")
        for sample_index in range(int(config["samples"])):
            G = np.asarray(snapshots[sample_index, cycle_index])
            nlayer = G.shape[0]
            correlation = np.asarray(G, dtype=np.complex128).copy()
            correlation.flat[:: nlayer + 1] += 1.0
            correlation *= 0.5
            hermiticity_error = float(np.max(np.abs(G - G.conj().T)))
            global_quantum_variance = float(
                np.trace(correlation).real - np.square(np.abs(correlation)).sum()
            )
            state_checks.append(
                {
                    **case,
                    "sample_index": sample_index,
                    "cycle": 50,
                    "hermiticity_max_abs_error": hermiticity_error,
                    "global_quantum_variance": global_quantum_variance,
                    "global_purity_deficit_per_mode": abs(global_quantum_variance)
                    / nlayer,
                    "pass": hermiticity_error < 1e-10
                    and abs(global_quantum_variance) / nlayer < 1e-10,
                }
            )
            ay, variance, auxiliaries = periodic_full_x_charge_curve(
                G, case["Nx"], case["Ny"]
            )
            x = _chord_coordinate(case["Ny"], ay)
            mask = ay >= FIT_AY_MIN
            fit = _linear_fit(x[mask], variance[mask])
            charge_fits.append(
                {
                    **case,
                    "sample_index": sample_index,
                    "cycle": 50,
                    "Ay_fit_min": FIT_AY_MIN,
                    "Ay_fit_max": int(ay[-1]),
                    "charge_slope": fit["slope"],
                    "k_wall": np.pi**2 * float(fit["slope"]),
                    "charge_r2": fit["r2"],
                }
            )
            charge_curves.extend(
                {
                    **case,
                    "sample_index": sample_index,
                    "cycle": 50,
                    "Ay": int(length),
                    "log_chord": float(coordinate),
                    "charge_variance": float(value),
                }
                for length, coordinate, value in zip(ay, x, variance)
            )
            if sample_index == 0:
                formula_checks.append(
                    {
                        **case,
                        "max_abs_error": _direct_charge_check(
                            snapshots[sample_index, cycle_index],
                            case["Nx"],
                            case["Ny"],
                            variance,
                        ),
                        **auxiliaries,
                    }
                )
        if compute_stochastic_renyi:
            print(
                f"    [Result 21] {case_id}: exact all-origin S1/S2/S3 reduction",
                flush=True,
            )
            case_curves, case_fits, case_spectrum_check = _stochastic_renyi_case(
                snapshots,
                cycle_index,
                case["Nx"],
                case["Ny"],
                workers,
            )
            renyi_curves.extend({**case, "cycle": 50, **row} for row in case_curves)
            renyi_fits.extend({**case, "cycle": 50, **row} for row in case_fits)
            renyi_spectrum_checks.append({**case, **case_spectrum_check})
            print(f"    [Result 21] {case_id}: complete", flush=True)

    if max(row["max_abs_error"] for row in formula_checks) >= 1e-10:
        raise AssertionError(f"Origin-average formula check failed: {formula_checks}")
    if not all(row["pass"] for row in state_checks):
        raise AssertionError("Result 21 purity/Hermiticity checks failed")
    if compute_stochastic_renyi and any(
        row["outside_tolerance_values"] for row in renyi_spectrum_checks
    ):
        raise AssertionError("Result 21 restricted spectra exceed the range tolerance")

    summaries: list[dict[str, Any]] = []
    paired_fits: list[dict[str, Any]] = []
    sensitivity_fit_rows: list[dict[str, Any]] = []
    sensitivity_summary_rows: list[dict[str, Any]] = []
    bootstrap_details: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["case_id"]
        selected_charge = [row for row in charge_fits if row["case_id"] == case_id]
        charge_curve_by_sample = {
            sample_index: sorted(
                (
                    item
                    for item in charge_curves
                    if item["case_id"] == case_id
                    and item["sample_index"] == sample_index
                ),
                key=lambda item: item["Ay"],
            )
            for sample_index in range(10)
        }
        renyi_curve_by_sample = {
            sample_index: sorted(
                (
                    item
                    for item in renyi_curves
                    if item["case_id"] == case_id
                    and item["sample_index"] == sample_index
                ),
                key=lambda item: item["Ay"],
            )
            for sample_index in range(10)
        }
        for window in _fit_windows(case["Ny"]):
            for sample_index in range(10):
                charge_curve = charge_curve_by_sample[sample_index]
                window_charge = [
                    item
                    for item in charge_curve
                    if int(window["Ay_fit_min"])
                    <= item["Ay"]
                    <= int(window["Ay_fit_max"])
                ]
                charge_fit = _linear_fit(
                    np.asarray([item["log_chord"] for item in window_charge]),
                    np.asarray([item["charge_variance"] for item in window_charge]),
                )
                sensitivity: dict[str, Any] = {
                    **case,
                    "sample_index": sample_index,
                    "cycle": 50,
                    **window,
                    "k_wall": np.pi**2 * float(charge_fit["slope"]),
                    "charge_slope": charge_fit["slope"],
                    "charge_r2": charge_fit["r2"],
                }
                if compute_stochastic_renyi:
                    renyi_curve = [
                        item
                        for item in renyi_curve_by_sample[sample_index]
                        if int(window["Ay_fit_min"])
                        <= item["Ay"]
                        <= int(window["Ay_fit_max"])
                    ]
                    for order in RENyi_ORDERS:
                        entropy_fit = _linear_fit(
                            np.asarray([item["log_chord"] for item in renyi_curve]),
                            np.asarray([item[f"S_{order}"] for item in renyi_curve]),
                        )
                        converted = coefficient_estimates(
                            entropy_fit["slope"], charge_fit["slope"], order
                        )
                        sensitivity.update(
                            {
                                f"c_{order}": converted["c_wall"],
                                f"delta_{order}": converted["delta_n"],
                                f"normalized_delta_{order}": converted[
                                    "normalized_delta"
                                ],
                                f"entropy_r2_{order}": entropy_fit["r2"],
                            }
                        )
                    compatibility_fit = _linear_fit(
                        np.asarray(
                            [item["log_chord"] for item in renyi_curve]
                        ),
                        np.asarray(
                            [
                                item["S_1_archive_compatibility"]
                                for item in renyi_curve
                            ]
                        ),
                    )
                    sensitivity["c_1_archive_compatibility"] = 3 * float(
                        compatibility_fit["slope"]
                    )
                    sensitivity[
                        "archive_compatibility_entropy_r2_1"
                    ] = compatibility_fit["r2"]
                sensitivity_fit_rows.append(sensitivity)

            selected_window = [
                item
                for item in sensitivity_fit_rows
                if item["case_id"] == case_id
                and item["window"] == window["window"]
            ]
            labels = ["k_wall"]
            if compute_stochastic_renyi:
                labels.extend(f"c_{order}" for order in RENyi_ORDERS)
                labels.extend(f"normalized_delta_{order}" for order in RENyi_ORDERS)
            matrix = np.asarray(
                [[item[label] for label in labels] for item in selected_window],
                dtype=float,
            )
            bootstrap = _bootstrap_joint(
                matrix, labels, f"result21:{case_id}:{window['window']}"
            )
            bootstrap_details.append(
                {"case_id": case_id, **window, **bootstrap}
            )
            sensitivity_summary: dict[str, Any] = {
                **case,
                "cycle": 50,
                "trajectories": 10,
                **window,
                "bootstrap_seed": bootstrap["seed"],
                "bootstrap_replicates": bootstrap["replicates"],
            }
            for label in labels:
                sensitivity_summary[label] = bootstrap["mean"][label]
                sensitivity_summary[f"{label}_ci_low"] = bootstrap["ci_low"][label]
                sensitivity_summary[f"{label}_ci_high"] = bootstrap["ci_high"][label]
            sensitivity_summary_rows.append(sensitivity_summary)

        primary = next(
            item
            for item in sensitivity_summary_rows
            if item["case_id"] == case_id and item["window"] == "primary"
        )
        k_mean, k_sem = _mean_sem(row["k_wall"] for row in selected_charge)
        row = {
            **case,
            "cycle": 50,
            "trajectories": len(selected_charge),
            "origin_count_per_trajectory": case["Ny"],
            "Ay_fit_min": FIT_AY_MIN,
            "Ay_fit_max": case["Ny"] // 2,
            "k_wall": k_mean,
            "k_wall_sem": k_sem,
            "k_wall_ci_low": primary["k_wall_ci_low"],
            "k_wall_ci_high": primary["k_wall_ci_high"],
            "bootstrap_seed": primary["bootstrap_seed"],
            "bootstrap_replicates": primary["bootstrap_replicates"],
            "mean_charge_r2": float(
                np.mean([item["charge_r2"] for item in selected_charge])
            ),
            "c_1_archived": entropy_coefficients[case_id]["c_eff"],
            "c_1_archived_regression_error": entropy_coefficients[case_id][
                "c_eff_err"
            ],
        }
        if compute_stochastic_renyi:
            selected_primary = [
                item
                for item in sensitivity_fit_rows
                if item["case_id"] == case_id and item["window"] == "primary"
            ]
            paired_fits.extend(selected_primary)
            for order in RENyi_ORDERS:
                _, sem = _mean_sem(item[f"c_{order}"] for item in selected_primary)
                _, delta_sem = _mean_sem(
                    item[f"normalized_delta_{order}"] for item in selected_primary
                )
                row[f"c_{order}"] = primary[f"c_{order}"]
                row[f"c_{order}_sem"] = sem
                row[f"c_{order}_ci_low"] = primary[f"c_{order}_ci_low"]
                row[f"c_{order}_ci_high"] = primary[f"c_{order}_ci_high"]
                row[f"normalized_delta_{order}"] = primary[
                    f"normalized_delta_{order}"
                ]
                row[f"normalized_delta_{order}_sem"] = delta_sem
                row[f"normalized_delta_{order}_ci_low"] = primary[
                    f"normalized_delta_{order}_ci_low"
                ]
                row[f"normalized_delta_{order}_ci_high"] = primary[
                    f"normalized_delta_{order}_ci_high"
                ]
            row["c_1_archive_compatibility"] = float(
                np.mean(
                    [
                        item["c_1_archive_compatibility"]
                        for item in selected_primary
                    ]
                )
            )
            row["endpoint_exact_minus_archive_c_1"] = float(
                row["c_1"] - row["c_1_archived"]
            )
            row["archive_compatibility_S1_reproduction_error"] = abs(
                row["c_1_archive_compatibility"] - row["c_1_archived"]
            )
            row["archive_compatibility_S1_reproduction_pass"] = bool(
                row["archive_compatibility_S1_reproduction_error"] < 1e-10
            )
            if not row["archive_compatibility_S1_reproduction_pass"]:
                raise AssertionError(
                    f"Archive-compatible S1 mismatch for {case_id}: {row}"
                )
        summaries.append(row)

    expected_checks = []
    for row in summaries:
        deviation = abs(row["k_wall"] - EXPECTED_RESULT21_K[row["case_id"]])
        expected_checks.append(
            {
                "case_id": row["case_id"],
                "rounded_expected_k": EXPECTED_RESULT21_K[row["case_id"]],
                "absolute_deviation": deviation,
                "pass": deviation < 5e-5,
            }
        )
    if not all(row["pass"] for row in expected_checks):
        raise AssertionError(f"Result 21 regression failed: {expected_checks}")
    return {
        "summary_rows": summaries,
        "charge_curve_rows": charge_curves,
        "charge_fit_rows": charge_fits,
        "renyi_curve_rows": renyi_curves,
        "renyi_fit_rows": renyi_fits,
        "paired_fit_rows": paired_fits,
        "sensitivity_fit_rows": sensitivity_fit_rows,
        "sensitivity_summary_rows": sensitivity_summary_rows,
        "bootstrap_details": bootstrap_details,
        "formula_checks": formula_checks,
        "state_checks": state_checks,
        "renyi_spectrum_checks": renyi_spectrum_checks,
        "expected_checks": expected_checks,
        "sources": sources,
        "stochastic_renyi_computed": compute_stochastic_renyi,
    }


def _stream_production_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    manifest: dict[str, Any] | None = None
    selected_payload: bytes | None = None
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            normalized = member.name.lstrip("./")
            if normalized == "manifest.json":
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unreadable manifest in {path}")
                manifest = json.load(extracted)
            elif normalized.endswith("/selected_observables.npz"):
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Unreadable selected observables in {path}")
                selected_payload = extracted.read()
            if manifest is not None and selected_payload is not None:
                break
    if manifest is None or selected_payload is None:
        raise ValueError(f"Archive lacks required compact products: {path}")
    expected_hash = manifest["products"]["selected_observables"]["sha256"]
    if _sha256_bytes(selected_payload) != expected_hash:
        raise ValueError(f"Selected-observable hash mismatch in {path}")
    return manifest, selected_payload


def _case_kind(case_id: str) -> tuple[str, bool, int]:
    if "explicit_interface" in case_id:
        construction = "explicit_interface"
    elif "support_terminated" in case_id:
        construction = "support_terminated"
    else:
        raise ValueError(f"Unknown W1 construction: {case_id}")
    matched_trivial = case_id.endswith("_matched_trivial")
    ny_token = next(token for token in case_id.split("_") if token.startswith("N20x"))
    return construction, matched_trivial, int(ny_token.removeprefix("N20x"))


def _reduce_prior_production(repo_root: Path) -> dict[str, Any]:
    base = (
        repo_root
        / "00_WORKSPACE/LARGE_RESULTS/classA_final_production_outputs/"
        "production_10sample_v3_fixed_nx20_ny20_30_40_50_60/01_bulk_width_gate"
    )
    sources: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    sensitivity_trajectory_rows: list[dict[str, Any]] = []
    compact_state_checks: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []
    for archive_path in sorted(base.glob("*.tar.gz")):
        receipt_path = archive_path.with_name(archive_path.name + ".receipt.json")
        receipt_source = _source_row(
            repo_root, receipt_path, "prior-production archive receipt"
        )
        receipt = _json_load(receipt_path)
        archive_hash = sha256_file(archive_path)
        if int(receipt["archive_bytes"]) != archive_path.stat().st_size:
            raise ValueError(f"Receipt size mismatch: {archive_path}")
        if receipt["archive_sha256"] != archive_hash:
            raise ValueError(f"Receipt hash mismatch: {archive_path}")
        archive_source = {
            "path": archive_path.relative_to(repo_root).as_posix(),
            "role": "prior-production W1 immutable archive",
            "bytes": archive_path.stat().st_size,
            "sha256": archive_hash,
        }
        sources.extend([archive_source, receipt_source])
        manifest, selected_payload = _stream_production_archive(archive_path)
        case_id = manifest["case_id"]
        construction, matched_trivial, ny = _case_kind(case_id)
        sample_indices = [int(value) for value in manifest["global_sample_indices"]]
        archive_rows.append(
            {
                "archive": archive_path.name,
                "case_id": case_id,
                "shard_index": int(manifest["shard_index"]),
                "global_sample_indices": sample_indices,
            }
        )
        with np.load(io.BytesIO(selected_payload), allow_pickle=False) as selected:
            configuration = json.loads(str(selected["config_json"].item()))
            model = configuration["case"]["model"]
            run = configuration["case"]["run"]
            archive_source["metadata"] = {
                "case_id": case_id,
                "shard_index": int(manifest["shard_index"]),
                "sample_indices": sample_indices,
                "dtype": model["dtype"],
                "engine_sha256": configuration["canonical_engine_sha256"],
                "initialization": model["init_mode"],
                "sequence": run["sequence"],
                "shard_generator_seed": configuration["shard_generator_seed"],
                "covariance_convention": (
                    "stored compact strip entropy/FCS observables; transient covariance "
                    "not archived"
                ),
                "selected_observables_schema": str(selected["schema"].item()),
            }
            if (
                int(model["Nx"]) != 20
                or int(model["Ny"]) != ny
                or str(model["dtype"]) != "complex128"
                or str(run["sequence"]) != "random"
                or not bool(run["perfect_correction"])
            ):
                raise ValueError(f"Prior-production contract mismatch for {case_id}")
            final_cycle = 2 * ny
            entropy_key = f"strip_entropy_cycle_{final_cycle:04d}"
            charge_key = f"strip_charge_k2_cycle_{final_cycle:04d}"
            entropy = np.asarray(selected[entropy_key], dtype=float)
            charge = np.asarray(selected[charge_key], dtype=float)
            if entropy.shape[0] != len(sample_indices) or charge.shape != entropy.shape:
                raise ValueError(f"Unexpected compact array shape in {archive_path}")
            ay = np.arange(entropy.shape[1], dtype=int)
            cycles = np.asarray(selected["cycles"], dtype=int)
            final_observation_index = int(np.flatnonzero(cycles == final_cycle)[0])
            total_entropy = np.asarray(selected["total_entropy"], dtype=float)[
                :, final_observation_index
            ]
            total_charge_variance = np.asarray(
                selected["total_charge_variance"], dtype=float
            )[:, final_observation_index]
            for local_index, global_index in enumerate(sample_indices):
                compact_state_checks.append(
                    {
                        "case_id": case_id,
                        "sample_index": global_index,
                        "cycle": final_cycle,
                        "total_entropy": float(total_entropy[local_index]),
                        "total_charge_variance": float(
                            total_charge_variance[local_index]
                        ),
                        "pass": abs(float(total_entropy[local_index])) < 1e-7
                        and abs(float(total_charge_variance[local_index])) < 1e-7,
                    }
                )
                for window in _fit_windows(ny):
                    fit_mask = (
                        (ay >= int(window["Ay_fit_min"]))
                        & (ay <= int(window["Ay_fit_max"]))
                    )
                    x_log = _chord_coordinate(ny, ay[fit_mask])
                    y_entropy = entropy[local_index, fit_mask]
                    y_charge = charge[local_index, fit_mask]
                    entropy_log_aic, entropy_fit = _linear_model_aic(
                        x_log, y_entropy
                    )
                    charge_log_aic, charge_fit = _linear_model_aic(x_log, y_charge)
                    entropy_linear_aic, _ = _linear_model_aic(
                        ay[fit_mask], y_entropy
                    )
                    charge_linear_aic, _ = _linear_model_aic(
                        ay[fit_mask], y_charge
                    )
                    entropy_constant_aic = _constant_model_aic(y_entropy)
                    charge_constant_aic = _constant_model_aic(y_charge)
                    converted = coefficient_estimates(
                        entropy_fit["slope"], charge_fit["slope"], 1
                    )
                    sensitivity_row = {
                        "case_id": case_id,
                        "construction": construction,
                        "matched_trivial": matched_trivial,
                        "Nx": 20,
                        "Ny": ny,
                        "cycle": final_cycle,
                        "sample_index": global_index,
                        **window,
                        "c_1": converted["c_wall"],
                        "k_wall": converted["k_wall"],
                        "delta_1": converted["delta_n"],
                        "normalized_delta_1": converted["normalized_delta"],
                        "entropy_r2": entropy_fit["r2"],
                        "charge_r2": charge_fit["r2"],
                        "entropy_delta_aic_linear_minus_log": entropy_linear_aic
                        - entropy_log_aic,
                        "charge_delta_aic_linear_minus_log": charge_linear_aic
                        - charge_log_aic,
                        "entropy_delta_aic_constant_minus_log": entropy_constant_aic
                        - entropy_log_aic,
                        "charge_delta_aic_constant_minus_log": charge_constant_aic
                        - charge_log_aic,
                        "engine_sha256": configuration["canonical_engine_sha256"],
                        "archive": archive_path.name,
                    }
                    sensitivity_trajectory_rows.append(sensitivity_row)
                    if window["window"] == "primary":
                        trajectory_rows.append(dict(sensitivity_row))

    case_ids = sorted({row["case_id"] for row in trajectory_rows})
    summary_rows: list[dict[str, Any]] = []
    sensitivity_summary_rows: list[dict[str, Any]] = []
    bootstrap_details: list[dict[str, Any]] = []
    accepted_trajectories: list[dict[str, Any]] = []
    accepted_sensitivity_trajectories: list[dict[str, Any]] = []
    for case_id in case_ids:
        selected = [row for row in trajectory_rows if row["case_id"] == case_id]
        indices = sorted(row["sample_index"] for row in selected)
        if indices != list(range(10)):
            exclusions.append(
                {
                    "case_id": case_id,
                    "reason": "incomplete case; requires ten unique trajectory slots",
                    "observed_sample_indices": indices,
                }
            )
            continue
        accepted_trajectories.extend(selected)
        accepted_sensitivity_trajectories.extend(
            row
            for row in sensitivity_trajectory_rows
            if row["case_id"] == case_id
        )
        first = selected[0]
        c_mean, c_sem = _mean_sem(row["c_1"] for row in selected)
        k_mean, k_sem = _mean_sem(row["k_wall"] for row in selected)
        delta_mean, delta_sem = _mean_sem(row["normalized_delta_1"] for row in selected)
        for window in _fit_windows(first["Ny"]):
            selected_window = [
                row
                for row in sensitivity_trajectory_rows
                if row["case_id"] == case_id
                and row["window"] == window["window"]
            ]
            labels = ("c_1", "k_wall", "normalized_delta_1")
            matrix = np.asarray(
                [[row[label] for label in labels] for row in selected_window],
                dtype=float,
            )
            bootstrap = _bootstrap_joint(
                matrix, labels, f"prior:{case_id}:{window['window']}"
            )
            bootstrap_details.append({"case_id": case_id, **window, **bootstrap})
            sensitivity_summary = {
                "case_id": case_id,
                "construction": first["construction"],
                "matched_trivial": first["matched_trivial"],
                "Nx": 20,
                "Ny": first["Ny"],
                "cycle": first["cycle"],
                "trajectories": 10,
                **window,
                "bootstrap_seed": bootstrap["seed"],
                "bootstrap_replicates": bootstrap["replicates"],
                "mean_entropy_r2": float(
                    np.mean([row["entropy_r2"] for row in selected_window])
                ),
                "mean_charge_r2": float(
                    np.mean([row["charge_r2"] for row in selected_window])
                ),
                "mean_entropy_delta_aic_linear_minus_log": float(
                    np.mean(
                        [
                            row["entropy_delta_aic_linear_minus_log"]
                            for row in selected_window
                        ]
                    )
                ),
                "mean_charge_delta_aic_linear_minus_log": float(
                    np.mean(
                        [
                            row["charge_delta_aic_linear_minus_log"]
                            for row in selected_window
                        ]
                    )
                ),
                "mean_entropy_delta_aic_constant_minus_log": float(
                    np.mean(
                        [
                            row["entropy_delta_aic_constant_minus_log"]
                            for row in selected_window
                        ]
                    )
                ),
                "mean_charge_delta_aic_constant_minus_log": float(
                    np.mean(
                        [
                            row["charge_delta_aic_constant_minus_log"]
                            for row in selected_window
                        ]
                    )
                ),
                "engine_sha256": first["engine_sha256"],
            }
            for label in labels:
                sensitivity_summary[label] = bootstrap["mean"][label]
                sensitivity_summary[f"{label}_ci_low"] = bootstrap["ci_low"][label]
                sensitivity_summary[f"{label}_ci_high"] = bootstrap["ci_high"][label]
            if first["matched_trivial"]:
                sensitivity_summary["trivial_abs_ci_bound"] = 1e-5
                sensitivity_summary["trivial_control_gate_pass"] = all(
                    max(
                        abs(float(sensitivity_summary[f"{label}_ci_low"])),
                        abs(float(sensitivity_summary[f"{label}_ci_high"])),
                    )
                    < 1e-5
                    for label in ("c_1", "k_wall")
                )
            sensitivity_summary_rows.append(sensitivity_summary)

        primary = next(
            row
            for row in sensitivity_summary_rows
            if row["case_id"] == case_id and row["window"] == "primary"
        )
        summary_rows.append(
            {
                **primary,
                "c_1_sem": c_sem,
                "k_wall_sem": k_sem,
                "normalized_delta_1_sem": delta_sem,
            }
        )
    summary_rows.sort(key=lambda row: (row["Ny"], row["construction"], row["matched_trivial"]))
    if len(summary_rows) != 16 or len(accepted_trajectories) != 160:
        raise AssertionError(
            f"Expected 16 complete W1 cases/160 trajectories, got "
            f"{len(summary_rows)}/{len(accepted_trajectories)}"
        )
    if not all(row["pass"] for row in compact_state_checks):
        raise AssertionError("Prior-production compact purity checks failed")
    trivial_gates = [
        row
        for row in sensitivity_summary_rows
        if row["matched_trivial"]
    ]
    if not all(row["trivial_control_gate_pass"] for row in trivial_gates):
        raise AssertionError("Matched-trivial confidence interval gate failed")
    return {
        "summary_rows": summary_rows,
        "trajectory_rows": accepted_trajectories,
        "sensitivity_trajectory_rows": accepted_sensitivity_trajectories,
        "sensitivity_summary_rows": sensitivity_summary_rows,
        "bootstrap_details": bootstrap_details,
        "compact_state_checks": compact_state_checks,
        "archive_rows": archive_rows,
        "exclusions": exclusions,
        "sources": sources,
    }


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["CMU Sans Serif", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.75,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    return plt


def _panel(ax: Any, label: str, title: str | None = None) -> None:
    ax.text(
        -0.14,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )
    if title:
        ax.set_title(title, loc="left", pad=4)


def _save_figure(fig: Any, stem: Path) -> list[Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(
        pdf,
        metadata={
            "Title": TITLE,
            "Author": "Bhuiyan, Pan, and Jian",
            "Creator": "build_validation_figures.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        png,
        dpi=300,
        metadata={"Title": TITLE, "Software": "build_validation_figures.py"},
    )
    return [pdf, png]


def _figure_geometry_pipeline(plt: Any, output: Path) -> list[Path]:
    from matplotlib.patches import FancyArrowPatch, Rectangle

    red, blue, green = "#c53b32", "#3569a8", "#3a8c5b"
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), gridspec_kw={"width_ratios": [1, 1.55]})
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.add_patch(Rectangle((0.8, 0.8), 8.4, 8.4, facecolor="white", edgecolor="black", lw=0.8))
    ax.add_patch(Rectangle((3.0, 0.8), 4.0, 8.4, facecolor="#e9f0f8", edgecolor="none"))
    for wall_x, color in ((3.0, red), (7.0, blue)):
        ax.plot([wall_x, wall_x], [0.8, 9.2], color=color, lw=2.0)
    ax.add_patch(Rectangle((0.8, 3.0), 8.4, 3.9, fill=False, edgecolor=green, lw=1.4, ls="--"))
    ax.annotate(r"interval $A_y$", (9.35, 5.0), color=green, fontsize=8, rotation=90, va="center")
    ax.annotate("topological slab", (5.0, 8.5), ha="center", fontsize=7)
    ax.annotate("trivial", (1.9, 8.5), ha="center", fontsize=7)
    ax.annotate("trivial", (8.1, 8.5), ha="center", fontsize=7)
    ax.annotate(
        r"$y\sim y+N_y$",
        xy=(0.35, 1.0),
        xytext=(0.35, 9.0),
        arrowprops={"arrowstyle": "<->", "lw": 0.8},
        fontsize=7,
        ha="center",
    )
    ax.set_axis_off()
    _panel(ax, "a", "two-wall periodic strip")

    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.set_axis_off()
    boxes = [
        (0.2, 2.55, 2.0, 1.25, r"$G_{\xi,A}$"),
        (3.0, 2.55, 2.0, 1.25, r"$\{\nu_a\}$"),
        (5.8, 4.35, 2.35, 1.25, r"$S_{n,\xi}(A)$"),
        (5.8, 0.75, 2.35, 1.25, r"$\kappa_{j,\xi}(A)$"),
        (8.95, 2.55, 2.6, 1.25, r"$c_{w,n},\ k_w,\ \delta_n$"),
    ]
    for x0, y0, width, height, text in boxes:
        ax.add_patch(Rectangle((x0, y0), width, height, facecolor="white", edgecolor="black", lw=0.8))
        ax.text(x0 + width / 2, y0 + height / 2, text, ha="center", va="center", fontsize=9)
    arrows = [
        ((2.2, 3.18), (3.0, 3.18)),
        ((5.0, 3.18), (5.8, 4.98)),
        ((5.0, 3.18), (5.8, 1.38)),
        ((8.15, 4.98), (9.0, 3.55)),
        ((8.15, 1.38), (9.0, 2.82)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=9, lw=0.8, color="black"))
    ax.text(7.0, 6.25, "trajectory first; origins internal", ha="center", fontsize=7)
    ax.text(10.25, 1.15, "paired slopes", ha="center", fontsize=7)
    _panel(ax, "b", "observable and validation pipeline")
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.05, top=0.9, wspace=0.2)
    paths = _save_figure(fig, output / "figure_01_geometry_pipeline")
    plt.close(fig)
    return paths


def _figure_modular_kernels(plt: Any, output: Path) -> list[Path]:
    red, green, blue, black = "#c53b32", "#3a8c5b", "#3569a8", "#222222"
    epsilon = np.linspace(-8, 8, 1001)
    occupation = 1 / (1 + np.exp(epsilon))
    fig, ax = plt.subplots(figsize=(3.375, 2.55))
    styles = {
        1: (red, "^", ":"),
        2: (green, "s", "--"),
        3: (blue, "o", "-"),
    }
    for order in RENyi_ORDERS:
        values = np.array(
            [entropy_renyi(np.asarray([value]), order) for value in occupation]
        )
        color, marker, linestyle = styles[order]
        ax.plot(
            epsilon,
            values,
            color=color,
            ls=linestyle,
            lw=1.15,
            marker=marker,
            markevery=125,
            ms=3.0,
            mfc="white",
            label=rf"$s_{{{order}}}(\epsilon)$",
        )
    susceptibility = susceptibility_kernel(epsilon)
    ax.plot(
        epsilon,
        susceptibility,
        color=black,
        ls="-.",
        lw=1.0,
        label=r"$v(\epsilon)$",
    )
    ax.axvline(0, color="0.55", ls="--", lw=0.7)
    ax.set(
        xlabel=r"modular energy $\epsilon=\log[(1-\nu)/\nu]$",
        ylabel="single-mode kernel",
        xlim=(-8, 8),
        ylim=(0, 0.75),
    )
    ax.legend(frameon=False, ncol=2, loc="upper right")
    _panel(ax, "a", "shared susceptibility spectrum")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.2, top=0.91)
    paths = _save_figure(fig, output / "figure_02_modular_kernels")
    plt.close(fig)
    return paths


def _figure_exact_b0(plt: Any, output: Path, b0: Mapping[str, Any]) -> list[Path]:
    red, green, blue, black = "#c53b32", "#3a8c5b", "#3569a8", "#222222"
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55))
    coupled = [row for row in b0["curves"] if row["construction"] == "coupled" and row["Ay"] >= FIT_AY_MIN]
    x = np.asarray([row["log_chord"] for row in coupled], dtype=float)
    for order, color, marker, linestyle in (
        (1, red, "^", ":"),
        (2, green, "s", "--"),
        (3, blue, "o", "-"),
    ):
        y = np.asarray([row[f"S_{order}"] for row in coupled], dtype=float)
        fit = _linear_fit(x, y)
        axes[0].plot(x, y, linestyle="none", marker=marker, ms=3.0, mfc="white", color=color, label=rf"$S_{order}$")
        axes[0].plot(x, float(fit["slope"]) * x + float(fit["intercept"]), color=color, ls=linestyle, lw=1.0)
    axes[0].set(xlabel=r"$\log d(A_y)$", ylabel=r"$S_n(A_y)$")
    axes[0].legend(frameon=False)
    _panel(axes[0], "a", "exact coupled wall")

    for construction, color, marker, label in (
        ("coupled", red, "^", "coupled"),
        ("hard_exterior", blue, "o", "hard exterior"),
    ):
        selected = [row for row in b0["curves"] if row["construction"] == construction and row["Ay"] >= FIT_AY_MIN]
        xx = np.asarray([row["log_chord"] for row in selected], dtype=float)
        yy = np.asarray([row["charge_variance"] for row in selected], dtype=float)
        fit = _linear_fit(xx, yy)
        axes[1].plot(xx, yy, linestyle="none", marker=marker, ms=3.0, mfc="white", color=color, label=label)
        axes[1].plot(xx, float(fit["slope"]) * xx + float(fit["intercept"]), color=color, lw=0.9)
    axes[1].set(xlabel=r"$\log d(A_y)$", ylabel=r"$F_A$")
    axes[1].legend(frameon=False)
    _panel(axes[1], "b", "level from charge FCS")

    labels = [r"$n=1$", r"$n=2$", r"$n=3$"]
    positions = np.arange(3)
    for offset, construction, color, marker, label in (
        (-0.09, "coupled", red, "^", "coupled"),
        (0.09, "hard_exterior", blue, "o", "hard exterior"),
    ):
        row = next(item for item in b0["rows"] if item["construction"] == construction)
        values = [
            row["normalized_delta_1"],
            row["normalized_delta_2"],
            row["normalized_delta_3"],
        ]
        axes[2].plot(positions + offset, values, linestyle="none", marker=marker, ms=4.0, mfc="white", color=color, label=label)
    axes[2].axhline(0, color=black, ls="--", lw=0.8)
    axes[2].set(xticks=positions, xticklabels=labels, ylabel=r"$c_{w,n}-k_w$", ylim=(-0.0065, 0.003))
    axes[2].legend(frameon=False, loc="lower left")
    _panel(axes[2], "c", "paired null test")
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.2, top=0.89, wspace=0.42)
    paths = _save_figure(fig, output / "figure_03_exact_b0_validation")
    plt.close(fig)
    return paths


def _summary_curve(
    rows: Sequence[Mapping[str, Any]], key: str, seed_label: str
) -> list[dict[str, float]]:
    output = []
    for ay in sorted({int(row["Ay"]) for row in rows}):
        selected = [float(row[key]) for row in rows if int(row["Ay"]) == ay]
        mean, sem = _mean_sem(selected)
        bootstrap = _bootstrap_joint(
            np.asarray(selected, dtype=float)[:, None],
            ("value",),
            f"curve:{seed_label}:Ay{ay}",
        )
        coordinate = next(float(row["log_chord"]) for row in rows if int(row["Ay"]) == ay)
        output.append(
            {
                "Ay": ay,
                "log_chord": coordinate,
                "mean": mean,
                "sem": sem,
                "ci_low": bootstrap["ci_low"]["value"],
                "ci_high": bootstrap["ci_high"]["value"],
            }
        )
    return output


def _figure_stochastic(
    plt: Any,
    output: Path,
    result21: Mapping[str, Any],
    prior: Mapping[str, Any],
) -> list[Path]:
    red, green, blue, black = "#c53b32", "#3a8c5b", "#3569a8", "#222222"
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.15))
    representative_rows = [
        row
        for row in result21["charge_curve_rows"]
        if row["Ny"] == 40 and row["nshell"] == 1
    ]
    curve = _summary_curve(
        representative_rows, "charge_variance", "result21:N40:nsh1"
    )
    x = np.asarray([row["log_chord"] for row in curve])
    y = np.asarray([row["mean"] for row in curve])
    lower = np.asarray([row["mean"] - row["ci_low"] for row in curve])
    upper = np.asarray([row["ci_high"] - row["mean"] for row in curve])
    ay = np.asarray([row["Ay"] for row in curve])
    mask = ay >= FIT_AY_MIN
    fit = _linear_fit(x[mask], y[mask])
    axes[0, 0].errorbar(x, y, yerr=np.vstack((lower, upper)), linestyle="none", marker="o", mfc="white", ms=3.0, color=blue, capsize=1.4, label="trajectory mean; 95% CI")
    axes[0, 0].axvspan(x[mask].min(), x[mask].max(), color="0.92", zorder=0, label=r"$A_y\geq8$")
    axes[0, 0].plot(x, float(fit["slope"]) * x + float(fit["intercept"]), color=black, ls="--", lw=0.9, label=rf"$k={np.pi**2*float(fit['slope']):.3f}$")
    axes[0, 0].set(xlabel=r"$\log d(A_y)$", ylabel=r"$\overline{F_A}$")
    axes[0, 0].legend(frameon=False, loc="upper left")
    _panel(axes[0, 0], "a", r"$N_y=40$, $n_{\rm shell}=1$")

    styles = {1: (red, "^"), 2: (blue, "o")}
    for nshell in (1, 2):
        selected = sorted((row for row in result21["summary_rows"] if row["nshell"] == nshell), key=lambda row: row["Ny"])
        color, marker = styles[nshell]
        use_raw = result21["stochastic_renyi_computed"]
        x_values = [row["c_1"] if use_raw else row["c_1_archived"] for row in selected]
        if use_raw:
            xerr = np.vstack(
                (
                    [row["c_1"] - row["c_1_ci_low"] for row in selected],
                    [row["c_1_ci_high"] - row["c_1"] for row in selected],
                )
            )
        else:
            xerr = [row["c_1_archived_regression_error"] for row in selected]
        yerr = np.vstack(
            (
                [row["k_wall"] - row["k_wall_ci_low"] for row in selected],
                [row["k_wall_ci_high"] - row["k_wall"] for row in selected],
            )
        )
        axes[0, 1].errorbar(
            x_values,
            [row["k_wall"] for row in selected],
            xerr=xerr,
            yerr=yerr,
            linestyle="none",
            marker=marker,
            color=color,
            mfc="white",
            capsize=2,
            label=rf"$n_{{\rm shell}}={nshell}$",
        )
        for row in selected:
            x_value = row["c_1"] if use_raw else row["c_1_archived"]
            axes[0, 1].annotate(str(row["Ny"]), (x_value, row["k_wall"]), xytext=(3, -6), textcoords="offset points", fontsize=6)
    axes[0, 1].plot([0.98, 1.15], [0.98, 1.15], color=black, ls="--", lw=0.8)
    axes[0, 1].set(xlabel=r"entropy coefficient $c_1$", ylabel=r"charge level $k$", xlim=(0.98, 1.15), ylim=(0.98, 1.15))
    axes[0, 1].legend(frameon=False)
    _panel(axes[0, 1], "b", "same-ensemble coefficient test")

    topological = [row for row in prior["summary_rows"] if not row["matched_trivial"]]
    for construction, color, marker, linestyle, label in (
        ("explicit_interface", red, "^", ":", "explicit interface"),
        ("support_terminated", blue, "o", "-", "support terminated"),
    ):
        selected = sorted((row for row in topological if row["construction"] == construction), key=lambda row: row["Ny"])
        c_err = np.vstack((
            [row["c_1"] - row["c_1_ci_low"] for row in selected],
            [row["c_1_ci_high"] - row["c_1"] for row in selected],
        ))
        k_err = np.vstack((
            [row["k_wall"] - row["k_wall_ci_low"] for row in selected],
            [row["k_wall_ci_high"] - row["k_wall"] for row in selected],
        ))
        axes[1, 0].errorbar([row["Ny"] for row in selected], [row["c_1"] for row in selected], yerr=c_err, color=color, marker=marker, mfc="white", ls=linestyle, capsize=2, label=label + r" $c_1$")
        axes[1, 0].errorbar([row["Ny"] for row in selected], [row["k_wall"] for row in selected], yerr=k_err, color=color, marker=marker, mfc=color, ls="--", capsize=2, label=label + r" $k$")
    controls = [row for row in prior["summary_rows"] if row["matched_trivial"]]
    for construction, marker, offset in (("explicit_interface", "x", -0.35), ("support_terminated", "+", 0.35)):
        selected = sorted((row for row in controls if row["construction"] == construction), key=lambda row: row["Ny"])
        axes[1, 0].plot([row["Ny"] + offset for row in selected], [row["c_1"] for row in selected], linestyle="none", marker=marker, color="0.35", ms=4.0, label=("matched trivial controls" if construction == "explicit_interface" else None))
    axes[1, 0].axhline(1, color=black, ls="--", lw=0.8)
    axes[1, 0].set(xlabel=r"circumference $N_y$", ylabel="per-wall coefficient", xticks=[20, 30, 40, 50], ylim=(-0.08, 1.35))
    axes[1, 0].legend(frameon=False, ncol=2, fontsize=5.7)
    _panel(axes[1, 0], "c", r"prior production, $S=10$")

    if result21["paired_fit_rows"]:
        values = []
        labels = []
        colors = []
        for order, color in ((1, red), (2, green), (3, blue)):
            values.append([row[f"normalized_delta_{order}"] for row in result21["paired_fit_rows"]])
            labels.append(rf"$c_{order}-k$")
            colors.append(color)
        box = axes[1, 1].boxplot(values, labels=labels, patch_artist=True, showfliers=True, widths=0.55, medianprops={"color": black, "lw": 0.9}, whiskerprops={"lw": 0.8}, capprops={"lw": 0.8}, boxprops={"lw": 0.8})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        axes[1, 1].axhline(0, color=black, ls="--", lw=0.8)
        axes[1, 1].set(ylabel="paired normalized residual")
        title = "40 trajectory-resolved pairs"
    else:
        histogram = [row["k_wall"] for row in result21["charge_fit_rows"] if row["Ny"] == 40]
        axes[1, 1].hist(histogram, bins=np.linspace(0.75, 1.55, 11), color=blue, edgecolor=black, alpha=0.55)
        axes[1, 1].axvline(1, color=black, ls="--", lw=0.8)
        axes[1, 1].set(xlabel=r"trajectory $k_\xi$", ylabel="count")
        title = "stochastic Rényi reduction skipped"
    _panel(axes[1, 1], "d", title)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.1, top=0.94, wspace=0.35, hspace=0.38)
    paths = _save_figure(fig, output / "figure_04_stochastic_legacy_validation")
    plt.close(fig)
    return paths


def _scientific_claim_status(
    result21: Mapping[str, Any], prior: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate claim licensing separately from implementation integrity.

    A primary-window paired 95% interval that excludes zero blocks the strict
    minimal-U(1)_1 license.  Endpoint-deletion rows are retained as sensitivity
    evidence but are not silently promoted to independent samples.
    """

    evaluated: list[dict[str, Any]] = []
    if result21["stochastic_renyi_computed"]:
        result21_rows = result21["sensitivity_summary_rows"]
    else:
        result21_rows = []
    for row in result21_rows:
        for order in RENyi_ORDERS:
            label = f"normalized_delta_{order}"
            low = float(row[f"{label}_ci_low"])
            high = float(row[f"{label}_ci_high"])
            evaluated.append(
                {
                    "dataset": "result21",
                    "case_id": row["case_id"],
                    "Nx": int(row["Nx"]),
                    "Ny": int(row["Ny"]),
                    "nshell": int(row["nshell"]),
                    "window": row["window"],
                    "Ay_fit_min": int(row["Ay_fit_min"]),
                    "Ay_fit_max": int(row["Ay_fit_max"]),
                    "renyi_order": order,
                    "normalized_delta": float(row[label]),
                    "ci_low": low,
                    "ci_high": high,
                    "confidence": BOOTSTRAP_CONFIDENCE,
                    "bootstrap_seed": int(row["bootstrap_seed"]),
                    "bootstrap_replicates": int(row["bootstrap_replicates"]),
                    "zero_excluded": bool(low > 0 or high < 0),
                }
            )
    for row in prior["sensitivity_summary_rows"]:
        if row["matched_trivial"]:
            continue
        label = "normalized_delta_1"
        low = float(row[f"{label}_ci_low"])
        high = float(row[f"{label}_ci_high"])
        evaluated.append(
            {
                "dataset": "prior_production",
                "case_id": row["case_id"],
                "construction": row["construction"],
                "Nx": int(row["Nx"]),
                "Ny": int(row["Ny"]),
                "window": row["window"],
                "Ay_fit_min": int(row["Ay_fit_min"]),
                "Ay_fit_max": int(row["Ay_fit_max"]),
                "renyi_order": 1,
                "normalized_delta": float(row[label]),
                "ci_low": low,
                "ci_high": high,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "bootstrap_seed": int(row["bootstrap_seed"]),
                "bootstrap_replicates": int(row["bootstrap_replicates"]),
                "zero_excluded": bool(low > 0 or high < 0),
            }
        )
    evaluated.sort(
        key=lambda row: (
            row["dataset"],
            row["case_id"],
            row["window"],
            row["renyi_order"],
        )
    )
    primary = [row for row in evaluated if row["window"] == "primary"]
    offending_primary = [row for row in primary if row["zero_excluded"]]
    offending_sensitivity = [
        row
        for row in evaluated
        if row["window"] != "primary" and row["zero_excluded"]
    ]
    minimal_status = (
        "blocked_by_paired_null"
        if offending_primary
        else "supported_by_paired_null"
    )
    trivial_primary_rows = [
        row
        for row in prior["sensitivity_summary_rows"]
        if row["matched_trivial"] and row["window"] == "primary"
    ]
    trivial_model_preference_rows = [
        {
            "case_id": row["case_id"],
            "construction": row["construction"],
            "Nx": int(row["Nx"]),
            "Ny": int(row["Ny"]),
            "window": row["window"],
            "Ay_fit_min": int(row["Ay_fit_min"]),
            "Ay_fit_max": int(row["Ay_fit_max"]),
            "entropy_constant_minus_log_aic": float(
                row["mean_entropy_delta_aic_constant_minus_log"]
            ),
            "charge_constant_minus_log_aic": float(
                row["mean_charge_delta_aic_constant_minus_log"]
            ),
            "c_1_ci_low": float(row["c_1_ci_low"]),
            "c_1_ci_high": float(row["c_1_ci_high"]),
            "k_wall_ci_low": float(row["k_wall_ci_low"]),
            "k_wall_ci_high": float(row["k_wall_ci_high"]),
        }
        for row in trivial_primary_rows
        if row["mean_entropy_delta_aic_constant_minus_log"] > 0
        or row["mean_charge_delta_aic_constant_minus_log"] > 0
    ]
    coefficient_null_pass = all(
        row["trivial_control_gate_pass"]
        for row in prior["sensitivity_summary_rows"]
        if row["matched_trivial"]
    )
    literal_model_preference_pass = not trivial_model_preference_rows
    all_scientific_criteria_met = (
        minimal_status == "supported_by_paired_null"
        and coefficient_null_pass
        and literal_model_preference_pass
    )
    return {
        "schema_version": "scientific-claim-status-v1",
        "separation_from_pipeline_integrity": (
            "These are scientific claim licenses. A blocked claim is a valid "
            "analysis result and does not imply an implementation-gate failure."
        ),
        "overall_scientific_acceptance": {
            "status": (
                "all_prespecified_criteria_met"
                if all_scientific_criteria_met
                else "qualified_not_met"
            ),
            "all_prespecified_criteria_met": all_scientific_criteria_met,
            "reason": (
                "strict paired minimal-U(1)_1 null and literal trivial-control "
                "model-preference criteria are not both met"
                if not all_scientific_criteria_met
                else "all prespecified scientific criteria are met"
            ),
        },
        "checks": {
            "trivial_control_coefficient_null": {
                "status": "pass" if coefficient_null_pass else "fail",
                "criterion": (
                    "all matched-trivial c1 and k 95% bootstrap intervals remain "
                    "inside the declared absolute 1e-5 numerical-null band"
                ),
            },
            "trivial_control_log_model_preference": {
                "status": (
                    "pass"
                    if literal_model_preference_pass
                    else "not_met_numerical_floor"
                ),
                "criterion": (
                    "constant-minus-log AIC must be nonpositive for both entropy "
                    "and charge in every primary matched-trivial case"
                ),
                "interpretation": (
                    "positive AIC preferences occur on curves whose fitted "
                    "coefficients are at the numerical floor; the literal model "
                    "criterion is still not met"
                ),
                "offending_rows": trivial_model_preference_rows,
            },
        },
        "claims": {
            "level_one_u1_current_sector": {
                "status": "supported_near_one_with_finite_size_limitations",
                "basis": (
                    "B0, Result 21, and prior-production conditional quantum "
                    "charge slopes give k_w near one; no thermodynamic extrapolation."
                ),
            },
            "central_charge_like_entropy": {
                "status": "supported_near_one_with_finite_size_limitations",
                "basis": (
                    "Matched S1 and Result-21 S2/S3 logarithmic coefficients are "
                    "near one on the archived finite systems."
                ),
            },
            "minimal_u1_1": {
                "status": minimal_status,
                "decision_rule": (
                    "block if any prespecified primary-window paired whole-trajectory "
                    "95% bootstrap CI for normalized_delta_n=c_w,n-k_w excludes zero"
                ),
                "primary_tests_evaluated": len(primary),
                "offending_primary_count": len(offending_primary),
                "offending_primary_rows": offending_primary,
                "sensitivity_tests_evaluated": len(evaluated) - len(primary),
                "offending_sensitivity_count": len(offending_sensitivity),
                "offending_sensitivity_rows": offending_sensitivity,
                "preserved_evidence": (
                    "The block preserves the separately reported near-one entropy "
                    "and current-level coefficients."
                ),
            },
            "chirality": {
                "status": "not_evaluated_by_static_observables",
                "basis": "requires a separate signed packet, response, or flux test",
            },
        },
    }


def _write_tables(
    table_dir: Path,
    b0: Mapping[str, Any],
    result21: Mapping[str, Any],
    prior: Mapping[str, Any],
    scientific_claim_status: Mapping[str, Any],
) -> list[Path]:
    outputs: list[Path] = []
    table_map = {
        "exact_b0_regression.csv": b0["rows"],
        "exact_b0_sensitivity.csv": b0["sensitivity_rows"],
        "exact_b0_curves.csv": b0["curves"],
        "result21_regression.csv": result21["summary_rows"],
        "result21_sensitivity_summary.csv": result21["sensitivity_summary_rows"],
        "result21_sensitivity_trajectory_fits.csv": result21[
            "sensitivity_fit_rows"
        ],
        "result21_charge_curves.csv": result21["charge_curve_rows"],
        "result21_trajectory_charge_fits.csv": result21["charge_fit_rows"],
        "result21_formula_checks.csv": result21["formula_checks"],
        "result21_state_checks.csv": result21["state_checks"],
        "prior_production_regression.csv": prior["summary_rows"],
        "prior_production_sensitivity_summary.csv": prior[
            "sensitivity_summary_rows"
        ],
        "prior_production_sensitivity_trajectory_fits.csv": prior[
            "sensitivity_trajectory_rows"
        ],
        "prior_production_state_checks.csv": prior["compact_state_checks"],
        "prior_production_trajectory_fits.csv": prior["trajectory_rows"],
        "prior_production_archive_index.csv": prior["archive_rows"],
    }
    if result21["renyi_curve_rows"]:
        table_map["result21_renyi_curves.csv"] = result21["renyi_curve_rows"]
        table_map["result21_trajectory_renyi_fits.csv"] = result21["renyi_fit_rows"]
        table_map["result21_paired_fits.csv"] = result21["paired_fit_rows"]
    for filename, rows in table_map.items():
        path = table_dir / filename
        _write_csv(path, rows)
        outputs.append(path)

    exact_tex = table_dir / "exact_b0_regression.tex"
    _write_tex_table(
        exact_tex,
        (
            ("construction", "construction", "s"),
            ("k_wall", r"$k_w$", ".7f"),
            ("c_1", r"$c_{w,1}$", ".7f"),
            ("c_2", r"$c_{w,2}$", ".7f"),
            ("c_3", r"$c_{w,3}$", ".7f"),
        ),
        b0["rows"],
    )
    outputs.append(exact_tex)
    result_tex = table_dir / "result21_regression.tex"
    result_tex_rows = [
        {
            **row,
            "c_2_display": row.get("c_2"),
            "c_3_display": row.get("c_3"),
        }
        for row in result21["summary_rows"]
    ]
    _write_tex_table(
        result_tex,
        (
            ("Ny", r"$N_y$", ".0f"),
            ("nshell", r"$n_{\rm shell}$", ".0f"),
            ("k_wall", r"$k_w$", ".4f"),
            ("k_wall_sem", "SE", ".4f"),
            ("c_1_archived", r"$c_{w,1}$", ".4f"),
            ("c_2_display", r"$c_{w,2}$", ".4f"),
            ("c_3_display", r"$c_{w,3}$", ".4f"),
        ),
        result_tex_rows,
    )
    outputs.append(result_tex)
    prior_tex = table_dir / "prior_production_regression.tex"
    prior_tex_rows = [
        {
            **row,
            "construction_short": (
                "explicit" if row["construction"] == "explicit_interface" else "support"
            ),
            "sector": "control" if row["matched_trivial"] else "wall",
            "c_1_ci_half_width": 0.5
            * (float(row["c_1_ci_high"]) - float(row["c_1_ci_low"])),
            "k_wall_ci_half_width": 0.5
            * (
                float(row["k_wall_ci_high"])
                - float(row["k_wall_ci_low"])
            ),
        }
        for row in prior["summary_rows"]
    ]
    _write_tex_table(
        prior_tex,
        (
            ("construction_short", "construction", "s"),
            ("sector", "sector", "s"),
            ("Ny", r"$N_y$", ".0f"),
            ("c_1", r"$c_{w,1}$", ".3f"),
            ("c_1_ci_half_width", r"95\% CI/2", ".3f"),
            ("k_wall", r"$k_w$", ".3f"),
            ("k_wall_ci_half_width", r"95\% CI/2", ".3f"),
        ),
        prior_tex_rows,
    )
    outputs.append(prior_tex)
    validation_path = table_dir / "validation_summary.json"
    _write_json(
        validation_path,
        {
            "exact_b0": b0["validations"],
            "result21_formula_checks": result21["formula_checks"],
            "result21_state_checks": result21["state_checks"],
            "result21_renyi_spectrum_checks": result21[
                "renyi_spectrum_checks"
            ],
            "result21_bootstrap": result21["bootstrap_details"],
            "result21_expected_checks": result21["expected_checks"],
            "prior_production_bootstrap": prior["bootstrap_details"],
            "prior_production_state_checks": prior["compact_state_checks"],
            "prior_production_exclusions": prior["exclusions"],
        },
    )
    outputs.append(validation_path)
    claim_status_path = table_dir / "scientific_claim_status.json"
    _write_json(claim_status_path, scientific_claim_status)
    outputs.append(claim_status_path)
    return outputs


def _deduplicate_sources(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = str(row["path"])
        if path in deduplicated and deduplicated[path]["sha256"] != row["sha256"]:
            raise ValueError(f"Conflicting source fingerprints for {path}")
        if path not in deduplicated:
            deduplicated[path] = dict(row)
        else:
            roles = sorted(
                set(str(deduplicated[path]["role"]).split("; "))
                | set(str(row["role"]).split("; "))
            )
            deduplicated[path]["role"] = "; ".join(roles)
    return [deduplicated[key] for key in sorted(deduplicated)]


def run_analysis(
    project_root: Path | str | None = None,
    repo_root: Path | str | None = None,
    *,
    compute_stochastic_renyi: bool = True,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the complete immutable-input reduction and return its manifest."""

    script_path = Path(__file__).resolve()
    canonical_project = script_path.parent
    project = Path(project_root).resolve() if project_root else canonical_project
    if project != canonical_project:
        raise ValueError(
            "Refusing noncanonical project_root: analysis outputs may be written only "
            f"below {canonical_project}"
        )
    repo = Path(repo_root).resolve() if repo_root else _discover_repo_root(project)
    if _discover_repo_root(project) != repo:
        raise ValueError("Explicit repo_root does not match the project ancestry")
    source_roots = [(repo / relative).resolve() for relative in SOURCE_ROOTS_RELATIVE]
    overlapping = [root for root in source_roots if _paths_overlap(project, root)]
    if overlapping:
        raise ValueError(
            "Refusing output/source overlap with immutable campaign roots: "
            + ", ".join(path.as_posix() for path in overlapping)
        )
    figure_dir = project / "figures"
    table_dir = project / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    worker_count = workers if workers is not None else min(32, os.cpu_count() or 1)
    if worker_count < 1:
        raise ValueError("workers must be positive")

    print("[1/6] reducing exact B0 spectra", flush=True)
    b0 = _reduce_exact_b0(repo)
    print("[2/6] reducing Result 21 covariances", flush=True)
    result21 = _reduce_result21(repo, compute_stochastic_renyi, worker_count)
    print("[3/6] reducing compact prior-production W1 archives", flush=True)
    prior = _reduce_prior_production(repo)
    scientific_claim_status = _scientific_claim_status(result21, prior)
    print("[4/6] writing machine-readable and TeX tables", flush=True)
    table_outputs = _write_tables(
        table_dir, b0, result21, prior, scientific_claim_status
    )
    print("[5/6] rendering figures", flush=True)
    plt = _setup_matplotlib()
    figure_outputs: list[Path] = []
    figure_outputs.extend(_figure_geometry_pipeline(plt, figure_dir))
    figure_outputs.extend(_figure_modular_kernels(plt, figure_dir))
    figure_outputs.extend(_figure_exact_b0(plt, figure_dir, b0))
    figure_outputs.extend(_figure_stochastic(plt, figure_dir, result21, prior))

    all_sources = _deduplicate_sources(
        [*b0["sources"], *result21["sources"], *prior["sources"]]
    )
    output_rows = []
    for path in sorted([*table_outputs, *figure_outputs]):
        output_rows.append(
            {
                "path": path.relative_to(project).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "kind": "figure" if path.parent == figure_dir else "table",
            }
        )
    topological_primary = [
        row
        for row in prior["sensitivity_summary_rows"]
        if not row["matched_trivial"] and row["window"] == "primary"
    ]
    trivial_sensitivities = [
        row
        for row in prior["sensitivity_summary_rows"]
        if row["matched_trivial"]
    ]
    gate_ledger = [
        {
            "gate": "source_fingerprints",
            "required": True,
            "status": "pass"
            if all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in all_sources)
            else "fail",
            "detail": f"{len(all_sources)} immutable inputs hashed before numerical load",
        },
        {
            "gate": "exact_b0_regression",
            "required": True,
            "status": "pass"
            if all(row["pass"] for row in b0["validations"])
            else "fail",
            "detail": "raw NaN-padded spectra reproduce locked c1,c2,c3,k values",
        },
        {
            "gate": "fit_window_sensitivity",
            "required": True,
            "status": "pass"
            if len(b0["sensitivity_rows"]) == 6
            and len(result21["sensitivity_summary_rows"]) == 12
            and len(prior["sensitivity_summary_rows"]) == 48
            else "fail",
            "detail": "primary, drop-largest, and drop-smallest interval fits evaluated",
        },
        {
            "gate": "result21_origin_formula",
            "required": True,
            "status": "pass"
            if max(row["max_abs_error"] for row in result21["formula_checks"])
            < 1e-10
            else "fail",
            "detail": "block-correlation formula agrees with explicit restricted submatrices",
        },
        {
            "gate": "result21_purity_and_hermiticity",
            "required": True,
            "status": "pass"
            if all(row["pass"] for row in result21["state_checks"])
            else "fail",
            "detail": "all 40 cycle-50 covariance snapshots checked",
        },
        {
            "gate": "result21_locked_k_regression",
            "required": True,
            "status": "pass"
            if all(row["pass"] for row in result21["expected_checks"])
            else "fail",
            "detail": "trajectory-resolved level estimates reproduce atlas Result 21",
        },
        {
            "gate": "result21_archived_s1_reproduction",
            "required": True,
            "status": (
                "pass"
                if compute_stochastic_renyi
                and all(
                    row.get("archive_compatibility_S1_reproduction_pass", False)
                    for row in result21["summary_rows"]
                )
                else "not_evaluated"
                if not compute_stochastic_renyi
                else "fail"
            ),
            "detail": (
                "legacy 1e-12 endpoint-floor S1 reproduces the archived table at "
                "1e-10; canonical endpoint-exact offsets remain explicit"
            ),
        },
        {
            "gate": "stochastic_renyi_reduction",
            "required": True,
            "status": (
                "pass"
                if compute_stochastic_renyi
                and len(result21["paired_fit_rows"]) == 40
                else "not_evaluated"
                if not compute_stochastic_renyi
                else "fail"
            ),
            "detail": "exact all-origin S1,S2,S3 fits on the same forty trajectories",
        },
        {
            "gate": "restricted_spectrum_range",
            "required": True,
            "status": (
                "pass"
                if compute_stochastic_renyi
                and all(
                    row["outside_tolerance_values"] == 0
                    for row in result21["renyi_spectrum_checks"]
                )
                else "not_evaluated"
                if not compute_stochastic_renyi
                else "fail"
            ),
            "detail": "range excursions, applied corrections, and endpoint counts recorded",
        },
        {
            "gate": "paired_whole_trajectory_bootstrap",
            "required": True,
            "status": (
                "pass"
                if compute_stochastic_renyi
                and len(result21["bootstrap_details"]) == 12
                and len(prior["bootstrap_details"]) == 48
                else "not_evaluated"
                if not compute_stochastic_renyi
                else "fail"
            ),
            "detail": (
                f"{BOOTSTRAP_REPLICATES} fixed-seed paired resamples per case/window; "
                "joint covariance retained"
            ),
        },
        {
            "gate": "prior_production_completeness",
            "required": True,
            "status": "pass"
            if len(prior["trajectory_rows"]) == 160
            else "fail",
            "detail": "16 complete S=10 cases accepted; incomplete Ny=60 case excluded",
        },
        {
            "gate": "prior_compact_purity",
            "required": True,
            "status": "pass"
            if all(row["pass"] for row in prior["compact_state_checks"])
            else "fail",
            "detail": "stored final total entropy and total charge variance checked",
        },
        {
            "gate": "wall_log_chord_vs_constant",
            "required": True,
            "status": "pass"
            if all(
                row["mean_entropy_delta_aic_constant_minus_log"] > 0
                and row["mean_charge_delta_aic_constant_minus_log"] > 0
                for row in topological_primary
            )
            else "fail",
            "detail": "positive constant-minus-log AIC favors the log-chord wall model",
        },
        {
            "gate": "matched_trivial_null",
            "required": True,
            "status": "pass"
            if all(row["trivial_control_gate_pass"] for row in trivial_sensitivities)
            else "fail",
            "detail": "all c1 and k bootstrap intervals lie within the declared 1e-5 numerical-null band",
        },
    ]
    all_required_checks_pass = all(
        row["status"] == "pass" for row in gate_ledger if row["required"]
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "title": TITLE,
        "analysis_entry_point": script_path.relative_to(repo).as_posix(),
        "analysis_entry_point_sha256": sha256_file(script_path),
        "read_only_source_policy": (
            "All campaign inputs are opened read-only. Writes are restricted to this "
            "project's figures, tables, and analysis_manifest.json."
        ),
        "pipeline_integrity": {
            "status": "pass" if all_required_checks_pass else "fail",
            "all_required_checks_pass": all_required_checks_pass,
            "meaning": (
                "implementation, provenance, numerical, fit, and control checks; "
                "scientific claim licensing is reported separately"
            ),
        },
        "scientific_claim_status": scientific_claim_status,
        "conventions": {
            "correlation_matrix": "C=(G+I)/2 for archived Result-21 G arrays",
            "entropy_units": "natural logarithms (nats)",
            "exact_b0_estimator_reconciliation": (
                "locked B0 c_n values use the archive's explicit 1e-12 endpoint "
                "floor; endpoint-exact logaddexp values and their difference are "
                "recorded alongside them"
            ),
            "result21_estimator_reconciliation": (
                "archived c1 is validated with its legacy 1e-12 endpoint-floor "
                "estimator; endpoint-exact S1,S2,S3 define the new paired analysis"
            ),
            "geometry": "two identical wall branches in a full-x periodic-y interval",
            "chord_coordinate": "log[(Ny/pi) sin(pi Ay/Ny)]",
            "fit_window": "Ay >= 8 through Ny/2, inclusive",
            "per_wall_coefficients": {
                "c_w_n": "6*a_Sn/(1+1/n)",
                "k_w": "pi^2*a_F",
                "delta_n": "a_Sn-(pi^2/6)*(1+1/n)*a_F",
            },
            "sample_unit": "whole trajectory; y origins remain internal correlated variance-reduction samples",
            "uncertainty": (
                "95% deterministic percentile confidence interval from paired whole-"
                "trajectory bootstrap; published Result-21 SEM retained in separate fields"
            ),
            "bootstrap": {
                "base_seed": BOOTSTRAP_BASE_SEED,
                "replicates": BOOTSTRAP_REPLICATES,
                "confidence": BOOTSTRAP_CONFIDENCE,
                "seed_derivation": "first 32 bits of SHA256(base_seed:case:window)",
            },
        },
        "workloads": {
            "exact_b0": "two deterministic Nx=20, Ny=48 constructions",
            "result21": "four S=10 cases at Nx=16, Ny=30/40, cycle 50",
            "prior_production": "16 complete S=10 W1 cases (160 trajectories)",
            "stochastic_renyi_computed": compute_stochastic_renyi,
            "stochastic_renyi_workers": worker_count if compute_stochastic_renyi else 0,
        },
        "results": {
            "exact_b0": b0["rows"],
            "exact_b0_sensitivity": b0["sensitivity_rows"],
            "result21": result21["summary_rows"],
            "result21_sensitivity": result21["sensitivity_summary_rows"],
            "prior_production": prior["summary_rows"],
            "prior_production_sensitivity": prior["sensitivity_summary_rows"],
        },
        "validation": {
            "gate_ledger": gate_ledger,
            "exact_b0": b0["validations"],
            "result21_formula_checks": result21["formula_checks"],
            "result21_state_checks": result21["state_checks"],
            "result21_renyi_spectrum_checks": result21[
                "renyi_spectrum_checks"
            ],
            "result21_bootstrap": result21["bootstrap_details"],
            "result21_expected_checks": result21["expected_checks"],
            "prior_production_state_checks": prior["compact_state_checks"],
            "prior_production_bootstrap": prior["bootstrap_details"],
            "prior_production_exclusions": prior["exclusions"],
            "all_required_checks_pass": all_required_checks_pass,
        },
        "censoring_and_missingness": {
            "result21": "none; all four cases have ten trajectories",
            "prior_production": prior["exclusions"],
            "policy": "receipt-only or incomplete slots are excluded, never imputed",
        },
        "limitations": [
            "Result 21 has only two circumferences and ten trajectories per configuration.",
            "The prior-production W1 archives use an older canonical-engine hash; no cross-version pooling claim is made.",
            "The incomplete Ny=60 W1 case is provenance-checked but excluded from coefficient tables.",
            "Static entropy and charge coefficients do not determine chirality.",
            "No new circuit dynamics are run by this builder.",
            "Accepted B0 entropy targets retain their historical 1e-12 endpoint-floor convention; the endpoint-exact recomputation is separately recorded and differs only at the 1e-8 entropy level.",
        ],
        "sources": all_sources,
        "outputs": output_rows,
    }
    print("[6/6] writing provenance manifest", flush=True)
    _write_json(project / "analysis_manifest.json", manifest)
    print(
        f"done: {len(figure_outputs)} figure files, {len(table_outputs)} table files, "
        f"{len(all_sources)} immutable inputs",
        flush=True,
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--skip-stochastic-renyi",
        action="store_true",
        help="skip the costly all-origin S2/S3 reduction; all accepted S1/k products remain",
    )
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    run_analysis(
        project_root=args.project_root,
        repo_root=args.repo_root,
        compute_stochastic_renyi=not args.skip_stochastic_renyi,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
