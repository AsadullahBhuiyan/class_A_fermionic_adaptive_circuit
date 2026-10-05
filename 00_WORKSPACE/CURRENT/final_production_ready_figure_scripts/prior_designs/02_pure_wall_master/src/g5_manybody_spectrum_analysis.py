"""Exact fixed-word Gaussian-operator spectrum reconstruction.

For a trajectory word K, the identity-input replay gives
``rho = K K^dagger / Z`` and ``log Z = N_orb log(2) + log p_K``.  Natural
occupations ``nu_i`` determine ``j_i = 2 sqrt(nu_i (1-nu_i))`` and the full
normalized spectrum ``lambda_n = product_i nu_i**n_i (1-nu_i)**(1-n_i)``;
absolute singular values are ``sqrt(Z lambda_n)``.  A flip relative to the
dominant occupation has charge ``1-2*n_i`` and singular-amplitude gap
``0.5*abs(log(nu_i/(1-nu_i))) = arcosh(1/j_i)``.  The code stores these
factors and low charge-sector subset sums rather than a Fock-space matrix.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from h3_twist_observables import BranchWeightObserver
from production_runtime import (
    require_a100,
    save_npz_atomic,
    sha256_file,
    verify_archive_receipt,
    write_json_atomic,
)
from record_observables import load_ordered_record


G5_PRODUCT_SCHEMA = "G5_identity_probe_factorized_spectrum_v1"
G5_SUMMARY_SCHEMA = "G5_tangent_manybody_summary_v1"
G5_RECEIPT_SCHEMA = "G5_derived_receipt_v1"
G5_PROTOCOLS = (
    "explicit_interface",
    "explicit_interface_matched_trivial",
)
G5_SIZES = (20, 30, 40, 50, 60)
G5_SECTORS = np.arange(-3, 4, dtype=np.int8)
G5_LEVELS_PER_SECTOR = 128
G5_CAP_TOLERANCE = 1.0e-12


def _root_manifest_from_tar(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.name.lstrip("./") == "manifest.json"
        ]
        if len(members) != 1:
            raise ValueError(f"{path}: expected one root manifest, found {len(members)}")
        handle = archive.extractfile(members[0])
        if handle is None:
            raise ValueError(f"{path}: root manifest is unreadable")
        return json.loads(handle.read().decode("utf-8"))


def _safe_extract(path: Path, destination: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        root = destination.resolve()
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe archive member {member.name!r}")
        archive.extractall(destination, filter="data")


def _selected_parent(manifest: dict[str, Any]) -> bool:
    case = manifest.get("run_config", {}).get("case", {})
    model = case.get("model", {})
    case_id = str(case.get("case_id", ""))
    if case.get("campaign") != "S1_T1_B2_MASTER":
        return False
    if not case_id.startswith("MASTER_"):
        return False
    return (
        int(model.get("Nx", -1)) == 20
        and int(model.get("Ny", -1)) in G5_SIZES
        and any(case_id.endswith(protocol) for protocol in G5_PROTOCOLS)
    )


def _physical_parent_identity(manifest: dict[str, Any]) -> str:
    case = manifest["run_config"]["case"]
    run = case["run"]
    payload = {
        "case_id": case["case_id"],
        "model": case["model"],
        "run": {
            key: run.get(key)
            for key in (
                "cycles",
                "samples",
                "sequence",
                "perfect_correction",
                "postselect",
                "postselect_probability",
            )
        },
        "root_seed": manifest.get("root_seed"),
        "shard_generator_seed": manifest.get("shard_generator_seed"),
        "global_sample_indices": manifest.get("global_sample_indices"),
        "canonical_engine_sha256": manifest.get("run_config", {}).get(
            "canonical_engine_sha256", manifest.get("canonical_engine_sha256")
        ),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def discover_parents(archive_roots: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    """Select one verified parent per case/shard in declared root-priority order."""

    selected: dict[tuple[str, int], tuple[Path, dict[str, Any], str]] = {}
    for priority, archive_root in enumerate(Path(root) for root in archive_roots):
        if not archive_root.is_dir():
            print(f"[G5 parent root missing] priority={priority}: {archive_root}", flush=True)
            continue
        for path in tqdm(
            sorted(archive_root.glob("*.tar.gz")),
            desc=f"G5 parent receipt scan [{priority}]",
            unit="archive",
        ):
            manifest = _root_manifest_from_tar(path)
            if not _selected_parent(manifest):
                continue
            verify_archive_receipt(path)
            case = manifest["run_config"]["case"]
            key = (str(case["case_id"]), int(manifest["shard_index"]))
            identity = _physical_parent_identity(manifest)
            previous = selected.get(key)
            if previous is not None:
                if previous[2] != identity:
                    raise RuntimeError(
                        f"incompatible checksum-verified G5 parents for {key}: "
                        f"{previous[0]} and {path}"
                    )
                print(
                    f"[G5 parent lower-priority duplicate skipped] {path}; "
                    f"using {previous[0]}",
                    flush=True,
                )
                continue
            selected[key] = (path, manifest, identity)
    parents = [(path, manifest) for path, manifest, _ in selected.values()]
    return sorted(
        parents,
        key=lambda item: (
            int(item[1]["run_config"]["case"]["model"]["Ny"]),
            str(item[1]["run_config"]["case"]["case_id"]),
            int(item[1]["shard_index"]),
        ),
    )


def spectrum_factors(
    occupations: np.ndarray,
    *,
    cap_tolerance: float = G5_CAP_TOLERANCE,
) -> dict[str, np.ndarray]:
    """Convert natural occupations into the exact factorized operator spectrum.

    ``density_gap`` is the gap in ``K K^dagger/Z``. ``amplitude_gap`` is half
    as large and is therefore the gap in the singular amplitudes of ``K``.
    ``flip_charge`` is measured relative to the most probable binary string.
    """

    nu = np.asarray(occupations, dtype=np.float64)
    if not np.all(np.isfinite(nu)):
        raise FloatingPointError("natural occupations contain a nonfinite value")
    tolerance = max(float(cap_tolerance), 64.0 * np.finfo(np.float64).eps)
    if np.min(nu) < -tolerance or np.max(nu) > 1.0 + tolerance:
        raise FloatingPointError(
            f"natural occupations leave [0,1]: [{np.min(nu)}, {np.max(nu)}]"
        )
    nu = np.clip(nu, 0.0, 1.0)
    dominant_occupation = (nu >= 0.5).astype(np.int8)
    flip_charge = (1 - 2 * dominant_occupation).astype(np.int8)
    cap_orientation = np.zeros(nu.shape, dtype=np.int8)
    cap_orientation[nu <= tolerance] = -1
    cap_orientation[nu >= 1.0 - tolerance] = 1
    j = 2.0 * np.sqrt(nu * (1.0 - nu))
    j = np.clip(j, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        density_gap = np.abs(np.log(nu) - np.log1p(-nu))
    density_gap[np.isclose(nu, 0.5, atol=tolerance, rtol=0.0)] = 0.0
    amplitude_gap = 0.5 * density_gap
    dominant_probability = np.maximum(nu, 1.0 - nu)
    leading_log_normalized_eigenvalue = np.sum(
        np.log(dominant_probability), axis=-1
    )
    return {
        "occupation": nu,
        "j": j,
        "mu_minus": 0.5 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - j * j))),
        "mu_plus": 0.5 * (1.0 + np.sqrt(np.maximum(0.0, 1.0 - j * j))),
        "dominant_occupation": dominant_occupation,
        "cap_orientation": cap_orientation,
        "flip_charge": flip_charge,
        "density_gap": density_gap,
        "amplitude_gap": amplitude_gap,
        "leading_log_normalized_eigenvalue": leading_log_normalized_eigenvalue,
    }


def lowest_charge_resolved_levels(
    amplitude_gaps: np.ndarray,
    flip_charges: np.ndarray,
    *,
    sectors: Iterable[int] = G5_SECTORS,
    levels_per_sector: int = G5_LEVELS_PER_SECTOR,
    maximum_states: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return low subset sums without constructing the full ``2**N`` spectrum."""

    gaps = np.asarray(amplitude_gaps, dtype=np.float64).reshape(-1)
    charges = np.asarray(flip_charges, dtype=np.int8).reshape(-1)
    if gaps.shape != charges.shape:
        raise ValueError("amplitude_gaps and flip_charges must have the same shape")
    finite = np.isfinite(gaps)
    order = np.argsort(gaps[finite], kind="stable")
    gaps = gaps[finite][order]
    charges = charges[finite][order]
    sector_array = np.asarray(list(sectors), dtype=np.int64)
    # An absent finite level is an exact infinite-amplitude gap, not missing data.
    output = np.full((sector_array.size, int(levels_per_sector)), np.inf)
    counts = np.zeros(sector_array.size, dtype=np.int32)
    sector_index = {int(value): idx for idx, value in enumerate(sector_array)}
    heap: list[tuple[float, tuple[int, ...], int]] = [(0.0, (), 0)]
    popped = 0
    while heap and np.any(counts < levels_per_sector):
        energy, subset, charge = heapq.heappop(heap)
        popped += 1
        idx = sector_index.get(int(charge))
        if idx is not None and counts[idx] < levels_per_sector:
            output[idx, counts[idx]] = energy
            counts[idx] += 1
        if popped >= int(maximum_states):
            break
        if gaps.size == 0:
            continue
        if not subset:
            heapq.heappush(heap, (float(gaps[0]), (0,), int(charges[0])))
            continue
        last = subset[-1]
        next_index = last + 1
        if next_index < gaps.size:
            heapq.heappush(
                heap,
                (
                    energy + float(gaps[next_index]),
                    subset + (next_index,),
                    int(charge + charges[next_index]),
                ),
            )
            heapq.heappush(
                heap,
                (
                    energy - float(gaps[last]) + float(gaps[next_index]),
                    subset[:-1] + (next_index,),
                    int(charge - charges[last] + charges[next_index]),
                ),
            )
    return output, counts, popped


class NaturalOccupationObserver:
    def __init__(self, *, samples: int, dimension: int, checkpoints: Iterable[int]) -> None:
        self.checkpoints = np.asarray(sorted({int(value) for value in checkpoints}), dtype=np.int64)
        self._checkpoint_index = {int(value): idx for idx, value in enumerate(self.checkpoints)}
        self.occupations = np.full(
            (int(samples), self.checkpoints.size, int(dimension)), np.nan, dtype=np.float64
        )

    def __call__(
        self,
        *,
        cycle: int,
        state: Any,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        checkpoint_index = self._checkpoint_index.get(int(cycle))
        if checkpoint_index is None:
            return
        import torch

        if hasattr(state, "frame"):
            raise RuntimeError("G5 identity-input replay must use covariance state")
        hermitian_g = 0.5 * (state + state.mH)
        values = 0.5 * (torch.linalg.eigvalsh(hermitian_g).real + 1.0)
        self.occupations[
            int(batch_start) : int(batch_start) + int(batch_count), checkpoint_index
        ] = values.detach().cpu().numpy()


def _product_paths(output_root: Path, case_id: str, shard_index: int) -> tuple[Path, Path]:
    safe = case_id.replace("/", "_")
    product = output_root / "products" / f"{safe}_shard-{int(shard_index):03d}.npz"
    receipt = product.with_suffix(product.suffix + ".receipt.json")
    return product, receipt


def _receipt_matches(
    product: Path,
    receipt: Path,
    *,
    parent_sha256: str,
    source_sha256: str,
    engine_sha256: str,
) -> bool:
    if not product.exists() and not receipt.exists():
        return False
    if not product.exists() or not receipt.exists():
        raise RuntimeError(f"incomplete G5 product/receipt pair: {product}, {receipt}")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    expected = {
        "schema": G5_RECEIPT_SCHEMA,
        "parent_archive_sha256": parent_sha256,
        "source_sha256": source_sha256,
        "engine_sha256": engine_sha256,
        "product_sha256": sha256_file(product),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"stale or corrupted G5 descendant {product}: {key} "
                f"is {payload.get(key)!r}, expected {value!r}"
            )
    return True


def replay_parent(
    *,
    bundle_root: Path,
    parent_archive: Path,
    parent_manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    import torch
    from classA_U1FGTN_gpu import classA_U1FGTN_gpu

    source_path = Path(__file__).resolve()
    source_sha256 = sha256_file(source_path)
    engine_path = bundle_root / "src" / "classA_U1FGTN_gpu.py"
    engine_sha256 = sha256_file(engine_path)
    parent_sha256 = sha256_file(parent_archive)
    case = parent_manifest["run_config"]["case"]
    case_id = str(case["case_id"])
    shard_index = int(parent_manifest["shard_index"])
    product, receipt = _product_paths(output_root, case_id, shard_index)
    product.parent.mkdir(parents=True, exist_ok=True)
    if _receipt_matches(
        product,
        receipt,
        parent_sha256=parent_sha256,
        source_sha256=source_sha256,
        engine_sha256=engine_sha256,
    ):
        return {"status": "verified_existing", "product": str(product)}

    with tempfile.TemporaryDirectory(prefix="classA_g5_parent_") as temporary:
        extracted = Path(temporary)
        _safe_extract(parent_archive, extracted)
        parent_shard = extracted / "shards" / f"shard_{shard_index:03d}"
        record_path = parent_shard / "ordered_record.npz"
        record = load_ordered_record(record_path)
        samples = int(record["site_ids"].shape[0])
        ny = int(case["model"]["Ny"])
        nx = int(case["model"]["Nx"])
        cycles = int(case["run"]["cycles"])
        checkpoints = np.asarray([ny, 3 * ny // 2, 2 * ny], dtype=np.int64)
        if cycles != int(checkpoints[-1]):
            raise ValueError(f"{case_id}: expected 2Ny cycles, found {cycles}")
        dimension = 2 * nx * ny

        model_config = dict(case["model"])
        model_config.pop("init_mode", None)
        meas_slab_only = bool(model_config.pop("meas_slab_only"))
        model = classA_U1FGTN_gpu(**model_config)
        occupations = NaturalOccupationObserver(
            samples=samples, dimension=dimension, checkpoints=checkpoints
        )
        branch = BranchWeightObserver(samples=samples, cycles=cycles)
        run_args = dict(case["run"])
        for key in list(run_args):
            if key.startswith("lyapunov_"):
                run_args.pop(key)
        run_args.update(
            {
                "samples": samples,
                "batch_size": samples,
                "init_mode": "maxmix",
                "meas_slab_only": meas_slab_only,
                "G_history": False,
                "save": False,
                "return_data": False,
                "frozen_schedule": record["site_ids"],
                "frozen_outcomes": record["outcomes"],
                "record_observer": branch,
                "native_cycle_observer": occupations,
                "state_representation": "covariance",
                "track_choi": False,
                "progress": False,
            }
        )
        started = time.perf_counter()
        result = model.run_markov_circuit(**run_args)
        elapsed = time.perf_counter() - started
        if bool(result.get("choi_tracked")):
            raise RuntimeError("G5 unexpectedly propagated a Choi state")
        log_probability_per_cycle = np.asarray(
            branch.log_probability_per_cycle.detach().cpu().numpy()
            if hasattr(branch.log_probability_per_cycle, "detach")
            else branch.log_probability_per_cycle,
            dtype=np.float64,
        )
        cumulative_log_probability = np.cumsum(log_probability_per_cycle, axis=1)
        log_z = dimension * math.log(2.0) + cumulative_log_probability[:, checkpoints - 1]

        factors = spectrum_factors(occupations.occupations)
        leading_log_sigma = 0.5 * (
            log_z + factors["leading_log_normalized_eigenvalue"]
        )
        sector_levels = np.full(
            (samples, checkpoints.size, G5_SECTORS.size, G5_LEVELS_PER_SECTOR),
            np.nan,
            dtype=np.float64,
        )
        sector_counts = np.zeros(
            (samples, checkpoints.size, G5_SECTORS.size), dtype=np.int32
        )
        enumeration_states = np.zeros((samples, checkpoints.size), dtype=np.int64)
        for sample in range(samples):
            for checkpoint_index in range(checkpoints.size):
                levels, counts, popped = lowest_charge_resolved_levels(
                    factors["amplitude_gap"][sample, checkpoint_index],
                    factors["flip_charge"][sample, checkpoint_index],
                )
                sector_levels[sample, checkpoint_index] = levels
                sector_counts[sample, checkpoint_index] = counts
                enumeration_states[sample, checkpoint_index] = popped
        cap_mu_sum_error = np.max(
            np.abs(factors["mu_minus"] + factors["mu_plus"] - 1.0)
        )
        j_identity_error = np.max(
            np.abs(
                factors["j"]
                - 2.0
                * np.sqrt(
                    factors["occupation"] * (1.0 - factors["occupation"])
                )
            )
        )
        save_npz_atomic(
            product,
            schema=np.asarray(G5_PRODUCT_SCHEMA),
            checkpoints=checkpoints,
            global_sample_indices=np.asarray(
                parent_manifest["global_sample_indices"], dtype=np.int64
            ),
            sectors=G5_SECTORS,
            occupation=factors["occupation"],
            tangent_singular_value=factors["j"],
            mu_minus=factors["mu_minus"],
            mu_plus=factors["mu_plus"],
            dominant_occupation=factors["dominant_occupation"],
            cap_orientation=factors["cap_orientation"],
            flip_charge=factors["flip_charge"],
            radial_density_gap=factors["density_gap"],
            singular_amplitude_gap=factors["amplitude_gap"],
            log_Z_K=log_z,
            leading_log_normalized_eigenvalue=factors[
                "leading_log_normalized_eigenvalue"
            ],
            leading_log_singular_amplitude=leading_log_sigma,
            charge_resolved_low_levels=sector_levels,
            charge_resolved_level_counts=sector_counts,
            subset_states_examined=enumeration_states,
        )
        payload = {
            "schema": G5_RECEIPT_SCHEMA,
            "status": "complete",
            "case_id": case_id,
            "protocol": next(
                protocol for protocol in G5_PROTOCOLS if case_id.endswith(protocol)
            ),
            "Nx": nx,
            "Ny": ny,
            "shard_index": shard_index,
            "global_sample_indices": parent_manifest["global_sample_indices"],
            "parent_archive": str(parent_archive),
            "parent_archive_sha256": parent_sha256,
            "parent_record_sha256": sha256_file(record_path),
            "source_sha256": source_sha256,
            "engine_sha256": engine_sha256,
            "product": str(product),
            "product_sha256": sha256_file(product),
            "elapsed_seconds": elapsed,
            "identity_probe": {
                "input_correlation": "one-particle identity / 2",
                "physical_ensemble": False,
                "track_choi": False,
            },
            "validation": {
                "max_mu_pair_sum_error": float(cap_mu_sum_error),
                "max_j_identity_error": float(j_identity_error),
                "minimum_branch_probability": float(np.min(branch.minimum_probability)),
            },
        }
        write_json_atomic(receipt, payload)
    return {"status": "computed", "product": str(product), "receipt": str(receipt)}


def _bootstrap_ci(
    values: np.ndarray,
    statistic: Any,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values)
    estimate = float(statistic(array))
    if array.shape[0] < 2 or replicates <= 0:
        return estimate, float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        selected = rng.integers(0, array.shape[0], size=array.shape[0])
        draws[index] = statistic(array[selected])
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def fit_tower_means(
    sizes: np.ndarray,
    neutral_spacing: np.ndarray,
    charged_plus: np.ndarray,
    charged_minus: np.ndarray,
) -> dict[str, float]:
    sizes = np.asarray(sizes, dtype=np.float64)
    neutral = np.asarray(neutral_spacing, dtype=np.float64)
    charged = 0.5 * (
        np.asarray(charged_plus, dtype=np.float64)
        + np.asarray(charged_minus, dtype=np.float64)
    )
    if (
        sizes.size < 3
        or not np.all(np.isfinite(neutral))
        or not np.all(np.isfinite(charged))
        or np.any(neutral <= 0.0)
        or np.any(charged <= 0.0)
    ):
        raise ValueError("tower fit requires at least three positive finite-size points")
    inverse_size = 1.0 / sizes
    v_by_size = sizes * neutral / (2.0 * math.pi)
    k_by_size = neutral / (2.0 * charged)
    v_coeff = np.polyfit(inverse_size, v_by_size, deg=1)
    k_coeff = np.polyfit(inverse_size, k_by_size, deg=1)
    v = float(v_coeff[-1])
    k = float(k_coeff[-1])
    predicted_neutral = (2.0 * math.pi * v / sizes)
    predicted_charged = predicted_neutral / (2.0 * k)
    return {
        "v": v,
        "k": k,
        "neutral_relative_rms": float(
            np.sqrt(np.mean(((neutral - predicted_neutral) / neutral) ** 2))
        ),
        "charged_relative_rms": float(
            np.sqrt(np.mean(((charged - predicted_charged) / charged) ** 2))
        ),
    }


def fit_casimir(
    sizes: np.ndarray,
    total_wall_excess_density: np.ndarray,
    *,
    velocity: float,
    wall_count: int,
    include_l4: bool = False,
) -> dict[str, float]:
    """Fit the chiral Casimir coefficient after explicit wall-count division."""

    sizes = np.asarray(sizes, dtype=np.float64)
    total = np.asarray(total_wall_excess_density, dtype=np.float64)
    if sizes.shape != total.shape or sizes.size < (3 if include_l4 else 2):
        raise ValueError("Casimir fit has an invalid finite-size matrix")
    if int(wall_count) <= 0 or not np.isfinite(velocity) or velocity <= 0.0:
        raise ValueError("wall_count and velocity must be positive")
    per_wall = total / float(wall_count)
    inverse_square = 1.0 / sizes**2
    columns = [np.ones_like(sizes), inverse_square]
    if include_l4:
        columns.append(inverse_square**2)
    coefficients, *_ = np.linalg.lstsq(np.column_stack(columns), per_wall, rcond=None)
    slope = float(coefficients[1])
    return {
        "intercept": float(coefficients[0]),
        "slope": slope,
        "c_eff": float(-12.0 * slope / (math.pi * float(velocity))),
        "l4_coefficient": float(coefficients[2]) if include_l4 else 0.0,
        "wall_count": int(wall_count),
    }


def _partition_level_sequence(levels: int) -> np.ndarray:
    """Integers n repeated with chiral-boson partition degeneracy p(n)."""

    partition = [1]
    output: list[int] = []
    n = 0
    while len(output) < int(levels):
        if n >= len(partition):
            total = 0
            k = 1
            while True:
                pentagonal_a = k * (3 * k - 1) // 2
                pentagonal_b = k * (3 * k + 1) // 2
                if pentagonal_a > n:
                    break
                sign = 1 if k % 2 else -1
                total += sign * partition[n - pentagonal_a]
                if pentagonal_b <= n:
                    total += sign * partition[n - pentagonal_b]
                k += 1
            partition.append(total)
        output.extend([n] * partition[n])
        n += 1
    return np.asarray(output[: int(levels)], dtype=np.float64)


def _bootstrap_universal_fits(
    raw: dict[tuple[str, int], dict[str, np.ndarray]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(int(seed))
    tower_draws = np.full((int(replicates), 2), np.nan)
    c_draws = np.full(int(replicates), np.nan)
    sizes = np.asarray(G5_SIZES, dtype=np.float64)
    for draw in range(int(replicates)):
        neutral, plus, minus, excess = [], [], [], []
        for ny in G5_SIZES:
            wall = raw[(G5_PROTOCOLS[0], ny)]
            control = raw[(G5_PROTOCOLS[1], ny)]
            wall_indices = rng.integers(
                0, wall["neutral"].size, size=wall["neutral"].size
            )
            control_indices = rng.integers(
                0, control["neutral"].size, size=control["neutral"].size
            )
            neutral.append(float(np.mean(wall["neutral"][wall_indices])))
            plus.append(float(np.mean(wall["charged_plus"][wall_indices])))
            minus.append(float(np.mean(wall["charged_minus"][wall_indices])))
            excess.append(
                -(
                    float(np.mean(wall["leading"][wall_indices]))
                    - float(np.mean(control["leading"][control_indices]))
                )
                / (2.0 * float(ny))
            )
        try:
            tower = fit_tower_means(sizes, np.asarray(neutral), np.asarray(plus), np.asarray(minus))
            tower_draws[draw] = [tower["v"], tower["k"]]
            casimir = fit_casimir(
                sizes,
                2.0 * np.asarray(excess),
                velocity=tower["v"],
                wall_count=2,
            )
            c_draws[draw] = casimir["c_eff"]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    result = {}
    for name, values in (
        ("v_bootstrap95", tower_draws[:, 0]),
        ("k_bootstrap95", tower_draws[:, 1]),
        ("c_eff_bootstrap95", c_draws),
    ):
        finite = values[np.isfinite(values)]
        if finite.size:
            result[name] = np.quantile(finite, [0.025, 0.975]).tolist()
    return result


def summarize(
    *,
    output_root: Path,
    bootstrap_replicates: int,
    seed: int = 20260823,
) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = defaultdict(list)
    for receipt_path in sorted((output_root / "products").glob("*.npz.receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        product_path = Path(receipt["product"])
        if sha256_file(product_path) != receipt["product_sha256"]:
            raise RuntimeError(f"G5 product checksum mismatch: {product_path}")
        with np.load(product_path, allow_pickle=False) as data:
            product = {key: np.array(data[key], copy=True) for key in data.files}
        groups[(str(receipt["protocol"]), int(receipt["Ny"]))].append(
            (receipt, product)
        )

    rows: list[dict[str, Any]] = []
    raw: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    for (protocol, ny), products in sorted(groups.items()):
        shards = sorted(int(receipt["shard_index"]) for receipt, _ in products)
        sample_indices = np.concatenate(
            [product["global_sample_indices"] for _, product in products]
        )
        if shards != [0, 1] or sorted(sample_indices.tolist()) != list(range(10)):
            continue
        levels = np.concatenate(
            [product["charge_resolved_low_levels"] for _, product in products], axis=0
        )
        leading = np.concatenate(
            [product["leading_log_singular_amplitude"] for _, product in products], axis=0
        )
        checkpoints = products[0][1]["checkpoints"]
        sectors = products[0][1]["sectors"]
        final_index = len(checkpoints) - 1
        duration = float(checkpoints[final_index])
        q0 = int(np.where(sectors == 0)[0][0])
        qp = int(np.where(sectors == 1)[0][0])
        qm = int(np.where(sectors == -1)[0][0])
        neutral = levels[:, final_index, q0, 1] / duration
        charged_plus = levels[:, final_index, qp, 0] / duration
        charged_minus = levels[:, final_index, qm, 0] / duration
        all_levels_per_cycle = levels / checkpoints[None, :, None, None]
        all_leading_per_cycle = leading / checkpoints[None, :]
        raw[(protocol, ny)] = {
            "neutral": neutral,
            "charged_plus": charged_plus,
            "charged_minus": charged_minus,
            "leading": all_leading_per_cycle[:, final_index],
            "levels": levels[:, final_index] / duration,
            "all_levels": all_levels_per_cycle,
            "all_leading": all_leading_per_cycle,
            "checkpoints": checkpoints,
            "sectors": sectors,
        }
        row = {
            "protocol": protocol,
            "Ny": ny,
            "samples": int(neutral.size),
            "prefix_cycle": int(checkpoints[final_index]),
        }
        for name, values in (
            ("neutral_current_spacing_per_cycle", neutral),
            ("q_plus_one_primary_gap_per_cycle", charged_plus),
            ("q_minus_one_primary_gap_per_cycle", charged_minus),
        ):
            estimate, low, high = _bootstrap_ci(
                values,
                np.mean,
                replicates=bootstrap_replicates,
                seed=seed + ny + len(name),
            )
            row[name] = estimate
            row[f"{name}_bootstrap95"] = [low, high]
        rows.append(row)

    required = {(protocol, ny) for protocol in G5_PROTOCOLS for ny in G5_SIZES}
    status = "complete" if required.issubset(raw) else "incomplete"
    fits: dict[str, Any] = {}
    product_status = status
    if status == "complete":
        fit_values = np.concatenate(
            [
                raw[(G5_PROTOCOLS[0], ny)][name]
                for ny in G5_SIZES
                for name in ("neutral", "charged_plus", "charged_minus")
            ]
        )
        if not np.all(np.isfinite(fit_values)) or np.any(fit_values <= 0.0):
            status = "complete_products_tower_unresolved"
            fits["status"] = "failed"
            fits["failure_reason"] = (
                "one or more required q=0 or q=+-1 low levels is nonfinite or nonpositive; "
                "retain factorized products and apply the G5 downgrade"
            )
    if status == "complete":
        sizes = np.asarray(G5_SIZES, dtype=np.float64)
        wall = [raw[(G5_PROTOCOLS[0], ny)] for ny in G5_SIZES]
        neutral = np.asarray([np.mean(item["neutral"]) for item in wall])
        charged_plus = np.asarray([np.mean(item["charged_plus"]) for item in wall])
        charged_minus = np.asarray([np.mean(item["charged_minus"]) for item in wall])
        fits["tower"] = fit_tower_means(
            sizes, neutral, charged_plus, charged_minus
        )
        fits["tower_delete_smallest"] = fit_tower_means(
            sizes[1:], neutral[1:], charged_plus[1:], charged_minus[1:]
        )
        prefix_fits = []
        for checkpoint_index, aspect_ratio in enumerate((1.0, 1.5, 2.0)):
            prefix_neutral, prefix_plus, prefix_minus = [], [], []
            for item in wall:
                sector_values = item["all_levels"][:, checkpoint_index]
                q0 = int(np.where(item["sectors"] == 0)[0][0])
                qp = int(np.where(item["sectors"] == 1)[0][0])
                qm = int(np.where(item["sectors"] == -1)[0][0])
                prefix_neutral.append(float(np.mean(sector_values[:, q0, 1])))
                prefix_plus.append(float(np.mean(sector_values[:, qp, 0])))
                prefix_minus.append(float(np.mean(sector_values[:, qm, 0])))
            prefix_fits.append(
                {
                    "T_over_L": aspect_ratio,
                    **fit_tower_means(
                        sizes,
                        np.asarray(prefix_neutral),
                        np.asarray(prefix_plus),
                        np.asarray(prefix_minus),
                    ),
                }
            )
        fits["prefix_tower_stability"] = prefix_fits
        expected_descendants = _partition_level_sequence(32)
        descendant_residuals = []
        for ny, item in zip(G5_SIZES, wall):
            spacing = float(np.mean(item["neutral"]))
            for sector_index, _sector in enumerate(item["sectors"]):
                mean_levels = np.mean(item["levels"][:, sector_index, :32], axis=0)
                if not np.all(np.isfinite(mean_levels)):
                    continue
                observed = (mean_levels - mean_levels[0]) / spacing
                descendant_residuals.extend((observed - expected_descendants).tolist())
        fits["tower"]["descendant_integer_degeneracy_rms"] = float(
            np.sqrt(np.mean(np.square(descendant_residuals)))
        )
        fits["tower"]["descendant_reference"] = (
            "integer n with chiral-boson partition degeneracy p(n), first 32 levels per sector"
        )
        v = float(fits["tower"]["v"])
        wall_excess = []
        for ny in G5_SIZES:
            wall_leading = raw[(G5_PROTOCOLS[0], ny)]["leading"]
            control_leading = raw[(G5_PROTOCOLS[1], ny)]["leading"]
            wall_excess.append(
                -(
                    float(np.mean(wall_leading))
                    - float(np.mean(control_leading))
                )
                / (2.0 * float(ny))
            )
        casimir = fit_casimir(
            sizes,
            2.0 * np.asarray(wall_excess),
            velocity=v,
            wall_count=2,
        )
        casimir_l4 = fit_casimir(
            sizes,
            2.0 * np.asarray(wall_excess),
            velocity=v,
            wall_count=2,
            include_l4=True,
        )
        casimir_delete = fit_casimir(
            sizes[1:],
            2.0 * np.asarray(wall_excess[1:]),
            velocity=float(fits["tower_delete_smallest"]["v"]),
            wall_count=2,
        )
        fits["casimir"] = {
            "per_wall_excess_free_energy_density": wall_excess,
            "fit_model": "f_wall(L)=f_infinity-pi*c_eff*v/(12 L^2)",
            **casimir,
            "include_l4_sensitivity": casimir_l4,
            "delete_smallest_sensitivity": casimir_delete,
            "matched_trivial_subtraction": True,
        }
        bootstrap_fit_intervals = _bootstrap_universal_fits(
            raw,
            replicates=bootstrap_replicates,
            seed=seed + 100_000,
        )
        fits["tower"].update(
            {
                key: value
                for key, value in bootstrap_fit_intervals.items()
                if key in ("v_bootstrap95", "k_bootstrap95")
            }
        )
        if "c_eff_bootstrap95" in bootstrap_fit_intervals:
            fits["casimir"]["c_eff_bootstrap95"] = bootstrap_fit_intervals[
                "c_eff_bootstrap95"
            ]
    payload = {
        "schema": G5_SUMMARY_SCHEMA,
        "status": status,
        "product_status": product_status,
        "required_groups": len(required),
        "complete_groups": len(required.intersection(raw)),
        "rows": rows,
        "fits": fits,
        "bootstrap": {
            "unit": "parent trajectory",
            "replicates": int(bootstrap_replicates),
            "seed": int(seed),
        },
        "interpretation": {
            "spectrum": "fixed-word many-body singular amplitudes",
            "identity_input": "algebraic probe only",
            "choi_propagated": False,
            "record_scgf": "separate supplemental large-deviation diagnostic",
        },
    }
    write_json_atomic(output_root / "G5_manybody_spectrum_summary.json", payload)
    return payload


def run(
    *,
    bundle_root: Path,
    archive_roots: Iterable[Path],
    output_root: Path,
    bootstrap_replicates: int,
    mode: str,
) -> dict[str, Any]:
    if mode not in ("production", "pilot", "smoke"):
        raise ValueError(f"unsupported G5 mode {mode!r}")
    require_a100(smoke=mode == "smoke")
    archive_roots = [Path(root) for root in archive_roots]
    parents = discover_parents(archive_roots)
    results = []
    for archive, manifest in tqdm(parents, desc="G5 fixed-word replays", unit="shard"):
        results.append(
            replay_parent(
                bundle_root=bundle_root,
                parent_archive=archive,
                parent_manifest=manifest,
                output_root=output_root,
            )
        )
    summary = summarize(
        output_root=output_root, bootstrap_replicates=bootstrap_replicates
    )
    summary["replay_inventory"] = {
        "archive_roots_in_priority_order": [str(path) for path in archive_roots],
        "discovered_parent_shards": len(parents),
        "computed": sum(item["status"] == "computed" for item in results),
        "verified_existing": sum(
            item["status"] == "verified_existing" for item in results
        ),
    }
    write_json_atomic(output_root / "G5_manybody_spectrum_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Bundle-02 words from the one-particle identity/2 and reconstruct "
            "their exact factorized many-body singular spectra without a Choi state"
        )
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--archive-root",
        type=Path,
        action="append",
        required=True,
        help="parent Bundle-02 archive directory; repeat in preference order",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--mode", choices=("production", "pilot", "smoke"), default="production")
    args = parser.parse_args(argv)
    payload = run(
        bundle_root=args.bundle_root,
        archive_roots=args.archive_root,
        output_root=args.output_root,
        bootstrap_replicates=args.bootstrap_replicates,
        mode=args.mode,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
