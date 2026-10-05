from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from production_runtime import save_npz_atomic, sha256_file


CHANNELS = ("Ap", "Am", "Bp", "Bm")
CHANNEL_TARGETS = (0, 1, 0, 1)
STATIC_SCHEMA = "b1_controller_frame_static_v1"
OBSERVABLE_SCHEMA = "b1_controller_frame_observables_v1"


def validate_b1_config(config: dict[str, Any]) -> None:
    expected_contract = {
        "samples": 10,
        "sample_shard_size": 5,
        "cycles_rule": "2*Ny",
        "physical_burn_in_cycles": 0,
        "sequence": "random",
        "dtype": "complex128",
        "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit",
    }
    if config.get("schema_version") != 1 or config.get("bundle") != "06_b1_controller_frame":
        raise ValueError("not a schema-v1 B1 controller-frame configuration")
    locked = config.get("locked_contract", {})
    mismatches = {
        key: (locked.get(key), value)
        for key, value in expected_contract.items()
        if locked.get(key) != value
    }
    if mismatches:
        raise ValueError(f"B1 locked contract mismatch: {mismatches}")
    matrix = config.get("B1", {})
    if int(matrix.get("Nx", -1)) != 20 or int(matrix.get("Ny", -1)) != 40:
        raise ValueError("B1 production geometry must be Nx=20, Ny=40")
    if matrix.get("protocols") != ["explicit_interface", "explicit_interface_matched_trivial"]:
        raise ValueError("B1 requires exactly the wall and matched-trivial protocols")
    if int(matrix.get("bootstrap_draws", -1)) != 2000:
        raise ValueError("B1 bootstrap_draws must remain 2000")
    numerical_tolerance = float(matrix.get("numerical_tolerance", -1.0))
    charge_tolerance = float(matrix.get("charge_integer_tolerance", -1.0))
    if not (0.0 < numerical_tolerance <= 1e-8):
        raise ValueError("B1 numerical_tolerance must lie in (0, 1e-8]")
    if not (numerical_tolerance <= charge_tolerance <= 1e-8):
        raise ValueError(
            "B1 charge_integer_tolerance must be between numerical_tolerance and 1e-8"
        )
    if config.get("output_bundle") != "06_b1_controller_frame_frame_native_v2":
        raise ValueError("B1 corrected production must use its versioned output bundle")


def b1_cases(config: dict[str, Any], *, smoke: bool = False) -> list[dict[str, Any]]:
    validate_b1_config(config)
    matrix = config["B1"]
    nx = 4 if smoke else int(matrix["Nx"])
    ny = 6 if smoke else int(matrix["Ny"])
    samples = 5 if smoke else int(config["locked_contract"]["samples"])
    cases = []
    for protocol in matrix["protocols"]:
        wall = protocol == "explicit_interface"
        cases.append(
            {
                "case_id": f"B1_N{nx}x{ny}_{protocol}",
                "protocol": protocol,
                "model": {
                    "Nx": nx,
                    "Ny": ny,
                    "DW": wall,
                    "nshell": int(matrix["nshell"]),
                    "alpha_1": float(matrix["alpha_top"] if wall else matrix["alpha_triv"]),
                    "alpha_2": float(matrix["alpha_triv"]),
                    "filling_frac": 0.5,
                    "trial_orbitals": "X",
                    "dw_truncation": False,
                    "device": "cpu" if smoke else "cuda:0",
                    "dtype": "complex128",
                    "backend": "local",
                },
                "run": {
                    "samples": samples,
                    "cycles": 2 * ny,
                    "G_history": False,
                    "progress": False,
                    "save": False,
                    "return_data": False,
                    "init_mode": "default",
                    "sequence": "random",
                    "perfect_correction": True,
                    "postselect": False,
                    "postselect_probability": 0.0,
                    "meas_slab_only": False,
                },
            }
        )
    return cases


def _sha256_arrays(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _fermi_cluster(eigenvalues: Any, rank: int, tolerance: float) -> tuple[int, int]:
    values = np.asarray(
        eigenvalues.detach().cpu().numpy() if hasattr(eigenvalues, "detach") else eigenvalues,
        dtype=np.float64,
    )
    rank = int(rank)
    if rank <= 0 or rank >= values.size:
        return rank, rank
    if values[rank] - values[rank - 1] > float(tolerance):
        return rank, rank
    reference = 0.5 * (values[rank - 1] + values[rank])
    left = rank - 1
    while left > 0 and abs(values[left - 1] - reference) <= float(tolerance):
        left -= 1
    right = rank + 1
    while right < values.size and abs(values[right] - reference) <= float(tolerance):
        right += 1
    return left, right


def residual_bounds(
    mode_weights: np.ndarray,
    targets: np.ndarray,
    eigenvalues: np.ndarray,
    rank: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    rank = int(rank)
    occupancy = np.sum(mode_weights[:rank], axis=0)
    residual = np.where(targets == 1, 1.0 - occupancy, occupancy)
    left, right = _fermi_cluster(eigenvalues, rank, tolerance)
    if left == right:
        return residual, residual.copy(), residual.copy(), (left, right)
    base = np.sum(mode_weights[:left], axis=0)
    cluster = np.sum(mode_weights[left:right], axis=0)
    selected = rank - left
    cluster_dimension = right - left
    if selected == 0:
        low_occ = high_occ = base
    elif selected == cluster_dimension:
        low_occ = high_occ = base + cluster
    else:
        low_occ, high_occ = base, base + cluster
    lower = np.where(targets == 1, 1.0 - high_occ, low_occ)
    upper = np.where(targets == 1, 1.0 - low_occ, high_occ)
    return residual, lower, upper, (left, right)


@dataclass
class ControllerFrame:
    vectors: Any
    targets: Any
    operator: Any
    eigenvalues: Any
    eigenvectors: Any
    active_indices: Any
    center_x: np.ndarray
    center_y: np.ndarray
    channel_indices: np.ndarray
    target_rank: int
    cluster: tuple[int, int]
    target_sum: float
    eigenvalue_prefix: Any
    below_projector: Any
    cluster_vectors: Any
    cluster_selected: int
    static_payload: dict[str, Any]


def construct_controller_frame(model: Any, *, degeneracy_tolerance: float = 1e-10) -> ControllerFrame:
    import torch

    active = model.active_top_layer_indices(meas_slab_only=False).to(
        device=model.device, dtype=torch.long
    )
    sequence = model._sequence_helper("raster_y", meas_slab_only=False)
    coords = [(int(x), int(y)) for x, y in sequence["coords_for_len"]]
    vectors, targets, xs, ys, channels = [], [], [], [], []
    for x, y in coords:
        site_id = int(x + model.Nx * y)
        for channel_index, (channel, target) in enumerate(zip(CHANNELS, CHANNEL_TARGETS)):
            vector = getattr(model, f"WF_{channel}_sites")[site_id].index_select(0, active)
            vector = vector / torch.linalg.vector_norm(vector)
            vectors.append(vector)
            targets.append(target)
            xs.append(x)
            ys.append(y)
            channels.append(channel_index)
    frame = torch.stack(vectors, dim=1)
    target_tensor = torch.as_tensor(targets, dtype=torch.int64, device=model.device)
    signs = 1.0 - 2.0 * target_tensor.to(model.real_dtype)
    operator = (frame * signs[None, :]) @ frame.conj().transpose(0, 1)
    operator = 0.5 * (operator + operator.conj().transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(operator)
    dimension = int(frame.shape[0])
    rank = dimension // 2
    cluster = _fermi_cluster(eigenvalues, rank, degeneracy_tolerance)
    left, right = cluster
    below = eigenvectors[:, :left] @ eigenvectors[:, :left].conj().transpose(0, 1)
    cluster_vectors = eigenvectors[:, left:right]
    mode_weights = torch.abs(eigenvectors.conj().transpose(0, 1) @ frame) ** 2
    mode_weights_cpu = mode_weights.detach().cpu().numpy()
    targets_cpu = target_tensor.detach().cpu().numpy().astype(np.int8)
    values_cpu = eigenvalues.detach().cpu().numpy()
    residual, lower, upper, _ = residual_bounds(
        mode_weights_cpu, targets_cpu, values_cpu, rank, degeneracy_tolerance
    )
    occupancy = np.sum(mode_weights_cpu[:rank], axis=0)
    f_direct = float(np.sum(np.where(targets_cpu == 1, 1.0 - occupancy, occupancy)))
    target_sum = float(np.sum(targets_cpu))
    f_formula = target_sum + float(np.sum(values_cpu[:rank]))
    residual_map = np.full((model.Nx, model.Ny, len(CHANNELS)), np.nan, dtype=np.float64)
    lower_map = residual_map.copy()
    upper_map = residual_map.copy()
    x_array = np.asarray(xs, dtype=np.int64)
    y_array = np.asarray(ys, dtype=np.int64)
    channel_array = np.asarray(channels, dtype=np.int64)
    residual_map[x_array, y_array, channel_array] = residual
    lower_map[x_array, y_array, channel_array] = lower
    upper_map[x_array, y_array, channel_array] = upper
    reference = eigenvectors[:, :rank] @ eigenvectors[:, :rank].conj().transpose(0, 1)
    projector_hash = _sha256_arrays(reference)
    manifold_hash = _sha256_arrays(below, cluster_vectors)
    prefix = torch.cat(
        (
            torch.zeros((1,), dtype=model.real_dtype, device=model.device),
            torch.cumsum(eigenvalues.real, dim=0),
        )
    )
    static = {
        "schema": np.asarray(STATIC_SCHEMA),
        "Nx": np.asarray(model.Nx, dtype=np.int64),
        "Ny": np.asarray(model.Ny, dtype=np.int64),
        "dimension": np.asarray(dimension, dtype=np.int64),
        "constraint_count": np.asarray(frame.shape[1], dtype=np.int64),
        "target_rank": np.asarray(rank, dtype=np.int64),
        "operator_eigenvalues": values_cpu,
        "target_occupancies": targets_cpu,
        "center_x": x_array,
        "center_y": y_array,
        "channel_indices": channel_array,
        "residuals_half_filling": residual,
        "residual_lower_half_filling": lower,
        "residual_upper_half_filling": upper,
        "residual_map_half_filling": residual_map,
        "residual_map_lower_half_filling": lower_map,
        "residual_map_upper_half_filling": upper_map,
        "fermi_cluster": np.asarray(cluster, dtype=np.int64),
        "selection_gap": np.asarray(values_cpu[rank] - values_cpu[rank - 1]),
        "f_star_half_filling": np.asarray(f_direct),
        "f_star_formula_half_filling": np.asarray(f_formula),
        "f_star_residual": np.asarray(abs(f_direct - f_formula)),
        "hermiticity_residual": np.asarray(
            torch.linalg.matrix_norm(operator - operator.conj().transpose(0, 1)).item()
        ),
        "frame_sha256": np.asarray(_sha256_arrays(frame, target_tensor)),
        "reference_projector_sha256": np.asarray(projector_hash),
        "ground_manifold_sha256": np.asarray(manifold_hash),
        "dense_arrays_archived": np.asarray(0, dtype=np.int64),
    }
    return ControllerFrame(
        vectors=frame,
        targets=target_tensor,
        operator=operator,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        active_indices=active,
        center_x=x_array,
        center_y=y_array,
        channel_indices=channel_array,
        target_rank=rank,
        cluster=cluster,
        target_sum=target_sum,
        eigenvalue_prefix=prefix,
        below_projector=below,
        cluster_vectors=cluster_vectors,
        cluster_selected=rank - left,
        static_payload=static,
    )


def save_static_frame(path: Path | str, frame: ControllerFrame) -> dict[str, Any]:
    path = Path(path)
    save_npz_atomic(path, **frame.static_payload)
    return {
        "schema": STATIC_SCHEMA,
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dense_arrays_archived": 0,
    }


class ControllerFrameObserver:
    def __init__(self, *, frame: ControllerFrame, samples: int, cycles: int) -> None:
        self.frame = frame
        self.samples = int(samples)
        self.cycles = int(cycles)
        shape = (self.samples, self.cycles + 1)
        self.total_charge = np.full(shape, np.nan, dtype=np.float64)
        self.charge_integer_residual = np.full(shape, np.nan, dtype=np.float64)
        self.controller_cost = np.full(shape, np.nan, dtype=np.float64)
        self.ky_fan_excess = np.full(shape, np.nan, dtype=np.float64)
        self.manifold_distance = np.full(shape, np.nan, dtype=np.float64)
        self.purity_defect = np.full(shape, np.nan, dtype=np.float64)
        self.successive_delta = np.full(shape, np.nan, dtype=np.float64)
        self._previous: Any | None = None
        self._device_arrays: dict[str, Any] | None = None

    def _ensure_device_arrays(self, device: Any) -> None:
        if self._device_arrays is not None:
            return
        import torch

        shape = (self.samples, self.cycles + 1)
        self._device_arrays = {
            name: torch.full(shape, torch.nan, dtype=torch.float64, device=device)
            for name in (
                "total_charge",
                "charge_integer_residual",
                "controller_cost",
                "ky_fan_excess",
                "manifold_distance",
                "purity_defect",
                "successive_delta",
            )
        }

    def _materialize_device_arrays(self) -> None:
        if self._device_arrays is None:
            return
        for name, value in self._device_arrays.items():
            setattr(self, name, value.detach().cpu().numpy())
        self._device_arrays = None

    def __call__(
        self,
        *,
        cycle: int,
        G: Any = None,
        state: Any = None,
        batch_start: int,
        batch_count: int,
        **_: Any,
    ) -> None:
        import torch

        start, stop = int(batch_start), int(batch_start) + int(batch_count)
        live = state if state is not None else G
        active = self.frame.active_indices
        frame_native = hasattr(live, "frame") and hasattr(live, "ranks")
        if frame_native:
            restricted_frame = live.frame.index_select(1, active)
            dimension = int(active.numel())
            charge = restricted_frame.abs().square().sum(dim=(1, 2)).real
            reduced_gram = restricted_frame.mH @ restricted_frame
            device = restricted_frame.device
        else:
            restricted = live.index_select(1, active).index_select(2, active)
            restricted = 0.5 * (restricted + restricted.mH)
            dimension = int(restricted.shape[-1])
            eye = torch.eye(dimension, dtype=restricted.dtype, device=restricted.device)
            occupation = 0.5 * (restricted + eye[None])
            charge = torch.diagonal(occupation, dim1=-2, dim2=-1).real.sum(dim=1)
            device = restricted.device
        rank = torch.round(charge).to(torch.long)
        rank_for_lookup = torch.clamp(rank, min=0, max=dimension)
        if frame_native:
            operator_action = self.frame.operator @ restricted_frame
            cost = self.frame.target_sum + torch.sum(
                restricted_frame.conj() * operator_action, dim=(1, 2)
            ).real
        else:
            cost = self.frame.target_sum + torch.einsum(
                "ij,bji->b", self.frame.operator, occupation
            ).real
        f_star = self.frame.target_sum + self.frame.eigenvalue_prefix.index_select(
            0, rank_for_lookup
        )
        excess = cost - f_star
        if frame_native:
            c_square_trace = reduced_gram.abs().square().sum(dim=(1, 2)).real
            below_action = self.frame.below_projector @ restricted_frame
            below_overlap = torch.sum(
                restricted_frame.conj() * below_action, dim=(1, 2)
            ).real
        else:
            c_square_trace = torch.sum(torch.abs(occupation) ** 2, dim=(1, 2)).real
            below_overlap = torch.einsum(
                "ij,bji->b", self.frame.below_projector, occupation
            ).real
        cluster_overlap = torch.zeros_like(below_overlap)
        if self.frame.cluster_selected > 0:
            cluster = self.frame.cluster_vectors
            if frame_native:
                cluster_frame = cluster.mH[None] @ restricted_frame
                reduced = cluster_frame @ cluster_frame.mH
            else:
                reduced = cluster.mH[None] @ occupation @ cluster[None]
            reduced = 0.5 * (reduced + reduced.conj().transpose(-2, -1))
            cluster_values = torch.linalg.eigvalsh(reduced).real
            cluster_overlap = torch.sum(
                cluster_values[:, -self.frame.cluster_selected :], dim=1
            )
        distance_squared = torch.clamp(
            c_square_trace
            + float(self.frame.target_rank)
            - 2.0 * (below_overlap + cluster_overlap),
            min=0.0,
        )
        if frame_native:
            singular_occupations = torch.linalg.svdvals(restricted_frame).square().real
            purity = torch.sqrt(
                torch.sum(
                    (singular_occupations.square() - singular_occupations).square(), dim=1
                )
            )
        else:
            purity = torch.linalg.matrix_norm(occupation @ occupation - occupation, ord="fro")
        distance = torch.sqrt(distance_squared / max(1, dimension))
        delta = None
        if self._previous is not None:
            if frame_native:
                previous = self._previous
                previous_gram_norm = (previous.mH @ previous).abs().square().sum(dim=(1, 2))
                current_gram_norm = reduced_gram.abs().square().sum(dim=(1, 2))
                cross_norm = (previous.mH @ restricted_frame).abs().square().sum(dim=(1, 2))
                delta = torch.sqrt(
                    (previous_gram_norm + current_gram_norm - 2.0 * cross_norm).real.clamp_min(0.0)
                ) / max(1, dimension)
            else:
                delta = torch.linalg.matrix_norm(restricted - self._previous, ord="fro") / max(
                    1, dimension
                )
        self._previous = (
            restricted_frame.detach().clone() if frame_native else restricted.detach().clone()
        )

        if device.type == "cuda":
            self._ensure_device_arrays(device)
            arrays = self._device_arrays
            index = (slice(start, stop), int(cycle))
            arrays["total_charge"][index] = charge.to(torch.float64)
            arrays["charge_integer_residual"][index] = torch.abs(
                charge - rank.to(charge.dtype)
            ).to(torch.float64)
            arrays["controller_cost"][index] = cost.to(torch.float64)
            arrays["ky_fan_excess"][index] = excess.to(torch.float64)
            arrays["manifold_distance"][index] = distance.to(torch.float64)
            arrays["purity_defect"][index] = (
                purity.to(torch.float64) / max(1, dimension)
            )
            if delta is not None:
                arrays["successive_delta"][index] = delta.to(torch.float64)
            return

        if bool(torch.any(rank < 0)) or bool(torch.any(rank > dimension)):
            raise RuntimeError("trajectory charge lies outside the active one-particle space")
        if delta is not None:
            self.successive_delta[start:stop, int(cycle)] = delta.detach().cpu().numpy()
        self.total_charge[start:stop, int(cycle)] = charge.detach().cpu().numpy()
        self.charge_integer_residual[start:stop, int(cycle)] = (
            torch.abs(charge - rank.to(charge.dtype)).detach().cpu().numpy()
        )
        self.controller_cost[start:stop, int(cycle)] = cost.detach().cpu().numpy()
        self.ky_fan_excess[start:stop, int(cycle)] = excess.detach().cpu().numpy()
        self.manifold_distance[start:stop, int(cycle)] = distance.detach().cpu().numpy()
        self.purity_defect[start:stop, int(cycle)] = (
            purity.detach().cpu().numpy() / max(1, dimension)
        )

    def validate(
        self,
        *,
        tolerance: float = 1e-10,
        charge_integer_tolerance: float | None = None,
    ) -> dict[str, Any]:
        self._materialize_device_arrays()
        numerical_tolerance = float(tolerance)
        charge_tolerance = (
            numerical_tolerance
            if charge_integer_tolerance is None
            else float(charge_integer_tolerance)
        )
        if numerical_tolerance <= 0.0 or charge_tolerance < numerical_tolerance:
            raise ValueError(
                "charge_integer_tolerance must be at least the positive numerical tolerance"
            )
        arrays = (
            self.total_charge,
            self.charge_integer_residual,
            self.controller_cost,
            self.ky_fan_excess,
            self.manifold_distance,
            self.purity_defect,
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise FloatingPointError("B1 observer contains missing or non-finite values")
        dimension = int(self.frame.active_indices.numel())
        if np.any(self.total_charge < -charge_tolerance) or np.any(
            self.total_charge > dimension + charge_tolerance
        ):
            raise RuntimeError(
                "trajectory charge lies outside the active one-particle space"
            )
        minimum_excess = float(np.min(self.ky_fan_excess))
        if minimum_excess < -numerical_tolerance:
            raise RuntimeError(f"Ky Fan excess became negative: {minimum_excess}")
        max_charge_error = float(np.max(self.charge_integer_residual))
        if max_charge_error > charge_tolerance:
            raise RuntimeError(f"trajectory charge is not integer: {max_charge_error}")
        return {
            "schema": OBSERVABLE_SCHEMA,
            "samples": self.samples,
            "cycles": self.cycles,
            "minimum_ky_fan_excess": minimum_excess,
            "maximum_charge_integer_residual": max_charge_error,
            "numerical_tolerance": numerical_tolerance,
            "charge_integer_tolerance": charge_tolerance,
            "maximum_purity_defect": float(np.max(self.purity_defect)),
        }

    def save_raw(self, path: Path | str) -> dict[str, Any]:
        """Persist compact histories before any scientific acceptance check."""

        self._materialize_device_arrays()
        path = Path(path)
        save_npz_atomic(
            path,
            schema=np.asarray(OBSERVABLE_SCHEMA),
            cycle=np.arange(self.cycles + 1, dtype=np.int64),
            total_charge=self.total_charge,
            charge_integer_residual=self.charge_integer_residual,
            controller_cost=self.controller_cost,
            ky_fan_excess=self.ky_fan_excess,
            half_filled_manifold_distance=self.manifold_distance,
            purity_defect=self.purity_defect,
            successive_covariance_delta=self.successive_delta,
            storage_contract=np.asarray("scalar_histories_only;no_covariance_or_projector"),
        )
        return {
            "schema": OBSERVABLE_SCHEMA,
            "samples": self.samples,
            "cycles": self.cycles,
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "permanent_dense_matrix_bytes": 0,
        }

    def save(
        self,
        path: Path | str,
        *,
        tolerance: float = 1e-10,
        charge_integer_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Persist first, then validate, so a failed check cannot erase the data."""

        product = self.save_raw(path)
        diagnostics = self.validate(
            tolerance=tolerance,
            charge_integer_tolerance=charge_integer_tolerance,
        )
        return {**product, **diagnostics}
