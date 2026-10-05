from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


FORBIDDEN_FINITE_SHELL_TOKENS = ("ky", "momentum", "branch")


def wall_interval(nx: int, rule: str) -> tuple[int, int]:
    half = int(nx) // 2
    if rule == "canonical":
        width = max(1, int(nx) // 4)
    elif rule == "legacy":
        width = max(1, int(math.floor(0.2 * int(nx))))
    else:
        raise ValueError(f"unknown wall rule {rule!r}")
    x0 = max(0, half - width)
    x1 = min(int(nx), half + width + 1)
    return x0, x1 - 1


def validate_selected_observable_schema(nshell: int | None, arrays: dict[str, Any]) -> None:
    if nshell is None:
        required = {
            "ky",
            "occupation_spectrum_ky",
            "wall_branch_occupations_ky",
            "wall_branch_weights_ky",
        }
        missing = sorted(required.difference(arrays))
        if missing:
            raise ValueError(f"untruncated product is missing momentum observables: {missing}")
        return
    forbidden = sorted(
        key
        for key in arrays
        if any(token in key.lower() for token in FORBIDDEN_FINITE_SHELL_TOKENS)
    )
    if forbidden:
        raise ValueError(
            f"finite-shell products may not contain momentum-resolved fields: {forbidden}"
        )


@dataclass
class StationarySolution:
    ky: np.ndarray
    covariance_blocks: np.ndarray
    damping_blocks: np.ndarray
    v_minus_blocks: np.ndarray
    v_plus_blocks: np.ndarray
    damping_eigenvalues: np.ndarray
    damping_eigenvectors: np.ndarray
    residual_by_ky: np.ndarray


class MeanChannelLindbladCPU:
    """CPU-native deterministic gain/loss channel and continuous generator.

    The evolved object is the number covariance ``C`` with eigenvalues in ``[0,1]``.
    The finite map is an exact completely positive Gaussian channel: exact gain and
    loss semigroups are composed with Strang splitting.  At fixed physical time the
    product approaches the continuous generator as its substep ``p`` tends to zero.
    No trajectory or random schedule is sampled.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        alpha_top: float,
        alpha_triv: float,
        domain_wall: bool = True,
        dw_truncation: bool = False,
        wall_rule: str = "canonical",
        nshell: int | None = None,
        n_a: float = 0.5,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        self.alpha_top, self.alpha_triv = float(alpha_top), float(alpha_triv)
        self.domain_wall = bool(domain_wall)
        self.dw_truncation = bool(dw_truncation)
        self.wall_rule = str(wall_rule)
        self.nshell = None if nshell is None else int(nshell)
        self.n_a = float(n_a)
        if self.nx < 2 or self.ny < 2:
            raise ValueError("Nx and Ny must both be at least two")
        if not 0.0 <= self.n_a <= 1.0:
            raise ValueError("n_a must lie in [0,1]")
        if self.nshell is not None and self.nshell < 0:
            raise ValueError("nshell must be nonnegative or None")
        if self.dw_truncation and not self.domain_wall:
            raise ValueError("dw_truncation=True requires domain_wall=True")
        self.n = 2 * self.nx * self.ny
        self.d = 2 * self.nx
        self.centers = self.nx * self.ny
        self.wall_locations: tuple[int, int] | None = None
        self.alpha_profile = self._alpha_profile()

    def _alpha_profile(self) -> np.ndarray:
        alpha = np.full(
            (self.nx, self.ny),
            self.alpha_triv if self.domain_wall else self.alpha_top,
            dtype=np.float64,
        )
        if self.domain_wall:
            x0, x1 = wall_interval(self.nx, self.wall_rule)
            alpha[x0 : x1 + 1, :] = self.alpha_top
            self.wall_locations = (x0, x1)
        return alpha

    def _flattened_hamiltonian(self) -> np.ndarray:
        kx = 2 * np.pi * np.fft.fftfreq(self.nx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny)
        kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
        nxv = np.sin(kxg)[:, :, None, None]
        nyv = np.sin(kyg)[:, :, None, None]
        nzv = (
            self.alpha_profile[None, None, :, :]
            - np.cos(kxg)[:, :, None, None]
            - np.cos(kyg)[:, :, None, None]
        )
        norm = np.sqrt(nxv**2 + nyv**2 + nzv**2)
        norm = np.maximum(norm, 1e-15)
        nxv, nyv, nzv = nxv / norm, nyv / norm, nzv / norm
        h = np.empty(
            (self.nx, self.ny, self.nx, self.ny, 2, 2), dtype=np.complex128
        )
        h[..., 0, 0] = nzv
        h[..., 1, 1] = -nzv
        h[..., 0, 1] = nxv - 1j * nyv
        h[..., 1, 0] = nxv + 1j * nyv
        return h

    def _frame(self, h: np.ndarray, *, band_sign: int, trial_sign: int) -> np.ndarray:
        trial = np.asarray([1.0, float(trial_sign)], dtype=np.complex128) / math.sqrt(2.0)
        projector = 0.5 * (np.eye(2, dtype=np.complex128) + int(band_sign) * h)
        psi = np.einsum("m,...mn->...n", trial.conj(), projector, optimize=True)
        kx = 2 * np.pi * np.fft.fftfreq(self.nx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny)
        rx = np.arange(self.nx)
        ry = np.arange(self.ny)
        phase = np.exp(
            1j
            * (
                kx[:, None, None, None] * rx[None, None, :, None]
                + ky[None, :, None, None] * ry[None, None, None, :]
            )
        )
        transformed = np.fft.fft2(phase[..., None] * psi, axes=(0, 1))
        frame = np.transpose(transformed, (0, 1, 4, 2, 3)).copy()
        if self.nshell is not None:
            x = np.arange(self.nx)[:, None, None, None]
            y = np.arange(self.ny)[None, :, None, None]
            rx_i = np.arange(self.nx)[None, None, :, None]
            ry_i = np.arange(self.ny)[None, None, None, :]
            dx = (x - rx_i + self.nx // 2) % self.nx - self.nx // 2
            dy = (y - ry_i + self.ny // 2) % self.ny - self.ny // 2
            frame *= ((np.abs(dx) <= self.nshell) & (np.abs(dy) <= self.nshell))[:, :, None]
        if self.dw_truncation:
            if self.wall_locations is None:
                raise ValueError("dw_truncation requires a valid domain-wall interval")
            x0, x1 = self.wall_locations
            topological_x = np.zeros(self.nx, dtype=bool)
            topological_x[x0 : x1 + 1] = True
            real_region = topological_x[:, None, None, None, None]
            center_region = topological_x[None, None, None, :, None]
            frame *= real_region == center_region
        norms = np.sqrt(np.sum(np.abs(frame) ** 2, axis=(0, 1, 2), keepdims=True))
        return frame / np.maximum(norms, 1e-15)

    def _operator_blocks(self, frame: np.ndarray) -> np.ndarray:
        # The dense convention is C_k = U_y^\dagger C U_y with U_y the
        # forward-FFT matrix.  Frames therefore transform with U_y^\dagger,
        # i.e. NumPy's orthonormal inverse FFT on their physical-y index.
        frame_k = np.fft.ifft(frame, axis=1, norm="ortho")
        columns = np.transpose(frame_k, (1, 0, 2, 3, 4)).reshape(
            self.ny, self.d, self.centers
        )
        return columns @ np.swapaxes(columns.conj(), -2, -1)

    def build_frame_operator_blocks(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        h = self._flattened_hamiltonian()
        v_minus = np.zeros((self.ny, self.d, self.d), dtype=np.complex128)
        v_plus = np.zeros_like(v_minus)
        checks: dict[str, float] = {}
        for band_name, band_sign, accumulator in (
            ("minus", -1, v_minus),
            ("plus", +1, v_plus),
        ):
            for trial_name, trial_sign in (("A", +1), ("B", -1)):
                frame = self._frame(h, band_sign=band_sign, trial_sign=trial_sign)
                norms = np.sum(np.abs(frame) ** 2, axis=(0, 1, 2))
                checks[f"{trial_name}_{band_name}_max_norm_error"] = float(
                    np.max(np.abs(norms - 1.0))
                )
                accumulator += self._operator_blocks(frame)
        return v_minus, v_plus, checks

    def stationary_solution(self) -> tuple[StationarySolution, dict[str, Any]]:
        started = time.perf_counter()
        v_minus, v_plus, checks = self.build_frame_operator_blocks()
        damping = 0.5 * (self.n_a * v_minus + (1.0 - self.n_a) * v_plus)
        source = self.n_a * v_minus
        damping = 0.5 * (damping + np.swapaxes(damping.conj(), -2, -1))
        source = 0.5 * (source + np.swapaxes(source.conj(), -2, -1))
        values, vectors = np.linalg.eigh(damping)
        source_eigen = np.swapaxes(vectors.conj(), -2, -1) @ source @ vectors
        denominator = values[:, :, None] + values[:, None, :]
        c_eigen = np.divide(
            source_eigen,
            denominator,
            out=np.zeros_like(source_eigen),
            where=np.abs(denominator) > 1e-13,
        )
        covariance = vectors @ c_eigen @ np.swapaxes(vectors.conj(), -2, -1)
        covariance = 0.5 * (covariance + np.swapaxes(covariance.conj(), -2, -1))
        residual = damping @ covariance + covariance @ damping - source
        residual_by_ky = np.linalg.norm(residual, axis=(-2, -1)) / np.maximum(
            np.linalg.norm(source, axis=(-2, -1)), 1e-300
        )
        solution = StationarySolution(
            ky=2 * np.pi * np.fft.fftfreq(self.ny),
            covariance_blocks=covariance,
            damping_blocks=damping,
            v_minus_blocks=v_minus,
            v_plus_blocks=v_plus,
            damping_eigenvalues=values,
            damping_eigenvectors=vectors,
            residual_by_ky=residual_by_ky,
        )
        return solution, {
            **checks,
            "stationary_relative_residual_max": float(np.max(residual_by_ky)),
            "damping_eigenvalue_min": float(np.min(values)),
            "damping_eigenvalue_max": float(np.max(values)),
            "frame_and_solve_seconds": float(time.perf_counter() - started),
        }

    def initial_blocks(self, mode: str) -> np.ndarray:
        identity = np.eye(self.d, dtype=np.complex128)
        if mode == "maxmix":
            return np.broadcast_to(0.5 * identity, (self.ny, self.d, self.d)).copy()
        if mode == "empty":
            return np.zeros((self.ny, self.d, self.d), dtype=np.complex128)
        if mode == "filled":
            return np.broadcast_to(identity, (self.ny, self.d, self.d)).copy()
        raise ValueError(f"unsupported deterministic initial covariance {mode!r}")

    def evolve_continuous(
        self, solution: StationarySolution, *, times: list[float], init_mode: str
    ) -> np.ndarray:
        initial = self.initial_blocks(init_mode)
        vectors = solution.damping_eigenvectors
        delta_eigen = np.swapaxes(vectors.conj(), -2, -1) @ (
            initial - solution.covariance_blocks
        ) @ vectors
        snapshots = []
        for value in times:
            decay = np.exp(-solution.damping_eigenvalues * float(value))
            evolved = decay[:, :, None] * delta_eigen * decay[:, None, :]
            block = solution.covariance_blocks + vectors @ evolved @ np.swapaxes(
                vectors.conj(), -2, -1
            )
            snapshots.append(0.5 * (block + np.swapaxes(block.conj(), -2, -1)))
        return np.stack(snapshots)

    @staticmethod
    def _matrix_exponential_from_eigh(
        values: np.ndarray, vectors: np.ndarray, coefficient: float
    ) -> np.ndarray:
        weights = np.exp(-values * float(coefficient))
        return (vectors * weights[:, None, :]) @ np.swapaxes(vectors.conj(), -2, -1)

    @staticmethod
    def _affine_channel_power(
        a: np.ndarray, b: np.ndarray, exponent: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact binary power of ``C -> A C A^dagger + B``.

        Composition stays inside the affine Gaussian-channel representation and
        reduces an O(T/p) time loop to O(log(T/p)) batched matrix products.
        """

        if int(exponent) < 0:
            raise ValueError("affine-channel exponent must be nonnegative")
        identity = np.broadcast_to(
            np.eye(a.shape[-1], dtype=np.complex128), a.shape
        ).copy()
        result_a = identity
        result_b = np.zeros_like(b)
        base_a = np.asarray(a, dtype=np.complex128)
        base_b = np.asarray(b, dtype=np.complex128)
        power = int(exponent)
        while power:
            if power & 1:
                result_b = base_a @ result_b @ np.swapaxes(base_a.conj(), -2, -1) + base_b
                result_a = base_a @ result_a
            base_b = base_a @ base_b @ np.swapaxes(base_a.conj(), -2, -1) + base_b
            base_a = base_a @ base_a
            power >>= 1
        return result_a, result_b

    def evolve_finite_channel(
        self,
        solution: StationarySolution,
        *,
        physical_time: float,
        p_values: list[float],
        init_mode: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        vm_values, vm_vectors = np.linalg.eigh(solution.v_minus_blocks)
        vp_values, vp_vectors = np.linalg.eigh(solution.v_plus_blocks)
        identity = np.broadcast_to(
            np.eye(self.d, dtype=np.complex128), (self.ny, self.d, self.d)
        )
        initial = self.initial_blocks(init_mode)
        exact = self.evolve_continuous(
            solution, times=[float(physical_time)], init_mode=init_mode
        )[0]
        endpoints = []
        steps = []
        errors = []
        stationary = []
        violations = []
        entropies = []
        histograms = []
        histogram_edges = np.linspace(0.0, 1.0, 201)
        for p in [float(value) for value in p_values]:
            nsteps = int(round(float(physical_time) / p))
            if nsteps < 1 or not math.isclose(
                nsteps * p, float(physical_time), rel_tol=1e-10, abs_tol=1e-12
            ):
                raise ValueError("physical_time must be an integer multiple of every p")
            gain_half = self._matrix_exponential_from_eigh(
                vm_values, vm_vectors, 0.25 * self.n_a * p
            )
            loss_full = self._matrix_exponential_from_eigh(
                vp_values, vp_vectors, 0.5 * (1.0 - self.n_a) * p
            )
            gain_offset = identity - gain_half @ gain_half
            one_step_a = gain_half @ loss_full @ gain_half
            one_step_b = (
                gain_half @ loss_full @ gain_offset @ loss_full @ gain_half
                + gain_offset
            )
            powered_a, powered_b = self._affine_channel_power(
                one_step_a, one_step_b, nsteps
            )
            block = powered_a @ initial @ np.swapaxes(powered_a.conj(), -2, -1) + powered_b
            block = 0.5 * (block + np.swapaxes(block.conj(), -2, -1))
            occupations = np.linalg.eigvalsh(block).real
            endpoints.append(occupations)
            steps.append(nsteps)
            errors.append(
                np.linalg.norm(block - exact) / max(np.linalg.norm(exact), 1e-300)
            )
            stationary.append(
                np.linalg.norm(block - solution.covariance_blocks)
                / math.sqrt(self.ny * self.d)
            )
            violations.append(
                max(0.0, float(-occupations.min()), float(occupations.max() - 1.0))
            )
            clipped = np.clip(occupations, 1e-15, 1.0 - 1e-15)
            entropies.append(
                float(np.sum(-clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped)))
            )
            histograms.append(np.histogram(np.clip(occupations, 0, 1), bins=histogram_edges)[0])
        arrays: dict[str, np.ndarray] = {
            "finite_channel_p": np.asarray(p_values, dtype=np.float64),
            "finite_channel_steps": np.asarray(steps, dtype=np.int64),
            "finite_channel_relative_error_to_continuous": np.asarray(errors),
            "finite_channel_stationary_distance": np.asarray(stationary),
            "finite_channel_physicality_violation": np.asarray(violations),
            "finite_channel_entropy": np.asarray(entropies),
            "finite_channel_occupation_histogram_edges": histogram_edges,
            "finite_channel_occupation_histogram_counts": np.asarray(histograms),
        }
        if self.nshell is None:
            arrays["finite_channel_occupation_spectrum_ky"] = np.asarray(endpoints)
        return arrays, {
            "finite_channel_definition": (
                "exact completely-positive Gaussian gain/loss semigroups composed by "
                "Strang splitting; p is the substep time"
            ),
            "finite_channel_splitting": "strang",
            "finite_channel_power_algorithm": "exact affine binary exponentiation",
            "finite_channel_physical_time": float(physical_time),
        }

    def finite_channel_homogeneous_power(
        self,
        solution: StationarySolution,
        *,
        physical_time: float,
        p: float,
    ) -> np.ndarray:
        """Return the homogeneous block of the exact finite channel at fixed time."""

        p = float(p)
        nsteps = int(round(float(physical_time) / p))
        if nsteps < 1 or not math.isclose(
            nsteps * p, float(physical_time), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("physical_time must be an integer multiple of p")
        vm_values, vm_vectors = np.linalg.eigh(solution.v_minus_blocks)
        vp_values, vp_vectors = np.linalg.eigh(solution.v_plus_blocks)
        gain_half = self._matrix_exponential_from_eigh(
            vm_values, vm_vectors, 0.25 * self.n_a * p
        )
        loss_full = self._matrix_exponential_from_eigh(
            vp_values, vp_vectors, 0.5 * (1.0 - self.n_a) * p
        )
        one_step_a = gain_half @ loss_full @ gain_half
        powered_a, _ = self._affine_channel_power(
            one_step_a, np.zeros_like(one_step_a), nsteps
        )
        return powered_a

    def _response_profiles_for_propagator(
        self,
        solution: StationarySolution,
        propagator: np.ndarray,
        *,
        epsilon: float,
        source_y: int,
        wall_window_columns: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return probe profiles, full-density L1 norms, and wall retention.

        The central density-phase response is rank four (two source orbitals), so only
        propagated vectors are materialized.  No full response covariance is formed.
        """

        source_y = int(source_y) % self.ny
        window = int(wall_window_columns)
        if window < 0:
            raise ValueError("wall_window_columns must be nonnegative")
        probe_locations = (
            self.wall_locations
            if self.wall_locations is not None
            else wall_interval(self.nx, self.wall_rule)
        )
        profiles = []
        norms = []
        retention = []
        sinc = math.sin(float(epsilon)) / float(epsilon)
        for probe_x in probe_locations:
            source_real = np.zeros((self.ny, self.d, 2), dtype=np.complex128)
            for orbital in (0, 1):
                source_real[source_y, 2 * int(probe_x) + orbital, orbital] = 1.0
            source_k = np.fft.ifft(source_real, axis=0, norm="ortho")
            covariance_source_k = solution.covariance_blocks @ source_k
            evolved_source_k = propagator @ source_k
            evolved_covariance_source_k = propagator @ covariance_source_k
            evolved_source = np.fft.fft(evolved_source_k, axis=0, norm="ortho")
            evolved_covariance_source = np.fft.fft(
                evolved_covariance_source_k, axis=0, norm="ortho"
            )
            diagonal = -2.0 * sinc * np.imag(
                np.sum(evolved_covariance_source * evolved_source.conj(), axis=2)
            )
            density_xy = diagonal.reshape(self.ny, self.nx, 2).sum(axis=2).T
            columns = sorted(
                {(int(probe_x) + offset) % self.nx for offset in range(-window, window + 1)}
            )
            profile = density_xy[columns].sum(axis=0)
            total_norm = float(np.sum(np.abs(density_xy)))
            profiles.append(profile)
            norms.append(total_norm)
            retention.append(
                float(np.sum(np.abs(density_xy[columns]))) / max(total_norm, 1e-300)
            )
        return np.asarray(profiles), np.asarray(norms), np.asarray(retention)

    def density_phase_response(
        self,
        solution: StationarySolution,
        *,
        epsilon: float,
        epsilon_multipliers: list[float],
        time_step: float,
        time_fraction: float,
        fit_time_min: float,
        fit_time_max_fraction: float,
        wall_window_columns: int,
        finite_channel_p: list[float],
        source_y: int = 0,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Compute the exact central-difference density response without sampling."""

        epsilon = float(epsilon)
        if epsilon <= 0.0:
            raise ValueError("response epsilon must be positive")
        horizon = float(time_fraction) * self.ny
        time_step = float(time_step)
        if horizon <= 0.0 or time_step <= 0.0:
            raise ValueError("response time_fraction and time_step must be positive")
        count = int(round(horizon / time_step))
        if not math.isclose(count * time_step, horizon, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("response horizon must be an integer multiple of time_step")
        times = np.arange(count + 1, dtype=np.float64) * time_step
        vectors = solution.damping_eigenvectors
        vectors_dagger = np.swapaxes(vectors.conj(), -2, -1)
        profiles = []
        norms = []
        retention = []
        for time_value in times:
            decay = np.exp(-solution.damping_eigenvalues * float(time_value))
            propagator = (vectors * decay[:, None, :]) @ vectors_dagger
            profile, norm, retained = self._response_profiles_for_propagator(
                solution,
                propagator,
                epsilon=epsilon,
                source_y=source_y,
                wall_window_columns=wall_window_columns,
            )
            profiles.append(profile)
            norms.append(norm)
            retention.append(retained)
        response = np.transpose(np.asarray(profiles), (1, 0, 2))
        response_norm = np.transpose(np.asarray(norms), (1, 0))
        wall_retention = np.transpose(np.asarray(retention), (1, 0))

        displacement = (np.arange(self.ny) + self.ny // 2) % self.ny - self.ny // 2
        absolute = np.abs(response)
        peak_index = np.argmax(absolute, axis=2)
        peak_position = displacement[peak_index].astype(np.float64)
        peak_position = np.unwrap(2.0 * np.pi * peak_position / self.ny, axis=1) * self.ny / (2.0 * np.pi)
        positive_weight = np.maximum(response, 0.0)
        negative_weight = np.maximum(-response, 0.0)
        positive_center = np.divide(
            positive_weight @ displacement,
            np.sum(positive_weight, axis=2),
            out=np.zeros(response.shape[:2], dtype=np.float64),
            where=np.sum(positive_weight, axis=2) > 1e-300,
        )
        negative_center = np.divide(
            negative_weight @ displacement,
            np.sum(negative_weight, axis=2),
            out=np.zeros(response.shape[:2], dtype=np.float64),
            where=np.sum(negative_weight, axis=2) > 1e-300,
        )
        signed_dipole = response @ displacement
        normalized_signed_dipole = np.divide(
            signed_dipole,
            response_norm,
            out=np.zeros_like(signed_dipole),
            where=response_norm > 1e-300,
        )
        positive = displacement > 0
        negative = displacement < 0
        right = np.sum(absolute[:, :, positive], axis=2)
        left = np.sum(absolute[:, :, negative], axis=2)
        asymmetry = np.divide(
            right - left,
            right + left,
            out=np.zeros_like(right),
            where=(right + left) > 1e-300,
        )
        fit_mask = (times >= float(fit_time_min)) & (
            times <= float(fit_time_max_fraction) * self.ny
        )
        velocities = np.full(response.shape[0], np.nan, dtype=np.float64)
        velocity_intercepts = np.full_like(velocities, np.nan)
        velocity_r2 = np.full_like(velocities, np.nan)
        mean_asymmetry = np.full_like(velocities, np.nan)
        mean_directionality = np.full_like(velocities, np.nan)
        for wall_index in range(response.shape[0]):
            amplitude = response_norm[wall_index]
            if float(np.max(amplitude)) <= 1e-14:
                velocities[wall_index] = 0.0
                velocity_intercepts[wall_index] = 0.0
                velocity_r2[wall_index] = 1.0
                mean_asymmetry[wall_index] = 0.0
                mean_directionality[wall_index] = 0.0
                continue
            active = fit_mask & (amplitude > max(float(np.max(amplitude)) * 1e-8, 1e-15))
            if np.count_nonzero(active) >= 3:
                slope, intercept = np.polyfit(times[active], positive_center[wall_index, active], 1)
                predicted = slope * times[active] + intercept
                observed = positive_center[wall_index, active]
                denominator = float(np.sum((observed - np.mean(observed)) ** 2))
                residual = float(np.sum((observed - predicted) ** 2))
                velocities[wall_index] = float(slope)
                velocity_intercepts[wall_index] = float(intercept)
                velocity_r2[wall_index] = 1.0 - residual / denominator if denominator > 0 else 1.0
                mean_asymmetry[wall_index] = float(np.mean(asymmetry[wall_index, active]))
                mean_directionality[wall_index] = float(
                    np.mean(normalized_signed_dipole[wall_index, active])
                )

        final_continuous = response[:, -1]
        finite_errors = []
        for p in [float(value) for value in finite_channel_p]:
            finite_propagator = self.finite_channel_homogeneous_power(
                solution, physical_time=horizon, p=p
            )
            finite_profile, _, _ = self._response_profiles_for_propagator(
                solution,
                finite_propagator,
                epsilon=epsilon,
                source_y=source_y,
                wall_window_columns=wall_window_columns,
            )
            finite_errors.append(
                np.linalg.norm(finite_profile - final_continuous)
                / max(np.linalg.norm(final_continuous), 1e-300)
            )

        multipliers = np.asarray(epsilon_multipliers, dtype=np.float64)
        sinc_reference = math.sin(epsilon) / epsilon
        linearity = np.abs(
            np.sin(epsilon * multipliers) / (epsilon * multipliers) / sinc_reference - 1.0
        )
        occupations = np.linalg.eigvalsh(solution.covariance_blocks).real
        arrays = {
            "response_times": times,
            "response_probe_x": np.asarray(
                self.wall_locations
                if self.wall_locations is not None
                else wall_interval(self.nx, self.wall_rule),
                dtype=np.int64,
            ),
            "response_source_y": np.asarray(int(source_y) % self.ny, dtype=np.int64),
            "response_density_ty": response,
            "response_norm_time": response_norm,
            "response_wall_retention_time": wall_retention,
            "response_peak_position_time": peak_position,
            "response_positive_center_time": positive_center,
            "response_negative_center_time": negative_center,
            "response_signed_dipole_time": signed_dipole,
            "response_normalized_signed_dipole_time": normalized_signed_dipole,
            "response_right_left_asymmetry_time": asymmetry,
            "response_velocity": velocities,
            "response_velocity_intercept": velocity_intercepts,
            "response_velocity_r2": velocity_r2,
            "response_mean_asymmetry": mean_asymmetry,
            "response_mean_directionality": mean_directionality,
            "response_epsilon_multipliers": multipliers,
            "response_epsilon_relative_error": linearity,
            "response_finite_channel_p": np.asarray(finite_channel_p, dtype=np.float64),
            "response_finite_channel_relative_error": np.asarray(finite_errors),
        }
        metadata = {
            "response_probe": "exact_plus_minus_epsilon_local_density_phase_unitary",
            "response_epsilon": epsilon,
            "response_horizon": horizon,
            "response_time_step": time_step,
            "response_fit_window": [float(fit_time_min), float(fit_time_max_fraction) * self.ny],
            "response_velocity_center_method": "positive_signed_response_center",
            "response_absolute_peak_is_diagnostic_only": True,
            "response_kick_occupation_min": float(np.min(occupations)),
            "response_kick_occupation_max": float(np.max(occupations)),
            "response_kick_physicality_violation": max(
                0.0, float(-np.min(occupations)), float(np.max(occupations) - 1.0)
            ),
            "response_covariance_materialized": False,
        }
        return arrays, metadata

    def analyze(
        self,
        solution: StationarySolution,
        snapshots: np.ndarray,
        *,
        times: list[float],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        occupations_time = np.linalg.eigvalsh(snapshots).real
        occupations, eigenvectors = np.linalg.eigh(solution.covariance_blocks)
        occupations = occupations.real
        x_weights = np.sum(
            np.abs(eigenvectors.reshape(self.ny, self.nx, 2, self.d)) ** 2, axis=2
        )
        clipped = np.clip(occupations_time, 1e-15, 1.0 - 1e-15)
        entropy_time = np.sum(
            -clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped),
            axis=(1, 2),
        )
        variance_time = np.sum(occupations_time * (1.0 - occupations_time), axis=(1, 2))
        distance_time = np.sqrt(
            np.sum(np.linalg.norm(snapshots - solution.covariance_blocks[None], axis=(-2, -1)) ** 2, axis=1)
        ) / math.sqrt(self.ny * self.d)
        histogram_edges = np.linspace(0.0, 1.0, 201)
        histogram_counts = np.histogram(
            np.clip(occupations.reshape(-1), 0.0, 1.0), bins=histogram_edges
        )[0]
        flat_order = np.argsort(np.abs(occupations.reshape(-1) - 0.5))
        selected = flat_order[: min(2 * self.ny, flat_order.size)]
        k_index, mode_index = np.unravel_index(selected, occupations.shape)
        midgap_profile = np.mean(x_weights[k_index, :, mode_index], axis=0)
        rates = (
            solution.damping_eigenvalues[:, :, None]
            + solution.damping_eigenvalues[:, None, :]
        ).reshape(-1)
        leading_rates = np.sort(rates)[: min(128, rates.size)]
        wall_union: set[int] = set()
        if self.wall_locations is not None:
            for wall in self.wall_locations:
                wall_union.update((wall + delta) % self.nx for delta in (-1, 0, 1))
        wall_total = (
            np.sum(x_weights[:, sorted(wall_union), :], axis=1)
            if wall_union
            else np.zeros_like(occupations)
        )
        bulk_mask = wall_total < 0.25
        bulk_gap: float | None = (
            float(np.min(np.abs(occupations[bulk_mask] - 0.5)))
            if np.any(bulk_mask)
            else None
        )
        arrays: dict[str, np.ndarray] = {
            "times": np.asarray(times, dtype=np.float64),
            "entropy_time": entropy_time,
            "intrinsic_charge_variance_time": variance_time,
            "stationary_distance_time": distance_time,
            "occupation_histogram_edges": histogram_edges,
            "occupation_histogram_counts": histogram_counts,
            "wall_midgap_x_profile": midgap_profile,
            "leading_integrated_decay_rates": leading_rates,
        }
        branch_slopes: list[dict[str, Any]] = []
        pairing_residual_max: float | None = None
        if self.nshell is None:
            negative = (-np.arange(self.ny)) % self.ny
            pairing = occupations + occupations[negative, ::-1] - 1.0
            pairing_residual_max = float(np.max(np.abs(pairing)))
            branch_occupations = []
            branch_weights = []
            all_wall_weights = []
            if self.wall_locations is not None:
                for wall in self.wall_locations:
                    indices = sorted({(wall + delta) % self.nx for delta in (-1, 0, 1)})
                    weights = np.sum(x_weights[:, indices, :], axis=1)
                    # Select the interface mode nearest half occupation, subject to a
                    # genuine wall-localization threshold.  This avoids an arbitrary
                    # switch to a nearly filled wall-localized bulk mode at an exact
                    # degeneracy (especially k_y=0).
                    eligible = weights >= 0.25
                    cost = np.where(
                        eligible,
                        np.abs(occupations - 0.5) + 0.02 * (1.0 - weights),
                        np.inf,
                    )
                    mode = np.argmin(cost, axis=1)
                    missing = ~np.any(eligible, axis=1)
                    if np.any(missing):
                        mode[missing] = np.argmax(weights[missing], axis=1)
                    selected_occupation = occupations[np.arange(self.ny), mode]
                    selected_weight = weights[np.arange(self.ny), mode]
                    cutoff = max(2.5 * 2.0 * np.pi / self.ny, 0.22)
                    fit_mask = np.abs(solution.ky) <= cutoff
                    slope, intercept = np.polyfit(
                        solution.ky[fit_mask], selected_occupation[fit_mask], 1
                    )
                    branch_slopes.append(
                        {
                            "wall_x": int(wall),
                            "slope_dnu_dky": float(slope),
                            "intercept": float(intercept),
                            "fit_points": int(np.sum(fit_mask)),
                            "minimum_selected_wall_weight": float(np.min(selected_weight)),
                        }
                    )
                    all_wall_weights.append(weights)
                    branch_occupations.append(selected_occupation)
                    branch_weights.append(selected_weight)
            arrays.update(
                {
                    "ky": solution.ky,
                    "occupation_spectrum_ky": occupations,
                    "x_weights_ky": x_weights,
                    "occupation_pairing_residual_ky": pairing,
                    "stationary_residual_ky": solution.residual_by_ky,
                    "single_particle_damping_rates_ky": solution.damping_eigenvalues,
                    "wall_branch_occupations_ky": np.asarray(branch_occupations),
                    "wall_branch_weights_ky": np.asarray(branch_weights),
                    "all_wall_mode_weights_ky": np.asarray(all_wall_weights),
                }
            )
        diagnostics = {
            "occupation_min": float(np.min(occupations)),
            "occupation_max": float(np.max(occupations)),
            "stationary_half_occupation_gap": float(np.min(np.abs(occupations - 0.5))),
            "bulk_half_occupation_gap": bulk_gap,
            "stationary_global_gaussian_entropy": float(entropy_time[-1]),
            "stationary_entropy_per_circumference": float(entropy_time[-1] / self.ny),
            "stationary_intrinsic_charge_variance": float(variance_time[-1]),
            "occupation_pairing_residual_max": pairing_residual_max,
            "translation_invariance": "exact by y-translation block construction",
            "translation_residual": 0.0,
            "branch_slopes": branch_slopes,
            "wall_locations": None if self.wall_locations is None else list(self.wall_locations),
        }
        validate_selected_observable_schema(self.nshell, arrays)
        return arrays, diagnostics


def run_gain_loss_case(case: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    model = case["model"]
    run = case["run"]
    solver = MeanChannelLindbladCPU(
        nx=model["Nx"],
        ny=model["Ny"],
        alpha_top=model["alpha_top"],
        alpha_triv=model["alpha_triv"],
        domain_wall=model["domain_wall"],
        dw_truncation=model["dw_truncation"],
        wall_rule=model["wall_rule"],
        nshell=model["nshell"],
        n_a=model["n_a"],
    )
    solution, solve = solver.stationary_solution()
    fractions = [float(value) for value in run["observation_time_fractions"]]
    times = [fraction * solver.ny for fraction in fractions]
    snapshots = solver.evolve_continuous(
        solution, times=times, init_mode=run["init_mode"]
    )
    arrays, diagnostics = solver.analyze(solution, snapshots, times=times)
    finite_arrays, finite_metadata = solver.evolve_finite_channel(
        solution,
        physical_time=float(run["physical_time_fraction"]) * solver.ny,
        p_values=run["finite_channel_p"],
        init_mode=run["init_mode"],
    )
    arrays.update(finite_arrays)
    response_metadata: dict[str, Any] = {"response_enabled": False}
    response_config = run.get("response", {})
    if bool(response_config.get("enabled", False)):
        response_arrays, response_metadata = solver.density_phase_response(
            solution,
            epsilon=float(response_config["epsilon"]),
            epsilon_multipliers=list(response_config["epsilon_multipliers"]),
            time_step=float(response_config["time_step"]),
            time_fraction=float(response_config["time_fraction"]),
            fit_time_min=float(response_config["fit_time_min"]),
            fit_time_max_fraction=float(response_config["fit_time_max_fraction"]),
            wall_window_columns=int(response_config["wall_window_columns"]),
            finite_channel_p=list(run["finite_channel_p"]),
        )
        arrays.update(response_arrays)
        response_metadata = {"response_enabled": True, **response_metadata}
    validate_selected_observable_schema(solver.nshell, arrays)
    metadata = {
        "schema": "mean_channel_lindblad_cpu_case_v2_hybrid_response",
        "case_id": case["case_id"],
        "campaign": case["campaign"],
        "canonical_entry_point": "mean_channel_lindblad_cpu.run_gain_loss_case",
        "backend": "numpy_scipy_cpu_complex128",
        "trajectory_samples": 0,
        "schedule_samples": 0,
        "saved_covariance_history": False,
        "permanent_covariance_bytes": 0,
        "model_family": "finite Gaussian bath channel and continuous gain/loss generator",
        "historical_fresh_ancilla_replacement_channel_is_distinct": True,
        "conditioned_trajectory_dynamics_is_distinct": True,
        "model": model,
        "run": run,
        "solve": solve,
        "diagnostics": diagnostics,
        "response": response_metadata,
        **finite_metadata,
    }
    return arrays, metadata


def run_dephasing_control(case: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Small-system full historical two-point closure, including dephasing."""

    model = case["model"]
    run = case["run"]
    solver = MeanChannelLindbladCPU(
        nx=model["Nx"], ny=model["Ny"], alpha_top=model["alpha_top"],
        alpha_triv=model["alpha_triv"], domain_wall=model["domain_wall"],
        dw_truncation=model["dw_truncation"], wall_rule=model["wall_rule"],
        nshell=model["nshell"], n_a=model["n_a"],
    )
    h = solver._flattened_hamiltonian()
    families: dict[str, np.ndarray] = {}
    for name, band_sign, trial_sign in (
        ("A_minus", -1, +1), ("B_minus", -1, -1),
        ("A_plus", +1, +1), ("B_plus", +1, -1),
    ):
        families[name] = solver._frame(
            h, band_sign=band_sign, trial_sign=trial_sign
        ).reshape(solver.n, solver.centers)
    v_family = {name: frame @ frame.conj().T for name, frame in families.items()}
    v_minus = v_family["A_minus"] + v_family["B_minus"]
    v_plus = v_family["A_plus"] + v_family["B_plus"]

    def action(covariance: np.ndarray) -> np.ndarray:
        out = solver.n_a * (
            v_minus - 0.5 * (v_minus @ covariance + covariance @ v_minus)
        ) - 0.5 * (1.0 - solver.n_a) * (
            v_plus @ covariance + covariance @ v_plus
        )
        for name, coefficient in (
            ("A_minus", 2.0 - solver.n_a), ("B_minus", 2.0 - solver.n_a),
            ("A_plus", 1.0 + solver.n_a), ("B_plus", 1.0 + solver.n_a),
        ):
            frame = families[name]
            v = v_family[name]
            expectation = np.einsum(
                "ia,ij,ja->a", frame.conj(), covariance, frame, optimize=True
            )
            projected = (frame * expectation[None, :]) @ frame.conj().T
            out += -0.5 * coefficient * (
                v @ covariance + covariance @ v - 2.0 * projected
            )
        return out

    dt = float(run["dt"])
    tmax = float(run["physical_time_fraction"]) * solver.ny
    steps = int(round(tmax / dt))
    if not math.isclose(steps * dt, tmax, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("dephasing tmax must be an integer multiple of dt")
    fractions = [float(value) for value in run["observation_time_fractions"]]
    save_steps = {int(round(fraction * solver.ny / dt)): index for index, fraction in enumerate(fractions)}
    if run["init_mode"] == "maxmix":
        covariance = 0.5 * np.eye(solver.n, dtype=np.complex128)
    elif run["init_mode"] == "empty":
        covariance = np.zeros((solver.n, solver.n), dtype=np.complex128)
    elif run["init_mode"] == "filled":
        covariance = np.eye(solver.n, dtype=np.complex128)
    else:
        raise ValueError(f"unsupported dephasing initial mode {run['init_mode']!r}")
    snapshots: list[np.ndarray | None] = [None] * len(fractions)
    if 0 in save_steps:
        snapshots[save_steps[0]] = covariance.copy()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        k1 = action(covariance)
        k2 = action(covariance + 0.5 * dt * k1)
        k3 = action(covariance + 0.5 * dt * k2)
        k4 = action(covariance + dt * k3)
        covariance = covariance + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        covariance = 0.5 * (covariance + covariance.conj().T)
        if step in save_steps:
            snapshots[save_steps[step]] = covariance.copy()
    if any(snapshot is None for snapshot in snapshots):
        raise RuntimeError("dephasing observation grid did not align with integration steps")
    values = np.stack([np.linalg.eigvalsh(snapshot).real for snapshot in snapshots])
    clipped = np.clip(values, 1e-15, 1.0 - 1e-15)
    entropy = np.sum(
        -clipped * np.log(clipped) - (1.0 - clipped) * np.log(1.0 - clipped), axis=1
    )
    variance = np.sum(values * (1.0 - values), axis=1)
    arrays = {
        "times": np.asarray(fractions) * solver.ny,
        "entropy_time": entropy,
        "intrinsic_charge_variance_time": variance,
        "successive_observation_distance": np.asarray(
            [np.nan] + [np.linalg.norm(snapshots[i] - snapshots[i - 1]) / math.sqrt(solver.n) for i in range(1, len(snapshots))]
        ),
        "occupation_histogram_edges": np.linspace(0.0, 1.0, 201),
        "occupation_histogram_counts": np.histogram(
            np.clip(values[-1], 0, 1), bins=np.linspace(0.0, 1.0, 201)
        )[0],
    }
    validate_selected_observable_schema(model["nshell"] if model["nshell"] is not None else 0, arrays)
    metadata = {
        "schema": "historical_dephasing_control_v1",
        "case_id": case["case_id"],
        "campaign": case["campaign"],
        "canonical_entry_point": "mean_channel_lindblad_cpu.run_dephasing_control",
        "backend": "numpy_scipy_cpu_complex128_rk4",
        "trajectory_samples": 0,
        "schedule_samples": 0,
        "saved_covariance_history": False,
        "permanent_covariance_bytes": 0,
        "model_family": "historical bath-density two-point closure with dephasing",
        "model": model,
        "run": run,
        "diagnostics": {
            "occupation_min": float(values[-1].min()),
            "occupation_max": float(values[-1].max()),
            "physicality_violation": max(0.0, float(-values[-1].min()), float(values[-1].max() - 1.0)),
            "elapsed_seconds": float(time.perf_counter() - started),
            "translation_invariance": "not reported as momentum-resolved output",
        },
    }
    return arrays, metadata


def run_case(case: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if case["kind"] == "gain_loss":
        return run_gain_loss_case(case)
    if case["kind"] == "dephasing_control":
        return run_dephasing_control(case)
    raise ValueError(f"unknown CPU campaign case kind {case['kind']!r}")
