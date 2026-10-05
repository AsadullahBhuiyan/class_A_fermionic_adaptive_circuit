"""Matched two-point Lindblad dynamics for canonical OW projectors.

This module evolves the physical correlation matrix

    G_ij = Tr(rho c_i^dagger c_j)

for the infinitesimal perfect-correction channel.  It intentionally consumes
the ``WF_Ap``, ``WF_Am``, ``WF_Bp``, and ``WF_Bm`` arrays constructed by
``classA_U1FGTN`` instead of rebuilding overcomplete-Wannier (OW) states.

The number-dephasing jumps make the many-body state non-Gaussian, but their
action on the two-point function closes exactly and remains affine-linear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
from tqdm.auto import tqdm


Array = np.ndarray

FAMILY_NAMES = ("A_minus", "B_minus", "A_plus", "B_plus")
LOWER_FAMILIES = ("A_minus", "B_minus")
UPPER_FAMILIES = ("A_plus", "B_plus")
CANONICAL_FRAME_ATTRIBUTES = {
    "A_minus": "WF_Am",
    "B_minus": "WF_Bm",
    "A_plus": "WF_Ap",
    "B_plus": "WF_Bp",
}


def _as_complex_matrix(matrix: Any, dimension: int, *, name: str) -> Array:
    result = np.asarray(matrix, dtype=np.complex128)
    expected = (int(dimension), int(dimension))
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite entries.")
    return result


def _projector_from_mode(mode: Any) -> Array:
    vector = np.asarray(mode, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-15:
        raise ValueError("mode must have a finite nonzero norm.")
    vector = vector / norm
    return np.outer(vector, vector.conj())


def linear_loss_rhs(correlation: Any, mode: Any) -> Array:
    """Return the two-point action of ``D[chi]`` for one normalized mode."""

    projector = _projector_from_mode(mode)
    matrix = _as_complex_matrix(correlation, projector.shape[0], name="correlation")
    return -0.5 * (projector @ matrix + matrix @ projector)


def linear_gain_rhs(correlation: Any, mode: Any) -> Array:
    """Return the two-point action of ``D[chi^dagger]`` for one mode."""

    projector = _projector_from_mode(mode)
    matrix = _as_complex_matrix(correlation, projector.shape[0], name="correlation")
    return projector - 0.5 * (projector @ matrix + matrix @ projector)


def number_dephasing_rhs(correlation: Any, mode: Any) -> Array:
    """Return the exact two-point action of ``D[chi^dagger chi]``."""

    projector = _projector_from_mode(mode)
    matrix = _as_complex_matrix(correlation, projector.shape[0], name="correlation")
    return -0.5 * (
        projector @ matrix
        + matrix @ projector
        - 2.0 * projector @ matrix @ projector
    )


def perfect_correction_mode_rhs(
    correlation: Any,
    mode: Any,
    *,
    target_occupied: bool,
    include_number_dephasing: bool,
) -> Array:
    """Infinitesimal perfect-correction action for one target mode."""

    if bool(target_occupied):
        result = linear_gain_rhs(correlation, mode)
    else:
        result = linear_loss_rhs(correlation, mode)
    if include_number_dephasing:
        result = result + number_dephasing_rhs(correlation, mode)
    return result


def dense_to_y_momentum(correlation: Any, *, nx: int, ny: int) -> Array:
    """Return ``F_y G F_y^dagger`` with shape ``(ky,d,ky',d)``.

    The canonical single-particle ordering is ``mu + 2*x + 2*Nx*y`` and
    ``d = 2*Nx``.  Both transforms use the orthonormal FFT convention.
    """

    nx, ny = int(nx), int(ny)
    dimension, block_dimension = 2 * nx * ny, 2 * nx
    matrix = _as_complex_matrix(correlation, dimension, name="correlation")
    tensor = matrix.reshape(ny, block_dimension, ny, block_dimension)
    return np.fft.ifft(
        np.fft.fft(tensor, axis=0, norm="ortho"),
        axis=2,
        norm="ortho",
    )


def y_momentum_to_dense(momentum_matrix: Any, *, nx: int, ny: int) -> Array:
    """Inverse of :func:`dense_to_y_momentum`."""

    nx, ny = int(nx), int(ny)
    block_dimension = 2 * nx
    tensor = np.asarray(momentum_matrix, dtype=np.complex128)
    expected = (ny, block_dimension, ny, block_dimension)
    if tensor.shape != expected:
        raise ValueError(
            f"momentum_matrix must have shape {expected}, got {tensor.shape}."
        )
    real_tensor = np.fft.fft(
        np.fft.ifft(tensor, axis=0, norm="ortho"),
        axis=2,
        norm="ortho",
    )
    return real_tensor.reshape(2 * nx * ny, 2 * nx * ny)


def extract_q_sector(momentum_matrix: Any, q_index: int) -> Array:
    """Extract blocks ``X_q(k)=G(k,k-q)`` from a y-momentum matrix."""

    tensor = np.asarray(momentum_matrix, dtype=np.complex128)
    if tensor.ndim != 4 or tensor.shape[0] != tensor.shape[2]:
        raise ValueError("momentum_matrix must have shape (Ny,d,Ny,d).")
    if tensor.shape[1] != tensor.shape[3]:
        raise ValueError("momentum_matrix row and column block sizes must agree.")
    ny = tensor.shape[0]
    q_index = int(q_index) % ny
    k = np.arange(ny)
    return np.asarray(tensor[k, :, (k - q_index) % ny, :], dtype=np.complex128)


def embed_q_sector(sector: Any, q_index: int) -> Array:
    """Embed ``X_q(k)`` into an otherwise-zero y-momentum matrix."""

    blocks = np.asarray(sector, dtype=np.complex128)
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("sector must have shape (Ny,d,d).")
    ny, block_dimension, _ = blocks.shape
    q_index = int(q_index) % ny
    result = np.zeros(
        (ny, block_dimension, ny, block_dimension), dtype=np.complex128
    )
    k = np.arange(ny)
    result[k, :, (k - q_index) % ny, :] = blocks
    return result


@dataclass(frozen=True)
class RK4Evolution:
    """Selected snapshots from fixed-step fourth-order Runge--Kutta evolution."""

    times: Array
    steps: Array
    states: Array
    dt: float


def integrate_rk4(
    rhs: Callable[[Array], Array],
    initial: Any,
    *,
    dt: float,
    observation_times: Any,
    hermitize: bool = False,
    progress: bool = False,
) -> RK4Evolution:
    """Integrate an autonomous array-valued ODE at grid-aligned times.

    Only requested observations are retained.  Requiring observations to align
    with the fixed time step keeps campaign time labels exact and reproducible.
    """

    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite scalar.")
    times = np.asarray(observation_times, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("observation_times must contain at least one time.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("observation_times must be finite and nonnegative.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("observation_times must be sorted in nondecreasing order.")
    steps = np.rint(times / dt).astype(np.int64)
    if not np.allclose(steps * dt, times, rtol=1e-11, atol=1e-13):
        raise ValueError("every observation time must be an integer multiple of dt.")

    state = np.asarray(initial, dtype=np.complex128).copy()
    if not np.all(np.isfinite(state)):
        raise ValueError("initial state contains non-finite entries.")

    def _hermitize(value: Array) -> Array:
        return 0.5 * (value + np.swapaxes(value.conj(), -1, -2))

    snapshots: list[Array] = []
    save_positions: dict[int, list[int]] = {}
    for position, step in enumerate(steps.tolist()):
        save_positions.setdefault(int(step), []).append(position)
    ordered: list[Array | None] = [None] * times.size
    if 0 in save_positions:
        saved = _hermitize(state) if hermitize else state
        for position in save_positions[0]:
            ordered[position] = saved.copy()

    for step in tqdm(range(1, int(steps[-1]) + 1), desc='Lindblad RK4',
                     unit='step', disable=not progress):
        k1 = np.asarray(rhs(state), dtype=np.complex128)
        k2 = np.asarray(rhs(state + 0.5 * dt * k1), dtype=np.complex128)
        k3 = np.asarray(rhs(state + 0.5 * dt * k2), dtype=np.complex128)
        k4 = np.asarray(rhs(state + dt * k3), dtype=np.complex128)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if hermitize:
            state = _hermitize(state)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError(f"non-finite RK4 state at step {step}.")
        if step in save_positions:
            for position in save_positions[step]:
                ordered[position] = state.copy()

    if any(snapshot is None for snapshot in ordered):
        raise RuntimeError("an RK4 observation was not populated.")
    snapshots = [np.asarray(snapshot) for snapshot in ordered]
    return RK4Evolution(
        times=times,
        steps=steps,
        states=np.stack(snapshots, axis=0),
        dt=dt,
    )


class PerfectCorrectionLindblad:
    """Perfect-correction two-point generator built from canonical OW frames."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        frames: Mapping[str, Any],
        twist_y: float = 0.0,
        normalization_tolerance: float = 1e-10,
    ) -> None:
        self.nx, self.ny = int(nx), int(ny)
        if self.nx < 1 or self.ny < 1:
            raise ValueError("nx and ny must be positive integers.")
        self.dimension = 2 * self.nx * self.ny
        self.block_dimension = 2 * self.nx
        self.centers = self.nx * self.ny
        self.twist_y = float(twist_y)
        if not np.isfinite(self.twist_y):
            raise ValueError("twist_y must be finite.")
        normalization_tolerance = float(normalization_tolerance)
        if normalization_tolerance <= 0.0:
            raise ValueError("normalization_tolerance must be positive.")

        missing = sorted(set(FAMILY_NAMES).difference(frames))
        if missing:
            raise ValueError(f"missing OW frame families: {missing}")
        self.frames: dict[str, Array] = {}
        self.frame_norm_error: dict[str, float] = {}
        expected = (self.dimension, self.nx, self.ny)
        for name in FAMILY_NAMES:
            frame = np.asarray(frames[name], dtype=np.complex128)
            if frame.shape != expected:
                raise ValueError(
                    f"frame {name!r} must have shape {expected}, got {frame.shape}."
                )
            if not np.all(np.isfinite(frame)):
                raise ValueError(f"frame {name!r} contains non-finite entries.")
            norms = np.sum(np.abs(frame) ** 2, axis=0)
            error = float(np.max(np.abs(norms - 1.0)))
            if error > normalization_tolerance:
                raise ValueError(
                    f"frame {name!r} is not normalized per center; max error={error:.3e}."
                )
            self.frames[name] = np.ascontiguousarray(frame)
            self.frame_norm_error[name] = error

        self._dense_operators: dict[str, Array] | None = None
        self._momentum_modes: dict[str, Array] | None = None
        self._momentum_operators: dict[str, Array] | None = None
        self._translation_residual: float | None = None

    @classmethod
    def from_canonical_model(
        cls,
        model: Any,
        *,
        construct_if_missing: bool = True,
        normalization_tolerance: float = 1e-10,
    ) -> "PerfectCorrectionLindblad":
        """Consume OW arrays from a ``classA_U1FGTN``-compatible instance."""

        missing_attributes = [
            attribute
            for attribute in CANONICAL_FRAME_ATTRIBUTES.values()
            if not hasattr(model, attribute)
        ]
        if missing_attributes and construct_if_missing:
            model.construct_OW_projectors(
                nshell=model.nshell,
                DW=model.DW,
                trial_orbitals=model.trial_orbitals,
                dw_truncation=model.dw_truncation,
                twist_y=getattr(model, "twist_y", 0.0),
            )
            missing_attributes = [
                attribute
                for attribute in CANONICAL_FRAME_ATTRIBUTES.values()
                if not hasattr(model, attribute)
            ]
        if missing_attributes:
            raise ValueError(
                "canonical model is missing OW arrays: "
                + ", ".join(missing_attributes)
            )
        frames = {
            family: getattr(model, attribute)
            for family, attribute in CANONICAL_FRAME_ATTRIBUTES.items()
        }
        return cls(
            nx=int(model.Nx),
            ny=int(model.Ny),
            frames=frames,
            twist_y=float(getattr(model, "twist_y", 0.0)),
            normalization_tolerance=normalization_tolerance,
        )

    def _ensure_dense_operators(self) -> dict[str, Array]:
        if self._dense_operators is None:
            operators = {}
            for name, frame in self.frames.items():
                columns = frame.reshape(self.dimension, self.centers)
                operators[name] = columns @ columns.conj().T
            self._dense_operators = operators
        return self._dense_operators

    @property
    def dense_frame_operators(self) -> Mapping[str, Array]:
        """Per-family frame operators ``sum_R |W_R><W_R|``."""

        return self._ensure_dense_operators()

    def dense_homogeneous_rhs(
        self,
        correlation: Any,
        *,
        include_number_dephasing: bool,
    ) -> Array:
        """Return the source-free dense perfect-correction generator action."""

        matrix = _as_complex_matrix(
            correlation, self.dimension, name="correlation"
        )
        operators = self._ensure_dense_operators()
        v_minus = operators["A_minus"] + operators["B_minus"]
        v_plus = operators["A_plus"] + operators["B_plus"]
        total = v_minus + v_plus
        result = -0.5 * (total @ matrix + matrix @ total)
        if include_number_dephasing:
            for name in FAMILY_NAMES:
                frame = self.frames[name].reshape(self.dimension, self.centers)
                operator = operators[name]
                expectation = np.einsum(
                    "ia,ij,ja->a",
                    frame.conj(),
                    matrix,
                    frame,
                    optimize=True,
                )
                recycled = (frame * expectation[None, :]) @ frame.conj().T
                result += -0.5 * (
                    operator @ matrix + matrix @ operator - 2.0 * recycled
                )
        return result

    def dense_rhs(
        self,
        correlation: Any,
        *,
        include_number_dephasing: bool,
    ) -> Array:
        """Return ``dG/dt`` for the dense affine perfect-correction equation."""

        operators = self._ensure_dense_operators()
        source = operators["A_minus"] + operators["B_minus"]
        return source + self.dense_homogeneous_rhs(
            correlation,
            include_number_dephasing=include_number_dephasing,
        )

    def _ensure_momentum_frames(self) -> None:
        if self._momentum_modes is not None:
            return
        if not np.isclose(self.twist_y, 0.0, rtol=0.0, atol=1e-14):
            raise NotImplementedError(
                "q-sector helpers currently require twist_y=0; dense evolution remains valid."
            )
        modes: dict[str, Array] = {}
        operators: dict[str, Array] = {}
        maximum_residual = 0.0
        momenta = 2.0 * np.pi * np.fft.fftfreq(self.ny)
        ry = np.arange(self.ny)
        phase = np.exp(-1j * momenta[:, None] * ry[None, :])
        for name, frame in self.frames.items():
            frame_y = frame.reshape(
                self.ny,
                self.block_dimension,
                self.nx,
                self.ny,
            )
            transformed = np.fft.fft(frame_y, axis=0, norm="ortho")
            reference = transformed[:, :, :, 0]
            predicted = reference[:, :, :, None] * phase[:, None, None, :]
            maximum_residual = max(
                maximum_residual,
                float(np.max(np.abs(transformed - predicted))),
            )
            mode = np.sqrt(self.ny) * reference
            modes[name] = mode
            operators[name] = np.einsum(
                "kir,kjr->kij", mode, mode.conj(), optimize=True
            )
        if maximum_residual > 5e-10:
            raise ValueError(
                "canonical OW frames are not y-translation covariant; "
                f"maximum residual={maximum_residual:.3e}."
            )
        self._momentum_modes = modes
        self._momentum_operators = operators
        self._translation_residual = maximum_residual

    @property
    def y_translation_residual(self) -> float:
        self._ensure_momentum_frames()
        assert self._translation_residual is not None
        return self._translation_residual

    @property
    def momentum_modes(self) -> Mapping[str, Array]:
        self._ensure_momentum_frames()
        assert self._momentum_modes is not None
        return self._momentum_modes

    @property
    def momentum_frame_operators(self) -> Mapping[str, Array]:
        self._ensure_momentum_frames()
        assert self._momentum_operators is not None
        return self._momentum_operators

    def q_sector_homogeneous_rhs(
        self,
        sector: Any,
        *,
        q_index: int,
        include_number_dephasing: bool,
    ) -> Array:
        """Return the source-free generator on ``X_q(k)=G(k,k-q)``."""

        self._ensure_momentum_frames()
        assert self._momentum_modes is not None
        assert self._momentum_operators is not None
        blocks = np.asarray(sector, dtype=np.complex128)
        expected_tail = (self.ny, self.block_dimension, self.block_dimension)
        if blocks.ndim < 3 or blocks.shape[-3:] != expected_tail:
            raise ValueError(
                "sector must have shape (...,Ny,d,d) with trailing shape "
                f"{expected_tail}, got {blocks.shape}."
            )
        q_index = int(q_index) % self.ny
        k_minus_q = (np.arange(self.ny) - q_index) % self.ny
        operators = self._momentum_operators
        v_minus = operators["A_minus"] + operators["B_minus"]
        v_plus = operators["A_plus"] + operators["B_plus"]
        total = v_minus + v_plus
        result = -0.5 * (total @ blocks + blocks @ total[k_minus_q])
        if include_number_dephasing:
            for name in FAMILY_NAMES:
                modes = self._momentum_modes[name]
                shifted_modes = modes[k_minus_q]
                # Writing the contractions as batched matrix products is
                # materially faster than a three-operand einsum at production
                # sizes and naturally retains any leading probe/source axes.
                projected = blocks @ shifted_modes
                expectation = np.sum(
                    modes.conj() * projected,
                    axis=(-3, -2),
                ) / float(self.ny)
                recycled = (
                    modes * expectation[..., None, None, :]
                ) @ np.swapaxes(shifted_modes.conj(), -1, -2)
                operator = operators[name]
                result += -0.5 * (
                    operator @ blocks
                    + blocks @ operator[k_minus_q]
                    - 2.0 * recycled
                )
        return result

    def q_sectors_homogeneous_rhs(
        self,
        sectors: Any,
        *,
        q_indices: Any,
        include_number_dephasing: bool,
    ) -> Array:
        """Vectorized homogeneous action for several q sectors.

        ``sectors`` has shape ``(Nq,...,Ny,d,d)``.  The optional axes between
        ``Nq`` and ``Ny`` are independent right-hand sides (for example the two
        wall-source kicks).  This is algebraically identical to calling
        :meth:`q_sector_homogeneous_rhs` for each q, while amortizing the small
        batched contractions needed by response calculations.
        """

        self._ensure_momentum_frames()
        assert self._momentum_modes is not None
        assert self._momentum_operators is not None
        blocks = np.asarray(sectors, dtype=np.complex128)
        expected_tail = (self.ny, self.block_dimension, self.block_dimension)
        if blocks.ndim < 4 or blocks.shape[-3:] != expected_tail:
            raise ValueError(
                "sectors must have shape (Nq,...,Ny,d,d) with trailing shape "
                f"{expected_tail}, got {blocks.shape}."
            )
        q_values = np.asarray(q_indices, dtype=np.int64).reshape(-1) % self.ny
        if q_values.size != blocks.shape[0]:
            raise ValueError("q_indices length must match the leading sector axis")
        extra_axes = blocks.ndim - 4
        singleton = (slice(None),) + (None,) * extra_axes
        k_minus_q = (
            np.arange(self.ny, dtype=np.int64)[None, :] - q_values[:, None]
        ) % self.ny

        operators = self._momentum_operators
        total = (
            operators["A_minus"]
            + operators["B_minus"]
            + operators["A_plus"]
            + operators["B_plus"]
        )
        left = total[(None,) * (1 + extra_axes) + (slice(None), slice(None), slice(None))]
        shifted_total = total[k_minus_q][
            (slice(None),) + (None,) * extra_axes + (slice(None), slice(None), slice(None))
        ]
        result = -0.5 * (left @ blocks + blocks @ shifted_total)
        if include_number_dephasing:
            for name in FAMILY_NAMES:
                modes = self._momentum_modes[name]
                shifted_modes = modes[k_minus_q]
                left_modes = modes[
                    (None,) * (1 + extra_axes)
                    + (slice(None), slice(None), slice(None))
                ]
                right_modes = shifted_modes[
                    (slice(None),)
                    + (None,) * extra_axes
                    + (slice(None), slice(None), slice(None))
                ]
                projected = blocks @ right_modes
                expectation = np.sum(
                    left_modes.conj() * projected,
                    axis=(-3, -2),
                ) / float(self.ny)
                recycled = (
                    left_modes * expectation[..., None, None, :]
                ) @ np.swapaxes(right_modes.conj(), -1, -2)
                operator = operators[name]
                left_operator = operator[
                    (None,) * (1 + extra_axes)
                    + (slice(None), slice(None), slice(None))
                ]
                shifted_operator = operator[k_minus_q][
                    (slice(None),)
                    + (None,) * extra_axes
                    + (slice(None), slice(None), slice(None))
                ]
                result += -0.5 * (
                    left_operator @ blocks
                    + blocks @ shifted_operator
                    - 2.0 * recycled
                )
        return result

    def q_sector_rhs(
        self,
        sector: Any,
        *,
        q_index: int,
        include_number_dephasing: bool,
    ) -> Array:
        """Return the affine generator on one momentum-transfer sector."""

        self._ensure_momentum_frames()
        assert self._momentum_operators is not None
        result = self.q_sector_homogeneous_rhs(
            sector,
            q_index=q_index,
            include_number_dephasing=include_number_dephasing,
        )
        if int(q_index) % self.ny == 0:
            result = result + (
                self._momentum_operators["A_minus"]
                + self._momentum_operators["B_minus"]
            )
        return result

    def dense_q_sector(self, correlation: Any, *, q_index: int) -> Array:
        """Transform a dense correlation matrix and extract one q sector."""

        momentum = dense_to_y_momentum(
            correlation, nx=self.nx, ny=self.ny
        )
        return extract_q_sector(momentum, q_index)

    def q_sector_to_dense(self, sector: Any, *, q_index: int) -> Array:
        """Embed one q sector and transform it to the canonical real-space basis."""

        return y_momentum_to_dense(
            embed_q_sector(sector, q_index), nx=self.nx, ny=self.ny
        )

    def integrate_dense(
        self,
        initial: Any,
        *,
        dt: float,
        observation_times: Any,
        include_number_dephasing: bool,
        progress: bool = False,
    ) -> RK4Evolution:
        """RK4 evolution of a physical dense correlation matrix."""

        return integrate_rk4(
            lambda matrix: self.dense_rhs(
                matrix,
                include_number_dephasing=include_number_dephasing,
            ),
            _as_complex_matrix(initial, self.dimension, name="initial"),
            dt=dt,
            observation_times=observation_times,
            hermitize=True,
            progress=progress,
        )

    def integrate_q_sector(
        self,
        initial: Any,
        *,
        q_index: int,
        dt: float,
        observation_times: Any,
        include_number_dephasing: bool,
        homogeneous: bool = False,
        progress: bool = False,
    ) -> RK4Evolution:
        """RK4 evolution confined to a y momentum-transfer sector."""

        q_index = int(q_index) % self.ny
        action = (
            self.q_sector_homogeneous_rhs
            if bool(homogeneous)
            else self.q_sector_rhs
        )
        return integrate_rk4(
            lambda blocks: action(
                blocks,
                q_index=q_index,
                include_number_dephasing=include_number_dephasing,
            ),
            initial,
            dt=dt,
            observation_times=observation_times,
            hermitize=(q_index == 0),
            progress=progress,
        )


__all__ = [
    "CANONICAL_FRAME_ATTRIBUTES",
    "FAMILY_NAMES",
    "LOWER_FAMILIES",
    "PerfectCorrectionLindblad",
    "RK4Evolution",
    "UPPER_FAMILIES",
    "dense_to_y_momentum",
    "embed_q_sector",
    "extract_q_sector",
    "integrate_rk4",
    "linear_gain_rhs",
    "linear_loss_rhs",
    "number_dephasing_rhs",
    "perfect_correction_mode_rhs",
    "y_momentum_to_dense",
]
