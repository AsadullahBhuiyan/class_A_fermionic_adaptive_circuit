from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def binary_entropy(occupations: np.ndarray, *, tolerance: float = 1e-14) -> float:
    values = np.clip(np.real(np.asarray(occupations, dtype=np.float64)), 0.0, 1.0)
    interior = values[(values > tolerance) & (values < 1.0 - tolerance)]
    if interior.size == 0:
        return 0.0
    return float(
        -np.sum(interior * np.log(interior) + (1.0 - interior) * np.log1p(-interior))
    )


def gaussian_subsystem_entropy(state: Any, rows: Sequence[int]) -> float:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    block = np.asarray(state.physical_frame[rows], dtype=np.complex128)
    correlation = block @ block.conj().T
    occupations = np.linalg.eigvalsh(0.5 * (correlation + correlation.conj().T))
    return binary_entropy(occupations)


def gaussian_mutual_information(
    state: Any, left_rows: Sequence[int], right_rows: Sequence[int]
) -> tuple[float, float, float, float]:
    left = tuple(int(value) for value in left_rows)
    right = tuple(int(value) for value in right_rows)
    if set(left) & set(right):
        raise ValueError("reference subsystems must be disjoint")
    entropy_left = gaussian_subsystem_entropy(state, left)
    entropy_right = gaussian_subsystem_entropy(state, right)
    entropy_union = gaussian_subsystem_entropy(state, left + right)
    mutual_information = max(0.0, entropy_left + entropy_right - entropy_union)
    return mutual_information, entropy_left, entropy_right, entropy_union


def _append_empty_reference_row(state: Any) -> int:
    if str(state.representation) != "physical_frame":
        raise ValueError("reference insertion currently requires a pure physical_frame state")
    old_dimension = int(state.physical_dimension)
    state.frame = np.vstack(
        (state.frame, np.zeros((1, state.rank), dtype=np.complex128))
    )
    state.physical_dimension = old_dimension + 1
    return old_dimension


def _occupy_empty_reference(state: Any, reference_row: int) -> None:
    column = np.zeros((state.physical_dimension, 1), dtype=np.complex128)
    column[int(reference_row), 0] = 1.0
    state.frame = np.column_stack((state.frame, column))
    state.max_rank = max(int(state.max_rank), int(state.rank))


def _beam_splitter_with_reference(
    state: Any,
    *,
    support_idx: np.ndarray,
    orbital_local: np.ndarray,
    reference_row: int,
) -> None:
    support = np.asarray(support_idx, dtype=np.int64).reshape(-1)
    orbital = np.asarray(orbital_local, dtype=np.complex128).reshape(-1)
    if support.size != orbital.size:
        raise ValueError("support and orbital lengths differ")
    norm = float(np.linalg.norm(orbital))
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise ValueError("probe orbital must be normalized")
    chi = np.zeros(state.physical_dimension, dtype=np.complex128)
    chi[support] = orbital
    physical_amplitude = chi.conj() @ state.frame
    reference_amplitude = state.frame[int(reference_row)].copy()
    inv_sqrt_two = 1.0 / np.sqrt(2.0)
    transformed_physical = inv_sqrt_two * (
        physical_amplitude + reference_amplitude
    )
    transformed_reference = inv_sqrt_two * (
        -physical_amplitude + reference_amplitude
    )
    state.frame += np.outer(chi, transformed_physical - physical_amplitude)
    state.frame[int(reference_row)] = transformed_reference


def insert_reference_mode(
    state: Any,
    *,
    support_idx: Sequence[int],
    orbital_local: Sequence[complex],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Measure one physical mode and replace it by a Gaussian Bell pair.

    The new reference is prepared with occupation opposite to the measured
    physical mode.  A 50:50 number-conserving beam splitter then creates one
    fermion delocalized across the physical/reference pair.  Measurement makes
    the physical mode factorize first, so the resulting pair is maximally
    entangled and decoupled from the rest at insertion.
    """

    support = np.asarray(support_idx, dtype=np.int64).reshape(-1)
    orbital = np.asarray(orbital_local, dtype=np.complex128).reshape(-1)
    probability_occupied = float(state.occupation_probability_local(support, orbital))
    if probability_occupied <= state.zero_tolerance:
        outcome_occupied = False
    elif probability_occupied >= 1.0 - state.zero_tolerance:
        outcome_occupied = True
    else:
        outcome_occupied = bool(rng.random() < probability_occupied)
    if outcome_occupied:
        selected_probability = float(state.project_occupied_local(support, orbital))
    else:
        selected_probability = float(state.project_empty_local(support, orbital))

    reference_row = _append_empty_reference_row(state)
    if not outcome_occupied:
        _occupy_empty_reference(state, reference_row)
    _beam_splitter_with_reference(
        state,
        support_idx=support,
        orbital_local=orbital,
        reference_row=reference_row,
    )
    residual = float(state.gram_residual())
    if not np.isfinite(residual) or residual > 1e-9:
        raise FloatingPointError(
            f"reference insertion violated frame orthonormality: residual={residual:.3e}"
        )
    return {
        "reference_row": int(reference_row),
        "probability_occupied": probability_occupied,
        "outcome_occupied": bool(outcome_occupied),
        "selected_probability": selected_probability,
    }


def insert_reference_cell(
    state: Any,
    *,
    nx: int,
    x: int,
    y: int,
    system_dimension: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Insert a two-mode reference maximally entangled with one unit cell."""

    if int(state.physical_dimension) < int(system_dimension):
        raise ValueError("state has fewer rows than the declared system")
    events = []
    reference_rows = []
    for orbital_index in (0, 1):
        system_row = orbital_index + 2 * int(x) + 2 * int(nx) * int(y)
        event = insert_reference_mode(
            state,
            support_idx=[system_row],
            orbital_local=[1.0],
            rng=rng,
        )
        event["orbital_index"] = orbital_index
        event["system_row"] = system_row
        events.append(event)
        reference_rows.append(int(event["reference_row"]))
    entropy = gaussian_subsystem_entropy(state, reference_rows)
    np.testing.assert_allclose(entropy, 2.0 * np.log(2.0), atol=2e-10, rtol=0.0)
    return {
        "x": int(x),
        "y": int(y),
        "reference_rows": tuple(reference_rows),
        "events": tuple(events),
        "insertion_entropy": entropy,
    }


@dataclass
class ReferencePairObserver:
    nx: int
    ny: int
    tau1: int
    tau2: int
    follow_cycles: int
    first_site: tuple[int, int]
    second_site: tuple[int, int]
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.system_dimension = 2 * int(self.nx) * int(self.ny)
        self.reference_one: dict[str, Any] | None = None
        self.reference_two: dict[str, Any] | None = None
        length = int(self.follow_cycles) + 1
        self.cycles = np.full(length, -1, dtype=np.int64)
        self.mutual_information = np.full(length, np.nan, dtype=np.float64)
        self.entropy_r1 = np.full(length, np.nan, dtype=np.float64)
        self.entropy_r2 = np.full(length, np.nan, dtype=np.float64)
        self.entropy_r12 = np.full(length, np.nan, dtype=np.float64)

    def __call__(self, *, cycle: int, state: Any, **_: Any) -> None:
        cycle = int(cycle)
        if cycle == int(self.tau1):
            self.reference_one = insert_reference_cell(
                state,
                nx=self.nx,
                x=self.first_site[0],
                y=self.first_site[1],
                system_dimension=self.system_dimension,
                rng=self.rng,
            )
        if cycle == int(self.tau2):
            self.reference_two = insert_reference_cell(
                state,
                nx=self.nx,
                x=self.second_site[0],
                y=self.second_site[1],
                system_dimension=self.system_dimension,
                rng=self.rng,
            )
        if cycle < int(self.tau2) or self.reference_two is None:
            return
        relative_cycle = cycle - int(self.tau2)
        if relative_cycle > int(self.follow_cycles):
            return
        rows_one = self.reference_one["reference_rows"]
        rows_two = self.reference_two["reference_rows"]
        mutual_information, entropy_one, entropy_two, entropy_union = (
            gaussian_mutual_information(state, rows_one, rows_two)
        )
        self.cycles[relative_cycle] = cycle
        self.mutual_information[relative_cycle] = mutual_information
        self.entropy_r1[relative_cycle] = entropy_one
        self.entropy_r2[relative_cycle] = entropy_two
        self.entropy_r12[relative_cycle] = entropy_union

    def assert_complete(self) -> None:
        if self.reference_one is None or self.reference_two is None:
            raise RuntimeError("both reference cells were not inserted")
        if np.any(self.cycles < 0):
            raise RuntimeError("reference mutual-information time series is incomplete")
        for values in (
            self.mutual_information,
            self.entropy_r1,
            self.entropy_r2,
            self.entropy_r12,
        ):
            if not np.all(np.isfinite(values)):
                raise RuntimeError("reference entropy time series contains non-finite values")
        if np.min(self.mutual_information) < -1e-12:
            raise RuntimeError("mutual information became negative beyond tolerance")

    def payload(self) -> dict[str, Any]:
        self.assert_complete()
        return {
            "cycles": self.cycles.copy(),
            "mutual_information": self.mutual_information.copy(),
            "entropy_r1": self.entropy_r1.copy(),
            "entropy_r2": self.entropy_r2.copy(),
            "entropy_r12": self.entropy_r12.copy(),
            "reference_one": self.reference_one,
            "reference_two": self.reference_two,
        }


def matching_time(
    temporal_separations: Sequence[float],
    temporal_values: Sequence[float],
    spatial_value: float,
) -> float:
    separations = np.asarray(temporal_separations, dtype=np.float64)
    temporal = np.asarray(temporal_values, dtype=np.float64)
    target = float(spatial_value)
    if separations.ndim != 1 or temporal.shape != separations.shape or separations.size < 2:
        return float("nan")
    order = np.argsort(separations)
    separations = separations[order]
    temporal = temporal[order]
    if not np.all(np.isfinite(temporal)) or not np.isfinite(target) or target <= 0.0:
        return float("nan")
    difference = temporal - target
    # The CFT time-like correlator decreases from short to long separation.  If
    # the shortest sampled separation is already below the spatial target, a
    # later upward noise fluctuation followed by a downward fluctuation is not a
    # physical matching bracket.
    if difference[0] <= 0.0:
        return float("nan")
    for index in range(separations.size - 1):
        if difference[index] == 0.0:
            return float(separations[index])
        if difference[index] > 0.0 and difference[index + 1] < 0.0:
            fraction = difference[index] / (difference[index] - difference[index + 1])
            return float(
                separations[index]
                + fraction * (separations[index + 1] - separations[index])
            )
    if difference[-1] == 0.0:
        return float(separations[-1])
    return float("nan")


def anisotropy_from_matching_time(circumference: int, time_star: float) -> float:
    if not np.isfinite(time_star) or float(time_star) <= 0.0:
        return float("nan")
    return float(np.log1p(np.sqrt(2.0)) * int(circumference) / (np.pi * time_star))
