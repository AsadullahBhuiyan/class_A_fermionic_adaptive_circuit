"""Narrow integration boundary for the two production dynamics engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from markov_adapter import run_markov_channel_case as _run_markov_channel_case

try:
    from lindblad_adapter import run_lindblad_case as _run_lindblad_case
except ModuleNotFoundError as exc:
    if exc.name != "lindblad_adapter":
        raise
    _run_lindblad_case = None


MARKOV_ADAPTER_READY = True
LINDBLAD_ADAPTER_READY = _run_lindblad_case is not None
ADAPTERS_READY = MARKOV_ADAPTER_READY and LINDBLAD_ADAPTER_READY


class AdapterNotIntegratedError(RuntimeError):
    pass


@dataclass
class AdapterResult:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _pending(name: str) -> AdapterResult:
    raise AdapterNotIntegratedError(
        f"{name} is an explicit integration stub. Connect it to the pinned dynamics "
        "engine, set ADAPTERS_READY=True, and pass the smoke gates before production."
    )


def run_markov_channel_case(case: dict[str, Any], config: dict[str, Any]) -> AdapterResult:
    """Run one case through ``classA_U1FGTN.run_markov_channel``.

    Integration requirements: enforce walls x=5,15 without changing the canonical
    class default globally; seed the random site schedule from ``matched_seed``; use
    ``perfect_correction=True`` and ``decoh=True``; return physical
    ``G=(Q+identity)/2``; stream every declared scalar at cycles 0..2Ny; and accumulate
    the inclusive late-cycle matrix average without archiving a dense history.
    """

    result = _run_markov_channel_case(case, config)
    return AdapterResult(arrays=dict(result.arrays), metadata=dict(result.metadata))


def run_lindblad_case(case: dict[str, Any], config: dict[str, Any]) -> AdapterResult:
    """Run the matched perfect-correction infinitesimal continuous Lindblad equation.

    Integration requirements: consume the exact OW vectors used by the channel case;
    use unit gain/loss coefficients and, when requested, unit number-dephasing
    coefficients; integrate through t=2Ny with the configured RK4 step; stream the same
    cycle observables; and retain only final and late-average dense checkpoints.
    """

    if _run_lindblad_case is None:
        return _pending("run_lindblad_case")
    result = _run_lindblad_case(case, config)
    return AdapterResult(arrays=dict(result.arrays), metadata=dict(result.metadata))
