"""Canonical model construction shared by the matched campaign adapters.

The campaign deliberately keeps model construction in one place.  Both the
ordered Markov channel and the infinitesimal Lindblad solver therefore consume
the same normalized overcomplete-Wannier (OW) arrays, including the explicit
inclusive domain-wall interval ``x=5,...,15``.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.fgtn.classA_U1FGTN import classA_U1FGTN


def build_model(
    case: dict[str, Any],
    *,
    quiet: bool = True,
) -> classA_U1FGTN:
    """Build one pinned canonical model and materialize its OW arrays.

    ``nshell=None`` is the full-frame control.  No campaign adapter is allowed
    to recreate or post-process the OW vectors independently of
    :meth:`classA_U1FGTN.construct_OW_projectors`.
    """

    model_spec = case["model"]
    if not bool(model_spec.get("domain_wall", False)):
        raise ValueError("the matched campaign requires domain_wall=True")
    walls = tuple(int(value) for value in model_spec["wall_locations"])
    if len(walls) != 2 or walls != tuple(sorted(walls)):
        raise ValueError("wall_locations must be an ordered pair")
    nx, ny = int(model_spec["Nx"]), int(model_spec["Ny"])
    if not (0 <= walls[0] <= walls[1] < nx):
        raise ValueError("wall_locations must define an inclusive interval in x")
    if str(model_spec["trial_orbitals"]).upper() != "X":
        raise ValueError("the matched campaign locks trial_orbitals='X'")

    constructor = lambda: classA_U1FGTN(
        nx,
        ny,
        DW=True,
        nshell=model_spec["nshell"],
        alpha_1=float(model_spec["alpha_run_in"]),
        alpha_2=float(model_spec["alpha_run_out"]),
        trial_orbitals="X",
        dw_truncation=bool(model_spec["dw_truncation"]),
        twist_y=0.0,
        dw_interval=walls,
    )
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            model = constructor()
    else:
        model = constructor()
    construct = lambda: model.construct_OW_projectors(
        nshell=model_spec["nshell"],
        DW=True,
        trial_orbitals="X",
        dw_truncation=bool(model_spec["dw_truncation"]),
        twist_y=0.0,
    )
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            construct()
    else:
        construct()

    if tuple(int(value) for value in model.DW_loc) != walls:
        raise RuntimeError("canonical model did not retain the requested wall interval")
    return model


__all__ = ["build_model"]
