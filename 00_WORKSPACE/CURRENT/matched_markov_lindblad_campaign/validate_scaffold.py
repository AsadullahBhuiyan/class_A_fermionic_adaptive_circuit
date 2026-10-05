#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from campaign_schema import (
    EXPECTED_CASES,
    case_requires_response,
    expand_cases,
    load_config,
    late_cycle_bounds,
)
from observables import (
    exact_y_twirl,
    ky_blocks_from_twirled,
    late_cycle_average,
    translation_residual,
    y_translate,
)


HERE = Path(__file__).resolve().parent


def validate_python_sources() -> None:
    for path in sorted(HERE.glob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_grid() -> None:
    config, _ = load_config(HERE / "campaign_config.v1.json")
    cases = expand_cases(config)
    assert len(cases) == EXPECTED_CASES == 105
    arms: dict[tuple[str, bool], int] = {}
    for case in cases:
        key = (case["dynamics"]["family"], case["dynamics"]["dephasing"])
        arms[key] = arms.get(key, 0) + 1
        assert case["model"]["wall_locations"] == [5, 15]
        assert case["dynamics"]["cycles"] == 2 * case["model"]["Ny"]
        start, end = late_cycle_bounds(case)
        assert (start, end) == (case["model"]["Ny"] + 1, 2 * case["model"]["Ny"])
    assert arms == {
        ("markov_channel", True): 27,
        ("lindblad", False): 39,
        ("lindblad", True): 39,
    }
    schedule = [case for case in cases if "channel_schedule_seed_control" in case["campaign_roles"]]
    assert len(schedule) == 1
    assert len(schedule[0]["dynamics"]["sample_seeds"]) == 10
    assert len(set(schedule[0]["dynamics"]["sample_seeds"])) == 10
    assert sum(case_requires_response(case, config) for case in cases) == 22


def validate_late_average_and_twirl() -> None:
    cycles = np.arange(0, 9)
    states = np.stack([np.eye(4) * cycle for cycle in cycles])
    observed = late_cycle_average(states, cycles, start_cycle=5, end_cycle=8)
    assert np.allclose(observed, 6.5 * np.eye(4))
    for malformed_cycle in (
        np.asarray([0, 1, 2, 3, 4, 5, 6, 6, 8]),
        np.asarray([0, 1, 2, 3, 4, 5, 6, 8, 9]),
    ):
        try:
            late_cycle_average(states, malformed_cycle, start_cycle=5, end_cycle=8)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed late-cycle coordinate was accepted")

    rng = np.random.default_rng(17)
    raw = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    correlation = raw @ raw.conj().T
    correlation /= np.linalg.norm(correlation)
    twirled = exact_y_twirl(correlation, nx=2, ny=3)
    assert translation_residual(twirled, nx=2, ny=3) < 1e-14
    assert np.allclose(twirled, twirled.conj().T)
    assert np.allclose(np.trace(twirled), np.trace(correlation))
    assert np.min(np.linalg.eigvalsh(twirled)) > -1e-13
    assert np.allclose(exact_y_twirl(twirled, 2, 3), twirled)
    shifted_twirl = exact_y_twirl(y_translate(correlation, 2, 3), 2, 3)
    assert np.allclose(twirled, shifted_twirl)

    # Canonical class index convention: i=mu+2*x+2*Nx*y.  A positive
    # translation moves both matrix indices from y to (y+1) mod Ny.
    nx, ny = 2, 3
    tagged = np.zeros((2 * nx * ny, 2 * nx * ny), dtype=np.complex128)
    source_i = 1 + 2 * 0 + 2 * nx * 1
    source_j = 0 + 2 * 1 + 2 * nx * 2
    target_i = 1 + 2 * 0 + 2 * nx * 2
    target_j = 0 + 2 * 1 + 2 * nx * 0
    tagged[source_i, source_j] = 7.0 - 3.0j
    shifted = y_translate(tagged, nx, ny, shift=1)
    assert shifted[target_i, target_j] == tagged[source_i, source_j]
    assert np.count_nonzero(shifted) == 1

    # A block-circulant matrix must be recovered exactly in FFT ordering.
    rng = np.random.default_rng(41)
    momenta = 2.0 * np.pi * np.fft.fftfreq(ny)
    prescribed = []
    for _ in range(ny):
        raw_block = rng.normal(size=(2 * nx, 2 * nx)) + 1j * rng.normal(
            size=(2 * nx, 2 * nx)
        )
        prescribed.append(0.5 * (raw_block + raw_block.conj().T))
    prescribed = np.asarray(prescribed)
    circulant = np.zeros((2 * nx * ny, 2 * nx * ny), dtype=np.complex128)
    for y in range(ny):
        for yp in range(ny):
            spatial_block = sum(
                np.exp(1j * momentum * (y - yp)) * prescribed[k]
                for k, momentum in enumerate(momenta)
            ) / ny
            for x in range(nx):
                for mu in range(2):
                    i = mu + 2 * x + 2 * nx * y
                    local_i = mu + 2 * x
                    for xp in range(nx):
                        for mup in range(2):
                            j = mup + 2 * xp + 2 * nx * yp
                            local_j = mup + 2 * xp
                            circulant[i, j] = spatial_block[local_i, local_j]
    observed_ky, observed_blocks = ky_blocks_from_twirled(circulant, nx, ny)
    assert np.allclose(observed_ky, momenta)
    assert np.allclose(observed_blocks, prescribed)


def validate_notebook_contract() -> None:
    notebook = json.loads((HERE / "analyze_matched_campaign.ipynb").read_text(encoding="utf-8"))
    cells = notebook["cells"]
    first_code = next(cell for cell in cells if cell["cell_type"] == "code")
    source = "".join(first_code["source"])
    assert "CPU_RANGE" in source and "sched_setaffinity" in source
    assert "Raw diagnostics" in "\n".join(
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "markdown"
    )


def validate_document_contract() -> None:
    main = (HERE / "docs" / "matched_channel_lindblad_main.tex").read_text(encoding="utf-8")
    supplement = (HERE / "docs" / "matched_channel_lindblad_supplement.tex").read_text(encoding="utf-8")
    for source in (main, supplement):
        assert r"\mathbb{1}" not in source
        assert r"\mathbbm{1}" not in source
        assert r"\mathscr" not in source
        assert "overcomplete Wannier (OW)" in source
    assert "run\\_markov\\_channel" in main
    assert "$A+$ (loss), $A-$ (gain), $B+$" in main
    assert r"\mathcal E_{\boldsymbol s}=\mathcal I+\mathcal L" in main
    assert "Exactly 22 endpoint cases" in main
    assert "charge-variance proxy" in main
    assert "nine-page" in main


def main() -> int:
    validate_python_sources()
    validate_grid()
    validate_late_average_and_twirl()
    validate_notebook_contract()
    validate_document_contract()
    print("matched campaign scaffold: all static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
