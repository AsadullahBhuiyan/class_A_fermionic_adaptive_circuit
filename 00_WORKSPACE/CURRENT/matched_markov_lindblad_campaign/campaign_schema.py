from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "matched_markov_lindblad_campaign_config_v1"
EXPECTED_CASES = 105


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    validate_config(config)
    return config, sha256_bytes(raw)


def _tag(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return str(value).replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def _matched_seed(root_seed: int, match_key: str) -> int:
    payload = f"{int(root_seed)}:{match_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"expected config schema {SCHEMA!r}")
    model = config["model"]
    if model["Nx"] != 20 or model["wall_locations"] != [5, 15]:
        raise ValueError("the matched campaign locks Nx=20 and walls x=5,15")
    if model["finite_nshell"] != [1, 2]:
        raise ValueError("the matched finite-shell grid must be [1,2]")
    if model["alpha_run_out"] != 30.0 or model["init_mode"] != "maxmix":
        raise ValueError("alpha_run_out=30 and maxmix initialization are locked")
    if not model["perfect_correction"]:
        raise ValueError("the campaign is perfect-correction only")
    dynamics = config["dynamics"]
    if dynamics["markov_channel"]["dephasing"] != [True]:
        raise ValueError("production Markov-channel cases are full-dephasing only")
    if dynamics["lindblad"]["dephasing"] != [False, True]:
        raise ValueError("Lindblad production must contain matched dephasing off/on arms")
    if dynamics["markov_channel"]["root_seed"] != 20260814:
        raise ValueError("the shared root seed is locked to 20260814")


def expand_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand and de-duplicate the four declared grids.

    A physical case can serve both the endpoint and alpha-scan roles.  Such a case is
    run once and records both role labels.  Schedule seeds omit dynamics, dephasing,
    and DW truncation, providing matched random orders across every comparable arm.
    """

    validate_config(config)
    model = config["model"]
    root_seed = int(config["dynamics"]["markov_channel"]["root_seed"])
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for role, grid in config["grids"].items():
        dynamics_names = grid.get("dynamics", ["markov_channel", "lindblad"])
        for ny in grid["Ny"]:
            for alpha_in in grid["alpha_run_in"]:
                for nshell in grid["nshell"]:
                    for dw_truncation in grid.get("dw_truncation", model["dw_truncation"]):
                        for dynamics_name in dynamics_names:
                            dephasing_values = config["dynamics"][dynamics_name]["dephasing"]
                            for dephasing in dephasing_values:
                                physics_key = (
                                    int(model["Nx"]), int(ny), float(alpha_in),
                                    float(model["alpha_run_out"]), nshell,
                                    bool(dw_truncation), dynamics_name, bool(dephasing),
                                )
                                match_key = (
                                    f"N{model['Nx']}x{ny}_ain{_tag(float(alpha_in))}"
                                    f"_aout{_tag(float(model['alpha_run_out']))}_nsh{_tag(nshell)}"
                                )
                                # The site word depends only on the lattice geometry.  Use
                                # one matched random schedule across alpha, shell, wall-mask,
                                # and dynamics comparisons at a given size.
                                schedule_match_key = f"N{model['Nx']}x{ny}:random-site-word"
                                if physics_key in merged:
                                    merged[physics_key]["campaign_roles"].append(role)
                                    continue
                                case_id = (
                                    f"{match_key}_dwtrunc{int(bool(dw_truncation))}"
                                    f"_{dynamics_name}_deph{int(bool(dephasing))}_pc1"
                                )
                                merged[physics_key] = {
                                    "schema": "matched_markov_lindblad_case_spec_v1",
                                    "case_id": case_id,
                                    "campaign_roles": [role],
                                    "model": {
                                        "Nx": int(model["Nx"]),
                                        "Ny": int(ny),
                                        "domain_wall": True,
                                        "wall_locations": list(model["wall_locations"]),
                                        "alpha_run_in": float(alpha_in),
                                        "alpha_run_out": float(model["alpha_run_out"]),
                                        "trial_orbitals": model["trial_orbitals"],
                                        "nshell": nshell,
                                        "dw_truncation": bool(dw_truncation),
                                    },
                                    "dynamics": {
                                        "family": dynamics_name,
                                        "dephasing": bool(dephasing),
                                        "perfect_correction": True,
                                        "init_mode": model["init_mode"],
                                        "cycles": 2 * int(ny),
                                        "physical_time": 2.0 * int(ny),
                                        "site_schedule": config["dynamics"]["markov_channel"]["site_schedule"],
                                        "root_seed": root_seed,
                                        "matched_seed": _matched_seed(root_seed, schedule_match_key),
                                        "schedule_match_key": schedule_match_key,
                                        "match_key": match_key,
                                        "sample_ids": [0],
                                        "sample_seeds": [_matched_seed(root_seed, schedule_match_key)],
                                    },
                                }
    control = config["schedule_seed_control"]
    if control.get("enabled", False):
        ny = int(control["Ny"])
        alpha_in = float(control["alpha_run_in"])
        nshell = control["nshell"]
        match_key = (
            f"N{model['Nx']}x{ny}_ain{_tag(alpha_in)}"
            f"_aout{_tag(float(model['alpha_run_out']))}_nsh{_tag(nshell)}"
        )
        schedule_match_key = f"N{model['Nx']}x{ny}:random-site-word"
        sample_count = int(control["independent_samples"])
        seeds = [
            _matched_seed(root_seed, f"{schedule_match_key}:schedule-control:{sample_id}")
            for sample_id in range(sample_count)
        ]
        control_case = {
            "schema": "matched_markov_lindblad_case_spec_v1",
            "case_id": (
                f"{match_key}_dwtrunc1_markov_channel_deph1_pc1_scheduleS{sample_count}"
            ),
            "campaign_roles": ["channel_schedule_seed_control"],
            "model": {
                "Nx": int(model["Nx"]), "Ny": ny, "domain_wall": True,
                "wall_locations": list(model["wall_locations"]),
                "alpha_run_in": alpha_in,
                "alpha_run_out": float(model["alpha_run_out"]),
                "trial_orbitals": model["trial_orbitals"],
                "nshell": nshell, "dw_truncation": True,
            },
            "dynamics": {
                "family": "markov_channel", "dephasing": True,
                "perfect_correction": True, "init_mode": model["init_mode"],
                "cycles": 2 * ny, "physical_time": 2.0 * ny,
                "site_schedule": config["dynamics"]["markov_channel"]["site_schedule"],
                "root_seed": root_seed, "matched_seed": seeds[0],
                "match_key": match_key, "schedule_match_key": schedule_match_key,
                "sample_ids": list(range(sample_count)),
                "sample_seeds": seeds,
            },
        }
        merged[("schedule_seed_control",)] = control_case
    cases = sorted(merged.values(), key=lambda row: row["case_id"])
    for case in cases:
        case["campaign_roles"] = sorted(set(case["campaign_roles"]))
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected {EXPECTED_CASES} unique cases, obtained {len(cases)}")
    return cases


def late_cycle_bounds(case: dict[str, Any]) -> tuple[int, int]:
    ny = int(case["model"]["Ny"])
    return ny + 1, 2 * ny


def spectral_checkpoint_cycles(case: dict[str, Any]) -> list[int]:
    """Return the locked sparse cycle coordinate for dense spectral estimators."""

    ny = int(case["model"]["Ny"])
    return [0, ny // 2, ny, (3 * ny) // 2, 2 * ny]


def case_requires_response(case: dict[str, Any], config: dict[str, Any]) -> bool:
    """Whether a case belongs to the preregistered endpoint-response subset."""

    response = config.get("response", {})
    if not bool(response.get("enabled", False)):
        return False
    roles = set(case.get("campaign_roles", ()))
    if "smoke" in roles:
        return True
    included = set(response.get("campaign_roles", ()))
    excluded = set(response.get("exclude_roles", ()))
    if not bool(roles.intersection(included)) or bool(roles.intersection(excluded)):
        return False
    model = case["model"]
    if float(model["alpha_run_in"]) not in {
        float(value) for value in response.get("alpha_run_in", [model["alpha_run_in"]])
    }:
        return False
    nshell = model["nshell"]
    ny = int(model["Ny"])
    if nshell is None:
        return ny in {int(value) for value in response.get("full_frame_Ny", [ny])}
    return (
        nshell in response.get("finite_nshell", [nshell])
        and ny in {int(value) for value in response.get("finite_Ny", [ny])}
    )
