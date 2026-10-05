from __future__ import annotations

from contextvars import ContextVar
from typing import Any


_DECLARED_SAMPLES: ContextVar[int] = ContextVar("declared_samples", default=10)


def _case_id(campaign: str, protocol: str, nx: int, ny: int, **tags: Any) -> str:
    suffix = "_".join(
        f"{key}-{str(value).replace('.', 'p')}" for key, value in sorted(tags.items())
    )
    base = f"{campaign}_N{int(nx)}x{int(ny)}_{protocol}"
    return base if not suffix else f"{base}_{suffix}"


def _model(
    *,
    nx: int,
    ny: int,
    protocol: str,
    init_mode: str = "default",
    nshell: int | None = 1,
    alpha_in: float = 1.0,
    alpha_out: float = 30.0,
) -> dict[str, Any]:
    common = {
        "Nx": int(nx),
        "Ny": int(ny),
        "nshell": None if nshell is None else int(nshell),
        "filling_frac": 0.5,
        "trial_orbitals": "X",
        "device": "cuda:0",
        "dtype": "complex128",
        "backend": "dense" if nshell is None else "local",
    }
    if protocol == "uniform_topological":
        geometry = {
            "DW": False,
            "alpha_1": 1.0,
            "alpha_2": 1.0,
            "dw_truncation": False,
            "meas_slab_only": False,
        }
    elif protocol in ("uniform_trivial", "explicit_interface_matched_trivial"):
        geometry = {
            "DW": False,
            "alpha_1": 30.0,
            "alpha_2": 30.0,
            "dw_truncation": False,
            "meas_slab_only": False,
        }
    elif protocol in ("explicit_interface", "alpha_wall", "noisy_wall"):
        geometry = {
            "DW": True,
            "alpha_1": float(alpha_in),
            "alpha_2": float(alpha_out),
            "dw_truncation": False,
            "meas_slab_only": False,
        }
    elif protocol in ("support_terminated", "support_terminated_matched_trivial"):
        alpha_top = 30.0 if protocol.endswith("matched_trivial") else float(alpha_in)
        geometry = {
            "DW": True,
            "alpha_1": alpha_top,
            "alpha_2": 30.0,
            "dw_truncation": True,
            "meas_slab_only": True,
        }
    else:
        raise ValueError(f"unknown physical protocol {protocol!r}")
    return {**common, **geometry, "init_mode": str(init_mode)}


def _run(ny: int, *, samples: int | None = None, **overrides: Any) -> dict[str, Any]:
    if samples is None:
        samples = _DECLARED_SAMPLES.get()
    return {
        "cycles": 2 * int(ny),
        "samples": int(samples),
        "sequence": "random",
        "perfect_correction": True,
        "postselect": False,
        "postselect_probability": 0.0,
        "G_history": False,
        "save": False,
        "return_data": False,
        **overrides,
    }


def _selected_cycles(ny: int) -> list[int]:
    return sorted({ny // 4, ny // 2, 3 * ny // 4, ny, 3 * ny // 2, 2 * ny})


def expand_cases(
    config: dict[str, Any],
    *,
    accepted_width: int | None = None,
    m3_wall_sigma: list[float] | None = None,
) -> list[dict[str, Any]]:
    bundle = config["bundle"]
    _DECLARED_SAMPLES.set(int(config.get("locked_contract", {}).get("samples", 10)))
    cases: list[dict[str, Any]] = []

    if bundle == "00_validation":
        smoke = config["validation_matrix"]["production_path_smoke"]
        nx, ny = int(smoke["Nx"]), int(smoke["Ny"])
        cases.append(
            {
                "case_id": "V0_production_path_smoke",
                "campaign": "V0",
                "kind": "validation",
                "model": _model(nx=nx, ny=ny, protocol="explicit_interface"),
                "run": _run(
                    ny, samples=int(smoke["samples"]), cycles=int(smoke["cycles"])
                ),
                "observation_cycles": [int(smoke["cycles"])],
            }
        )
        return cases

    if bundle == "01_bulk_width_gate":
        p1 = config["P1"]
        p1_revision = str(p1["campaign_revision"])
        if bool(p1.get("enabled", True)):
            for size in p1["square_sizes"]:
                for protocol in p1["pure_protocols"]:
                    for nshell in p1["nshell_values"]:
                        shell_tag = "none" if nshell is None else int(nshell)
                        cases.append(
                            {
                                "case_id": _case_id(
                                    "P1",
                                    protocol,
                                    size,
                                    size,
                                    init="pure",
                                    nsh=shell_tag,
                                    rev=p1_revision,
                                ),
                                "campaign": "P1",
                                "kind": "stochastic",
                                "model": _model(
                                    nx=size, ny=size, protocol=protocol, nshell=nshell
                                ),
                                "run": _run(size),
                                "observation_cycles": _selected_cycles(size),
                                "local_marker_cycles": [2 * size],
                                "bott_cycles": [2 * size],
                            }
                        )
            for size in p1["maxmix_sizes"]:
                for protocol in p1["maxmix_protocols"]:
                    for nshell in p1["nshell_values"]:
                        shell_tag = "none" if nshell is None else int(nshell)
                        cases.append(
                            {
                                "case_id": _case_id(
                                    "P1",
                                    protocol,
                                    size,
                                    size,
                                    init="maxmix",
                                    nsh=shell_tag,
                                    rev=p1_revision,
                                ),
                                "campaign": "P1",
                                "kind": "stochastic",
                                "model": _model(
                                    nx=size,
                                    ny=size,
                                    protocol=protocol,
                                    init_mode="maxmix",
                                    nshell=nshell,
                                ),
                                "run": _run(size),
                                "observation_cycles": _selected_cycles(size),
                                "local_marker_cycles": [2 * size],
                                "bott_cycles": [2 * size],
                            }
                        )
        w1 = config["W1"]
        raw_nx = w1["Nx"]
        nx_values = [int(raw_nx)] if isinstance(raw_nx, (int, float)) else [int(v) for v in raw_nx]
        if str(config.get("sampling_revision", "")).startswith(
            ("production_10sample_v3", "production_10sample_v4")
        ) and nx_values != [20]:
            raise ValueError("the v3 baseline contract fixes Nx=20")
        for ny in w1["Ny"]:
            for nx in nx_values:
                for protocol in w1["constructions"]:
                    cases.append(
                        {
                            "case_id": _case_id("W1", protocol, nx, ny),
                            "campaign": "W1",
                            "kind": "stochastic",
                            "model": _model(nx=nx, ny=ny, protocol=protocol),
                            "run": _run(
                                ny, lyapunov_nvec=16, lyapunov_start_cycle=1
                            ),
                            "observation_cycles": sorted(
                                {max(1, round(k * ny / 4)) for k in range(1, 9)}
                            ),
                            "strip_entropy_cycles": [ny, 3 * ny // 2, 2 * ny],
                            "local_marker_cycles": [2 * ny],
                            "tangent_alignment_stop": ny,
                        }
                    )
        return cases

    fixed = config.get("fixed_geometry", {})
    nx = int(fixed.get("Nx", 20))
    if nx != 20:
        raise ValueError("the v3 production contract fixes Nx=20")
    if accepted_width is not None and int(accepted_width) != nx:
        raise ValueError(
            f"deprecated accepted-width override {accepted_width} conflicts with fixed Nx={nx}"
        )

    if bundle == "02_pure_wall_master":
        families = config["trajectory_families"]
        mappings = {
            "primary_explicit_interface": (
                "explicit_interface",
                "explicit_interface_matched_trivial",
            ),
            "secondary_support_terminated": (
                "support_terminated",
                "support_terminated_matched_trivial",
            ),
        }
        for family_name, family in families.items():
            protocols = mappings[family_name]
            for ny in family["Ny"]:
                for protocol in protocols:
                    origins = sorted(
                        {
                            ny,
                            round(7 * ny / 6),
                            round(4 * ny / 3),
                            3 * ny // 2,
                        }
                    )
                    snapshots = sorted({ny, 3 * ny // 2, 2 * ny, *origins})
                    cases.append(
                        {
                            "case_id": _case_id("MASTER", protocol, nx, ny),
                            "campaign": "S1_T1_B2_MASTER",
                            "kind": "stochastic_parent",
                            "model": _model(nx=nx, ny=ny, protocol=protocol),
                            "run": _run(
                                ny,
                                lyapunov_nvec=int(config["T1"]["lyapunov_nvec"]),
                                lyapunov_start_cycle=1,
                            ),
                            "observation_cycles": sorted(
                                {*snapshots, 5 * ny // 4, 7 * ny // 4}
                            ),
                            "strip_entropy_cycles": [ny, 3 * ny // 2, 2 * ny],
                            "local_marker_cycles": [ny, 3 * ny // 2, 2 * ny],
                            "h2_origin_cycles": origins,
                            "inline_descendant_stages": [
                                stage
                                for stage in ("H1", "H2")
                                if family_name == "primary_explicit_interface"
                                and int(ny)
                                in {
                                    int(value)
                                    for value in config.get(stage, {}).get("Ny", [])
                                }
                            ],
                            "tangent_alignment_stop": ny,
                        }
                    )
        for ny in config.get("R1", {}).get("Ny", []):
            for protocol in config["R1"]["protocols"]:
                cases.append(
                    {
                        "case_id": _case_id("R1", protocol, nx, ny),
                        "campaign": "R1_RECORD_SPECTRUM",
                        "kind": "stochastic_parent",
                        "model": _model(nx=nx, ny=ny, protocol=protocol),
                        "run": _run(ny),
                        "observation_cycles": [ny, 3 * ny // 2, 2 * ny],
                        "local_marker_cycles": [2 * ny],
                        "inline_descendant_stages": [],
                    }
                )
        return cases

    if bundle == "03_chirality_replay":
        contract = config["H3"]["production_contract"]
        ny = int(contract["Ny"])
        for parent_protocol in contract["protocols"]:
            record_indices = [
                int(value) for value in contract["record_indices_within_parent_shard"]
            ]
            cases.append(
                {
                    "case_id": _case_id(
                        "H3",
                        parent_protocol,
                        nx,
                        ny,
                        gauge=contract["gauge"],
                        grid=int(contract["grid_points"]),
                        records=len(record_indices),
                        seam=int(contract["seam_shift"]),
                    ),
                    "campaign": "H3",
                    "kind": "deterministic_descendant",
                    "Nx": nx,
                    "Ny": ny,
                    "parent_protocol": parent_protocol,
                    "grid_points": int(contract["grid_points"]),
                    "seam_shift": int(contract["seam_shift"]),
                    "gauge": str(contract["gauge"]),
                    "record_indices": record_indices,
                }
            )
        return cases

    if bundle == "04_maxmix_master":
        for ny in config["P2"]["Ny"]:
            for protocol in (
                "explicit_interface",
                "explicit_interface_matched_trivial",
            ):
                late = {
                    max(1, int(round(value))) for value in np_geomspace(1, 2 * ny, 16)
                }
                observations = sorted(
                    set(range(1, min(16, ny // 2) + 1))
                    | late
                    | {ny, 3 * ny // 2, 2 * ny}
                )
                cases.append(
                    {
                        "case_id": _case_id("P2", protocol, nx, ny, init="maxmix"),
                        "campaign": "P2",
                        "kind": "stochastic_parent",
                        "model": _model(
                            nx=nx, ny=ny, protocol=protocol, init_mode="maxmix"
                        ),
                        "run": _run(
                            ny,
                            lyapunov_nvec=int(config["P2"]["lyapunov_nvec"]),
                            lyapunov_start_cycle=1,
                        ),
                        "observation_cycles": observations,
                        "strip_entropy_cycles": [ny, 3 * ny // 2, 2 * ny],
                        "local_marker_cycles": [ny, 2 * ny],
                    }
                )
        return cases

    if bundle == "05_scans_and_controls":
        s2 = config["S2"]
        s2_points = [
            *[
                (int(ny), float(alpha))
                for ny in s2["Ny_full_scan"]
                for alpha in s2["alpha_in"]
            ],
            *[
                (int(ny), float(alpha))
                for ny in s2["Ny_scaling"]
                for alpha in s2["alpha_scaling"]
            ],
        ]
        if len(s2_points) != len(set(s2_points)):
            raise ValueError("S2 full-scan and scaling matrices overlap")
        for ny, alpha in s2_points:
            for init_mode in s2["init_modes"]:
                init_tag = "pure" if init_mode == "default" else "maxmix"
                if init_mode == "maxmix":
                    late = {
                        max(1, int(round(value)))
                        for value in np_geomspace(1, 2 * ny, 16)
                    }
                    observations = sorted(
                        set(range(1, min(16, ny // 2) + 1))
                        | late
                        | {ny, 3 * ny // 2, 2 * ny}
                    )
                else:
                    observations = [ny, 3 * ny // 2, 2 * ny]
                case = {
                    "case_id": _case_id(
                        "S2",
                        "alpha_wall",
                        nx,
                        ny,
                        alpha=alpha,
                        init=init_tag,
                    ),
                    "campaign": "S2",
                    "kind": "stochastic_scan",
                    "model": _model(
                        nx=nx,
                        ny=ny,
                        protocol="alpha_wall",
                        init_mode=init_mode,
                        alpha_in=float(alpha),
                        alpha_out=float(config["S2"]["alpha_out"]),
                    ),
                    "run": _run(
                        ny,
                        lyapunov_nvec=int(s2["lyapunov_nvec"]),
                        lyapunov_start_cycle=1,
                    ),
                    "observation_cycles": observations,
                    "strip_entropy_cycles": [ny, 3 * ny // 2, 2 * ny],
                    "local_marker_cycles": [2 * ny],
                }
                cases.append(case)
        # A24: pure-state mirror of S2 using the support-terminated construction.
        # Existing S2 case dictionaries above are intentionally untouched so v1 and
        # unversioned archives remain byte-for-byte compatible with their case contract.
        for ny, alpha in s2_points:
            cases.append(
                {
                    "case_id": _case_id(
                        "S2",
                        "support_terminated_alpha_wall",
                        nx,
                        ny,
                        alpha=alpha,
                        init="pure",
                    ),
                    "campaign": "S2",
                    "kind": "stochastic_scan",
                    "model": _model(
                        nx=nx,
                        ny=ny,
                        protocol="support_terminated",
                        init_mode="default",
                        alpha_in=float(alpha),
                        alpha_out=float(config["S2"]["alpha_out"]),
                    ),
                    "run": _run(
                        ny,
                        lyapunov_nvec=int(s2["lyapunov_nvec"]),
                        lyapunov_start_cycle=1,
                    ),
                    "observation_cycles": [ny, 3 * ny // 2, 2 * ny],
                    "strip_entropy_cycles": [ny, 3 * ny // 2, 2 * ny],
                    "local_marker_cycles": [2 * ny],
                }
            )
        m2 = config["M2"]
        for ny in m2["Ny_full_scan"]:
            cases.append(
                {
                    "case_id": _case_id("M2", "perfect", nx, ny),
                    "campaign": "M2",
                    "kind": "stochastic_control",
                    "model": _model(nx=nx, ny=ny, protocol="explicit_interface"),
                    "run": _run(ny),
                    "observation_cycles": [ny, 2 * ny],
                }
            )
            for error in m2["imperfect_error_probability"]:
                success = 1.0 - float(error)
                cases.append(
                    {
                        "case_id": _case_id("M2", "imperfect", nx, ny, error=error),
                        "campaign": "M2",
                        "kind": "stochastic_control",
                        "model": _model(nx=nx, ny=ny, protocol="explicit_interface"),
                        "run": _run(
                            ny,
                            perfect_correction=False,
                            p_gain=success,
                            p_loss=success,
                        ),
                        "observation_cycles": [ny, 2 * ny],
                    }
                )
        for ny in m2["Ny_scaling"]:
            for error in [0.0, *m2["scaling_error_probability"]]:
                perfect = float(error) == 0.0
                success = 1.0 - float(error)
                cases.append(
                    {
                        "case_id": _case_id(
                            "M2",
                            "perfect" if perfect else "imperfect",
                            nx,
                            ny,
                            **({} if perfect else {"error": error}),
                        ),
                        "campaign": "M2",
                        "kind": "stochastic_control",
                        "model": _model(nx=nx, ny=ny, protocol="explicit_interface"),
                        "run": _run(
                            ny,
                            **(
                                {}
                                if perfect
                                else {
                                    "perfect_correction": False,
                                    "p_gain": success,
                                    "p_loss": success,
                                }
                            ),
                        ),
                        "observation_cycles": [ny, 2 * ny],
                    }
                )
        m3 = config["M3"]
        dense_geometry = m3["bulk_full_scan_geometry"]
        bulk_points = [
            *[
                (int(dense_geometry["Nx"]), int(dense_geometry["Ny"]), float(sigma))
                for sigma in m3["sigma"]
            ],
            *[
                (nx, int(ny), float(sigma))
                for ny in m3["bulk_scaling_Ny"]
                for sigma in m3["bulk_scaling_sigma"]
            ],
        ]
        if len(bulk_points) != len(set(bulk_points)):
            raise ValueError("M3 full-scan and scaling matrices overlap")
        for bulk_nx, bulk_ny, sigma in bulk_points:
            cases.append(
                {
                    "case_id": _case_id(
                        "M3BULK", "uniform_topological", bulk_nx, bulk_ny, sigma=sigma
                    ),
                    "campaign": "M3_BULK",
                    "kind": "stochastic_noise_scan",
                    "model": _model(
                        nx=bulk_nx, ny=bulk_ny, protocol="uniform_topological"
                    ),
                    "run": _run(
                        bulk_ny,
                        onsite_phase_noise_sigma=float(sigma),
                        lyapunov_nvec=16,
                        lyapunov_start_cycle=1,
                    ),
                    "observation_cycles": [bulk_ny, 2 * bulk_ny],
                    "local_marker_cycles": [2 * bulk_ny],
                    "bott_cycles": [2 * bulk_ny],
                    "tangent_alignment_stop": bulk_ny,
                }
            )
        if m3_wall_sigma:
            for ny in m3["wall_Ny_after_bulk_gate"]:
                for sigma in m3_wall_sigma:
                    for protocol in m3.get(
                        "wall_protocols",
                        ["explicit_interface", "explicit_interface_matched_trivial"],
                    ):
                        cases.append(
                            {
                                "case_id": _case_id(
                                    "M3WALL", protocol, nx, ny, sigma=sigma
                                ),
                                "campaign": "M3_WALL",
                                "kind": "stochastic_noise_scan",
                                "model": _model(nx=nx, ny=ny, protocol=protocol),
                                "run": _run(
                                    ny,
                                    onsite_phase_noise_sigma=float(sigma),
                                    lyapunov_nvec=16,
                                    lyapunov_start_cycle=1,
                                ),
                                "observation_cycles": [ny, 3 * ny // 2, 2 * ny],
                                "strip_entropy_cycles": [ny, 3 * ny // 2, 2 * ny],
                                "local_marker_cycles": [ny, 2 * ny],
                                "tangent_alignment_stop": ny,
                            }
                        )
        return cases

    raise ValueError(f"unsupported bundle {bundle!r}")


def np_geomspace(start: int, stop: int, count: int) -> list[float]:
    if start <= 0 or stop < start or count < 2:
        raise ValueError("invalid geometric grid")
    ratio = (float(stop) / float(start)) ** (1.0 / float(count - 1))
    return [float(start) * ratio**index for index in range(int(count))]


def case_index(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {case["case_id"]: case for case in cases}
    if len(output) != len(cases):
        raise ValueError("campaign expansion produced duplicate case IDs")
    return output
