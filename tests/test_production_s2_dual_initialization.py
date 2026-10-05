from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "00_WORKSPACE" / "CURRENT" / "final_production_ready_figure_scripts"
PRIOR_DESIGNS = PRODUCTION / "prior_designs"
SHARED = PRODUCTION / "_shared_src"
sys.path.insert(0, str(SHARED))

from campaign_cases import expand_cases  # noqa: E402


def test_s2_has_explicit_pair_and_support_terminated_pure_mirror() -> None:
    config = json.loads(
        (PRIOR_DESIGNS / "05_scans_and_controls" / "production_config.json").read_text()
    )
    cases = [
        case
        for case in expand_cases(config, accepted_width=20)
        if case["campaign"] == "S2"
    ]
    expected_per_initialization = (
        len(config["S2"]["Ny_full_scan"]) * len(config["S2"]["alpha_in"])
        + len(config["S2"]["Ny_scaling"]) * len(config["S2"]["alpha_scaling"])
    )
    assert len(cases) == 3 * expected_per_initialization

    grouped = {"default": [], "maxmix": []}
    for case in cases:
        grouped[case["model"]["init_mode"]].append(case)
    assert len(grouped["default"]) == 2 * expected_per_initialization
    assert len(grouped["maxmix"]) == expected_per_initialization

    for pure in grouped["default"]:
        ny = pure["model"]["Ny"]
        assert "_init-pure" in pure["case_id"]
        assert pure["observation_cycles"] == [ny, 3 * ny // 2, 2 * ny]
        assert "bridge_cycles" not in pure
        if "support_terminated_alpha_wall" in pure["case_id"]:
            assert pure["model"]["dw_truncation"] is True
            assert pure["model"]["meas_slab_only"] is True
        else:
            assert pure["model"]["dw_truncation"] is False
            assert pure["model"]["meas_slab_only"] is False

    for maxmix in grouped["maxmix"]:
        ny = maxmix["model"]["Ny"]
        assert "_init-maxmix" in maxmix["case_id"]
        assert "bridge_cycles" not in maxmix
        assert maxmix["observation_cycles"][0] == 1
        assert maxmix["observation_cycles"][-1] == 2 * ny
