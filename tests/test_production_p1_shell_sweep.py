from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "00_WORKSPACE" / "CURRENT" / "final_production_ready_figure_scripts"
PRIOR_DESIGNS = PRODUCTION / "prior_designs"
SHARED = PRODUCTION / "_shared_src"
sys.path.insert(0, str(SHARED))

from campaign_cases import expand_cases  # noqa: E402


def test_p1_bulk_baseline_runs_all_three_shell_choices() -> None:
    config_path = (
        PRIOR_DESIGNS
        / "01_p1_existing_completion"
        / "production_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cases = expand_cases(config)
    p1 = [case for case in cases if case["campaign"] == "P1"]

    assert len(p1) == 48
    assert Counter(case["model"]["nshell"] for case in p1) == {
        2: 16,
        1: 16,
        None: 16,
    }
    assert all("rev-shell_sweep_v2" in case["case_id"] for case in p1)
    assert all(
        case["model"]["backend"] == (
            "dense" if case["model"]["nshell"] is None else "local"
        )
        for case in p1
    )
