from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "00_WORKSPACE" / "CURRENT" / "final_production_ready_figure_scripts"
PRIOR_DESIGNS = PRODUCTION / "prior_designs"
SHARED = PRODUCTION / "_shared_src"
sys.path.insert(0, str(SHARED))

from campaign_cases import expand_cases  # noqa: E402
from run_core_shard import (  # noqa: E402
    _enforce_no_dense_choi_run_args,
    preflight,
)


def _config(bundle: str) -> dict:
    return json.loads((PRIOR_DESIGNS / bundle / "production_config.json").read_text())


def test_p2_and_s2_case_matrices_never_request_choi_tracking() -> None:
    p2 = expand_cases(_config("04_maxmix_master"), accepted_width=20)
    s2 = [
        case
        for case in expand_cases(_config("05_scans_and_controls"), accepted_width=20)
        if case["campaign"] == "S2"
    ]
    assert len(p2) == 10
    assert len(s2) == 72
    for case in [*p2, *s2]:
        assert "bridge_cycles" not in case
        assert "track_choi" not in case
        assert "track_choi" not in case["run"]
        plan = preflight(case, shard_samples=5)
        assert plan["choi_tracking_enabled"] is False
        assert plan["choi_bridge_transient_bytes"] == 0
        assert plan["choi_bridge_transient_GiB"] == 0.0


def test_stale_choi_case_and_run_arguments_are_rejected() -> None:
    case = expand_cases(_config("04_maxmix_master"), accepted_width=20)[0]
    stale = dict(case)
    stale["bridge_cycles"] = [1]
    with pytest.raises(ValueError, match="dense Choi tracking is forbidden"):
        preflight(stale, shard_samples=5)
    stale = dict(case)
    stale["dense_choi_tracking"] = False
    with pytest.raises(ValueError, match="dense Choi tracking is forbidden"):
        preflight(stale, shard_samples=5)

    assert _enforce_no_dense_choi_run_args({"cycles": 4})["track_choi"] is False
    assert _enforce_no_dense_choi_run_args({"track_choi": False})["track_choi"] is False
    with pytest.raises(ValueError, match="dense Choi tracking is forbidden"):
        _enforce_no_dense_choi_run_args({"track_choi": True})
    with pytest.raises(ValueError, match="dense Choi tracking is forbidden"):
        _enforce_no_dense_choi_run_args({"choi_observer": object()})


def test_selected_observable_v3_has_no_bridge_named_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    import selected_observables

    monkeypatch.setattr(
        selected_observables, "build_chern_partition_indices", lambda **_: {}
    )
    monkeypatch.setattr(
        selected_observables,
        "real_space_chern_batch_torch",
        lambda covariance, _: torch.zeros(
            covariance.shape[0], dtype=torch.float64, device=covariance.device
        ),
    )
    observer = selected_observables.SelectedCovarianceObserver(
        nx=1,
        ny=1,
        samples=1,
        physical_cycles=1,
        observation_cycles=[1],
        compute_correlator=False,
    )
    covariance = torch.zeros((1, 2, 2), dtype=torch.complex128)
    observer(cycle=0, G=covariance, batch_start=0, batch_count=1)
    observer(cycle=1, G=covariance, batch_start=0, batch_count=1)
    path = tmp_path / "selected.npz"
    observer.save(path, config={})
    with np.load(path, allow_pickle=False) as payload:
        assert payload["schema"].item() == "selected_native_state_observables_v4"
        assert "low_mode_occupation" in payload.files
        assert not any("bridge" in name.lower() or "choi" in name.lower() for name in payload.files)
