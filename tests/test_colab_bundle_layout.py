from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NEW_CAMPAIGN = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs"
PRIOR_CAMPAIGN = ROOT / "00_WORKSPACE/CURRENT/final_production_ready_figure_scripts"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW_LAYOUT = _load(NEW_CAMPAIGN / "bundle_layout.py", "new_bundle_layout")
PRIOR_LAYOUT = _load(PRIOR_CAMPAIGN / "bundle_layout.py", "prior_bundle_layout")


def test_redesigned_and_prior_campaigns_have_independent_layouts() -> None:
    redesigned = NEW_LAYOUT.validate_bundle_layout(NEW_CAMPAIGN)
    prior = PRIOR_LAYOUT.validate_bundle_layout(PRIOR_CAMPAIGN)
    assert set(redesigned) == set(NEW_LAYOUT.NEW_DESIGN_BUNDLES)
    assert set(prior) == set(PRIOR_LAYOUT.PRIOR_DESIGN_BUNDLES)
    assert not (PRIOR_CAMPAIGN / "new_designs").exists()


def test_redesigned_index_and_flat_paths_are_authoritative() -> None:
    index = json.loads((NEW_CAMPAIGN / "bundle_index.json").read_text(encoding="utf-8"))
    assert index["campaign_parent"] == "final_production_new_designs"
    assert set(index["standalone_contracts"]) == set(NEW_LAYOUT.NEW_DESIGN_BUNDLES)
    for bundle in NEW_LAYOUT.NEW_DESIGN_BUNDLES:
        assert NEW_LAYOUT.bundle_path(NEW_CAMPAIGN, bundle) == NEW_CAMPAIGN / bundle
        for relative in NEW_LAYOUT.REQUIRED_BUNDLE_FILES[bundle]:
            assert (NEW_CAMPAIGN / bundle / relative).is_file()

